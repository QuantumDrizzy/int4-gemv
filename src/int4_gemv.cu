// int4-gemv - benchmark: both formats on the real projection shapes.
//
// Build: scripts/build.bat

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// Host: quantise, reference, benchmark
// ---------------------------------------------------------------------------
struct Shape { const char* name; int out_f; int in_f; };

static const Shape SHAPES[] = {
    {"q_proj",     3584,  3584},
    {"k_proj",      512,  3584},
    {"v_proj",      512,  3584},
    {"o_proj",     3584,  3584},
    {"gate_proj", 18944,  3584},
    {"up_proj",   18944,  3584},
    {"down_proj",  3584, 18944},
};

struct Result { float us; double gbs; float rel; };

// Correctness first, then timing. The reference is computed on the dequantised
// weights, so this measures the kernel's arithmetic and layout handling rather
// than the quantiser's accuracy.
static Result run_variant(const std::vector<float>& deq,
                          const std::vector<float>& xh,
                          int out_f, int in_f, double bytes,
                          void* d_q, void* d_s, half* d_x, float* d_y,
                          bool nf4)
{
    std::vector<float> ref(out_f, 0.0f);
    for (int r = 0; r < out_f; ++r) {
        float a = 0.0f;
        for (int i = 0; i < in_f; ++i) a += deq[(size_t)r * in_f + i] * xh[i];
        ref[r] = a;
    }

    const int block = 256;                    // 8 warps
    const int grid = (out_f + (block / 32) - 1) / (block / 32);

    // One warp per row leaves the device short of warps when out_features is
    // small, and diag_aspect.cu measures the knee at roughly 9k warps. Below it,
    // splitting each row across two warps of one block recovers some of the
    // gap - but only when the row is long enough to be worth splitting. On
    // Qwen2.5-Coder-7B only down_proj (3584 rows of 2368 words) qualifies:
    // q_proj has the same row count but 448-word rows and gains nothing.
    //
    // Both thresholds are fitted to one GPU. They are a measured heuristic, not
    // a portable rule.
    const int words_per_row = in_f >> 3;
    const bool use_block_per_row = nf4 && out_f < 8192 && words_per_row >= 1024;
    const int wpb = 2;

    auto launch = [&]() {
        if (use_block_per_row)
            nf4_gemv_block_per_row<<<out_f, wpb * 32, wpb * sizeof(float)>>>(
                (const uint32_t*)d_q, (const float*)d_s, d_x, d_y, in_f, out_f);
        else if (nf4)
            nf4_gemv_warp_per_row<<<grid, block>>>((const uint32_t*)d_q,
                                                   (const float*)d_s, d_x, d_y,
                                                   in_f, out_f);
        else
            int4_gemv_warp_per_row<<<grid, block>>>((const uint32_t*)d_q,
                                                    (const half*)d_s, d_x, d_y,
                                                    in_f, out_f);
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(out_f);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, (size_t)out_f * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float max_err = 0.0f, ref_scale = 0.0f;
    for (int r = 0; r < out_f; ++r) {
        max_err = std::max(max_err, std::fabs(got[r] - ref[r]));
        ref_scale = std::max(ref_scale, std::fabs(ref[r]));
    }
    const float rel = (ref_scale > 0.0f) ? max_err / ref_scale : max_err;
    if (rel > 2e-2f) {
        std::fprintf(stderr, "FAILED correctness (%s): rel max err %.5f\n",
                     nf4 ? "nf4" : "int4", rel);
        std::exit(2);
    }

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 25; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(300);
    for (int i = 0; i < 300; ++i) {
        CUDA_CHECK(cudaEventRecord(a));
        launch();
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        times.push_back(ms / 1e3f);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));

    const float t = median(times);
    return Result{t * 1e6f, bytes / t / 1e9, rel};
}

int main()
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("==============================================================================================\n");
    std::printf("int4-gemv - fused 4-bit dequantisation GEMV, warp per row\n");
    std::printf("==============================================================================================\n");
    std::printf("  %s  sm_%d%d  %d SMs\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("  INT4 = symmetric, group %d, FP16 scales\n", GROUP);
    std::printf("  NF4  = the bitsandbytes format: block %d, FP32 absmax, NormalFloat table\n\n",
                NF4_BLOCK);

    std::printf("  %-11s %13s   %9s %9s %9s   %9s %9s %9s\n",
                "shape", "out x in", "int4 us", "int4 GB/s", "int4 err",
                "nf4 us", "nf4 GB/s", "nf4 err");
    std::printf("  ------------------------------------------------------------------------------------------\n");

    FILE* js = std::fopen("bench/kernel.json", "w");
    std::fprintf(js, "{\n  \"device\": \"%s sm_%d%d %d SMs\",\n  \"shapes\": [\n",
                 prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const int n_shapes = sizeof(SHAPES) / sizeof(SHAPES[0]);
    for (int si = 0; si < n_shapes; ++si) {
        const Shape sh = SHAPES[si];
        const size_t n_w = (size_t)sh.out_f * sh.in_f;

        std::vector<float> W(n_w), xh(sh.in_f);
        std::srand(0);
        for (size_t i = 0; i < n_w; ++i) W[i] = ((float)std::rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < sh.in_f; ++i) xh[i] = ((float)std::rand() / RAND_MAX - 0.5f);

        std::vector<half> xhalf(sh.in_f);
        for (int i = 0; i < sh.in_f; ++i) xhalf[i] = __float2half(xh[i]);

        half* d_x = nullptr; float* d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, xhalf.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)sh.out_f * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, xhalf.data(), xhalf.size() * sizeof(half),
                              cudaMemcpyHostToDevice));

        // ---- INT4 symmetric, group 128, FP16 scales ----------------------
        std::vector<uint32_t> p4; std::vector<half> s4; std::vector<float> d4;
        quantise(W, sh.out_f, sh.in_f, p4, s4, d4);
        uint32_t* dq4 = nullptr; half* ds4 = nullptr;
        CUDA_CHECK(cudaMalloc(&dq4, p4.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds4, s4.size() * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(dq4, p4.data(), p4.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds4, s4.data(), s4.size() * sizeof(half), cudaMemcpyHostToDevice));
        const double b4 = (double)p4.size() * 4 + (double)s4.size() * 2;
        const Result r4 = run_variant(d4, xh, sh.out_f, sh.in_f, b4,
                                      dq4, ds4, d_x, d_y, false);
        CUDA_CHECK(cudaFree(dq4)); CUDA_CHECK(cudaFree(ds4));

        // ---- NF4, the bitsandbytes format --------------------------------
        std::vector<uint32_t> pn; std::vector<float> an, dn;
        quantise_nf4(W, sh.out_f, sh.in_f, pn, an, dn);
        uint32_t* dqn = nullptr; float* dan = nullptr;
        CUDA_CHECK(cudaMalloc(&dqn, pn.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dan, an.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dqn, pn.data(), pn.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dan, an.data(), an.size() * sizeof(float), cudaMemcpyHostToDevice));
        const double bn = (double)pn.size() * 4 + (double)an.size() * 4;
        const Result rn = run_variant(dn, xh, sh.out_f, sh.in_f, bn,
                                      dqn, dan, d_x, d_y, true);
        CUDA_CHECK(cudaFree(dqn)); CUDA_CHECK(cudaFree(dan));

        char dims[32];
        std::snprintf(dims, sizeof(dims), "%dx%d", sh.out_f, sh.in_f);
        std::printf("  %-11s %13s   %8.1fu %9.1f %9.5f   %8.1fu %9.1f %9.5f\n",
                    sh.name, dims, r4.us, r4.gbs, r4.rel, rn.us, rn.gbs, rn.rel);
        std::fflush(stdout);

        std::fprintf(js,
            "    %s{\"name\": \"%s\", \"out_features\": %d, \"in_features\": %d, "
            "\"int4\": {\"bytes\": %.0f, \"kernel_us\": %.3f, \"gbs\": %.2f, \"rel_max_err\": %.6f}, "
            "\"nf4\": {\"bytes\": %.0f, \"kernel_us\": %.3f, \"gbs\": %.2f, \"rel_max_err\": %.6f}}\n",
            si ? ", " : "  ", sh.name, sh.out_f, sh.in_f,
            b4, r4.us, r4.gbs, r4.rel, bn, rn.us, rn.gbs, rn.rel);

        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    }

    std::fprintf(js, "  ]\n}\n");
    std::fclose(js);
    std::printf("\n  wrote bench/kernel.json\n");
    return 0;
}
