// Diagnostic: why does down_proj lose?
//
// down_proj is 3584x18944 and gate_proj is 18944x3584 — the same weights,
// transposed — and the kernel is slower on the first. Two candidates:
//
//   (a) warp count. 3584 against 18944, so fewer waves to hide latency and a
//       tail that is a larger fraction of the run.
//   (b) the size of x, which every warp reads in full: 37 KB against 7 KB. The
//       weight-only roofline does not count that traffic, and x is the one thing
//       in this kernel that is re-read rather than streamed.
//
// Sweep 1 walks the aspect ratio at constant weight count, which shows the
// effect but cannot attribute it: warps fall and x grows together. Sweeps 2 and
// 3 hold one variable fixed each, which is what actually separates them.
//
// Build: scripts/build.bat

#include "kernels.cuh"

struct Case { int out_f; int in_f; const char* note; };

static void run_sweep(const char* title, const char* explain,
                      const Case* cases, int n)
{
    std::printf("\n  %s\n", title);
    std::printf("  %s\n", explain);
    std::printf("  %13s %9s %9s %9s %10s %10s   %s\n",
                "out x in", "warps", "x KB", "kernel us", "GB/s", "MW/ms", "note");
    std::printf("  ----------------------------------------------------------------------------------------\n");

    for (int c = 0; c < n; ++c) {
        const Case a = cases[c];
        const size_t n_w = (size_t)a.out_f * a.in_f;
        std::vector<float> W(n_w), xh(a.in_f);
        std::srand(0);
        for (size_t i = 0; i < n_w; ++i) W[i] = ((float)std::rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < a.in_f; ++i) xh[i] = ((float)std::rand() / RAND_MAX - 0.5f);

        std::vector<uint32_t> pn; std::vector<float> an, dn;
        quantise_nf4(W, a.out_f, a.in_f, pn, an, dn);

        std::vector<half> xhalf(a.in_f);
        for (int i = 0; i < a.in_f; ++i) xhalf[i] = __float2half(xh[i]);

        uint32_t* dq = nullptr; float* ds = nullptr; half* dx = nullptr; float* dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dq, pn.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds, an.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dx, xhalf.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)a.out_f * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dq, pn.data(), pn.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds, an.data(), an.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, xhalf.data(), xhalf.size() * sizeof(half), cudaMemcpyHostToDevice));

        const int block = 256;
        const int grid = (a.out_f + (block / 32) - 1) / (block / 32);
        auto launch = [&]() {
            nf4_gemv_warp_per_row<<<grid, block>>>(dq, ds, dx, dy, a.in_f, a.out_f);
        };

        for (int i = 0; i < 25; ++i) launch();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        std::vector<float> times;
        for (int i = 0; i < 200; ++i) {
            CUDA_CHECK(cudaEventRecord(e0));
            launch();
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            times.push_back(ms / 1e3f);
        }
        const float t = median(times);
        const double bytes = (double)pn.size() * 4 + (double)an.size() * 4;

        char dims[32];
        std::snprintf(dims, sizeof(dims), "%dx%d", a.out_f, a.in_f);
        // MW/ms = millions of weights per millisecond: work rate, comparable
        // across sweeps where the total weight count is NOT held constant.
        std::printf("  %13s %9d %9.1f %8.1fu %10.1f %10.1f   %s\n",
                    dims, a.out_f, a.in_f * 2 / 1024.0, t * 1e6,
                    bytes / t / 1e9, n_w / (t * 1e3) / 1e6, a.note);
        std::fflush(stdout);

        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(ds));
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
}

static const Case SWEEP_ASPECT[] = {
    {18944,  3584, "gate_proj"}, { 9472,  7168, ""}, { 4736, 14336, ""},
    { 3584, 18944, "down_proj"}, { 2368, 28672, ""}, { 1184, 57344, ""},
};

// x fixed at 7 KB; only the warp count moves.
static const Case SWEEP_WARPS[] = {
    {  1184, 3584, ""}, {  2368, 3584, ""}, {  4736, 3584, ""},
    {  9472, 3584, ""}, { 18944, 3584, "gate_proj"}, { 37888, 3584, ""},
};

// warps fixed at 3584; only x moves.
static const Case SWEEP_X[] = {
    {3584,  3584, ""}, {3584,  7168, ""}, {3584, 14336, ""},
    {3584, 18944, "down_proj"}, {3584, 28672, ""}, {3584, 57344, ""},
};

// Split-K sweep. y must be zeroed before each launch, and that memset is a
// launch of its own, so it is inside the timed region: it is real work the
// caller cannot avoid.
static void run_split_sweep(const Case* cases, int n, const int* splits, int ns)
{
    std::printf("\n  SWEEP 4 - split-K, the fix the sweeps above point at\n");
    std::printf("  Warps become out_features x splits. Timing includes the memset of y.\n");
    std::printf("  %13s %8s %9s %10s %10s %9s   %s\n",
                "out x in", "splits", "warps", "kernel us", "MW/ms", "vs 1", "note");
    std::printf("  ----------------------------------------------------------------------------------------\n");

    for (int c = 0; c < n; ++c) {
        const Case a = cases[c];
        const size_t n_w = (size_t)a.out_f * a.in_f;
        std::vector<float> W(n_w), xh(a.in_f);
        std::srand(0);
        for (size_t i = 0; i < n_w; ++i) W[i] = ((float)std::rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < a.in_f; ++i) xh[i] = ((float)std::rand() / RAND_MAX - 0.5f);

        std::vector<uint32_t> pn; std::vector<float> an, dn;
        quantise_nf4(W, a.out_f, a.in_f, pn, an, dn);
        std::vector<half> xhalf(a.in_f);
        for (int i = 0; i < a.in_f; ++i) xhalf[i] = __float2half(xh[i]);

        std::vector<float> ref(a.out_f, 0.0f);
        for (int r = 0; r < a.out_f; ++r) {
            float s = 0.0f;
            for (int i = 0; i < a.in_f; ++i) s += dn[(size_t)r * a.in_f + i] * xh[i];
            ref[r] = s;
        }

        uint32_t* dq = nullptr; float* ds = nullptr; half* dx = nullptr; float* dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dq, pn.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds, an.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dx, xhalf.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)a.out_f * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dq, pn.data(), pn.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds, an.data(), an.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, xhalf.data(), xhalf.size() * sizeof(half), cudaMemcpyHostToDevice));

        double base_rate = 0.0;
        for (int k = 0; k < ns; ++k) {
            const int S = splits[k];
            const int block = 256;
            const int warps = a.out_f * S;
            const int grid = (warps + (block / 32) - 1) / (block / 32);
            auto launch = [&]() {
                CUDA_CHECK(cudaMemsetAsync(dy, 0, (size_t)a.out_f * sizeof(float)));
                nf4_gemv_split_k<<<grid, block>>>(dq, ds, dx, dy, a.in_f, a.out_f, S);
            };

            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> got(a.out_f);
            CUDA_CHECK(cudaMemcpy(got.data(), dy, (size_t)a.out_f * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            float me = 0.0f, rs = 0.0f;
            for (int r = 0; r < a.out_f; ++r) {
                me = std::max(me, std::fabs(got[r] - ref[r]));
                rs = std::max(rs, std::fabs(ref[r]));
            }
            if (rs > 0.0f && me / rs > 2e-2f) {
                std::printf("  split %d FAILED correctness: %.5f\n", S, me / rs);
                std::exit(2);
            }

            for (int i = 0; i < 25; ++i) launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<float> times;
            for (int i = 0; i < 200; ++i) {
                CUDA_CHECK(cudaEventRecord(e0));
                launch();
                CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                times.push_back(ms / 1e3f);
            }
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            const float t = median(times);
            const double rate = n_w / (t * 1e3) / 1e6;
            if (k == 0) base_rate = rate;

            char dims[32];
            std::snprintf(dims, sizeof(dims), "%dx%d", a.out_f, a.in_f);
            std::printf("  %13s %8d %9d %9.1fu %10.1f %8.2fx   %s\n",
                        k ? "" : dims, S, warps, t * 1e6, rate,
                        rate / base_rate, k ? "" : a.note);
            std::fflush(stdout);
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(ds));
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
}

// Block-per-row: same warp multiplication, no memset and no global atomics.
// Compared against the plain one-warp-per-row kernel, which is the number both
// have to beat.
static void run_block_sweep(const Case* cases, int n)
{
    std::printf("\n  SWEEP 5 - block per row vs one warp per row\n");
    std::printf("  Warps become out_features x warps_per_block. No memset, no atomics.\n");
    std::printf("  %13s %10s %9s %10s %10s %9s   %s\n",
                "out x in", "warps/blk", "warps", "kernel us", "MW/ms", "vs 1 warp", "note");
    std::printf("  ----------------------------------------------------------------------------------------\n");

    const int WPB[] = {1, 2, 4, 8, 16};
    for (int c = 0; c < n; ++c) {
        const Case a = cases[c];
        const size_t n_w = (size_t)a.out_f * a.in_f;
        std::vector<float> W(n_w), xh(a.in_f);
        std::srand(0);
        for (size_t i = 0; i < n_w; ++i) W[i] = ((float)std::rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < a.in_f; ++i) xh[i] = ((float)std::rand() / RAND_MAX - 0.5f);

        std::vector<uint32_t> pn; std::vector<float> an, dn;
        quantise_nf4(W, a.out_f, a.in_f, pn, an, dn);
        std::vector<half> xhalf(a.in_f);
        for (int i = 0; i < a.in_f; ++i) xhalf[i] = __float2half(xh[i]);

        std::vector<float> ref(a.out_f, 0.0f);
        for (int r = 0; r < a.out_f; ++r) {
            float s = 0.0f;
            for (int i = 0; i < a.in_f; ++i) s += dn[(size_t)r * a.in_f + i] * xh[i];
            ref[r] = s;
        }

        uint32_t* dq = nullptr; float* ds = nullptr; half* dx = nullptr; float* dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dq, pn.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds, an.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dx, xhalf.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)a.out_f * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dq, pn.data(), pn.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds, an.data(), an.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, xhalf.data(), xhalf.size() * sizeof(half), cudaMemcpyHostToDevice));

        // Baseline: the plain one-warp-per-row kernel, no memset, no atomics.
        double base_rate = 0.0;
        {
            const int block = 256;
            const int grid = (a.out_f + 7) / 8;
            auto launch = [&]() {
                nf4_gemv_warp_per_row<<<grid, block>>>(dq, ds, dx, dy, a.in_f, a.out_f);
            };
            for (int i = 0; i < 25; ++i) launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<float> times;
            for (int i = 0; i < 200; ++i) {
                CUDA_CHECK(cudaEventRecord(e0)); launch(); CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                times.push_back(ms / 1e3f);
            }
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            const float t = median(times);
            base_rate = n_w / (t * 1e3) / 1e6;
            char dims[32];
            std::snprintf(dims, sizeof(dims), "%dx%d", a.out_f, a.in_f);
            std::printf("  %13s %10s %9d %9.1fu %10.1f %8.2fx   %s\n",
                        dims, "1 (plain)", a.out_f, t * 1e6, base_rate, 1.0, a.note);
        }

        for (int k = 0; k < 5; ++k) {
            const int wpb = WPB[k];
            const int block = wpb * 32;
            const size_t shmem = wpb * sizeof(float);
            auto launch = [&]() {
                nf4_gemv_block_per_row<<<a.out_f, block, shmem>>>(dq, ds, dx, dy,
                                                                  a.in_f, a.out_f);
            };
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> got(a.out_f);
            CUDA_CHECK(cudaMemcpy(got.data(), dy, (size_t)a.out_f * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            float me = 0.0f, rs = 0.0f;
            for (int r = 0; r < a.out_f; ++r) {
                me = std::max(me, std::fabs(got[r] - ref[r]));
                rs = std::max(rs, std::fabs(ref[r]));
            }
            if (rs > 0.0f && me / rs > 2e-2f) {
                std::printf("  wpb %d FAILED correctness: %.5f\n", wpb, me / rs);
                std::exit(2);
            }

            for (int i = 0; i < 25; ++i) launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<float> times;
            for (int i = 0; i < 200; ++i) {
                CUDA_CHECK(cudaEventRecord(e0)); launch(); CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                times.push_back(ms / 1e3f);
            }
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            const float t = median(times);
            const double rate = n_w / (t * 1e3) / 1e6;
            std::printf("  %13s %10d %9d %9.1fu %10.1f %8.2fx\n",
                        "", wpb, a.out_f * wpb, t * 1e6, rate, rate / base_rate);
            std::fflush(stdout);
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(ds));
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
}

static const Case SPLIT_CASES[] = {
    { 512,  3584, "k_proj"},
    {3584, 18944, "down_proj"},
    {3584,  3584, "q_proj"},
    {18944, 3584, "gate_proj"},
};
static const int SPLIT_FACTORS[] = {1, 2, 4, 8, 16};

int main()
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("==========================================================================================\n");
    std::printf("why does down_proj lose? - three sweeps, NF4 kernel\n");
    std::printf("==========================================================================================\n");
    std::printf("  %s  sm_%d%d  %d SMs  L2 %.1f MB  shared/SM %zu KB\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.l2CacheSize / 1048576.0,
                (size_t)prop.sharedMemPerMultiprocessor / 1024);

    run_sweep("SWEEP 1 - aspect ratio, weight count held constant",
              "Shows the effect. Cannot attribute it: warps fall as x grows.",
              SWEEP_ASPECT, 6);

    run_sweep("SWEEP 2 - warp count, x held at 7 KB",
              "If warp count is the cause, MW/ms falls at the low end.",
              SWEEP_WARPS, 6);

    run_sweep("SWEEP 3 - size of x, warps held at 3584",
              "If x residency is the cause, MW/ms falls as x grows.",
              SWEEP_X, 6);

    run_split_sweep(SPLIT_CASES, 4, SPLIT_FACTORS, 5);
    run_block_sweep(SPLIT_CASES, 4);

    std::printf("\n  MW/ms is the work rate, comparable across sweeps 2 and 3 where the\n");
    std::printf("  total weight count is not constant. GB/s counts weights and scales\n");
    std::printf("  only - it never counted the traffic from x, which is the point.\n");
    return 0;
}
