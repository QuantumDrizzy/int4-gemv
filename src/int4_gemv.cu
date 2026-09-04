// int4-gemv — fused INT4 dequantisation GEMV for LLM decode.
//
// ADR-0001, Option B: one warp per output row, dequantisation fused into the
// load so FP16 weights are never materialised. At batch 1 this is a pure
// memory-streaming problem — two flops per weight, every weight read once — so
// the only target that means anything is a fraction of achieved bandwidth.
//
// Layout, per output row:
//   packed weights  in_f/8 uint32 words, 8 int4 codes per word, little-endian
//                   nibble order. Value = (code - 8) * scale.
//   scales          in_f/GROUP half, one per group along the reduction axis.
//
// GROUP is a multiple of 8, so all eight codes in a word share one scale and
// the scale load is hoisted out of the inner unroll.
//
// Build:
//   nvcc -O3 -arch=sm_120 -o int4_gemv src/int4_gemv.cu

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define GROUP 128
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void int4_gemv_warp_per_row(const uint32_t* __restrict__ qw,
                                       const half* __restrict__ scales,
                                       const half* __restrict__ x,
                                       float* __restrict__ y,
                                       int in_f, int out_f)
{
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= out_f) return;

    const int words = in_f >> 3;                 // 8 int4 codes per uint32
    const uint32_t* __restrict__ wrow = qw + (size_t)row * words;
    const half* __restrict__ srow = scales + (size_t)row * (in_f / GROUP);

    float acc = 0.0f;

    // Lanes stride by 32 words, so consecutive lanes read consecutive words:
    // the row is walked in fully coalesced 128-byte transactions.
    for (int w = lane; w < words; w += 32) {
        const uint32_t packed = __ldg(&wrow[w]);
        const int base = w << 3;
        const float s = __half2float(__ldg(&srow[base / GROUP]));

#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int   code = (packed >> (k * 4)) & 0xF;
            const float wv   = (float)(code - 8) * s;
            acc = fmaf(wv, __half2float(__ldg(&x[base + k])), acc);
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// NF4 — the like-for-like variant (ADR-0001, action item 7)
//
// Same format bitsandbytes uses, verified against its quant_state: block 64,
// FP32 absmax, no double quantisation, and the canonical NormalFloat code
// table. Value = NF4[code] * absmax, so this moves the same bytes and lands on
// the same numbers; only the kernel differs.
//
// The lookup is the whole difference against INT4, where dequantisation is one
// multiply. A 16-entry table read from memory with divergent indices serialises,
// so it is held in registers instead: lane L keeps entry L, and __shfl_sync with
// a per-thread source lane turns the warp's register file into the table. One
// instruction per lookup, no memory traffic, no bank conflicts.
// ---------------------------------------------------------------------------
__constant__ float NF4_TABLE[16] = {
    -1.0f,              -0.6961928009986877f, -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
    -0.09105003625154495f, 0.0f,               0.07958029955625534f,
     0.16093020141124725f,  0.24611230194568634f, 0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,  0.7229568362236023f, 1.0f};

#define NF4_BLOCK 64                       // bitsandbytes default, = 8 words

__global__ void nf4_gemv_warp_per_row(const uint32_t* __restrict__ qw,
                                      const float* __restrict__ absmax,
                                      const half* __restrict__ x,
                                      float* __restrict__ y,
                                      int in_f, int out_f)
{
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= out_f) return;

    // Lanes 0-15 hold the code table; lanes 16-31 are never a shuffle source
    // because a 4-bit code cannot exceed 15.
    const float lut = (lane < 16) ? NF4_TABLE[lane] : 0.0f;

    const int words = in_f >> 3;
    const uint32_t* __restrict__ wrow = qw + (size_t)row * words;
    const float* __restrict__ arow = absmax + (size_t)row * (in_f / NF4_BLOCK);

    float acc = 0.0f;

    // Every lane runs the same number of iterations: __shfl_sync is undefined
    // if a lane in the mask has already exited, and in_f is not guaranteed to
    // make `words` a multiple of 32.
    const int iters = (words + 31) >> 5;
    for (int it = 0; it < iters; ++it) {
        const int w = (it << 5) + lane;
        const bool active = (w < words);
        const uint32_t packed = active ? __ldg(&wrow[w]) : 0u;
        const float s = active ? __ldg(&arow[w >> 3]) : 0.0f;
        const int base = w << 3;

#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int code = (packed >> (k * 4)) & 0xF;
            const float wv = __shfl_sync(0xffffffffu, lut, code) * s;
            if (active)
                acc = fmaf(wv, __half2float(__ldg(&x[base + k])), acc);
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[row] = acc;
}

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

// Symmetric group-wise INT4. Returns the dequantised weights alongside the
// packed form so the reference GEMV tests the kernel, not the quantiser.
static void quantise(const std::vector<float>& W, int out_f, int in_f,
                     std::vector<uint32_t>& packed, std::vector<half>& scales,
                     std::vector<float>& deq)
{
    const int groups = in_f / GROUP;
    packed.assign((size_t)out_f * (in_f / 8), 0u);
    scales.resize((size_t)out_f * groups);
    deq.resize((size_t)out_f * in_f);

    for (int r = 0; r < out_f; ++r) {
        for (int g = 0; g < groups; ++g) {
            const int off = r * in_f + g * GROUP;
            float amax = 0.0f;
            for (int i = 0; i < GROUP; ++i) amax = std::max(amax, std::fabs(W[off + i]));
            const float s = (amax > 0.0f) ? amax / 7.0f : 1.0f;
            scales[(size_t)r * groups + g] = __float2half(s);
            const float s_used = __half2float(scales[(size_t)r * groups + g]);

            for (int i = 0; i < GROUP; ++i) {
                int code = (int)std::lround(W[off + i] / s_used) + 8;
                code = std::min(15, std::max(0, code));
                const int idx = g * GROUP + i;
                packed[(size_t)r * (in_f / 8) + idx / 8] |= ((uint32_t)code) << ((idx % 8) * 4);
                deq[off + i] = (float)(code - 8) * s_used;
            }
        }
    }
}

// NF4, matching bitsandbytes: block 64, FP32 absmax, nearest code from the
// canonical table. Same host-side table as the device constant.
static const float NF4_HOST[16] = {
    -1.0f,              -0.6961928009986877f, -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
    -0.09105003625154495f, 0.0f,               0.07958029955625534f,
     0.16093020141124725f,  0.24611230194568634f, 0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,  0.7229568362236023f, 1.0f};

static void quantise_nf4(const std::vector<float>& W, int out_f, int in_f,
                         std::vector<uint32_t>& packed, std::vector<float>& absmax,
                         std::vector<float>& deq)
{
    const int blocks = in_f / NF4_BLOCK;
    packed.assign((size_t)out_f * (in_f / 8), 0u);
    absmax.resize((size_t)out_f * blocks);
    deq.resize((size_t)out_f * in_f);

    for (int r = 0; r < out_f; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const int off = r * in_f + b * NF4_BLOCK;
            float amax = 0.0f;
            for (int i = 0; i < NF4_BLOCK; ++i) amax = std::max(amax, std::fabs(W[off + i]));
            if (amax == 0.0f) amax = 1.0f;
            absmax[(size_t)r * blocks + b] = amax;

            for (int i = 0; i < NF4_BLOCK; ++i) {
                const float norm = W[off + i] / amax;
                int best = 0;
                float bd = std::fabs(norm - NF4_HOST[0]);
                for (int c = 1; c < 16; ++c) {
                    const float d = std::fabs(norm - NF4_HOST[c]);
                    if (d < bd) { bd = d; best = c; }
                }
                const int idx = b * NF4_BLOCK + i;
                packed[(size_t)r * (in_f / 8) + idx / 8] |= ((uint32_t)best) << ((idx % 8) * 4);
                deq[off + i] = NF4_HOST[best] * amax;
            }
        }
    }
}

static float median(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
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

    auto launch = [&]() {
        if (nf4)
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
