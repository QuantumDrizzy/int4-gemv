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

static float median(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main()
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("============================================================================\n");
    std::printf("int4-gemv - fused INT4 dequantisation GEMV, warp per row\n");
    std::printf("============================================================================\n");
    std::printf("  %s  sm_%d%d  %d SMs\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    std::printf("  %-11s %13s %11s %11s %10s\n",
                "shape", "out x in", "kernel", "GB/s", "max err");
    std::printf("  --------------------------------------------------------------\n");

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

        std::vector<uint32_t> packed;
        std::vector<half> scales;
        std::vector<float> deq;
        quantise(W, sh.out_f, sh.in_f, packed, scales, deq);

        // Reference on the dequantised weights: this checks the kernel's
        // arithmetic and layout handling, not the quantiser's accuracy.
        std::vector<float> ref(sh.out_f, 0.0f);
        for (int r = 0; r < sh.out_f; ++r) {
            float a = 0.0f;
            for (int i = 0; i < sh.in_f; ++i) a += deq[(size_t)r * sh.in_f + i] * xh[i];
            ref[r] = a;
        }

        std::vector<half> xhalf(sh.in_f);
        for (int i = 0; i < sh.in_f; ++i) xhalf[i] = __float2half(xh[i]);

        uint32_t* d_q = nullptr; half* d_s = nullptr; half* d_x = nullptr; float* d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_q, packed.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_s, scales.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_x, xhalf.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)sh.out_f * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_q, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, scales.data(), scales.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, xhalf.data(), xhalf.size() * sizeof(half), cudaMemcpyHostToDevice));

        const int block = 256;                  // 8 warps
        const int grid = (sh.out_f + (block / 32) - 1) / (block / 32);

        int4_gemv_warp_per_row<<<grid, block>>>(d_q, d_s, d_x, d_y, sh.in_f, sh.out_f);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- correctness gate, before any timing is recorded --------------
        std::vector<float> got(sh.out_f);
        CUDA_CHECK(cudaMemcpy(got.data(), d_y, (size_t)sh.out_f * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0.0f, ref_scale = 0.0f;
        for (int r = 0; r < sh.out_f; ++r) {
            max_err = std::max(max_err, std::fabs(got[r] - ref[r]));
            ref_scale = std::max(ref_scale, std::fabs(ref[r]));
        }
        const float rel = (ref_scale > 0.0f) ? max_err / ref_scale : max_err;
        if (rel > 2e-2f) {
            std::printf("  %-11s FAILED correctness: rel max err %.4f\n", sh.name, rel);
            std::exit(2);
        }

        // ---- timing --------------------------------------------------------
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
        for (int i = 0; i < 25; ++i)
            int4_gemv_warp_per_row<<<grid, block>>>(d_q, d_s, d_x, d_y, sh.in_f, sh.out_f);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times;
        times.reserve(300);
        for (int i = 0; i < 300; ++i) {
            CUDA_CHECK(cudaEventRecord(a));
            int4_gemv_warp_per_row<<<grid, block>>>(d_q, d_s, d_x, d_y, sh.in_f, sh.out_f);
            CUDA_CHECK(cudaEventRecord(b));
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            times.push_back(ms / 1e3f);
        }
        const float t = median(times);

        const double bytes = (double)packed.size() * sizeof(uint32_t)
                           + (double)scales.size() * sizeof(half);
        const double gbs = bytes / t / 1e9;

        char dims[32];
        std::snprintf(dims, sizeof(dims), "%dx%d", sh.out_f, sh.in_f);
        std::printf("  %-11s %13s %10.1fu %11.1f %10.5f\n",
                    sh.name, dims, t * 1e6, gbs, rel);
        std::fflush(stdout);

        std::fprintf(js, "    %s{\"name\": \"%s\", \"out_features\": %d, \"in_features\": %d, "
                         "\"bytes\": %.0f, \"kernel_us\": %.3f, \"gbs\": %.2f, \"rel_max_err\": %.6f}\n",
                     si ? ", " : "  ", sh.name, sh.out_f, sh.in_f, bytes, t * 1e6, gbs, rel);

        CUDA_CHECK(cudaEventDestroy(a));
        CUDA_CHECK(cudaEventDestroy(b));
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    std::fprintf(js, "  ]\n}\n");
    std::fclose(js);
    std::printf("\n  wrote bench/kernel.json\n");
    return 0;
}
