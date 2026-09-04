// Kernels and the shared quantisation formats. Split out so the benchmark
// and the diagnostic sweep build against the same code.
#pragma once

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
// NF4 split-K (ADR-0001, Option C)
//
// One warp per row leaves the device short of warps whenever out_features is
// small. The aspect sweep in diag_aspect.cu measured that directly: at constant
// x, the work rate climbs from 329 MW/ms at 1184 warps to 559 at 18944, and
// saturates around 9-19k. down_proj launches 3584 warps and k_proj launches 512,
// so both sit well below the knee.
//
// Splitting the reduction dimension multiplies the warp count by `splits`. Each
// (row, split) pair computes a partial dot product and atomically accumulates
// into y, so y MUST be zeroed before launch.
//
// The atomics make the summation order non-deterministic, which is why the
// correctness check for this kernel is a tolerance rather than an equality.
// ---------------------------------------------------------------------------
__global__ void nf4_gemv_split_k(const uint32_t* __restrict__ qw,
                                 const float* __restrict__ absmax,
                                 const half* __restrict__ x,
                                 float* __restrict__ y,
                                 int in_f, int out_f, int splits)
{
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane  = threadIdx.x & 31;
    const int row   = gwarp / splits;
    const int part  = gwarp - row * splits;
    if (row >= out_f) return;

    const float lut = (lane < 16) ? NF4_TABLE[lane] : 0.0f;

    const int words = in_f >> 3;
    const uint32_t* __restrict__ wrow = qw + (size_t)row * words;
    const float* __restrict__ arow = absmax + (size_t)row * (in_f / NF4_BLOCK);

    // Split on a 32-word boundary so every slice stays coalesced.
    const int chunk = ((words + splits - 1) / splits + 31) & ~31;
    const int begin = part * chunk;
    const int end   = min(begin + chunk, words);

    float acc = 0.0f;
    const int iters = (end > begin) ? ((end - begin + 31) >> 5) : 0;
    for (int it = 0; it < iters; ++it) {
        const int w = begin + (it << 5) + lane;
        const bool active = (w < end);
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

    if (lane == 0 && iters > 0) atomicAdd(&y[row], acc);
}

// ---------------------------------------------------------------------------
// NF4 block-per-row — split-K's job without split-K's costs
//
// Split-K multiplies the warp count, which is what the sweeps say is missing,
// but it pays for it twice: y has to be zeroed by a separate launch, and every
// warp closes with a global atomicAdd. Measured on down_proj those costs eat
// almost exactly the occupancy they buy.
//
// Same idea, one block per row instead: the block's warps each take a strided
// slice, then reduce through shared memory and a single warp writes y[row]. The
// warp count still multiplies by warps-per-block, but there is no memset, no
// global atomic, and the result is deterministic.
// ---------------------------------------------------------------------------
__global__ void nf4_gemv_block_per_row(const uint32_t* __restrict__ qw,
                                       const float* __restrict__ absmax,
                                       const half* __restrict__ x,
                                       float* __restrict__ y,
                                       int in_f, int out_f)
{
    extern __shared__ float partial[];
    const int row     = blockIdx.x;
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    if (row >= out_f) return;

    const float lut = (lane < 16) ? NF4_TABLE[lane] : 0.0f;

    const int words = in_f >> 3;
    const uint32_t* __restrict__ wrow = qw + (size_t)row * words;
    const float* __restrict__ arow = absmax + (size_t)row * (in_f / NF4_BLOCK);

    // Warp w walks words w*32 + lane, then strides by n_warps*32: consecutive
    // lanes still read consecutive words, so every access stays coalesced.
    const int stride = n_warps << 5;
    const int start  = (warp << 5) + lane;
    const int iters  = (words > start) ? ((words - start + stride - 1) / stride) : 0;
    const int max_iters = (words + stride - 1) / stride;

    float acc = 0.0f;
    for (int it = 0; it < max_iters; ++it) {
        const int w = start + it * stride;
        const bool active = (it < iters) && (w < words);
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

    if (lane == 0) partial[warp] = acc;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < n_warps; ++i) total += partial[i];
        y[row] = total;
    }
}

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
