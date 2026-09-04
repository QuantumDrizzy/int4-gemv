# int4-gemv

A hand-written CUDA kernel for the hot path of quantised LLM decode: `y = W·x`
where `W` is 4-bit and the dequantisation is **fused into the load**, so FP16
weights are never materialised in memory.

Measured on an RTX 5060 Ti (sm_120 Blackwell, 36 SMs), on the seven real
projection shapes of Qwen2.5-Coder-7B at batch 1.

```
shape            out x in   roofline    bnb NF4       ours    bnb %BW  ours %BW   ratio
q_proj          3584x3584      15.5u      29.1u      22.6u      58.3%     68.8%   1.29x
k_proj           512x3584       2.2u       8.5u       6.4u      28.4%     34.7%   1.33x
o_proj          3584x3584      15.5u      29.1u      22.7u      58.3%     68.5%   1.28x
gate_proj      18944x3584      82.2u     135.7u      96.5u      66.1%     85.2%   1.41x
up_proj        18944x3584      82.2u     135.7u      96.5u      66.1%     85.1%   1.41x
down_proj      3584x18944      82.2u     129.0u     104.4u      69.5%     78.7%   1.23x

measured read bandwidth 425.9 GB/s · empty-launch floor 5.28 us
```

**85 % of the memory roofline** on the projections that dominate the parameter
count, against 66 % for the library people actually run.

## What the ratio column is not

It is **not** "1.4× faster than bitsandbytes". The two kernels do not compute the
same thing:

- **ours** — INT4 symmetric, group 128, one FP16 scale per group. Dequantisation
  is one multiply per weight.
- **bitsandbytes** — NF4 with double quantisation. A non-uniform format decoded
  through a 16-entry lookup, with the scales themselves quantised. Strictly the
  harder decode, and a different accuracy/size trade.

Comparing them and reporting the ratio as a speedup would be choosing an easier
problem and calling it an optimisation. The honest column is **`%BW`** — each
kernel against its own byte count over the same measured bandwidth. The
like-for-like number needs an NF4 variant of this kernel, which is not written
yet ([ADR-0001](docs/ADR-0001-the-kernel.md), action item 7).

`k_proj` and `v_proj` sit near the 5.28 µs launch floor. Most of those rows is
dispatch, not kernel, and they are reported rather than dropped.

## The kernel

One warp per output row. The 32 lanes stride along the reduction dimension in
`uint32` words — 8 int4 codes per load, fully coalesced 128-byte transactions —
dequantise in registers, accumulate in FP32, and close the row with a
`__shfl_down_sync` reduction.

```cuda
for (int w = lane; w < words; w += 32) {
    const uint32_t packed = __ldg(&wrow[w]);
    const float s = __half2float(__ldg(&srow[(w << 3) / GROUP]));
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int code = (packed >> (k * 4)) & 0xF;
        acc = fmaf((float)(code - 8) * s, __half2float(__ldg(&x[(w << 3) + k])), acc);
    }
}
```

`GROUP` is a multiple of 8, so all eight codes in a word share one scale and the
scale load hoists out of the unroll.

No tensor cores, deliberately. At batch 1 `mma` needs M ≥ 8, so seven eighths of
every tile would be padding — and this kernel is memory-bound, which is not a
problem tensor cores solve. The reasoning is recorded in the ADR so it does not
get re-litigated.

## Where this came from, and where it went wrong

[Unibit](https://github.com/QuantumDrizzy/unibit) measured decode at ~6 % of the
bandwidth roofline and attributed the gap to dequantisation arithmetic. This
repository was started to close that gap.

**The harness disproved that premise before any CUDA was written.** Isolating the
GEMV showed bitsandbytes already at 58–70 % of roofline: the dequant-GEMV kernel
is not where the 15× goes. The headline was corrected from 15× to ~1.5× in the
ADR, above the reasoning it invalidates, and where the token-level gap actually
lives became an open question rather than a settled diagnosis.

What did survive the measurement: **fusion is worth 3–6×.** Dequantising to FP16
and calling cuBLAS costs 88.3 µs against 29.1 µs at 3584×3584, and 793.4 µs
against 135.7 µs at 18944×3584.

## Method

- Every timing gated on a correctness check first. The kernel is validated
  against an FP32 CPU reference computed on the **dequantised** weights, so it
  tests the kernel's arithmetic and layout rather than the quantiser's accuracy.
  Max relative error lands at 1.6–2.1 × 10⁻⁴; the gate rejects above 2 × 10⁻².
- CUDA-event timing, 25 warmup iterations, median of 300.
- Bandwidth measured on the device at run time, not quoted from a datasheet.
- The empty-launch floor is measured and stated separately, because at small
  shapes it *is* the measurement.

## Build and run

```bash
scripts\build.bat          # vcvars64 + nvcc -O3 -arch=sm_120
int4_gemv.exe              # kernel   -> bench/kernel.json

python bench/baseline.py   # roofline, cuBLAS FP16, bitsandbytes NF4, control A
python bench/compare.py    # the table above
```

`bench/baseline.py` needs PyTorch and bitsandbytes. The kernel itself needs only
nvcc.

## Limits

- One GPU, one architecture (sm_120). Nothing here is tuned for anything else,
  and the split-K threshold in the ADR is not implemented yet.
- Batch 1 only. This is decode; prefill is a different, compute-bound problem.
- INT4 symmetric group-wise only. The NF4 variant that would make the
  bitsandbytes comparison like-for-like is still to be written.
- No fused activation, no bias, no attention, no KV-cache. This is one kernel,
  not an inference engine.

## Licence

Proprietary. Copyright (c) 2026 Antonio Zambudio Rodriguez, all rights reserved.
See [LICENSE](LICENSE). Visibility of this repository does not grant a licence to
use, copy or derive from it.
