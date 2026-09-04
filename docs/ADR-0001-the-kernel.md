# ADR-0001: A fused INT4 dequantisation GEMV for LLM decode

**Status:** Proposed
**Date:** 2026-09-04
**Deciders:** Antonio Zambudio Rodriguez

---

## Context

Single-token decode in a quantised LLM is not a compute problem. At batch size 1
every projection is a matrix-**vector** product: two floating-point operations
per weight, and every weight read exactly once. Arithmetic intensity is fixed at
~2 flops/byte before quantisation and worse after. The kernel is memory-bound by
construction, and the only honest target is a fraction of achieved bandwidth.

Measured on the machine this will be built on — an RTX 5060 Ti (sm_120 Blackwell,
36 SMs, 16 GB), read bandwidth **384.95 GB/s**, against an Intel Xeon E5-2683 v4
at **39.88 GB/s**:

| Quantity | Value |
|---|---|
| Qwen2.5-Coder-7B at 4 bits + scales | ~3.8 GB of weights per token |
| Roofline at ~426 GB/s | ~8.9 ms/token → **~112 tok/s** |
| Measured decode, bitsandbytes NF4 | **6.5 tok/s** |
| Fraction of roofline achieved | **~6 %** |

A 15× gap at the token level, on a path whose theoretical behaviour is completely
understood. The [Unibit](https://github.com/QuantumDrizzy/unibit) cost study
measured this decode path and attributed the gap to dequantisation arithmetic.

> ### ⚠ That attribution was wrong, and this repository's own harness disproved it
>
> Before writing any CUDA, `bench/baseline.py` measured the isolated GEMV on the
> real projection shapes. **bitsandbytes' NF4 kernel already achieves 58–70 % of
> the roofline** on the projections that dominate the parameter count:
>
> | shape | out × in | roofline | bnb NF4 | % of roofline |
> |---|---:|---:|---:|---:|
> | `q_proj` / `o_proj` | 3584×3584 | 17.0 µs | 29.1 µs | **58 %** |
> | `gate_proj` / `up_proj` | 18944×3584 | 89.7 µs | 135.7 µs | **66 %** |
> | `down_proj` | 3584×18944 | 89.7 µs | 129.0 µs | **70 %** |
> | `k_proj` / `v_proj` | 512×3584 | 2.4 µs | 8.5 µs | 28 % |
>
> The dequant-GEMV kernel is **not** where the 15× goes. At most ~1.7× is
> available here, and on the small GQA projections the measurement is dominated
> by the 5.28 µs empty-launch floor rather than by the kernel at all.
>
> The token-level gap is therefore somewhere else — attention, the KV-cache, the
> Python loop, per-layer framework dispatch. That is now an open question rather
> than a settled diagnosis, and it is tracked as ADR-0002.

The premise is corrected, not abandoned: a hand-written kernel that closes
58 % → ~90 % on the dominant shape is still worth having, and writing one from
scratch that matches a mature library is the demonstration this repository is
for. But the honest headline is **1.5×, not 15×**, and it is stated here before
any code exists so that no later result can be presented as more than it is.

### Constraints

- **One consumer GPU, 16 GB.** No multi-GPU, no NVLink, no datacenter part.
- **sm_120 Blackwell, nvcc 13.0.88**, MSVC toolchain (vcvars64 + nvcc + Ninja;
  the Visual Studio generator does not pick up the CUDA toolset on this machine).
  Note the version skew: the kernels compile against CUDA 13.0, while the PyTorch
  used for baselines is a `2.12.0.dev+cu128` build. Both list sm_120; the skew is
  recorded here so a later discrepancy is not mistaken for a kernel bug.
- **Solo, evenings.** The scope must be finishable, not a serving stack.
- Every number published carries its baseline and its limits.

---

## Decision

Build a **kernel library, not an inference engine**: a small set of CUDA kernels
that compute `y = W·x` where `W` is 4-bit quantised and `x` is FP16, with
dequantisation **fused into the load** so FP16 weights are never materialised in
memory. Ship it with a benchmark harness that reports **percentage of measured
memory bandwidth** as the primary metric, and correctness against an FP32
reference as a gate on every result.

Not in scope: attention, KV-cache, sampling, batching, a server, a Python
framework integration beyond what the benchmark needs.

---

## Options Considered

### Option A: Separate dequantisation kernel, then cuBLAS

Dequantise the whole weight matrix to FP16 into scratch memory, then call
`cublasHgemv`.

| Dimension | Assessment |
|---|---|
| Complexity | Low — two library calls |
| Performance | Poor: writes and re-reads the FP16 matrix, so it moves **4× the bytes** of the fused version |
| Correctness risk | Low |
| What it teaches | Nothing about kernel design |

**Pros:** trivial to write; useful as an intermediate baseline that isolates how
much the fusion itself is worth.
**Cons:** defeats the entire purpose of quantisation at decode time — the memory
saving is spent immediately. Cannot beat FP16 cuBLAS because it *is* FP16 cuBLAS
plus overhead.

**Verdict:** rejected as the design, kept as a measured baseline. It is the
control that proves fusion is the mechanism.

### Option B: Fused dequant-GEMV, one warp per output row *(chosen)*

Each warp owns one row of `W`. The 32 lanes stride along the reduction
dimension, each loading packed 4-bit codes as `uint32_t` (8 weights per load),
dequantising in registers, multiplying against `x` held in shared memory, and
accumulating in FP32. A warp-level `__shfl_down_sync` reduction closes the row.

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Performance | Reads weights exactly once; coalesced; no scratch traffic |
| Occupancy risk | Rows must be numerous enough to fill 36 SMs — true for 3584 and 18944, marginal for the 512-wide GQA projections |
| Team familiarity | Warp reductions and vectorised loads are well-trodden |

**Pros:** the minimum possible byte movement; simple enough to reason about and
to profile in Nsight Compute; degrades gracefully.
**Cons:** for short-and-wide matrices (`k_proj`, `v_proj` at 3584×512) there are
too few rows to saturate the device.

### Option C: Split-K with atomics

Partition the reduction dimension across blocks; each computes a partial dot
product and `atomicAdd`s into `y`.

| Dimension | Assessment |
|---|---|
| Complexity | Medium-high — needs a zeroed output and a deterministic-order story |
| Performance | Recovers occupancy exactly where Option B loses it |
| Reproducibility | FP32 atomics are non-deterministic in ordering |

**Pros:** the right answer for the narrow projections.
**Cons:** non-deterministic summation order makes bit-exact regression tests
impossible; must be validated by tolerance, not equality.

**Verdict:** implement as a **second kernel** selected by shape, not as the
default. Option B for wide matrices, Option C when rows < a tuned threshold.

### Option D: Tensor cores (MMA)

| Dimension | Assessment |
|---|---|
| Applicability | **None at batch 1** |

`mma` instructions require M ≥ 8; decode has M = 1, so seven eighths of every
tile would be padding. Consumer Blackwell also has no INT4 tensor-core path.
Recording this explicitly because "use tensor cores" is the reflexive suggestion
and it is wrong here — the kernel is memory-bound, and tensor cores solve a
compute problem this workload does not have.

**Verdict:** rejected, with reasons, permanently.

---

## Quantisation format

Two formats, for two different jobs:

- **INT4 symmetric, group-wise (group = 128), FP16 scales.** The primary format.
  Dequantisation is one multiply per weight, which is what makes fusion cheap.
  This is what GPTQ/AWQ-style production serving actually uses.
- **NF4 (normal-float 4), 16-entry lookup table.** Implemented second, purely so
  the comparison against bitsandbytes is *the same numerical format* rather than
  a favourable substitution. Without this, any speedup over bitsandbytes is
  confounded by having quietly chosen an easier problem.

The NF4 path is expected to be **slower** than INT4 — a LUT lookup per weight
against a multiply. If it is not, that is a result worth investigating, not
celebrating.

---

## Trade-off analysis

The central trade is **determinism against occupancy**. Option B is exactly
reproducible and leaves the narrow projections underutilised; Option C fills the
device and gives up bit-exact reproducibility. Shipping both, dispatched by
shape, costs a second kernel to maintain and a tolerance-based test for the
split-K path. That is the right price: pretending one decomposition fits every
projection shape in a transformer would be the actual mistake.

The second trade is **format against comparability**. INT4 group-wise is the
faster and more useful format; NF4 is the one that makes the headline comparison
honest. Building only the first would produce a better number and a worse claim.

---

## How results will be reported

Primary metric: **achieved bandwidth as a percentage of 384.95 GB/s measured on
this device**, per projection shape. Speedups are reported only alongside it, and
always name their baseline.

Three baselines, all measured on the same machine in the same session:

1. **bitsandbytes NF4** — the thing a person actually runs today.
2. **cuBLAS FP16 GEMV** — unquantised speed of light for this shape.
3. **The roofline** — bytes moved ÷ measured bandwidth. The floor no kernel beats.

Rules carried over from Unibit, because they are what makes the numbers worth
anything:

- Kernel-only and end-to-end timings reported **separately**, never blended.
- CUDA-event timing, warmed, medians over ≥ 1000 iterations.
- An **empty-kernel launch cost** measured and stated (on this machine an empty
  CUDA matmul costs 45.66 µs — at small shapes that *is* the measurement).
- Every kernel validated against an FP32 CPU reference before any timing is
  published. A fast wrong answer is not a result.
- If a hypothesis fails, the negative result ships. Unibit's headline finding was
  negative; this repository inherits that standard.

---

## Consequences

**Easier:** a defensible answer to "can you write a fast kernel and prove it" —
in the one domain both open contract roles name explicitly. It also closes an arc
that is already half-published: Unibit measured the bottleneck, this fixes it.

**Harder:** two kernels instead of one, and a tolerance-based test for the
non-deterministic path. Windows/MSVC + sm_120 is a narrower toolchain path than
Linux, and **bitsandbytes support for sm_120 is unverified** — if its NF4 kernels
do not run on this device, baseline 1 has to be replaced (with a CPU NF4
reference, or a documented gap).

**To revisit:** whether group size 128 is right (32 costs more scale traffic but
quantises better); whether the split-K threshold generalises past this GPU;
whether an INT8 path is worth adding for comparison.

---

## Verified before accepting this ADR

The one assumption that could have invalidated the benchmark plan — that
bitsandbytes has no working NF4 path on consumer Blackwell — **does not hold.
Checked on the target device:**

```
torch 2.12.0.dev+cu128 · bitsandbytes 0.49.2
NVIDIA GeForce RTX 5060 Ti · sm_120 · 36 SMs · 17.1 GB
arch list includes sm_120

quantize_nf4 on a 3584x3584 fp16 projection:
  packed 6 422 528 B vs fp16 25 690 112 B          -> 4.00x, as specified
  mean |dequant - original|            0.07279      (weights ~ N(0,1))
  relative GEMV error vs fp16 weights  0.0902
```

Baseline 1 stands. The 9 % relative GEMV error is NF4 doing its job at 4 bits and
is the accuracy this kernel must match, not beat.

---

## Action items

1. [x] ~~Confirm bitsandbytes runs NF4 on sm_120~~ — **done, see above.**
2. [x] ~~Harness~~ — `bench/baseline.py`. All seven Qwen2.5-Coder-7B projection
       shapes, FP32 CPU reference as a correctness gate, CUDA-event timing,
       measured bandwidth and empty-launch floor. Results in `bench/baseline.json`.
       **It disproved the premise above; see the warning box.**
3. [x] ~~Option A control (dequant-then-cuBLAS)~~ — **hypothesis held.** It loses
       by 3.0× on 3584×3584 (88.3 µs vs 29.1 µs) and 5.8× on 18944×3584
       (793.4 µs vs 135.7 µs). Fusion is worth 3–6×, which is the one part of the
       original reasoning the measurement supports.
4. [ ] Option B kernel: INT4 group-wise, warp-per-row, FP32 accumulate.
5. [ ] Correctness gate against the FP32 reference before any timing is recorded.
6. [x] ~~Option C: split-K~~ — **implemented, measured, and not used.** The
       diagnosis first: `diag_aspect.cu` holds x fixed while varying warps and
       vice versa, and the answer is warp count (329 → 563 MW/ms from 1184 to
       37888 warps) not x residency (flat from 14 KB up). So the premise for
       Option C was right. The implementation was not worth it: split-K buys
       1.14× over itself at 4 splits on `down_proj` and gives it straight back to
       the memset of `y` and the global atomics, netting 0.7 % against the plain
       kernel. **Block-per-row** does the same job without either — the block's
       warps take strided slices and reduce through shared memory — and wins
       1.05 %, enough to flip `down_proj` from 0.98× to 1.03× against
       bitsandbytes. Split-K stays in the tree as the measured negative.

       Also wrong in this ADR: Option C was called "the right answer for the
       narrow projections". It is not. `k_proj` runs in 8.3 µs against a 5.28 µs
       launch floor, so it is dispatch-bound, and split-K adds a launch. Fusing
       QKV into one kernel is the real fix and is not in scope here.
7. [x] ~~NF4 variant, for the like-for-like comparison~~ — **done, and it cut the
       result.** Against bitsandbytes in its own format (block 64, FP32 absmax,
       canonical NormalFloat table, verified against its `quant_state`) the
       kernel wins by **1.03–1.10× and loses `down_proj` at 0.98×**, not the
       1.23–1.41× the INT4-vs-NF4 comparison showed. Holding the kernel constant
       and changing only the format costs 19–31 %, so most of that 1.4× was the
       format. The suspicion this action item was written to test turned out to
       be correct, and the headline is revised down a second time.
8. [ ] Nsight Compute profile: confirm the kernel is bandwidth-bound and report
       the achieved-vs-peak figure it gives.
9. [ ] README with the bandwidth table, the roofline, and the known limits.
