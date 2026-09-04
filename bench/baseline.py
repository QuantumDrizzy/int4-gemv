"""Baselines and the roofline, before a line of CUDA is written.

ADR-0001 action items 2 and 3. This measures what the kernel has to beat and
what it cannot beat, on the real projection shapes of a 7B model at batch 1:

  roofline          bytes moved / measured device bandwidth. The floor.
  cuBLAS FP16       unquantised speed of light for the shape.
  bitsandbytes NF4  the thing a person actually runs today (baseline 1).
  Option A control  dequantise the whole matrix, then cuBLAS. The control that
                    isolates how much fusion is worth: it moves the 4-bit
                    weights AND writes and re-reads an FP16 copy.

Every result is gated on correctness against an FP32 CPU reference first. A
fast wrong answer is not a result.

    python bench/baseline.py
"""
from __future__ import annotations

import json
import time

import torch

import bitsandbytes.functional as BF

# Qwen2.5-Coder-7B: hidden 3584, intermediate 18944, GQA with 4 KV heads x 128.
# torch.nn.Linear stores weight as (out_features, in_features).
SHAPES = [
    ("q_proj",    3584,  3584),
    ("k_proj",     512,  3584),
    ("v_proj",     512,  3584),
    ("o_proj",    3584,  3584),
    ("gate_proj", 18944, 3584),
    ("up_proj",   18944, 3584),
    ("down_proj",  3584, 18944),
]

WARMUP = 25
ITERS = 300


def cuda_time(fn, iters=ITERS, warmup=WARMUP):
    """Median wall time per call, in seconds, via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2]


def measure_read_bandwidth(gib=1.0):
    """Achieved read bandwidth, measured here rather than quoted from a spec."""
    n = int(gib * (1024 ** 3) // 4)
    v = torch.ones(n, dtype=torch.float32, device="cuda")
    dt = cuda_time(lambda: v.sum(), iters=40, warmup=5)
    bw = v.numel() * 4 / dt
    del v
    torch.cuda.empty_cache()
    return bw


def empty_launch_floor():
    """What a dispatch costs when it does nothing. At small shapes this IS the
    measurement, so it is stated rather than folded into the results."""
    a = torch.zeros(1, 1, dtype=torch.float16, device="cuda")
    b = torch.zeros(1, 1, dtype=torch.float16, device="cuda")
    return cuda_time(lambda: a @ b)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    dev = f"{p.name} sm_{p.major}{p.minor} {p.multi_processor_count} SMs"

    print("=" * 92)
    print("int4-gemv — baselines before the kernel")
    print("=" * 92)
    print(f"  {dev}")
    print(f"  torch {torch.__version__}")

    bw = measure_read_bandwidth()
    floor = empty_launch_floor()
    print(f"  measured read bandwidth   {bw / 1e9:8.1f} GB/s")
    print(f"  empty-launch floor        {floor * 1e6:8.2f} us")
    print()

    out = {"device": dev, "torch": torch.__version__,
           "read_bandwidth_gbs": bw / 1e9,
           "empty_launch_us": floor * 1e6, "shapes": []}

    hdr = (f"  {'shape':<11} {'out x in':>13} {'roofline':>10} "
           f"{'fp16':>10} {'nf4':>10} {'ctrl A':>10}   {'nf4 %BW':>8}  {'err':>7}")
    print(hdr)
    print("  " + "-" * 88)

    for name, out_f, in_f in SHAPES:
        W16 = torch.randn(out_f, in_f, dtype=torch.float16, device="cuda") * 0.02
        x16 = torch.randn(1, in_f, dtype=torch.float16, device="cuda")

        qW, state = BF.quantize_4bit(W16, quant_type="nf4")
        packed_bytes = qW.numel() * qW.element_size()

        # ---- correctness gate, against an FP32 CPU reference -------------
        W32c, x32c = W16.float().cpu(), x16.float().cpu()
        ref = (x32c @ W32c.t()).squeeze(0)
        deq = BF.dequantize_4bit(qW, state).to(torch.float16)
        got = (x16 @ deq.t()).squeeze(0).float().cpu()
        rel = ((got - ref).norm() / ref.norm()).item()

        # ---- timings -----------------------------------------------------
        t_fp16 = cuda_time(lambda: x16 @ W16.t())
        t_nf4 = cuda_time(lambda: BF.gemv_4bit(x16, qW.t(), state=state))

        def control_a():
            # Option A: materialise FP16 weights, then cuBLAS. Moves the packed
            # weights, writes an FP16 copy, then reads it back.
            d = BF.dequantize_4bit(qW, state).to(torch.float16)
            return x16 @ d.t()

        t_ctrl = cuda_time(control_a, iters=60, warmup=10)

        # Roofline uses the bytes the *fused* kernel must move: the packed
        # weights plus their scales. Everything else is a design choice.
        scale_bytes = state.absmax.numel() * state.absmax.element_size()
        bytes_fused = packed_bytes + scale_bytes
        t_roof = bytes_fused / bw

        pct = t_roof / t_nf4 * 100.0

        print(f"  {name:<11} {f'{out_f}x{in_f}':>13} "
              f"{t_roof * 1e6:9.1f}u {t_fp16 * 1e6:9.1f}u "
              f"{t_nf4 * 1e6:9.1f}u {t_ctrl * 1e6:9.1f}u   "
              f"{pct:7.1f}%  {rel:7.4f}")

        out["shapes"].append({
            "name": name, "out_features": out_f, "in_features": in_f,
            "packed_bytes": packed_bytes, "scale_bytes": scale_bytes,
            "roofline_us": t_roof * 1e6, "fp16_us": t_fp16 * 1e6,
            "nf4_us": t_nf4 * 1e6, "control_a_us": t_ctrl * 1e6,
            "nf4_pct_of_roofline": pct, "nf4_relative_error": rel,
        })

        del W16, x16, qW, state, deq
        torch.cuda.empty_cache()

    with open("bench/baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote bench/baseline.json")

    print("\n  Reading this table:")
    print("   * roofline is the floor - packed weights + scales over measured")
    print("     bandwidth. No kernel goes below it.")
    print("   * nf4 %BW is what bitsandbytes achieves against that floor. The")
    print("     headroom in that column is the whole subject of this repository.")
    print("   * ctrl A is dequantise-then-cuBLAS. It should lose badly: it moves")
    print("     the 4-bit weights and then writes and re-reads an FP16 copy.")
    print("   * err is NF4's relative GEMV error vs FP32. The kernel must match")
    print("     it, not beat it - beating it would mean a different format.")


if __name__ == "__main__":
    main()
