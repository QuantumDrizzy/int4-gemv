"""Put the kernel next to the baselines, and say plainly what is not comparable.

    python bench/baseline.py     # -> bench/baseline.json  (roofline, fp16, bnb NF4)
    int4_gemv.exe                # -> bench/kernel.json    (this repository)
    python bench/compare.py
"""
import json


def main():
    base = json.load(open("bench/baseline.json", encoding="utf-8"))
    kern = json.load(open("bench/kernel.json", encoding="utf-8"))
    kmap = {k["name"]: k for k in kern["shapes"]}

    bw = base["read_bandwidth_gbs"]
    print("=" * 96)
    print("int4-gemv - kernel vs baselines")
    print("=" * 96)
    print(f"  {base['device']}")
    print(f"  measured read bandwidth  {bw:.1f} GB/s"
          f"   empty-launch floor  {base['empty_launch_us']:.2f} us\n")

    print(f"  {'shape':<11} {'out x in':>13} {'roofline':>10} {'bnb NF4':>10} "
          f"{'ours':>10}   {'bnb %BW':>8} {'ours %BW':>9}  {'ratio':>6}")
    print("  " + "-" * 92)

    for s in base["shapes"]:
        k = kmap.get(s["name"])
        if k is None:
            continue
        # Roofline recomputed from THIS kernel's byte count: the two formats do
        # not move the same number of bytes, so each is measured against its own
        # floor rather than against a single borrowed one.
        roof_ours = k["bytes"] / (bw * 1e9) * 1e6
        pct_ours = roof_ours / k["kernel_us"] * 100.0
        ratio = s["nf4_us"] / k["kernel_us"]
        dims = f"{s['out_features']}x{s['in_features']}"
        print(f"  {s['name']:<11} {dims:>13} "
              f"{roof_ours:9.1f}u {s['nf4_us']:9.1f}u {k['kernel_us']:9.1f}u   "
              f"{s['nf4_pct_of_roofline']:7.1f}% {pct_ours:8.1f}%  {ratio:5.2f}x")

    print("\n  What this table does and does not say")
    print("  " + "-" * 92)
    print("  * ours is INT4 symmetric, group 128, one FP16 scale per group.")
    print("    bnb is NF4 with double quantisation - a different format, and the")
    print("    harder decode: a 16-entry lookup against a single multiply.")
    print("  * So the ratio column is NOT 'faster than bitsandbytes'. It is two")
    print("    formats measured on the same shapes and hardware. The like-for-like")
    print("    comparison needs the NF4 variant of this kernel, which is not")
    print("    written yet (ADR-0001, action item 7).")
    print("  * %BW is each kernel against its own byte count over measured")
    print("    bandwidth. That column is comparable; the ratio column is not.")
    print("  * k_proj / v_proj sit near the 5.28 us launch floor, so most of that")
    print("    row is dispatch rather than kernel.")


if __name__ == "__main__":
    main()
