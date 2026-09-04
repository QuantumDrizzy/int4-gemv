"""Put the kernels next to the baselines, on equal footing where it exists.

    python bench/baseline.py     # -> bench/baseline.json  (roofline, fp16, bnb NF4)
    int4_gemv.exe                # -> bench/kernel.json    (ours, INT4 and NF4)
    python bench/compare.py
"""
import json


def main():
    base = json.load(open("bench/baseline.json", encoding="utf-8"))
    kern = json.load(open("bench/kernel.json", encoding="utf-8"))
    kmap = {k["name"]: k for k in kern["shapes"]}
    bw = base["read_bandwidth_gbs"] * 1e9

    print("=" * 100)
    print("int4-gemv - kernels vs baselines")
    print("=" * 100)
    print(f"  {base['device']}")
    print(f"  measured read bandwidth  {base['read_bandwidth_gbs']:.1f} GB/s"
          f"   empty-launch floor  {base['empty_launch_us']:.2f} us\n")

    print("  LIKE-FOR-LIKE: our NF4 vs bitsandbytes NF4 - same format, same bytes")
    print("  " + "-" * 96)
    print(f"  {'shape':<11} {'out x in':>13} {'roofline':>10} {'bnb NF4':>10} "
          f"{'our NF4':>10}   {'bnb %BW':>8} {'our %BW':>8}  {'ratio':>6}")
    for s in base["shapes"]:
        k = kmap.get(s["name"])
        if k is None:
            continue
        roof = k["nf4"]["bytes"] / bw * 1e6
        ours = k["nf4"]["kernel_us"]
        pct_o = roof / ours * 100.0
        pct_b = roof / s["nf4_us"] * 100.0
        dims = f"{s['out_features']}x{s['in_features']}"
        flag = "" if ours <= s["nf4_us"] else "   <- bnb wins"
        print(f"  {s['name']:<11} {dims:>13} {roof:9.1f}u {s['nf4_us']:9.1f}u "
              f"{ours:9.1f}u   {pct_b:7.1f}% {pct_o:7.1f}%  {s['nf4_us']/ours:5.2f}x{flag}")

    print("\n  NOT LIKE-FOR-LIKE: our INT4 vs bitsandbytes NF4 - different formats")
    print("  " + "-" * 96)
    print(f"  {'shape':<11} {'out x in':>13} {'roofline':>10} {'bnb NF4':>10} "
          f"{'our INT4':>10}   {'bnb %BW':>8} {'our %BW':>8}  {'ratio':>6}")
    for s in base["shapes"]:
        k = kmap.get(s["name"])
        if k is None:
            continue
        roof = k["int4"]["bytes"] / bw * 1e6
        ours = k["int4"]["kernel_us"]
        dims = f"{s['out_features']}x{s['in_features']}"
        print(f"  {s['name']:<11} {dims:>13} {roof:9.1f}u {s['nf4_us']:9.1f}u "
              f"{ours:9.1f}u   {s['nf4_pct_of_roofline']:7.1f}% "
              f"{roof / ours * 100.0:7.1f}%  {s['nf4_us']/ours:5.2f}x")

    print("\n  OUR OWN TWO FORMATS: same kernel structure, same author, same machine")
    print("  " + "-" * 96)
    print(f"  {'shape':<11} {'our INT4':>12} {'our NF4':>12} "
          f"{'format cost':>13}")
    for s in base["shapes"]:
        k = kmap.get(s["name"])
        if k is None:
            continue
        i, n = k["int4"]["kernel_us"], k["nf4"]["kernel_us"]
        print(f"  {s['name']:<11} {i:11.1f}u {n:11.1f}u {(n / i - 1) * 100:12.1f}%")

    print("\n  What these three tables say")
    print("  " + "-" * 96)
    print("  * The first is the only fair one. Same NF4 format, same block 64,")
    print("    same FP32 absmax, same NormalFloat table, same bytes moved.")
    print("    We win by ~1.1x on most shapes and LOSE on down_proj.")
    print("  * The second is the tempting one and it is confounded: INT4 group-128")
    print("    with FP16 scales moves fewer bytes and decodes with one multiply")
    print("    instead of a lookup. Reporting its ratio as a kernel speedup would")
    print("    be crediting the format to the kernel.")
    print("  * The third isolates exactly that: our own two kernels differ only in")
    print("    format, so the last column is what the NF4 format costs, with the")
    print("    implementation held constant.")
    print("  * k_proj / v_proj sit near the launch floor; most of those rows is")
    print("    dispatch rather than kernel.")


if __name__ == "__main__":
    main()
