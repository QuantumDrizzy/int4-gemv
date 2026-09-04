"""Chart for the README: the like-for-like comparison, with the confound shown.

Two panels, deliberately. Left is the fair test — our NF4 against bitsandbytes
NF4, same format and same bytes. Right is what the INT4 format buys on top,
kept visually separate so it cannot be read as kernel performance.

    python bench/plot.py      # -> docs/img/roofline.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, TEXT, GRID = "#ffffff", "#1a1d21", "#d8dce0"
BNB, OURS, INT4, CEIL = "#d95f02", "#0b6ea8", "#6a3d9a", "#e3e8ec"


def main():
    base = json.load(open("bench/baseline.json", encoding="utf-8"))
    kern = json.load(open("bench/kernel.json", encoding="utf-8"))
    kmap = {k["name"]: k for k in kern["shapes"]}
    bw = base["read_bandwidth_gbs"] * 1e9

    keep = ["k_proj", "q_proj", "down_proj", "gate_proj"]
    rows = []
    for s in base["shapes"]:
        if s["name"] not in keep:
            continue
        k = kmap[s["name"]]
        roof_n = k["nf4"]["bytes"] / bw * 1e6
        roof_i = k["int4"]["bytes"] / bw * 1e6
        rows.append((
            f"{s['name']}\n{s['out_features']}x{s['in_features']}",
            roof_n / s["nf4_us"] * 100.0,              # bnb NF4
            roof_n / k["nf4"]["kernel_us"] * 100.0,    # our NF4
            roof_i / k["int4"]["kernel_us"] * 100.0,   # our INT4
        ))

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
        "grid.color": GRID, "font.family": "DejaVu Sans Mono", "font.size": 9.5,
    })

    labels = [r[0] for r in rows]
    xs = np.arange(len(rows))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- left: the fair test ------------------------------------------------
    w = 0.36
    axl.bar(xs, [100] * len(rows), width=0.86, color=CEIL, zorder=1)
    axl.bar(xs - w / 2, [r[1] for r in rows], width=w, color=BNB, zorder=3,
            label="bitsandbytes NF4")
    axl.bar(xs + w / 2, [r[2] for r in rows], width=w, color=OURS, zorder=3,
            label="ours, NF4")
    for x, r in enumerate(rows):
        axl.text(x - w / 2, r[1] + 1.6, f"{r[1]:.0f}%", ha="center", fontsize=8.5, color=BNB)
        axl.text(x + w / 2, r[2] + 1.6, f"{r[2]:.0f}%", ha="center", fontsize=8.5,
                 color=OURS, fontweight="bold")
    axl.set_title("Like-for-like: same NF4 format, same bytes\n"
                  "block 64, FP32 absmax, NormalFloat table",
                  fontsize=10.5, fontweight="bold", pad=10)
    axl.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # ---- right: what the format buys ---------------------------------------
    axr.bar(xs, [100] * len(rows), width=0.86, color=CEIL, zorder=1)
    axr.bar(xs - w / 2, [r[2] for r in rows], width=w, color=OURS, zorder=3,
            label="ours, NF4")
    axr.bar(xs + w / 2, [r[3] for r in rows], width=w, color=INT4, zorder=3,
            label="ours, INT4 g128")
    for x, r in enumerate(rows):
        axr.text(x - w / 2, r[2] + 1.6, f"{r[2]:.0f}%", ha="center", fontsize=8.5, color=OURS)
        axr.text(x + w / 2, r[3] + 1.6, f"{r[3]:.0f}%", ha="center", fontsize=8.5,
                 color=INT4, fontweight="bold")
    axr.set_title("Same kernel, different format\n"
                  "the gap here is the format, not the code",
                  fontsize=10.5, fontweight="bold", pad=10)
    axr.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    for ax in (axl, axr):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", alpha=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axl.set_ylabel("percentage of the measured memory roofline")

    fig.suptitle("int4-gemv - fused 4-bit dequantisation GEMV, batch 1\n"
                 "Qwen2.5-Coder-7B projections, RTX 5060 Ti (sm_120), "
                 "425.9 GB/s measured. Grey is the roofline.",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs("docs/img", exist_ok=True)
    fig.savefig("docs/img/roofline.png", dpi=150, facecolor=BG)
    print("wrote docs/img/roofline.png")


if __name__ == "__main__":
    main()
