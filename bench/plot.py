"""Chart for the README: each kernel against the memory roofline.

Plots %BW, not the ratio. The two formats do not move the same bytes, so each
is measured against its own floor — see the note in the README.

    python bench/plot.py      # -> docs/img/roofline.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, TEXT, GRID = "#ffffff", "#1a1d21", "#d8dce0"
OURS, BNB, CEIL = "#0b6ea8", "#d95f02", "#c9d6e0"


def main():
    base = json.load(open("bench/baseline.json", encoding="utf-8"))
    kern = json.load(open("bench/kernel.json", encoding="utf-8"))
    kmap = {k["name"]: k for k in kern["shapes"]}
    bw = base["read_bandwidth_gbs"] * 1e9

    # One row per distinct shape; q/o and gate/up are duplicates.
    keep = ["k_proj", "q_proj", "down_proj", "gate_proj"]
    rows = []
    for s in base["shapes"]:
        if s["name"] not in keep:
            continue
        k = kmap[s["name"]]
        rows.append((
            f"{s['name']}\n{s['out_features']}x{s['in_features']}",
            s["nf4_pct_of_roofline"],
            k["bytes"] / bw * 1e6 / k["kernel_us"] * 100.0,
        ))

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
        "grid.color": GRID, "font.family": "DejaVu Sans Mono", "font.size": 10,
    })

    labels = [r[0] for r in rows]
    bnb = [r[1] for r in rows]
    ours = [r[2] for r in rows]
    xs = np.arange(len(rows))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(xs, [100] * len(rows), width=0.84, color=CEIL, zorder=1,
           label="memory roofline (no kernel goes above)")
    ax.bar(xs - w / 2, bnb, width=w, color=BNB, zorder=3, label="bitsandbytes NF4")
    ax.bar(xs + w / 2, ours, width=w, color=OURS, zorder=3, label="ours (INT4 g128)")

    for x, (b, o) in enumerate(zip(bnb, ours)):
        ax.text(x - w / 2, b + 1.6, f"{b:.0f}%", ha="center", fontsize=9, color=BNB)
        ax.text(x + w / 2, o + 1.6, f"{o:.0f}%", ha="center", fontsize=9,
                color=OURS, fontweight="bold")

    # k_proj is dominated by dispatch, not by the kernel. Say so on the chart
    # rather than letting the short bars read as a failure.
    k_idx = labels.index(next(l for l in labels if l.startswith("k_proj")))
    ax.annotate("launch-floor bound\n(5.28 us empty dispatch)",
                xy=(k_idx, 37), xytext=(k_idx, 66), ha="center", fontsize=8.5,
                style="italic", color="#5b6672",
                arrowprops=dict(arrowstyle="->", color="#5b6672", lw=0.9))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("percentage of the measured memory roofline")
    ax.set_ylim(0, 112)
    ax.set_title("int4-gemv - fused INT4 dequantisation GEMV, batch 1\n"
                 "Qwen2.5-Coder-7B projections on RTX 5060 Ti (sm_120), "
                 "425.9 GB/s measured",
                 fontsize=11, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    os.makedirs("docs/img", exist_ok=True)
    fig.savefig("docs/img/roofline.png", dpi=150, facecolor=BG)
    print("wrote docs/img/roofline.png")


if __name__ == "__main__":
    main()
