#!/usr/bin/env python3
"""
Usage: python3 plot_figure5.py <output_dir> [save_path.png]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_rows

ROBUST_COLOR = "#1f77b4"
UNROBUST_COLOR = "#d98419"

DATASET_ORDER = ["breast_cancer", "diabetes", "ijcnn", "webspam", "higgs", "binary_mnist"]
DISPLAY_NAME = {
    "breast_cancer": "Breast\nCancer", "diabetes": "Diabetes", "ijcnn": "IJCNN",
    "webspam": "WEBSPAM", "higgs": "HIGGS", "binary_mnist": "BINARY\nMNIST",
}

CAPTION = ("Robust vs. unrobust models. Robust training reduces but does not "
           "eliminate glitch-exhibiting features.")


def robust_vs_unrobust_pct(rows):
    p1 = [r for r in rows if r["problem"] == 1 and r["modeltype"] in ("robust", "unrobust")]
    stats = {}
    for r in p1:
        key = (r["modelname"], r["modeltype"])
        stats.setdefault(key, [0, 0])
        stats[key][1] += 1
        if r["sat"] == "sat":
            stats[key][0] += 1
    return stats


def build_figure(rows, plt):
    stats = robust_vs_unrobust_pct(rows)

    names = [n for n in DATASET_ORDER if (n, "robust") in stats or (n, "unrobust") in stats]
    if not names:
        return None

    def count(name, kind):
        f, _t = stats.get((name, kind), (0, 0))
        return f

    robust_vals = [count(n, "robust") for n in names]
    unrobust_vals = [count(n, "unrobust") for n in names]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(names)), 5.2))

    ax.bar([i - width / 2 for i in x], robust_vals, width, color=ROBUST_COLOR, label="robust")
    ax.bar([i + width / 2 for i in x], unrobust_vals, width, color=UNROBUST_COLOR, label="unrobust")

    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY_NAME.get(n, n) for n in names], fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Number of features exhibiting glitches (per feature, $\\alpha=1$)", fontsize=11)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    ax.grid(True, axis="y", linewidth=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.01, CAPTION, ha="center", va="bottom", fontsize=9, wrap=True)
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_figure5.py <output_dir> [save_path.png]")
        sys.exit(1)
    outroot = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outroot, "figure5_rob_vs_unrob.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(outroot)
    fig = build_figure(rows, plt)
    if fig is None:
        print("No robust/unrobust Problem-1 data found in this results directory.")
        sys.exit(1)

    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
