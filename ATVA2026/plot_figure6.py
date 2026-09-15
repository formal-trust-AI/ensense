#!/usr/bin/env python3
"""
Usage: python3 plot_figure6.py <output_dir> [save_path.png] [timelimit_seconds]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_rows

SOLVED_COLOR = "#2ca02c"
PURPLE = "#800080"
RED = "#d62728"

ALPHA_THRESHOLDS = [5, 25, 125, 625, 3125, 15625]
ALPHA_COLORS = {
    5: "#2a78d6", 25: "#eb6834", 125: "#1baf7a",
    625: "#eda100", 3125: "#e87ba4", 15625: "#008300",
}

PANELS = [
    (1, "(a) TE_GLITCH(α, i)"),
    (2, "(b) TE_GLITCH(α)"),
    (3, "(c) TE_GLITCH (max)"),
]

CAPTION = ("Cactus plot reports the runtime (a) for TE_GLITCH(α,i), (b) for TE_GLITCH(α) "
           "and (c) TE_GLITCH.")


def set_sensible_ylim(ax, all_ys, timelimit):
   
    if not all_ys:
        return
    ymin, ymax = min(all_ys), max(all_ys)
    if ymin == ymax:
        ax.set_ylim(ymin * 0.1, max(ymax * 1.2, timelimit * 1.1))
    else:
        ax.set_ylim(ymin * 0.5, max(ymax, timelimit) * 1.2)


def has_incumbent(row):
    return row.get("threepoints") not in (None, "", "[]")


def panel_data(rows, problem):
    prob_rows = [r for r in rows if r["problem"] == problem and r["time"] is not None]
    solved = sorted(r["time"] for r in prob_rows if r["sat"] in ("sat", "unsat"))
    if problem == 3:
        purple = sum(1 for r in prob_rows if r["sat"] == "timelimit" and has_incumbent(r))
        red = sum(1 for r in prob_rows if r["sat"] == "timelimit" and not has_incumbent(r))
    else:
        purple = 0
        red = sum(1 for r in prob_rows if r["sat"] == "timelimit")
    return solved, purple, red, len(prob_rows)


def panel_b_data_by_alpha(rows):
    """One (solved_times, n_timeout, n_total) tuple per alpha threshold present."""
    prob_rows = [r for r in rows if r["problem"] == 2 and r["time"] is not None]
    out = {}
    for a in ALPHA_THRESHOLDS:
        a_rows = [r for r in prob_rows if int(r["alpha"]) == a]
        if not a_rows:
            continue
        solved = sorted(r["time"] for r in a_rows if r["sat"] in ("sat", "unsat"))
        n_timeout = sum(1 for r in a_rows if r["sat"] == "timelimit")
        out[a] = (solved, n_timeout, len(a_rows))
    return out


def build_figure(rows, timelimit, plt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (problem, title) in zip(axes, PANELS):
        if problem == 2:
            by_alpha = panel_b_data_by_alpha(rows)
            n_solved_total = sum(len(s) for s, _, _ in by_alpha.values())
            n_total = sum(t for _, _, t in by_alpha.values())
            all_ys = []
            for a in ALPHA_THRESHOLDS:
                if a not in by_alpha:
                    continue
                solved, n_timeout, a_total = by_alpha[a]
                color = ALPHA_COLORS[a]
                n_s = len(solved)
                xs = list(range(1, n_s + 1))
                ys = list(solved)
                if n_timeout:
                    xs += list(range(n_s + 1, n_s + 1 + n_timeout))
                    ys += [timelimit] * n_timeout
                all_ys += ys
                ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=3,
                        label=f"α={a}", zorder=3)
            ax.axhline(timelimit, color="gray", linewidth=1, linestyle="--", zorder=1, label="time limit")
            ax.set_yscale("log")
            set_sensible_ylim(ax, all_ys, timelimit)
            ax.set_xlabel("Number of Instances")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"{title}", fontsize=13, style="italic")
            ax.grid(True, which="both", linewidth=0.5, color="0.85", zorder=0)
            ax.legend(fontsize=8, loc="lower right", frameon=True)
            continue

        solved, n_purple, n_red, n_total = panel_data(rows, problem)
        n_solved = len(solved)
        xs = list(range(1, n_solved + 1))
        ys = list(solved)
        if n_purple or n_red:
            xs += list(range(n_solved + 1, n_solved + 1 + n_purple + n_red))
            ys += [timelimit] * (n_purple + n_red)

        seg_bounds = [("SAT", 0, n_solved, SOLVED_COLOR)]
        if n_purple:
            seg_bounds.append(("feasible sol", n_solved, n_solved + n_purple, PURPLE))
        if n_red:
            seg_bounds.append(("timeout", n_solved + n_purple, n_solved + n_purple + n_red, RED))

        for label, lo, hi, color in seg_bounds:
            lo_i = max(lo - 1, 0)
            hi_i = min(hi + 1, len(xs))
            if hi_i - lo_i < 1:
                continue
            ax.plot(xs[lo_i:hi_i], ys[lo_i:hi_i], color=color, linewidth=2.2,
                     marker="o", markersize=3, label=label, zorder=3)

        ax.axhline(timelimit, color="gray", linewidth=1, linestyle="--", zorder=1, label="time limit")
        ax.set_yscale("log")
        set_sensible_ylim(ax, ys, timelimit)
        ax.set_xlabel("Number of Instances")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{title}", fontsize=13, style="italic")
        ax.grid(True, which="both", linewidth=0.5, color="0.85", zorder=0)
        ax.legend(fontsize=9, loc="lower right", frameon=True)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.01, CAPTION, ha="center", va="bottom", fontsize=9, wrap=True)
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_figure6.py <output_dir> [save_path.png] [timelimit_seconds]")
        sys.exit(1)
    outroot = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outroot, "figure6_cactus.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(outroot)
    all_times = [r["time"] for r in rows if r["time"] is not None]
    timelimit = float(sys.argv[3]) if len(sys.argv) > 3 else (max(all_times) if all_times else 45.0)

    fig = build_figure(rows, timelimit, plt)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}  (timelimit used for unsolved points: {timelimit:.1f}s)")


if __name__ == "__main__":
    main()
