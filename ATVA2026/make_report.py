#!/usr/bin/env python3
"""
Combines Figure 5 (robust vs. unrobust), Figure 6 (cactus plot), Table 2 (glitch
prevalence), and Table 3 (glitch magnitudes) into a single multi-page PDF report for
a given results directory.

Usage: python3 make_report.py <output_dir> [save_path.pdf] [timelimit_seconds]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_rows
from common import SELF_TRAINED_DATASETS, SELF_TRAINED_CONFIGS, VERIFICATION_DATASETS
import plot_figure5
import plot_figure6
import tables as tables_mod



TABLE2_CAPTION = ("Features exhibiting glitches under TE_GLITCH(α,i) with α=1. Percentage\n"
                   "of features with glitches across six tabular datasets and six tree\n"
                   "configurations (t = number of trees, d = maximum depth).")
TABLE3_CAPTION = "Glitch magnitudes reported for TE_GLITCH."


def render_table_page(fig, plt, page_title, caption, subtables, footnote=None):
    """subtables: list of (subtitle, col_labels, row_labels, cell_text_rows)."""
    n_caption_lines = caption.count("\n") + 1
    top = 0.80 - 0.035 * (n_caption_lines - 1)   
    fig.subplots_adjust(top=top, bottom=0.08, hspace=0.6)

    fig.suptitle(page_title, fontsize=14, y=0.97)
    fig.text(0.5, 0.90, caption, ha="center", va="top", fontsize=9.5, style="italic")

    n = len(subtables)
    for i, (subtitle, col_labels, row_labels, cell_rows) in enumerate(subtables):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.axis("off")
        ax.set_title(subtitle, fontsize=10, loc="left")
        tbl = ax.table(cellText=cell_rows, rowLabels=row_labels, colLabels=col_labels,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
    if footnote:
        fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=8, color="0.3")


def table2_page(fig, plt, rows):
    self_trained, verif = tables_mod.table2_prevalence(rows)
    dn = tables_mod.DATASET_DISPLAY

    st_cols = [dn[n] for n in SELF_TRAINED_DATASETS]
    st_rows = [tables_mod.CONFIG_DISPLAY[cfg] for cfg in SELF_TRAINED_CONFIGS]
    st_cells = [[tables_mod._fmt_pct(self_trained[cfg][n]) for n in SELF_TRAINED_DATASETS] for cfg in SELF_TRAINED_CONFIGS]

    vf_cols = ["Robust", "Unrobust"]
    vf_rows = [dn[n] for n in VERIFICATION_DATASETS]
    vf_cells = [[tables_mod._fmt_pct(verif[n]["robust"]), tables_mod._fmt_pct(verif[n]["unrobust"])] for n in VERIFICATION_DATASETS]

    render_table_page(fig, plt, "Table 2", TABLE2_CAPTION, [
        ("(a) Self-trained datasets x configs -- % of tested features with a glitch", st_cols, st_rows, st_cells),
        ("(b) Verification models, robust vs. unrobust -- % of tested features with a glitch", vf_cols, vf_rows, vf_cells),
    ], footnote="'--' = config not run")


def table3_page(fig, plt, rows):
    self_trained, verif = tables_mod.table3_magnitudes(rows)
    dn = tables_mod.DATASET_DISPLAY

    st_cols = [dn[n] for n in SELF_TRAINED_DATASETS]
    st_rows = [tables_mod.CONFIG_DISPLAY[cfg] for cfg in SELF_TRAINED_CONFIGS]
    st_cells = [[tables_mod._fmt_mag(self_trained[cfg][n]) for n in SELF_TRAINED_DATASETS] for cfg in SELF_TRAINED_CONFIGS]

    vf_cols = ["Robust", "Unrobust"]
    vf_rows = [dn[n] for n in VERIFICATION_DATASETS]
    vf_cells = [[tables_mod._fmt_mag(verif[n]["robust"]), tables_mod._fmt_mag(verif[n]["unrobust"])] for n in VERIFICATION_DATASETS]

    render_table_page(fig, plt, "Table 3", TABLE3_CAPTION, [
        ("(a) Self-trained datasets x configs -- alpha magnitude", st_cols, st_rows, st_cells),
        ("(b) Verification models, robust vs. unrobust -- alpha magnitude", vf_cols, vf_rows, vf_cells),
    ], footnote="'TO' = timeout, no solution   'unsat' = no glitch exists   '--' = config not run")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 make_report.py <output_dir> [save_path.pdf] [timelimit_seconds]")
        sys.exit(1)
    outroot = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outroot, "report.pdf")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = load_rows(outroot)
    all_times = [r["time"] for r in rows if r["time"] is not None]
    timelimit = float(sys.argv[3]) if len(sys.argv) > 3 else (max(all_times) if all_times else 45.0)

    with PdfPages(save_path) as pdf:
        fig5 = plot_figure5.build_figure(rows, plt)
        if fig5 is not None:
            pdf.savefig(fig5)
            plt.close(fig5)
        else:
            print("Skipping Figure 5: no robust/unrobust Problem-1 data in this results directory.")

        fig6 = plot_figure6.build_figure(rows, timelimit, plt)
        pdf.savefig(fig6)
        plt.close(fig6)

        fig_t2 = plt.figure(figsize=(11, 8.5))
        table2_page(fig_t2, plt, rows)
        pdf.savefig(fig_t2)
        plt.close(fig_t2)

        fig_t3 = plt.figure(figsize=(11, 8.5))
        table3_page(fig_t3, plt, rows)
        pdf.savefig(fig_t3)
        plt.close(fig_t3)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
