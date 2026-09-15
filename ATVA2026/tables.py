#!/usr/bin/env python3
"""
Reproduces ATVA26 Table 2 (tab:glitch_count) and Table 3 (tab:robustUnrobustGlitchMagnitudes).

Table 2: percentage of features exhibiting a glitch under TE_GLITCH(alpha=1, i) -- Problem 1.
  (a) self-trained datasets x {t,d} configs
  (b) verification models, robust vs. unrobust

Table 3: glitch magnitude (the alpha found) under TE_GLITCH-max -- Problem 3, "TO" for timeout.
  (a) self-trained datasets x {t,d} configs
  (b) verification models, robust vs. unrobust

Both tables report whatever configs are present in the given results directory -- entries for
configs that weren't run at all show as blank ("--"), distinct from "TO" (ran, timed out).

Usage: python3 tables.py <output_dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_rows
from common import SELF_TRAINED_DATASETS, SELF_TRAINED_CONFIGS, VERIFICATION_DATASETS

DATASET_DISPLAY = {
    "adult": "Adult", "churn": "Churn", "pimadiabetes": "Pima Diabetes",
    "german_credit": "German Credit", "breast_cancer": "Breast Cancer", "spambase": "Spambase",
    "diabetes": "Diabetes", "ijcnn": "IJCNN", "webspam": "Webspam",
    "binary_mnist": "Binary MNIST", "higgs": "Higgs",
}


def _paper_config_label(cfg):
    t_part, d_part = cfg.split("_")
    return f"{t_part}_d{int(d_part[1:]) - 1}"


CONFIG_DISPLAY = {cfg: _paper_config_label(cfg) for cfg in SELF_TRAINED_CONFIGS}


def has_incumbent(row):
    return row.get("threepoints") not in (None, "", "[]")


def table2_prevalence(rows):
    """Returns (self_trained_grid, verif_grid): {row_key: {col_key: pct_or_None}}."""
    p1 = [r for r in rows if r["problem"] == 1]

    def pct_for(name, cfg):
        matches = [r for r in p1 if r["modelname"] == name and r["modeltype"] == cfg]
        if not matches:
            return None
        found = sum(1 for r in matches if r["sat"] == "sat")
        return 100 * found / len(matches)

    self_trained = {
        cfg: {name: pct_for(name, cfg) for name in SELF_TRAINED_DATASETS}
        for cfg in SELF_TRAINED_CONFIGS
    }
    verif = {
        name: {kind: pct_for(name, kind) for kind in ("robust", "unrobust")}
        for name in VERIFICATION_DATASETS
    }
    return self_trained, verif


def table3_magnitudes(rows):

    p3 = [r for r in rows if r["problem"] == 3]

    def cell_for(name, cfg):
        matches = [r for r in p3 if r["modelname"] == name and r["modeltype"] == cfg]
        if not matches:
            return None
        r = matches[0]
        if r["sat"] == "unsat":
            return "unsat"
        if r["sat"] == "sat" or (r["sat"] == "timelimit" and has_incumbent(r)):
            return float(r["alpha_found"])
        return "TO"

    self_trained = {
        cfg: {name: cell_for(name, cfg) for name in SELF_TRAINED_DATASETS}
        for cfg in SELF_TRAINED_CONFIGS
    }
    verif = {
        name: {kind: cell_for(name, kind) for kind in ("robust", "unrobust")}
        for name in VERIFICATION_DATASETS
    }
    return self_trained, verif


def _fmt_pct(v):
    return "--" if v is None else f"{v:.2f}"


def _fmt_mag(v):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    return f"{v:.2f}"


def print_table2(rows):
    self_trained, verif = table2_prevalence(rows)
    print("\n=== Table 2: Features exhibiting glitches under TE_GLITCH(alpha=1, i) ===")
    print("\n(a) Self-trained datasets x configs (% of tested features with a glitch)")
    header = "Config".ljust(10) + "".join(DATASET_DISPLAY[n].ljust(16) for n in SELF_TRAINED_DATASETS)
    print(header)
    for cfg in SELF_TRAINED_CONFIGS:
        row = CONFIG_DISPLAY[cfg].ljust(10) + "".join(_fmt_pct(self_trained[cfg][n]).ljust(16) for n in SELF_TRAINED_DATASETS)
        print(row)

    print("\n(b) Verification models, robust vs. unrobust (% of tested features with a glitch)")
    print(f"{'Model':<16}{'Robust':<12}{'Unrobust'}")
    for name in VERIFICATION_DATASETS:
        print(f"{DATASET_DISPLAY[name]:<16}{_fmt_pct(verif[name]['robust']):<12}{_fmt_pct(verif[name]['unrobust'])}")


def print_table3(rows):
    self_trained, verif = table3_magnitudes(rows)
    print("\n=== Table 3: Glitch magnitudes under TE_GLITCH (max) ===")
    print("\n(a) Self-trained datasets x configs (alpha magnitude, 'TO' = timeout no solution, 'unsat' = no glitch)")
    header = "Config".ljust(10) + "".join(DATASET_DISPLAY[n].ljust(16) for n in SELF_TRAINED_DATASETS)
    print(header)
    for cfg in SELF_TRAINED_CONFIGS:
        row = CONFIG_DISPLAY[cfg].ljust(10) + "".join(_fmt_mag(self_trained[cfg][n]).ljust(16) for n in SELF_TRAINED_DATASETS)
        print(row)

    print("\n(b) Verification models, robust vs. unrobust (alpha magnitude)")
    print(f"{'Model':<16}{'Robust':<12}{'Unrobust'}")
    for name in VERIFICATION_DATASETS:
        print(f"{DATASET_DISPLAY[name]:<16}{_fmt_mag(verif[name]['robust']):<12}{_fmt_mag(verif[name]['unrobust'])}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tables.py <output_dir>")
        sys.exit(1)
    rows = load_rows(sys.argv[1])
    print_table2(rows)
    print_table3(rows)


if __name__ == "__main__":
    main()
