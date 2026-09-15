#!/usr/bin/env python3
"""
Aggregate results.csv files from a run_smoke.py / run_full.py output directory into
RQ1 (prevalence), RQ2 (robust vs unrobust), and RQ3 (scalability) style tables.

Usage: python3 analyze.py <output_dir>
"""
import csv
import os
import statistics
import sys

VERIFICATION_PAIRS = ["binary_mnist", "breast_cancer", "diabetes", "higgs", "ijcnn", "webspam"]


def load_rows(outroot):

    rows = []
    for tag in sorted(os.listdir(outroot)):
        job_dir = os.path.join(outroot, tag)
        if not os.path.isdir(job_dir):
            continue
        csv_path = os.path.join(job_dir, "results.csv")
        base = {"tag": tag, "sat": "MISSING", "time": None, "alpha_found": None,
                "problem": None, "modelname": None, "modeltype": None, "alpha": None}
        if not os.path.isfile(csv_path):
            rows.append(base)
            continue
        with open(csv_path) as f:
            r = next(csv.DictReader(f), None)
        if r is None:
            rows.append({**base, "sat": "EMPTY"})
            continue
        rows.append({
            "tag": tag,
            "problem": int(r["problem"]) if r.get("problem") not in (None, "") else None,
            "features": [int(r["feature"])] if r.get("feature") not in (None, "") else None,
            "modelname": r.get("modelname"),
            "modeltype": r.get("modeltype"),
            "alpha": r.get("givenalpha"),
            "sat": r.get("sat"),
            "time": float(r["time"]) if r.get("time") not in (None, "") else None,
            "alpha_found": r.get("alpha"),
            "threepoints": r.get("threepoints", ""),
        })
    return rows


def rq1_prevalence(rows):
    p1 = [r for r in rows if r["problem"] == 1]
    if not p1:
        return
    print("\n=== RQ1: Prevalence -- TE_GLITCH(alpha=1, i), fixed feature ===")
    found = sum(1 for r in p1 if r["sat"] == "sat")
    print(f"Overall: {found}/{len(p1)} feature-instances have a glitch ({100*found/len(p1):.1f}%)")

    per_model = {}
    for r in p1:
        key = (r["modelname"], r["modeltype"])
        per_model.setdefault(key, [0, 0])
        per_model[key][1] += 1
        if r["sat"] == "sat":
            per_model[key][0] += 1
    print(f"{'model':<20}{'config':<10}{'found/total':<14}{'%'}")
    for (name, cfg), (f, t) in sorted(per_model.items()):
        print(f"{name:<20}{cfg:<10}{f}/{t:<12}{100*f/t:.1f}%")

    p2 = [r for r in rows if r["problem"] == 2]
    if p2:
        print("\n--- TE_GLITCH(alpha), any feature, by alpha threshold ---")
        by_alpha = {}
        for r in p2:
            by_alpha.setdefault(r["alpha"], [0, 0])
            by_alpha[r["alpha"]][1] += 1
            if r["sat"] == "sat":
                by_alpha[r["alpha"]][0] += 1
        for a, (f, t) in sorted(by_alpha.items(), key=lambda kv: float(kv[0])):
            print(f"  alpha={a:<10} {f}/{t} solved-with-glitch")

    p3 = [r for r in rows if r["problem"] == 3]
    if p3:
        print("\n--- TE_GLITCH (max), any feature ---")
        found3 = sum(1 for r in p3 if r["sat"] in ("sat",) or r.get("threepoints") not in (None, "", "[]"))
        print(f"  {found3}/{len(p3)} instances found some glitch (proven-optimal or timeout-with-incumbent)")


def rq2_robust_vs_unrobust(rows):
    p1 = [r for r in rows if r["problem"] == 1 and r["modeltype"] in ("robust", "unrobust")]
    if not p1:
        return
    print("\n=== RQ2: Robust vs Unrobust -- % of features that are glitch-bearing ===")
    stats = {}
    for r in p1:
        key = (r["modelname"], r["modeltype"])
        stats.setdefault(key, [0, 0])
        stats[key][1] += 1
        if r["sat"] == "sat":
            stats[key][0] += 1
    print(f"{'dataset':<16}{'robust %':<14}{'unrobust %'}")
    for name in VERIFICATION_PAIRS:
        rk, uk = (name, "robust"), (name, "unrobust")
        if rk not in stats or uk not in stats:
            continue
        rf, rt = stats[rk]
        uf, ut = stats[uk]
        print(f"{name:<16}{f'{100*rf/rt:.1f}% ({rf}/{rt})':<14}{100*uf/ut:.1f}% ({uf}/{ut})")


def rq3_scalability(rows):
    print("\n=== RQ3: Scalability ===")
    for problem, label in [(1, "TE_GLITCH(alpha,i)"), (2, "TE_GLITCH(alpha)"), (3, "TE_GLITCH (max)")]:
        p = [r for r in rows if r["problem"] == problem]
        if not p:
            continue
        times = [r["time"] for r in p if r["time"] is not None]
        n_sat = sum(1 for r in p if r["sat"] == "sat")
        n_timelimit_sol = sum(1 for r in p if r["sat"] == "timelimit" and r.get("threepoints") not in (None, "", "[]"))
        n_timelimit_nosol = sum(1 for r in p if r["sat"] == "timelimit" and r.get("threepoints") in (None, "", "[]"))
        n_unsat = sum(1 for r in p if r["sat"] == "unsat")
        print(f"\n{label}: {len(p)} instances")
        print(f"  proven sat/optimal: {n_sat}   proven unsat: {n_unsat}   "
              f"timeout w/ incumbent: {n_timelimit_sol}   timeout no solution: {n_timelimit_nosol}")
        if times:
            print(f"  solve time: min={min(times):.1f}s  median={statistics.median(times):.1f}s  "
                  f"max={max(times):.1f}s  mean={statistics.mean(times):.1f}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <output_dir>")
        sys.exit(1)
    outroot = sys.argv[1]
    rows = load_rows(outroot)
    missing = sum(1 for r in rows if r["sat"] in ("MISSING", "EMPTY"))
    print(f"Loaded {len(rows)} job records from {outroot} ({missing} missing/empty -- run may be incomplete)")

    rq1_prevalence(rows)
    rq2_robust_vs_unrobust(rows)
    rq3_scalability(rows)


if __name__ == "__main__":
    main()
