#!/usr/bin/env python3
"""
Usage: python3 run_smoke.py [output_dir] [max_workers]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common

TIMELIMIT = {1: 30, 2: 30, 3: 300}
MAX_WORKERS = 2
TOPK_OVERRIDES = {}

SUBSET_CONFIGS = [
    (name, kind, common.verification_path(name, kind))
    for name in ["breast_cancer", "diabetes", "ijcnn"]
    for kind in common.VERIFICATION_TYPES
] + [
    (name, "t200_d5", common.self_trained_path(name, "t200_d5"))
    for name in ["adult", "pimadiabetes"]
]


def main():
    outroot = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(common.REPO_ROOT, "experiment_results", "atva_smoke"))
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_WORKERS
    os.makedirs(outroot, exist_ok=True)

    configs, missing = common.present(SUBSET_CONFIGS)
    common.report_missing(missing)
    if not configs:
        print("No models from the smoke subset are on disk -- nothing to run.")
        sys.exit(1)

    print(f"Enumerating eligible features across {len(configs)} configs "
          f"(loads each model once)...")
    jobs = (
        common.build_problem1_jobs(configs, alpha=1, topk_overrides=TOPK_OVERRIDES)
        + common.build_problem2_jobs(configs)
        + common.build_problem3_jobs(configs)
    )
    by_problem = {p: sum(1 for j in jobs if j["problem"] == p) for p in (1, 2, 3)}
    print(f"Smoke run: {len(jobs)} jobs "
          f"(P1 {by_problem[1]}, P2 {by_problem[2]}, P3 {by_problem[3]}), "
          f"timelimits {TIMELIMIT}, workers={max_workers}")

    t0 = time.time()
    results = common.run_all(jobs, TIMELIMIT, outroot, max_workers=max_workers)
    elapsed = time.time() - t0

    failed = [r for r in results if r.get("returncode") not in (0,)]
    print(f"\nDone in {elapsed/60:.1f} min. "
          f"{len(results)-len(failed)}/{len(results)} jobs exited cleanly.")
    if failed:
        print("Jobs that did not exit cleanly (see run.log in each job dir):")
        for r in failed:
            print(" ", r["tag"])
    report(outroot, max(TIMELIMIT.values()))


def report(outroot, timelimit):
    print(f"\nResults written under: {outroot}")
    for label, args in [
        ("analyze.py", ["analyze.py", outroot]),
        ("plot_figure5.py (robust vs. unrobust)", ["plot_figure5.py", outroot]),
        ("plot_figure6.py (cactus plot)",
         ["plot_figure6.py", outroot, os.path.join(outroot, "figure6_cactus.png"), str(timelimit)]),
        ("tables.py (Table 2 + Table 3)", ["tables.py", outroot]),
        ("make_report.py (combined PDF)",
         ["make_report.py", outroot, os.path.join(outroot, "report.pdf"), str(timelimit)]),
    ]:
        print(f"\n--- {label} ---")
        subprocess.run([sys.executable, os.path.join(HERE, args[0])] + args[1:])


if __name__ == "__main__":
    main()
