#!/usr/bin/env python3
"""
Full-scale run of the ATVA26 glitch experiments through Ensense
(src/sensitive.py --solver glitch), across every (model, config) combination at
the paper's own 3600s per-instance limit.  Multi-hour to multi-day depending on
hardware -- see run_smoke.py for a representative subset.

  Problem 1 (TE_GLITCH(alpha,i), alpha=1):  one job per eligible feature per config
  Problem 2 (TE_GLITCH(alpha)):             one job per config per alpha threshold
  Problem 3 (TE_GLITCH max):                one job per config

The self-trained grid is discovered from models/ rather than asserted, so a dataset
is only asked for configs it actually has.  Anything absent is reported and shows as
'--' in Tables 2 and 3, distinct from 'TO' (ran, timed out).

Safe to interrupt and re-run: any job whose directory already has results.csv is
skipped.

Usage: python3 run_full.py [output_dir] [max_workers]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from run_smoke import report

TIMELIMIT = {1: 3600, 2: 3600, 3: 3600}


def main():
    outroot = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(common.REPO_ROOT, "experiment_results", "atva_full"))
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    os.makedirs(outroot, exist_ok=True)

    configs, missing = common.present(common.all_configs())
    common.report_missing(missing)
    print(f"Enumerating eligible features across {len(configs)} configs "
          f"(loads each model once)...")

    jobs = (
        common.build_problem1_jobs(configs, alpha=1)
        + common.build_problem2_jobs(configs)
        + common.build_problem3_jobs(configs)
    )
    by_problem = {p: sum(1 for j in jobs if j["problem"] == p) for p in (1, 2, 3)}
    worst_h = sum(TIMELIMIT[j["problem"]] for j in jobs) / 3600 / max_workers
    print(f"Total jobs: {len(jobs)} (P1 {by_problem[1]}, P2 {by_problem[2]}, "
          f"P3 {by_problem[3]}); worst case {worst_h:.1f}h with {max_workers} workers")

    t0 = time.time()
    results = common.run_all(jobs, TIMELIMIT, outroot, max_workers=max_workers)
    elapsed = time.time() - t0

    skipped = sum(1 for r in results if r.get("skipped"))
    failed = [r for r in results if not r.get("skipped") and r.get("returncode") != 0]
    print(f"\nDone in {elapsed/3600:.2f}h. {skipped} skipped (already done), "
          f"{len(failed)} did not exit cleanly, "
          f"{len(results)-skipped-len(failed)} ran cleanly.")
    if failed:
        print("Jobs that did not exit cleanly (see run.log in each job dir):")
        for r in failed:
            print(" ", r["tag"])
    report(outroot, max(TIMELIMIT.values()))


if __name__ == "__main__":
    main()
