#!/usr/bin/env python3
"""Unit-test runner for TriST sensitivity workflows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import random
from modeldetails import MODELDETAILS, GAP_LB, GAP_UB

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = ROOT / "src" / "sensitive.py"

#! running it for same gap lb and ub .. need to think to convert gap -> lb and ub

@dataclass
class TestCase:
    name: str
    args: list[str]

def _feature_set(model_cfg: dict[str, object]) -> tuple[str, list[str]]:
    # includes only single testing
    total_features = int(model_cfg.get("feature", 0))
    if total_features <= 0:
        sys.exit(1)
    feature_list = [f for f in range(0,total_features)]
    single_options = feature_list
    multi_options = [single_options]    
    return single_options, multi_options

def _output_gap(gap) -> tuple[str, str]:
    gap_options = []
    lb = str(random.choice(GAP_LB))
    ub = str(random.choice(GAP_UB))
    gap_options.append([lb,ub])
    return gap_options
    
def one_cases(model_name: str,mode,timeout=100) -> list[TestCase]:
    model = MODELDETAILS[model_name]
    single_feat, multi_feats = _feature_set(model)
    single_baselist = [model["model"], "--features", single_feat, '--timeout', str(timeout)]
    multi_baselist = [model["model"], "--features", *multi_feats, '--timeout', str(timeout)]
    lb,ub = _output_gap()
    core = [
        # basic
        TestCase(f"{mode}.{model_name}.singlefeat", single_baselist),
        TestCase(f"{mode}.{model_name}.multifeat", multi_baselist),
        #for detail file
        TestCase(f"{mode}.{model_name}.singlefeat_details", single_baselist + 
                 ["--details", model["details"]],
        ),
        TestCase(f"{mode}.{model_name}.multifeat", multi_baselist + 
                 ["--details", model["details"]],
        ),
        # for allopt
        TestCase(
            f"{mode}.{model_name}.singlefeat_allopt", single_baselist +
            [
                "--details", model["details"], "--all_opt",
            ],
        ),
        TestCase(
            f"{mode}.{model_name}.multifeat_allopt", multi_baselist +
            [
                "--details", model["details"], "--all_opt",
            ],
        ),
        # for prob-data-aware
        TestCase(
            f"{mode}.{model_name}.singlefeat_allopt", single_baselist +
            [
                "--details", model["details"], "--all_opt", "--prob",
                "--output_gap", lb, ub,
                "--compute_data_distance", "--data_file", model["data"],
            ],
        ),
        TestCase(
            f"{mode}.{model_name}.multifeat_allopt", multi_baselist +
            [
                "--details", model["details"], "--all_opt", "--prob",
                "--output_gap", lb, ub,
                "--compute_data_distance", "--data_file", model["data"],
            ],
        ),
        # for clause-data-aware
        TestCase(
            f"{mode}.{model_name}.singlefeat_allopt", single_baselist +
            [
                "--details", model["details"], "--all_opt",
                "--output_gap", lb, ub,
                "--in_distro_clauses", model["clause"],
                "--compute_data_distance", "--data_file", model["data"],
            ],
        ),
        TestCase(
            f"{mode}.{model_name}.multifeat_allopt", multi_baselist +
            [
                "--details", model["details"], "--all_opt",
                "--output_gap", lb, ub,
                "--in_distro_clauses", model["clause"],
                "--compute_data_distance", "--data_file", model["data"],
            ],
        ),
        # for prob-clause-data-aware
        TestCase(
            f"{mode}.{model_name}.singlefeat_allopt", single_baselist +
            [
                "--details", model["details"], "--all_opt",
                "--output_gap", lb, ub,
                "--prob",
                "--in_distro_clauses", model["clause"],
                "--compute_data_distance", "--data_file", model["data"],
            ],
        ),
        TestCase(
            f"{mode}.{model_name}.multifeat_allopt", multi_baselist +
            [
                "--details", model["details"], "--all_opt",
                "--output_gap", lb, ub,
                "--prob",
                "--in_distro_clauses", model["clause"],
                "--compute_data_distance", "--data_file", model["data"],
            ],
        ),
        
    ]
    return core


def pb_cases(model_name: str,timeout:int) -> list[TestCase]:
    core = one_cases(model_name,"pb",timeout)
    addition = ["--solver", "pb"]
    for test in core:
        test.args = test.args + addition
    return core

def milp_cases(model_name:str,timeout) -> list[TestCase]:
    core = one_cases(model_name,"milp",timeout)
    addition = ["--solver", "milp"]
    for test in core:
        test.args = test.args + addition
    return core
    
def _run_case(case: TestCase, dry_run: bool, verbose: bool,timeout:int) -> int:
    cmd = [sys.executable, str(SENSITIVE), *case.args]
    print(f"[RUN] {case.name}")
    print("      " + " ".join(cmd))
    if dry_run:
        return 0

    env = os.environ.copy()
    old_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not old_path else f"{ROOT}:{old_path}"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout+100,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {case.name}")
        return 1

    if proc.returncode == 0:
        print(f"[PASS] {case.name}")
        return 0

    print(f"[FAIL] {case.name} (code={proc.returncode})")
    if verbose:
        if proc.stdout.strip():
            print("------ stdout ------")
            print(proc.stdout[-3000:])
        if proc.stderr.strip():
            print("------ stderr ------")
            print(proc.stderr[-3000:])
    return 1


def _collect_cases(suite: str, model_name: str,timeout :int) -> list[TestCase]:
    cases: list[TestCase] = []
    if suite in ("pb", "all"):
        cases.extend(pb_cases(model_name,timeout))
    if suite in ("milp", "all"):
        cases.extend(milp_cases(model_name,timeout))
    #! TODO MONITOR
    # for case in cases:
    #     print(case)
    # input()
    return cases


def _validate_model(model_name: str) -> None:
    if model_name not in MODELDETAILS:
        available = ", ".join(sorted(MODELDETAILS))
        raise SystemExit(f"Unknown model '{model_name}'. Available: {available}")

    required = {"model", "details", "data", "test", "feature", "clause"}
    missing = required - set(MODELDETAILS[model_name].keys())
    if missing:
        raise SystemExit(
            f"Model '{model_name}' is missing required keys in MODELDETAILS: {sorted(missing)}"
        )

def _list_models():
    print("Available models:")
    for name in sorted(MODELDETAILS):
        print(f"  - {name}")
    return

def main() -> None:
    parser = argparse.ArgumentParser(description="Run TriST command-line unit tests.")
    parser.add_argument("--model", default="brcR", help="Model key from unit-tests/modeldetails.py")
    parser.add_argument("--suite", choices=["milp", "pb", "all"], default="all")
    parser.add_argument("--timeout",type=int,default=100,help="timeout")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list_models: _list_models()
    _validate_model(args.model)
    cases = _collect_cases(args.suite, args.model,args.timeout)
    if not cases:
        print("No test cases selected.")
        return

    failed = 0
    for case in cases:
        rc = _run_case(case, args.dry_run, args.verbose,args.timeout)
        failed += rc
        if rc:
            break

    passed = len(cases) - failed
    print(f"\nSummary: {passed}/{len(cases)} passed, {failed} failed")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
