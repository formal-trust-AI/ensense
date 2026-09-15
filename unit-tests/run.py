#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_TESTS_DIR = SCRIPT_DIR / "tests"
SENSITIVE_TAG = "# Sensitive "
INSENSITIVE_TAG = "# Insensitive "

def list_tests(tests_dir):
    return sorted(s for s in os.listdir(tests_dir) if (tests_dir / s).is_dir())

def extract_sensitive(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(SENSITIVE_TAG):
            return "sensitive"
        if line.startswith(INSENSITIVE_TAG):
            return "insensitive"
    return None  # error or crash — tag not found

def run_command(cmd: str, timeout: int) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True,
                          text=True, timeout=timeout, shell=True,
                          executable="/bin/bash", env=env)
    return (proc.stdout or "") + (proc.stderr or "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS_DIR)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--test", type=int, default=-1)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-diff-lines", type=int, default=120)
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument("--show-command", action="store_true")
    parser.add_argument("--output", action="store_true")

    args = parser.parse_args()

    if (args.output or args.show_command) and args.test == -1:
        print(f"--{'output' if args.output else 'show-command'} requires --test to specify which test")
        return

    tests = list_tests(args.tests_dir)
    if args.test >= len(tests):
        print(f"Error: out of index total test found {len(tests)}")
        return
    if args.test != -1:
        tests = [tests[args.test]]
    if args.max_tests is not None:
        tests = tests[:args.max_tests]

    passed = failed = 0
    for i, test in enumerate(tests):
        idx = args.test if args.test != -1 else i
        test_dir = args.tests_dir / test
        option = (test_dir / "option.txt").read_text().strip()
        expected_text = (test_dir / "output.txt").read_text()

        if args.output:
            print(expected_text)
            return

        if args.show_command:
            print(f"[test{idx:03d}] CMD: {option}")
            return

        if args.dry_run:
            print(f"[test{idx:03d}] DRY-RUN  {option}")
            continue

        try:
            actual_text = run_command(option, args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[test{idx:03d}] TIMEOUT")
            failed += 1
            if args.stop_on_fail:
                break
            continue

        expected_label = extract_sensitive(expected_text)
        actual_label   = extract_sensitive(actual_text)

        if actual_label is None:
            print(f"[test{idx:03d}] ERROR ")#  (no sensitive/insensitive tag found)")
            if args.show_diff:
                print(actual_text[:args.show_diff_lines])
            failed += 1
        elif expected_label == actual_label:
            print(f"[test{idx:03d}] PASSED") # ({actual_label})")
            passed += 1
        else:
            print(f"[test{idx:03d}] FAILED  expected={expected_label}, actual={actual_label}")
            if args.show_diff:
                print(actual_text[:args.show_diff_lines])
            failed += 1

        if args.stop_on_fail and failed > 0:
            break

    print(f"\nSummary: {passed}/{passed+failed} passed, {failed} failed")

if __name__ == "__main__":
    main()
