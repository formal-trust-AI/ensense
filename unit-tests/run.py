#!/usr/bin/env python3
"""Run unit-test commands from tmp.txt and compare against golden outputs."""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
# DEFAULT_COMMANDS_FILE = SCRIPT_DIR / "testcmd.txt"
DEFAULT_TESTS_DIR = SCRIPT_DIR / "tests"
DEFAULT_OUTPUT_DIR = Path("/tmp") / "testing"

import json

TAGS = [
        "# Model name:",
        "# Trees per class:",
        "# Number of classes:",
        "# Number of features:",
        "# Base score:",
        "# Max depth:",
        "# Feature names:",
        "# Running the solver with precision level:",
    ]
# TOLERANCE_TAGS : <tag>:(<type>,tolerance)

TOLERANCE_TAGS = {
        'Time:': (float,10),
        '#Time:': (float,10),
        'Sensitive:': (list,0) ,
        'Sensitive sample 1:':(list,1e-1),
        'Sensitive sample 2:':(list,1e-1),
        'Output values:':(list,1e-2),
        'Output Values:':(list,1e-2),
        'Objective Value:':(float,1e1),
        'Distance from data distype L0:' :(float,1e-4) ,
        'Distance from data distype L1:':(float,1e-4),
        'Distance from data distype Linf:':(float,1e-4),
}

WARNING_LINE_RE = re.compile(
    r"""
    (?:^/.*\.py:\d+:.*Warning:)   # file.py:line: WarningType:
    |(?:^\s*warnings\.warn\()     # continuation: warnings.warn(...)
    |(?:\b(?:Future|User|Deprecation|Runtime)Warning\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

@dataclass
class CaseResult:
    test_id: str
    command_ok: bool
    output_ok: bool
    tagged : bool
    message: str = ""
    

    @property
    def ok(self) -> bool:
        return self.command_ok and self.output_ok
    @property
    def tag_ok(self) -> bool:
        return self.command_ok and self.tagged

def list_tests():
    test_names  = [ s for s in os.listdir(DEFAULT_TESTS_DIR) if os.path.isdir(DEFAULT_TESTS_DIR / s) ]
    test_names.sort()
    return test_names

def _read_commands(path: Path) -> list[str]:
    commands: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(line)
    return commands


def _normalize_output(text: str) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]

    return lines

def readfile(filename):
    with open(filename,'r',encoding= "utf-8") as f:
        content = f.read()
    return content

def extract_numbers(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _run_command(cmd: str, timeout: int) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    old_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) if not old_path else f"{ROOT_DIR}:{old_path}"

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=True,
        executable="/bin/bash",
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined

def _write_dff(diff,diff_file,limit):
    with diff_file.open("a", encoding="utf-8") as f:
        f.write("------ diff ------\n")
        f.write("\n".join(diff) + "\n")
        if len(diff) > limit:
                f.write(f"... (diff truncated, total lines: {len(diff)})")

def _print_diff(expected: list[str],
                actual: list[str],
                expected_file: Path,
                actual_file: Path,
                limit: int,
                diff_file:Path,
                message:str) -> None:
    diff = list(
        difflib.unified_diff(
            expected,
            actual,
            fromfile=str(expected_file),
            tofile=str(actual_file),
            lineterm="",
        )
    )
    if not diff:
        with diff_file.open("a", encoding="utf-8") as f:
            f.write("------no diff ------\n")
    if diff:
        _write_dff(diff,diff_file,limit)
     


def tag_comparision(
    expected: list[str],
    actual: list[str],
    expected_file: Path,
    actual_file: Path,
    limit: int,
    diff_file: Path,
) -> bool:
    
    all_tags = TAGS + list(TOLERANCE_TAGS.keys())
    expected_tagged = {tag: next((line for line in expected if line.startswith(tag)), "") for tag in all_tags}
    actual_tagged = {tag: next((line for line in actual if line.startswith(tag)), "") for tag in all_tags}
    diff_summary = {}
    passed = 0
    failed = 0
    actual_tmp =""
    expected_tmp = ""
    for tag in all_tags:
        
        # print(tag,actual_tagged[tag],expected_tagged[tag])
        if tag in TAGS:
            actual_tmp += f"{actual_tagged[tag]}"
            expected_tmp += f"{expected_tagged[tag]}"
        else:
            type_, tolerance = TOLERANCE_TAGS[tag]
            actual_output = extract_numbers(actual_tagged[tag])
            expected_output = extract_numbers(expected_tagged[tag])
            if not expected_output and not actual_output:
                    diff_summary[tag] = 'PASSED'
                    passed +=1
                    continue
            if type_ is float:
                if expected_output and actual_output:
                    if abs(expected_output[0] - actual_output[0]) < tolerance:
                        diff_summary[tag] = 'PASSED'
                        passed +=1
                    else:
                        diff_summary[tag] = f"FAILED: actual:{actual_output} , expected: {expected_output}" 
                        failed +=1
                else:
                    diff_summary[tag] = f"FAILED: actual:{actual_output} , expected: {expected_output}" 
                    failed +=1
            elif type_ is list:
                if len(actual_output) != len(expected_output):
                    diff_summary[tag] = f"FAILED: length did not matched  actual:{actual_output} , expected: {expected_output}"
                    failed +=1
                elif actual_output and expected_output:
                    tmp = True
                    for a,o in zip(expected_output,actual_output):
                        if abs(a-o) > tolerance:
                            diff_summary[tag] = f"FAILED: on tolerance actual:{actual_output} , expected: {expected_output}"
                            tmp = False
                            break
                    if tmp:
                        diff_summary[tag] = 'Passed'
                else:
                    diff_summary[tag] = f"FAILED: actual:{actual_output} , expected: {expected_output}"
                    failed +=1

    diff = list(
        difflib.unified_diff(
            expected_tmp,
            actual_tmp,
            fromfile=str(expected_file),
            tofile=str(actual_file),
            lineterm="",
        )
    )
    if failed == 0 and not diff:
        with diff_file.open("a", encoding="utf-8") as f:
            f.write("------ no tagged diff ------\n")
        return True

    with diff_file.open("a", encoding="utf-8") as f:
        f.write("------ tagged diff ------\n")
        f.write("\n".join(diff))
        f.write(json.dumps(diff_summary, indent=2, ensure_ascii=False) + "\n")

    return False


def _compare_case(
    idx: int,
    command: str,
    tests_dir: Path,
    timeout: int,
    dry_run: bool,
    show_diff_lines: int,
) -> CaseResult:
    test_id = f"test{idx:03d}"
    test_dir = tests_dir / test_id
    option_file = test_dir / "option.txt"
    expected_output_file = test_dir / "output.txt"
    new_output_file = DEFAULT_OUTPUT_DIR / "output.new.txt"
    diff_file = DEFAULT_OUTPUT_DIR / "diff.txt"
    diff_file.write_text("", encoding="utf-8")
    
    if not test_dir.exists():
        return CaseResult(test_id, False, False,False, f"Missing directory: {test_dir}")

    # print(f"[RUN] {test_id}")
    # print(f"      {command}")

    if dry_run:
        return CaseResult(test_id, True, True,True, "dry-run")

    try:
        rc, output = _run_command(command, timeout)
    except subprocess.TimeoutExpired:
        new_output_file.write_text(f"TIMEOUT after {timeout}s\n", encoding="utf-8")
        return CaseResult(test_id, True, False, False, f"Timeout after {timeout}s")

    new_output_file.write_text(output, encoding="utf-8")

    if not expected_output_file.exists():
        return CaseResult(test_id, rc == 0, False, False, f"Missing expected output file: {expected_output_file}")

    expected = expected_output_file.read_text(encoding="utf-8")
    expected_norm = _normalize_output(expected)
    output_norm = _normalize_output(output)
    same = expected_norm == output_norm

    if same:
        return CaseResult(test_id, rc == 0, True,True,'PASSED')
    tag_same = tag_comparision(
        expected_norm,
        output_norm,
        expected_output_file,
        new_output_file,
        show_diff_lines,
        diff_file,
    )
    message = ''
    if tag_same:
        message += 'PASSED'
    else:
        message += 'FAILED'

    _print_diff(expected_norm, output_norm, expected_output_file, new_output_file, show_diff_lines, diff_file,message)
    status_msg = message
    return CaseResult(test_id, rc == 0, False,tag_same, status_msg)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run commands from tmp.txt and compare with golden outputs.")
    # parser.add_argument("--commands-file", type=Path, default=DEFAULT_COMMANDS_FILE)
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS_DIR)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-diff-lines", type=int, default=120)
    parser.add_argument("--show-diff",action="store_true")
    parser.add_argument("--test", type=int, default=-1)
    args = parser.parse_args()

    # commands_file = args.commands_file
    tests_dir = args.tests_dir

    # if not commands_file.exists():
    #     raise SystemExit(f"Commands file not found: {commands_file}")
    if not tests_dir.exists():
        raise SystemExit(f"Tests directory not found: {tests_dir}")

    tests = list_tests()
    if args.test != -1:
        if args.test >= len(tests):
            print( f"Number of available test is less than {args.test}" )
        tests = [tests[args.test]]
    
    commands = []
    for test in tests:
        option_path = DEFAULT_TESTS_DIR / test / "option.txt"
        f = open(option_path, "r")
        commands.append(f.read().strip())
        f.close()   

    os.makedirs( DEFAULT_OUTPUT_DIR, exist_ok=True )
    
    # commands = _read_commands(commands_file)
    
    if args.max_tests is not None: commands = commands[: args.max_tests]

    if not commands:
        print("No commands to run.")
        return

    results: list[CaseResult] = []
    for idx, command in enumerate(commands):
        if args.test != -1:
            idx = args.test
        result = _compare_case(
            idx=idx,
            command=command,
            tests_dir=tests_dir,
            timeout=args.timeout,
            dry_run=args.dry_run,
            show_diff_lines=args.show_diff_lines,
        )
        results.append(result)

        if result.ok:
            print(f"[RUN] {result.test_id} [PASSED] {result.message}")
        else:
            print(f"[RUN] {result.test_id}: {result.message}")
            diff_file = DEFAULT_OUTPUT_DIR / "diff.txt"
            if args.show_diff:
                if os.path.exists(diff_file):
                    print(readfile(diff_file))
                else: print('diff file not found')
            if args.stop_on_fail:
                break

    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    tag_passed = sum(1 for r in results if r.tag_ok)
    tag_failed = len(results) - tag_passed
    print(f"\nOutput Summary: {passed}/{len(results)} passed, {failed} failed")
    print(f"\nTagged Summary: {tag_passed}/{len(results)} passed, {tag_failed} failed")

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
