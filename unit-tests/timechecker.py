#!/usr/bin/env python3
"""Run all tests and write combined output to output.txt and timing to time.txt."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
TESTS_DIR  = SCRIPT_DIR / "tests"
OUTPUT_FILE = SCRIPT_DIR / "diffoutputalloptimizations.txt"
TIME_FILE   = SCRIPT_DIR / "timemid.txt"

TIMEOUT = 240  # seconds per test


def list_tests() -> list[str]:
    names = [s for s in os.listdir(TESTS_DIR) if os.path.isdir(TESTS_DIR / s)]
    names.sort()
    return names


def run_command(cmd: str) -> tuple[float, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) if not old else f"{ROOT_DIR}:{old}"

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            shell=True,
            executable="/bin/bash",
            env=env,
        )
        elapsed = time.perf_counter() - t0
        combined = (proc.stdout or "") + (proc.stderr or "")
        return elapsed, combined
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return elapsed, f"TIMEOUT after {TIMEOUT}s\n"


def main() -> None:
    tests = list_tests()

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f, \
         TIME_FILE.open("w", encoding="utf-8") as time_f:

        for test in tests:
            option_path = TESTS_DIR / test / "option.txt"
            if not option_path.exists():
                print(f"[SKIP] {test}: no option.txt")
                continue

            cmd = option_path.read_text(encoding="utf-8").strip()
            print(f"[RUN]  {test}: {cmd}")

            elapsed, output = run_command(cmd)

            # --- output.txt ---
            out_f.write(f"{'='*60}\n")
            out_f.write(f"TEST: {test}\n")
            out_f.write(f"CMD:  {cmd}\n")
            out_f.write(f"{'='*60}\n")
            out_f.write(output)
            if not output.endswith("\n"):
                out_f.write("\n")

            # --- time.txt ---
            time_f.write(f"{test}: {elapsed:.3f}s  |  {cmd}\n")

            print(f"       done in {elapsed:.3f}s")

    print(f"\nOutputs written to: {OUTPUT_FILE}")
    print(f"Timings written to: {TIME_FILE}")


if __name__ == "__main__":
    main()
