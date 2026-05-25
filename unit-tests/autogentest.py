#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import argparse

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT_DIR=SCRIPT_DIR.parent
COMMANDS_FILE=SCRIPT_DIR / "testcmd.txt"
OUT_DIR=SCRIPT_DIR / "tests"


def main(args):
    commands_file = COMMANDS_FILE
    out_dir = OUT_DIR
    root_dir = ROOT_DIR

    if not commands_file.is_file():
        print(f"Commands file not found: {commands_file}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    passed = 0
    failed = 0

    with commands_file.open("r") as f:
        for raw_line in f:
            cmd = raw_line.rstrip("\n").rstrip("\r")
            idx += 1

            if not cmd.strip() or cmd.lstrip().startswith("#"):
                continue

            test_id = f"test{idx - 1:03d}"
            test_dir = out_dir / test_id
            test_dir.mkdir(parents=True, exist_ok=True)

            option_file = test_dir / "option.txt"
            output_file = test_dir / "output.txt"

            option_file.write_text(cmd + "\n")

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "ignore::FutureWarning"

            try:
                result = subprocess.run(
                    ["bash", "-lc", cmd],
                    cwd=root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                output = result.stdout
                rc = result.returncode
            except Exception as e:
                output = str(e) + "\n"
                rc = 99

            output_file.write_text(output)
            if args.show_output:
                print(f"      {cmd}")
                print(output, end="")

            if rc == 0:
                print(f"[RUN] {test_id} [PASS] ")
                passed += 1
            else:
                print(f"[RUN] {test_id} [FAIL] (exit={rc})")
                failed += 1
                if args.stop_on_fail:
                    break

    total = passed + failed
    print(f"PASSED: {passed}\nFAILED: {failed}")
    print(f"Artifacts: {out_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("auto deneration of test cases from testcmd.txt")
    parser.add_argument('--start',action='store_true', help='to run the script')
    parser.add_argument('--show-output',action='store_true',help='show output of each command')
    parser.add_argument('--stop-on-fail',action='store_true',help='stop on first failure')
    args = parser.parse_args()
    if args.start:
        main(args)
    else:
        print(f"please use --start option to run the script")