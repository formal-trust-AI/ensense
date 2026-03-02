#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMANDS_FILE="${SCRIPT_DIR}/testcmd.txt"
OUT_DIR="${SCRIPT_DIR}/tests"
STOP_ON_FAIL=0

# echo "THIS SCRIPT IS HARDBLOCK! UNCOMMENT THE FOLLOWING EXIT TO GEN TEST!"
# exit 1

if [[ ! -f "$COMMANDS_FILE" ]]; then
  echo "Commands file not found: $COMMANDS_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

idx=0
passed=0
failed=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  cmd="${raw_line%%$'\r'}"
  idx=$((idx + 1))
  if [[ -z "$cmd" || "$cmd" =~ ^[[:space:]]*# ]]; then
    continue
  fi

  # if [[ $((idx - 1)) -le 23 ]]; then
  #   continue
  # fi

  test_id="test$(printf '%03d' "$((idx-1))")"
  test_dir="$OUT_DIR/$test_id"
  mkdir -p "$test_dir"

  option_file="$test_dir/option.txt"
  output_file="$test_dir/output.txt"

  printf '%s\n' "$cmd" > "$option_file"

  echo "[RUN] $test_id"
  echo "      $cmd"

  (
    cd "$ROOT_DIR" || exit 99
    PYTHONWARNINGS="ignore::FutureWarning" bash -lc "$cmd"
  ) > "$output_file" 2>&1  | tee "$output_file"
  rc=$?

  cat "$output_file" | tee "$output_file"

  if [[ $rc -eq 0 ]]; then
    echo "[PASS] $test_id"
    passed=$((passed + 1))
  else
    echo "[FAIL] $test_id (exit=$rc)"
    failed=$((failed + 1))
    if [[ $STOP_ON_FAIL -eq 1 ]]; then
      break
    fi
  fi

done < "$COMMANDS_FILE"

total=$((passed + failed))
echo
echo "Artifacts: $OUT_DIR"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
