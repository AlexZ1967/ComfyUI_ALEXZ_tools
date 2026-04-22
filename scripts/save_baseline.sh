#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
RESULTS_TSV="$TMP_DIR/results.tsv"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage: scripts/save_baseline.sh [--output PATH]

Options:
  -o, --output PATH   Path to the generated baseline JSON file
  -h, --help          Show this help message

Environment:
  BASELINE_OUTPUT_PATH  Default output path when --output is not provided
EOF
}

OUTPUT_PATH="${BASELINE_OUTPUT_PATH:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      if [[ $# -lt 2 ]]; then
        echo "[baseline] missing value for $1" >&2
        usage >&2
        exit 2
      fi
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$OUTPUT_PATH" ]]; then
        OUTPUT_PATH="$1"
        shift
      else
        echo "[baseline] unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/baseline.json}"

extract_summary() {
  local log_path="$1"
  python3 - "$log_path" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
text = text.replace("\r", "\n")
ansi = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
lines = [ansi.sub("", line).strip() for line in text.splitlines()]
lines = [line for line in lines if line and "%|" not in line]

patterns = (
    re.compile(r".*\b(?:passed|failed|error|errors|warning|warnings|skipped|xfailed|xpassed)\b.*\bin [0-9.]+s\b", re.IGNORECASE),
    re.compile(r"^=+ .* =+$"),
    re.compile(r".*\b(?:passed|failed|error|errors|warning|warnings|skipped|xfailed|xpassed)\b.*", re.IGNORECASE),
)

for pattern in patterns:
    for line in reversed(lines):
        if pattern.search(line):
            print(line)
            raise SystemExit(0)

if lines:
    print(lines[-1])
PY
}

CHECK_NAMES=(
  "docs_check"
  "phase0_and_contracts"
  "smoke_nodes"
  "js_check"
  "full_pytest"
)

CHECK_COMMANDS=(
  "conda run -n p313 python utils/docs_check.py"
  "conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_api_contracts_golden.py"
  "conda run -n p313 pytest -q tests/test_smoke_nodes.py"
  "conda run -n p313 node --check web/widget_visibility_profiles.js"
  "conda run -n p313 pytest -q"
)

cd "$ROOT_DIR"
mkdir -p "$(dirname "$OUTPUT_PATH")"
: > "$RESULTS_TSV"

for idx in "${!CHECK_NAMES[@]}"; do
  name="${CHECK_NAMES[$idx]}"
  cmd="${CHECK_COMMANDS[$idx]}"
  log_path="$TMP_DIR/${name}.log"

  echo "[baseline] running: $name"
  if output="$(bash -lc "$cmd" 2>&1)"; then
    status="passed"
  else
    printf '%s\n' "$output" >&2
    echo "[baseline] failed: $name" >&2
    exit 1
  fi

  printf '%s' "$output" > "$log_path"
  sha256="$(sha256sum "$log_path" | awk '{print $1}')"
  summary="$(extract_summary "$log_path" | tr '\t' ' ' | tr -d '\r')"
  summary="${summary:-OK}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$sha256" "$summary" "$cmd" >> "$RESULTS_TSV"
done

python3 - "$OUTPUT_PATH" "$RESULTS_TSV" "$ROOT_DIR" <<'PY'
import csv
import datetime as dt
import json
import pathlib
import subprocess
import sys

output_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
root_dir = pathlib.Path(sys.argv[3])

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=root_dir,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_commit = ""

checks = []
with results_path.open("r", encoding="utf-8", newline="") as fh:
    reader = csv.reader(fh, delimiter="\t")
    for row in reader:
        if not row:
            continue
        checks.append(
            {
                "name": row[0],
                "status": row[1],
                "output_sha256": row[2],
                "summary": row[3],
                "command": row[4],
            }
        )

payload = {
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "git_commit": git_commit,
    "checks": checks,
}

output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[baseline] saved: {output_path}")
PY
