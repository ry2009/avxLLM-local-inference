#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-tests/data/test_tokenizer.json}
INPUT=${2:-tests/bench/prompts.txt}
PREFIX_K=${PREFIX_K:-128}
PREFIX_CACHE=${PREFIX_CACHE:-2048}
BUILD_DIR=${BUILD_DIR:-build}
REPORT_DIR=reports

: "${TOKENIZER_MIN_SPEEDUP:=5.0}"

export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"

BIN="${BUILD_DIR}/tools/bench/infeng_bench_tokenizer_gb"
PY_BASELINE=tools/bench/python_tokenizer_baseline.py
COMPARE=tools/bench/compare_tokenizers.py

if [[ ! -x "$BIN" ]]; then
  echo "Benchmark binary $BIN not found or not executable" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"
rm -f "$REPORT_DIR"/tokenizer.csv

THREADS=(1 2 4 8)
for t in "${THREADS[@]}"; do
  "$BIN" --model="$MODEL" --input_file="$INPUT" --threads=$t --prefix_k=${PREFIX_K} --prefix_cache_entries=${PREFIX_CACHE} --benchmark_min_time=200ms
done

python3 "$PY_BASELINE" --model "$MODEL" --input_file "$INPUT" --threads 1 --output "$REPORT_DIR"/tokenizer_py.csv

python3 "$COMPARE" --native "$REPORT_DIR"/tokenizer.csv --python "$REPORT_DIR"/tokenizer_py.csv --baseline tools/perf/tokenizer_baseline.json --min-speedup "${TOKENIZER_MIN_SPEEDUP}"

python3 - <<"PY"
import csv, json
from pathlib import Path

native_rows = []
with Path("reports/tokenizer.csv").open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        native_rows.append({
            "bench": row["bench"],
            "threads": int(row["threads"]),
            "tokens_per_s": float(row["tokens_per_s"])
        })

python_tokens = 0.0
with Path("reports/tokenizer_py.csv").open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        python_tokens = float(row["tokens_per_s"])

summary = {
    "native": native_rows,
    "python_tokens_per_s": python_tokens
}
Path("reports/tokenizer_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
