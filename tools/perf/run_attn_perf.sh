#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
OUT_CSV="reports/attn_prefill.csv"
SUMMARY_PATH=""
BASELINE="tools/perf/attn_prefill_baseline.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --out)
      OUT_CSV="$2"
      shift 2
      ;;
    --summary)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    --baseline)
      BASELINE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

DEFAULT_SUMMARY="$(dirname "$OUT_CSV")/attn_prefill_summary.json"
SUMMARY_PATH=${SUMMARY_PATH:-$DEFAULT_SUMMARY}

"${BUILD_DIR}/tools/bench/infeng_bench_attn_prefill"

DEFAULT_REPORT="reports/attn_prefill.csv"
CSV_PATH="$DEFAULT_REPORT"
if [[ "$OUT_CSV" != "$DEFAULT_REPORT" ]]; then
  mkdir -p "$(dirname "$OUT_CSV")"
  cp "$DEFAULT_REPORT" "$OUT_CSV"
  CSV_PATH="$OUT_CSV"
fi

python3 tools/perf/compare_attn_prefill.py \
  --csv "$CSV_PATH" \
  --baseline "$BASELINE" \
  --summary "$SUMMARY_PATH"
