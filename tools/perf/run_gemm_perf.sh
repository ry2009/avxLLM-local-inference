#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
OUT_CSV="reports/gemm_lora.csv"
SUMMARY_PATH=""
BASELINE="tools/perf/gemm_baseline.json"
ARGS=()
COMPARE_ARGS=()

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
    --baseline-out|--baseline_out)
      COMPARE_ARGS+=("$1" "$2")
      shift 2
      ;;
    --write-baseline|--update-baseline)
      COMPARE_ARGS+=("$1")
      shift 1
      ;;
    --regression-threshold)
      COMPARE_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

DEFAULT_SUMMARY="$(dirname "$OUT_CSV")/gemm_summary.json"
SUMMARY_PATH=${SUMMARY_PATH:-$DEFAULT_SUMMARY}

"${BUILD_DIR}/tools/bench/infeng_bench_gemm_lora"

DEFAULT_REPORT="reports/gemm_lora.csv"
CSV_PATH="$DEFAULT_REPORT"
if [[ "$OUT_CSV" != "$DEFAULT_REPORT" ]]; then
  mkdir -p "$(dirname "$OUT_CSV")"
  cp "$DEFAULT_REPORT" "$OUT_CSV"
  CSV_PATH="$OUT_CSV"
fi

PY_CMD=(
  python3 tools/perf/compare_gemm.py
  --csv "$CSV_PATH"
  --baseline "$BASELINE"
  --summary "$SUMMARY_PATH"
)

if [[ ${#COMPARE_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${COMPARE_ARGS[@]}")
fi

"${PY_CMD[@]}"
