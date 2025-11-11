#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
OUT_CSV="reports/sched_zipf.csv"
SUMMARY_PATH=""
BASELINE="tools/perf/sched_zipf_baseline.json"
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
    --alpha|--adapters|--mean_adapters|--rank|--load_rps|--duration_s|--interactive_ratio|\
    --decode_threads|--adapter_threads|--hot_reload_rate|--hot_reload_latency_ms|\
    --base_decode_ms|--adapter_decode_ms|--prompt_tokens_mean|--max_batch_size|--seed)
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --pin_cores)
      ARGS+=("$1" "true")
      shift 1
      ;;
    --pin_cores=*)
      ARGS+=("--pin_cores" "${1#*=}")
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

DEFAULT_SUMMARY="$(dirname "$OUT_CSV")/sched_zipf_summary.json"
SUMMARY_PATH=${SUMMARY_PATH:-$DEFAULT_SUMMARY}

DEFAULT_REPORT="reports/sched_zipf.csv"
rm -f "$DEFAULT_REPORT"

CMD=("${BUILD_DIR}/tools/perf/cpp/infeng_perf_sched_zipf" "--out" "$DEFAULT_REPORT")
for ((idx=0; idx<${#ARGS[@]}; idx+=2)); do
  CMD+=("${ARGS[idx]}" "${ARGS[idx+1]}")
done
"${CMD[@]}"

CSV_PATH="$DEFAULT_REPORT"
if [[ "$OUT_CSV" != "$DEFAULT_REPORT" ]]; then
  mkdir -p "$(dirname "$OUT_CSV")"
  cp "$DEFAULT_REPORT" "$OUT_CSV"
  CSV_PATH="$OUT_CSV"
fi

PY_CMD=(
  python3 tools/perf/compare_sched_zipf.py
  --csv "$CSV_PATH"
  --baseline "$BASELINE"
  --summary "$SUMMARY_PATH"
)
if [[ ${#COMPARE_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${COMPARE_ARGS[@]}")
fi
"${PY_CMD[@]}"
