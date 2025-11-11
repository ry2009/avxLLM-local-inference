#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
OUT_CSV="reports/attn_decode.csv"
SUMMARY_PATH=""
BASELINE="tools/perf/attn_decode_baseline.json"
TILE_TOKENS=""
FORCE_PATH=""

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
    --tile-tokens)
      TILE_TOKENS="$2"
      shift 2
      ;;
    --tile-tokens=*)
      TILE_TOKENS="${1#*=}"
      shift 1
      ;;
    --force-path)
      FORCE_PATH="$2"
      shift 2
      ;;
    --force-path=*)
      FORCE_PATH="${1#*=}"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

DEFAULT_SUMMARY="$(dirname "$OUT_CSV")/attn_decode_summary.json"
SUMMARY_PATH=${SUMMARY_PATH:-$DEFAULT_SUMMARY}

DEFAULT_REPORT="reports/attn_decode.csv"
rm -f "$DEFAULT_REPORT"

CMD=("${BUILD_DIR}/tools/perf/cpp/infeng_perf_attn_decode")
if [[ -n "$TILE_TOKENS" ]]; then
  CMD+=("--tile_tokens" "$TILE_TOKENS")
fi
if [[ -n "$FORCE_PATH" ]]; then
  CMD+=("--force_path" "$FORCE_PATH")
fi
"${CMD[@]}"

CSV_PATH="$DEFAULT_REPORT"
if [[ "$OUT_CSV" != "$DEFAULT_REPORT" ]]; then
  mkdir -p "$(dirname "$OUT_CSV")"
  cp "$DEFAULT_REPORT" "$OUT_CSV"
  CSV_PATH="$OUT_CSV"
fi

python3 tools/perf/compare_attn_decode.py \
  --csv "$CSV_PATH" \
  --baseline "$BASELINE" \
  --summary "$SUMMARY_PATH"
