#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import datetime as _datetime
import os
import platform
from pathlib import Path
from typing import Dict


def _hardware_meta() -> Dict[str, object]:
  uname = platform.uname()
  machine = uname.machine or "unknown"
  processor = uname.processor or ""
  mhz = 0.0
  try:
    with open("/proc/cpuinfo", "r", encoding="utf-8") as fp:
      for line in fp:
        if line.lower().startswith("cpu mhz"):
          mhz = float(line.split(":", 1)[1].strip())
          break
  except OSError:
    pass
  try:
    page = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    ram_gb = page * pages / (1024 ** 3)
  except (ValueError, OSError, AttributeError):
    ram_gb = 0.0
  flags = []
  if machine.lower().endswith("64"):
    flags.append(machine.lower())
  return {
      "hardware": machine,
      "processor": processor,
      "cores": os.cpu_count() or 0,
      "cpu_mhz": mhz,
      "ram_gb": ram_gb,
      "flags": flags,
      "timestamp": _datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
  }


def load_baseline(path: Path) -> Dict[str, float]:
  if not path or not path.exists():
    return {}
  data = json.loads(path.read_text())
  result: Dict[str, float] = {}
  for key, value in data.items():
    if key == "_meta":
      continue
    if isinstance(value, dict):
      result[key] = float(value.get("tokens_per_s", 0.0))
    else:
      result[key] = float(value)
  return result


def parse_csv(path: Path) -> Dict[str, Dict[str, float]]:
  results: Dict[str, Dict[str, float]] = {}
  with path.open("r", encoding="utf-8") as fp:
    reader = csv.DictReader(fp)
    for row in reader:
      if row.get("bench") != "rope_fht_kv":
        continue
      key = f"seq{row['seq_len']}_dim{row['head_dim']}"
      tokens = float(row.get("tokens_per_s", 0.0))
      speedup = float(row.get("speedup", 0.0))
      diff = float(row.get("l2_diff", 0.0))
      perplexity_delta = float(row.get("perplexity_delta", 0.0))
      results[key] = {
          "tokens_per_s": tokens,
          "speedup": speedup,
          "l2_diff": diff,
          "perplexity_delta": perplexity_delta,
      }
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare attention prefill perf")
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--baseline-out", "--baseline_out", type=Path)
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--regression-threshold", type=float, default=0.10)
  parser.add_argument("--update-baseline", action="store_true")
  parser.add_argument("--write-baseline", action="store_true")
  args = parser.parse_args()

  results = parse_csv(args.csv)
  baseline_path = args.baseline or args.baseline_out
  baseline = load_baseline(baseline_path)
  args.summary.parent.mkdir(parents=True, exist_ok=True)
  args.summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

  write_baseline = args.update_baseline or args.write_baseline

  for key, metrics in results.items():
    if not write_baseline:
      if metrics["speedup"] < 1.25:
        raise SystemExit(f"Prefill speedup below 1.25× for {key} (got {metrics['speedup']:.2f}×)")
      if metrics["perplexity_delta"] > 0.3:
        raise SystemExit(f"Perplexity delta exceeded 0.3pp for {key} (got {metrics['perplexity_delta']:.3f}pp)")
    base = baseline.get(key, 0.0)
    if not write_baseline and base > 0.0 and metrics["tokens_per_s"] < (1.0 - args.regression_threshold) * base:
      raise SystemExit(f"Attn perf regression for {key}: {metrics['tokens_per_s']:.2f} < {base:.2f}")

  if write_baseline:
    target = args.baseline_out or args.baseline
    if not target:
      raise SystemExit("--baseline-out (or --baseline) required when updating baseline")
    payload = {key: {"tokens_per_s": value["tokens_per_s"]} for key, value in results.items()}
    payload["_meta"] = _hardware_meta()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
