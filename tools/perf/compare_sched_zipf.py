#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
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


def load_baseline(path: Path) -> Dict[str, Dict[str, float]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text())
    result: Dict[str, Dict[str, float]] = {}
    for key, value in data.items():
        if key == "_meta":
            continue
        result[key] = {
            "p50_ms": float(value.get("p50_ms", 0.0)),
            "p95_ms": float(value.get("p95_ms", 0.0)),
            "base_p95_ms": float(value.get("base_p95_ms", 0.0)),
            "lora_tax_pct": float(value.get("lora_tax_pct", 0.0)),
            "hot_reload_ms_p95": float(value.get("hot_reload_ms_p95", 0.0)),
            "tokens_per_s": float(value.get("tokens_per_s", 0.0)),
            "queue_ms_p95": float(value.get("queue_ms_p95", 0.0)),
            "decode_util": float(value.get("decode_util", 0.0)),
            "adapter_util": float(value.get("adapter_util", 0.0)),
            "overlap_ratio": float(value.get("overlap_ratio", 0.0)),
        }
    return result


def parse_csv(path: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            bench = row.get("bench") or row.get("sched")
            if bench != "scheduler_zipf":
                continue
            key = f"load_{row.get('load_rps', 'unknown')}"
            results[key] = {
                "p50_ms": float(row.get("p50_ms", 0.0)),
                "p95_ms": float(row.get("p95_ms", 0.0)),
                "base_p95_ms": float(row.get("base_p95_ms", 0.0)),
                "lora_tax_pct": float(row.get("lora_tax_pct", 0.0)),
                "hot_reload_ms_p95": float(row.get("hot_reload_ms_p95", 0.0)),
                "tokens_per_s": float(row.get("tokens_per_s", 0.0)),
                "queue_ms_p95": float(row.get("queue_ms_p95", 0.0)),
                "load_rps": float(row.get("load_rps", 0.0)),
                "decode_util": float(row.get("decode_util", 0.0)),
                "adapter_util": float(row.get("adapter_util", 0.0)),
                "overlap_ratio": float(row.get("overlap_ratio", 0.0)),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scheduler Zipf metrics")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-out", "--baseline_out", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    results = parse_csv(args.csv)
    baseline_path = args.baseline or args.baseline_out
    baseline = load_baseline(baseline_path)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    write_baseline = args.update_baseline or args.write_baseline
    if not results:
        raise SystemExit("No scheduler_zipf rows found in CSV")

    if baseline and not write_baseline:
        for key, metrics in results.items():
            base = baseline.get(key, {})
            if base:
                if metrics.get("tokens_per_s", 0.0) < 0.9 * base.get("tokens_per_s", 0.0):
                    raise SystemExit(f"Scheduler throughput regression detected for {key}")
            lora_tax = metrics.get("lora_tax_pct", 0.0)
            if lora_tax > 10.0:
                print(f"[warn] LoRA tax {lora_tax:.2f}% exceeds 10% for {key}", file=os.sys.stderr)
            queue_p95 = metrics.get("queue_ms_p95", 0.0)
            if queue_p95 > 200.0:
                print(f"[warn] Queue wait {queue_p95:.2f} ms looks high for {key}", file=os.sys.stderr)

    if write_baseline:
        target = args.baseline_out or args.baseline
        if not target:
            raise SystemExit("--baseline-out (or --baseline) required when updating baseline")
        payload = {key: value for key, value in results.items()}
        payload["_meta"] = _hardware_meta()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
