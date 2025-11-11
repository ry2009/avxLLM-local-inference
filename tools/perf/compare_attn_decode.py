#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import os
import platform
import sys
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
    if platform.machine().lower().endswith("64"):
        flags.append("x86_64")
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
            "tokens_per_s": float(value.get("tokens_per_s", 0.0)),
            "scratch_ratio": float(value.get("scratch_ratio", 0.0)),
            "tile_tokens": float(value.get("tile_tokens", 0.0)),
            "l2_bytes_per_token": float(value.get("l2_bytes_per_token", 0.0)),
        }
    return result


def parse_csv(path: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("bench") != "attn_decode_int8_qk":
                continue
            key = f"seq{row['seq_len']}_dim{row['head_dim']}_h{row['heads']}_b{row['batch']}"
            results[key] = {
                "seq_len": int(row.get("seq_len", 0)),
                "head_dim": int(row.get("head_dim", 0)),
                "heads": int(row.get("heads", 0)),
                "batch": int(row.get("batch", 0)),
                "tokens_per_s": float(row.get("tokens_per_s", 0.0)),
                "baseline_tokens_per_s": float(row.get("baseline_tokens_per_s", 0.0)),
                "speedup": float(row.get("speedup", 0.0)),
                "l2_diff": float(row.get("l2_diff", 0.0)),
                "perplexity_delta": float(row.get("perplexity_delta", 0.0)),
                "scratch_ratio": float(row.get("scratch_ratio", 0.0)),
                "scratch_bytes": float(row.get("scratch_bytes", 0.0)),
                "tile_tokens": float(row.get("tile_tokens", 0.0)) if row.get("tile_tokens") else 0.0,
                "acc_width": row.get("acc_width", ""),
                "has_avx2": bool(int(row.get("has_avx2", 0))) if row.get("has_avx2") else False,
                "has_vnni": bool(int(row.get("has_vnni", 0))) if row.get("has_vnni") else False,
                "csr_outliers": bool(int(row.get("csr_outliers", 0))) if row.get("csr_outliers") else False,
                "l2_bytes_per_token": float(row.get("l2_bytes_per_token", 0.0)),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attention decode performance")
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

    primary_key = "seq2048_dim64_h32_b1"
    primary_speedup = None

    for key, metrics in results.items():
        if key == primary_key:
            primary_speedup = metrics.get("speedup", 0.0)
        if not write_baseline:
            if metrics["l2_diff"] > 1e-2:
                raise SystemExit(f"Decode L2 diff too large for {key}: {metrics['l2_diff']:.5f}")
            if abs(metrics.get("perplexity_delta", 0.0)) > 0.3:
                raise SystemExit(
                    f"Decode perplexity delta exceeded 0.3pp for {key} (got {metrics['perplexity_delta']:.3f}pp)"
                )
            if metrics.get("seq_len", 0) >= 2048 and metrics.get("speedup", 0.0) < 1.15:
                raise SystemExit(
                    f"Decode speedup below 1.15× for {key} (got {metrics['speedup']:.2f}×)"
                )
            if metrics.get("scratch_ratio", 0.0) > 1.25:
                raise SystemExit(
                    f"Decode scratch usage regression for {key}: {metrics['scratch_ratio']:.2f}× (>1.25×)"
                )
            if metrics.get("head_dim") == 128 and metrics.get("speedup", 0.0) < 1.08:
                print(
                    f"[warn] Decode speedup {metrics['speedup']:.2f}× < 1.08× target for {key}",
                    file=sys.stderr,
                )
        base = baseline.get(key)
        if not write_baseline and base:
            limit = (1.0 - args.regression_threshold) * base.get("tokens_per_s", 0.0)
            if metrics["tokens_per_s"] < limit:
                raise SystemExit(
                    f"Decode perf regression for {key}: {metrics['tokens_per_s']:.2f} < {limit:.2f}"
                )

    if not write_baseline:
        if primary_speedup is None:
            raise SystemExit(
                f"Primary decode row {primary_key} missing from results; cannot evaluate gate"
            )
        if primary_speedup < 1.15:
            raise SystemExit(
                f"Primary decode speedup below 1.15× for {primary_key} (got {primary_speedup:.2f}×)"
            )

    if write_baseline:
        target = args.baseline_out or args.baseline
        if not target:
            raise SystemExit("--baseline-out (or --baseline) required when updating baseline")
        payload = {
            key: {
                "tokens_per_s": value["tokens_per_s"],
                "scratch_ratio": value.get("scratch_ratio", 0.0),
                "tile_tokens": value.get("tile_tokens", 0.0),
                "l2_bytes_per_token": value.get("l2_bytes_per_token", 0.0),
            }
            for key, value in results.items()
        }
        tile_meta: Dict[str, float] = {}
        for key, value in results.items():
            head_dim = value.get("head_dim")
            tile = value.get("tile_tokens")
            if head_dim and tile and f"dim{head_dim}" not in tile_meta:
                tile_meta[f"dim{head_dim}"] = tile
        meta = _hardware_meta()
        if tile_meta:
            meta["tile_tokens"] = tile_meta
        payload["_meta"] = meta
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
