#!/usr/bin/env python3
"""Compare tokenizer benchmarks and enforce minimum speedup."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_native(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values[int(row["threads"])] = float(row["tokens_per_s"])
    if not values:
        raise SystemExit(f"no data in {path}")
    return values


def read_python_tokens(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            return float(row["tokens_per_s"])
    raise SystemExit(f"no data in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=Path("reports/tokenizer.csv"))
    parser.add_argument("--python", type=Path, default=Path("reports/tokenizer_py.csv"))
    parser.add_argument("--min-speedup", type=float, default=5.0)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()

    native = read_native(args.native)
    python = read_python_tokens(args.python)
    best_native = max(native.values())
    speedup = best_native / python if python else 0.0

    print("Native throughput by threads:")
    for threads, value in sorted(native.items()):
        print(f"  threads={threads}: {value:.2f} tokens/s")
    print(f"Python baseline: {python:.2f} tokens/s")
    print(f"Tokenizer speedup (best/native vs python): {speedup:.2f}x")

    if speedup < args.min_speedup:
        sys.exit(f"Speedup {speedup:.2f}x below threshold {args.min_speedup:.2f}x")

    if args.baseline:
        data = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_native = data.get("native", {})
        for threads_str, expected in baseline_native.items():
            threads = int(threads_str)
            actual = native.get(threads)
            if actual is None:
                continue
            if expected > 0.0 and actual < 0.9 * expected:
                sys.exit(
                    f"Native throughput for threads={threads} ({actual:.2f}) below 90% of baseline {expected:.2f}"
                )

if __name__ == "__main__":
    main()
