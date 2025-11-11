#!/usr/bin/env python3
"""Summarize tokenizer perf CSV outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reports/tokenizer.csv")
    parser.add_argument("--csv", default="reports/tokenizer.csv")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    tokenizer = df[df["bench"] == "tokenizer"]
    grouped = tokenizer.groupby("threads")["tokens_per_s"].mean().reset_index()
    best = grouped.sort_values("tokens_per_s", ascending=False).iloc[0]
    summary = {
        "best_threads": int(best["threads"]),
        "best_tokens_per_s": float(best["tokens_per_s"]),
        "entries": grouped.to_dict("records"),
    }
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
