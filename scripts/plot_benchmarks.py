#!/usr/bin/env python3
"""Visualize benchmark matrix results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px


def load_matrix(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for run in data.get("runs", []):
        metrics = run.get("metrics")
        if not metrics:
            continue
        rows.append(
            {
                "name": run.get("name"),
                "engine": run.get("engine"),
                "base_model": run.get("base_model"),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "seq_per_second": metrics.get("seq_per_second"),
                "avg_latency_s": metrics.get("avg_latency_s"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark throughput vs. engine.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("reports/benchmark_matrix.json"),
        help="Path to aggregated benchmark JSON.",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("reports/benchmark_tokens.html"),
        help="Output HTML for tokens/sec bar chart.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/benchmark_tokens.csv"),
        help="Optional CSV export of summarized metrics.",
    )
    args = parser.parse_args()

    df = load_matrix(args.matrix)
    if df.empty:
        raise ValueError(f"No metrics found in {args.matrix}")

    df.sort_values(by="tokens_per_second", ascending=False, inplace=True)
    fig = px.bar(
        df,
        x="name",
        y="tokens_per_second",
        color="engine",
        hover_data=["base_model", "seq_per_second", "avg_latency_s"],
        title="Benchmark Tokens/sec by Engine",
    )
    fig.write_html(args.out_html)
    df.to_csv(args.out_csv, index=False)
    print(f"[ok] Wrote {args.out_html} and {args.out_csv}")


if __name__ == "__main__":
    main()
