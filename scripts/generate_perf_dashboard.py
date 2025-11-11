#!/usr/bin/env python3
"""Generate Plotly dashboard for perf report data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

ATTN_CSV = Path("reports/attn_decode_relwithdeb.csv")
PREFILL_JSON = Path("reports/attn_prefill_summary.json")
TOKENIZER_CSV = Path("reports/tokenizer.csv")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build perf dashboard HTML")
    parser.add_argument("--output", default="reports/perf_dashboard.html")
    return parser.parse_args(argv)


def _decode_fig(df: pd.DataFrame) -> go.Figure:
    subset = df[df["bench"].str.contains("attn_decode_int8_qk")]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset["seq_len"], y=subset["tokens_per_s"] / 1e6, name="int8 tokens/s"))
    fig.add_trace(go.Bar(x=subset["seq_len"], y=subset["baseline_tokens_per_s"] / 1e6, name="baseline tokens/s"))
    fig.update_layout(title="Decode throughput (millions tokens/s)", xaxis_title="Sequence length")
    return fig


def _prefill_fig(data: dict) -> go.Figure:
    labels = list(data.keys())
    tokens = [data[label]["tokens_per_s"] / 1e3 for label in labels]
    fig = go.Figure(go.Bar(x=labels, y=tokens, name="Prefill tokens/s (K)"))
    fig.update_layout(title="Prefill throughput", xaxis_title="Scenario")
    return fig


def _tokenizer_fig(df: pd.DataFrame) -> go.Figure:
    subset = df[df["bench"] == "tokenizer"]
    fig = go.Figure(go.Scatter(x=subset["threads"], y=subset["tokens_per_s"], mode="lines+markers"))
    fig.update_layout(title="Rust tokenizer tokens/s vs threads", xaxis_title="threads")
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    decode_fig = _decode_fig(pd.read_csv(ATTN_CSV))
    prefill_fig = _prefill_fig(json.loads(PREFILL_JSON.read_text()))
    tokenizer_fig = _tokenizer_fig(pd.read_csv(TOKENIZER_CSV))

    html = "\n".join(fig.to_html(full_html=False, include_plotlyjs="cdn") for fig in [decode_fig, prefill_fig, tokenizer_fig])
    out = Path(args.output)
    out.write_text("<html><body>" + html + "</body></html>")
    print(f"Dashboard written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
