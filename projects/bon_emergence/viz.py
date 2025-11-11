from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No records found in {path}")
    return records


def aggregate_pass(records: List[dict]) -> Tuple[List[int], List[float]]:
    totals: Dict[int, float] = {}
    for rec in records:
        for k_str, value in rec["metrics"]["pass_at"].items():
            k = int(k_str)
            totals.setdefault(k, 0.0)
            totals[k] += float(value)
    ks = sorted(totals)
    count = len(records)
    values = [totals[k] / count for k in ks]
    return ks, values


def minimal_k(records: List[dict]) -> List[int]:
    mins: List[int] = []
    for rec in records:
        solved_k = None
        for k_str, value in sorted(rec["metrics"]["pass_at"].items(), key=lambda kv: int(kv[0])):
            if float(value) >= 1.0:
                solved_k = int(k_str)
                break
        if solved_k is not None:
            mins.append(solved_k)
    return mins


def plot_pass_curves(base_k: List[int], base_vals: List[float], adapter_k: List[int], adapter_vals: List[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(base_k, base_vals, marker="o", label="Base model", linewidth=2, color="#1f77b4")
    plt.plot(adapter_k, adapter_vals, marker="o", label="Distilled adapter", linewidth=2, color="#ff7f0e")
    plt.xlabel("k (number of samples)")
    plt.ylabel("pass@k")
    plt.ylim(-0.05, 1.05)
    plt.xscale("log", base=2)
    plt.xticks(base_k, [str(k) for k in base_k])
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.title("Best-of-N performance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_emergence_hist(min_k_values: List[int], out_path: Path) -> None:
    if not min_k_values:
        return
    plt.figure(figsize=(5, 3))
    bins = sorted(set(min_k_values))
    plt.hist(min_k_values, bins=[b - 0.5 for b in bins] + [bins[-1] + 0.5], edgecolor="black", color="#6699cc")
    plt.xlabel("Smallest k where base succeeds")
    plt.ylabel("Number of prompts")
    plt.title("BoN emergence (base model)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(records: List[dict]) -> Dict[str, float]:
    latencies = [rec["metrics"].get("avg_duration_s", 0.0) for rec in records]
    entropies = [rec["metrics"].get("entropy", 0.0) for rec in records]
    unique = [rec["metrics"].get("unique_frac", 0.0) for rec in records]
    summary = {
        "avg_latency_s": sum(latencies) / len(latencies),
        "avg_entropy": sum(entropies) / len(entropies),
        "avg_unique_frac": sum(unique) / len(unique),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise BoN runs")
    parser.add_argument("--base", type=Path, required=True, help="Base run JSONL")
    parser.add_argument("--adapter", type=Path, required=True, help="Adapter run JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for charts")
    args = parser.parse_args()

    base_records = load_records(args.base)
    adapter_records = load_records(args.adapter)

    base_k, base_vals = aggregate_pass(base_records)
    adapter_k, adapter_vals = aggregate_pass(adapter_records)
    plot_pass_curves(base_k, base_vals, adapter_k, adapter_vals, args.out / "pass_curves.png")

    mins = minimal_k(base_records)
    plot_emergence_hist(mins, args.out / "emergence_hist.png")

    summary = {
        "base": summarize(base_records),
        "adapter": summarize(adapter_records),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved charts to {args.out}")


if __name__ == "__main__":
    main()
