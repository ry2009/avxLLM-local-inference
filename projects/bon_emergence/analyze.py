from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_run_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob("*.jsonl"))


def _load_records(paths: Iterable[Path]) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    if not records:
        raise SystemExit("No records found")
    return records


def aggregate(records: List[dict]) -> dict:
    pass_keys = set()
    for record in records:
        pass_keys.update(record["metrics"]["pass_at"].keys())
    pass_totals = {key: 0.0 for key in pass_keys}
    for record in records:
        pa = record["metrics"]["pass_at"]
        for key in pass_keys:
            pass_totals[key] += float(pa.get(key, 0.0))
    n = len(records)
    pass_avg = {key: value / n for key, value in pass_totals.items()}
    unique = statistics.mean(r["metrics"].get("unique_frac", 0.0) for r in records)
    entropy = statistics.mean(r["metrics"].get("entropy", 0.0) for r in records)
    avg_duration = statistics.mean(r["metrics"].get("avg_duration_s", 0.0) for r in records)
    avg_chars_per_sec = statistics.mean(r["metrics"].get("avg_chars_per_sec", 0.0) for r in records)
    solved = sum(1 for r in records if any(v >= 1.0 for k, v in r["metrics"]["pass_at"].items() if k != "1"))
    return {
        "num_prompts": n,
        "pass_at": pass_avg,
        "unique_frac": unique,
        "entropy": entropy,
        "avg_duration_s": avg_duration,
        "avg_chars_per_sec": avg_chars_per_sec,
        "coverage": solved / n if n else 0.0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarise BoN run files")
    parser.add_argument("--runs", type=Path, required=True, help="Collector run JSONL or directory")
    parser.add_argument("--out", type=Path, help="Optional JSON summary path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    records = _load_records(_iter_run_files(args.runs))
    summary = aggregate(records)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
