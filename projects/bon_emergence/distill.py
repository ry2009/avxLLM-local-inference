from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _iter_run_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.glob("*.jsonl")):
        if candidate.is_file():
            yield candidate


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
        raise SystemExit("No BoN run records found")
    return records


def build_dataset(records: List[dict], out_path: Path, skip_unsolved: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for record in records:
            samples = record.get("samples", [])
            best = next((sample for sample in samples if sample.get("correct")), None)
            if best is None:
                if skip_unsolved:
                    skipped += 1
                    continue
                if not samples:
                    skipped += 1
                    continue
                best = samples[0]
            combined = f"{record['prompt']} {best['completion']}"
            obj = {
                "prompt": record["prompt"],
                "completion": best["completion"],
                "text": combined,
            }
            fp.write(json.dumps(obj) + "\n")
            kept += 1
    print(f"Wrote {kept} training pairs to {out_path} (skipped {skipped})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create best-of-N distillation dataset")
    parser.add_argument("--runs", type=Path, required=True, help="Collector run JSONL or directory")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL dataset path")
    parser.add_argument("--keep-unsolved", action="store_true", help="Include prompts without correct samples")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_files = list(_iter_run_files(args.runs))
    records = _load_records(run_files)
    build_dataset(records, args.out, skip_unsolved=not args.keep_unsolved)


if __name__ == "__main__":
    main()
