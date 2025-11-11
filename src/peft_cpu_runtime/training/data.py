from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Dataset

from .config import DatasetConfig


def _load_json_lines(path: Path, field: str) -> List[str]:
    values: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if field not in payload:
                raise KeyError(f"Field '{field}' missing in JSONL record from {path}")
            values.append(payload[field])
    return values


def _load_json(path: Path, field: str) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if field not in payload:
            raise KeyError(f"Field '{field}' missing in JSON object from {path}")
        data = payload[field]
        if not isinstance(data, list):
            raise TypeError(f"Expected list for '{field}' in {path}, found {type(data)}")
        return [str(item) for item in data]
    if isinstance(payload, list):
        values = []
        for item in payload:
            if field not in item:
                raise KeyError(f"Field '{field}' missing in JSON array item from {path}")
            values.append(item[field])
        return values
    raise TypeError(f"Unsupported JSON format in {path}")


def _load_csv(path: Path, field: str) -> List[str]:
    values = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if field not in row:
                raise KeyError(f"Field '{field}' missing in CSV row from {path}")
            values.append(row[field])
    return values


def load_prompts(cfg: DatasetConfig, fallback: Optional[Iterable[str]] = None) -> List[str]:
    """Load prompts either from the configured dataset or from the provided fallback."""

    prompts: List[str]
    if cfg.path:
        path = Path(cfg.path)
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".jsonl.gz"}:
            prompts = _load_json_lines(path, cfg.field)
        elif suffix == ".json":
            prompts = _load_json(path, cfg.field)
        elif suffix in {".csv", ".tsv"}:
            prompts = _load_csv(path, cfg.field)
        else:
            raise ValueError(f"Unsupported dataset format: {cfg.path}")
    else:
        prompts = list(fallback or [])

    if cfg.max_samples:
        prompts = prompts[: cfg.max_samples]
    if cfg.shuffle:
        from random import shuffle

        shuffle(prompts)
    return prompts


def build_hf_dataset(prompts: List[str]) -> Dataset:
    return Dataset.from_dict({"text": prompts})


__all__ = ["load_prompts", "build_hf_dataset"]
