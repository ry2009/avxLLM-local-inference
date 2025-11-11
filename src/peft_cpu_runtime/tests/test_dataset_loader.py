from __future__ import annotations

import json
from pathlib import Path
import sys

SYS_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SYS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_SRC_ROOT))

from peft_cpu_runtime.training.config import DatasetConfig
from peft_cpu_runtime.training.data import load_prompts


def test_load_prompts_jsonl(tmp_path: Path) -> None:
    data = tmp_path / "samples.jsonl"
    data.write_text('\n'.join([json.dumps({"text": "A"}), json.dumps({"text": "B"})]))
    cfg = DatasetConfig(path=data, field="text", max_samples=1, shuffle=False)
    prompts = load_prompts(cfg)
    assert prompts == ["A"]


def test_load_prompts_json(tmp_path: Path) -> None:
    data = tmp_path / "samples.json"
    data.write_text(json.dumps([{"text": "one"}, {"text": "two"}]))
    cfg = DatasetConfig(path=data, field="text", shuffle=False)
    prompts = load_prompts(cfg)
    assert prompts == ["one", "two"]


def test_load_prompts_csv(tmp_path: Path) -> None:
    data = tmp_path / "samples.csv"
    data.write_text("text\nalpha\nbeta\n")
    cfg = DatasetConfig(path=data, field="text", shuffle=False)
    prompts = load_prompts(cfg)
    assert prompts == ["alpha", "beta"]


def test_load_prompts_fallback_list() -> None:
    cfg = DatasetConfig(path=None, shuffle=False)
    prompts = load_prompts(cfg, fallback=["hello", "world"])
    assert prompts == ["hello", "world"]
