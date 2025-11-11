from __future__ import annotations

import json
from pathlib import Path

from projects.bon_emergence.analyze import aggregate
from projects.bon_emergence.distill import build_dataset


def _fake_record(prompt: str, solved: bool) -> dict:
    samples = []
    for idx in range(3):
        samples.append(
            {
                "completion": f"answer-{idx}",
                "correct": solved and idx == 1,
                "details": {},
                "duration_s": 0.5,
                "length_chars": 5,
            }
        )
    pass_at = {"1": 1.0 if solved else 0.0, "8": 1.0 if solved else 0.0}
    return {
        "prompt": prompt,
        "answer": "answer-1",
        "samples": samples,
        "metrics": {
            "pass_at": pass_at,
            "unique_frac": 0.5,
            "entropy": 0.7,
            "avg_completion_length": 5.0,
            "avg_duration_s": 0.5,
            "avg_chars_per_sec": 10.0,
        },
        "adapter": {
            "base_model": "demo",
            "adapter_name": "demo",
            "adapter_path": "adapters/demo",
        },
        "scorer": "exact",
        "metadata": {},
        "timestamp": "20250101T000000Z",
    }


def test_distill_dataset(tmp_path: Path):
    records = [_fake_record("P1", True), _fake_record("P2", False)]
    out = tmp_path / "distill.jsonl"
    build_dataset(records, out, skip_unsolved=False)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert entries[0]["completion"] == "answer-1"


def test_aggregate_metrics():
    records = [_fake_record("P1", True), _fake_record("P2", False)]
    summary = aggregate(records)
    assert summary["num_prompts"] == 2
    assert summary["pass_at"]["1"] == 0.5
    assert "avg_duration_s" in summary
