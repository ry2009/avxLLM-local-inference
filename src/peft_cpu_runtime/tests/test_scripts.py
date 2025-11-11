from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def _load_script(filename: str):
    spec = importlib.util.spec_from_file_location(filename, SCRIPTS_DIR / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[filename] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


run_local_inference = _load_script("run_local_inference")
run_local_eval = _load_script("run_local_eval")
check_mac_env = _load_script("check_mac_env")


class _FakeRuntime:
    def __init__(self, base_model_id, adapter_map, torch_dtype):
        self.base_model_id = base_model_id
        self.adapter_map = adapter_map
        self.telemetry = False

    def enable_profiling(self, enabled: bool) -> None:
        self.telemetry = enabled

    def generate(self, batch):
        return [f"out-{idx}" for idx, _ in enumerate(batch.requests)]

    def benchmark(self, batch, num_warmup, num_iters):
        return {"tokens_per_second": 100, "iterations": []}


@pytest.fixture(autouse=True)
def _clear_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    yield


def test_run_local_inference_script(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_local_inference, "_download", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(run_local_inference, "CpuPeftRuntime", _FakeRuntime)

    exit_code = run_local_inference.main([
        "--model-id",
        "foo/base",
        "--model-dir",
        str(tmp_path / "model"),
        "--prompts",
        "Hello",
        "there",
        "--telemetry",
        "--benchmark-iters",
        "1",
    ])
    assert exit_code == 0
    output = capsys.readouterr().out
    main_json = output.split("\nTelemetry", 1)[0].strip()
    payload = json.loads(main_json)
    assert payload["prompts"] == ["Hello", "there"]


def test_run_local_eval_script(monkeypatch, tmp_path, capsys):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"text": "Hi"}\n{"text": "There"}\n')

    monkeypatch.setattr(run_local_eval, "_download", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(run_local_eval, "CpuPeftRuntime", _FakeRuntime)

    metrics_path = tmp_path / "metrics.json"
    exit_code = run_local_eval.main([
        "--model-id",
        "foo/base",
        "--model-dir",
        str(tmp_path / "model"),
        "--prompts-file",
        str(prompt_file),
        "--max-prompts",
        "2",
        "--telemetry",
        "--metrics-out",
        str(metrics_path),
    ])
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["tokens_per_second"] == 100
    eval_json = stdout.split("\nSaved telemetry", 1)[0].strip()
    payload = json.loads(eval_json)
    assert payload["prompts"] == ["Hi", "There"]


def test_check_mac_env_json(monkeypatch, capsys):
    fake_state: Dict[str, str | None] = {"cmake": "/usr/bin/cmake"}

    def fake_which(cmd: str) -> str | None:
        return fake_state.get(cmd)

    monkeypatch.setattr(check_mac_env, "REQUIREMENTS", {"cmake": "hint"})
    monkeypatch.setattr(check_mac_env.shutil, "which", fake_which)

    exit_code = check_mac_env.main(["--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cmake"]["available"] is True
