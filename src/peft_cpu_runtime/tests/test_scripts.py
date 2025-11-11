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
run_end_to_end = _load_script("run_end_to_end")
run_telemetry_matrix = _load_script("run_telemetry_matrix")
run_ci_smoke = _load_script("run_ci_smoke")
run_throughput_sweep = _load_script("run_throughput_sweep")
download_manifest = _load_script("download_manifest")
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


def test_download_manifest(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "models": [{"id": "foo/base", "dir": str(tmp_path / "model")}] ,
        "adapters": [{"id": "foo/adapter", "dir": str(tmp_path / "adapter")}] ,
    }))

    calls = []

    def fake_download(*, repo_id, local_dir, local_dir_use_symlinks, revision, token, resume_download):
        calls.append(repo_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    monkeypatch.setattr(download_manifest, "snapshot_download", fake_download)

    exit_code = download_manifest.main([str(manifest)])
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert {entry["id"] for entry in result} == {"foo/base", "foo/adapter"}
    assert calls == ["foo/base", "foo/adapter"]


def test_run_telemetry_matrix(monkeypatch, tmp_path, capsys):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("line1\nline2\n")

    monkeypatch.setattr(run_telemetry_matrix, "snapshot_download", lambda **kwargs: tmp_path)
    monkeypatch.setattr(run_telemetry_matrix, "CpuPeftRuntime", _FakeRuntime)

    exit_code = run_telemetry_matrix.main(
        [
            "--model-id",
            "foo/base",
            "--model-dir",
            str(tmp_path / "model"),
            "--adapter",
            "demo=foo/adapter",
            "--prompts",
            str(prompts_file),
            "--out",
            str(tmp_path / "matrix.json"),
        ]
    )
    assert exit_code == 0
    matrix = json.loads((tmp_path / "matrix.json").read_text())
    assert matrix[0]["adapter"] == "demo"


def test_run_ci_smoke(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_ci_smoke, "snapshot_download", lambda **kwargs: tmp_path)
    monkeypatch.setattr(run_ci_smoke, "CpuPeftRuntime", _FakeRuntime)

    exit_code = run_ci_smoke.main([
        "--model-id",
        "foo/base",
        "--model-dir",
        str(tmp_path / "model"),
        "--metrics",
        str(tmp_path / "metrics.json"),
    ])
    assert exit_code == 0
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert "metrics" in payload


def test_run_throughput_sweep(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_throughput_sweep, "snapshot_download", lambda **kwargs: tmp_path)
    monkeypatch.setattr(run_throughput_sweep, "CpuPeftRuntime", _FakeRuntime)

    exit_code = run_throughput_sweep.main(
        [
            "--model-id",
            "foo/base",
            "--model-dir",
            str(tmp_path / "model"),
            "--lengths",
            "8",
            "--out",
            str(tmp_path / "sweep.json"),
        ]
    )
    assert exit_code == 0
    data = json.loads((tmp_path / "sweep.json").read_text())
    assert data[0]["length"] == 8


def test_throughput_sweep_threshold(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_throughput_sweep, "snapshot_download", lambda **kwargs: tmp_path)

    class _SlowRuntime(_FakeRuntime):
        def benchmark(self, batch, num_warmup, num_iters):
            return {"tokens_per_second": 0.1, "avg_ttft_s": 1.5, "iterations": []}

    monkeypatch.setattr(run_throughput_sweep, "CpuPeftRuntime", _SlowRuntime)

    with pytest.raises(SystemExit):
        run_throughput_sweep.main(
            [
                "--lengths",
                "8",
                "--min-tps",
                "1.0",
                "--max-ttft",
                "1.0",
            ]
        )


def test_run_end_to_end_script(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text": "prompt"}\n')
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text('{"text": "Eval"}\n')

    monkeypatch.setattr(run_end_to_end, "snapshot_download", lambda **kwargs: tmp_path)

    def fake_train(cfg):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return adapter_dir

    monkeypatch.setattr(run_end_to_end, "train_lora_adapter", fake_train)
    monkeypatch.setattr(run_end_to_end, "CpuPeftRuntime", _FakeRuntime)

    exit_code = run_end_to_end.main(
        [
            "--model-id",
            "foo/base",
            "--model-dir",
            str(tmp_path / "model"),
            "--dataset",
            str(dataset),
            "--eval-prompts",
            str(eval_file),
            "--eval-limit",
            "1",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["prompts"] == ["Eval"]
