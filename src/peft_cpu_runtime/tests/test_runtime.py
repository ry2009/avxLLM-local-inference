from __future__ import annotations

from pathlib import Path
import sys

import torch
import pytest

SYS_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SYS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_SRC_ROOT))

from peft_cpu_runtime import runtime as runtime_mod


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = 0

    def __call__(self, prompts, padding=True, return_tensors="pt"):
        max_len = max(len(p.split()) + 1 for p in prompts)
        encoded = []
        masks = []
        for prompt in prompts:
            length = len(prompt.split()) + 1
            tokens = list(range(1, length + 1))
            pad = [0] * (max_len - len(tokens))
            encoded.append(tokens + pad)
            masks.append([1] * len(tokens) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens=True):
        return f"decoded({len(tokens)})"

    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        return list(range(len(tokens) or 1))


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeTokenizer()


class _FakeModel:
    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, generation_config):
        batch, _ = input_ids.shape
        new_tokens = torch.full((batch, generation_config.max_new_tokens), 7, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeModel()


@pytest.mark.parametrize("enable_profiling", [True, False])
def test_runtime_generates_and_emits_metrics(monkeypatch, enable_profiling):
    monkeypatch.setattr(runtime_mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(runtime_mod, "AutoModelForCausalLM", _FakeAutoModel)

    runtime = runtime_mod.CpuPeftRuntime(
        base_model_id="dummy",
        adapter_map={},
        torch_dtype=torch.float32,
    )
    runtime.enable_profiling(enable_profiling)

    batch = runtime_mod.RequestBatch(
        requests=[
            runtime_mod.InferenceRequest(prompt="Hello world"),
            runtime_mod.InferenceRequest(prompt="How are you"),
        ],
        trace_config=runtime_mod.InferenceTraceConfig(max_new_tokens=2),
    )

    outputs = runtime.generate(batch)
    assert outputs and all(output.startswith("decoded") for output in outputs)

    metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)
    assert metrics["tokens_per_second"] > 0
    if enable_profiling:
        assert metrics["avg_ttft_s"] is not None
        iter_entry = metrics["iterations"][0]
        assert "ttft_s" in iter_entry and iter_entry["ttft_s"] >= 0
    else:
        assert metrics["avg_ttft_s"] is None


def test_runtime_respects_num_threads(monkeypatch):
    monkeypatch.setattr(runtime_mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(runtime_mod, "AutoModelForCausalLM", _FakeAutoModel)

    captured = {}

    def fake_set_num_threads(value: int) -> None:
        captured["value"] = value

    monkeypatch.setattr(runtime_mod.torch, "set_num_threads", fake_set_num_threads)

    runtime = runtime_mod.CpuPeftRuntime(
        base_model_id="dummy",
        adapter_map={},
        torch_dtype=torch.float32,
        num_threads=6,
    )
    assert runtime.num_threads == 6
    assert captured["value"] == 6


def test_runtime_env_threads(monkeypatch):
    monkeypatch.setattr(runtime_mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(runtime_mod, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setenv("INFENG_NUM_THREADS", "4")

    captured = {}

    def fake_set_num_threads(value: int) -> None:
        captured["value"] = value

    monkeypatch.setattr(runtime_mod.torch, "set_num_threads", fake_set_num_threads)

    runtime = runtime_mod.CpuPeftRuntime(
        base_model_id="dummy",
        adapter_map={},
        torch_dtype=torch.float32,
    )
    assert runtime.num_threads == 4
    assert captured["value"] == 4
