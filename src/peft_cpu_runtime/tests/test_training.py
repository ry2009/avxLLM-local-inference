from __future__ import annotations

from pathlib import Path
import sys

SYS_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SYS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_SRC_ROOT))

import types

import pytest
import torch
from peft_cpu_runtime.training import TrainingConfig, train_lora_adapter
from peft_cpu_runtime.training.rewards import get_reward
from peft_cpu_runtime.training import sft as sft_module


class _DummyTokenizer:
    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        size = len(texts)
        input_ids = torch.zeros((size, max_length), dtype=torch.long)
        attention = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention}

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}")


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_text("stub")

    def forward(self, input_ids, attention_mask, labels):
        return types.SimpleNamespace(loss=self.param.sum())


def test_train_lora_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sft_module, "prepare_tokenizer", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr(sft_module, "prepare_lora_model", lambda **kwargs: _DummyModel())

    prompts = ["Hello", "World"]
    cfg = TrainingConfig(
        base_model="sshleifer/tiny-gpt2",
        adapter_name="test-adapter",
        output_dir=tmp_path,
        epochs=1,
        batch_size=1,
        max_seq_len=32,
    )
    adapter_path = train_lora_adapter(prompts, cfg)
    assert adapter_path.exists()


def test_builtin_reward_registry() -> None:
    reward = get_reward("length")
    scores = reward(["hello"], ["short completion"], None)
    assert len(scores) == 1
    assert scores[0] > 0.0
