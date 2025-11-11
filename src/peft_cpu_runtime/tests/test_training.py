from __future__ import annotations

from pathlib import Path
import sys

SYS_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SYS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_SRC_ROOT))

import pytest
import torch
from peft_cpu_runtime.training import TrainingConfig, train_lora_adapter
from peft_cpu_runtime.training.rewards import get_reward


@pytest.mark.skipif(torch.cuda.is_available(), reason="skip on CUDA hosts to keep runtime low")
def test_train_lora_adapter(tmp_path: Path) -> None:
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

