from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence


def _expand_prompts(prompts: Optional[Sequence[str]] | None) -> list[str]:
    return list(prompts) if prompts else []


@dataclass
class DatasetConfig:
    """Generic dataset specification used across training modes."""

    path: Optional[Path] = None
    field: str = "text"
    max_samples: Optional[int] = None
    shuffle: bool = True


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class OptimizerConfig:
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0


@dataclass
class TrainingConfig:
    """Backwards-compatible LoRA fine-tune config (SFT)."""

    base_model: str
    adapter_name: str
    output_dir: Path
    prompts: list[str] = field(default_factory=list)
    epochs: int = 1
    batch_size: int = 1
    max_seq_len: int = 256
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model_dtype: str = "float32"
    base_safetensors: Optional[Path] = None
    hf_cache_dir: Optional[Path] = None

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            base_model=data["base_model"],
            adapter_name=data["adapter_name"],
            output_dir=Path(data["output_dir"]),
            prompts=_expand_prompts(data.get("prompts")),
            epochs=data.get("epochs", 1),
            batch_size=data.get("batch_size", 1),
            max_seq_len=data.get("max_seq_len", 256),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            model_dtype=data.get("model_dtype", "float32"),
            base_safetensors=Path(data["base_safetensors"]) if data.get("base_safetensors") else None,
            hf_cache_dir=Path(data["hf_cache_dir"]) if data.get("hf_cache_dir") else None,
        )


SFTConfig = TrainingConfig


@dataclass
class PretrainConfig:
    base_model: str
    output_dir: Path
    dataset: DatasetConfig
    epochs: int = 1
    batch_size: int = 1
    max_seq_len: int = 256
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model_dtype: str = "float32"

    @classmethod
    def from_json(cls, path: Path) -> "PretrainConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            base_model=data["base_model"],
            output_dir=Path(data["output_dir"]),
            dataset=DatasetConfig(**data.get("dataset", {})),
            epochs=data.get("epochs", 1),
            batch_size=data.get("batch_size", 1),
            max_seq_len=data.get("max_seq_len", 256),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            model_dtype=data.get("model_dtype", "float32"),
        )


RewardFn = Callable[[list[str], list[str], dict | None], list[float]]


@dataclass
class RLConfig:
    base_model: str
    adapter_name: str
    output_dir: Path
    dataset: DatasetConfig
    reward_fn: RewardFn
    max_new_tokens: int = 64
    epochs: int = 1
    batch_size: int = 1
    mini_batch_size: int = 1
    gamma: float = 0.99
    kl_coef: float = 0.01
    baseline_momentum: float = 0.9
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    model_dtype: str = "float32"
    hf_cache_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: dict, reward_fn: RewardFn) -> "RLConfig":
        return cls(
            base_model=data["base_model"],
            adapter_name=data["adapter_name"],
            output_dir=Path(data["output_dir"]),
            dataset=DatasetConfig(**data.get("dataset", {})),
            reward_fn=reward_fn,
            max_new_tokens=data.get("max_new_tokens", 64),
            epochs=data.get("epochs", 1),
            batch_size=data.get("batch_size", 1),
            mini_batch_size=data.get("mini_batch_size", 1),
            gamma=data.get("gamma", 0.99),
            kl_coef=data.get("kl_coef", 0.01),
            baseline_momentum=data.get("baseline_momentum", 0.9),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            model_dtype=data.get("model_dtype", "float32"),
            hf_cache_dir=Path(data["hf_cache_dir"]) if data.get("hf_cache_dir") else None,
        )

    @classmethod
    def from_json(cls, path: Path, reward_resolver: Callable[[str], RewardFn]) -> "RLConfig":
        data = json.loads(Path(path).read_text())
        reward_spec = data.get("reward")
        if not reward_spec:
            raise ValueError("RL config JSON must include 'reward'")
        reward_fn = reward_resolver(reward_spec)
        return cls.from_dict(data, reward_fn)


__all__ = [
    "DatasetConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "SFTConfig",
    "PretrainConfig",
    "RLConfig",
    "RewardFn",
]
