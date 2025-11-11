"""
Utilities for local fine-tuning (SFT/pre-training) and lightweight RL on CPU-only Macs.
"""

from .config import (
    DatasetConfig,
    LoRAConfig,
    OptimizerConfig,
    PretrainConfig,
    RLConfig,
    SFTConfig,
    TrainingConfig,
)
from .sft import evaluate_adapter, export_adapter_to_infq, train_lora_adapter
from .rl import train_policy_rl
from .pretrain import train_causal_lm

__all__ = [
    "DatasetConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "SFTConfig",
    "PretrainConfig",
    "RLConfig",
    "train_lora_adapter",
    "evaluate_adapter",
    "export_adapter_to_infq",
    "train_causal_lm",
    "train_policy_rl",
]
