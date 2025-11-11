"""
CPU-first runtime utilities for serving PEFT (LoRA) adapters.
"""

from .runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from .training import (
    DatasetConfig,
    LoRAConfig,
    OptimizerConfig,
    TrainingConfig,
    train_lora_adapter,
    evaluate_adapter,
    export_adapter_to_infq,
    train_policy_rl,
    train_causal_lm,
)

try:  # optional dependency for GGUF / llama.cpp-backed inference
    from .quantized import LlamaCppConfig, LlamaCppPeftRuntime  # type: ignore
except ImportError:
    LlamaCppConfig = None  # type: ignore[assignment]
    LlamaCppPeftRuntime = None  # type: ignore[assignment]

__all__ = [
    "CpuPeftRuntime",
    "InferenceRequest",
    "InferenceTraceConfig",
    "RequestBatch",
    "DatasetConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "train_lora_adapter",
    "evaluate_adapter",
    "export_adapter_to_infq",
    "train_policy_rl",
    "train_causal_lm",
]

if LlamaCppConfig is not None and LlamaCppPeftRuntime is not None:
    __all__ += ["LlamaCppConfig", "LlamaCppPeftRuntime"]
