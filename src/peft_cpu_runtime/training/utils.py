from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional
import warnings

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import LoRAConfig


def resolve_dtype(name: Optional[str], default: torch.dtype = torch.float32) -> torch.dtype:
    if not name:
        return default
    normalized = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "single": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "bfloat": torch.bfloat16,
    }
    if normalized in ("auto", "default"):
        return default
    try:
        dtype = mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of: {', '.join(sorted(mapping))}.") from exc
    if dtype == torch.float16 and not torch.cuda.is_available():
        warnings.warn(
            "float16 is not fully supported on CPU; defaulting to float32 for stability.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.float32
    return dtype


def prepare_tokenizer(model_id: str, cache_dir: Optional[Path] = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=str(cache_dir) if cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_lora_model(
    base_model: str,
    lora_cfg: LoRAConfig,
    cache_dir: Optional[Path] = None,
    dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    cache = str(cache_dir) if cache_dir else None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=cache,
            use_safetensors=True,
            torch_dtype=dtype,
        )
    except (OSError, ValueError) as exc:
        error_message = str(exc)
        if "safetensor" in error_message.lower() or "safetensors" in error_message.lower():
            raise RuntimeError(
                f"{base_model} does not provide safetensors weights. "
                "Pick a safetensors-backed checkpoint or upgrade PyTorch to >=2.6 to load legacy .bin files."
            ) from exc
        raise
    lora_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
    )
    peft_model = get_peft_model(model, lora_cfg)
    if dtype is not None:
        peft_model.to(dtype=dtype)
    return peft_model


@contextmanager
def torch_no_grad():
    old = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(old)


def chunked(seq: Iterable, size: int):
    chunk = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = [
    "prepare_tokenizer",
    "prepare_lora_model",
    "torch_no_grad",
    "chunked",
    "resolve_dtype",
]
