from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import TrainingConfig
from .data import build_hf_dataset, load_prompts
from .utils import prepare_lora_model, prepare_tokenizer, resolve_dtype
from ..runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch


def _build_dataloader(prompts: List[str], tokenizer: AutoTokenizer, cfg: TrainingConfig) -> DataLoader:
    dataset = build_hf_dataset(prompts)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=cfg.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return DataLoader(tokenized, batch_size=cfg.batch_size)


def train_lora_adapter(
    prompts: Union[Iterable[str], TrainingConfig, None],
    cfg: Optional[TrainingConfig] = None,
) -> Path:
    if isinstance(prompts, TrainingConfig):
        cfg = prompts
        prompt_source: Optional[Iterable[str]] = None
    else:
        prompt_source = prompts

    if cfg is None:
        raise ValueError("TrainingConfig must be provided")

    prompt_list = load_prompts(cfg.dataset, prompt_source or cfg.prompts)
    if not prompt_list:
        raise ValueError("No prompts provided for SFT training")

    tokenizer = prepare_tokenizer(cfg.base_model, cfg.hf_cache_dir)
    dtype = resolve_dtype(cfg.model_dtype)
    model = prepare_lora_model(
        base_model=cfg.base_model,
        lora_cfg=cfg.lora,
        cache_dir=cfg.hf_cache_dir,
        dtype=dtype,
    )
    model.train()

    dataloader = _build_dataloader(prompt_list, tokenizer, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    device = torch.device("cpu")
    model.to(device)

    for _ in range(cfg.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            labels = batch["input_ids"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
            )
            outputs.loss.backward()
            optimizer.step()

    adapter_path = cfg.output_dir / cfg.adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(cfg.output_dir / "tokenizer")
    return adapter_path


def evaluate_adapter(cfg: TrainingConfig, adapter_path: Path, eval_prompts: List[str]) -> List[str]:
    runtime = CpuPeftRuntime(
        base_model_id=cfg.base_model,
        adapter_map={cfg.adapter_name: str(adapter_path)},
    )
    batch = RequestBatch(
        requests=[InferenceRequest(prompt=p, adapter_name=cfg.adapter_name) for p in eval_prompts],
        trace_config=InferenceTraceConfig(max_new_tokens=64),
    )
    return runtime.generate(batch)


def export_adapter_to_infq(cfg: TrainingConfig, adapter_path: Path) -> Path:
    from tools.convert.safetensors_to_infq import main as convert

    adapter_safetensors = adapter_path / "adapter_model.safetensors"
    if not adapter_safetensors.exists():
        raise FileNotFoundError(f"Adapter safetensors not found at {adapter_safetensors}")

    if cfg.base_safetensors and cfg.base_safetensors.exists():
        base_safetensors = cfg.base_safetensors
    else:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                cfg.base_model,
                cache_dir=str(cfg.hf_cache_dir) if cfg.hf_cache_dir else None,
                local_files_only=False,
            )
        )
        safetensors = sorted(snapshot_path.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError(f"No safetensors found in snapshot for {cfg.base_model}")
        base_safetensors = safetensors[0]

    out_dir = cfg.output_dir / "infq"
    cmd_args = [
        "--base",
        str(base_safetensors),
        "--adapter",
        f"{cfg.adapter_name}={adapter_safetensors}",
        "--out",
        str(out_dir),
        "--outlier-threshold",
        "0",
    ]
    convert(cmd_args)
    return out_dir


__all__ = [
    "train_lora_adapter",
    "evaluate_adapter",
    "export_adapter_to_infq",
]
