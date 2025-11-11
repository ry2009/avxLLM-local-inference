from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from .config import PretrainConfig
from .data import build_hf_dataset, load_prompts
from .utils import prepare_tokenizer, resolve_dtype


def _build_loader(prompts, tokenizer, cfg: PretrainConfig) -> DataLoader:
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


def train_causal_lm(cfg: PretrainConfig) -> Path:
    prompts = load_prompts(cfg.dataset, None)
    if not prompts:
        raise ValueError("Dataset yielded no samples for pretraining")

    tokenizer = prepare_tokenizer(cfg.base_model)
    dtype = resolve_dtype(cfg.model_dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            use_safetensors=True,
            torch_dtype=dtype,
        )
    except (OSError, ValueError) as exc:
        message = str(exc)
        if "safetensor" in message.lower() or "safetensors" in message.lower():
            raise RuntimeError(
                f"{cfg.base_model} is missing safetensors weights required for CPU pretraining "
                "on this environment. Choose a safetensors checkpoint or upgrade PyTorch to >=2.6."
            ) from exc
        raise

    dataloader = _build_loader(prompts, tokenizer, cfg)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    for _ in range(cfg.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir / "tokenizer")
    return cfg.output_dir


__all__ = ["train_causal_lm"]
