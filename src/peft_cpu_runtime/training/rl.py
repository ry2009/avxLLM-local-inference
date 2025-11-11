from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional

import torch
import torch.nn.functional as F

from .config import RLConfig
from .data import load_prompts
from .utils import chunked, prepare_lora_model, prepare_tokenizer, resolve_dtype, torch_no_grad


@dataclass
class RLStepResult:
    sequences: torch.Tensor
    prompt_lengths: List[int]
    completions: List[str]
    logprob_sums: torch.Tensor
    ref_logprob_sums: torch.Tensor | None


def _generate_batch(model, tokenizer, prompts: List[str], cfg: RLConfig) -> Tuple[torch.Tensor, List[int], List[str]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    with torch_no_grad():
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

    completions = []
    for idx, length in enumerate(prompt_lengths):
        completion_ids = generated[idx, length:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return generated, prompt_lengths, completions


def _sequence_logprobs(model, sequences: torch.Tensor, prompt_lengths: List[int]) -> torch.Tensor:
    outputs = model(input_ids=sequences, attention_mask=(sequences != model.config.pad_token_id).long())
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    target = sequences[:, 1:]
    token_logprobs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    sums = []
    for idx, prompt_len in enumerate(prompt_lengths):
        start = max(prompt_len - 1, 0)
        sums.append(token_logprobs[idx, start:].sum())
    return torch.stack(sums)


def _rl_step(
    model,
    ref_model,
    tokenizer,
    prompts: List[str],
    cfg: RLConfig,
) -> RLStepResult:
    sequences, prompt_lengths, completions = _generate_batch(model, tokenizer, prompts, cfg)
    logprob_sums = _sequence_logprobs(model, sequences, prompt_lengths)
    ref_logprob_sums = None
    if ref_model is not None and cfg.kl_coef > 0:
        with torch.no_grad():
            ref_logprob_sums = _sequence_logprobs(ref_model, sequences, prompt_lengths)
    return RLStepResult(
        sequences=sequences,
        prompt_lengths=prompt_lengths,
        completions=completions,
        logprob_sums=logprob_sums,
        ref_logprob_sums=ref_logprob_sums,
    )


def train_policy_rl(prompts: Union[Iterable[str], RLConfig, None], cfg: Optional[RLConfig] = None) -> Path:
    if isinstance(prompts, RLConfig):
        cfg = prompts
        prompt_source: Iterable[str] | None = None
    else:
        prompt_source = prompts

    if cfg is None:
        raise ValueError("RLConfig must be provided")

    prompt_list = load_prompts(cfg.dataset, list(prompt_source) if prompt_source else None)
    if not prompt_list:
        raise ValueError("No prompts supplied for RL training")

    tokenizer = prepare_tokenizer(cfg.base_model, cfg.hf_cache_dir)
    dtype = resolve_dtype(cfg.model_dtype)
    model = prepare_lora_model(
        base_model=cfg.base_model,
        lora_cfg=cfg.lora,
        cache_dir=cfg.hf_cache_dir,
        dtype=dtype,
    )
    model.train()
    model.config.pad_token_id = tokenizer.pad_token_id

    ref_model = None
    if cfg.kl_coef > 0:
        from transformers import AutoModelForCausalLM

        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            cache_dir=str(cfg.hf_cache_dir) if cfg.hf_cache_dir else None,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        ref_model.eval()
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        for param in ref_model.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    device = torch.device("cpu")
    model.to(device)
    if ref_model:
        ref_model.to(device)

    baseline = 0.0
    step = 0

    for _ in range(cfg.epochs):
        for batch_prompts in chunked(prompt_list, cfg.batch_size):
            result = _rl_step(model, ref_model, tokenizer, batch_prompts, cfg)
            rewards = torch.tensor(
                cfg.reward_fn(batch_prompts, result.completions, None),
                dtype=result.logprob_sums.dtype,
                device=device,
            )
            if rewards.shape[0] != result.logprob_sums.shape[0]:
                raise ValueError("Reward function returned mismatched batch size")

            baseline = cfg.baseline_momentum * baseline + (1.0 - cfg.baseline_momentum) * rewards.float().mean().item()
            advantages = rewards - rewards.new_tensor(baseline)

            policy_loss = -(advantages * result.logprob_sums).mean()

            if cfg.kl_coef > 0 and result.ref_logprob_sums is not None:
                kl_term = result.logprob_sums.detach() - result.ref_logprob_sums
                policy_loss = policy_loss + cfg.kl_coef * kl_term.mean()

            policy_loss.backward()
            step += 1

            if step % cfg.mini_batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

    adapter_path = cfg.output_dir / cfg.adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(cfg.output_dir / "tokenizer")
    return adapter_path


__all__ = ["train_policy_rl"]
