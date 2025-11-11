"""Built-in reward helpers for lightweight RL on CPU."""

from __future__ import annotations

from typing import Callable, List, Optional

RewardFn = Callable[[List[str], List[str], Optional[dict]], List[float]]


def exact_match(prompts: List[str], completions: List[str], _: Optional[dict] = None) -> List[float]:
    """Returns 1.0 when the completion exactly matches the prompt suffix."""

    scores: List[float] = []
    for prompt, completion in zip(prompts, completions):
        scores.append(1.0 if completion.strip() == prompt.strip() else 0.0)
    return scores


def length_reward(prompts: List[str], completions: List[str], _: Optional[dict] = None) -> List[float]:
    """Encourages shorter completions by rewarding inverse token length."""

    scores: List[float] = []
    for completion in completions:
        length = max(len(completion.strip().split()), 1)
        scores.append(1.0 / length)
    return scores


REGISTRY: dict[str, RewardFn] = {
    "exact_match": exact_match,
    "length": length_reward,
}


def get_reward(name: str) -> RewardFn:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown built-in reward '{name}'. Available: {sorted(REGISTRY)}") from exc


__all__ = ["RewardFn", "exact_match", "length_reward", "REGISTRY", "get_reward"]
