from __future__ import annotations

from typing import List, Optional


REFERENCES = {
    "2+2=": "4",
}


def compute_reward(prompts: List[str], completions: List[str], _: Optional[dict] = None) -> List[float]:
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        target = REFERENCES.get(prompt.strip(), "")
        reward = 1.0 if completion.strip().startswith(target) else 0.0
        rewards.append(reward)
    return rewards
