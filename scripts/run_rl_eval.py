#!/usr/bin/env python3
"""Evaluate RL adapter vs base using reward functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.rewards import get_reward


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reward scores base vs adapter")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompts", default="data/math_prompts.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--reward", default="exact_match")
    parser.add_argument("--out", default="reports/rl_eval.json")
    return parser.parse_args(argv)


def _load_prompts(path: Path, limit: int = 32) -> List[str]:
    prompts: List[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            prompts.append(record.get("prompt") or record.get("text") or line)
        except json.JSONDecodeError:
            prompts.append(line)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise SystemExit("Prompt file empty")
    return prompts


def _run(runtime: CpuPeftRuntime, prompts: List[str], adapter: Optional[str], max_tokens: int) -> List[str]:
    batch = RequestBatch(
        requests=[InferenceRequest(prompt=p, adapter_name=adapter) for p in prompts],
        trace_config=InferenceTraceConfig(max_new_tokens=max_tokens),
    )
    return runtime.generate(batch)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    reward_fn = get_reward(args.reward)
    prompts = _load_prompts(Path(args.prompts))

    runtime = CpuPeftRuntime(base_model_id=args.model_id, adapter_map={args.adapter: args.adapter})
    base_outputs = _run(runtime, prompts, None, args.max_new_tokens)
    adapter_outputs = _run(runtime, prompts, args.adapter, args.max_new_tokens)

    base_scores = reward_fn(prompts, base_outputs, None)
    adapter_scores = reward_fn(prompts, adapter_outputs, None)
    payload = {
        "prompts": prompts,
        "base_scores": base_scores,
        "adapter_scores": adapter_scores,
        "delta": sum(adapter_scores) - sum(base_scores),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
