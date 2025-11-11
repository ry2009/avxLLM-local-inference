#!/usr/bin/env python3
import argparse
import json
import math
import random
from itertools import cycle
from typing import Dict, List, Optional, Sequence

from peft_cpu_runtime import (
    CpuPeftRuntime,
    InferenceRequest,
    InferenceTraceConfig,
    LlamaCppConfig,
    LlamaCppPeftRuntime,
    RequestBatch,
)

DEFAULT_PROMPTS = [
    "Summarize the latest quarterly earnings for a fictional analytics company in three bullet points.",
    "Draft a polite email requesting feedback on an internal research memo about LoRA adapter serving.",
    "Explain in 100 words why overlapping CPU and GPU workloads can improve LoRA inference throughput.",
]


def parse_adapter_map(entries: Optional[Sequence[str]]) -> Dict[str, str]:
    adapter_map: Dict[str, str] = {}
    if not entries:
        return adapter_map
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Adapter entry '{entry}' must be of the form name=repo_or_path.")
        name, repo = entry.split("=", maxsplit=1)
        adapter_map[name.strip()] = repo.strip()
    return adapter_map


def sample_zipf(names: Sequence[Optional[str]], num_samples: int, alpha: float, rng: random.Random) -> List[Optional[str]]:
    if not names:
        return [None] * num_samples
    if len(names) == 1 or alpha <= 0:
        return [names[idx % len(names)] for idx in range(num_samples)]
    weights = [1.0 / (math.pow(rank + 1, alpha)) for rank in range(len(names))]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    indices = rng.choices(range(len(names)), weights=probabilities, k=num_samples)
    return [names[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU LoRA serving throughput.")
    parser.add_argument("--engine", default="torch", choices=("torch", "llama_cpp"), help="Runtime backend.")
    parser.add_argument(
        "--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF repo id or local path."
    )
    parser.add_argument(
        "--adapter",
        action="append",
        help="Adapter specification as name=repo_or_path. Repeat for multiple adapters.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-requests", type=int, default=None, help="Total requests per batch.")
    parser.add_argument("--zipf-alpha", type=float, default=1.2, help="Zipf skew for adapter sampling; <=0 disables.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for reproducible sampling.")
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include the base model (no adapter) in the sampling pool.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture tokenizer vs. generation timings for each adapter group.",
    )
    parser.add_argument(
        "--tokenize-overlap-workers",
        type=int,
        default=0,
        help="Enable tokenizer overlap with the given worker count (torch engine only).",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window for llama.cpp runtime.")
    parser.add_argument("--n-threads", type=int, default=None, help="Thread pool size for llama.cpp runtime.")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scaling factor for llama.cpp runtime.")
    parser.add_argument(
        "--llama-verbose",
        action="store_true",
        help="Enable verbose logging from llama.cpp runtime.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--print-completions", action="store_true")
    parser.add_argument(
        "--metrics-out",
        type=str,
        help="Optional path to write benchmark metrics JSON.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Custom prompt to include in the batch. Repeat to add multiple prompts.",
    )
    args = parser.parse_args()

    adapter_map = parse_adapter_map(args.adapter)

    if args.engine == "torch":
        runtime = CpuPeftRuntime(
            base_model_id=args.base_model,
            adapter_map=adapter_map,
        )
        if args.profile:
            runtime.enable_profiling(True)
        if args.tokenize_overlap_workers > 0:
            runtime.enable_tokenize_overlap(True, args.tokenize_overlap_workers)
    else:
        config = LlamaCppConfig(
            model_path=args.base_model,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            seed=args.seed,
            verbose=args.llama_verbose,
            lora_scale=args.lora_scale,
        )
        runtime = LlamaCppPeftRuntime(
            config=config,
            adapter_map=adapter_map,
        )
        if args.profile:
            print("Profiling is not supported for llama.cpp runtime; ignoring --profile flag.")

    prompts = args.prompt or DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("At least one prompt is required.")
    num_requests = args.num_requests or len(prompts)
    if num_requests <= 0:
        raise ValueError("num_requests must be > 0.")

    rng = random.Random(args.seed)

    adapter_names: List[Optional[str]] = list(adapter_map.keys())
    if args.include_base or not adapter_names:
        adapter_sampling_pool: List[Optional[str]] = adapter_names + [None]
    else:
        adapter_sampling_pool = adapter_names

    adapter_assignments = sample_zipf(adapter_sampling_pool, num_requests, args.zipf_alpha, rng)

    prompt_cycle = cycle(prompts)
    requests = []
    for idx in range(num_requests):
        requests.append(
            InferenceRequest(
                prompt=next(prompt_cycle),
                adapter_name=adapter_assignments[idx],
                max_new_tokens=args.max_new_tokens,
            )
        )

    batch = RequestBatch(
        requests=requests,
        trace_config=InferenceTraceConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        ),
    )

    metrics = runtime.benchmark(
        batch=batch,
        num_warmup=args.warmup,
        num_iters=args.iters,
    )

    print(json.dumps(metrics, indent=2))
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    if args.print_completions:
        outputs = runtime.generate(batch)
        for idx, output in enumerate(outputs):
            adapter_name = requests[idx].adapter_name or "<base>"
            print(f"\n--- Completion {idx} (adapter={adapter_name}) ---\n{output.strip()}\n")


if __name__ == "__main__":
    main()
