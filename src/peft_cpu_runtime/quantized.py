from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .runtime import InferenceRequest, InferenceTraceConfig, RequestBatch

try:
    from llama_cpp import Llama, llama_cpp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("llama-cpp-python is required for quantized GGUF inference.") from exc


@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    seed: int = 0
    verbose: bool = False
    lora_scale: float = 1.0


class LlamaCppPeftRuntime:
    """
    Quantized runtime backed by llama.cpp (GGUF) with multi-adapter LoRA support.
    """

    def __init__(self, config: LlamaCppConfig, adapter_map: Dict[str, str]) -> None:
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"model_path does not exist: {config.model_path}")

        self.config = config
        self._llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            seed=config.seed,
            lora_scale=config.lora_scale,
            logits_all=False,
            verbose=config.verbose,
        )
        self._adapter_handles: Dict[str, llama_cpp.llama_adapter_lora_p] = {}
        self._load_adapters(adapter_map)

    def _load_adapters(self, adapter_map: Dict[str, str]) -> None:
        for name, path in adapter_map.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Adapter path '{path}' not found.")
            handle = llama_cpp.llama_adapter_lora_init(self._llm.model, path.encode("utf-8"))
            if handle is None:
                raise RuntimeError(f"Failed to initialize LoRA adapter '{name}' from {path}.")
            self._adapter_handles[name] = handle

    def _activate_adapter(self, adapter_name: Optional[str]) -> None:
        if adapter_name is None:
            llama_cpp.llama_clear_adapter_lora(self._llm.ctx)
            return
        handle = self._adapter_handles.get(adapter_name)
        if handle is None:
            raise KeyError(f"Adapter '{adapter_name}' is not loaded. Available: {sorted(self._adapter_handles)}")
        llama_cpp.llama_set_adapter_lora(self._llm.ctx, handle, self.config.lora_scale)

    def generate(self, batch: RequestBatch) -> List[str]:
        outputs: List[str] = []
        grouped = defaultdict(list)
        for req in batch.requests:
            grouped[req.adapter_name].append(req)

        for adapter_name, requests in grouped.items():
            self._activate_adapter(adapter_name)
            for request in requests:
                trace = batch.trace_config
                max_tokens = request.max_new_tokens or trace.max_new_tokens
                temperature = trace.temperature if trace.do_sample else 0.0
                top_p = trace.top_p if trace.do_sample else 1.0
                stop = trace.stop_sequences
                result = self._llm.create_completion(
                    prompt=request.prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    stream=False,
                )
                outputs.append(result["choices"][0]["text"])
        return outputs

    def benchmark(
        self,
        batch: RequestBatch,
        num_warmup: int = 1,
        num_iters: int = 3,
    ) -> Dict[str, object]:
        if num_iters <= 0:
            raise ValueError("num_iters must be > 0")

        for _ in range(num_warmup):
            _ = self.generate(batch)

        total_time = 0.0
        total_new_tokens = 0
        total_sequences = 0
        iter_stats = []

        for _ in range(num_iters):
            start = time.perf_counter()
            completions = self.generate(batch)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_sequences += len(completions)
            iter_tokens = 0
            for completion in completions:
                token_ids = self._llm.tokenize(completion.encode("utf-8"), add_bos=False)
                iter_tokens += len(token_ids)
            total_new_tokens += iter_tokens
            iter_stats.append(
                {
                    "latency_s": elapsed,
                    "sequences": len(completions),
                    "new_tokens": iter_tokens,
                    "tokens_per_second": iter_tokens / elapsed if elapsed else 0.0,
                    "seq_per_second": len(completions) / elapsed if elapsed else 0.0,
                }
            )

        return {
            "avg_latency_s": total_time / num_iters,
            "seq_per_second": total_sequences / total_time if total_time else 0.0,
            "tokens_per_second": total_new_tokens / total_time if total_time else 0.0,
            "total_new_tokens": total_new_tokens,
            "iterations": iter_stats,
        }

    def unload(self) -> None:
        for handle in self._adapter_handles.values():
            llama_cpp.llama_clear_adapter_lora(self._llm.ctx)
            llama_cpp.llama_adapter_lora_free(handle)
        self._adapter_handles.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.unload()
        except Exception:
            pass
