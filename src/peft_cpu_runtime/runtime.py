from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


@dataclass
class InferenceTraceConfig:
    """Shared generation parameters used across batched requests."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
    do_sample: bool = False
    stop_sequences: Optional[List[str]] = None

    def to_generation_config(self) -> GenerationConfig:
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        if self.stop_sequences:
            gen_kwargs["eos_token_id"] = None  # explicit stop sequences managed manually
        return GenerationConfig(**gen_kwargs)


@dataclass
class InferenceRequest:
    """Single inference payload routed to a specific adapter."""

    prompt: str
    adapter_name: Optional[str] = None
    max_new_tokens: Optional[int] = None


@dataclass
class RequestBatch:
    """Collection of inference requests sharing a trace configuration."""

    requests: List[InferenceRequest]
    trace_config: InferenceTraceConfig = field(default_factory=InferenceTraceConfig)

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("RequestBatch must contain at least one request.")


class CpuPeftRuntime:
    """
    Minimal CPU-first runtime that supports multi-adapter LoRA serving on top of a shared
    base model. Designed to mimic the scheduling and adapter hot-swapping patterns of
    GPU runtimes while enabling early experimentation on CPU-only hardware.
    """

    def __init__(
        self,
        base_model_id: str,
        adapter_map: Dict[str, str],
        torch_dtype: torch.dtype = torch.float32,
        use_fast_tokenizer: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        base_model_id:
            Hugging Face model identifier for the base causal LM.
        adapter_map:
            Mapping from logical adapter names to Hugging Face adapter repositories or
            local paths.
        torch_dtype:
            Desired dtype for the base model. float32 is safest on CPU; bf16 requires AVX512.
        use_fast_tokenizer:
            Whether to use the fast tokenizer implementations where available.
        """
        self.base_model_id = base_model_id
        self.adapter_map = adapter_map
        self.torch_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            use_fast=use_fast_tokenizer,
        )
        if self.tokenizer.pad_token_id is None:
            # Ensure padding token exists for batch inference
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
        )
        model.eval()

        if adapter_map:
            logical_name, adapter_id = next(iter(adapter_map.items()))
            peft_model = PeftModel.from_pretrained(
                model,
                adapter_id,
                adapter_name=logical_name,
                is_trainable=False,
            )
            self.available_adapters = {logical_name}
            for name, adapter in list(adapter_map.items())[1:]:
                peft_model.load_adapter(adapter, adapter_name=name, is_trainable=False)
                self.available_adapters.add(name)
            self.model: torch.nn.Module = peft_model
        else:
            self.available_adapters = set()
            self.model = model
        self._profiling_enabled = False
        self._last_profile: Optional[List[Dict[str, float]]] = None
        self._overlap_enabled = False
        self._overlap_workers = 2

    @property
    def is_peft_model(self) -> bool:
        return isinstance(self.model, PeftModel)

    def _activate_adapter(self, adapter_name: Optional[str]) -> None:
        if not self.is_peft_model:
            return
        peft_model: PeftModel = self.model  # type: ignore[assignment]
        if adapter_name is None:
            peft_model.disable_adapter()
            return
        if adapter_name not in self.available_adapters:
            raise KeyError(f"Adapter '{adapter_name}' not loaded. Available: {sorted(self.available_adapters)}")
        peft_model.set_adapter(adapter_name)

    def generate(self, batch: RequestBatch) -> List[str]:
        """Generate completions for a batch of requests."""
        outputs: List[str] = []
        grouped: Dict[Optional[str], List[InferenceRequest]] = defaultdict(list)
        for req in batch.requests:
            grouped[req.adapter_name].append(req)

        generation_config = batch.trace_config.to_generation_config()
        profile_groups: Optional[List[Dict[str, float]]] = [] if self._profiling_enabled else None
        group_items = list(grouped.items())

        executor: Optional[ThreadPoolExecutor] = None
        token_futures: List = []
        if self._overlap_enabled and len(group_items) > 1:
            executor = ThreadPoolExecutor(max_workers=self._overlap_workers)
            for _, requests in group_items:
                token_futures.append(executor.submit(self._tokenize_requests, requests))

        try:
            for idx, (adapter_name, requests) in enumerate(group_items):
                self._activate_adapter(adapter_name)
                if executor:
                    inputs, tokenize_elapsed = token_futures[idx].result()
                else:
                    inputs, tokenize_elapsed = self._tokenize_requests(requests)
                max_new_tokens_override = [
                    req.max_new_tokens or batch.trace_config.max_new_tokens for req in requests
                ]
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                input_lengths = attention_mask.sum(dim=1)
                device = torch.device("cpu")
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                adapter_outputs = []
                generation_elapsed = 0.0
                group_start = time.perf_counter() if profile_groups is not None else None
                ttft = None
                for row_idx, request in enumerate(requests):
                    req_config = GenerationConfig.from_dict(generation_config.to_dict())
                    req_config.max_new_tokens = max_new_tokens_override[row_idx]
                    with torch.no_grad():
                        gen_start = time.perf_counter() if profile_groups is not None else None
                        generated = self.model.generate(
                            input_ids=input_ids[row_idx : row_idx + 1],
                            attention_mask=attention_mask[row_idx : row_idx + 1],
                            generation_config=req_config,
                        )
                        if gen_start is not None:
                            gen_end = time.perf_counter()
                            generation_elapsed += gen_end - gen_start
                            if ttft is None and group_start is not None:
                                ttft = gen_end - group_start
                    generated_tokens = generated[0][input_lengths[row_idx] :]
                    decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    if batch.trace_config.stop_sequences:
                        decoded = self._truncate_on_stop(decoded, batch.trace_config.stop_sequences)
                    adapter_outputs.append(decoded)
                outputs.extend(adapter_outputs)
                if profile_groups is not None:
                    profile_groups.append(
                        {
                            "adapter": adapter_name or "<base>",
                            "tokenize_s": tokenize_elapsed or 0.0,
                            "generate_s": generation_elapsed,
                            "num_requests": len(requests),
                            "ttft_s": ttft or 0.0,
                        }
                    )
        finally:
            if executor:
                executor.shutdown(wait=True)
        if profile_groups is not None:
            self._last_profile = profile_groups
        else:
            self._last_profile = None
        return outputs

    @staticmethod
    def _truncate_on_stop(text: str, stop_sequences: Iterable[str]) -> str:
        truncated = text
        for stop in stop_sequences:
            idx = truncated.find(stop)
            if idx != -1:
                truncated = truncated[:idx]
        return truncated

    def benchmark(
        self,
        batch: RequestBatch,
        num_warmup: int = 1,
        num_iters: int = 3,
    ) -> Dict[str, object]:
        """
        Run repeated generations to estimate throughput. Returns aggregate metrics.
        """
        if num_iters <= 0:
            raise ValueError("num_iters must be > 0")

        for _ in range(num_warmup):
            _ = self.generate(batch)

        total_time = 0.0
        total_new_tokens = 0
        total_sequences = 0
        total_ttft = 0.0
        ttft_samples = 0
        iter_stats = []

        for _ in range(num_iters):
            start = time.perf_counter()
            completions = self.generate(batch)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_sequences += len(completions)
            iter_tokens = 0
            for completion in completions:
                iter_tokens += len(self.tokenizer.encode(completion, add_special_tokens=False))
            total_new_tokens += iter_tokens
            iter_entry = {
                "latency_s": elapsed,
                "sequences": len(completions),
                "new_tokens": iter_tokens,
                "tokens_per_second": iter_tokens / elapsed if elapsed else 0.0,
                "seq_per_second": len(completions) / elapsed if elapsed else 0.0,
            }
            if self._profiling_enabled and self._last_profile is not None:
                iter_entry["profiling"] = self._last_profile
                ttft_values = [grp.get("ttft_s", 0.0) for grp in self._last_profile if grp.get("ttft_s", 0.0) > 0]
                if ttft_values:
                    ttft_sample = min(ttft_values)
                    iter_entry["ttft_s"] = ttft_sample
                    total_ttft += ttft_sample
                    ttft_samples += 1
            iter_stats.append(iter_entry)

        return {
            "avg_latency_s": total_time / num_iters,
            "seq_per_second": total_sequences / total_time if total_time else 0.0,
            "tokens_per_second": total_new_tokens / total_time if total_time else 0.0,
            "total_new_tokens": total_new_tokens,
            "avg_ttft_s": (total_ttft / ttft_samples) if ttft_samples else None,
            "iterations": iter_stats,
        }

    def enable_profiling(self, enabled: bool = True) -> None:
        self._profiling_enabled = enabled
        if not enabled:
            self._last_profile = None

    def get_last_profile(self) -> Optional[List[Dict[str, float]]]:
        return self._last_profile

    def enable_tokenize_overlap(self, enabled: bool = True, max_workers: int = 2) -> None:
        self._overlap_enabled = enabled
        self._overlap_workers = max(1, max_workers)

    def _tokenize_requests(self, requests: List[InferenceRequest]):
        start = time.perf_counter()
        prompts = [req.prompt for req in requests]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        elapsed = time.perf_counter() - start
        return inputs, elapsed
