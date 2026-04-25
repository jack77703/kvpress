# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import gc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from fire import Fire
from transformers import DynamicCache, pipeline

from kvpress import KVzapPrunePress, KVzipPress


def parse_csv_floats(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def build_context(tokenizer, target_tokens: int) -> str:
    paragraph = (
        "This is a synthetic long-context benchmark passage about GPU memory, key value caches, "
        "and long document question answering. The passage repeats so timing and memory can be "
        "measured at a fixed context length. "
    )
    context = paragraph
    while len(tokenizer.encode(context, add_special_tokens=False)) < target_tokens:
        context += paragraph
    return context


def make_press(method: str, compression_ratio: float):
    if method == "no_press":
        return None
    if method == "kvzip":
        return KVzipPress(compression_ratio=compression_ratio)
    if method == "kvzap_prune_mlp":
        return KVzapPrunePress(compression_ratio=compression_ratio, model_type="mlp")
    raise ValueError(f"Unknown method: {method}")


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_gb(value: int) -> float:
    return value / 1024**3


def benchmark_method(
    pipe, method: str, compression_ratio: float, context: str, max_context_length: int, max_new_tokens: int
):
    press = make_press(method, compression_ratio)
    cache = DynamicCache()

    input_tensors = pipe.preprocess(
        context,
        questions=["\nSummarize the benchmark passage in one short sentence."],
        answer_prefix="",
        max_context_length=max_context_length,
        enable_thinking=False,
    )
    context_ids = input_tensors["context_ids"].to(pipe.model.device)
    question_ids = input_tensors["questions_ids"][0].to(pipe.model.device)
    original_context_tokens = context_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_allocated_gb = memory_gb(torch.cuda.memory_allocated())
    else:
        model_allocated_gb = 0.0

    synchronize()
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()

    prefill_context = press(pipe.model) if press is not None else nullcontext()
    with prefill_context:
        pipe.model.model(input_ids=context_ids, past_key_values=cache)

    synchronize()
    prefill_seconds = time.perf_counter() - prefill_start
    compressed_cache_tokens = cache.get_seq_length()

    answer = pipe.generate_answer(
        question_ids=question_ids,
        cache=cache,
        context_length=original_context_tokens,
        max_new_tokens=max_new_tokens,
    )

    synchronize()
    total_seconds = time.perf_counter() - total_start

    if torch.cuda.is_available():
        peak_allocated_gb = memory_gb(torch.cuda.max_memory_allocated())
        peak_reserved_gb = memory_gb(torch.cuda.max_memory_reserved())
    else:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0

    row = {
        "method": method,
        "compression_ratio": compression_ratio,
        "original_context_tokens": original_context_tokens,
        "compressed_cache_tokens": compressed_cache_tokens,
        "actual_compression_ratio": 1 - compressed_cache_tokens / original_context_tokens,
        "max_new_tokens": max_new_tokens,
        "prefill_seconds": prefill_seconds,
        "total_seconds": total_seconds,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "model_allocated_gb": model_allocated_gb,
        "answer_preview": answer[:120].replace("\n", " "),
    }

    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def main(
    model: str = "Qwen/Qwen3-8B",
    methods: str = "no_press,kvzip,kvzap_prune_mlp",
    compression_ratios: str = "0.25,0.50,0.75",
    max_context_length: int = 16384,
    max_new_tokens: int = 64,
    output_csv: str = "/scratch/cc9171/results/kv_microbenchmark_16k.csv",
):
    pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto")
    context = build_context(pipe.tokenizer, max_context_length + 256)
    rows = []
    ratios = parse_csv_floats(compression_ratios)

    for method in methods.split(","):
        method = method.strip()
        if not method:
            continue
        method_ratios = [0.0] if method == "no_press" else ratios
        for ratio in method_ratios:
            print(f"Running {method} with compression_ratio={ratio}", flush=True)
            rows.append(benchmark_method(pipe, method, ratio, context, max_context_length, max_new_tokens))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote benchmark results to {output_path}", flush=True)


if __name__ == "__main__":
    Fire(main)
