"""Repeated-inference statistics and GPU memory helpers.

Ported from the research repo's ``src/utils/utils.py`` (``get_stats``,
``run_multiinference``) and ``src/utils/memory_tracker.py``. Unchanged except that
the NVML/KV-cache helpers, which generation does not use, are left out.
"""
from __future__ import annotations

from contambench.gen import require_extra

require_extra()

import functools  # noqa: E402
import gc  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def get_stats(all_responses, all_response_lens, train_max_len):
    responses = list(zip(all_responses, all_response_lens))
    df = pd.DataFrame(responses, columns=["response", "response_len"])
    df = df.sort_values(by="response_len")
    max_len = int(df["response_len"].max()) if not df.empty else 0
    min_len = int(df["response_len"].min()) if not df.empty else 0
    longest_response = df.loc[df["response_len"] == max_len, "response"].iloc[0] if not df.empty else ""
    shortest_response = df.loc[df["response_len"] == min_len, "response"].iloc[0] if not df.empty else ""
    mean = float(df["response_len"].mean()) if not df.empty else 0.0
    median = float(df["response_len"].median()) if not df.empty else 0.0
    median_idx = len(df) // 2
    median_response = df["response"].iloc[median_idx] if len(df) > 0 else ""
    mode_series = df["response_len"].mode()
    mode = int(mode_series.iloc[0]) if not mode_series.empty else None
    variance = df["response_len"].var()
    range_len = max_len - min_len
    freq = df["response_len"].value_counts(normalize=True)
    entropy = float(-(freq * np.log2(freq)).sum()) if not freq.empty else 0.0
    max_runaway = int((df["response_len"] > train_max_len).sum())
    return {
        "max_response_len": max_len,
        "min_response_len": min_len,
        "longest_response": longest_response,
        "shortest_response": shortest_response,
        "avg_response_len": mean,
        "median_response_len": median,
        "median_response": median_response,
        "mode": mode,
        "variance": None if pd.isna(variance) else float(variance),
        "range": range_len,
        "entropy": entropy,
        "max_runaway": max_runaway,
        "num_samples": len(df),
    }


def run_multiinference(model, prompt: str, temperature=0.7, batch_size=16, total_runs=32):
    print(f"running multiple inference with temperature [{temperature}] and batchsize [{batch_size}] on prompt {prompt!r}")
    all_responses, all_response_lens, input_lens = [], [], []
    for _ in range(total_runs // batch_size):
        batch_response, input_lens, response_lens = model.batch_infer([prompt] * batch_size, temperature=temperature)
        all_responses.extend(batch_response)
        all_response_lens.extend(response_lens)
    stats = get_stats(all_responses, all_response_lens, model.get_response_limit())
    return stats | {"prompt_len": input_lens[0]}


def memory_usage():
    """GPU memory allocated, in percent of device 0."""
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        return torch.cuda.memory_allocated() / total_mem * 100
    return 0


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def track_memory(func: Callable) -> Callable:
    """Report GPU memory before and after ``func``."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem_before = get_gpu_memory_usage()
        print(f"GPU Memory before {func.__name__}: {mem_before:.2f}MB")
        try:
            return func(*args, **kwargs)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mem_after = get_gpu_memory_usage()
            print(f"GPU Memory after {func.__name__}: {mem_after:.2f}MB (difference {mem_after - mem_before:.2f}MB)")
    return wrapper
