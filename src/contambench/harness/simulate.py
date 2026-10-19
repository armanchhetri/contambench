"""Offline simulation client.

``MockChatClient`` mimics :class:`client.ChatClient` but fabricates plausible
latency/TTFT numbers instead of issuing HTTP requests. It is used for
``--simulate`` runs so the whole experiment + analysis pipeline can be exercised
(and regression-tested) without a live endpoint.

No model is served, so **nothing produced by a simulate run is a measurement**: the
numbers are drawn from the synthetic model below, and must never be reported as
serving results. Measuring contamination effects requires a real endpoint.

The synthetic model deliberately encodes the two effects the study targets:

* a **direct effect** - adversarial requests are intrinsically slower (longer
  generations), and
* an **indirect / noisy-neighbour effect** - benign requests slow down as the
  global contamination ratio and concurrency rise, because they share queueing
  resources with the heavier adversarial traffic.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from contambench.harness.client import RequestResult


class MockChatClient:
    """Drop-in stand-in for :class:`client.ChatClient` (synthetic latencies)."""

    def __init__(
        self,
        model: Optional[str] = "mock-model",
        max_tokens: Optional[int] = 256,
        temperature: Optional[float] = 0.0,
        contamination_ratio: float = 0.0,
        num_clients: int = 1,
        arrival_rate: float = 1.0,
        seed: int = 23,
        sleep_scale: float = 0.0,
        backend: str = "openai",
        endpoint: str = "mock://local",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.contamination_ratio = contamination_ratio
        self.num_clients = num_clients
        self.arrival_rate = arrival_rate
        self.sleep_scale = sleep_scale
        self.backend = backend
        self.endpoint = endpoint
        self.url = f"{endpoint}/v1/chat/completions"
        self._rng = random.Random(seed)

    @property
    def sends_model_field(self) -> bool:
        return bool(self.model)

    def _load_factor(self) -> float:
        # Concurrency raises baseline contention; contamination amplifies it.
        c = self.contamination_ratio
        n = self.num_clients
        return 1.0 + 0.15 * (n - 1) + 2.5 * c * (1.0 + 0.3 * (n - 1))

    def stream_chat(
        self,
        content: str,
        index: int = 0,
        is_attack: bool = False,
        pair_index: int = 0,
        preview_chars: int = 200,
    ) -> RequestResult:
        rng = self._rng
        load = self._load_factor()

        # Base service characteristics.
        base_ttft = 0.05 * rng.uniform(0.8, 1.2)
        # Benign requests feel only the shared-resource (indirect) interference.
        # Adversarial requests additionally pay a direct generation penalty.
        ttft = base_ttft * load
        if is_attack:
            out_tokens = int(rng.uniform(0.7, 1.0) * (self.max_tokens or 256) * 4)
            per_tok = 0.010 * load
        else:
            out_tokens = int(rng.uniform(0.3, 0.7) * (self.max_tokens or 256))
            per_tok = 0.009 * load
        out_tokens = max(8, out_tokens)
        decode = out_tokens * per_tok * rng.uniform(0.9, 1.1)
        total = ttft + decode

        # Rare failures, slightly more likely under heavy load.
        fail_p = 0.005 * load
        success = rng.random() > fail_p

        if self.sleep_scale > 0:
            time.sleep(total * self.sleep_scale)

        result = RequestResult(
            index=index,
            is_attack=is_attack,
            pair_index=pair_index,
            success=success,
            model=self.model,
            http_status=200 if success else 500,
            prompt_preview=content[:preview_chars],
        )
        if not success:
            result.error = "RuntimeError: simulated failure"
            result.total_s = total
            return result

        result.ttft_s = ttft
        result.total_s = total
        result.decode_s = decode
        result.itl_mean_s = per_tok * rng.uniform(0.9, 1.1)
        result.output_tokens = out_tokens
        result.usage_completion_tokens = out_tokens
        result.usage_prompt_tokens = max(8, len(content) // 4)
        result.output_chars = out_tokens * 4
        result.finish_reason = "length" if is_attack else "stop"
        result.generated_text = ("x" * (out_tokens * 4))[:1000]
        result.generated_preview = result.generated_text[:preview_chars]
        return result
