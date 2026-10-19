"""Experiment configuration for contamination-aware stress testing.

A configuration describes one *experiment*, which may sweep several
experimental factors:

* ``contamination_ratio`` (c)   - proportion of adversarial requests
* ``num_clients`` (N)           - number of concurrent clients
* ``arrival_rate`` (lambda)     - requests/second *per client*

Any of those three may be given as a single scalar or as a list; the runner
takes the Cartesian product and repeats every combination ``num_trials`` times.

The endpoint defaults to a RunPod serverless OpenAI-compatible vLLM endpoint
(resolved from ``RUNPOD_ENDPOINT_ID``), as required by the study.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is a hard dependency for YAML files
    yaml = None  # noqa: N816

# Contamination ratios mandated by the study design.
DEFAULT_CONTAMINATION_RATIOS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

# SLO latency budgets (ms) evaluated as service-quality regimes, not a single
# ground-truth threshold.
DEFAULT_LATENCY_SLOS_MS = [2000, 5000, 10000]
DEFAULT_TTFT_SLOS_MS = [500, 1000, 2000]
DEFAULT_TBT_SLOS_MS = [50, 100, 200]

VALID_ARRIVAL_MODELS = ("fixed", "poisson", "bursty")
VALID_BACKENDS = ("auto", "openai", "runpod")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@dataclass
class ExperimentConfig:
    """Full specification of a (possibly multi-factor) experiment."""

    name: str = "contamination_study"

    # --- endpoint / model -------------------------------------------------
    endpoint: Optional[str] = None
    backend: str = "runpod"
    runpod_endpoint_id: Optional[str] = None
    model: Optional[str] = None  # request-body model; omit on RunPod
    attack_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    attack_file: Optional[str] = None
    workload_size: Optional[int] = None  # top-n prompts by success_rate; None = all
    api_key: Optional[str] = None

    # --- experimental factors (scalar or list) ---------------------------
    contamination_ratio: Any = field(
        default_factory=lambda: list(DEFAULT_CONTAMINATION_RATIOS)
    )
    num_clients: Any = 1
    arrival_rate: Any = 1.0  # requests/second per client
    arrival_model: str = "poisson"

    # --- per-run sizing ---------------------------------------------------
    requests_per_client: int = 20
    num_trials: int = 1

    # --- generation params ------------------------------------------------
    max_tokens: Any = 256  # scalar or list — swept like other factors when a list
    temperature: Optional[float] = 0.0
    timeout_s: float = 600.0
    # Pause after each factor combination so the endpoint can drain queues /
    # finish stragglers before the next c/N/lambda/trial run starts.
    config_cooldown_s: float = 0.0

    # --- SLOs -------------------------------------------------------------
    # When auto_slo is true, fixed budgets below are ignored at analysis time;
    # budgets = benign c=0 p99 at the selected workload × 1.0 (latency, TTFT, TBT).
    auto_slo: bool = False
    latency_slo_ms: Any = field(default_factory=lambda: list(DEFAULT_LATENCY_SLOS_MS))
    ttft_slo_ms: Any = field(default_factory=lambda: list(DEFAULT_TTFT_SLOS_MS))
    tbt_slo_ms: Any = field(default_factory=lambda: list(DEFAULT_TBT_SLOS_MS))

    # --- arrival shaping (bursty model) ----------------------------------
    burst_size: int = 5
    burst_factor: float = 10.0  # within-burst rate = arrival_rate * burst_factor

    # --- prompt salting (lower vLLM prefix-cache hits) -------------------
    # With a small prompt pool, identical prompts give a high prefix-cache hit
    # rate. Salting prepends a short per-request random id so inputs vary like
    # real traffic. Applied to benign and adversarial prompts alike.
    prompt_salt: bool = False
    salt_length: int = 8  # chars of random id prepended when prompt_salt is on
    # 0 = unique id per request (~0% prefix reuse). >0 draws from a fixed pool of
    # that many ids so prefixes repeat: smaller pool → higher cache-hit rate.
    salt_pool_size: int = 0

    # --- reproducibility / storage ---------------------------------------
    seed: int = 23
    results_dir: Optional[str] = None  # default: ./contambench-results
    response_text_max_chars: Optional[int] = 2000  # null = store full response text
    max_workers: Optional[int] = None  # thread-pool size; default derives from N

    # --- offline simulation (no network) ---------------------------------
    simulate: bool = False

    # ---------------------------------------------------------------------
    # factor accessors
    # ---------------------------------------------------------------------
    @property
    def contamination_ratios(self) -> list[float]:
        return [float(c) for c in _as_list(self.contamination_ratio)]

    @property
    def client_counts(self) -> list[int]:
        return [int(n) for n in _as_list(self.num_clients)]

    @property
    def arrival_rates(self) -> list[float]:
        return [float(r) for r in _as_list(self.arrival_rate)]

    @property
    def latency_slos(self) -> list[float]:
        return [float(x) for x in _as_list(self.latency_slo_ms)]

    @property
    def ttft_slos(self) -> list[float]:
        return [float(x) for x in _as_list(self.ttft_slo_ms)]

    @property
    def tbt_slos(self) -> list[float]:
        return [float(x) for x in _as_list(self.tbt_slo_ms)]

    @property
    def max_tokens_values(self) -> list[Optional[int]]:
        raw = _as_list(self.max_tokens)
        if not raw:
            return [256]
        out: list[Optional[int]] = []
        for v in raw:
            out.append(None if v is None else int(v))
        return out

    # ---------------------------------------------------------------------
    def validate(self) -> None:
        if self.arrival_model not in VALID_ARRIVAL_MODELS:
            raise ValueError(
                f"arrival_model must be one of {VALID_ARRIVAL_MODELS}, got {self.arrival_model!r}"
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {self.backend!r}")
        for c in self.contamination_ratios:
            if not 0.0 <= c <= 1.0:
                raise ValueError(f"contamination_ratio out of [0,1]: {c}")
        if any(n <= 0 for n in self.client_counts):
            raise ValueError("num_clients must be positive")
        if any(r <= 0 for r in self.arrival_rates):
            raise ValueError("arrival_rate must be positive")
        if self.requests_per_client <= 0:
            raise ValueError("requests_per_client must be positive")
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if self.workload_size is not None and self.workload_size <= 0:
            raise ValueError("workload_size must be positive when set")
        if self.config_cooldown_s < 0:
            raise ValueError("config_cooldown_s must be non-negative")
        for mt in self.max_tokens_values:
            if mt is not None and mt <= 0:
                raise ValueError(f"max_tokens must be positive when set, got {mt}")

    # ---------------------------------------------------------------------
    def resolved_endpoint(self) -> str:
        """Resolve the OpenAI-compatible base URL (RunPod by default)."""
        from contambench.harness.client import default_endpoint, resolve_runpod_base

        endpoint_id = self.runpod_endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
        if self.endpoint:
            return self.endpoint
        if endpoint_id:
            return resolve_runpod_base(endpoint_id)
        return default_endpoint()

    def to_metadata(self) -> dict:
        data = asdict(self)
        # Normalise the sweep factors into explicit lists for provenance.
        data["contamination_ratios"] = self.contamination_ratios
        data["client_counts"] = self.client_counts
        data["arrival_rates"] = self.arrival_rates
        data["latency_slos"] = self.latency_slos
        data["ttft_slos"] = self.ttft_slos
        data["tbt_slos"] = self.tbt_slos
        data["max_tokens_values"] = self.max_tokens_values
        return data


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs (uv sync, or uv pip install pyyaml)")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> ExperimentConfig:
    known = {f for f in ExperimentConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = ExperimentConfig(**raw)
    cfg.validate()
    return cfg
