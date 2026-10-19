"""Concurrent, open-loop multi-client workload scheduler.

Launches ``num_clients`` clients in parallel. Each client issues
``requests_per_client`` requests whose dispatch times follow the configured
arrival model (fixed / Poisson / bursty) at the per-client ``arrival_rate``.
Dispatch is *open-loop*: a slow in-flight request never delays the next
scheduled arrival, which is essential for studying queueing under load.

For every request a client independently chooses a benign prompt with
probability ``1 - c`` and an adversarial prompt with probability ``c`` (global
contamination; no permanently benign/malicious clients).
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional

from contambench.harness.arrival import build_offsets
from contambench.harness.dataset import PromptPair, make_salt_pool, select_prompt

# A client object only needs a ``stream_chat`` method matching ChatClient.
ClientFactory = Callable[[int], object]


class WorkloadInterrupted(Exception):
    """Raised when the user interrupts a workload; carries partial results."""

    def __init__(self, result: "WorkloadResult"):
        self.result = result
        super().__init__("workload interrupted by user")


@dataclass
class RunSpec:
    """One concrete factor combination (a single raw run)."""

    contamination_ratio: float
    num_clients: int
    arrival_rate: float
    arrival_model: str
    requests_per_client: int
    trial: int = 0
    burst_size: int = 5
    burst_factor: float = 10.0
    seed: int = 23
    response_text_max_chars: Optional[int] = 2000
    max_workers: Optional[int] = None
    prompt_salt_chars: int = 0  # >0 prepends a per-request id to lower prefix-cache hits
    salt_pool_size: int = 0  # >0 draws salts from a fixed pool (partial, tunable cache reuse)
    max_tokens: Optional[int] = None

    @property
    def total_requests(self) -> int:
        return self.num_clients * self.requests_per_client

    def config_id(self) -> str:
        mt = f"_mt{self.max_tokens}" if self.max_tokens is not None else ""
        return (
            f"c{self.contamination_ratio:g}"
            f"_N{self.num_clients}"
            f"_lam{self.arrival_rate:g}"
            f"_{self.arrival_model}"
            f"_t{self.trial}"
            f"{mt}"
        )


@dataclass
class RequestRecord:
    request_id: str
    client_id: int
    trial: int
    config_id: str
    timestamp_start: float
    timestamp_first_token: Optional[float]
    timestamp_end: float
    prompt_type: str  # "benign" | "adversarial"
    contamination_ratio: float
    arrival_rate: float
    arrival_model: str
    num_clients: int
    success: bool
    http_status: Optional[int]
    input_tokens: Optional[int]
    output_tokens: int
    ttft_ms: Optional[float]
    tbt_ms: Optional[float]  # mean inter-token latency (time-between-tokens), ms
    latency_ms: Optional[float]
    finish_reason: Optional[str]
    error: Optional[str]
    pair_index: int
    prompt_id: int  # index in attack summary.json / res_<N>.json
    model: Optional[str]
    input_text: str
    output_text: str
    decode_residency_ms: Optional[float]  # last_token - first_token (decode phase only)
    max_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkloadResult:
    spec: RunSpec
    records: List[RequestRecord]
    wall_start: float
    wall_end: float

    @property
    def duration_s(self) -> float:
        return max(1e-9, self.wall_end - self.wall_start)

    @property
    def achieved_rate(self) -> float:
        return len(self.records) / self.duration_s

    @property
    def target_rate(self) -> float:
        return self.spec.arrival_rate * self.spec.num_clients


def _default_max_workers(spec: RunSpec) -> int:
    if spec.max_workers is not None:
        return spec.max_workers
    # Open-loop can queue many tasks; cap workers so high N does not spawn
    # tens of thousands of threads (e.g. N=1500 → 12000 workers).
    return min(max(8, spec.num_clients * 2), 512)


def _interruptible_sleep(stop: threading.Event, seconds: float) -> bool:
    """Sleep up to *seconds*; return True if *stop* was set before it elapsed."""
    if seconds <= 0:
        return stop.is_set()
    return stop.wait(timeout=seconds)


def _drain_futures(
    futures: list[Future],
    *,
    verbose: bool = False,
    per_future_timeout_s: Optional[float] = None,
) -> None:
    pending = sum(1 for f in futures if not f.done())
    if verbose and pending:
        print(f"  draining {pending}/{len(futures)} in-flight requests...")
    for fut in futures:
        if fut.done():
            try:
                fut.result()
            except Exception:
                pass
            continue
        try:
            fut.result(timeout=per_future_timeout_s)
        except Exception:
            pass


def run_workload(
    spec: RunSpec,
    client_factory: ClientFactory,
    pairs: List[PromptPair],
    verbose: bool = False,
) -> WorkloadResult:
    """Execute one factor combination and return the collected request records."""
    records: List[RequestRecord] = []
    records_lock = threading.Lock()
    futures_lock = threading.Lock()
    stop = threading.Event()
    cap = spec.response_text_max_chars
    interrupted = False

    def do_request(client, client_id: int, idx: int, sched) -> None:
        if stop.is_set():
            return
        ts_start = time.time()
        try:
            res = client.stream_chat(
                sched.content,
                index=idx,
                is_attack=sched.is_attack,
                pair_index=sched.pair_index,
            )
        except Exception as exc:  # noqa: BLE001 — worker must never kill the pool
            from contambench.harness.client import RequestResult

            res = RequestResult(
                index=idx,
                is_attack=sched.is_attack,
                pair_index=sched.pair_index,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            res.total_s = time.time() - ts_start
        latency_s = res.total_s if res.total_s is not None else (time.time() - ts_start)
        ttft_ms = res.ttft_s * 1000.0 if res.ttft_s is not None else None
        tbt_ms = res.itl_mean_s * 1000.0 if res.itl_mean_s is not None else None
        ts_first = ts_start + res.ttft_s if res.ttft_s is not None else None
        ts_end = ts_start + latency_s
        if res.decode_s is not None:
            decode_residency_ms = res.decode_s * 1000.0
        elif ts_first is not None:
            decode_residency_ms = (ts_end - ts_first) * 1000.0
        else:
            decode_residency_ms = None
        prompt = sched.content
        full_text = getattr(res, "generated_text", "") or res.generated_preview or ""
        stored_text = full_text[:cap] if cap is not None else full_text
        record = RequestRecord(
            request_id=uuid.uuid4().hex,
            client_id=client_id,
            trial=spec.trial,
            config_id=spec.config_id(),
            timestamp_start=ts_start,
            timestamp_first_token=ts_first,
            timestamp_end=ts_end,
            prompt_type="adversarial" if sched.is_attack else "benign",
            contamination_ratio=spec.contamination_ratio,
            arrival_rate=spec.arrival_rate,
            arrival_model=spec.arrival_model,
            num_clients=spec.num_clients,
            success=bool(res.success),
            http_status=res.http_status,
            input_tokens=res.usage_prompt_tokens,
            output_tokens=res.output_tokens,
            ttft_ms=ttft_ms,
            tbt_ms=tbt_ms,
            latency_ms=latency_s * 1000.0,
            finish_reason=res.finish_reason,
            error=res.error,
            pair_index=sched.pair_index,
            prompt_id=sched.pair_index,
            model=res.model,
            input_text=prompt,
            output_text=stored_text,
            decode_residency_ms=decode_residency_ms,
            max_tokens=spec.max_tokens,
        )
        with records_lock:
            records.append(record)
        if verbose:
            tag = "ATK" if sched.is_attack else "BEN"
            status = "ok" if res.success else f"FAIL({res.error})"
            ttft_str = "n/a" if ttft_ms is None else f"{ttft_ms:.0f}ms"
            print(
                f"  [client {client_id} req {idx}] {tag} {status} "
                f"ttft={ttft_str} latency={latency_s * 1000.0:.0f}ms tokens={res.output_tokens}"
            )

    max_workers = _default_max_workers(spec)
    futures: list[Future] = []
    start_barrier = threading.Barrier(spec.num_clients)
    wall_start_box: list[float] = [0.0]
    start_lock = threading.Lock()

    # Shared across all clients so salted prefixes actually repeat (partial reuse).
    salt_pool = (
        make_salt_pool(spec.seed, spec.salt_pool_size, spec.prompt_salt_chars) or None
        if spec.prompt_salt_chars > 0
        else None
    )

    def submit_request(client, client_id: int, idx: int, sched) -> bool:
        """Schedule one HTTP request. Returns False if dispatch should stop."""
        if stop.is_set():
            return False
        try:
            fut = pool.submit(do_request, client, client_id, idx, sched)
        except RuntimeError:
            # Pool is shutting down (interrupt or interpreter exit).
            return False
        with futures_lock:
            futures.append(fut)
        return True

    def client_loop(client_id: int) -> None:
        rng = random.Random((spec.seed * 1_000_003) ^ (client_id * 2_654_435_761) ^ spec.trial)
        client = client_factory(client_id)
        offsets = build_offsets(
            spec.arrival_model,
            spec.arrival_rate,
            spec.requests_per_client,
            rng,
            burst_size=spec.burst_size,
            burst_factor=spec.burst_factor,
        )
        try:
            start_barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            return
        if stop.is_set():
            return
        with start_lock:
            if wall_start_box[0] == 0.0:
                wall_start_box[0] = time.time()
        start = wall_start_box[0]
        for i in range(spec.requests_per_client):
            if stop.is_set():
                break
            sched = select_prompt(
                pairs, spec.contamination_ratio, rng, index=i,
                salt_length=spec.prompt_salt_chars, salt_pool=salt_pool,
            )
            target = start + offsets[i]
            now = time.time()
            if target > now and _interruptible_sleep(stop, target - now):
                break
            if not submit_request(client, client_id, i, sched):
                break

    client_threads = [
        threading.Thread(target=client_loop, args=(cid,), name=f"client-{cid}", daemon=True)
        for cid in range(spec.num_clients)
    ]

    pool = ThreadPoolExecutor(max_workers=max_workers)
    wall_end = time.time()
    try:
        for t in client_threads:
            t.start()
        for t in client_threads:
            while t.is_alive():
                t.join(timeout=0.5)
                if stop.is_set():
                    break
        _drain_futures(futures, verbose=verbose)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        if verbose:
            print("  interrupted; stopping new dispatches and draining in-flight requests...")
        for t in client_threads:
            t.join(timeout=2.0)
        _drain_futures(futures, verbose=verbose, per_future_timeout_s=5.0)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=False)
        wall_end = time.time()

    records.sort(key=lambda r: (r.client_id, r.timestamp_start))
    result = WorkloadResult(
        spec=spec,
        records=records,
        wall_start=wall_start_box[0] or wall_end,
        wall_end=wall_end,
    )
    if interrupted:
        raise WorkloadInterrupted(result)
    return result
