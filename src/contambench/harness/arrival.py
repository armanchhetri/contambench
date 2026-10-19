"""Arrival-rate controllers / workload schedulers.

Each controller turns a target rate ``lambda`` (requests/second, *per client*)
into a sequence of absolute dispatch offsets (seconds from the client's start).
The workload scheduler dispatches request ``i`` at ``t_start + offsets[i]`` in an
*open-loop* fashion, so request latency does not delay later arrivals.

Supported models
----------------
fixed    Deterministic spacing of ``1/lambda`` seconds.
poisson  Exponentially distributed inter-arrival gaps (a Poisson process).
bursty   Tight bursts of ``burst_size`` requests separated by idle gaps, with
         the long-run average held at ``lambda``.
"""

from __future__ import annotations

import random
from typing import List


def fixed_offsets(rate: float, n: int) -> List[float]:
    if rate <= 0:
        raise ValueError("rate must be positive")
    step = 1.0 / rate
    return [i * step for i in range(n)]


def poisson_offsets(rate: float, n: int, rng: random.Random) -> List[float]:
    if rate <= 0:
        raise ValueError("rate must be positive")
    offsets: List[float] = []
    t = 0.0
    for i in range(n):
        if i > 0:
            t += rng.expovariate(rate)
        offsets.append(t)
    return offsets


def bursty_offsets(
    rate: float,
    n: int,
    rng: random.Random,
    burst_size: int = 5,
    burst_factor: float = 10.0,
) -> List[float]:
    """Bursty arrivals with long-run mean rate ``rate``.

    Within a burst, requests are spaced at ``rate * burst_factor`` (fast). Between
    bursts an idle gap is inserted so that the average inter-arrival time over a
    whole cycle equals ``1/rate``.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    burst_size = max(1, int(burst_size))
    intra = 1.0 / (rate * max(1.0, burst_factor))
    # A cycle of `burst_size` requests should span burst_size/rate on average.
    cycle_target = burst_size / rate
    burst_span = (burst_size - 1) * intra
    idle_mean = max(0.0, cycle_target - burst_span)

    offsets: List[float] = []
    t = 0.0
    i = 0
    while i < n:
        for _ in range(burst_size):
            if i >= n:
                break
            offsets.append(t)
            t += intra
            i += 1
        # remove the trailing intra added after the last in-burst request and
        # replace it with a (jittered) idle gap before the next burst.
        if i < n:
            t = t - intra + (rng.expovariate(1.0 / idle_mean) if idle_mean > 0 else 0.0)
    return offsets


def build_offsets(
    arrival_model: str,
    rate: float,
    n: int,
    rng: random.Random,
    burst_size: int = 5,
    burst_factor: float = 10.0,
) -> List[float]:
    """Dispatch offsets for one client under the requested arrival model."""
    if arrival_model == "fixed":
        return fixed_offsets(rate, n)
    if arrival_model == "poisson":
        return poisson_offsets(rate, n, rng)
    if arrival_model == "bursty":
        return bursty_offsets(rate, n, rng, burst_size=burst_size, burst_factor=burst_factor)
    raise ValueError(f"unknown arrival_model: {arrival_model!r}")
