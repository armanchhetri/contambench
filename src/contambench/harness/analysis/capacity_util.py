"""Capacity-utilization helpers for cross-model workload selection."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .common import successful

DEFAULT_TARGET_UTILIZATION = 0.90


def measured_utilization(
    df: pd.DataFrame,
    workload_key: dict,
    capacity_req_s: float,
) -> float:
    """Return achieved throughput / capacity for benign c=0 at *workload_key*."""
    if capacity_req_s <= 0:
        return float("nan")
    nc = int(workload_key["num_clients"])
    lam = float(workload_key["arrival_rate"])
    wk = successful(
        df[
            (df["num_clients"].astype(int) == nc)
            & np.isclose(df["arrival_rate"].astype(float), lam)
            & np.isclose(df["contamination_ratio"].astype(float), 0.0)
            & (df["prompt_type"].astype(str).str.lower() == "benign")
        ]
    )
    if wk.empty or "timestamp_start" not in wk.columns or "timestamp_end" not in wk.columns:
        return float("nan")
    thrs: list[float] = []
    for _, g in wk.groupby("trial"):
        dur = max(1e-9, float(g["timestamp_end"].max()) - float(g["timestamp_start"].min()))
        thrs.append(len(g) / dur)
    if not thrs:
        return float("nan")
    return float(np.mean(thrs) / capacity_req_s)


def pick_workload_near_utilization(
    df: pd.DataFrame,
    candidate_lambdas: Sequence[float],
    capacity_req_s: float,
    *,
    num_clients: int = 50,
    arrival_model: str = "poisson",
    target_utilization: float = DEFAULT_TARGET_UTILIZATION,
) -> tuple[dict, float]:
    """Pick (N, λ) from *candidate_lambdas* whose utilization is closest to *target*."""
    best_key: Optional[dict] = None
    best_util = float("nan")
    best_gap = float("inf")
    for lam in candidate_lambdas:
        key = {
            "num_clients": int(num_clients),
            "arrival_rate": float(lam),
            "arrival_model": str(arrival_model),
        }
        util = measured_utilization(df, key, capacity_req_s)
        if np.isnan(util):
            continue
        gap = abs(util - target_utilization)
        if gap < best_gap:
            best_gap = gap
            best_util = util
            best_key = key
    if best_key is None:
        raise ValueError("no candidate workload with measurable utilization")
    return best_key, best_util
