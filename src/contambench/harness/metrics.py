"""Statistical metric computation and the summary generator.

Computes, for an arbitrary set of request rows, the latency/TTFT distribution
statistics (mean, median, p95, p99, max, std, 95% CI), the success / timeout
rates, and per-SLO-budget violation rates (latency, TTFT, combined).

The summary generator splits requests into *all*, *benign* and *adversarial*
groups, matching the study requirement to report each separately.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    _scipy_stats = None

_TIMEOUT_RE = re.compile(r"timeout|timed out|read timed out|ReadTimeout|ConnectTimeout", re.I)

# Percentiles reported everywhere.
_PCTS = {"p95": 95, "p99": 99}


def _t_critical(conf: float, df: int) -> float:
    """Two-sided t critical value; falls back to the normal approx."""
    if df <= 0:
        return float("nan")
    alpha = 1.0 - conf
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df))
    # Normal approximation (good for large df).
    return 1.959963984540054


def confidence_interval(values: Sequence[float], conf: float = 0.95) -> tuple[Optional[float], Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = arr.size
    if n == 0:
        return (None, None)
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / math.sqrt(n))
    h = _t_critical(conf, n - 1) * sem
    return (mean - h, mean + h)


def summary_stats(values: Iterable[float], conf: float = 0.95) -> dict:
    """Distribution statistics for one numeric vector."""
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=float,
    )
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0, "mean": None, "median": None, "p95": None, "p99": None,
            "max": None, "std": None, "ci95_low": None, "ci95_high": None,
        }
    low, high = confidence_interval(arr, conf)
    return {
        "n": n,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
    }


def _is_timeout(row: pd.Series) -> bool:
    if bool(row.get("success")):
        return False
    err = row.get("error")
    if isinstance(err, str) and _TIMEOUT_RE.search(err):
        return True
    return False


def violation_rate(values: Sequence[float], budget_ms: float) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return None
    return float((arr > budget_ms).mean())


def _combined_violation_rate(
    lat: np.ndarray,
    ttft: np.ndarray,
    lat_budget: float,
    ttft_budget: float,
    tbt: Optional[np.ndarray] = None,
    tbt_budget: Optional[float] = None,
) -> Optional[float]:
    """Fraction violating latency ∨ TTFT, and ∨ TBT when a TBT budget is set."""
    mask = ~(np.isnan(lat) | np.isnan(ttft))
    lat, ttft = lat[mask], ttft[mask]
    if lat.size == 0:
        return None
    viol = (lat > lat_budget) | (ttft > ttft_budget)
    if tbt is not None and tbt_budget is not None and tbt.size == mask.size:
        tbt_ok = tbt[mask]
        viol = viol | (tbt_ok > tbt_budget)
    return float(viol.mean())


def compute_group_metrics(
    df: pd.DataFrame,
    latency_slos: Sequence[float],
    ttft_slos: Sequence[float],
    tbt_slos: Sequence[float] = (),
) -> dict:
    """Full metric block for one group of requests (all / benign / adversarial)."""
    n_total = int(len(df))
    success = df[df["success"] == True] if n_total else df  # noqa: E712
    n_success = int(len(success))

    lat = success["latency_ms"].to_numpy(dtype=float) if n_success else np.array([])
    ttft = success["ttft_ms"].to_numpy(dtype=float) if n_success else np.array([])
    tbt = (
        success["tbt_ms"].to_numpy(dtype=float)
        if n_success and "tbt_ms" in success.columns
        else np.array([])
    )

    n_timeout = int(df.apply(_is_timeout, axis=1).sum()) if n_total else 0

    slo_block = {"latency": {}, "ttft": {}, "tbt": {}, "combined": {}}
    for b in latency_slos:
        slo_block["latency"][str(int(b))] = {
            "budget_ms": float(b),
            "latency_slo_violation_rate": violation_rate(lat, b),
        }
    for b in ttft_slos:
        slo_block["ttft"][str(int(b))] = {
            "budget_ms": float(b),
            "ttft_slo_violation_rate": violation_rate(ttft, b),
        }
    for b in tbt_slos:
        slo_block["tbt"][str(int(b))] = {
            "budget_ms": float(b),
            "tbt_slo_violation_rate": violation_rate(tbt, b),
        }
    tbt_budget = float(tbt_slos[0]) if tbt_slos else None
    for lb in latency_slos:
        for tb in ttft_slos:
            key = f"lat{int(lb)}_ttft{int(tb)}"
            entry = {
                "latency_budget_ms": float(lb),
                "ttft_budget_ms": float(tb),
                "combined_slo_violation_rate": _combined_violation_rate(
                    lat if lat.size else np.array([]),
                    ttft if ttft.size else np.array([]),
                    lb,
                    tb,
                    tbt if tbt.size else None,
                    tbt_budget,
                ),
            }
            if tbt_budget is not None:
                entry["tbt_budget_ms"] = tbt_budget
            slo_block["combined"][key] = entry

    return {
        "n": n_total,
        "n_success": n_success,
        "request_success_rate": (n_success / n_total) if n_total else None,
        "timeout_rate": (n_timeout / n_total) if n_total else None,
        "latency_ms": summary_stats(lat),
        "ttft_ms": summary_stats(ttft),
        "tbt_ms": summary_stats(tbt),
        "output_tokens": summary_stats(success["output_tokens"].to_numpy(dtype=float)) if n_success else summary_stats([]),
        "decode_residency_ms": (
            summary_stats(success["decode_residency_ms"].to_numpy(dtype=float))
            if n_success and "decode_residency_ms" in success.columns
            else summary_stats([])
        ),
        "slo": slo_block,
    }


def generate_summaries(
    df: pd.DataFrame,
    latency_slos: Sequence[float],
    ttft_slos: Sequence[float],
    tbt_slos: Sequence[float] = (),
) -> dict:
    """Compute overall / benign / adversarial metric blocks for a run."""
    benign = df[df["prompt_type"] == "benign"]
    adversarial = df[df["prompt_type"] == "adversarial"]
    out = {
        "overall": compute_group_metrics(df, latency_slos, ttft_slos, tbt_slos),
        "benign": compute_group_metrics(benign, latency_slos, ttft_slos, tbt_slos),
        "adversarial": compute_group_metrics(adversarial, latency_slos, ttft_slos, tbt_slos),
    }
    try:
        from contambench.harness.validation import compute_validation_summary

        out["validation"] = compute_validation_summary(df)
    except Exception:
        pass
    return out
