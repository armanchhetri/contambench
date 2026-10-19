"""Step 1 — server capacity calibration at fixed N (default 50).

Sweeps arrival rate at contamination c=0 to find peak achievable throughput,
then recommends per-client λ for low (60%), medium (80%), and high (90%) of
that capacity. Plots λ vs achieved throughput, mean latency, and mean TTFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .common import confidence_interval, save_figure, save_table, setup_style, successful

DEFAULT_NUM_CLIENTS = 50
DEFAULT_LOAD_FRACTIONS: tuple[float, ...] = (0.6, 0.8, 0.9)
DEFAULT_LOAD_LABELS: tuple[str, ...] = ("low", "medium", "high")


def capacity_calibration_frame(
    df: pd.DataFrame,
    *,
    num_clients: int = DEFAULT_NUM_CLIENTS,
) -> pd.DataFrame:
    """Successful benign requests at c=0 for the target client count."""
    ok = successful(df)
    ben = ok[ok["prompt_type"].astype(str).str.lower() == "benign"]
    sub = ben[
        (ben["num_clients"].astype(int) == int(num_clients))
        & np.isclose(ben["contamination_ratio"].astype(float), 0.0)
    ]
    return sub.copy()


def _per_trial_cells(ben: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["num_clients", "arrival_rate", "arrival_model", "trial", "config_id"]
    for key, g in ben.groupby(group_cols, dropna=False):
        nc, lam, model, trial, config_id = key
        t0 = float(g["timestamp_start"].min())
        t1 = float(g["timestamp_end"].max())
        dur = max(1e-9, t1 - t0)
        lat = g["latency_ms"].dropna().to_numpy(dtype=float)
        ttft = g["ttft_ms"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "config_id": str(config_id),
                "num_clients": int(nc),
                "arrival_rate": float(lam),
                "arrival_model": str(model),
                "trial": int(trial),
                "offered_load_req_s": float(nc) * float(lam),
                "n_requests": int(len(g)),
                "duration_s": dur,
                "achieved_throughput_req_s": len(g) / dur,
                "latency_mean_ms": float(lat.mean()) if lat.size else np.nan,
                "ttft_mean_ms": float(ttft.mean()) if ttft.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_by_arrival_rate(trial_tbl: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for lam, g in trial_tbl.groupby("arrival_rate"):
        row: dict = {
            "arrival_rate": float(lam),
            "offered_load_req_s": float(g["offered_load_req_s"].iloc[0]),
            "n_trials": int(len(g)),
            "n_requests": int(g["n_requests"].sum()),
        }
        for col in (
            "achieved_throughput_req_s",
            "latency_mean_ms",
            "ttft_mean_ms",
        ):
            arr = g[col].dropna().to_numpy(dtype=float)
            lo, hi = confidence_interval(arr) if arr.size else (None, None)
            row[col] = float(arr.mean()) if arr.size else np.nan
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("arrival_rate").reset_index(drop=True)


def _interpolate_lambda(table: pd.DataFrame, target_throughput: float) -> Optional[float]:
    """λ whose achieved throughput best matches *target_throughput* (linear interp)."""
    t = table.dropna(subset=["achieved_throughput_req_s", "arrival_rate"]).sort_values("arrival_rate")
    if t.empty:
        return None
    x = t["arrival_rate"].to_numpy(dtype=float)
    y = t["achieved_throughput_req_s"].to_numpy(dtype=float)
    if target_throughput <= y[0]:
        return float(x[0])
    if target_throughput >= y[-1]:
        return float(x[-1])
    for i in range(len(y) - 1):
        if y[i] <= target_throughput <= y[i + 1]:
            if y[i + 1] == y[i]:
                return float(x[i + 1])
            frac = (target_throughput - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + frac * (x[i + 1] - x[i]))
    return float(x[-1])


def recommend_load_points(
    table: pd.DataFrame,
    *,
    fractions: Sequence[float] = DEFAULT_LOAD_FRACTIONS,
    labels: Sequence[str] = DEFAULT_LOAD_LABELS,
) -> list[dict]:
    cap = float(table["achieved_throughput_req_s"].max())
    recs: list[dict] = []
    t = table.dropna(subset=["achieved_throughput_req_s", "arrival_rate"]).copy()
    for frac, label in zip(fractions, labels):
        target = frac * cap
        if t.empty:
            recs.append(
                {
                    "label": label,
                    "fraction_of_capacity": float(frac),
                    "target_throughput_req_s": target,
                    "recommended_arrival_rate": None,
                    "expected_throughput_req_s": None,
                }
            )
            continue
        # Pick the calibrated λ whose achieved throughput is closest to the target.
        t = t.assign(
            throughput_gap=(t["achieved_throughput_req_s"] - target).abs()
        )
        best = t.loc[t["throughput_gap"].idxmin()]
        lam = float(best["arrival_rate"])
        achieved = float(best["achieved_throughput_req_s"])
        recs.append(
            {
                "label": label,
                "fraction_of_capacity": float(frac),
                "target_throughput_req_s": target,
                "recommended_arrival_rate": lam,
                "expected_throughput_req_s": achieved,
            }
        )
    return recs


def _plot_capacity_curves(table: pd.DataFrame, out_dir: Path, num_clients: int, capacity: float) -> None:
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    x = table["arrival_rate"]
    thr = table["achieved_throughput_req_s"]
    thr_lo = table.get("achieved_throughput_req_s_ci_low")
    thr_hi = table.get("achieved_throughput_req_s_ci_high")

    ax = axes[0]
    if thr_lo is not None and thr_hi is not None:
        yerr = np.vstack([
            np.clip(thr - thr_lo, 0, None),
            np.clip(thr_hi - thr, 0, None),
        ])
        ax.errorbar(x, thr, yerr=yerr, marker="o", capsize=4, label="achieved throughput")
    else:
        ax.plot(x, thr, "o-", label="achieved throughput")
    ax.axhline(capacity, color="gray", ls="--", lw=1, label=f"capacity ≈ {capacity:.2f} req/s")
    for frac, label in zip(DEFAULT_LOAD_FRACTIONS, DEFAULT_LOAD_LABELS):
        ax.axhline(frac * capacity, color="lightgray", ls=":", lw=0.8)
        ax.text(x.max(), frac * capacity, f" {label} {int(frac*100)}%", va="bottom", fontsize=10)
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title(f"Capacity calibration: N={num_clients}, c=0 (benign)")
    ax.legend()

    for ax, col, ylabel in (
        (axes[1], "latency_mean_ms", "Mean latency (ms)"),
        (axes[2], "ttft_mean_ms", "Mean TTFT (ms)"),
    ):
        y = table[col]
        lo, hi = table.get(f"{col}_ci_low"), table.get(f"{col}_ci_high")
        if lo is not None and hi is not None:
            yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
            ax.errorbar(x, y, yerr=yerr, marker="s", capsize=4)
        else:
            ax.plot(x, y, "s-")
        ax.set_ylabel(ylabel)

    axes[2].set_xlabel("Per-client arrival rate λ (req/s)")
    save_figure(fig, out_dir, "arrival_rate_vs_throughput")


def run(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    load_fractions: Sequence[float] = DEFAULT_LOAD_FRACTIONS,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ben = capacity_calibration_frame(df, num_clients=num_clients)
    if ben.empty:
        msg = {
            "error": f"no benign c=0 data at N={num_clients} — run capacity calibration first",
        }
        with open(out_dir / "capacity_calibration.json", "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2)
        return msg

    trial_tbl = _per_trial_cells(ben)
    agg = _aggregate_by_arrival_rate(trial_tbl)
    save_table(trial_tbl, out_dir, "capacity_by_trial")
    save_table(agg, out_dir, "capacity_by_arrival_rate")

    capacity = float(agg["achieved_throughput_req_s"].max())
    recommendations = recommend_load_points(agg, fractions=load_fractions)
    _plot_capacity_curves(agg, out_dir, num_clients, capacity)

    payload = {
        "num_clients": int(num_clients),
        "capacity_throughput_req_s": capacity,
        "n_calibration_points": int(len(agg)),
        "load_recommendations": recommendations,
        "recommended_arrival_rates": [
            r["recommended_arrival_rate"] for r in recommendations if r["recommended_arrival_rate"]
        ],
        "arrival_rate_table": agg.to_dict(orient="records"),
    }
    with open(out_dir / "capacity_calibration.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload
