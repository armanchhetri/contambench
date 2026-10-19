"""Benign workload calibration — map the safe operating envelope and derive SLOs.

Plots mean latency, TTFT, and TBT (time-between-tokens) vs concurrent clients,
with one line per arrival rate, using only benign requests at contamination c=0.

Empirical SLO budgets follow the evaluation practice in latency-sensitive LLM
serving work: use the p95 of a reference benign operating point (default: the
lightest offered load N×λ in the calibration data).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import confidence_interval, save_figure, save_table, setup_style, successful

_METRICS = (
    ("latency_mean_ms", "latency_ms", "mean", "Latency (mean)", "ms"),
    ("ttft_mean_ms", "ttft_ms", "mean", "TTFT (mean)", "ms"),
    ("tbt_mean_ms", "tbt_ms", "mean", "TBT (mean)", "ms"),
)


def benign_calibration_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Successful benign requests at c=0."""
    ok = successful(df)
    ben = ok[ok["prompt_type"].astype(str).str.lower() == "benign"]
    return ben[np.isclose(ben["contamination_ratio"].astype(float), 0.0)].copy()


def _trial_aggregates(ben: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["num_clients", "arrival_rate", "arrival_model", "trial"]
    for key, g in ben.groupby(group_cols, dropna=False):
        nc, lam, model, trial = key
        row = {
            "num_clients": int(nc),
            "arrival_rate": float(lam),
            "arrival_model": str(model),
            "trial": int(trial),
            "n_requests": len(g),
            "offered_load": float(nc) * float(lam),
        }
        for out_col, src_col, stat, _, _ in _METRICS:
            vals = g[src_col].dropna().to_numpy(dtype=float) if src_col in g.columns else np.array([])
            if vals.size == 0:
                row[out_col] = np.nan
                row[out_col.replace("_mean_", "_p95_")] = np.nan
                continue
            if stat == "mean":
                row[out_col] = float(vals.mean())
            row[out_col.replace("_mean_", "_p95_")] = float(np.percentile(vals, 95))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_across_trials(trial_tbl: pd.DataFrame) -> pd.DataFrame:
    """Mean of per-trial means, with 95% CI across trials."""
    value_cols = [m[0] for m in _METRICS]
    rows = []
    for (nc, lam, model), g in trial_tbl.groupby(["num_clients", "arrival_rate", "arrival_model"]):
        row = {
            "num_clients": int(nc),
            "arrival_rate": float(lam),
            "arrival_model": str(model),
            "offered_load": float(nc) * float(lam),
            "n_trials": len(g),
            "n_requests": int(g["n_requests"].sum()),
        }
        for col in value_cols:
            arr = g[col].dropna().to_numpy(dtype=float)
            if arr.size == 0:
                row[col] = np.nan
                row[f"{col}_ci_low"] = np.nan
                row[f"{col}_ci_high"] = np.nan
                continue
            lo, hi = confidence_interval(arr)
            row[col] = float(arr.mean())
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["arrival_rate", "num_clients"]).reset_index(drop=True)


def _plot_metric_vs_clients(table: pd.DataFrame, metric_col: str, ylabel: str, out_dir: Path, name: str) -> None:
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for lam in sorted(table["arrival_rate"].unique()):
        s = table[table["arrival_rate"] == lam].sort_values("num_clients")
        if s[metric_col].isna().all():
            continue
        yerr = None
        if s["n_trials"].max() > 1:
            val = s[metric_col].to_numpy(dtype=float)
            lo = s[f"{metric_col}_ci_low"].to_numpy(dtype=float)
            hi = s[f"{metric_col}_ci_high"].to_numpy(dtype=float)
            lower = np.clip(val - lo, 0, None)
            upper = np.clip(hi - val, 0, None)
            yerr = np.vstack([lower, upper])
        ax.errorbar(
            s["num_clients"],
            s[metric_col],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=f"λ={lam:g} req/s/client",
        )
    ax.set_xlabel("Concurrent clients N")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Benign calibration: {ylabel} vs N")
    ax.legend(title="Arrival rate")
    save_figure(fig, out_dir, name)


def _derive_reference_slos(
    ben: pd.DataFrame,
    ref_clients: Optional[int] = None,
    ref_rate: Optional[float] = None,
) -> dict:
    if ben.empty:
        return {"error": "no benign c=0 calibration data"}

    cell = ben.copy()
    if ref_clients is not None and ref_rate is not None:
        cell = cell[
            (cell["num_clients"].astype(int) == int(ref_clients))
            & np.isclose(cell["arrival_rate"].astype(float), float(ref_rate))
        ]
        ref_label = f"N={ref_clients}, λ={ref_rate}"
    else:
        cell["offered_load"] = cell["num_clients"].astype(float) * cell["arrival_rate"].astype(float)
        min_load = cell["offered_load"].min()
        cell = cell[cell["offered_load"] == min_load]
        sample = cell.iloc[0]
        ref_label = f"N={int(sample['num_clients'])}, λ={float(sample['arrival_rate']):g} (min N×λ)"

    if cell.empty:
        return {"error": "reference operating point not found in data"}

    ref_row = cell.iloc[0]
    ref = {
        "num_clients": int(ref_row["num_clients"]),
        "arrival_rate": float(ref_row["arrival_rate"]),
        "arrival_model": str(ref_row.get("arrival_model", "")),
        "label": ref_label,
    }

    lat = cell["latency_ms"].dropna().to_numpy(dtype=float)
    ttft = cell["ttft_ms"].dropna().to_numpy(dtype=float)
    tbt = cell["tbt_ms"].dropna().to_numpy(dtype=float) if "tbt_ms" in cell.columns else np.array([])

    slo = {
        "method": "empirical_p95_benign_c0",
        "description": "p95 over all benign c=0 requests at the reference operating point",
        "reference": ref,
        "n_requests": int(len(cell)),
        "latency_ms": {
            "p95": float(np.percentile(lat, 95)) if lat.size else None,
            "p99": float(np.percentile(lat, 99)) if lat.size else None,
            "mean": float(lat.mean()) if lat.size else None,
        },
        "ttft_ms": {
            "p95": float(np.percentile(ttft, 95)) if ttft.size else None,
            "p99": float(np.percentile(ttft, 99)) if ttft.size else None,
            "mean": float(ttft.mean()) if ttft.size else None,
        },
        "tbt_ms": {
            "p95": float(np.percentile(tbt, 95)) if tbt.size else None,
            "p99": float(np.percentile(tbt, 99)) if tbt.size else None,
            "mean": float(tbt.mean()) if tbt.size else None,
        },
        "suggested_fixed_slos": {
            "latency_ms_tight": float(np.percentile(lat, 95)) if lat.size else None,
            "ttft_ms_tight": float(np.percentile(ttft, 95)) if ttft.size else None,
            "tbt_ms_tight": float(np.percentile(tbt, 95)) if tbt.size else None,
        },
    }
    return slo


def run(
    df: pd.DataFrame,
    out_dir: Path,
    ref_clients: Optional[int] = None,
    ref_rate: Optional[float] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ben = benign_calibration_frame(df)
    if ben.empty:
        msg = {"error": "no benign c=0 rows — run experiment.py with contamination_ratio: 0.0 first"}
        with open(out_dir / "calibration_slos.json", "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2)
        return msg

    trial_tbl = _trial_aggregates(ben)
    agg = _aggregate_across_trials(trial_tbl)
    save_table(trial_tbl, out_dir, "benign_by_trial")
    save_table(agg, out_dir, "benign_vs_clients")

    has_tbt = "tbt_ms" in ben.columns and ben["tbt_ms"].notna().any()
    for out_col, _, _, ylabel, _ in _METRICS:
        if out_col == "tbt_mean_ms" and not has_tbt:
            continue
        if agg[out_col].isna().all():
            continue
        _plot_metric_vs_clients(agg, out_col, ylabel, out_dir, out_col.replace("_ms", ""))

    slos = _derive_reference_slos(ben, ref_clients=ref_clients, ref_rate=ref_rate)
    slos["calibration_points"] = len(agg)
    with open(out_dir / "calibration_slos.json", "w", encoding="utf-8") as f:
        json.dump(slos, f, indent=2, ensure_ascii=False, default=str)

    return slos
