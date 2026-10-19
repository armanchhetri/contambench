"""Score D3-format request logs with the SLO oracle used in the paper.

Oracle: for each arrival rate, the budget of each metric (latency, TTFT, TBT) is the
99th percentile over the c = 0 requests of the same run. A request violates when it
exceeds any of the three budgets. Violation rates are computed per trial and averaged,
with a Student-t 95% interval over trials.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("latency_ms", "ttft_ms", "tbt_ms")


def load(path: str | Path | list) -> pd.DataFrame:
    """One run directory or file, or several (concatenated, e.g. a benign and a contaminated run)."""
    if isinstance(path, (list, tuple)):
        return pd.concat([load(p) for p in path], ignore_index=True)
    p = Path(path)
    if p.is_dir():
        p = p / "requests.parquet" if (p / "requests.parquet").exists() else p / "requests.csv"
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df["success"] = df["success"].astype(bool)
    return df


def ci95(vals: np.ndarray) -> tuple[float, float, float]:
    m = float(vals.mean())
    if vals.size < 2:
        return m, m, m
    h = stats.t.ppf(0.975, vals.size - 1) * vals.std(ddof=1) / math.sqrt(vals.size)
    return m, m - h, m + h


def budgets(df: pd.DataFrame) -> dict[str, float]:
    base = df[(df.contamination_ratio == 0) & df.success]
    if base.empty:
        raise ValueError("no successful c = 0 requests at this arrival rate; the oracle needs a benign baseline")
    return {k: float(np.percentile(base[k].dropna(), 99)) for k in METRICS}


def violates(g: pd.DataFrame, b: dict[str, float]) -> pd.Series:
    return (g.latency_ms > b["latency_ms"]) | (g.ttft_ms > b["ttft_ms"]) | (g.tbt_ms > b["tbt_ms"])


def _per_trial(df: pd.DataFrame, fn) -> np.ndarray:
    return np.asarray([fn(g) for _, g in df.groupby("trial")], dtype=float)


def is_stressor(df: pd.DataFrame) -> pd.Series:
    return df.prompt_type != "benign"


def load_budgets(source: str | Path, arrival_rate: float | None = None) -> dict[str, float]:
    """Budgets from a slo_budgets.json file, or computed from the c = 0 requests of another run."""
    p = Path(source)
    if p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text())
        if arrival_rate is None:
            if len(data) != 1:
                raise ValueError(f"{p} has budgets for several arrival rates; pass --budget-rate")
            entry = next(iter(data.values()))
        else:
            key = min(data, key=lambda k: abs(float(k) - arrival_rate))
            if abs(float(key) - arrival_rate) > 1e-6:
                raise ValueError(f"{p} has no budgets at arrival rate {arrival_rate}")
            entry = data[key]
        return {m: float(entry[f"{m}_budget"]) for m in METRICS}
    df = load(p)
    if arrival_rate is not None:
        df = df[np.isclose(df.arrival_rate, arrival_rate)]
    return budgets(df)


def sweep(df: pd.DataFrame, arrival_rate: float, budget: dict[str, float] | None = None) -> dict:
    """Per-c rows at one arrival rate, as in the paper's sweep table.

    ``budget`` overrides the oracle; by default it comes from the c = 0 requests at this rate.
    """
    sub = df[np.isclose(df.arrival_rate, arrival_rate)]
    if sub.empty:
        raise ValueError(f"no requests at arrival rate {arrival_rate}")
    b = budget or budgets(sub)
    ok = sub[sub.success]
    rows, base_benign_p95 = [], None
    if not (sub.contamination_ratio == 0).any():
        base_benign_p95 = None
    for c, g in ok.groupby("contamination_ratio"):
        ben, adv = g[~is_stressor(g)], g[is_stressor(g)]
        viol = _per_trial(g, lambda t: float(violates(t, b).mean()))
        ben_p95 = float(_per_trial(ben, lambda t: float(np.percentile(t.latency_ms, 95))).mean()) if len(ben) else float("nan")
        if c == 0:
            base_benign_p95 = ben_p95
        m, lo, hi = ci95(viol)
        rows.append({
            "c": float(c),
            "n_requests": int((sub.contamination_ratio == c).sum()),
            "n_success": int(len(g)),
            "n_stressor": int(len(adv)),
            "trials": int(g.trial.nunique()),
            "p95_latency_s": float(_per_trial(g, lambda t: float(np.percentile(t.latency_ms, 95))).mean()) / 1000,
            "slo_violation": m,
            "slo_violation_ci": [lo, hi],
            "benign_slo_violation": float(_per_trial(ben, lambda t: float(violates(t, b).mean())).mean()) if len(ben) else None,
            "stressor_slo_violation": float(violates(adv, b).mean()) if len(adv) else None,
            "stressor_p95_latency_s": float(np.percentile(adv.latency_ms, 95)) / 1000 if len(adv) else None,
            "benign_p95_latency_s": ben_p95 / 1000,
            "benign_p95_inflation": ben_p95 / base_benign_p95 if base_benign_p95 else None,
            "stressor_cap_hit": float((adv.finish_reason == "length").mean()) if len(adv) else None,
        })
    return {"arrival_rate": float(arrival_rate), "budgets_ms": b, "rows": rows}


def stage_b(df: pd.DataFrame) -> dict:
    """Stage-B adequacy over every successful request of a run."""
    ok = df[df.success]
    adv, ben = ok[is_stressor(ok)], ok[~is_stressor(ok)]
    return {
        "n_stressor": int(len(adv)),
        "n_benign": int(len(ben)),
        "stressor_cap_hit": float((adv.finish_reason == "length").mean()) if len(adv) else None,
        "benign_cap_hit": float((ben.finish_reason == "length").mean()) if len(ben) else None,
        "stressor_mean_out": float(adv.output_tokens.mean()) if len(adv) else None,
        "benign_mean_out": float(ben.output_tokens.mean()) if len(ben) else None,
        "stressor_mean_residency_s": float(adv.decode_residency_ms.mean()) / 1000 if len(adv) else None,
        "benign_mean_residency_s": float(ben.decode_residency_ms.mean()) / 1000 if len(ben) else None,
    }


def _pc(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def format_sweep(s: dict) -> str:
    out = [f"arrival rate {s['arrival_rate']:g}   budgets: " + ", ".join(f"{k[:-3]} {v:.1f} ms" for k, v in s["budgets_ms"].items()),
           f"{'c':>5} {'p95 s':>7} {'viol %':>7} {'95% CI':>13} {'benign %':>9} {'stress %':>9} {'benign p95x':>11} {'cap %':>6}"]
    for r in s["rows"]:
        lo, hi = r["slo_violation_ci"]
        infl = "-" if r["c"] == 0 or r["benign_p95_inflation"] is None else f"{r['benign_p95_inflation']:.2f}"
        out.append(f"{r['c']:>5.2f} {r['p95_latency_s']:>7.1f} {100 * r['slo_violation']:>7.1f} "
                   f"{100 * max(lo, 0):>6.1f}-{100 * min(hi, 1):<6.1f} {_pc(r['benign_slo_violation']):>9} "
                   f"{_pc(r['stressor_slo_violation']):>9} {infl:>11} {_pc(r['stressor_cap_hit']):>6}")
    return "\n".join(out)
