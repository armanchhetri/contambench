"""RQ1 validation metrics: length-finish ASR and stretch ratio (no plotting)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from contambench.harness.metrics import summary_stats

LENGTH_FINISH_REASON = "length"
_PROMPT_TYPES = ("benign", "adversarial")


def is_length_saturated(finish_reason: object) -> bool:
    """True when generation stopped because ``max_tokens`` was reached."""
    if finish_reason is None:
        return False
    if isinstance(finish_reason, float) and math.isnan(finish_reason):
        return False
    return str(finish_reason).strip().lower() == LENGTH_FINISH_REASON


def _benign_baselines_by_prompt(ok: pd.DataFrame) -> pd.Series:
    if "prompt_id" not in ok.columns and "pair_index" in ok.columns:
        ok = ok.assign(prompt_id=ok["pair_index"])
    if "prompt_id" not in ok.columns:
        return pd.Series(dtype=float)
    benign = ok[ok["prompt_type"] == "benign"]
    if benign.empty:
        return pd.Series(dtype=float)
    lengths = pd.to_numeric(benign["output_tokens"], errors="coerce")
    grouped = benign.assign(_len=lengths).dropna(subset=["_len", "prompt_id"])
    if grouped.empty:
        return pd.Series(dtype=float)
    return grouped.groupby("prompt_id")["_len"].median()


def compute_stretch_table(ok: pd.DataFrame) -> pd.DataFrame:
    """One row per adversarial request with paired benign baseline length."""
    adv = ok[ok["prompt_type"] == "adversarial"].copy()
    if adv.empty:
        return pd.DataFrame()

    baselines = _benign_baselines_by_prompt(ok)
    adv["output_tokens"] = pd.to_numeric(adv["output_tokens"], errors="coerce")
    if "prompt_id" in adv.columns:
        adv["prompt_id"] = pd.to_numeric(adv["prompt_id"], errors="coerce")
    elif "pair_index" in adv.columns:
        adv["prompt_id"] = pd.to_numeric(adv["pair_index"], errors="coerce")
    adv["benign_baseline_tokens"] = adv["prompt_id"].map(baselines)
    adv["stretch_ratio"] = np.where(
        adv["benign_baseline_tokens"] > 0,
        adv["output_tokens"] / adv["benign_baseline_tokens"],
        np.nan,
    )
    cols = [
        "request_id", "run_id", "config_id", "trial", "prompt_id",
        "output_tokens", "benign_baseline_tokens", "stretch_ratio",
        "finish_reason", "decode_residency_ms", "model",
    ]
    present = [c for c in cols if c in adv.columns]
    return adv[present].dropna(subset=["stretch_ratio"])


def compute_asr_table(ok: pd.DataFrame) -> pd.DataFrame:
    """ASR = share of successful requests with ``finish_reason == 'length'``."""
    if "finish_reason" not in ok.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    models = ok["model"].dropna().unique().tolist() if "model" in ok.columns else [None]
    if not models:
        models = [None]

    for model in models:
        sub = ok if model is None else ok[ok["model"] == model]
        for ptype in _PROMPT_TYPES:
            part = sub[sub["prompt_type"] == ptype]
            known = part[part["finish_reason"].notna()]
            n = int(len(known))
            n_sat = int(known["finish_reason"].map(is_length_saturated).sum()) if n else 0
            rows.append({
                "model": model or "unknown",
                "prompt_type": ptype,
                "n": n,
                "n_saturated": n_sat,
                "asr": (n_sat / n) if n else None,
                "saturation_criterion": f"finish_reason={LENGTH_FINISH_REASON!r}",
            })
    return pd.DataFrame(rows)


def per_prompt_asr(ok: pd.DataFrame) -> pd.Series:
    """Adversarial ASR per ``prompt_id`` (length-finish rate)."""
    adv = ok[ok["prompt_type"] == "adversarial"].copy()
    if adv.empty or "finish_reason" not in adv.columns:
        return pd.Series(dtype=float)
    if "prompt_id" not in adv.columns and "pair_index" in adv.columns:
        adv["prompt_id"] = adv["pair_index"]
    if "prompt_id" not in adv.columns:
        return pd.Series(dtype=float)
    known = adv[adv["finish_reason"].notna()].copy()
    if known.empty:
        return pd.Series(dtype=float)
    known["_sat"] = known["finish_reason"].map(is_length_saturated).astype(float)
    return known.groupby("prompt_id")["_sat"].mean()


def compute_validation_summary(df: pd.DataFrame) -> dict:
    """Overall validation metrics for console / JSON summaries."""
    try:
        from contambench.harness.analysis.common import enrich_request_fields, successful

        ok = successful(enrich_request_fields(df))
    except ImportError:
        ok = df[df["success"] == True] if "success" in df.columns else df  # noqa: E712
    stretch = compute_stretch_table(ok)
    asr_table = compute_asr_table(ok)
    stretch_vals = (
        stretch["stretch_ratio"].dropna().to_numpy(dtype=float)
        if not stretch.empty
        else np.array([])
    )
    per_prompt = per_prompt_asr(ok)
    per_prompt_vals = per_prompt.to_numpy(dtype=float) if not per_prompt.empty else np.array([])

    adv_row = asr_table[asr_table["prompt_type"] == "adversarial"]
    all_known = ok[ok["finish_reason"].notna()] if "finish_reason" in ok.columns else ok.iloc[0:0]
    n_all = int(len(all_known))
    n_all_sat = (
        int(all_known["finish_reason"].map(is_length_saturated).sum())
        if n_all and "finish_reason" in all_known.columns
        else 0
    )

    return {
        "n_success": int(len(ok)),
        "n_benign": int((ok["prompt_type"] == "benign").sum()),
        "n_adversarial": int((ok["prompt_type"] == "adversarial").sum()),
        "asr": {
            "criterion": f"finish_reason={LENGTH_FINISH_REASON!r}",
            "overall": {
                "n": n_all,
                "n_saturated": n_all_sat,
                "rate": (n_all_sat / n_all) if n_all else None,
                "per_prompt_mean": float(np.mean(per_prompt_vals)) if per_prompt_vals.size else None,
                "per_prompt_median": float(np.median(per_prompt_vals)) if per_prompt_vals.size else None,
            },
            "adversarial": {
                "n": int(adv_row["n"].sum()) if not adv_row.empty else 0,
                "n_saturated": int(adv_row["n_saturated"].sum()) if not adv_row.empty else 0,
                "rate": (
                    float(adv_row["asr"].iloc[0])
                    if len(adv_row) == 1 and pd.notna(adv_row["asr"].iloc[0])
                    else None
                ),
            },
            "by_prompt_type": asr_table.to_dict(orient="records") if not asr_table.empty else [],
        },
        "stretch_ratio": summary_stats(stretch_vals),
    }


def format_validation_report(summary: dict) -> list[str]:
    """Human-readable lines for post-run / post-analyze console output."""
    lines: list[str] = []
    asr = summary.get("asr", {})
    overall = asr.get("overall", {})
    stretch = summary.get("stretch_ratio", {})

    pm = overall.get("per_prompt_mean")
    pmed = overall.get("per_prompt_median")
    if pm is not None and pmed is not None:
        lines.append(f"ASR (overall): mean={pm:.3f}, median={pmed:.3f} (per-prompt adversarial)")
    elif overall.get("rate") is not None:
        lines.append(f"ASR (overall): {overall['rate']:.3f}")

    adv = asr.get("adversarial", {})
    if adv.get("rate") is not None:
        lines.append(
            f"ASR (adversarial): {adv['rate']:.3f} "
            f"({adv.get('n_saturated', 0)}/{adv.get('n', 0)} length finishes)"
        )

    if stretch.get("mean") is not None:
        lines.append(
            f"Stretch ratio (overall): mean={stretch['mean']:.2f}, "
            f"median={stretch['median']:.2f} (n={stretch.get('n', 0)})"
        )
    return lines
