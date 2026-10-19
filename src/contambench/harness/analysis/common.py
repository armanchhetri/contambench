"""Shared helpers for the RQ analyses: data loading, trial aggregation, plotting.

All analyses operate on the pooled per-request table produced by the storage
layer (one row per request, tagged with the experimental factors). Statistics
are computed *per trial* and then aggregated across trials, so error bars and
confidence intervals reflect between-trial variability.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from contambench.harness.metrics import confidence_interval, violation_rate  # noqa: E402
from contambench.harness.storage import load_all_requests, results_root  # noqa: E402

WORKLOAD_COLS = ["num_clients", "arrival_rate", "arrival_model"]

TrialAggregation = Literal["mean", "median"]
DEFAULT_TRIAL_AGGREGATION: TrialAggregation = "mean"

SLO_REGIME_LABELS = ("baseline",)
# Auto SLO: budgets = benign c=0 p99 at the selected workload (1.0×, per metric).
SLO_REGIME_BASELINE_PERCENTILE = 99
SLO_REGIME_P99_MULTIPLIERS = {"baseline": 1.0}


@dataclass
class SLORegimes:
    """Latency, TTFT, and TBT SLO budgets keyed by regime (baseline = benign c=0 p99)."""

    auto: bool
    latency: dict[str, float]
    ttft: dict[str, float]
    tbt: dict[str, float] = field(default_factory=dict)
    baseline_n: int = 0
    baseline_description: str = ""
    workload: Optional[dict] = None
    workload_label: str = ""
    latency_p99_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tbt_p99_ms: Optional[float] = None

    def latency_pairs(self) -> list[tuple[str, float]]:
        return [(label, self.latency[label]) for label in SLO_REGIME_LABELS if label in self.latency]

    def ttft_pairs(self) -> list[tuple[str, float]]:
        return [(label, self.ttft[label]) for label in SLO_REGIME_LABELS if label in self.ttft]

    def tbt_pairs(self) -> list[tuple[str, float]]:
        return [(label, self.tbt[label]) for label in SLO_REGIME_LABELS if label in self.tbt]

    def latency_values(self) -> list[float]:
        return [budget for _, budget in self.latency_pairs()]

    def ttft_values(self) -> list[float]:
        return [budget for _, budget in self.ttft_pairs()]

    def tbt_values(self) -> list[float]:
        return [budget for _, budget in self.tbt_pairs()]

    def slo_col(self, metric: str, regime: str) -> str:
        """Column name for a regime's violation rate (e.g. slo_viol_latency_baseline)."""
        return f"slo_viol_{metric}_{regime}"

    def budget_stat(self, regime: str) -> str:
        """Effects-table stat label for a regime budget."""
        return f"budget_{regime}"

    def to_dict(self) -> dict:
        regimes = {}
        for label in SLO_REGIME_LABELS:
            entry: dict = {}
            if self.auto:
                entry["p99_multiplier"] = SLO_REGIME_P99_MULTIPLIERS.get(label)
            if label in self.latency:
                entry["latency_ms"] = self.latency[label]
            if label in self.ttft:
                entry["ttft_ms"] = self.ttft[label]
            if label in self.tbt:
                entry["tbt_ms"] = self.tbt[label]
            regimes[label] = entry
        out = {
            "auto": self.auto,
            "baseline_n": self.baseline_n,
            "baseline_description": self.baseline_description,
            "regimes": regimes,
        }
        if self.workload is not None:
            out["workload"] = self.workload
        if self.workload_label:
            out["workload_label"] = self.workload_label
        if self.latency_p99_ms is not None:
            out["latency_p99_ms"] = self.latency_p99_ms
        if self.ttft_p99_ms is not None:
            out["ttft_p99_ms"] = self.ttft_p99_ms
        if self.tbt_p99_ms is not None:
            out["tbt_p99_ms"] = self.tbt_p99_ms
        if self.auto:
            out["baseline_percentile"] = SLO_REGIME_BASELINE_PERCENTILE
            out["p99_multipliers"] = dict(SLO_REGIME_P99_MULTIPLIERS)
        return out

    @classmethod
    def from_fixed(
        cls,
        latency_slos: Sequence[float],
        ttft_slos: Sequence[float],
        tbt_slos: Sequence[float],
    ) -> SLORegimes:
        """Map fixed budgets to the single baseline regime (first value per metric)."""
        lat = float(latency_slos[0]) if latency_slos else 0.0
        ttft = float(ttft_slos[0]) if ttft_slos else 0.0
        tbt = float(tbt_slos[0]) if tbt_slos else 0.0
        return cls(
            auto=False,
            latency={"baseline": lat},
            ttft={"baseline": ttft},
            tbt={"baseline": tbt},
            baseline_description="fixed budgets from config",
        )


def workload_keys_match(a: Optional[dict], b: Optional[dict]) -> bool:
    """True when two workload keys denote the same (N, λ, arrival_model)."""
    if not a or not b:
        return False
    for col in WORKLOAD_COLS:
        if col not in a or col not in b:
            return False
        if col == "arrival_rate":
            if not np.isclose(float(a[col]), float(b[col])):
                return False
        elif col == "num_clients":
            if int(a[col]) != int(b[col]):
                return False
        elif str(a[col]) != str(b[col]):
            return False
    return True


def workload_key_from_frame(work: pd.DataFrame) -> dict:
    """Return the single (N, lambda, arrival_model) key present in *work*."""
    key: dict = {}
    for col in WORKLOAD_COLS:
        if col not in work.columns:
            continue
        uniq = work[col].dropna().unique()
        if len(uniq) != 1:
            raise ValueError(
                f"SLO calibration requires one workload configuration; "
                f"{col!r} has {len(uniq)} distinct values"
            )
        if col == "arrival_rate":
            key[col] = float(uniq[0])
        elif col == "num_clients":
            key[col] = int(uniq[0])
        else:
            key[col] = str(uniq[0])
    return key


def benign_c0_baseline(work: pd.DataFrame) -> pd.DataFrame:
    """Benign, successful requests at c=0 within an already workload-scoped frame."""
    ok = successful(work)
    return ok[
        (ok["prompt_type"] == "benign") & np.isclose(ok["contamination_ratio"], 0.0)
    ]


def benign_slo_baseline(work: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Benign baseline for auto-SLO: prefer c=0, else lowest contamination in frame."""
    baseline = benign_c0_baseline(work)
    if not baseline.empty:
        return baseline, "benign c=0"
    ok = successful(work)
    benign = ok[ok["prompt_type"].astype(str).str.lower() == "benign"]
    if benign.empty or "contamination_ratio" not in benign.columns:
        return baseline, "benign c=0"
    c_min = float(benign["contamination_ratio"].min())
    at_min = benign[np.isclose(benign["contamination_ratio"], c_min)]
    return at_min, f"benign c={c_min:g}"


def _p99_multiplier_budgets(values: np.ndarray) -> tuple[dict[str, float], float]:
    if values.size == 0:
        raise ValueError("cannot derive SLO budgets from an empty sample")
    p99 = float(np.percentile(values, SLO_REGIME_BASELINE_PERCENTILE))
    budgets = {
        label: p99 * SLO_REGIME_P99_MULTIPLIERS[label] for label in SLO_REGIME_LABELS
    }
    return budgets, p99


def compute_auto_slos(work: pd.DataFrame, workload_key: Optional[dict] = None) -> SLORegimes:
    """Derive SLO budgets from benign c=0 p99 at the selected workload (1.0×)."""
    frame_key = workload_key_from_frame(work)
    if workload_key is not None:
        for col in WORKLOAD_COLS:
            if col not in workload_key or col not in frame_key:
                continue
            expected = workload_key[col]
            actual = frame_key[col]
            if col == "arrival_rate":
                if not np.isclose(float(actual), float(expected)):
                    raise ValueError(
                        f"SLO workload mismatch on {col!r}: "
                        f"frame has {actual}, expected {expected}"
                    )
            elif str(actual) != str(expected):
                raise ValueError(
                    f"SLO workload mismatch on {col!r}: frame has {actual!r}, expected {expected!r}"
                )
    key = workload_key or frame_key
    label = workload_label(key)

    baseline, baseline_tag = benign_slo_baseline(work)
    if baseline.empty:
        raise ValueError(
            f"auto_slo requires benign requests for workload {label}"
        )
    lat_vals = baseline["latency_ms"].dropna().to_numpy(dtype=float)
    ttft_vals = baseline["ttft_ms"].dropna().to_numpy(dtype=float)
    if lat_vals.size == 0 or ttft_vals.size == 0:
        raise ValueError(f"auto_slo baseline missing latency/TTFT for workload {label}")
    lat_budgets, lat_p99 = _p99_multiplier_budgets(lat_vals)
    ttft_budgets, ttft_p99 = _p99_multiplier_budgets(ttft_vals)
    tbt_budgets: dict[str, float] = {}
    tbt_p99: Optional[float] = None
    if "tbt_ms" in baseline.columns:
        tbt_vals = baseline["tbt_ms"].dropna().to_numpy(dtype=float)
        if tbt_vals.size > 0:
            tbt_budgets, tbt_p99 = _p99_multiplier_budgets(tbt_vals)
    return SLORegimes(
        auto=True,
        latency=lat_budgets,
        ttft=ttft_budgets,
        tbt=tbt_budgets,
        baseline_n=int(len(baseline)),
        baseline_description=(
            f"{baseline_tag} p99 at {label} (1.0× latency, TTFT, TBT)"
        ),
        workload=key,
        workload_label=label,
        latency_p99_ms=lat_p99,
        ttft_p99_ms=ttft_p99,
        tbt_p99_ms=tbt_p99,
    )


def resolve_slo_regimes(
    work: pd.DataFrame,
    *,
    auto_slo: bool,
    latency_slos: Sequence[float],
    ttft_slos: Sequence[float],
    tbt_slos: Sequence[float],
    workload_key: Optional[dict] = None,
) -> SLORegimes:
    if auto_slo:
        return compute_auto_slos(work, workload_key=workload_key)
    regimes = SLORegimes.from_fixed(latency_slos, ttft_slos, tbt_slos)
    if workload_key is not None:
        regimes.workload = workload_key
        regimes.workload_label = workload_label(workload_key)
    return regimes


def resolve_regimes_for_workload(
    df: pd.DataFrame,
    workload_key: dict,
    *,
    auto_slo: bool,
    latency_slos: Sequence[float],
    ttft_slos: Sequence[float],
    tbt_slos: Sequence[float],
) -> SLORegimes:
    """Derive SLO budgets from benign c=0 p99 at *this* (N, λ) only."""
    work = filter_workload(df, workload_key)
    if work.empty:
        raise ValueError(
            f"no requests for workload {workload_label(workload_key)}; "
            "cannot derive per-λ SLO budgets"
        )
    return resolve_slo_regimes(
        work,
        auto_slo=auto_slo,
        latency_slos=latency_slos,
        ttft_slos=ttft_slos,
        tbt_slos=tbt_slos,
        workload_key=workload_key,
    )


def ensure_regimes_for_workload(
    df: pd.DataFrame,
    workload_key: Optional[dict],
    regimes: Optional[SLORegimes],
    *,
    auto_slo: bool,
    latency_slos: Sequence[float],
    ttft_slos: Sequence[float],
    tbt_slos: Sequence[float],
) -> SLORegimes:
    """Return SLO regimes pinned to *workload_key*; recompute when mismatched."""
    if workload_key is None:
        if regimes is None:
            raise ValueError("SLO regimes required when workload_key is unset")
        return regimes
    if regimes is not None and workload_keys_match(regimes.workload, workload_key):
        return regimes
    return resolve_regimes_for_workload(
        df,
        workload_key,
        auto_slo=auto_slo,
        latency_slos=latency_slos,
        ttft_slos=ttft_slos,
        tbt_slos=tbt_slos,
    )


def combined_request_slo_violation_rate(
    lat: np.ndarray,
    ttft: np.ndarray,
    tbt: np.ndarray,
    lat_budget: float,
    ttft_budget: float,
    tbt_budget: Optional[float] = None,
) -> Optional[float]:
    """Fraction of requests violating latency ∨ TTFT ∨ TBT budgets."""
    n = min(lat.size, ttft.size)
    if n == 0:
        return None
    lat, ttft = lat[:n], ttft[:n]
    mask = ~(np.isnan(lat) | np.isnan(ttft))
    lat, ttft = lat[mask], ttft[mask]
    if lat.size == 0:
        return None
    viol = (lat > lat_budget) | (ttft > ttft_budget)
    if tbt_budget is not None and tbt.size >= n:
        tbt_trim = tbt[:n][mask]
        if tbt_trim.size:
            viol = viol | (tbt_trim > tbt_budget)
    return float(viol.mean())


def infer_auto_slo_from_metadata(results_dir: Optional[str | Path] = None) -> Optional[bool]:
    """Read auto_slo from raw run metadata; None if unavailable or inconsistent."""
    root = results_root(results_dir)
    flags: list[bool] = []
    for meta_path in sorted(root.glob("raw/run_*/metadata.json")):
        try:
            with open(meta_path, encoding="utf-8") as f:
                cfg = json.load(f).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        if "auto_slo" in cfg:
            flags.append(bool(cfg["auto_slo"]))
    if not flags:
        return None
    if len(set(flags)) > 1:
        return None
    return flags[0]


def load_slo_config_from_runs(run_dirs: Sequence[Path]) -> Optional[dict]:
    """Read ``auto_slo`` and fixed budgets from the selected run metadata files.

    Used at analyze time so post-hoc edits to ``metadata.json`` (or values
    recorded at experiment start) apply without re-passing every CLI flag.
    Returns ``None`` when no metadata is available. Raises when selected runs
    disagree on SLO configuration.
    """
    configs: list[dict] = []
    for run_dir in run_dirs:
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                cfg = json.load(f).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        if cfg:
            configs.append(cfg)
    if not configs:
        return None

    out: dict = {}
    auto_vals = [bool(c["auto_slo"]) for c in configs if "auto_slo" in c]
    if auto_vals:
        if len(set(auto_vals)) > 1:
            raise ValueError(
                "Selected runs disagree on auto_slo; pass --auto-slo or --no-auto-slo explicitly"
            )
        out["auto_slo"] = auto_vals[0]

    if out.get("auto_slo") is False:
        for key in ("latency_slo_ms", "ttft_slo_ms", "tbt_slo_ms"):
            tuples: list[tuple[float, ...]] = []
            for cfg in configs:
                raw = cfg.get(key)
                if raw:
                    tuples.append(tuple(float(x) for x in raw[:3]))
            if tuples:
                if len(set(tuples)) > 1:
                    raise ValueError(
                        f"Selected runs disagree on {key}; pass explicit --{key.replace('_', '-')}"
                    )
                out[key] = list(tuples[0])
    return out or None


def save_slo_regimes(regimes: SLORegimes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regimes.to_dict(), f, indent=2, ensure_ascii=False, default=str)

STAT_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "p95": lambda x: float(np.percentile(x, 95)),
    "p99": lambda x: float(np.percentile(x, 99)),
    "max": lambda x: float(np.max(x)),
    "sum": lambda x: float(np.sum(x)),
}


# ---------------------------------------------------------------------------
# plotting style
# ---------------------------------------------------------------------------
def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "lines.linewidth": 2.2,
            "lines.markersize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
# data loading / preparation
# ---------------------------------------------------------------------------
def load_requests(
    results_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    df = load_all_requests(results_dir, run_id=run_id)
    if df.empty:
        return df
    for col in ("latency_ms", "ttft_ms", "tbt_ms", "output_tokens", "input_tokens",
                "decode_residency_ms", "contamination_ratio", "arrival_rate", "num_clients",
                "trial", "prompt_id", "pair_index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalise the success flag (csv may store it as text).
    if df["success"].dtype != bool:
        df["success"] = df["success"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "t", "yes")
        )
    df["prompt_type"] = df["prompt_type"].astype(str)
    return enrich_request_fields(df)


def enrich_request_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Back-fill ``prompt_id`` and ``decode_residency_ms`` for older raw runs."""
    if df.empty:
        return df
    out = df.copy()
    if "prompt_id" not in out.columns and "pair_index" in out.columns:
        out["prompt_id"] = out["pair_index"]
    elif "prompt_id" in out.columns and "pair_index" in out.columns:
        out["prompt_id"] = out["prompt_id"].fillna(out["pair_index"])
    if "decode_residency_ms" not in out.columns:
        out["decode_residency_ms"] = np.nan
    missing_decode = out["decode_residency_ms"].isna()
    if missing_decode.any():
        if {"timestamp_end", "timestamp_first_token"}.issubset(out.columns):
            ts_end = pd.to_numeric(out["timestamp_end"], errors="coerce")
            ts_first = pd.to_numeric(out["timestamp_first_token"], errors="coerce")
            derived = (ts_end - ts_first) * 1000.0
            out.loc[missing_decode, "decode_residency_ms"] = derived.loc[missing_decode]
        elif {"latency_ms", "ttft_ms"}.issubset(out.columns):
            derived = out["latency_ms"] - out["ttft_ms"]
            out.loc[missing_decode, "decode_residency_ms"] = derived.loc[missing_decode]
    return out


def successful(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["success"] == True]  # noqa: E712


def iter_workloads(df: pd.DataFrame):
    """Yield (workload_key_dict, sub_df) for each distinct workload config."""
    for key, sub in df.groupby(WORKLOAD_COLS):
        key_dict = dict(zip(WORKLOAD_COLS, key))
        yield key_dict, sub


def _workload_score(sub: pd.DataFrame) -> tuple[int, int]:
    return (sub["contamination_ratio"].nunique(), len(sub))


def filter_workload(df: pd.DataFrame, key: dict) -> pd.DataFrame:
    """Return rows matching a workload key (``num_clients``, ``arrival_rate``, ``arrival_model``)."""
    mask = pd.Series(True, index=df.index)
    for col in WORKLOAD_COLS:
        if col not in df.columns or col not in key:
            continue
        target = float(key[col]) if col == "arrival_rate" else key[col]
        series = df[col]
        if col == "arrival_rate":
            mask &= np.isclose(series.astype(float), float(target))
        else:
            mask &= series == target
    return df[mask].copy()


def list_workload_options(df: pd.DataFrame) -> list[dict]:
    """Enumerate distinct workloads with summary stats for interactive selection."""
    options = []
    for key_dict, sub in iter_workloads(df):
        options.append(
            {
                "key": key_dict,
                "label": workload_label(key_dict),
                "n_rows": len(sub),
                "n_contamination": sub["contamination_ratio"].nunique(),
                "score": _workload_score(sub),
            }
        )
    options.sort(key=lambda o: (int(o["key"]["num_clients"]), float(o["key"]["arrival_rate"]), str(o["key"]["arrival_model"])))
    return options


def primary_workload_index(options: Sequence[dict]) -> int:
    """Index of the auto-selected primary workload within *options*."""
    best_i, best_score = 0, (-1, -1)
    for i, opt in enumerate(options):
        if opt["score"] > best_score:
            best_i, best_score = i, opt["score"]
    return best_i


def select_primary_workload(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pick the workload with the most contamination levels (then most rows)."""
    options = list_workload_options(df)
    if not options:
        return {}, df
    idx = primary_workload_index(options)
    key = options[idx]["key"]
    return key, filter_workload(df, key)


def get_workload(df: pd.DataFrame, workload_key: Optional[dict] = None) -> tuple[dict, pd.DataFrame]:
    """Resolve a workload by explicit key or fall back to the primary auto-pick."""
    if workload_key:
        return workload_key, filter_workload(df, workload_key)
    return select_primary_workload(df)


def prompt_workload_selection(
    df: pd.DataFrame,
    *,
    workload_index: Optional[int] = None,
    interactive: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Show workload options and return the user-selected (or default) workload.

    *workload_index* is 0-based; when set, skips the prompt.
    When *interactive* is False or stdin is not a TTY, the primary workload is
    used without prompting.
    """
    options = list_workload_options(df)
    if not options:
        return {}, df
    if len(options) == 1:
        key = options[0]["key"]
        print(f"Workload: {options[0]['label']} (only option)")
        return key, filter_workload(df, key)

    default_i = primary_workload_index(options)

    if workload_index is not None:
        if not 0 <= workload_index < len(options):
            raise ValueError(
                f"workload_index {workload_index} out of range (0..{len(options) - 1})"
            )
        key = options[workload_index]["key"]
        return key, filter_workload(df, key)

    use_tty = interactive and sys.stdin.isatty()
    if not use_tty:
        key = options[default_i]["key"]
        print(f"Primary workload (non-interactive): {options[default_i]['label']}")
        return key, filter_workload(df, key)

    print("\nAvailable workloads:")
    for i, opt in enumerate(options):
        default_mark = "  [default]" if i == default_i else ""
        print(
            f"  {i + 1}. {opt['label']}"
            f"  — {opt['n_rows']} requests, {opt['n_contamination']} contamination levels"
            f"{default_mark}"
        )

    while True:
        try:
            raw = input(f"\nSelect workload [1-{len(options)}] (Enter = {default_i + 1}): ").strip()
        except EOFError:
            raw = ""
        if raw == "":
            idx = default_i
            break
        try:
            idx = int(raw) - 1
        except ValueError:
            print("  Enter a number from the list.")
            continue
        if 0 <= idx < len(options):
            break
        print(f"  Enter a number between 1 and {len(options)}.")

    key = options[idx]["key"]
    return key, filter_workload(df, key)


def workload_label(key: dict) -> str:
    return (
        f"N={int(key.get('num_clients', 0))}, "
        f"lambda={key.get('arrival_rate')}, "
        f"{key.get('arrival_model')}"
    )


# ---------------------------------------------------------------------------
# trial-level aggregation
# ---------------------------------------------------------------------------
def aggregate_trials(
    values: Sequence[float],
    method: TrialAggregation = DEFAULT_TRIAL_AGGREGATION,
) -> float:
    """Combine per-trial scalars (mean or median across trials)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("aggregate_trials requires at least one value")
    if method == "median":
        return float(np.median(arr))
    return float(arr.mean())


def stat_vs_contamination(
    df: pd.DataFrame,
    value_col: str,
    stat_name: str,
    trial_aggregation: TrialAggregation = DEFAULT_TRIAL_AGGREGATION,
) -> pd.DataFrame:
    """Per-trial ``stat_name`` of ``value_col``, aggregated across trials.

    ``df`` should already be restricted to one workload config and (optionally)
    one prompt type. Returns columns:
    contamination_ratio, stat, value, ci_low, ci_high, n_trials, std,
    trial_aggregation.
    """
    fn = STAT_FUNCS[stat_name]
    ok = successful(df)
    rows = []
    for c, g in ok.groupby("contamination_ratio"):
        per_trial = []
        for _, gt in g.groupby("trial"):
            vals = gt[value_col].dropna().to_numpy(dtype=float)
            if vals.size:
                per_trial.append(fn(vals))
        if not per_trial:
            continue
        arr = np.asarray(per_trial, dtype=float)
        lo, hi = confidence_interval(arr)
        rows.append(
            {
                "contamination_ratio": float(c),
                "stat": stat_name,
                "value": aggregate_trials(arr, trial_aggregation),
                "ci_low": lo,
                "ci_high": hi,
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n_trials": int(arr.size),
                "trial_aggregation": trial_aggregation,
            }
        )
    return pd.DataFrame(rows).sort_values("contamination_ratio").reset_index(drop=True)


def slo_violation_vs_contamination(
    df: pd.DataFrame,
    value_col: str,
    budget_ms: float,
    regime: Optional[str] = None,
    trial_aggregation: TrialAggregation = DEFAULT_TRIAL_AGGREGATION,
) -> pd.DataFrame:
    """Per-trial SLO violation rate of ``value_col`` vs contamination."""
    ok = successful(df)
    rows = []
    for c, g in ok.groupby("contamination_ratio"):
        per_trial = []
        for _, gt in g.groupby("trial"):
            vals = gt[value_col].dropna().to_numpy(dtype=float)
            if vals.size:
                per_trial.append(float((vals > budget_ms).mean()))
        if not per_trial:
            continue
        arr = np.asarray(per_trial, dtype=float)
        lo, hi = confidence_interval(arr)
        row = {
            "contamination_ratio": float(c),
            "budget_ms": float(budget_ms),
            "value": aggregate_trials(arr, trial_aggregation),
            "ci_low": lo,
            "ci_high": hi,
            "n_trials": int(arr.size),
            "trial_aggregation": trial_aggregation,
        }
        if regime is not None:
            row["regime"] = regime
        rows.append(row)
    return pd.DataFrame(rows).sort_values("contamination_ratio").reset_index(drop=True)


def yerr_from(table: pd.DataFrame) -> Optional[np.ndarray]:
    """Build a 2xK asymmetric error-bar array from ci columns (None if 1 trial)."""
    if table.empty or table["n_trials"].max() <= 1:
        return None
    val = table["value"].to_numpy(dtype=float)
    lo = table["ci_low"].to_numpy(dtype=float)
    hi = table["ci_high"].to_numpy(dtype=float)
    lower = np.clip(val - lo, 0, None)
    upper = np.clip(hi - val, 0, None)
    return np.vstack([lower, upper])


def baseline_value(table: pd.DataFrame, baseline_c: float = 0.0) -> Optional[float]:
    base = table[np.isclose(table["contamination_ratio"], baseline_c)]
    if base.empty:
        return None
    return float(base["value"].iloc[0])
