"""Structured storage layer for the contamination study.

Owns the canonical output tree::

    results/
      raw/
        <name>_run_<timestamp>/   # or run_<timestamp> when name omitted
          requests.csv
          requests.parquet
          metadata.json
      summaries/
        run_<timestamp>/
          overall_metrics.json
          benign_metrics.json
          adversarial_metrics.json
      rq1/  rq2/  rq3/  rq4/
      campaigns/
        <campaign_id>.json     # campaign registry
        <campaign_id>/         # campaign-scoped analysis outputs
          calibrate-capacity/
          rq2/  rq3/  …

A "run" here is one experiment invocation, which may contain many factor
combinations (raw rows are tagged with ``config_id``, ``trial``,
``contamination_ratio``, ``num_clients`` and ``arrival_rate``).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from contambench.harness.workload import RequestRecord

# Measurements a user takes are written under the working directory, never into the package.
DEFAULT_RESULTS_DIR = Path.cwd() / "contambench-results"

RAW_DIRNAME = "raw"
SUMMARY_DIRNAME = "summaries"
CAMPAIGNS_DIRNAME = "campaigns"
RQ_DIRNAMES = ("rq1", "rq2", "rq3", "rq4", "rq5")
ANALYSIS_DIRNAMES = ("calibrate", "calibrate-capacity", "effects", *RQ_DIRNAMES)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slug_run_name(name: Optional[str]) -> Optional[str]:
    """Filesystem-safe slug for prepending to run directory names."""
    if not name or not str(name).strip():
        return None
    slug = re.sub(r"[^\w.\-]+", "_", str(name).strip())
    slug = slug.strip("._")
    return slug or None


def results_root(results_dir: Optional[str | Path] = None) -> Path:
    return Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR


def campaign_analysis_root(results_dir: Path, campaign_id: str) -> Path:
    """Per-campaign analysis tree under ``results/campaigns/<campaign_id>/``."""
    return results_dir / CAMPAIGNS_DIRNAME / campaign_id


def analysis_output_dir(
    results_dir: Path,
    analysis_name: str,
    *,
    campaign_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Directory for one analysis kind (optional per-run subfolder)."""
    base = campaign_analysis_root(results_dir, campaign_id) if campaign_id else results_dir
    path = base / analysis_name
    if run_id:
        path = path / run_id
    return path


def analysis_results_base(results_dir: Path, *, campaign_id: Optional[str] = None) -> Path:
    """Root for shared analysis artifacts (e.g. ``slo_regimes.json``)."""
    if campaign_id:
        return campaign_analysis_root(results_dir, campaign_id)
    return results_dir


def resolve_capacity_calibration_path(
    results_dir: Path,
    *,
    campaign_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """Return ``capacity_calibration.json`` path (campaign dir first, then legacy)."""
    candidates: list[Path] = []
    if campaign_id:
        candidates.append(
            analysis_output_dir(
                results_dir, "calibrate-capacity", campaign_id=campaign_id, run_id=run_id,
            )
            / "capacity_calibration.json"
        )
    candidates.append(
        analysis_output_dir(results_dir, "calibrate-capacity", run_id=run_id)
        / "capacity_calibration.json"
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def ensure_tree(results_dir: Optional[str | Path] = None) -> Path:
    root = results_root(results_dir)
    (root / RAW_DIRNAME).mkdir(parents=True, exist_ok=True)
    (root / SUMMARY_DIRNAME).mkdir(parents=True, exist_ok=True)
    for rq in RQ_DIRNAMES:
        (root / rq).mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class RunPaths:
    run_id: str
    root: Path
    raw_dir: Path
    summary_dir: Path

    @property
    def requests_csv(self) -> Path:
        return self.raw_dir / "requests.csv"

    @property
    def requests_parquet(self) -> Path:
        return self.raw_dir / "requests.parquet"

    @property
    def metadata_json(self) -> Path:
        return self.raw_dir / "metadata.json"


def new_run(
    results_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    *,
    name: Optional[str] = None,
) -> RunPaths:
    root = ensure_tree(results_dir)
    if run_id:
        rid = run_id
    else:
        stamp = now_stamp()
        slug = slug_run_name(name)
        rid = f"{slug}_run_{stamp}" if slug else f"run_{stamp}"
    raw_dir = root / RAW_DIRNAME / rid
    summary_dir = root / SUMMARY_DIRNAME / rid
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=rid, root=root, raw_dir=raw_dir, summary_dir=summary_dir)


_SECRET_KEYS = {"api_key", "authorization", "runpod_api_key"}


def redact(obj):
    """Drop credentials before metadata reaches disk."""
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k.lower() in _SECRET_KEYS and v else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def records_to_frame(records: Iterable[RequestRecord]) -> pd.DataFrame:
    rows = [r.to_dict() for r in records]
    return pd.DataFrame(rows)


def write_raw(paths: RunPaths, records: List[RequestRecord], metadata: dict) -> pd.DataFrame:
    """Persist per-request rows (csv + parquet) and the run metadata."""
    df = records_to_frame(records)
    df.to_csv(paths.requests_csv, index=False)
    try:
        df.to_parquet(paths.requests_parquet, index=False)
    except Exception as exc:  # pragma: no cover - parquet engine missing
        print(f"  [storage] parquet write skipped: {exc}")
    with open(paths.metadata_json, "w", encoding="utf-8") as f:
        json.dump(redact(metadata), f, indent=2, ensure_ascii=False, default=str)
    return df


def checkpoint_run(
    paths: RunPaths,
    records: List[RequestRecord],
    metadata: dict,
    summaries: dict,
) -> pd.DataFrame:
    """Rewrite raw + summary artifacts with all data collected so far.

    Called after each factor combination so partial results are visible on disk
    while a long sweep is still running. Does not support resume-after-interrupt.
    """
    df = write_raw(paths, records, metadata)
    write_summaries(paths, summaries)
    return df


def write_summaries(paths: RunPaths, summaries: dict) -> None:
    """Write overall/benign/adversarial summary JSON files."""
    mapping = {
        "overall": "overall_metrics.json",
        "benign": "benign_metrics.json",
        "adversarial": "adversarial_metrics.json",
    }
    for key, fname in mapping.items():
        if key in summaries:
            with open(paths.summary_dir / fname, "w", encoding="utf-8") as f:
                json.dump(summaries[key], f, indent=2, ensure_ascii=False, default=str)


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "requests.csv").exists()


def resolve_results_root(results_dir: Optional[str | Path] = None) -> Path:
    """Return the results tree root, accepting either ``results/`` or ``results/raw/run_*``."""
    path = results_root(results_dir)
    if is_run_dir(path):
        # .../results/raw/run_<stamp>
        if path.parent.name == RAW_DIRNAME:
            return path.parent.parent
    return path


def list_raw_runs(
    results_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> List[Path]:
    path = results_root(results_dir)
    if is_run_dir(path):
        runs = [path]
    else:
        raw = path / RAW_DIRNAME
        if not raw.exists():
            return []
        runs = sorted(p for p in raw.iterdir() if is_run_dir(p))
    if run_id:
        rid = run_id.strip()
        runs = [p for p in runs if p.name == rid or p.name.endswith(rid)]
    return runs


def load_run_metadata(run_dir: Path) -> dict:
    """Load ``metadata.json`` for a raw run directory."""
    meta_path = run_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def patch_run_metadata(run_dir: Path, patch: dict) -> dict:
    """Merge *patch* into an existing run's metadata and rewrite the file."""
    meta = load_run_metadata(run_dir)
    meta.update(patch)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(redact(meta), f, indent=2, ensure_ascii=False, default=str)
    return meta


def delete_run(
    run_id: str,
    results_dir: Optional[str | Path] = None,
    *,
    analysis_dirs: Optional[Iterable[str]] = None,
    campaign_id: Optional[str] = None,
) -> list[Path]:
    """Remove raw + summary artifacts for *run_id*. Returns deleted paths."""
    root = results_root(results_dir)
    deleted: list[Path] = []
    for sub in (RAW_DIRNAME, SUMMARY_DIRNAME):
        path = root / sub / run_id
        if path.exists():
            shutil.rmtree(path)
            deleted.append(path)
    extra = analysis_dirs or ANALYSIS_DIRNAMES
    analysis_bases = [root]
    if campaign_id:
        analysis_bases = [campaign_analysis_root(root, campaign_id)]
    for base in analysis_bases:
        for sub in extra:
            path = base / sub / run_id
            if path.exists():
                shutil.rmtree(path)
                deleted.append(path)
    return deleted


def merge_runs(
    run_ids: Sequence[str],
    *,
    results_dir: Optional[str | Path] = None,
    name: Optional[str] = None,
) -> RunPaths:
    """Concatenate multiple raw runs into a new run (same schema, adds ``merged_from``)."""
    if len(run_ids) < 2:
        raise ValueError("merge_runs requires at least two run_ids")

    root = results_root(results_dir)
    frames: List[pd.DataFrame] = []
    sources: list[dict] = []
    for rid in run_ids:
        run_dir = root / RAW_DIRNAME / rid
        if not is_run_dir(run_dir):
            raise FileNotFoundError(f"missing raw run: {run_dir}")
        df = load_all_requests(run_dir, run_id=rid)
        if df.empty:
            raise ValueError(f"run {rid!r} has no request rows")
        frames.append(df)
        sources.append({"run_id": rid, "n_rows": int(len(df)), "metadata": load_run_metadata(run_dir)})

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop(columns=["run_id"], errors="ignore")
    paths = new_run(results_dir, name=name)
    base_meta = dict(sources[0]["metadata"])
    base_meta.pop("run_id", None)
    metadata = {
        **base_meta,
        "run_id": paths.run_id,
        "status": "completed",
        "merged_from": [s["run_id"] for s in sources],
        "merge_sources": sources,
        "total_requests": int(len(merged)),
        "num_runs_completed": base_meta.get("num_runs_completed"),
        "num_runs_planned": base_meta.get("num_runs_planned"),
    }
    merged.to_csv(paths.requests_csv, index=False)
    try:
        merged.to_parquet(paths.requests_parquet, index=False)
    except Exception as exc:  # pragma: no cover
        print(f"  [storage] parquet write skipped: {exc}")
    with open(paths.metadata_json, "w", encoding="utf-8") as f:
        json.dump(redact(metadata), f, indent=2, ensure_ascii=False, default=str)
    return paths


def load_all_requests(
    results_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load and concatenate raw per-request rows.

    Adds a ``run_id`` column. Prefers parquet, falling back to csv.
    Pass *run_id* (or a path to ``results/raw/run_*`` as *results_dir*) to
    restrict loading to one experiment run.
    """
    frames: List[pd.DataFrame] = []
    for run_dir in list_raw_runs(results_dir, run_id=run_id):
        parquet = run_dir / "requests.parquet"
        csv = run_dir / "requests.csv"
        if parquet.exists():
            try:
                df = pd.read_parquet(parquet)
            except Exception:
                df = pd.read_csv(csv)
        else:
            df = pd.read_csv(csv)
        df["run_id"] = run_dir.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
