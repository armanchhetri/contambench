"""Poll vLLM Prometheus ``/metrics`` and append key gauges to a JSONL log.

Works with RunPod pod proxies (``https://{pod}-8000.proxy.runpod.net/metrics``)
and local vLLM. Auth uses ``--api-key`` or ``$RUNPOD_API_KEY``.

Example
-------
    contam-bench telemetry --endpoint http://HOST:8000 --interval 1 \\
        --out contambench-results/vllm/metrics.jsonl

Run in the background while an experiment is active; stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError("The 'requests' package is required. Install it with 'pip install requests'.") from exc

from contambench.harness.client import default_api_key

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[-+0-9.eE]+|NaN|\+Inf|-Inf)"
    r"(?:\s+(?P<ts>\d+))?$"
)

# Gauges / counters we surface in the JSONL summary (per model).
_KEY_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
)
_KEY_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:num_requests_swapped_total",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_hits_total",
)
_KEY_HISTOGRAMS = (
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:time_per_output_token_seconds",
)
_SUCCESS_COUNTER = "vllm:request_success_total"

DEFAULT_OUT_DIR = Path.cwd() / "contambench-results" / "vllm"
DEFAULT_MAX_HTTP_ERRORS = 20


def _parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        labels[key.strip()] = val.strip().strip('"')
    return labels


def _parse_value(raw: str) -> Optional[float]:
    if raw in ("NaN", "+Inf", "-Inf"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    """Parse exposition text into flat samples: name, labels, value."""
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        val = _parse_value(m.group("value"))
        if val is None:
            continue
        labels = _parse_labels(m.group("labels") or "")
        samples.append({"name": m.group("name"), "labels": labels, "value": val})
    return samples


def resolve_metrics_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/metrics"):
        return base
    return f"{base}/metrics"


def _model_key(labels: dict[str, str]) -> str:
    return labels.get("model_name", labels.get("model", "_default"))


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact per-model summary from raw Prometheus samples."""
    by_model: dict[str, dict[str, Any]] = {}

    def bucket(model: str) -> dict[str, Any]:
        if model not in by_model:
            by_model[model] = {
                "model_name": None if model == "_default" else model,
                "queue": {
                    "requests_running": None,
                    "requests_waiting": None,
                    "requests_swapped": None,
                },
                "gpu_cache_usage_perc": None,
                "counters": {},
                "request_success": {},
                "histograms": {},
            }
        return by_model[model]

    for s in samples:
        name = s["name"]
        labels = s["labels"]
        value = s["value"]
        model = _model_key(labels)

        if name in _KEY_GAUGES:
            b = bucket(model)
            if name == "vllm:num_requests_running":
                b["queue"]["requests_running"] = value
            elif name == "vllm:num_requests_waiting":
                b["queue"]["requests_waiting"] = value
            elif name == "vllm:num_requests_swapped":
                b["queue"]["requests_swapped"] = value
            elif name == "vllm:gpu_cache_usage_perc":
                b["gpu_cache_usage_perc"] = value
            continue

        if name in _KEY_COUNTERS:
            short = name.removeprefix("vllm:").removesuffix("_total")
            bucket(model)["counters"][short] = value
            continue

        if name == _SUCCESS_COUNTER:
            reason = labels.get("finished_reason", "unknown")
            bucket(model)["request_success"][reason] = value
            continue

        for hist in _KEY_HISTOGRAMS:
            if name == f"{hist}_sum":
                bucket(model)["histograms"].setdefault(hist.removeprefix("vllm:"), {})["sum"] = value
            elif name == f"{hist}_count":
                bucket(model)["histograms"].setdefault(hist.removeprefix("vllm:"), {})["count"] = value

    # Derived fields.
    for b in by_model.values():
        queries = b["counters"].get("gpu_prefix_cache_queries")
        hits = b["counters"].get("gpu_prefix_cache_hits")
        if queries and queries > 0 and hits is not None:
            b["prefix_cache_hit_rate"] = hits / queries
        else:
            b["prefix_cache_hit_rate"] = None

        for hist_name, parts in b["histograms"].items():
            cnt = parts.get("count")
            total = parts.get("sum")
            if cnt and cnt > 0 and total is not None:
                parts["mean"] = total / cnt
            else:
                parts["mean"] = None

    if len(by_model) == 1:
        return next(iter(by_model.values()))
    return {"models": list(by_model.values())}


def fetch_metrics(
    url: str,
    *,
    api_key: Optional[str] = None,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    headers: dict[str, str] = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return parse_prometheus(resp.text)


def verify_metrics_endpoint(
    endpoint: str,
    *,
    api_key: Optional[str] = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Probe ``/metrics`` once; raise on network/auth/parse failure."""
    url = resolve_metrics_url(endpoint)
    samples = fetch_metrics(url, api_key=api_key, timeout_s=timeout_s)
    if not samples:
        raise RuntimeError(f"vLLM /metrics at {url} returned no parseable samples")
    summary = summarize_samples(samples)
    return summary


def build_record(
    endpoint: str,
    summary: dict[str, Any],
    *,
    scrape_duration_s: Optional[float] = None,
    http_status: int = 200,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "timestamp": now.isoformat(),
        "timestamp_unix": now.timestamp(),
        "endpoint": endpoint,
        "http_status": http_status,
        "scrape_duration_s": scrape_duration_s,
        "metrics": summary,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _metrics_root(summary: dict[str, Any]) -> dict[str, Any]:
    """Single-model summary, or first model when multiple are present."""
    if "models" in summary:
        models = summary.get("models") or []
        return models[0] if models else {}
    return summary


def format_status_line(summary: dict[str, Any]) -> str:
    """Compact one-line status for console logging."""
    m = _metrics_root(summary)
    q = m.get("queue") or {}
    counters = m.get("counters") or {}
    parts = [
        f"running={q.get('requests_running')}",
        f"waiting={q.get('requests_waiting')}",
        f"swapped={q.get('requests_swapped')}",
        f"preemptions={counters.get('num_preemptions')}",
    ]
    gpu = m.get("gpu_cache_usage_perc")
    if gpu is not None:
        parts.append(f"gpu_cache={gpu:.3f}" if isinstance(gpu, float) else f"gpu_cache={gpu}")
    return " ".join(parts)


def run_once(args: argparse.Namespace) -> int:
    url = resolve_metrics_url(args.endpoint)
    api_key = args.api_key or default_api_key()
    out_path = Path(args.out)
    t0 = time.perf_counter()
    try:
        samples = fetch_metrics(url, api_key=api_key, timeout_s=args.timeout)
        summary = summarize_samples(samples)
        record = build_record(args.endpoint, summary, scrape_duration_s=time.perf_counter() - t0)
        append_jsonl(out_path, record)
        print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
        print(f"\nWrote 1 record → {out_path}  {format_status_line(summary)}")
        return 0
    except requests.RequestException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run_loop(args: argparse.Namespace) -> int:
    url = resolve_metrics_url(args.endpoint)
    api_key = args.api_key or default_api_key()
    out_path = Path(args.out)
    stop = False

    def _handle_sigint(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"Polling {url}")
    print(f"Logging to {out_path} every {args.interval:g}s (Ctrl+C to stop)")
    if args.duration_s:
        print(f"Will stop after {args.duration_s:g}s")

    started = time.time()
    n_ok = 0
    n_err = 0
    consecutive_err = 0
    max_err = int(args.max_http_errors)

    while not stop:
        t0 = time.perf_counter()
        try:
            samples = fetch_metrics(url, api_key=api_key, timeout_s=args.timeout)
            summary = summarize_samples(samples)
            record = build_record(
                args.endpoint,
                summary,
                scrape_duration_s=time.perf_counter() - t0,
            )
            append_jsonl(out_path, record)
            n_ok += 1
            consecutive_err = 0

            if args.verbose:
                print(f"  {record['timestamp']} {format_status_line(summary)}")
            elif n_ok == 1 or n_ok % args.print_every == 0:
                print(f"  [{n_ok}] {format_status_line(summary)}")
        except requests.RequestException as exc:
            n_err += 1
            consecutive_err += 1
            print(f"  [error {consecutive_err}/{max_err}] {exc}", file=sys.stderr)
            if consecutive_err >= max_err:
                print(
                    f"Stopping after {max_err} consecutive HTTP errors.",
                    file=sys.stderr,
                )
                break

        if args.duration_s and (time.time() - started) >= args.duration_s:
            break
        if stop:
            break
        time.sleep(max(0.0, args.interval))

    stopped_on_errors = consecutive_err >= max_err
    print(f"Done. {n_ok} ok, {n_err} errors → {out_path}")
    if stopped_on_errors:
        return 1
    if n_ok == 0:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT", "").strip() or None,
        help="vLLM base URL (pod proxy or http://localhost:8000); /metrics appended if missing",
    )
    p.add_argument("--api-key", default=None, help="Bearer token (default: $RUNPOD_API_KEY)")
    p.add_argument(
        "--out",
        default=None,
        help=f"JSONL output path (default: {DEFAULT_OUT_DIR}/vllm_metrics_<utc>.jsonl)",
    )
    p.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds (default 1)")
    p.add_argument("--duration-s", type=float, default=None, help="Stop after this many seconds")
    p.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout per scrape")
    p.add_argument("-v", "--verbose", action="store_true", help="Print every sample")
    p.add_argument("--print-every", type=int, default=10, help="Status line every N ok scrapes")
    p.add_argument(
        "--max-http-errors",
        type=int,
        default=DEFAULT_MAX_HTTP_ERRORS,
        help=f"Stop after this many consecutive HTTP errors (default {DEFAULT_MAX_HTTP_ERRORS})",
    )
    p.add_argument("--once", action="store_true", help="Fetch once and exit (no loop)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.endpoint:
        print("error: pass --endpoint or set VLLM_ENDPOINT", file=sys.stderr)
        return 2

    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.out = str(DEFAULT_OUT_DIR / f"vllm_metrics_{stamp}.jsonl")

    if args.once:
        return run_once(args)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
