"""Convert a GuideLLM benchmark report into D3-format request rows.

GuideLLM (https://github.com/vllm-project/guidellm, Apache-2.0) is used as published:
run ``guidellm run`` with an exported ContamBench workload as its dataset file. This
module reads the ``benchmarks.json`` it writes, labels each request by matching its
prompt text against the export (stressor rows carry the stressor text), and writes the
same columns as the harness, so ``contam-bench score`` applies unchanged.

Field mapping from GuideLLM request records:
  request_latency (s)        -> latency_ms
  time_to_first_token_ms     -> ttft_ms
  inter_token_latency_ms     -> tbt_ms   (mean gap between output tokens)
  output_tokens              -> output_tokens
  output_tokens >= max_tokens -> finish_reason "length", else "stop"  (GuideLLM does not
                                 report a finish reason; this is the cap-hit rule)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def _user_text(request_args) -> str:
    if isinstance(request_args, str):
        try:
            request_args = json.loads(request_args)
        except json.JSONDecodeError:
            return ""
    body = (request_args or {}).get("body") or {}
    msgs = body.get("messages") or []
    if msgs:
        content = msgs[-1].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                (i.get("text") or "") if isinstance(i, dict) else str(i) for i in content
                if not isinstance(i, dict) or i.get("type") in (None, "text")
            )
    prompt = body.get("prompt")
    return prompt if isinstance(prompt, str) else ""


def load_requests(benchmarks: Path) -> tuple[list[dict], dict]:
    """Request records of the first benchmark in a GuideLLM report, plus its strategy."""
    data = json.loads(Path(benchmarks).read_text(encoding="utf-8"))
    benches = data.get("benchmarks") or []
    if not benches:
        raise ValueError(f"no benchmarks in {benchmarks}")
    if len(benches) > 1:
        raise ValueError(f"{benchmarks} holds {len(benches)} benchmarks; run one rate per report")
    b = benches[0]
    rows = []
    for status in ("successful", "errored", "incomplete"):
        for r in (b.get("requests") or {}).get(status) or []:
            rows.append({"status": status, **r})
    strategy = ((b.get("config") or {}).get("strategy")) or {}
    return rows, strategy


def load_labels(export_csv: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Map prompt text -> export row, separately for stressor and benign rows."""
    stress, benign = {}, {}
    with Path(export_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            (stress if str(row.get("is_attack", "0")) in ("1", "True", "true") else benign)[row["prompt"]] = row
    return stress, benign


def to_frame(benchmarks: Path, export_csv: Path, *, max_tokens: int, contamination_ratio: float | None = None,
             arrival_rate: float | None = None, trial: int = 0, model: str | None = None) -> tuple[pd.DataFrame, dict]:
    reqs, strategy = load_requests(benchmarks)
    stress, benign = load_labels(export_csv)
    if contamination_ratio is None:
        man = Path(export_csv).with_suffix(".manifest.json")
        if man.exists():
            contamination_ratio = float(json.loads(man.read_text())["contamination_ratio"])
        else:
            with Path(export_csv).open(newline="", encoding="utf-8") as fh:
                flags = [str(r.get("is_attack", "0")) in ("1", "True", "true") for r in csv.DictReader(fh)]
            contamination_ratio = round(sum(flags) / max(len(flags), 1), 4)
    rate = float(arrival_rate if arrival_rate is not None else strategy.get("rate") or 0.0)
    out, unmatched = [], 0
    for r in reqs:
        text = _user_text(r.get("request_args"))
        if text in stress:
            ptype, src = "attack", stress[text]
        elif text in benign:
            ptype, src = "benign", benign[text]
        else:
            ptype, src, unmatched = "unlabeled", {}, unmatched + 1
        lat = r.get("request_latency")
        ttft = r.get("time_to_first_token_ms")
        out_tok = r.get("output_tokens")
        out.append({
            "request_id": r.get("request_id"),
            "client_id": 0,
            "trial": trial,
            "config_id": f"guidellm_c{contamination_ratio:g}_r{rate:g}",
            "timestamp_start": r.get("request_start_time"),
            "timestamp_first_token": None,
            "timestamp_end": r.get("request_end_time"),
            "prompt_type": ptype,
            "contamination_ratio": contamination_ratio,
            "arrival_rate": rate,
            "arrival_model": strategy.get("type_", "guidellm"),
            "num_clients": 1,
            "success": r["status"] == "successful",
            "http_status": None,
            "input_tokens": r.get("prompt_tokens"),
            "output_tokens": out_tok,
            "ttft_ms": None if ttft is None else float(ttft),
            "tbt_ms": None if r.get("inter_token_latency_ms") is None else float(r["inter_token_latency_ms"]),
            "latency_ms": None if lat is None else float(lat) * 1000.0,
            "finish_reason": None if out_tok is None else ("length" if out_tok >= max_tokens else "stop"),
            "error": None if r["status"] == "successful" else r["status"],
            "pair_index": int(src["prompt_id"]) if src.get("prompt_id") not in (None, "") else None,
            "prompt_id": src.get("prompt_id"),
            "model": model,
            "input_text": text,
            "output_text": (r.get("output") or "")[:2000],
            "decode_residency_ms": None if lat is None or ttft is None else float(lat) * 1000.0 - float(ttft),
            "max_tokens": max_tokens,
        })
    df = pd.DataFrame(out)
    info = {
        "source": "guidellm",
        "benchmarks": str(benchmarks),
        "labels": str(export_csv),
        "strategy": strategy,
        "requests": len(df),
        "successful": int(df.success.sum()) if len(df) else 0,
        "stressor_requests": int((df.prompt_type == "attack").sum()) if len(df) else 0,
        "unlabeled_requests": unmatched,
        "contamination_ratio": contamination_ratio,
        "arrival_rate": rate,
        "max_tokens": max_tokens,
        "finish_reason_rule": "length when output_tokens >= max_tokens",
    }
    return df, info


def write(df: pd.DataFrame, info: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "requests.parquet", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({"config": info}, indent=2, default=str) + "\n")
    return out_dir
