"""Reproduce the paper's tables from the fetched data record (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contambench import data, score

# Campaign, label, and the high operating point used for the paper's sweep table.
HIGH_POINT = [
    ("aws_llama3b", "Llama-3.2-3B", 0.27),
    ("aws_qwen7b", "Qwen2.5-7B", 0.17),
    ("runpod_llama8b", "Llama-3.1-8B", 0.45),
    ("aws_mistral7b", "Mistral-7B", 0.28),
]


# Suites whose Stage-A run used greedy decoding (no sampling in the model's default
# generation config), so their 16 Stage-A samples are identical; no Stage-A rate is reported.
GREEDY_STAGE_A = {"mistral-7b-instruct-v0.3", "phi-4-mini-instruct", "granite-4.1-8b",
                  "seed-coder-8b-instruct", "glm-4-9b-0414"}


def table_suites(base: Path | None = None, top_k: int = 50) -> list[dict]:
    d1 = data.require("d1", base)
    rows = []
    for slug in data.SUITES:
        pairs = json.loads((d1 / "suites" / slug / "pairs.json").read_text())
        suite = json.loads((d1 / "suites" / slug / "suite.json").read_text())
        sr = np.asarray([p["success_rate"] for p in pairs], dtype=float)
        ln = np.asarray([p["avg_len"] for p in pairs], dtype=float)
        top = np.argsort(-sr, kind="stable")[:top_k]
        sb = d1 / "suites" / slug / "stage_b.json"
        sbj = json.loads(sb.read_text()) if sb.exists() else {}
        rows.append({
            "target": suite["model"], "pairs": len(pairs), "L_max": suite["stage_a"]["generation_cap_tokens"],
            "stage_a_top50": None if slug in GREEDY_STAGE_A else float(sr[top].mean()),
            "stage_a_top50_mean_len": None if slug in GREEDY_STAGE_A else float(ln[top].mean()),
            "stage_b": sbj.get("stressor_cap_hit"), "stressor_mean_out": sbj.get("stressor_mean_out"),
            "benign_mean_out": sbj.get("benign_mean_out"),
        })
    return rows


def table_sweep(base: Path | None = None) -> list[dict]:
    d3 = data.require("d3", base)
    out = []
    for campaign, label, lam in HIGH_POINT:
        df = score.load(d3 / "runs" / campaign / "rq2_sweep")
        out.append({"deployment": label, "campaign": campaign, **score.sweep(df, lam)})
    return out


def table_guidellm(base: Path | None = None) -> dict:
    """The two GuideLLM runs, imported from their benchmarks.json and scored against the
    Llama-3.2-3B campaign budgets at lambda = 0.27 (the same import any user runs)."""
    from contambench import guidellm_import

    d2, d3 = data.require("d2", base), data.require("d3", base)
    exports = d2 / "exports" / "llama-3.2-3b-instruct"
    sweep = score.load(d3 / "runs" / "aws_llama3b" / "rq2_sweep")
    b = score.budgets(sweep[np.isclose(sweep.arrival_rate, 0.27)])
    out = {}
    for key, label, csv_name, c in (("guidellm_benign", "benign", "benign_tiled1000.csv", 0.0),
                                    ("guidellm_c0p10", "c0p10", "c0p10_tiled1000.csv", 0.10)):
        df, info = guidellm_import.to_frame(d3 / "guidellm" / label / "benchmarks.json", exports / csv_name,
                                            max_tokens=4096, contamination_ratio=c)
        ok = df[df.success]
        adv, ben = ok[ok.prompt_type == "attack"], ok[ok.prompt_type == "benign"]
        out[key] = {
            "p95_latency_ms": float(np.percentile(ok.latency_ms, 95)),
            "p95_ttft_ms": float(np.percentile(ok.ttft_ms.dropna(), 95)),
            "p95_tbt_ms": float(np.percentile(ok.tbt_ms.dropna(), 95)),
            "slo_violation_rate": float(score.violates(ok, b).mean()),
            "n_completed": int(len(ok)),
            "n_stressor": int(len(adv)),
            "unlabeled": info["unlabeled_requests"],
            "stressor_slo_violation": float(score.violates(adv, b).mean()) if len(adv) else None,
            "benign_slo_violation": float(score.violates(ben, b).mean()) if len(ben) else None,
            "stressor_p95_latency_ms": float(np.percentile(adv.latency_ms, 95)) if len(adv) else None,
            "benign_p95_latency_ms": float(np.percentile(ben.latency_ms, 95)) if len(ben) else None,
        }
    for key in ("ours_c0", "ours_c0p10"):
        c = 0.0 if key == "ours_c0" else 0.10
        r = next(x for x in score.sweep(sweep, 0.27)["rows"] if abs(x["c"] - c) < 1e-9)
        out[key] = {"slo_violation_rate": r["slo_violation"], "p95_latency_ms": r["p95_latency_s"] * 1000}
    th = json.loads((d3 / "guidellm" / "c0p10" / "analysis" / "head2head_threeway.json").read_text())["runs"]
    for key in ("ours_c0", "ours_c0p10"):
        out[key]["p95_ttft_ms"] = th[key]["ttft_p95_ms"]
        out[key]["p95_tbt_ms"] = th[key]["tbt_p95_ms"]
    return out


def check_against_paper(sweeps: list[dict], base: Path | None = None) -> list[str]:
    """Compare recomputed sweep rows with the values printed in the paper."""
    ref = json.loads((data.require("d3", base) / "paper" / "tables.json").read_text())["sweeps"]
    diffs = []
    for got, exp in zip(sweeps, ref):
        for a, b in zip(got["rows"], exp["rows"]):
            for k in ("slo_violation", "p95_latency_s", "benign_slo_violation"):
                if abs(a[k] - b[k]) > 1e-9:
                    diffs.append(f"{got['deployment']} c={a['c']}: {k} {a[k]} != {b[k]}")
    return diffs


def render(base: Path | None = None, check: bool = True) -> str:
    lines = ["Table I. Released LoopLLM stressor suites (D1)",
             f"{'target':42} {'pairs':>5} {'L_max':>6} {'StageA':>7} {'StageB':>7} {'stress out':>10} {'benign out':>10}"]
    for r in table_suites(base):
        sb = "-" if r["stage_b"] is None else f"{r['stage_b']:.2f}"
        so = "-" if r["stressor_mean_out"] is None else f"{r['stressor_mean_out']:.0f}"
        bo = "-" if r["benign_mean_out"] is None else f"{r['benign_mean_out']:.0f}"
        lines.append(f"{r['target']:42} {r['pairs']:>5} {r['L_max']:>6} {'---' if r['stage_a_top50'] is None else format(r['stage_a_top50'], '.2f'):>7} {sb:>7} {so:>10} {bo:>10}")
    sweeps = table_sweep(base)
    lines += ["", "Table III. Contamination sweep at the high operating point (D3)"]
    for s in sweeps:
        lines += ["", s["deployment"], score.format_sweep(s)]
    g = table_guidellm(base)
    lines += ["", "Table IV. Reuse of a D2 workload in GuideLLM (Llama-3.2-3B)"]
    for key, label in (("guidellm_benign", "GuideLLM, benign pool"), ("guidellm_c0p10", "GuideLLM, D2 at c=0.10"),
                       ("ours_c0", "harness, c=0"), ("ours_c0p10", "harness, c=0.10")):
        r = g[key]
        lat = r.get("p95_latency_ms", r.get("latency_p95_ms"))
        ttft = r.get("p95_ttft_ms", r.get("ttft_p95_ms"))
        tbt = r.get("p95_tbt_ms", r.get("tbt_p95_ms"))
        viol = r.get("slo_violation_rate")
        v = "" if viol is None else f"{100 * viol:5.1f}%"
        lines.append(f"{label:26} p95 {lat / 1000:6.1f} s   TTFT {ttft:5.0f} ms   TBT {tbt:5.1f} ms   {v}")
    if check:
        diffs = check_against_paper(sweeps, base)
        lines += ["", "check against paper values: " + ("identical" if not diffs else f"{len(diffs)} differences")]
        lines += diffs[:10]
    return "\n".join(lines)
