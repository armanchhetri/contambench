"""Turn generation results into a D1 pairs file. CPU only, no torch.

``summarize_loopllm_results`` is the research repo's ``_summarize_loopllm_results``
(``src/sensitivity_analysis.py``): for each ``res_<i>.json`` it keeps the evaluated
step with the highest Stage-A ``success_rate`` (the first one on ties). It also carries
``repet`` and ``average_repet``, as in the released ``summary.json`` files.

``write_pairs`` writes ``{"format", "provenance", "prompts": [...], "average_*"}``.
The ``prompts`` rows have the D1 keys (``pair_index, prompt, adv_suffix, adv_prompt,
answer, success_rate, avg_len, repet``), so ``contam-bench export --pairs FILE`` and the
harness ``attack_file`` option read the file directly.
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT = "contambench.d1.pairs/1"
PAIR_KEYS = ("prompt", "adv_suffix", "adv_prompt", "answer", "success_rate", "avg_len", "repet")


def _res_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def summarize_loopllm_results(results_dir: Path) -> dict:
    """Best evaluated step (highest success rate) for each LoopLLM prompt."""
    prompt_summaries = []
    files = [p for p in Path(results_dir).glob("res_*.json") if p.stem.split("_")[-1].isdigit()]
    for res_path in sorted(files, key=_res_index):
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)
        best_entry = None
        best_success_rate = None
        for step in res.keys():
            if step == "-1" or not str(step).lstrip("-").isdigit():
                continue
            candidate = res[step]
            if "success_rate" not in candidate or "avg_len" not in candidate:
                continue
            success_rate = candidate["success_rate"]
            if best_success_rate is None or success_rate > best_success_rate:
                best_success_rate = success_rate
                best_entry = candidate
        if best_entry is None:
            continue
        prompt_summaries.append({k: best_entry.get(k) for k in PAIR_KEYS})
    return with_averages(prompt_summaries)


def with_averages(prompts: list[dict]) -> dict:
    if not prompts:
        return {"prompts": [], "average_asr": None, "average_len": None, "average_repet": None}
    n = len(prompts)
    reps = [p["repet"] for p in prompts if p.get("repet") is not None]
    return {
        "prompts": prompts,
        "average_asr": sum(p["success_rate"] for p in prompts) / n,
        "average_len": sum(p["avg_len"] for p in prompts) / n,
        "average_repet": sum(reps) / len(reps) if reps else None,
    }


def write_pairs(path: Path, summary: dict, provenance: dict) -> Path:
    """Write a D1 pairs file with a provenance block; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"pair_index": i, **{k: p.get(k) for k in PAIR_KEYS}} for i, p in enumerate(summary["prompts"])]
    doc = {
        "format": FORMAT,
        "provenance": provenance,
        "pairs": len(rows),
        "average_asr": summary.get("average_asr"),
        "average_len": summary.get("average_len"),
        "average_repet": summary.get("average_repet"),
        "prompts": rows,
    }
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def resolve_out(out: str | Path) -> tuple[Path, Path]:
    """``(pairs_file, work_dir)`` for ``--out``: a ``.json`` file or a directory."""
    out = Path(out).expanduser()
    if out.suffix.lower() == ".json":
        return out, out.with_name(out.stem + ".work")
    return out / "pairs.json", out / "work"
