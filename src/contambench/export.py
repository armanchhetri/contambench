"""Write a D2 labeled workload: a replayable CSV plus a manifest.

The benign pool is the suite's benign prompts in suite order, cycled up to ``rows``
rows (or a user-supplied CSV with ``prompt_id, prompt, success_rate``). A fraction
``c`` of rows is replaced by the matching stressor prompt. Stressor rows are chosen
deterministically: among rows whose prompt is in the top-``k`` prompts by Stage-A
success rate, take the ``round(c * rows)`` rows with the highest success rate, with
ties broken by row index. This reproduces the exports shipped in D2.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_pairs(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("prompts") or data.get("entries") or []
        if isinstance(data, dict):
            data = list(data.values())
    return [p for p in data if p.get("prompt") is not None and p.get("adv_prompt") is not None]


def benign_pool(pairs: list[dict], rows: int) -> list[dict]:
    n = len(pairs)
    return [
        {"prompt_id": str(i % n), "prompt": pairs[i % n]["prompt"], "success_rate": str(float(pairs[i % n]["success_rate"]))}
        for i in range(rows)
    ]


def read_benign_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("prompt_id", "prompt", "success_rate"):
        if rows and col not in rows[0]:
            raise ValueError(f"{path} is missing column {col!r}")
    return rows


def contaminate(benign: list[dict], pairs: list[dict], c: float, k: int) -> tuple[list[dict], dict]:
    if not 0.0 <= c <= 1.0:
        raise ValueError("c must be in [0, 1]")
    adv = {p["prompt"]: p["adv_prompt"] for p in pairs}
    missing = {r["prompt"] for r in benign} - adv.keys()
    if missing:
        raise ValueError(f"{len(missing)} benign prompts have no stressor in the suite")
    best: dict[str, float] = {}
    for r in benign:
        best[r["prompt_id"]] = max(best.get(r["prompt_id"], -1.0), float(r["success_rate"]))
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], int(kv[0])))
    eligible = {pid for pid, _ in ranked[:k]}
    idx = [i for i, r in enumerate(benign) if r["prompt_id"] in eligible]
    n_attack = int(round(c * len(benign)))
    if n_attack > len(idx):
        raise ValueError(f"need {n_attack} stressor rows but only {len(idx)} rows belong to the top-{k} prompts")
    idx.sort(key=lambda i: (-float(benign[i]["success_rate"]), i))
    chosen = sorted(idx[:n_attack])
    cs = set(chosen)
    out = [
        {
            "prompt_id": r["prompt_id"],
            "prompt": adv[r["prompt"]] if i in cs else r["prompt"],
            "success_rate": r["success_rate"],
            "is_attack": int(i in cs),
            "benign_prompt": r["prompt"],
            "row_index": i,
        }
        for i, r in enumerate(benign)
    ]
    manifest = {
        "n_rows": len(out),
        "contamination_ratio": c,
        "n_attack": n_attack,
        "n_benign": len(out) - n_attack,
        "realized_contamination_ratio": n_attack / len(out) if out else 0.0,
        "workload_size": k,
        "eligible_unique_prompts": len(eligible),
        "eligible_rows": len(idx),
        "selection": "rows whose prompt_id is among the top-workload_size prompts by Stage-A success_rate; "
        "take the n_attack rows with the highest success_rate, ties broken by row index",
        "attack_row_indices": chosen,
        "attack_prompt_ids": sorted({benign[i]["prompt_id"] for i in chosen}, key=int),
        "distinct_attack_prompts": len({benign[i]["prompt_id"] for i in chosen}),
        "text_column": "prompt",
    }
    return out, manifest


def write(out_csv: Path, rows: list[dict], manifest: dict) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "prompt", "success_rate", "is_attack", "benign_prompt", "row_index"])
        w.writeheader()
        w.writerows(rows)
    man = out_csv.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return man
