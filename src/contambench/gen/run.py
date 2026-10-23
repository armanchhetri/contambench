"""``contam-bench generate``: run LoopLLM or Engorgio against one model and write D1 pairs.

LoopLLM follows the research repo's ``main.py --task attack --attack_method loopllm``
(``loopllm_attack_run``): cycle loss, cyclic segment length 5, 30 suffix tokens,
output limit from ``modelsinfo.json``, Alpaca prompts, 100 steps, seed 23. Results are
written per prompt to ``<work>/res_<i>.json`` (a rerun resumes from the last one), then
summarized into the pairs file.

Engorgio follows one cell of ``max_len_trigg_len_grid_search`` (3000 iterations,
``log_interval`` 100, loss version 1, ``alpha`` 0.5, max length = output limit,
trigger length 32), and keeps the attacker's 30 best candidates in
``<work>/engorgio_raw.json`` in the research format. Engorgio triggers are
prompt-agnostic, so to obtain D1 pairs the best triggers (by mean response length) are
appended round-robin to the benign prompts, and every pair gets the same Stage-A
measurement as LoopLLM (16 sampled decodes with the user-only chat template; success
when the sequence reaches the generation cap minus 5 tokens).
"""
from __future__ import annotations

from contambench.gen import require_extra

require_extra()

import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from transformers import set_seed  # noqa: E402

from contambench import __version__  # noqa: E402
from contambench.gen.models import get_model  # noqa: E402
from contambench.gen.modelsinfo import get_modelinfo  # noqa: E402
from contambench.gen.summary import resolve_out, with_averages, write_pairs, summarize_loopllm_results  # noqa: E402

LOOPLLM_STEPS = 100
ENGORGIO_STEPS = 3000
ENGORGIO_KEEP = 30        # attacker.get_attack_prompts(30), as in the grid search
ENGORGIO_TRIGGERS = 5     # best triggers used to build pairs
STAGE_A_SAMPLES = 16      # test_suffix(sample_times=16)


def _versions() -> dict:
    import torch
    import transformers

    v = {"contambench": __version__, "torch": torch.__version__, "transformers": transformers.__version__,
         "python": platform.python_version()}
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name(0)
    return v


def _provenance(method, model, *, suffix_tokens, seed, steps, prompts, n_prompts, quantize, limit, extra,
                temperature) -> dict:
    return {
        "generator": method,
        "model": model.get_name(),
        "suffix_tokens": suffix_tokens,
        "seed": seed,
        "steps": steps,
        "benign_source": prompts,
        "dataset_length": n_prompts,
        "quantize": quantize,
        "output_limit_tokens": limit,
        "stage_a": {
            "samples_per_pair": STAGE_A_SAMPLES,
            "chat_template": "user message only, add_generation_prompt=True",
            "decoding": f"sampled, temperature {temperature} (do_sample forced on; other sampling settings from the checkpoint)",
            "generation_cap_tokens": limit,
            "success_rule": "prompt + output tokens >= generation cap - 5",
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": _versions(),
        **extra,
    }


def run_loopllm(model, work: Path, *, prompts, n_prompts, steps, seed, gpu_threshold, temperature) -> tuple[dict, dict]:
    from contambench.gen.loopllm.attack import LoopLLMAttacker, LoopLLMAttackParams
    from contambench.gen.loopllm.utils import DEFAULT_CYCLIC_SEGMENT_LEN, DEFAULT_LOSS_CONFIG

    params = LoopLLMAttackParams(
        results_dir=str(work),
        loss_config=DEFAULT_LOSS_CONFIG,
        cyclic_segment_len=DEFAULT_CYCLIC_SEGMENT_LEN,
        max_length=model.get_response_limit(),
        data_name=prompts,
        dataset_length=n_prompts,
        steps=steps,
        seed=seed,
        evaluate_collected=False,
        temperature=temperature,
    )
    attacker = LoopLLMAttacker(model)
    attacker.clear_attack()
    attacker.attack(params, log_prompts=True, gpu_threshold=gpu_threshold)
    summary = summarize_loopllm_results(work)
    extra = {"loss_config": params.loss_config, "cyclic_segment_len": params.cyclic_segment_len,
             "topk": params.topk, "num_candidate": params.num_candidate,
             "req_success_rate": params.req_success_rate, "results_dir": str(work)}
    return summary, {"suffix_tokens": params.adv_len, **extra}


def run_engorgio(model, work: Path, *, prompts, n_prompts, steps, seed, gpu_threshold, trigger_len,
                 cpuoffload) -> tuple[dict, dict]:
    from contambench.gen.engorgio.attack import EngorgioAttacker, EngorgioAttackParams
    from contambench.gen.loopllm.utils import get_chat_prompt, normalized_lz_complexity, read_data, test_suffix

    work.mkdir(parents=True, exist_ok=True)
    max_len = model.get_response_limit()
    params = EngorgioAttackParams(
        num_iters=steps,
        log_interval=max(1, min(100, steps // 20)),
        max_length=max_len,
        trigger_token_length=trigger_len,
        loss_version=1,
        cpuoffload=cpuoffload,
        alpha=0.5,
    )
    set_seed(seed)
    attacker = EngorgioAttacker(model)
    attacker.clear_attack()
    attacker.attack(params, log_prompts=True, gpu_threshold=gpu_threshold)
    raw = attacker.get_attack_prompts(ENGORGIO_KEEP)
    (work / "engorgio_raw.json").write_text(json.dumps(raw, ensure_ascii=False, default=float), encoding="utf-8")
    triggers = [r["trigger_text"].strip() for r in raw if (r.get("trigger_text") or "").strip()][:ENGORGIO_TRIGGERS]
    if not triggers:
        raise RuntimeError("Engorgio produced no candidate triggers (converged before the first log interval); "
                           "increase --steps")

    # Stage A on the D1 pairs, with LoopLLM's rule.
    set_seed(seed)
    benign = read_data(prompts, length=n_prompts)
    hf, tok = model.hf_model, model.tokenizer
    hf.generation_config.max_length = max_len
    rows = []
    for i, prompt in enumerate(benign):
        suffix = " " + triggers[i % len(triggers)]
        adv_prompt = f"{prompt} {suffix.strip()}"
        ids = get_chat_prompt(tok, adv_prompt, add_generation_prompt=True, return_tensors="pt")
        success_rate, avg_len, answer = test_suffix(hf, tok, ids, batch=8, sample_times=STAGE_A_SAMPLES)
        print(f"[stage A {i + 1}/{len(benign)}] success_rate={success_rate:.3f} avg_len={avg_len:.1f}")
        rows.append({"prompt": prompt, "adv_suffix": suffix, "adv_prompt": adv_prompt, "answer": answer,
                     "success_rate": success_rate, "avg_len": avg_len, "repet": normalized_lz_complexity(answer)})
    extra = {"trigger_len": trigger_len, "engorgio_iters": steps, "loss_version": params.loss_version,
             "triggers_used": len(triggers), "pairing": "trigger appended round-robin to benign prompts",
             "raw_candidates": str(work / "engorgio_raw.json")}
    return with_averages(rows), {"suffix_tokens": trigger_len, **extra}


def generate(method: str, model_name: str, out: str | Path, *, prompts: str = "alpaca", n_prompts: int = 195,
             steps: int | None = None, seed: int = 23, quantize: int | None = None, max_length: int | None = None,
             gpu_threshold: int = 85, trigger_len: int = 32, cpuoffload: bool = False,
             model_info: str | None = None, temperature: float = 0.7) -> Path:
    import torch

    if not torch.cuda.is_available():
        print("warning: no CUDA device found; generation on a CPU is impractically slow")
    pairs_file, work = resolve_out(out)
    work.mkdir(parents=True, exist_ok=True)
    if prompts.endswith((".json", ".jsonl")):
        prompts = str(Path(prompts).expanduser().resolve())
    info = get_modelinfo(model_name, model_info, limit=max_length)
    model = get_model(info, quantize=quantize)
    model.print_details()
    # Stage A always samples at `temperature`, also for checkpoints whose generation_config ships no
    # sampling settings (they would otherwise decode greedily and give 16 identical samples).
    model.hf_model.generation_config.do_sample = True
    model.hf_model.generation_config.temperature = temperature
    if method == "loopllm":
        steps = steps or LOOPLLM_STEPS
        summary, extra = run_loopllm(model, work, prompts=prompts, n_prompts=n_prompts, steps=steps, seed=seed,
                                     gpu_threshold=gpu_threshold, temperature=temperature)
    elif method == "engorgio":
        steps = steps or ENGORGIO_STEPS
        summary, extra = run_engorgio(model, work, prompts=prompts, n_prompts=n_prompts, steps=steps, seed=seed,
                                      gpu_threshold=gpu_threshold, trigger_len=trigger_len, cpuoffload=cpuoffload)
    else:
        raise ValueError(f"unknown method {method!r}")
    suffix_tokens = extra.pop("suffix_tokens")
    prov = _provenance(method, model, suffix_tokens=suffix_tokens, seed=seed, steps=steps, prompts=prompts,
                       n_prompts=n_prompts, quantize=quantize, limit=model.get_response_limit(), extra=extra,
                       temperature=temperature)
    write_pairs(pairs_file, summary, prov)
    n = len(summary["prompts"])
    asr = summary["average_asr"]
    print(f"wrote {pairs_file}: {n} pairs" + (f", mean Stage-A success rate {asr:.3f}" if asr is not None else ""))
    return pairs_file
