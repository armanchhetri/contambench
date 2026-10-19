"""Experiment runner for contamination-aware stress testing.

Sweeps the experimental factors (contamination ratio ``c``, concurrent clients
``N``, arrival rate ``lambda``) across ``num_trials`` repetitions, drives the
multi-client open-loop workload against an OpenAI-compatible vLLM endpoint
(RunPod by default), and writes the structured raw + summary artifacts.

Examples
--------
Contamination sweep from a config file::

    contam-bench run --config sweep.yaml --endpoint http://HOST:8000

Offline pipeline test (no endpoint needed)::

    contam-bench run --simulate --contamination-ratio 0,0.1 --requests-per-client 5
"""

from __future__ import annotations

import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Optional


import argparse  # noqa: E402

from contambench.harness.config import ExperimentConfig, config_from_dict, load_config  # noqa: E402
from contambench.harness.dataset import load_workload_pairs, mean_success_rate, resolve_attack_file  # noqa: E402
from contambench.harness.metrics import generate_summaries  # noqa: E402
from contambench.harness.validation import compute_validation_summary, format_validation_report  # noqa: E402
from contambench.harness.storage import checkpoint_run, new_run, records_to_frame  # noqa: E402
from contambench.harness.workload import RunSpec, WorkloadInterrupted, _default_max_workers, run_workload  # noqa: E402


def _build_real_client_factory(cfg: ExperimentConfig, *, max_tokens: Optional[int] = None):
    from contambench.harness.client import ChatClient, default_api_key, resolve_backend

    endpoint = cfg.resolved_endpoint()
    backend = resolve_backend(endpoint, cfg.backend)
    if cfg.model is not None:
        model = cfg.model
    else:
        # Omit model in the request body unless explicitly set; vLLM uses the
        # deployed/served model (required when the endpoint is not 3.2-3B).
        model = None
    api_key = cfg.api_key or default_api_key()
    mt = cfg.max_tokens if max_tokens is None else max_tokens

    def factory(client_id: int):
        return ChatClient(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            backend=backend,
            timeout_s=cfg.timeout_s,
            max_tokens=mt,
            temperature=cfg.temperature,
        )

    return factory, endpoint, backend, model


def _build_sim_client_factory(cfg: ExperimentConfig, spec: RunSpec):
    from contambench.harness.simulate import MockChatClient

    def factory(client_id: int):
        return MockChatClient(
            model=cfg.model or "mock-model",
            max_tokens=spec.max_tokens if spec.max_tokens is not None else cfg.max_tokens,
            temperature=cfg.temperature,
            contamination_ratio=spec.contamination_ratio,
            num_clients=spec.num_clients,
            arrival_rate=spec.arrival_rate,
            seed=cfg.seed + client_id + spec.trial * 7919,
        )

    return factory


def _resolve_attack_file(cfg: ExperimentConfig) -> Path:
    """``attack_file`` may be a path, or ``contambench:<path inside the data record>``."""
    if not cfg.attack_file:
        return resolve_attack_file(cfg.attack_model)
    if str(cfg.attack_file).startswith("contambench:"):
        from contambench import data

        return data.root() / str(cfg.attack_file).split(":", 1)[1]
    return Path(cfg.attack_file)


def run_experiment(cfg: ExperimentConfig, verbose: bool = True) -> Path:
    cfg.validate()
    attack_file = _resolve_attack_file(cfg)
    pairs, all_pairs = load_workload_pairs(attack_file, cfg.workload_size)

    paths = new_run(cfg.results_dir, name=cfg.name)
    if verbose:
        print(f"Run id: {paths.run_id}")
        print(f"Raw dir: {paths.raw_dir}")
        if cfg.workload_size:
            print(
                f"Attack file: {attack_file} "
                f"(workload_size={cfg.workload_size}: {len(pairs)}/{len(all_pairs)} prompt pairs)"
            )
        else:
            print(f"Attack file: {attack_file} ({len(pairs)} prompt pairs)")
        avg_sr = mean_success_rate(pairs)
        if avg_sr is not None:
            print(f"Selected workload avg success_rate: {avg_sr:.4f}")

    real_factory = None
    endpoint = backend = model = None

    combos = list(
        product(
            cfg.contamination_ratios,
            cfg.client_counts,
            cfg.arrival_rates,
            range(cfg.num_trials),
            cfg.max_tokens_values,
        )
    )
    if verbose:
        print(
            f"Sweep: {len(combos)} runs "
            f"(c={cfg.contamination_ratios} N={cfg.client_counts} "
            f"lambda={cfg.arrival_rates} max_tokens={cfg.max_tokens_values} "
            f"trials={cfg.num_trials})\n"
        )

    all_records = []
    per_config = []
    metadata = {
        "run_id": paths.run_id,
        "status": "in_progress",
        "config": cfg.to_metadata(),
        "endpoint": endpoint,
        "backend": backend,
        "request_model": model,
        "attack_file": str(attack_file),
        "workload_size": cfg.workload_size,
        "num_prompt_pairs_total": len(all_pairs),
        "num_prompt_pairs": len(pairs),
        "num_runs_planned": len(combos),
        "num_runs_completed": 0,
        "total_requests": 0,
        "per_config": per_config,
    }

    interrupted = False
    for idx, (c, n, lam, trial, mt) in enumerate(combos):
        spec = RunSpec(
            contamination_ratio=c,
            num_clients=n,
            arrival_rate=lam,
            arrival_model=cfg.arrival_model,
            requests_per_client=cfg.requests_per_client,
            trial=trial,
            burst_size=cfg.burst_size,
            burst_factor=cfg.burst_factor,
            seed=cfg.seed,
            response_text_max_chars=cfg.response_text_max_chars,
            max_workers=cfg.max_workers,
            prompt_salt_chars=(cfg.salt_length if cfg.prompt_salt else 0),
            salt_pool_size=(cfg.salt_pool_size if cfg.prompt_salt else 0),
            max_tokens=mt,
        )
        if cfg.simulate:
            factory = _build_sim_client_factory(cfg, spec)
            if verbose and idx == 0:
                print("Mode: SIMULATE (synthetic latencies, no network)")
        else:
            factory, endpoint, backend, model = _build_real_client_factory(cfg, max_tokens=mt)
            if verbose and idx == 0:
                print(f"Endpoint: {endpoint}  backend={backend}  model={model or '(worker default)'}")

        if verbose:
            print(
                f"[{idx + 1}/{len(combos)}] {spec.config_id()} "
                f"max_tokens={mt} "
                f"({spec.total_requests} requests, target {spec.arrival_rate * spec.num_clients:g} req/s, "
                f"pool={_default_max_workers(spec)} workers)"
            )
        try:
            result = run_workload(spec, factory, pairs, verbose=verbose)
        except WorkloadInterrupted as exc:
            interrupted = True
            result = exc.result
            if verbose:
                print("  interrupted; saving partial results for this config...")
        all_records.extend(result.records)
        per_config.append(
            {
                "config_id": spec.config_id(),
                "contamination_ratio": c,
                "num_clients": n,
                "arrival_rate": lam,
                "arrival_model": cfg.arrival_model,
                "trial": trial,
                "max_tokens": mt,
                "requests": len(result.records),
                "target_rate_req_s": result.target_rate,
                "achieved_rate_req_s": result.achieved_rate,
                "duration_s": result.duration_s,
                "config_cooldown_s": cfg.config_cooldown_s,
            }
        )
        metadata["num_runs_completed"] = idx + 1
        metadata["total_requests"] = len(all_records)
        metadata["per_config"] = per_config

        slo_lat = [] if cfg.auto_slo else cfg.latency_slos
        slo_ttft = [] if cfg.auto_slo else cfg.ttft_slos
        slo_tbt = [] if cfg.auto_slo else cfg.tbt_slos
        summaries = (
            generate_summaries(records_to_frame(all_records), slo_lat, slo_ttft, slo_tbt)
            if all_records
            else {}
        )
        metadata["status"] = "interrupted" if interrupted else metadata.get("status", "in_progress")
        checkpoint_run(paths, all_records, metadata, summaries)

        if verbose:
            ok = sum(1 for r in result.records if r.success)
            print(
                f"    done: {ok}/{len(result.records)} ok, "
                f"achieved {result.achieved_rate:.2f} req/s"
            )
            print(
                f"    checkpoint: {len(all_records)} rows -> {paths.requests_csv.name} "
                f"({idx + 1}/{len(combos)} configs)"
            )

        if interrupted:
            if verbose:
                print("\nExperiment interrupted; partial results saved.")
            break

        if cfg.config_cooldown_s > 0 and idx + 1 < len(combos):
            if verbose:
                print(
                    f"    cooldown: waiting {cfg.config_cooldown_s:g}s "
                    f"before next config ({combos[idx + 1][0]:g} c / "
                    f"N={combos[idx + 1][1]} / lam={combos[idx + 1][2]:g})..."
                )
            time.sleep(cfg.config_cooldown_s)

    if not interrupted:
        metadata["status"] = "completed"
    metadata["num_runs"] = len(combos)
    slo_lat = [] if cfg.auto_slo else cfg.latency_slos
    slo_ttft = [] if cfg.auto_slo else cfg.ttft_slos
    slo_tbt = [] if cfg.auto_slo else cfg.tbt_slos
    summaries = (
        generate_summaries(records_to_frame(all_records), slo_lat, slo_ttft, slo_tbt)
        if all_records
        else {}
    )
    checkpoint_run(paths, all_records, metadata, summaries)

    if verbose:
        ov = summaries.get("overall", {})
        print(f"\nSaved {len(all_records)} request rows -> {paths.requests_csv}")
        print(f"Summaries -> {paths.summary_dir}")
        lat = ov.get("latency_ms") or {}
        if lat.get("mean") is not None:
            print(
                f"Overall latency: mean={lat['mean']:.0f}ms p95={lat['p95']:.0f}ms "
                f"success={ov.get('request_success_rate', 0):.3f}"
            )
        else:
            print("Overall latency: n/a")
        if all_records:
            val = compute_validation_summary(records_to_frame(all_records))
            for line in format_validation_report(val):
                print(line)
    return paths.raw_dir


def _merge_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    def parse_list(val, cast):
        return [cast(x) for x in str(val).split(",") if x != ""]

    if args.contamination_ratio is not None:
        cfg.contamination_ratio = parse_list(args.contamination_ratio, float)
    if args.num_clients is not None:
        cfg.num_clients = parse_list(args.num_clients, int)
    if args.arrival_rate is not None:
        cfg.arrival_rate = parse_list(args.arrival_rate, float)
    if args.arrival_model is not None:
        cfg.arrival_model = args.arrival_model
    if args.requests_per_client is not None:
        cfg.requests_per_client = args.requests_per_client
    if args.num_trials is not None:
        cfg.num_trials = args.num_trials
    if args.endpoint is not None:
        cfg.endpoint = args.endpoint
    if args.runpod_endpoint_id is not None:
        cfg.runpod_endpoint_id = args.runpod_endpoint_id
    if args.backend is not None:
        cfg.backend = args.backend
    if args.model is not None:
        cfg.model = args.model
    if args.attack_model is not None:
        cfg.attack_model = args.attack_model
    if args.attack_file is not None:
        cfg.attack_file = args.attack_file
    if args.workload_size is not None:
        cfg.workload_size = args.workload_size
    if args.api_key is not None:
        cfg.api_key = args.api_key
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.latency_slo_ms is not None:
        cfg.latency_slo_ms = parse_list(args.latency_slo_ms, float)
    if args.ttft_slo_ms is not None:
        cfg.ttft_slo_ms = parse_list(args.ttft_slo_ms, float)
    if args.tbt_slo_ms is not None:
        cfg.tbt_slo_ms = parse_list(args.tbt_slo_ms, float)
    if args.auto_slo:
        cfg.auto_slo = True
    if args.no_auto_slo:
        cfg.auto_slo = False
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    if args.name is not None:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.config_cooldown_s is not None:
        cfg.config_cooldown_s = args.config_cooldown_s
    if args.simulate:
        cfg.simulate = True
    cfg.validate()
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None, help="YAML experiment config")
    p.add_argument("--simulate", action="store_true", help="Synthetic latencies, no endpoint")
    p.add_argument("--name", default=None,
                   help="Experiment label prepended to raw run folder (e.g. rq2_sweep_run_20260618_...)")
    p.add_argument("--contamination-ratio", default=None, help="Comma list, e.g. 0,0.05,0.1,0.5")
    p.add_argument("--num-clients", default=None, help="Comma list (default 1)")
    p.add_argument("--arrival-rate", default=None, help="Comma list, req/s per client")
    p.add_argument("--arrival-model", default=None, choices=["fixed", "poisson", "bursty"])
    p.add_argument("--requests-per-client", type=int, default=None)
    p.add_argument("--num-trials", type=int, default=None)
    p.add_argument("--endpoint", default=None)
    p.add_argument("--runpod-endpoint-id", default=None)
    p.add_argument("--backend", default=None, choices=["auto", "openai", "runpod"])
    p.add_argument("--model", default=None)
    p.add_argument("--attack-model", default=None)
    p.add_argument("--attack-file", default=None)
    p.add_argument(
        "--workload-size",
        type=int,
        default=None,
        help="Use top-n prompts by success_rate from the attack file (default: all)",
    )
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    slo_mode = p.add_mutually_exclusive_group()
    slo_mode.add_argument("--auto-slo", action="store_true", help="Skip fixed SLOs; calibrate at analyze time")
    slo_mode.add_argument("--no-auto-slo", action="store_true", help="Use fixed latency/ttft SLO budgets")
    p.add_argument("--latency-slo-ms", default=None, help="Comma list (ignored when auto_slo)")
    p.add_argument("--ttft-slo-ms", default=None, help="Comma list (ignored when auto_slo)")
    p.add_argument("--tbt-slo-ms", default=None, help="Comma list (ignored when auto_slo)")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--config-cooldown-s",
        type=float,
        default=None,
        help="Seconds to wait after each factor combo drains before starting the next",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = config_from_dict({"num_clients": 1})  # one client by default
    cfg = _merge_overrides(cfg, args)
    run_experiment(cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
