"""contam-bench command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contambench import __version__, data


def _data_base(args) -> Path | None:
    return Path(args.data_dir).expanduser() if getattr(args, "data_dir", None) else None


def cmd_data(args) -> int:
    base = _data_base(args)
    if args.action == "fetch":
        parts = [args.subset] if args.subset else None
        dest = data.fetch(parts, base=base, source=args.source)
        print(f"data ready at {dest}")
    elif args.action == "path":
        print(data.root(base))
    elif args.action == "status":
        print(json.dumps({"root": str(data.root(base)), **data.status(base)}, indent=2))
    return 0


def cmd_export(args) -> int:
    from contambench import export

    pairs_file = Path(args.pairs) if args.pairs else data.suite_file(args.suite, _data_base(args))
    pairs = export.load_pairs(pairs_file)
    benign = export.read_benign_csv(Path(args.benign)) if args.benign else export.benign_pool(pairs, args.rows)
    rows, manifest = export.contaminate(benign, pairs, args.c, args.k)
    manifest = {"suite": args.suite or str(pairs_file), "seed": args.seed,
                "seed_note": "recorded for compatibility; the selection rule is deterministic", **manifest}
    man = export.write(Path(args.out), rows, manifest)
    print(f"wrote {args.out} ({manifest['n_attack']}/{manifest['n_rows']} stressor rows, "
          f"{manifest['distinct_attack_prompts']} distinct prompts) and {man}")
    return 0


def cmd_run(args, extra: list[str]) -> int:
    from contambench.harness import experiment

    experiment.main(extra)
    return 0


def cmd_calibrate(args) -> int:
    import pandas as pd

    from contambench.harness import experiment
    from contambench.harness.analysis import calibrate_capacity
    from contambench.harness.config import config_from_dict

    if args.config:
        from contambench.harness.config import load_config

        cfg = load_config(args.config)
        cfg.contamination_ratio = [0.0]  # calibration is benign by definition
        if args.endpoint:
            cfg.endpoint = args.endpoint
        if args.api_key:
            cfg.api_key = args.api_key
        if args.simulate:
            cfg.simulate = True
        if args.results_dir:
            cfg.results_dir = args.results_dir
        run_dir = experiment.run_experiment(cfg)
        num_clients = max(cfg.client_counts)
    else:
        if not args.suite:
            raise SystemExit("calibrate needs --config or --suite")
        run_dir, num_clients = None, args.num_clients
    if run_dir is None:
        cfg = config_from_dict({
        "name": "capacity_calibrate",
        "endpoint": args.endpoint, "api_key": args.api_key, "model": args.model,
        "attack_model": data.SUITES[data.suite_slug(args.suite)],
        "attack_file": str(data.suite_file(args.suite, _data_base(args))),
        "contamination_ratio": [0.0], "num_clients": args.num_clients,
        "arrival_rate": [float(x) for x in args.arrival_rates.split(",")],
        "arrival_model": "poisson", "requests_per_client": args.requests_per_client,
        "num_trials": args.num_trials, "max_tokens": args.max_tokens, "temperature": 0.7,
        "auto_slo": True, "seed": args.seed, "results_dir": args.results_dir, "simulate": args.simulate,
        })
        run_dir = experiment.run_experiment(cfg)
    df = pd.read_parquet(Path(run_dir) / "requests.parquet")
    out = Path(run_dir) / "calibrate-capacity"
    res = calibrate_capacity.run(df, out, num_clients=num_clients)
    print(json.dumps({k: res.get(k) for k in ("capacity_throughput_req_s", "recommended_arrival_rates")}, indent=2))
    print(f"calibration written to {out}")
    return 0


def cmd_score(args) -> int:
    from contambench import score

    df = score.load(args.run)
    rates = [args.arrival_rate] if args.arrival_rate is not None else sorted(df.arrival_rate.unique())
    results = [score.sweep(df, r) for r in rates]
    if args.json:
        print(json.dumps({"sweeps": results, "stage_b": score.stage_b(df)}, indent=2))
    else:
        for s in results:
            print(score.format_sweep(s), end="\n\n")
        sb = score.stage_b(df)
        if sb["stressor_cap_hit"] is not None:
            print(f"Stage B: {100 * sb['stressor_cap_hit']:.1f}% of {sb['n_stressor']} stressor requests reached the "
                  f"output cap ({100 * sb['benign_cap_hit']:.2f}% of {sb['n_benign']} benign)")
    return 0


def cmd_examples(args) -> int:
    import shutil
    from importlib.resources import files

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = files("contambench") / "examples"
    for f in sorted(p for p in src.iterdir() if p.name.endswith(".yaml")):
        target = dest / f.name
        if target.exists() and not args.force:
            print(f"kept existing {target}")
            continue
        target.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"wrote {target}")
    return 0


def cmd_report(args) -> int:
    from contambench import report

    print(report.render(_data_base(args), check=not args.no_check))
    return 0


def cmd_telemetry(args, extra: list[str]) -> int:
    from contambench.harness import telemetry

    return telemetry.main(extra)


def cmd_guidellm(args, extra: list[str]) -> int:
    import shutil
    import subprocess

    exe = shutil.which("guidellm")
    if not exe:
        print("GuideLLM is not installed: pip install 'contambench[guidellm]'", file=sys.stderr)
        return 2
    return subprocess.call([exe, *extra])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="contam-bench", description=__doc__)
    p.add_argument("--version", action="version", version=f"contambench {__version__}")
    p.add_argument("--data-dir", help="data root (default: $CONTAMBENCH_DATA or ~/.cache/contambench/<version>)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="fetch, locate, or inspect the data record")
    d.add_argument("action", choices=["fetch", "path", "status"])
    d.add_argument("--subset", choices=list(data.PARTS), help="fetch one product only")
    d.add_argument("--from", dest="source", help="copy the record from a local directory instead of Zenodo")

    e = sub.add_parser("export", help="write a D2 labeled workload at contamination ratio c")
    e.add_argument("--suite", help=f"one of: {', '.join(data.SUITES)}")
    e.add_argument("--pairs", help="use a pairs file instead of a fetched suite")
    e.add_argument("--benign", help="benign pool CSV (prompt_id, prompt, success_rate); default: tile the suite")
    e.add_argument("--c", type=float, required=True)
    e.add_argument("--k", type=int, default=50, help="draw stressors from the top-k prompts (default 50)")
    e.add_argument("--rows", type=int, default=1000)
    e.add_argument("--seed", type=int, default=23)
    e.add_argument("--out", required=True)

    sub.add_parser("run", help="execute a workload against an endpoint (options as in the harness; see run --help)",
                   add_help=False)

    c = sub.add_parser("calibrate", help="benign arrival-rate sweep -> capacity and operating points")
    c.add_argument("--config", help="harness YAML; contamination is forced to 0")
    c.add_argument("--endpoint", required=False)
    c.add_argument("--api-key")
    c.add_argument("--model", help="model name sent in the request body")
    c.add_argument("--suite", help="suite whose benign prompts drive the sweep (when no --config)")
    c.add_argument("--arrival-rates", default="0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0")
    c.add_argument("--num-clients", type=int, default=50)
    c.add_argument("--requests-per-client", type=int, default=20)
    c.add_argument("--num-trials", type=int, default=1)
    c.add_argument("--max-tokens", type=int, default=4096)
    c.add_argument("--seed", type=int, default=23)
    c.add_argument("--results-dir")
    c.add_argument("--simulate", action="store_true", help="synthetic latencies, no endpoint")

    s = sub.add_parser("score", help="apply the SLO oracle and label split to a D3-format run")
    s.add_argument("run", help="run directory or requests.parquet/csv")
    s.add_argument("--arrival-rate", type=float)
    s.add_argument("--json", action="store_true")

    ex = sub.add_parser("examples", help="copy the example YAML campaign configurations")
    ex.add_argument("dest", nargs="?", default="examples")
    ex.add_argument("--force", action="store_true", help="overwrite existing files")

    r = sub.add_parser("report", help="reprint the paper's tables from the data record")
    r.add_argument("--no-check", action="store_true", help="skip the comparison with the paper's values")

    sub.add_parser("telemetry", help="poll a vLLM /metrics endpoint into a JSONL log (see telemetry --help)",
                   add_help=False)
    sub.add_parser("guidellm", help="run GuideLLM (pass its arguments after --)", add_help=False)
    return p


def _command_index(argv: list[str]) -> int | None:
    """Index of the subcommand token, skipping global options."""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--data-dir":
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    i = _command_index(argv)
    # `run`, `telemetry`, and `guidellm` forward everything after the subcommand.
    passthrough = {"run": cmd_run, "guidellm": cmd_guidellm, "telemetry": cmd_telemetry}
    if i is not None and argv[i] in passthrough:
        args = build_parser().parse_args(argv[: i + 1])
        extra = argv[i + 1:]
        if extra[:1] == ["--"]:
            extra = extra[1:]
        return passthrough[argv[i]](args, extra)
    args = build_parser().parse_args(argv)
    return {"data": cmd_data, "export": cmd_export, "calibrate": cmd_calibrate, "score": cmd_score,
            "report": cmd_report, "examples": cmd_examples}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
