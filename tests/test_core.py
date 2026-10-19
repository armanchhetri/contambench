"""CPU tests. Tests marked `needs_data` run when CONTAMBENCH_DATA points at an extracted record."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from contambench import cli, data, export, score

FIX = Path(__file__).parent / "fixtures"
HAVE_DATA = all(data.status().values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="set CONTAMBENCH_DATA to an extracted record")


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "contambench" in capsys.readouterr().out


def test_export_rule_is_deterministic():
    pairs = export.load_pairs(FIX / "pairs.json")
    pool = export.benign_pool(pairs, 100)
    rows, man = export.contaminate(pool, pairs, 0.10, 3)
    assert man["n_attack"] == 10
    assert man["distinct_attack_prompts"] <= 3
    again, _ = export.contaminate(pool, pairs, 0.10, 3)
    assert rows == again
    # stressor rows carry the stressor text and keep the benign text for joins
    for r in rows:
        assert r["prompt"].endswith("SUFFIX") == bool(r["is_attack"])
        assert r["benign_prompt"].startswith("benign prompt")


def test_export_rejects_too_small_pool():
    pairs = export.load_pairs(FIX / "pairs.json")
    with pytest.raises(ValueError):
        export.contaminate(export.benign_pool(pairs, 10), pairs, 0.5, 1)


def test_simulated_run_and_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["run", "--simulate", "--attack-file", str(FIX / "pairs.json"), "--contamination-ratio", "0,0.2",
              "--num-clients", "3", "--arrival-rate", "2", "--requests-per-client", "10", "--num-trials", "2",
              "--max-tokens", "64", "--quiet", "--results-dir", str(tmp_path / "res")])
    run = next((tmp_path / "res" / "raw").iterdir())
    df = score.load(run)
    s = score.sweep(df, 2.0)
    assert [r["c"] for r in s["rows"]] == [0.0, 0.2]
    assert 0.0 <= s["rows"][0]["slo_violation"] <= 1.0


@needs_data
def test_export_reproduces_shipped_workload(tmp_path):
    out = tmp_path / "w.csv"
    cli.main(["export", "--suite", "llama-3.2-3b-instruct", "--c", "0.10", "--out", str(out)])
    shipped = data.require("d2") / "exports" / "llama-3.2-3b-instruct" / "c0p10_tiled1000.csv"
    assert out.read_bytes() == shipped.read_bytes()
    man = json.loads(out.with_suffix(".manifest.json").read_text())
    assert (man["n_attack"], man["distinct_attack_prompts"]) == (100, 21)


@needs_data
def test_report_matches_paper():
    from contambench import report

    assert report.check_against_paper(report.table_sweep()) == []
    g = report.table_guidellm()
    assert round(100 * g["guidellm_benign"]["slo_violation_rate"], 1) == 9.0
    assert round(100 * g["guidellm_c0p10"]["slo_violation_rate"], 1) == 66.5


@needs_data
def test_shipped_files_match_schemas():
    jsonschema = pytest.importorskip("jsonschema")
    schemas = Path(__file__).resolve().parents[1] / "src" / "contambench" / "schemas"
    exp = json.loads((schemas / "contamination_export.schema.json").read_text())
    path = data.require("d2") / "exports" / "llama-3.2-3b-instruct" / "c0p10_tiled1000.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            row = {k: (float(v) if k == "success_rate" else int(v) if k in ("is_attack", "row_index") else v)
                   for k, v in row.items()}
            jsonschema.validate(row, exp)
            if i > 50:
                break
    trace = json.loads((schemas / "request_trace.schema.json").read_text())
    df = pd.read_parquet(data.require("d3") / "runs" / "aws_llama3b" / "rq2_sweep" / "requests.parquet").head(50)
    for rec in json.loads(df.to_json(orient="records")):
        jsonschema.validate(rec, trace)


def test_api_key_never_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["run", "--simulate", "--attack-file", str(FIX / "pairs.json"), "--contamination-ratio", "0",
              "--num-clients", "1", "--arrival-rate", "5", "--requests-per-client", "2", "--max-tokens", "16",
              "--api-key", "sk-test-secret-123456", "--quiet", "--results-dir", str(tmp_path / "res")])
    for f in (tmp_path / "res").rglob("*"):
        if f.is_file() and f.suffix in (".json", ".csv"):
            assert "sk-test-secret-123456" not in f.read_text(errors="replace"), f


def test_calibrate_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cal.yaml"
    cfg.write_text(
        f"attack_file: {FIX / 'pairs.json'}\nnum_clients: 2\narrival_rate: [2.0, 4.0]\nrequests_per_client: 3\n"
        "num_trials: 1\nmax_tokens: 16\nauto_slo: true\nseed: 23\n"
    )
    assert cli.main(["calibrate", "--config", str(cfg), "--simulate", "--results-dir", str(tmp_path / "res")]) == 0
    out = next((tmp_path / "res" / "raw").glob("*/calibrate-capacity/capacity_calibration.json"))
    assert json.loads(out.read_text())["capacity_throughput_req_s"] > 0


def test_import_guidellm_report(tmp_path):
    """A report written by stock GuideLLM 0.7.1 (mock server) imports and scores."""
    out = tmp_path / "gl"
    cli.main(["import-guidellm", str(FIX / "guidellm_mock.json"), "--labels", str(FIX / "guidellm_workload.csv"),
              "--max-tokens", "64", "--out", str(out)])
    df = score.load(out)
    assert len(df) == 8 and df.success.all()
    assert (df.prompt_type == "attack").sum() == 1 and (df.prompt_type == "unlabeled").sum() == 0
    assert df.contamination_ratio.iloc[0] == 0.1 and df.arrival_rate.iloc[0] == 20
    assert set(score.METRICS) <= set(df.columns) and df.latency_ms.notna().all()


@needs_data
def test_guidellm_case_reproduces_from_raw_report():
    from contambench import report

    g = report.table_guidellm()
    assert g["guidellm_c0p10"]["n_stressor"] == 99 and g["guidellm_c0p10"]["unlabeled"] == 0
    assert round(100 * g["guidellm_c0p10"]["stressor_slo_violation"], 1) == 78.8
    assert round(100 * g["guidellm_c0p10"]["benign_slo_violation"], 1) == 65.1
