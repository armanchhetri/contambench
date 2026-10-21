"""CPU tests for `contam-bench generate` and contambench.gen; no torch needed."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from contambench import cli, export, gen
from contambench.gen import modelsinfo, summary

FIX = Path(__file__).parent / "fixtures"
SCHEMAS = Path(__file__).resolve().parents[1] / "src" / "contambench" / "schemas"

# Runs a snippet in a fresh interpreter in which torch (and friends) cannot be imported.
BLOCK = textwrap.dedent("""
    import sys
    from importlib.abc import MetaPathFinder
    BLOCKED = {"torch", "transformers", "accelerate", "bitsandbytes", "datasets", "antropy"}
    class Block(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)")
            return None
    sys.meta_path.insert(0, Block())
""")


def _run_blocked(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK + textwrap.dedent(code)], capture_output=True, text=True)


def test_cli_help_and_light_imports_without_torch():
    r = _run_blocked("""
        import contextlib, io, sys
        from contambench import cli
        import contambench.gen, contambench.gen.summary, contambench.gen.modelsinfo
        for argv in (["--help"], ["generate", "--help"], ["export", "--help"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    cli.main(argv)
                except SystemExit as e:
                    assert e.code == 0, (argv, e.code)
            assert "usage" in buf.getvalue()
            if argv == ["--help"]:
                assert "generate" in buf.getvalue()
        assert cli.main(["generate", "--list-models"]) == 0
        heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "transformers", "pydantic"))
        assert not heavy, heavy
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "meta-llama/Llama-3.2-3B-Instruct" in r.stdout and r.stdout.rstrip().endswith("OK")


def test_generate_without_extra_prints_install_hint():
    r = _run_blocked("""
        import sys
        from contambench import cli
        sys.exit(cli.main(["generate", "--model", "meta-llama/Llama-3.2-3B-Instruct", "--out", "unused"]))
    """)
    assert r.returncode == 3
    assert "pip install 'contambench[gen]'" in r.stderr
    assert "torch" in r.stderr


def test_generate_hint_in_process(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(gen, "_find_spec", lambda name: None)
    assert cli.main(["generate", "--model", "m", "--out", str(tmp_path / "o")]) == 3
    assert gen.INSTALL_HINT in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
    assert "bitsandbytes" in gen.missing(quantize=True)


def test_heavy_modules_are_guarded():
    r = _run_blocked("""
        from contambench.gen import MissingExtraError
        for mod in ("contambench.gen.models", "contambench.gen.run", "contambench.gen.loopllm.attack",
                    "contambench.gen.loopllm.utils", "contambench.gen.engorgio.attack"):
            try:
                __import__(mod)
            except MissingExtraError as e:
                assert "pip install 'contambench[gen]'" in str(e), mod
            else:
                raise AssertionError(f"{mod} imported without the extra")
        print("OK")
    """)
    assert r.returncode == 0, r.stderr + r.stdout


def test_summarizer_picks_best_step():
    s = summary.summarize_loopllm_results(FIX / "gen_res")
    p = s["prompts"]
    assert [x["prompt"] for x in p] == ["benign prompt 0", "benign prompt 1", "benign prompt 2"]  # res_3 unevaluated
    assert [x["success_rate"] for x in p] == [0.75, 0.5, 0.5]
    assert [x["adv_suffix"].split()[0] for x in p] == ["sfx1", "sfx1", "sfx2"]  # ties keep the earlier step
    assert s["average_asr"] == pytest.approx(1.75 / 3)
    assert set(p[0]) == set(summary.PAIR_KEYS)


def test_pairs_file_feeds_export(tmp_path):
    s = summary.summarize_loopllm_results(FIX / "gen_res")
    prov = {"generator": "loopllm", "model": "test/model", "suffix_tokens": 30, "seed": 23,
            "versions": {"contambench": "test"}}
    pairs_file, work = summary.resolve_out(tmp_path / "suite")
    assert (pairs_file.name, work.name) == ("pairs.json", "work")
    summary.write_pairs(pairs_file, s, prov)
    doc = json.loads(pairs_file.read_text())
    assert doc["provenance"]["generator"] == "loopllm" and doc["pairs"] == 3
    assert [r["pair_index"] for r in doc["prompts"]] == [0, 1, 2]

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / "stressor_pair.schema.json").read_text())
    for row in doc["prompts"]:
        jsonschema.validate({k: v for k, v in row.items() if k != "pair_index"}, schema)

    out = tmp_path / "w.csv"
    assert cli.main(["export", "--pairs", str(pairs_file), "--c", "0.10", "--k", "3", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    attack = [r for r in rows if r["is_attack"] == "1"]
    assert len(rows) == 1000 and len(attack) == 100
    adv = {p["prompt"]: p["adv_prompt"] for p in export.load_pairs(pairs_file)}
    assert all(r["prompt"] == adv[r["benign_prompt"]] for r in attack)
    man = json.loads(out.with_suffix(".manifest.json").read_text())
    assert man["n_attack"] == 100


def test_resolve_out_file():
    f, w = summary.resolve_out("x/my_suite.json")
    assert f == Path("x/my_suite.json") and w == Path("x/my_suite.work")


def test_modelsinfo(tmp_path):
    info = modelsinfo.get_modelinfo("meta-llama/Llama-3.2-3B-Instruct")
    assert info.limit == 4096 and info.tokenizer is None
    assert modelsinfo.get_modelinfo("meta-llama/Llama-3.2-3B-Instruct", limit=512).limit == 512
    with pytest.warns(UserWarning):
        unk = modelsinfo.get_modelinfo("someone/unlisted-model")
    assert unk.limit == modelsinfo.DEFAULT_LIMIT and unk.stop_ids == []
    with pytest.raises(KeyError):
        modelsinfo.get_modelinfo("someone/unlisted-model", allow_unknown=False)
    f = tmp_path / "mi.json"
    f.write_text(json.dumps([{"name": "someone/unlisted-model", "limit": 2048, "stop_ids": [7], "eot": 9}]))
    own = modelsinfo.get_modelinfo("someone/unlisted-model", f)
    assert (own.limit, own.stop_ids, own.eot) == (2048, [7], 9)
    assert all(not modelsinfo.get_modelinfo(m).is_vlm for m in modelsinfo.known_models())

