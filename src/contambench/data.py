"""Locate, download, and verify the ContamBench data record.

The record holds three zips (D1, D2, D3) plus documentation and a SHA256SUMS file.
``fetch`` copies them from Zenodo (or from a local directory with ``source=``),
checks every hash, and extracts the zips into the data root.

Data root, in order of precedence: the ``root`` argument, ``$CONTAMBENCH_DATA``,
then ``~/.cache/contambench/<version>``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from contambench import __data_version__

# Zenodo record of data version 0.1.0 (DOI 10.5281/zenodo.23482023).
ZENODO_RECORD = os.environ.get("CONTAMBENCH_ZENODO_RECORD", "23482023")
ZENODO_FILES = "https://zenodo.org/records/{record}/files/{name}?download=1"

PARTS = {"d1": "d1_stressors", "d2": "d2_workloads", "d3": "d3_execution"}
DOCS = ["README.md", "DATASET_CARD.md", "LICENSE-DATA.md", "CITATION.cff", "runs_index.json"]

# Short suite names accepted on the command line, mapped to model ids.
SUITES = {
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-math-7b": "Qwen/Qwen2.5-Math-7B",
}
CAMPAIGNS = {
    "aws_llama3b": "llama-3.2-3b-instruct",
    "aws_qwen7b": "qwen2.5-7b-instruct",
    "runpod_llama8b": "llama-3.1-8b-instruct",
    "aws_mistral7b": "mistral-7b-instruct-v0.3",
}


class DataNotFound(RuntimeError):
    pass


def root(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    env = os.environ.get("CONTAMBENCH_DATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "contambench" / __data_version__


def require(part: str, base: Path | None = None) -> Path:
    """Path to an extracted product directory, with a helpful error if it is missing."""
    d = (base or root()) / PARTS[part]
    if not d.is_dir():
        raise DataNotFound(
            f"{PARTS[part]} not found under {d.parent}. Run `contam-bench data fetch` "
            "(or set CONTAMBENCH_DATA to an extracted record)."
        )
    return d


def suite_slug(name: str) -> str:
    """Accept a short suite name or a Hugging Face model id."""
    if name in SUITES:
        return name
    for slug, model in SUITES.items():
        if model.lower() == name.lower():
            return slug
    raise KeyError(f"unknown suite {name!r}; choose one of {', '.join(SUITES)}")


def suite_file(name: str, base: Path | None = None) -> Path:
    return require("d1", base) / "suites" / suite_slug(name) / "pairs.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.replace(dst)


def _get(name: str, dst: Path, source: Path | None) -> None:
    if source is not None:
        shutil.copyfile(source / name, dst)
        return
    if not ZENODO_RECORD:
        raise DataNotFound(
            "The Zenodo record id is not set in this build. Pass --from <dir> with a local copy "
            "of the record, or set CONTAMBENCH_ZENODO_RECORD."
        )
    _download(ZENODO_FILES.format(record=ZENODO_RECORD, name=name), dst)


def fetch(parts: list[str] | None = None, *, base: Path | None = None, source: str | None = None,
          log=print) -> Path:
    """Download (or copy), verify, and extract the requested products."""
    dest = base or root()
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(source).expanduser() if source else None
    _get("SHA256SUMS", dest / "SHA256SUMS", src)
    sums = {}
    for line in (dest / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    wanted = [f"{PARTS[p]}.zip" for p in (parts or list(PARTS))]
    names = wanted + [d for d in DOCS if d in sums] + (["schemas.zip"] if "schemas.zip" in sums else [])
    for name in names:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.exists() and _sha256(target) == sums[name]):
            log(f"fetching {name}")
            _get(name, target, src)
        if _sha256(target) != sums[name]:
            raise RuntimeError(f"checksum mismatch for {name}")
    for name in wanted + (["schemas.zip"] if "schemas.zip" in sums else []):
        out = dest / name[:-4]
        if out.is_dir():
            shutil.rmtree(out)
        with zipfile.ZipFile(dest / name) as z:
            z.extractall(dest)
        log(f"extracted {name[:-4]}/")
    return dest


def status(base: Path | None = None) -> dict[str, bool]:
    b = base or root()
    return {k: (b / v).is_dir() for k, v in PARTS.items()}
