"""Per-model metadata for generation (ported from the research repo's ``src/modelsinfo.py``).

The shipped ``data/modelsinfo.json`` holds the text-only models of the research
codebase. A model that is not listed gets defaults (no system-prompt fallback roles,
no extra stop ids, ``AutoTokenizer``, output limit 4096 unless ``limit`` is given), or
its entry can be supplied with ``--model-info FILE``. No torch import here.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

DEFAULT_LIMIT = 4096


@dataclass
class SysPrompt:
    sys_prompt: str = ""
    roles: tuple[str, str] = ("", "")


@dataclass
class ModelInfo:
    name: str
    sys_prompt: SysPrompt = field(default_factory=SysPrompt)
    limit: int = DEFAULT_LIMIT
    is_vlm: bool | None = False
    stop_ids: list[int] = field(default_factory=list)
    tokenizer: str | None = None
    model_loader: str | None = None
    eot: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        sp = d.get("sys_prompt") or {}
        if isinstance(sp, dict):
            sp = SysPrompt(sys_prompt=sp.get("sys_prompt", ""), roles=tuple(sp.get("roles", ("", ""))))
        return cls(
            name=d["name"],
            sys_prompt=sp,
            limit=int(d.get("limit") or DEFAULT_LIMIT),
            is_vlm=bool(d.get("is_vlm", False)),
            stop_ids=list(d.get("stop_ids") or []),
            tokenizer=d.get("tokenizer"),
            model_loader=d.get("model_loader"),
            eot=d.get("eot"),
        )


def _load(path: Path | None) -> dict[str, dict]:
    text = Path(path).read_text(encoding="utf-8") if path else \
        (files("contambench.gen") / "data" / "modelsinfo.json").read_text(encoding="utf-8")
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = [raw]
    return {e["name"]: e for e in raw if e.get("name")}


def known_models() -> list[str]:
    return sorted(_load(None))


def get_modelinfo(name: str, info_path: str | Path | None = None, limit: int | None = None,
                  allow_unknown: bool = True) -> ModelInfo:
    """Metadata for ``name``: from ``info_path`` if given, else the shipped file, else defaults."""
    entries = _load(None)
    if info_path:
        entries.update(_load(Path(info_path)))
    if name in entries:
        info = ModelInfo.from_dict(entries[name])
    elif allow_unknown:
        warnings.warn(
            f"{name} is not in contambench's modelsinfo.json; using defaults (output limit "
            f"{limit or DEFAULT_LIMIT}, the tokenizer's chat template, no extra stop ids). "
            "Pass --model-info FILE to set them.", UserWarning, stacklevel=2)
        info = ModelInfo(name=name)
    else:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(entries))}")
    if info.is_vlm:
        raise ValueError(f"{name} is a vision-language model; `generate` supports text-only models")
    if limit is not None:
        info.limit = int(limit)
    return info
