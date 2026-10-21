"""Stressor generation (``contam-bench generate``); needs the ``gen`` extra and a GPU.

This module is safe to import without the extra: it only checks what is installed.
The modules that do the work (``models``, ``attacker``, ``stats``, ``loopllm``,
``engorgio``, ``run``) import torch and call :func:`require_extra` first, so a missing
extra gives the install hint instead of a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib.util

INSTALL_HINT = "pip install 'contambench[gen]'"

# Import names the two generation paths need (see the `gen` extra in pyproject.toml).
REQUIRED = ("torch", "transformers", "accelerate", "pydantic", "datasets", "antropy", "tqdm")
QUANTIZE_REQUIRED = ("bitsandbytes",)


class MissingExtraError(ImportError):
    """The ``gen`` extra (torch and friends) is not installed."""


def _find_spec(name: str):
    return importlib.util.find_spec(name)


def missing(quantize: bool = False) -> list[str]:
    """Names of the required modules that are not importable."""
    names = REQUIRED + (QUANTIZE_REQUIRED if quantize else ())
    out = []
    for n in names:
        try:
            if _find_spec(n) is None:
                out.append(n)
        except (ImportError, ValueError):
            out.append(n)
    return out


def hint(names: list[str]) -> str:
    return (f"stressor generation needs the optional 'gen' extra (missing: {', '.join(names)}).\n"
            f"Install it with:  {INSTALL_HINT}\n"
            "Generation also needs a CUDA GPU; see 'Generate a suite for a new model' in the README.")


def require_extra(quantize: bool = False) -> None:
    names = missing(quantize)
    if names:
        raise MissingExtraError(hint(names))
