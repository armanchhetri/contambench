# Vendored from LoopLLM (https://github.com/neuron-insight-lab/LoopLLM), Apache-2.0; see ../LICENSE.
# Modified by the ContamBench authors (loss configurations, import paths).
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_COLLAPSE_TERMS = ("kl", "logit_mse", "cosine", "entropy")
DEFAULT_LOSS_CONFIG = "cycle"
DEFAULT_CYCLIC_SEGMENT_LEN = 5


@dataclass(frozen=True)
class LossConfig:
    name: str
    kind: Literal["cycle", "collapse"]
    terms: Optional[Sequence[str]] = None


LOSS_CONFIGS: dict[str, LossConfig] = {
    "cycle": LossConfig(name="cycle", kind="cycle"),
    "collapse_kl": LossConfig(name="collapse_kl", kind="collapse", terms=("kl",)),
    "collapse_logit_mse": LossConfig(
        name="collapse_logit_mse", kind="collapse", terms=("logit_mse",)
    ),
    "collapse_cosine": LossConfig(
        name="collapse_cosine", kind="collapse", terms=("cosine",)
    ),
    "collapse_entropy": LossConfig(
        name="collapse_entropy", kind="collapse", terms=("entropy",)
    ),
    "collapse_logit_mse_entropy": LossConfig(
        name="collapse_logit_mse_entropy",
        kind="collapse",
        terms=("logit_mse", "entropy"),
    ),
    "collapse_cosine_entropy": LossConfig(
        name="collapse_cosine_entropy",
        kind="collapse",
        terms=("cosine", "entropy"),
    ),
    "collapse_all": LossConfig(
        name="collapse_all",
        kind="collapse",
        terms=DEFAULT_COLLAPSE_TERMS,
    ),
}


def get_loss_config(name: str) -> LossConfig:
    if name not in LOSS_CONFIGS:
        valid = ", ".join(sorted(LOSS_CONFIGS))
        raise ValueError(f"Unknown loss config {name!r}. Valid: {valid}")
    return LOSS_CONFIGS[name]


LOSS_CONFIG_ALIASES: dict[str, str] = {
    "cycle": "cycle",
    "kl": "collapse_kl",
    "logit_mse": "collapse_logit_mse",
    "cosine": "collapse_cosine",
    "entropy": "collapse_entropy",
    **{name: name for name in LOSS_CONFIGS},
}


def resolve_loss_config(name: str) -> str:
    """Map CLI-friendly loss names (e.g. ``kl``) to ``LOSS_CONFIGS`` keys."""
    key = LOSS_CONFIG_ALIASES.get(name)
    if key is None:
        aliases = ", ".join(sorted(LOSS_CONFIG_ALIASES))
        raise ValueError(f"Unknown loss config {name!r}. Valid: {aliases}")
    get_loss_config(key)
    return key


def cycle_loss(
    logits: torch.Tensor,
    target_slice: slice,
    special_id,
    per_sample: bool = False,
):
    """BCE loss pushing cyclic trigger token probability toward 1 on the target region."""
    logits_t = logits[:, target_slice.start - 1 : -1, :]
    prob = torch.softmax(logits_t, dim=-1)
    special_p = prob[:, :, special_id]
    if isinstance(special_id, list):
        special_p = special_p.sum(dim=-1)

    loss_s = nn.BCELoss(reduction="none")(special_p, torch.ones_like(special_p))
    loss_s = loss_s.mean(dim=-1)

    if per_sample:
        return loss_s
    return loss_s.mean()


def compute_attack_loss(
    logits: torch.Tensor,
    suffix_manager,
    loss_config: LossConfig,
    per_sample: bool = False,
    k: int = 50,
    eps: float = 1e-8,
):
    if loss_config.kind == "cycle":
        return cycle_loss(
            logits,
            suffix_manager._target_slice,
            suffix_manager.adv_token_id,
            per_sample=per_sample,
        )
    return stationary_topk_collapse_loss(
        logits,
        suffix_manager._target_slice,
        suffix_manager.eos_token_id,
        k=k,
        eps=eps,
        per_sample=per_sample,
        terms=list(loss_config.terms) if loss_config.terms is not None else None,
    )

_TERM_WEIGHTS = {
    "kl": 1.0,
    "logit_mse": 0.5,
    "cosine": 0.5,
    "entropy": 0.05,
}


def _normalize_eos_token_ids(eos_token_ids):
    if isinstance(eos_token_ids, int):
        return [eos_token_ids]
    return list(eos_token_ids)


def _mask_eos_logits(logits: torch.Tensor, eos_token_ids) -> torch.Tensor:
    _, _, vocab_size = logits.shape
    eos_mask = torch.zeros(vocab_size, device=logits.device, dtype=torch.bool)
    for eos_id in _normalize_eos_token_ids(eos_token_ids):
        eos_mask[eos_id] = True
    return logits.masked_fill(eos_mask.view(1, 1, -1), -1e4)


def _compute_topk_shift_pair(
    work_logits: torch.Tensor,
    d: int,
    k: int,
    temp: float,
    sharpness: float,
    eps: float,
):
    z_p_full = work_logits[:, :-d, :]
    z_q_full = work_logits[:, d:, :]

    topk_vals, topk_idx = torch.topk(z_p_full / temp, k, dim=-1)
    p = F.softmax(topk_vals, dim=-1)
    q_logits = torch.gather(z_q_full, -1, topk_idx) / temp
    q = F.softmax(q_logits, dim=-1)
    z_p = topk_vals
    z_q = torch.gather(z_q_full, -1, topk_idx)

    weights = p.pow(sharpness)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)

    return {
        "z_p": z_p,
        "z_q": z_q,
        "p": p,
        "q": q,
        "weights": weights,
        "eps": eps,
    }


def stationary_topk_kl_penalty(pair: dict) -> torch.Tensor:
    """Weighted KL between consecutive top-k distributions. Shape: (batch, positions)."""
    p = pair["p"]
    q = pair["q"]
    weights = pair["weights"]
    eps = pair["eps"]
    return (weights * p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def stationary_topk_logit_mse_loss(pair: dict) -> torch.Tensor:
    """Weighted MSE between top-k logits. Shape: (batch, positions)."""
    z_p = pair["z_p"]
    z_q = pair["z_q"]
    weights = pair["weights"]
    return (weights * (z_p - z_q).pow(2)).sum(dim=-1)


def stationary_topk_logit_cosine_loss(pair: dict) -> torch.Tensor:
    """One minus cosine similarity of top-k logits. Shape: (batch, positions)."""
    return 1 - F.cosine_similarity(pair["z_p"], pair["z_q"], dim=-1)


def stationary_topk_entropy_loss(
    work_logits: torch.Tensor,
    k: int,
    temp: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Low-entropy penalty on the final tail window (top-k only). Shape: (batch,)."""
    _, seq_len, _ = work_logits.shape
    tail = work_logits[:, -min(32, seq_len) :, :]
    tail_topk = torch.topk(tail / temp, k, dim=-1).values
    tail_p = F.softmax(tail_topk, dim=-1)
    return -(tail_p * torch.log(tail_p + eps)).sum(dim=-1).mean(dim=-1)


def stationary_topk_collapse_loss(
    logits: torch.Tensor,
    target_slice: slice,
    eos_token_ids,
    k: int = 50,
    eps: float = 1e-8,
    per_sample: bool = False,
    terms=None,
):
    """
    Stationary-collapse loss on top-k logits for the target region only.

    Args:
        terms: loss components to include. Defaults to all four:
            ``kl``, ``logit_mse``, ``cosine``, ``entropy``.
    """
    if terms is None:
        terms = list(DEFAULT_COLLAPSE_TERMS)
    terms = list(terms)

    logits_t = logits[:, target_slice.start - 1 : -1, :]
    batch_size, seq_len, _ = logits_t.shape
    temp = 0.1
    sharpness = max(2.0, k / 4)
    horizon = min(10, seq_len - 1)

    work_logits = _mask_eos_logits(logits_t, eos_token_ids)
    total_loss = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)

    horizon_terms = [t for t in terms if t in ("kl", "logit_mse", "cosine")]
    if horizon_terms and horizon > 0:
        for d in range(1, horizon + 1):
            pair = _compute_topk_shift_pair(
                work_logits, d, k, temp, sharpness, eps
            )
            step_loss = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
            if "kl" in horizon_terms:
                step_loss = step_loss + _TERM_WEIGHTS["kl"] * stationary_topk_kl_penalty(pair).mean(dim=-1)
            if "logit_mse" in horizon_terms:
                step_loss = step_loss + _TERM_WEIGHTS["logit_mse"] * stationary_topk_logit_mse_loss(pair).mean(dim=-1)
            if "cosine" in horizon_terms:
                step_loss = step_loss + _TERM_WEIGHTS["cosine"] * stationary_topk_logit_cosine_loss(pair).mean(dim=-1)
            total_loss = total_loss + step_loss

        total_loss = total_loss / horizon

    if "entropy" in terms:
        total_loss = total_loss + _TERM_WEIGHTS["entropy"] * stationary_topk_entropy_loss(
            work_logits, k, temp=temp, eps=eps
        )

    if per_sample:
        return total_loss
    return total_loss.mean()
