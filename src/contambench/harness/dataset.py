"""Prompt loading and contamination scheduling for real-world endpoint tests.

The benign and adversarial prompts both come from the LoopLLM attack summary
(``attack_data/<model_folder>/loopllm/summary.json`` by default). Each entry exposes
a benign ``prompt`` and an adversarial ``adv_prompt``; the contamination ratio
controls what fraction of issued requests use ``adv_prompt``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


_SALT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def make_salt(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_SALT_ALPHABET) for _ in range(length))


def make_salt_pool(seed: int, size: int, length: int) -> list[str]:
    """Deterministically build a fixed pool of ``size`` distinct salt ids.

    Drawing per-request salts from a small shared pool makes prefixes *repeat*,
    so the vLLM prefix-cache hit rate settles at a tunable level (smaller pool →
    more reuse → higher hit rate) instead of collapsing to ~0% with unique salts.
    """
    if size <= 0 or length <= 0:
        return []
    rng = random.Random((seed * 1_000_003) ^ (size * 2_654_435_761) ^ (length * 40_503))
    pool: set[str] = set()
    guard = 0
    while len(pool) < size and guard < size * 50:
        pool.add(make_salt(rng, length))
        guard += 1
    return sorted(pool)


def _salt_prefix(content: str, salt: str) -> str:
    # A bracketed id reads like a request/session tag and leaves the original
    # prompt's semantics (benign or adversarial) intact.
    return f"[{salt}] {content}"


def apply_salt(
    content: str,
    rng: random.Random,
    length: int,
    pool: list[str] | None = None,
) -> str:
    """Prepend a short id to defeat / dampen vLLM prefix caching.

    vLLM hashes the prompt as a chain of fixed-size token blocks from the very
    start, so a varying *prefix* invalidates the prefix-cache chain for the
    request, while a suffix would not.

    * ``pool`` given  → pick a salt from the fixed pool (prefixes repeat → partial
      cache reuse at a tunable rate).
    * else ``length`` > 0 → fresh random salt per request (near-0% reuse).
    * else → returned unchanged.
    """
    if pool:
        return _salt_prefix(content, rng.choice(pool))
    if length > 0:
        return _salt_prefix(content, make_salt(rng, length))
    return content


def model_to_folder(model: str) -> str:
    return model.replace("/", "_")


def resolve_attack_file(model: str = DEFAULT_MODEL, attack_root: str | Path | None = None) -> Path:
    """Locate the stressor suite for ``model``.

    With ``attack_root`` set, looks for a research-repo layout
    (``<root>/<org>_<model>/loopllm/summary.json``). Otherwise resolves the D1
    suite in the fetched ContamBench archive.
    """
    if attack_root is not None:
        summary = Path(attack_root) / model_to_folder(model) / "loopllm" / "summary.json"
        if summary.is_file():
            return summary
        raise FileNotFoundError(f"No summary.json at {summary}")
    from contambench import data

    return data.suite_file(model)


@dataclass
class PromptPair:
    pair_index: int
    prompt: str
    adv_prompt: str
    success_rate: float | None = None
    avg_len: float | None = None


@dataclass
class ScheduledPrompt:
    index: int
    pair_index: int
    is_attack: bool
    content: str


def select_prompt(
    pairs: list["PromptPair"],
    contamination_ratio: float,
    rng: random.Random,
    index: int = 0,
    salt_length: int = 0,
    salt_pool: list[str] | None = None,
) -> ScheduledPrompt:
    """Independently select a benign or adversarial prompt for one request.

    Contamination is *global*: every request is adversarial with probability
    ``contamination_ratio`` and benign otherwise, regardless of which client
    issues it. There are no permanently benign or malicious clients.

    Salting (see :func:`apply_salt`) prepends a short id to both benign and
    adversarial prompts to lower vLLM prefix-cache hits when the prompt pool is
    small: ``salt_pool`` gives partial, tunable reuse; ``salt_length`` alone
    gives a unique id per request.
    """
    if not 0.0 <= contamination_ratio <= 1.0:
        raise ValueError("contamination_ratio must be in [0, 1]")
    if not pairs:
        raise ValueError("no prompt pairs available")
    is_attack = rng.random() < contamination_ratio
    pair = pairs[rng.randrange(len(pairs))]
    content = pair.adv_prompt if is_attack else pair.prompt
    content = apply_salt(content, rng, salt_length, salt_pool)
    return ScheduledPrompt(
        index=index,
        pair_index=pair.pair_index,
        is_attack=is_attack,
        content=content,
    )


def load_prompt_pairs(attack_file: str | Path) -> list[PromptPair]:
    with open(attack_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):  # D1 pairs.json
        entries = data
    else:  # research summary.json ("prompts") or a D1 control file ("entries")
        entries = data.get("prompts") or data.get("entries") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    pairs: list[PromptPair] = []
    for i, entry in enumerate(entries):
        i = int(entry.get("pair_index", i))
        prompt = entry.get("prompt")
        adv_prompt = entry.get("adv_prompt")
        if prompt is None or adv_prompt is None:
            continue
        pairs.append(
            PromptPair(
                pair_index=i,
                prompt=prompt,
                adv_prompt=adv_prompt,
                success_rate=entry.get("success_rate"),
                avg_len=entry.get("avg_len"),
            )
        )
    if not pairs:
        raise ValueError(f"No usable prompt/adv_prompt pairs in {attack_file}")
    return pairs


def mean_success_rate(pairs: list[PromptPair]) -> float | None:
    """Mean ``success_rate`` over pairs that record one; ``None`` if none do."""
    rates = [float(p.success_rate) for p in pairs if p.success_rate is not None]
    if not rates:
        return None
    return sum(rates) / len(rates)


def select_prompt_pairs(pairs: list[PromptPair], workload_size: int | None) -> list[PromptPair]:
    """Keep the top *workload_size* pairs ranked by ``success_rate`` (descending).

    Pairs without a recorded success rate sort last. When *workload_size* is
    ``None`` or not smaller than ``len(pairs)``, all pairs are returned.
    """
    if workload_size is None or workload_size <= 0 or workload_size >= len(pairs):
        return pairs

    def _rate(p: PromptPair) -> float:
        return float(p.success_rate) if p.success_rate is not None else float("-inf")

    ranked = sorted(pairs, key=_rate, reverse=True)
    return ranked[:workload_size]


def load_workload_pairs(
    attack_file: str | Path,
    workload_size: int | None = None,
) -> tuple[list[PromptPair], list[PromptPair]]:
    """Load prompt pairs and optionally restrict to the top *workload_size* by success rate."""
    all_pairs = load_prompt_pairs(attack_file)
    selected = select_prompt_pairs(all_pairs, workload_size)
    return selected, all_pairs


def build_schedule(
    pairs: list[PromptPair],
    num_requests: int,
    contamination_ratio: float = 0.0,
    seed: int = 23,
    salt_length: int = 0,
    salt_pool_size: int = 0,
) -> list[ScheduledPrompt]:
    """Create an interleaved request schedule.

    ``contamination_ratio`` is the fraction of the ``num_requests`` that use the
    adversarial ``adv_prompt``; the rest use the benign ``prompt``. Attack/benign
    positions are shuffled deterministically by ``seed``. Salting prepends a short
    id to lower vLLM prefix-cache hits: ``salt_pool_size`` > 0 gives partial,
    tunable reuse; ``salt_length`` alone gives a unique id per request.
    """
    if not 0.0 <= contamination_ratio <= 1.0:
        raise ValueError("contamination_ratio must be in [0, 1]")
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")

    rng = random.Random(seed)
    salt_pool = make_salt_pool(seed, salt_pool_size, salt_length) or None
    n_attack = round(contamination_ratio * num_requests)
    flags = [True] * n_attack + [False] * (num_requests - n_attack)
    rng.shuffle(flags)

    schedule: list[ScheduledPrompt] = []
    benign_cursor = 0
    attack_cursor = 0
    for i, is_attack in enumerate(flags):
        if is_attack:
            pair = pairs[attack_cursor % len(pairs)]
            attack_cursor += 1
            content = pair.adv_prompt
        else:
            pair = pairs[benign_cursor % len(pairs)]
            benign_cursor += 1
            content = pair.prompt
        content = apply_salt(content, rng, salt_length, salt_pool)
        schedule.append(
            ScheduledPrompt(
                index=i,
                pair_index=pair.pair_index,
                is_attack=is_attack,
                content=content,
            )
        )
    return schedule
