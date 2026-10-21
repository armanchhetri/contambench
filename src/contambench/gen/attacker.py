"""Attacker base class (ported from the research repo's ``src/attacks/attacker.py``)."""
from __future__ import annotations

from contambench.gen import require_extra

require_extra()

from abc import ABC, abstractmethod  # noqa: E402
from typing import Generic, List, TypeVar  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from contambench.gen.models import Model  # noqa: E402


class AttackParams(BaseModel):
    pass


T = TypeVar("T", bound=AttackParams)


class Attacker(ABC, Generic[T]):
    def __init__(self, model: Model):
        self.model = model
        self._attack_prompts: List[dict] = []

    @abstractmethod
    def attack(self, params: T, log_prompts=False, gpu_threshold=85):
        raise NotImplementedError("implement the attack")

    @abstractmethod
    def get_attack_prompts(self, count: int):
        raise NotImplementedError("implement the attack")

    def clear_attack(self):
        self._attack_prompts = []
