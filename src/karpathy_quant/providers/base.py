"""Base types for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HypothesisRequest:
    symbol: str
    ohlcv_stats: dict[str, Any]
    n_hypotheses: int = 10
    saturated_theories: list[str] = field(default_factory=list)
    hall_of_fame: list[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    code: str
    theory: str
    direction: str = "long"


class Provider(ABC):
    @abstractmethod
    async def generate(self, request: HypothesisRequest) -> list[Hypothesis]: ...
