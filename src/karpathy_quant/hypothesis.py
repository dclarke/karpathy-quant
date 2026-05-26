"""Hypothesis generation and provider wiring."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .providers.base import Hypothesis, HypothesisRequest, Provider

logger = logging.getLogger(__name__)


def create_provider(name: str, **kwargs: Any) -> Provider:
    """Create an LLM provider by name."""
    if name == "gemini":
        from .providers.gemini import GeminiProvider

        return GeminiProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}. Use 'gemini'.")


def compute_ohlcv_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Summary stats to inject into the LLM prompt."""
    desc = df[["open", "high", "low", "close", "volume"]].describe()
    stats: dict[str, Any] = {
        col: {k: round(v, 4) for k, v in desc[col].items()}
        for col in desc.columns
    }
    stats["n_bars"] = len(df)
    return stats


async def generate_hypotheses(
    provider: Provider,
    symbol: str,
    df: pd.DataFrame,
    n_hypotheses: int = 10,
    saturated_theories: list[str] | None = None,
    hall_of_fame: list[str] | None = None,
) -> list[Hypothesis]:
    """Ask the LLM to produce hypotheses and handle failures."""
    stats = compute_ohlcv_stats(df)
    request = HypothesisRequest(
        symbol=symbol,
        ohlcv_stats=stats,
        n_hypotheses=n_hypotheses,
        saturated_theories=saturated_theories or [],
        hall_of_fame=hall_of_fame or [],
    )

    try:
        hypotheses = await provider.generate(request)
        logger.info(
            "Generated %d/%d hypotheses for %s",
            len(hypotheses),
            n_hypotheses,
            symbol,
        )
        return hypotheses
    except Exception as e:
        logger.warning("Hypothesis generation failed: %s", e)
        return []
