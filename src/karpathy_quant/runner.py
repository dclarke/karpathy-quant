"""Main discovery loop — generate hypotheses, backtest, gate, repeat."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .backtest import BacktestConfig
from .data_loader import load_ohlcv, split_data
from .gatekeeper import GateResult, evaluate
from .hypothesis import create_provider, generate_hypotheses
from .providers.base import Hypothesis

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    hypothesis: Hypothesis
    gate_result: GateResult


@dataclass
class RunConfig:
    symbol: str
    iterations: int = 50
    hypotheses_per_call: int = 10
    hold_days: int = 5
    cost_per_side_bps: float = 5.0
    slippage_bps: float = 2.0
    holdout_pct: float = 0.2
    data_period: str = "5y"
    provider: str = "gemini"
    provider_kwargs: dict = field(default_factory=dict)


async def run(config: RunConfig) -> list[DiscoveryResult]:
    """Run the full generate -> backtest -> gate loop."""
    logger.info(
        "Starting discovery for %s (%d iterations, provider=%s)",
        config.symbol,
        config.iterations,
        config.provider,
    )

    df = load_ohlcv(config.symbol, period=config.data_period)
    data = split_data(df, holdout_pct=config.holdout_pct)

    provider = create_provider(config.provider, **config.provider_kwargs)
    bt_config = BacktestConfig(
        hold_days=config.hold_days,
        cost_per_side_bps=config.cost_per_side_bps,
        slippage_bps=config.slippage_bps,
    )

    results: list[DiscoveryResult] = []
    saturated: list[str] = []
    hall_of_fame: list[str] = []
    hypothesis_count = 0

    n_calls = math.ceil(config.iterations / config.hypotheses_per_call)

    for call_num in range(n_calls):
        batch_size = min(
            config.hypotheses_per_call,
            config.iterations - hypothesis_count,
        )

        logger.info(
            "Batch %d/%d: generating %d hypotheses",
            call_num + 1,
            n_calls,
            batch_size,
        )

        hypotheses = await generate_hypotheses(
            provider=provider,
            symbol=config.symbol,
            df=data.train,
            n_hypotheses=batch_size,
            saturated_theories=saturated,
            hall_of_fame=hall_of_fame,
        )

        if not hypotheses:
            logger.warning("No hypotheses returned for batch %d", call_num + 1)
            hypothesis_count += batch_size
            continue

        for h in hypotheses:
            if hypothesis_count >= config.iterations:
                break
            hypothesis_count += 1

            gate_result = evaluate(
                code=h.code,
                train_df=data.train,
                holdout_df=data.holdout,
                direction=h.direction,
                config=bt_config,
            )

            if gate_result.status != "reject":
                results.append(DiscoveryResult(hypothesis=h, gate_result=gate_result))
                hall_of_fame.append(h.theory)
                logger.info(
                    "PASS [%s] %s (sharpe=%.2f, return=%.1f%%)",
                    gate_result.status,
                    h.theory[:60],
                    gate_result.wf_metrics.sharpe,
                    gate_result.wf_metrics.annual_return * 100,
                )
            else:
                saturated.append(h.theory)

    logger.info(
        "Discovery complete: %d/%d hypotheses passed",
        len(results),
        hypothesis_count,
    )
    return results
