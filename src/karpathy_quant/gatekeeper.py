"""Two-stage gatekeeper: walk-forward + holdout validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest


@dataclass
class GateResult:
    status: str  # "active", "watchlist", "reject"
    wf_metrics: BacktestResult
    holdout_metrics: BacktestResult | None = None


def evaluate(
    code: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    direction: str = "long",
    config: BacktestConfig | None = None,
) -> GateResult:
    """Run walk-forward backtest on train data, then validate on holdout.

    Stage 1 (walk-forward): backtest on training data, classify.
    Stage 2 (holdout): if walk-forward passes, re-run on holdout data.
    Final status is the worse of the two stages.
    """
    config = config or BacktestConfig()

    # Stage 1: Walk-forward on training data
    wf_result = run_backtest(code, train_df, direction, config)

    if wf_result.status == "reject":
        return GateResult(status="reject", wf_metrics=wf_result)

    # Stage 2: Holdout validation
    if len(holdout_df) == 0:
        return GateResult(status=wf_result.status, wf_metrics=wf_result)

    ho_result = run_backtest(code, holdout_df, direction, config)

    if ho_result.status == "reject":
        return GateResult(
            status="reject",
            wf_metrics=wf_result,
            holdout_metrics=ho_result,
        )

    # Final status = worst of the two
    combined = _combine(wf_result.status, ho_result.status)
    return GateResult(
        status=combined,
        wf_metrics=wf_result,
        holdout_metrics=ho_result,
    )


def _combine(wf_status: str, ho_status: str) -> str:
    """Pick the worse of two statuses."""
    order = ["reject", "watchlist", "active"]
    a = order.index(wf_status) if wf_status in order else 0
    b = order.index(ho_status) if ho_status in order else 0
    return order[min(a, b)]
