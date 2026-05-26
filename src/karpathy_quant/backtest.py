"""Walk-forward backtester with MFE/MAE and Sharpe."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_ANNUALIZE_DAYS = 180


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    hold_days: int = 5
    cost_per_side_bps: float = 5.0
    slippage_bps: float = 2.0


@dataclass
class BacktestResult:
    """Full metrics from a backtest run."""

    n_trades: int = 0
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_mfe_pct: float = 0.0
    p50_mfe_pct: float = 0.0
    p75_mfe_pct: float = 0.0
    p90_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    p90_mae_pct: float = 0.0
    optimal_hold_bars: int = 0
    fire_rate: float = 0.0
    fire_dates: list[str] = field(default_factory=list)
    status: str = "reject"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def run_backtest(
    code: str,
    df: pd.DataFrame,
    direction: str = "long",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a walk-forward backtest for a single signal hypothesis.

    The signal code must define a `get_feature(df) -> pd.Series` function
    that returns values > 0.5 to indicate a trade entry.
    """
    config = config or BacktestConfig()
    round_trip_cost = 2.0 * (config.cost_per_side_bps + config.slippage_bps) / 10_000

    try:
        fire_series = _execute_signal(code, df)
        fire_dates = fire_series[fire_series].index

        if len(fire_dates) == 0:
            return BacktestResult(error="No fire dates", status="reject")

        trades = _simulate_trades(
            fire_dates=fire_dates,
            close=df["close"].astype(float),
            direction=direction,
            hold_bars=config.hold_days,
            round_trip_cost=round_trip_cost,
        )

        if not trades:
            return BacktestResult(error="No non-overlapping trades", status="reject")

        return _compute_metrics(trades, len(df), int(fire_series.sum()))

    except Exception as e:
        return BacktestResult(error=str(e), status="reject")


def _execute_signal(code: str, df: pd.DataFrame) -> pd.Series:
    """Run the user's signal code and get a boolean fire mask."""
    exec_globals: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "ohlcv_data": df,
        "close": df["close"],
        "high": df["high"],
        "low": df["low"],
        "open": df["open"],
        "volume": df["volume"],
    }
    local_env: dict[str, Any] = {}
    exec(code, exec_globals, local_env)  # noqa: S102

    feature_func = local_env.get("get_feature")
    if not feature_func:
        raise ValueError("Signal code must define a get_feature(df) function")

    raw = feature_func(df)
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]

    return (raw > 0.5).reindex(df.index).fillna(False)


def _simulate_trades(
    fire_dates: pd.Index,
    close: pd.Series,
    direction: str,
    hold_bars: int,
    round_trip_cost: float,
) -> list[dict[str, Any]]:
    """Simulate non-overlapping trades with MFE/MAE tracking."""
    trades: list[dict[str, Any]] = []
    position_end = pd.Timestamp("1900-01-01")

    for date in fire_dates:
        if date <= position_end or date not in close.index:
            continue

        idx = close.index.get_loc(date)
        exit_idx = min(idx + hold_bars, len(close) - 1)
        exit_date = close.index[exit_idx]
        entry_price = float(close.iloc[idx])

        window_closes = close.iloc[idx + 1 : exit_idx + 1]
        if len(window_closes) == 0:
            continue

        pct_changes = window_closes / entry_price - 1

        if direction == "short":
            mfe = float(abs(pct_changes.min()))
            mae = float(max(pct_changes.max(), 0.0))
            opt_bar = int(pct_changes.values.argmin()) + 1
            ret = -(float(close.iloc[exit_idx] / entry_price - 1)) - round_trip_cost
        else:
            mfe = float(pct_changes.max())
            mae = float(abs(pct_changes.min()))
            opt_bar = int(pct_changes.values.argmax()) + 1
            ret = float(close.iloc[exit_idx] / entry_price - 1) - round_trip_cost

        trades.append({
            "entry": date,
            "exit": exit_date,
            "return": ret,
            "mfe": mfe,
            "mae": mae,
            "optimal_bar": opt_bar,
        })
        position_end = exit_date

    return trades


def _annualize_sharpe(rets: pd.Series, date_range_days: int) -> float:
    """Annualize Sharpe based on how often trades actually happen."""
    n = len(rets)
    if n <= 1:
        return 0.0
    per_trade = float(rets.mean() / (rets.std() + 1e-9))
    if date_range_days < _MIN_ANNUALIZE_DAYS:
        return per_trade
    years = date_range_days / 365.25
    if years <= 0:
        return per_trade
    trades_per_year = n / years
    return per_trade * float(np.sqrt(trades_per_year))


def _compute_metrics(
    trades: list[dict[str, Any]],
    total_bars: int,
    fire_count: int,
) -> BacktestResult:
    """Roll up trade-level data into aggregate metrics."""
    rets = pd.Series([t["return"] for t in trades])
    mae_list = [t["mae"] for t in trades]
    mfe_list = [t["mfe"] for t in trades]
    opt_bars = [t["optimal_bar"] for t in trades]

    equity = (1 + rets).cumprod()
    max_dd = float(((equity - equity.cummax()) / equity.cummax()).min())
    total_ret = float((1 + rets).prod() - 1)

    date_range_days = (trades[-1]["exit"] - trades[0]["entry"]).days
    years = date_range_days / 365.25
    annual_ret = (
        float((1 + total_ret) ** (1 / years) - 1)
        if date_range_days >= _MIN_ANNUALIZE_DAYS
        else total_ret
    )

    win_rate = float((rets > 0).mean())
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    sharpe = _annualize_sharpe(rets, date_range_days)

    status = _classify(annual_ret, sharpe, max_dd, len(trades))

    return BacktestResult(
        n_trades=len(trades),
        total_return=round(total_ret, 6),
        annual_return=round(annual_ret, 6),
        sharpe=round(sharpe, 3),
        win_rate_pct=round(win_rate * 100, 1),
        avg_win_pct=round(avg_win * 100, 3),
        avg_loss_pct=round(avg_loss * 100, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        avg_mfe_pct=round(float(np.mean(mfe_list)) * 100, 2),
        p50_mfe_pct=round(float(np.percentile(mfe_list, 50)) * 100, 2),
        p75_mfe_pct=round(float(np.percentile(mfe_list, 75)) * 100, 2),
        p90_mfe_pct=round(float(np.percentile(mfe_list, 90)) * 100, 2),
        avg_mae_pct=round(float(np.mean(mae_list)) * 100, 2),
        p90_mae_pct=round(float(np.percentile(mae_list, 90)) * 100, 2),
        optimal_hold_bars=statistics.mode(opt_bars),
        fire_rate=round(fire_count / total_bars if total_bars > 0 else 0.0, 6),
        fire_dates=[str(t["entry"].date()) for t in trades],
        status=status,
    )


def _classify(annual_ret: float, sharpe: float, max_dd: float, n_trades: int) -> str:
    """Bucket into active / watchlist / reject based on thresholds."""
    if annual_ret >= 0.08 and sharpe >= 0.5 and max_dd >= -0.30 and n_trades >= 10:
        return "active"
    if annual_ret >= 0.03 and sharpe >= 0.3 and max_dd >= -0.50 and n_trades >= 5:
        return "watchlist"
    return "reject"
