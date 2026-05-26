import numpy as np
import pandas as pd
import pytest

from karpathy_quant.backtest import BacktestConfig
from karpathy_quant.gatekeeper import GateResult, _combine, evaluate


def _make_df(n, drift=0.05):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(drift, 0.5, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.5, n),
            "high": close + rng.uniform(0, 1, n),
            "low": close - rng.uniform(0, 1, n),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


_PERIODIC_SIGNAL = """
def get_feature(df):
    import pandas as pd
    s = pd.Series(0.0, index=df.index)
    s.iloc[::15] = 1.0
    return s
"""

_NEVER_FIRE = """
def get_feature(df):
    import pandas as pd
    return pd.Series(0.0, index=df.index)
"""


class TestCombine:
    def test_both_active(self):
        assert _combine("active", "active") == "active"

    def test_active_watchlist(self):
        assert _combine("active", "watchlist") == "watchlist"

    def test_watchlist_active(self):
        assert _combine("watchlist", "active") == "watchlist"

    def test_any_reject(self):
        assert _combine("active", "reject") == "reject"
        assert _combine("reject", "active") == "reject"
        assert _combine("reject", "reject") == "reject"


class TestEvaluate:
    def test_reject_when_no_fires(self):
        train = _make_df(200)
        holdout = _make_df(50)
        result = evaluate(_NEVER_FIRE, train, holdout)
        assert result.status == "reject"
        assert result.holdout_metrics is None  # should short-circuit

    def test_passes_through_with_good_signal(self):
        train = _make_df(500, drift=0.1)
        holdout = _make_df(100, drift=0.1)
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = evaluate(_PERIODIC_SIGNAL, train, holdout, config=config)

        assert isinstance(result, GateResult)
        assert result.wf_metrics.n_trades >= 1
        if result.status != "reject":
            assert result.holdout_metrics is not None

    def test_empty_holdout_skips_stage2(self):
        train = _make_df(500, drift=0.1)
        holdout = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = evaluate(_PERIODIC_SIGNAL, train, holdout, config=config)
        assert result.holdout_metrics is None

    def test_bad_code_returns_reject(self):
        result = evaluate("not valid python;;;", _make_df(100), _make_df(50))
        assert result.status == "reject"

    def test_both_stages_pass(self):
        # strong uptrend, no costs, frequent signals — both stages should pass
        train = _make_df(600, drift=0.15)
        holdout = _make_df(200, drift=0.15)
        config = BacktestConfig(hold_days=3, cost_per_side_bps=0, slippage_bps=0)
        signal = """
def get_feature(df):
    import pandas as pd
    s = pd.Series(0.0, index=df.index)
    s.iloc[::7] = 1.0
    return s
"""
        result = evaluate(signal, train, holdout, config=config)
        assert result.holdout_metrics is not None
        assert result.status in ("active", "watchlist")
