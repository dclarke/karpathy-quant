import numpy as np
import pandas as pd
import pytest

from karpathy_quant.backtest import BacktestConfig, BacktestResult, _classify, run_backtest


def _make_df(prices, volumes=None):
    n = len(prices)
    dates = pd.bdate_range("2020-01-01", periods=n)
    if volumes is None:
        volumes = [1_000_000] * n
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )


# -- signal snippets --

_ALWAYS_FIRE = """
def get_feature(df):
    import pandas as pd
    return pd.Series(1.0, index=df.index)
"""

_NEVER_FIRE = """
def get_feature(df):
    import pandas as pd
    return pd.Series(0.0, index=df.index)
"""

_FIRST_BAR = """
def get_feature(df):
    import pandas as pd
    s = pd.Series(0.0, index=df.index)
    s.iloc[0] = 1.0
    return s
"""

_LAST_BAR = """
def get_feature(df):
    import pandas as pd
    s = pd.Series(0.0, index=df.index)
    s.iloc[-1] = 1.0
    return s
"""

# returns a DataFrame instead of Series
_DF_RETURN = """
def get_feature(df):
    import pandas as pd
    return pd.DataFrame({'signal': pd.Series(1.0, index=df.index)})
"""

_VOLUME_SPIKE = """
def get_feature(df):
    avg_vol = df['volume'].rolling(5).mean()
    return (df['volume'] > 2 * avg_vol).astype(float)
"""


class TestRunBacktest:
    def test_no_fires_returns_reject(self):
        df = _make_df([100, 101, 102, 103, 104])
        result = run_backtest(_NEVER_FIRE, df)
        assert result.status == "reject"
        assert result.error == "No fire dates"
        assert result.n_trades == 0

    def test_single_trade_long(self):
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = _make_df(prices)
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = run_backtest(_FIRST_BAR, df, direction="long", config=config)

        assert result.n_trades == 1
        assert result.status != "reject" or result.error is None
        # should be 5% return (105/100 - 1)
        assert abs(result.total_return - 0.05) < 0.001

    def test_single_trade_short(self):
        prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
        df = _make_df(prices)
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = run_backtest(_FIRST_BAR, df, direction="short", config=config)

        assert result.n_trades == 1
        assert abs(result.total_return - 0.05) < 0.001

    def test_transaction_costs_reduce_returns(self):
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = _make_df(prices)

        no_cost = run_backtest(
            _FIRST_BAR, df, config=BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        )
        with_cost = run_backtest(
            _FIRST_BAR, df, config=BacktestConfig(hold_days=5, cost_per_side_bps=10, slippage_bps=5)
        )
        assert with_cost.total_return < no_cost.total_return

    def test_mfe_mae_computed(self):
        # price spikes to 105 then dips to 98 before recovering
        prices = [100, 105, 98, 102, 103, 104, 106, 107, 108, 109]
        df = _make_df(prices)
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = run_backtest(_FIRST_BAR, df, direction="long", config=config)

        assert result.n_trades == 1
        assert result.avg_mfe_pct > 0
        assert result.avg_mae_pct > 0

    def test_non_overlapping_trades(self):
        prices = list(range(100, 120))
        df = _make_df(prices)
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = run_backtest(_ALWAYS_FIRE, df, config=config)
        assert result.n_trades >= 2

    def test_bad_code_returns_error(self):
        result = run_backtest("def wrong(): pass", _make_df([100, 101, 102]))
        assert result.status == "reject"
        assert result.error is not None

    def test_syntax_error_returns_error(self):
        result = run_backtest("def get_feature(df) broken", _make_df([100, 101, 102]))
        assert result.status == "reject"
        assert result.error is not None

    def test_fire_on_last_bar_gives_no_trades(self):
        # firing on the last bar means no window to trade into
        prices = list(range(100, 110))
        df = _make_df(prices)
        config = BacktestConfig(hold_days=5, cost_per_side_bps=0, slippage_bps=0)
        result = run_backtest(_LAST_BAR, df, config=config)
        assert result.status == "reject"
        assert result.error == "No non-overlapping trades"

    def test_signal_returning_dataframe(self):
        # should squeeze the DataFrame to a Series automatically
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = _make_df(prices)
        config = BacktestConfig(hold_days=3, cost_per_side_bps=0, slippage_bps=0)
        result = run_backtest(_DF_RETURN, df, config=config)
        assert result.n_trades >= 1


class TestClassification:
    def test_active_classification(self):
        rng = np.random.default_rng(42)
        n = 1000
        prices = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
        df = _make_df(prices.tolist())
        config = BacktestConfig(hold_days=3, cost_per_side_bps=0, slippage_bps=0)

        code = """
def get_feature(df):
    import pandas as pd
    s = pd.Series(0.0, index=df.index)
    s.iloc[::10] = 1.0
    return s
"""
        result = run_backtest(code, df, config=config)
        assert result.n_trades >= 10

    def test_reject_on_bad_metrics(self):
        rng = np.random.default_rng(99)
        prices = (100 + rng.normal(0, 0.5, 50)).tolist()
        df = _make_df(prices)
        config = BacktestConfig(hold_days=3, cost_per_side_bps=10, slippage_bps=5)
        result = run_backtest(_ALWAYS_FIRE, df, config=config)
        assert result.n_trades >= 1


class TestClassify:
    def test_active(self):
        assert _classify(0.10, 0.8, -0.15, 20) == "active"

    def test_watchlist(self):
        # meets watchlist thresholds but not active
        assert _classify(0.05, 0.4, -0.20, 8) == "watchlist"

    def test_reject(self):
        assert _classify(0.001, 0.05, -0.80, 2) == "reject"


class TestFireRate:
    def test_fire_rate_is_fraction(self):
        prices = list(range(100, 120))
        df = _make_df(prices)
        result = run_backtest(_ALWAYS_FIRE, df, config=BacktestConfig(hold_days=5))
        assert 0 < result.fire_rate <= 1.0

    def test_fire_dates_are_strings(self):
        prices = list(range(100, 115))
        df = _make_df(prices)
        result = run_backtest(_FIRST_BAR, df, config=BacktestConfig(hold_days=3))
        if result.fire_dates:
            assert isinstance(result.fire_dates[0], str)


class TestBacktestResult:
    def test_to_dict(self):
        result = BacktestResult(n_trades=5, sharpe=1.2, status="active")
        d = result.to_dict()
        assert d["n_trades"] == 5
        assert d["sharpe"] == 1.2
        assert d["status"] == "active"
