from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from karpathy_quant.data_loader import DataSplit, _normalize, load_ohlcv, split_data


def _make_ohlcv(n=100):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


class TestNormalize:
    def test_lowercases_columns(self):
        df = _make_ohlcv()
        df.columns = [c.upper() for c in df.columns]
        result = _normalize(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_strips_timezone(self):
        df = _make_ohlcv()
        df.index = df.index.tz_localize("US/Eastern")
        result = _normalize(df)
        assert result.index.tz is None

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"open": [1], "close": [2]}, index=pd.to_datetime(["2020-01-01"]))
        with pytest.raises(ValueError, match="Missing required columns"):
            _normalize(df)

    def test_selects_only_ohlcv(self):
        df = _make_ohlcv()
        df["extra"] = 999
        result = _normalize(df)
        assert "extra" not in result.columns

    def test_string_index_gets_converted(self):
        df = pd.DataFrame(
            {"open": [1, 2], "high": [2, 3], "low": [0.5, 1], "close": [1.5, 2.5], "volume": [100, 200]},
            index=["2020-01-02", "2020-01-03"],
        )
        result = _normalize(df)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestSplitData:
    def test_split_proportions(self):
        df = _make_ohlcv(100)
        split = split_data(df, holdout_pct=0.2)
        assert isinstance(split, DataSplit)
        assert len(split.train) == 80
        assert len(split.holdout) == 20

    def test_split_is_contiguous(self):
        df = _make_ohlcv(100)
        split = split_data(df, holdout_pct=0.3)
        assert split.train.index[-1] < split.holdout.index[0]

    def test_invalid_holdout_pct(self):
        df = _make_ohlcv()
        with pytest.raises(ValueError, match="holdout_pct must be between"):
            split_data(df, holdout_pct=0.0)
        with pytest.raises(ValueError, match="holdout_pct must be between"):
            split_data(df, holdout_pct=1.0)

    def test_copies_data(self):
        df = _make_ohlcv(50)
        split = split_data(df, holdout_pct=0.2)
        # mutating the split shouldn't affect the original
        split.train.iloc[0, 0] = -9999
        assert df.iloc[0, 0] != -9999


class TestLoadOhlcv:
    @patch("karpathy_quant.data_loader.Path.exists", return_value=False)
    def test_fetches_and_caches(self, _mock_exists):
        mock_df = _make_ohlcv(50)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch("pandas.DataFrame.to_parquet"),
            patch("karpathy_quant.data_loader.Path.mkdir"),
        ):
            result = load_ohlcv("AAPL", period="1y", cache_dir="/tmp/test_cache")

        assert len(result) == 50
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        mock_ticker.history.assert_called_once_with(period="1y", auto_adjust=True)

    @patch("karpathy_quant.data_loader.Path.exists", return_value=True)
    def test_loads_from_cache(self, _mock_exists):
        mock_df = _make_ohlcv(30)
        with patch("pandas.read_parquet", return_value=mock_df):
            result = load_ohlcv("AAPL", period="1y", cache_dir="/tmp/test_cache")
        assert len(result) == 30

    @patch("karpathy_quant.data_loader.Path.exists", return_value=False)
    def test_raises_on_empty_data(self, _mock_exists):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            pytest.raises(ValueError, match="No data returned"),
        ):
            load_ohlcv("FAKESYMBOL", cache_dir="/tmp/test_cache")
