"""Integration tests for data loader — actually hits yfinance."""

import tempfile

import pandas as pd
import pytest

from karpathy_quant.data_loader import load_ohlcv, split_data


@pytest.mark.integration
class TestYfinanceLoader:
    def test_fetch_aapl(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            df = load_ohlcv("AAPL", period="1mo", cache_dir=cache_dir)
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 15  # ~20 trading days in a month
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_caching_works(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            df1 = load_ohlcv("AAPL", period="1mo", cache_dir=cache_dir)
            df2 = load_ohlcv("AAPL", period="1mo", cache_dir=cache_dir)
            pd.testing.assert_frame_equal(df1, df2)

    def test_split_real_data(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            df = load_ohlcv("AAPL", period="3mo", cache_dir=cache_dir)
            split = split_data(df, holdout_pct=0.2)
            assert len(split.train) + len(split.holdout) == len(df)
            assert split.train.index[-1] < split.holdout.index[0]
