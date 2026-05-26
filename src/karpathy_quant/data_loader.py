"""OHLCV data loading with yfinance + local parquet cache."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    train: pd.DataFrame
    holdout: pd.DataFrame


def load_ohlcv(
    symbol: str,
    period: str = "5y",
    cache_dir: str = ".cache",
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance, caching locally as parquet.

    Returns a DataFrame with columns [open, high, low, close, volume]
    and a DatetimeIndex.
    """
    cache_path = _cache_path(symbol, period, cache_dir)

    if cache_path.exists():
        logger.info("Loading cached data for %s from %s", symbol, cache_path)
        df = pd.read_parquet(cache_path)
        return _normalize(df)

    logger.info("Fetching %s data from yfinance (period=%s)", symbol, period)
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")

    df = _normalize(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Cached %d bars for %s", len(df), symbol)
    return df


def split_data(df: pd.DataFrame, holdout_pct: float = 0.2) -> DataSplit:
    """Split data into train and holdout sets (contiguous time split)."""
    if not 0 < holdout_pct < 1:
        raise ValueError(f"holdout_pct must be between 0 and 1, got {holdout_pct}")

    split_idx = int(len(df) * (1 - holdout_pct))
    return DataSplit(
        train=df.iloc[:split_idx].copy(),
        holdout=df.iloc[split_idx:].copy(),
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, check required fields, strip tz."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["open", "high", "low", "close", "volume"]]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def _cache_path(symbol: str, period: str, cache_dir: str) -> Path:
    """Deterministic cache path from symbol + period."""
    key = f"{symbol.upper()}_{period}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]  # noqa: S324
    return Path(cache_dir) / f"{symbol.upper()}_{period}_{h}.parquet"
