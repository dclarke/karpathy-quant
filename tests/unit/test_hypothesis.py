from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from karpathy_quant.hypothesis import compute_ohlcv_stats, create_provider, generate_hypotheses
from karpathy_quant.providers.base import Hypothesis, HypothesisRequest, Provider


def _make_df(n=100):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


class MockProvider(Provider):
    def __init__(self, hypotheses=None):
        self._hypotheses = hypotheses or [
            Hypothesis(
                code='def get_feature(df):\n    return (df["close"].pct_change() > 0.02).astype(float)',
                theory="Buy on 2% daily moves",
                direction="long",
            ),
        ]

    async def generate(self, request):
        return self._hypotheses[: request.n_hypotheses]


class TestComputeOhlcvStats:
    def test_has_required_keys(self):
        stats = compute_ohlcv_stats(_make_df())
        assert "close" in stats
        assert "volume" in stats
        assert "n_bars" in stats
        assert stats["n_bars"] == 100

    def test_stats_have_describe_keys(self):
        stats = compute_ohlcv_stats(_make_df())
        assert "mean" in stats["close"]
        assert "std" in stats["close"]


class TestCreateProvider:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")

    def test_gemini_import(self):
        try:
            provider = create_provider("gemini", api_key="test-key")
            assert provider is not None
        except ImportError:
            pytest.skip("google-genai not installed")


class TestGenerateHypotheses:
    @pytest.mark.asyncio
    async def test_returns_hypotheses(self):
        provider = MockProvider()
        results = await generate_hypotheses(provider, "AAPL", _make_df(), n_hypotheses=1)
        assert len(results) == 1
        assert "get_feature" in results[0].code

    @pytest.mark.asyncio
    async def test_respects_n_hypotheses(self):
        hypotheses = [
            Hypothesis(code=f"def get_feature(df): return df['close'] * {i}", theory=f"H{i}", direction="long")
            for i in range(5)
        ]
        provider = MockProvider(hypotheses)
        results = await generate_hypotheses(provider, "AAPL", _make_df(), n_hypotheses=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_handles_provider_failure(self):
        provider = MockProvider()
        provider.generate = AsyncMock(side_effect=Exception("LLM down"))
        results = await generate_hypotheses(provider, "AAPL", _make_df(), n_hypotheses=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_passes_saturated_and_hof(self):
        provider = MockProvider()
        provider.generate = AsyncMock(return_value=[])

        await generate_hypotheses(
            provider, "AAPL", _make_df(),
            saturated_theories=["bad theory"],
            hall_of_fame=["good theory"],
        )

        req = provider.generate.call_args[0][0]
        assert "bad theory" in req.saturated_theories
        assert "good theory" in req.hall_of_fame
