"""Integration tests for LLM hypothesis generation — requires API key."""

import os
import tempfile

import pandas as pd
import pytest

from karpathy_quant.data_loader import load_ohlcv
from karpathy_quant.hypothesis import create_provider, generate_hypotheses


@pytest.mark.integration
class TestGeminiGeneration:
    @pytest.fixture(autouse=True)
    def _check_api_key(self) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    @pytest.mark.asyncio
    async def test_generates_valid_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            df = load_ohlcv("AAPL", period="1mo", cache_dir=cache_dir)

        provider = create_provider("gemini")
        results = await generate_hypotheses(provider, "AAPL", df, n_hypotheses=3)

        assert len(results) >= 1
        for h in results:
            assert "get_feature" in h.code
            assert h.direction in ("long", "short")
            assert len(h.theory) > 0
