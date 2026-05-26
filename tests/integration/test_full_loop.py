"""Integration test — full discovery loop with real data and LLM."""

import os
import tempfile

import pytest

from karpathy_quant.runner import RunConfig, run


@pytest.mark.integration
class TestFullLoop:
    @pytest.fixture(autouse=True)
    def _check_api_key(self) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    @pytest.mark.asyncio
    async def test_end_to_end_discovery(self) -> None:
        config = RunConfig(
            symbol="AAPL",
            iterations=5,
            hypotheses_per_call=5,
            provider="gemini",
            data_period="1y",
            hold_days=5,
        )

        results = await run(config)

        # We don't assert that hypotheses pass (that depends on the LLM
        # and market data), but the loop should complete without errors
        assert isinstance(results, list)
        for r in results:
            assert r.gate_result.status in ("active", "watchlist")
            assert r.gate_result.wf_metrics.n_trades >= 1
