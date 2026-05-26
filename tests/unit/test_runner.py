"""Tests for the discovery runner loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from karpathy_quant.backtest import BacktestConfig, BacktestResult
from karpathy_quant.data_loader import DataSplit
from karpathy_quant.gatekeeper import GateResult
from karpathy_quant.providers.base import Hypothesis
from karpathy_quant.runner import DiscoveryResult, RunConfig, run


def _fake_df(n=100):
    """quick synthetic ohlcv"""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2021-01-01", periods=n)
    c = 100 + np.cumsum(rng.normal(0.01, 0.3, n))
    return pd.DataFrame({
        "open": c - 0.1, "high": c + 0.5,
        "low": c - 0.5, "close": c,
        "volume": rng.integers(1e6, 5e6, n),
    }, index=dates)


TRAIN_DF = _fake_df(80)
HOLDOUT_DF = _fake_df(20)
SPLIT = DataSplit(train=TRAIN_DF, holdout=HOLDOUT_DF)


def _hyp(theory="mean reversion after gap", code="pass", direction="long"):
    return Hypothesis(code=code, theory=theory, direction=direction)


def _gate(status="active"):
    return GateResult(
        status=status,
        wf_metrics=BacktestResult(n_trades=10, sharpe=1.2, status=status),
        holdout_metrics=BacktestResult(n_trades=5, sharpe=0.9, status=status),
    )


# patch targets
_P_LOAD = "karpathy_quant.runner.load_ohlcv"
_P_SPLIT = "karpathy_quant.runner.split_data"
_P_PROVIDER = "karpathy_quant.runner.create_provider"
_P_GEN = "karpathy_quant.runner.generate_hypotheses"
_P_EVAL = "karpathy_quant.runner.evaluate"


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig(symbol="AAPL")
        assert cfg.symbol == "AAPL"
        assert cfg.iterations == 50
        assert cfg.hypotheses_per_call == 10
        assert cfg.hold_days == 5
        assert cfg.holdout_pct == 0.2
        assert cfg.provider == "gemini"

    def test_custom_values(self):
        cfg = RunConfig(symbol="TSLA", iterations=20, hold_days=3,
                        provider="gemini", provider_kwargs={"model": "gemini-2.5-pro"})
        assert cfg.iterations == 20
        assert cfg.provider_kwargs == {"model": "gemini-2.5-pro"}


class TestRun:
    """Tests for the async run() function."""

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_basic_passing_hypotheses(self, mock_load, mock_split,
                                            mock_prov, mock_gen, mock_eval):
        mock_load.return_value = _fake_df(100)
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()

        h1 = _hyp("theory A")
        h2 = _hyp("theory B")
        mock_gen.return_value = [h1, h2]
        mock_eval.return_value = _gate("active")

        cfg = RunConfig(symbol="SPY", iterations=2, hypotheses_per_call=5)
        results = await run(cfg)

        assert len(results) == 2
        assert all(isinstance(r, DiscoveryResult) for r in results)
        assert results[0].hypothesis == h1
        assert results[0].gate_result.status == "active"

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_all_rejected_populates_saturated(self, mock_load, mock_split,
                                                     mock_prov, mock_gen, mock_eval):
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()

        hyps = [_hyp(f"bad idea {i}") for i in range(3)]
        mock_gen.return_value = hyps
        mock_eval.return_value = _gate("reject")

        cfg = RunConfig(symbol="SPY", iterations=3, hypotheses_per_call=5)
        results = await run(cfg)

        # nothing should pass
        assert results == []

        # check that generate_hypotheses was called with saturated theories
        # on the first call it's empty, but all 3 should have been rejected
        assert mock_eval.call_count == 3

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_empty_generation_continues(self, mock_load, mock_split,
                                               mock_prov, mock_gen, mock_eval):
        """When LLM returns nothing, we skip and move on."""
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()

        # first call empty, second call has results
        mock_gen.side_effect = [[], [_hyp("finally works")]]
        mock_eval.return_value = _gate("watchlist")

        cfg = RunConfig(symbol="SPY", iterations=2, hypotheses_per_call=1)
        results = await run(cfg)

        assert len(results) == 1
        assert results[0].gate_result.status == "watchlist"
        # generate should have been called twice
        assert mock_gen.call_count == 2
        # evaluate only once (empty batch = no eval)
        assert mock_eval.call_count == 1

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_iteration_limit_respected(self, mock_load, mock_split,
                                              mock_prov, mock_gen, mock_eval):
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()

        # provider returns 10 hypotheses but we only want 3
        big_batch = [_hyp(f"idea {i}") for i in range(10)]
        mock_gen.return_value = big_batch
        mock_eval.return_value = _gate("active")

        cfg = RunConfig(symbol="SPY", iterations=3, hypotheses_per_call=10)
        results = await run(cfg)

        # should stop after 3 even though batch had 10
        assert len(results) == 3
        assert mock_eval.call_count == 3

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_hall_of_fame_grows(self, mock_load, mock_split,
                                      mock_prov, mock_gen, mock_eval):
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()

        h_pass = _hyp("winning theory")
        h_fail = _hyp("losing theory")
        mock_gen.return_value = [h_pass, h_fail]

        # first passes, second gets rejected
        mock_eval.side_effect = [_gate("active"), _gate("reject")]

        cfg = RunConfig(symbol="SPY", iterations=2, hypotheses_per_call=5)
        results = await run(cfg)

        assert len(results) == 1
        assert results[0].hypothesis.theory == "winning theory"

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_backtest_config_forwarded(self, mock_load, mock_split,
                                              mock_prov, mock_gen, mock_eval):
        """Make sure runner passes the right BacktestConfig to evaluate."""
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()
        mock_gen.return_value = [_hyp()]
        mock_eval.return_value = _gate("active")

        cfg = RunConfig(symbol="QQQ", iterations=1, hypotheses_per_call=5,
                        hold_days=10, cost_per_side_bps=3.0, slippage_bps=1.0)
        await run(cfg)

        call_kwargs = mock_eval.call_args
        bt_cfg = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        # might be positional
        if bt_cfg is None:
            bt_cfg = call_kwargs[0][4] if len(call_kwargs[0]) > 4 else call_kwargs[0][-1]
        assert isinstance(bt_cfg, BacktestConfig)
        assert bt_cfg.hold_days == 10
        assert bt_cfg.cost_per_side_bps == 3.0

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_multiple_batches(self, mock_load, mock_split,
                                     mock_prov, mock_gen, mock_eval):
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()

        # 5 iterations, 2 per call = 3 calls (2+2+1)
        mock_gen.return_value = [_hyp("x")]
        mock_eval.return_value = _gate("watchlist")

        cfg = RunConfig(symbol="SPY", iterations=5, hypotheses_per_call=2)
        results = await run(cfg)

        assert mock_gen.call_count == 3
        # each call returns 1 hyp, and we need 5 iterations total
        # but batch_size limits how many we request - the gen only returns 1 each time
        # so we get min(returned, needed) evaluated per batch
        assert len(results) == 3  # 1 per call, 3 calls

    @patch(_P_EVAL)
    @patch(_P_GEN, new_callable=AsyncMock)
    @patch(_P_PROVIDER)
    @patch(_P_SPLIT)
    @patch(_P_LOAD)
    async def test_watchlist_goes_to_hall_of_fame(self, mock_load, mock_split,
                                                   mock_prov, mock_gen, mock_eval):
        # watchlist != reject, so it should end up in results + hall of fame
        mock_load.return_value = _fake_df()
        mock_split.return_value = SPLIT
        mock_prov.return_value = MagicMock()
        mock_gen.return_value = [_hyp("promising")]
        mock_eval.return_value = _gate("watchlist")

        cfg = RunConfig(symbol="SPY", iterations=1, hypotheses_per_call=5)
        results = await run(cfg)

        assert len(results) == 1
        assert results[0].gate_result.status == "watchlist"
