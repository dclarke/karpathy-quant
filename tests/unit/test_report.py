import json
import tempfile
from pathlib import Path

import pytest

from karpathy_quant.backtest import BacktestResult
from karpathy_quant.gatekeeper import GateResult
from karpathy_quant.providers.base import Hypothesis
from karpathy_quant.report import print_summary, save_json
from karpathy_quant.runner import DiscoveryResult


def _make_result(theory="Test theory", status="active"):
    wf = BacktestResult(
        n_trades=15, total_return=0.15, annual_return=0.12, sharpe=1.5,
        win_rate_pct=60.0, avg_win_pct=2.5, avg_loss_pct=-1.5,
        max_drawdown_pct=-10.0, avg_mfe_pct=3.0, p50_mfe_pct=2.5,
        p75_mfe_pct=4.0, p90_mfe_pct=6.0, avg_mae_pct=1.5,
        p90_mae_pct=3.0, optimal_hold_bars=3, fire_rate=0.05,
        fire_dates=["2020-01-15", "2020-03-10"], status=status,
    )
    return DiscoveryResult(
        hypothesis=Hypothesis(code="def get_feature(df): ...", theory=theory, direction="long"),
        gate_result=GateResult(status=status, wf_metrics=wf),
    )


def _make_result_with_holdout(status="active"):
    wf = BacktestResult(
        n_trades=15, total_return=0.15, annual_return=0.12, sharpe=1.5,
        win_rate_pct=60.0, avg_win_pct=2.5, avg_loss_pct=-1.5,
        max_drawdown_pct=-10.0, avg_mfe_pct=3.0, p50_mfe_pct=2.5,
        p75_mfe_pct=4.0, p90_mfe_pct=6.0, avg_mae_pct=1.5,
        p90_mae_pct=3.0, optimal_hold_bars=3, fire_rate=0.05,
        fire_dates=["2020-01-15", "2020-03-10"], status=status,
    )
    ho = BacktestResult(
        n_trades=8, total_return=0.09, annual_return=0.08, sharpe=1.1,
        win_rate_pct=55.0, avg_win_pct=2.0, avg_loss_pct=-1.2,
        max_drawdown_pct=-8.0, avg_mfe_pct=2.5, p50_mfe_pct=2.0,
        p75_mfe_pct=3.5, p90_mfe_pct=5.0, avg_mae_pct=1.2,
        p90_mae_pct=2.5, optimal_hold_bars=3, fire_rate=0.04,
        fire_dates=["2020-07-01"], status=status,
    )
    return DiscoveryResult(
        hypothesis=Hypothesis(code="def get_feature(df): ...", theory="Holdout test", direction="long"),
        gate_result=GateResult(status=status, wf_metrics=wf, holdout_metrics=ho),
    )


class TestPrintSummary:
    def test_no_results(self, capsys):
        print_summary([])
        assert "No passing hypotheses found" in capsys.readouterr().out

    def test_with_results(self, capsys):
        print_summary([_make_result("Buy the dip")])
        out = capsys.readouterr().out
        assert "Buy the dip" in out
        assert "Sharpe" in out
        assert "1 passing" in out

    def test_multiple_results(self, capsys):
        results = [_make_result(f"Theory {i}") for i in range(3)]
        print_summary(results)
        assert "3 passing" in capsys.readouterr().out

    def test_holdout_metrics_printed(self, capsys):
        print_summary([_make_result_with_holdout()])
        out = capsys.readouterr().out
        assert "Holdout" in out
        assert "1.10" in out  # holdout sharpe


class TestSaveJson:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json([_make_result()], output_dir=tmpdir)
            assert Path(path).exists()

    def test_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json([_make_result()], output_dir=tmpdir)
            data = json.loads(Path(path).read_text())
            assert isinstance(data, list)
            assert len(data) == 1

    def test_json_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json([_make_result("My theory", "watchlist")], output_dir=tmpdir)
            data = json.loads(Path(path).read_text())
            item = data[0]
            assert item["hypothesis"]["theory"] == "My theory"
            assert item["status"] == "watchlist"
            assert "walk_forward" in item
            assert item["walk_forward"]["sharpe"] == 1.5

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = str(Path(tmpdir) / "sub" / "dir")
            path = save_json([_make_result()], output_dir=nested)
            assert Path(path).exists()

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json([], output_dir=tmpdir)
            data = json.loads(Path(path).read_text())
            assert data == []
