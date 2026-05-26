"""Console + JSON output for discovery results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import DiscoveryResult

logger = logging.getLogger(__name__)


def print_summary(results: list[DiscoveryResult]) -> None:
    """Dump a quick summary table to stdout."""
    if not results:
        print("No passing hypotheses found.")  # noqa: T201
        return

    print(f"\n{'=' * 80}")  # noqa: T201
    print(f" Discovery Results: {len(results)} passing hypothesis(es)")  # noqa: T201
    print(f"{'=' * 80}\n")  # noqa: T201

    for i, r in enumerate(results, 1):
        gr = r.gate_result
        wf = gr.wf_metrics
        print(f"  [{i}] {r.hypothesis.theory}")  # noqa: T201
        print(f"      Direction: {r.hypothesis.direction} | Status: {gr.status}")  # noqa: T201
        print(  # noqa: T201
            f"      Sharpe: {wf.sharpe:.2f} | "
            f"Return: {wf.annual_return * 100:.1f}% | "
            f"Win Rate: {wf.win_rate_pct:.0f}% | "
            f"Trades: {wf.n_trades}"
        )
        print(  # noqa: T201
            f"      MFE p75: {wf.p75_mfe_pct:.1f}% | "
            f"MAE p90: {wf.p90_mae_pct:.1f}% | "
            f"Max DD: {wf.max_drawdown_pct:.1f}%"
        )
        if gr.holdout_metrics:
            ho = gr.holdout_metrics
            print(  # noqa: T201
                f"      Holdout — Sharpe: {ho.sharpe:.2f} | "
                f"Return: {ho.annual_return * 100:.1f}% | "
                f"Win Rate: {ho.win_rate_pct:.0f}%"
            )
        print()  # noqa: T201


def save_json(
    results: list[DiscoveryResult],
    output_dir: str = "results",
) -> str:
    """Write results to a timestamped JSON file. Returns the path."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"discovery_{timestamp}.json"
    filepath = out_path / filename

    output: list[dict[str, Any]] = []
    for r in results:
        output.append({
            "hypothesis": {
                "code": r.hypothesis.code,
                "theory": r.hypothesis.theory,
                "direction": r.hypothesis.direction,
            },
            "status": r.gate_result.status,
            "walk_forward": asdict(r.gate_result.wf_metrics),
            "holdout": asdict(r.gate_result.holdout_metrics) if r.gate_result.holdout_metrics else None,
        })

    filepath.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results saved to %s", filepath)
    return str(filepath)
