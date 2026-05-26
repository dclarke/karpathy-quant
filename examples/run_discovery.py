"""CLI entry point for running signal discovery."""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

# load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from karpathy_quant.report import print_summary, save_json
from karpathy_quant.runner import RunConfig, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


async def main() -> None:
    config = RunConfig(
        symbol="AAPL",
        iterations=50,
        hypotheses_per_call=1,
        provider="gemini",
        hold_days=5,
        data_period="5y",
    )

    results = await run(config)
    print_summary(results)

    if results:
        path = save_json(results)
        print(f"Results saved to {path}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
