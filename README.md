# karpathy-quant

Uses an LLM to generate trading signal hypotheses as Python code, backtests them against real OHLCV data from yfinance, and filters for winners using walk-forward + holdout validation. Inspired by Karpathy's approach to automated hypothesis search.

The loop is self-improving — rejected theories get fed back as "saturated" so the LLM stops suggesting them, and winners go into a "hall of fame" to steer it toward what's working.

## How it works

```
  yfinance OHLCV data
        |
  train/holdout split
        |
        v
  LLM generates N hypotheses (get_feature(df) functions)
        |
        v
  walk-forward backtest on train data
  (non-overlapping trades, MFE/MAE, transaction costs)
        |
        v
  gatekeeper (sharpe, return, drawdown thresholds)
        |
   pass? -----> holdout validation -----> results
        |                                    |
   fail? --> add to saturated list     pass? --> hall of fame
```

## Setup

```bash
git clone <repo-url> && cd karpathy-quant
pip install -e ".[all,dev]"
```

Set your API key via a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

Or export it directly:

```bash
export GEMINI_API_KEY=your-key-here
```

Then run:

```bash
python examples/run_discovery.py
```

## Usage

Tweak `RunConfig` to change what gets searched:

```python
import asyncio
from karpathy_quant.runner import RunConfig, run
from karpathy_quant.report import print_summary, save_json

config = RunConfig(
    symbol="AAPL",
    iterations=50,
    hypotheses_per_call=10,
    provider="gemini",
    hold_days=5,
    cost_per_side_bps=5.0,
    slippage_bps=2.0,
    holdout_pct=0.2,
    data_period="5y",
)

results = asyncio.run(run(config))
print_summary(results)
save_json(results)
```

## Hypotheses

Each hypothesis is a Python snippet defining `get_feature(df)` — takes a DataFrame with `[open, high, low, close, volume]` columns + DatetimeIndex, returns a Series where values > 0.5 mean "enter trade."

```python
def get_feature(df):
    # RSI below 30 + volume above 20d average
    rsi = 100 - 100 / (1 + df['close'].diff().clip(lower=0).rolling(14).mean() /
                            df['close'].diff().clip(upper=0).abs().rolling(14).mean())
    vol_above_avg = df['volume'] > df['volume'].rolling(20).mean()
    return ((rsi < 30) & vol_above_avg).astype(float)
```

## Gatekeeper

Signals get classified based on walk-forward metrics:

- **active**: sharpe >= 0.5, annual return >= 8%, max drawdown >= -30%, at least 10 trades
- **watchlist**: sharpe >= 0.3, annual return >= 3%, max drawdown >= -50%, at least 5 trades
- **reject**: everything else

Both walk-forward and holdout must pass. Final status is the worse of the two.

## Tests

```bash
pytest tests/unit/ -v                          # no network needed
pytest tests/integration/ -v -m integration    # needs API key + network
bash scripts/check_coverage.sh                 # coverage gate (default 90%)
```

## Project layout

```
src/karpathy_quant/
  data_loader.py     OHLCV fetching + parquet cache + train/holdout split
  backtest.py        walk-forward backtester (MFE/MAE/Sharpe)
  gatekeeper.py      two-stage gate
  hypothesis.py      LLM orchestration
  runner.py          main discovery loop
  report.py          console + JSON output
  providers/
    base.py          Provider ABC + data types
    gemini.py        Gemini API

tests/
  unit/              mocked, deterministic
  integration/       real data + real LLM calls
```
