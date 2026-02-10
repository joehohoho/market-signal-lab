# Market Signal Lab

**Local-first trading research, backtesting, and monitoring for US/Canada stocks and crypto.**

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Do not make investment decisions based solely on the output of this software. Past performance does not guarantee future results.

---

## Features

- **Data ingestion** -- pull OHLCV candles from Kraken (crypto) and Yahoo Finance (equities) with automatic gap detection and incremental updates
- **Three built-in strategies** -- SMA Crossover, RSI Mean Reversion, and Donchian Breakout, each configurable per timeframe
- **Backtesting engine** -- event-driven backtest with configurable fees, slippage, and ATR-based stop losses; outputs CAGR, Sharpe, max drawdown, win rate, profit factor, and exposure
- **Paper trading** -- forward-test strategies in real time without risking capital
- **Screener** -- scan watchlists or entire universes for fresh signals across multiple strategies and timeframes
- **Slack alerts** -- receive BUY/SELL notifications via Slack webhook
- **Web UI** -- FastAPI + Jinja2 dashboard with Plotly charts for interactive exploration
- **Optional ML module** -- scikit-learn-based feature engineering and walk-forward binary classifier that predicts forward returns
- **Optional AI module** -- local Ollama integration for natural-language market commentary (no data leaves your machine)
- **Storage** -- Parquet files for candle data, DuckDB for metadata catalogue; everything stays on disk, no cloud dependency

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/market-signal-lab.git
cd market-signal-lab

# 2. Install (editable, with dev dependencies)
pip install -e ".[dev]"

# 3. Copy and edit the configuration
cp config/config.example.yaml config/config.yaml

# 4. Ingest some data and run signals
python -m app.cli ingest --asset BTC-USD --tf 1d --start 2023-01-01
python -m app.cli ingest --asset AAPL --tf 1d --start 2023-01-01
python -m app.cli run-signals --watchlist
```

---

## Setup

### Laptop (development)

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows

# Install with all optional extras
pip install -e ".[dev,ml,ai]"

# Copy config
cp config/config.example.yaml config/config.yaml

# Run the test suite
pytest tests/ -v
```

### Mac mini (always-on deployment)

```bash
# 1. Clone
git clone https://github.com/<your-username>/market-signal-lab.git
cd market-signal-lab

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,ai]"

# 3. Configure
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml: enable Slack, set watchlists, choose strategies

# 4. Run the web dashboard
python -m app.cli serve          # default: http://0.0.0.0:8000

# 5. (Optional) Set up a cron job or launchd plist to run signals periodically
#    e.g. every 15 minutes:
#    */15 * * * * cd /path/to/market-signal-lab && .venv/bin/python -m app.cli run-signals --watchlist
```

---

## Demo Script

A full demo that ingests data, runs signals, backtests a strategy, scans a universe, and launches the web UI:

```bash
# Ingest daily candles
python -m app.cli ingest --asset BTC-USD --tf 1d --start 2023-01-01
python -m app.cli ingest --asset AAPL --tf 1d --start 2023-01-01

# Run signal evaluation for all watchlist assets
python -m app.cli run-signals --watchlist

# Backtest SMA crossover on BTC-USD
python -m app.cli backtest --asset BTC-USD --strategy sma_crossover --tf 1d --start 2023-01-01 --end 2024-01-01

# Screen the crypto universe for fresh signals
python -m app.cli screen --universe crypto --tf 1d

# Launch the web dashboard
python -m app.cli serve
```

Or use the Makefile shortcuts:

```bash
make ingest-demo   # Ingest BTC-USD and AAPL
make demo          # Full demo (ingest + signals + backtest + screener)
make test          # Run the test suite
make serve         # Start the web server
```

---

## Slack Webhook Configuration

1. Create a Slack app at <https://api.slack.com/apps> and add an Incoming Webhook.
2. Copy the webhook URL.
3. Set it in one of two ways:

   **Environment variable (recommended):**
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
   export SLACK_ENABLED=true
   ```

   **Or in `config/config.yaml`:**
   ```yaml
   alerts:
     slack:
       enabled: true
       webhook_url: "https://hooks.slack.com/services/T.../B.../..."
   ```

4. Alerts are sent whenever a strategy emits a BUY or SELL signal for a watched asset.

---

## Configuration Reference

The configuration file lives at `config/config.yaml` (copy from `config/config.example.yaml`). Key sections:

| Section | Description |
|---|---|
| `storage` | Paths for DuckDB (`db_path`) and Parquet candle files (`parquet_dir`) |
| `providers` | Default data provider per asset class (`crypto` -> `kraken`, `equity` -> `yahoo_daily`) |
| `watchlist` | List of assets to monitor with their timeframes and asset class |
| `strategies` | Per-strategy, per-timeframe parameter overrides (periods, thresholds, multipliers) |
| `fee_presets` | Named fee/slippage/spread presets (`crypto_major`, `liquid_stock`, `penny_stock`) |
| `risk` | Stop-loss mode (`none`, `fixed_percent`, `atr_multiple`), position sizing |
| `screener` | Universe definitions and filtering thresholds |
| `alerts.slack` | Webhook URL, enable/disable, trigger filters |
| `ml` | Enable/disable, forward horizon, score threshold, walk-forward windows |
| `ai` | Enable/disable, adapter selection (`null` or `ollama`), model and base URL |
| `server` | Host and port for the FastAPI web server |

Environment variables override YAML values. See `src/app/config.py` for the full mapping.

---

## Architecture Overview

```
src/
  app/          CLI entry point, config loader, structured logging
  data/
    providers/  Data provider adapters (Kraken, Yahoo)
    storage/    ParquetStore (candle files) + DuckDBStore (metadata)
    ingest.py   Orchestrates fetching, gap detection, deduplication
  indicators/   Pure-function technical indicators (SMA, EMA, RSI, MACD, ATR, Donchian, Bollinger, etc.)
  strategies/   Strategy base class + concrete implementations (SMA Crossover, RSI Mean Reversion, Donchian Breakout)
  signals/      Signal evaluation engine
  backtest/     Event-driven backtest engine + performance metrics
  paper/        Paper trading module
  screener/     Multi-asset signal scanner
  alerts/       Notification dispatchers (Slack)
  ml/           Feature engineering + walk-forward ML classifier
  ai/           Local LLM adapter (Ollama)
  api/          FastAPI REST endpoints
  ui/           Jinja2 templates + Plotly charts
tests/          pytest test suite
config/         YAML configuration (example committed, local copy gitignored)
```

Data flows through the system as follows:

1. **Ingest** fetches OHLCV candles from a provider and persists them as Parquet files with DuckDB metadata.
2. **Indicators** compute technical features from the candle data (pure functions, no side effects).
3. **Strategies** consume indicators and emit `SignalResult` objects (BUY / SELL / HOLD with strength and explanation).
4. **Backtest** replays historical signals through a simulated portfolio with configurable fees and risk controls.
5. **Screener** evaluates strategies across a universe and surfaces fresh signals.
6. **Alerts** dispatch notable signals to Slack.
7. **Web UI** presents everything interactively.

---

## ML Module (Optional)

The ML module adds a scikit-learn-based binary classifier that predicts whether forward returns will be positive over a configurable horizon.

### How to enable

1. Install the ML extras:
   ```bash
   pip install -e ".[ml]"
   ```
2. Set `ml.enabled: true` in `config/config.yaml`.

### What it does

- **Feature engineering** (`src/ml/features.py`): computes RSI, MACD histogram, normalised ATR, volatility, SMA ratios, volume ratio, lagged returns, and calendar features -- all using only past data (no lookahead bias).
- **Target**: binary label based on whether the close price is higher after `horizon_bars` bars.
- **Walk-forward validation**: the model is retrained on expanding windows to avoid overfitting to a single train/test split.
- **Score threshold**: signals are only boosted when the model's predicted probability exceeds `ml.score_threshold` (default 0.55).

---

## AI Module (Optional)

The AI module connects to a local Ollama instance to generate natural-language commentary on signals and market conditions. No data is sent to any external service.

### Ollama setup

1. Install Ollama: <https://ollama.com/download>
2. Pull a model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
3. Ollama runs on `http://localhost:11434` by default.

### Recommended models

| Model | Size | Notes |
|---|---|---|
| `qwen2.5:7b-instruct` | ~4 GB | Good balance of speed and quality; runs on 8 GB RAM |
| `llama3.1:8b` | ~4.7 GB | Strong general reasoning |
| `mistral:7b` | ~4 GB | Fast inference, solid for summarisation |

### Configuration

```yaml
ai:
  enabled: true
  adapter: "ollama"
  ollama:
    model: "qwen2.5:7b-instruct"
    base_url: "http://localhost:11434"
```

Or via environment variables:
```bash
export OLLAMA_MODEL="qwen2.5:7b-instruct"
export OLLAMA_BASE_URL="http://localhost:11434"
```

---

## ATR Explanation

**ATR (Average True Range)** measures typical price movement over a given period. It accounts for gaps between sessions by considering the true range -- the greatest of:

- Current high minus current low
- Absolute value of current high minus previous close
- Absolute value of current low minus previous close

An **ATR-multiple stop** adapts to the asset's current volatility. For example, with a 2x ATR stop:
- In a volatile market (ATR = $5), the stop is set $10 below entry, giving the trade room to breathe.
- In a calm market (ATR = $1), the stop is set $2 below entry, keeping risk tight.

This approach avoids the common pitfall of fixed-percentage stops that are too tight in volatile conditions and too loose in quiet ones.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test module
pytest tests/test_indicators.py -v
```

---

## License

MIT
