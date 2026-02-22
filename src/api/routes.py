"""FastAPI application with routes for the Market Signal Lab web UI.

Serves HTMX-powered pages for watchlist, asset detail, backtest, and
screener.  Also provides JSON API endpoints for signals and health.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly
import plotly.graph_objects as go
from fastapi import FastAPI, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_config, load_config
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestResult
from backtest.optimizer import ParameterOptimizer
from data.providers.yfinance_provider import YFinanceProvider
from data.storage.parquet_store import ParquetStore
from indicators.core import ema, sma
from paper.simulator import PaperTradingSimulator
from screener.scanner import Screener
from signals.engine import SignalEngine
from strategies import STRATEGY_REGISTRY, get_strategy
from strategies.base import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent
_TEMPLATES_DIR = _SRC_DIR / "ui" / "templates"
_STATIC_DIR = _SRC_DIR / "ui" / "static"
_SCENARIOS_DIR = _PROJECT_ROOT / "data" / "scenarios"
_WATCHLIST_JSON = _PROJECT_ROOT / "data" / "watchlist.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store() -> ParquetStore:
    """Return a ParquetStore using the configured directory."""
    try:
        storage_cfg = get_config("storage")
        parquet_dir = storage_cfg.get("parquet_dir", "data/candles")
    except KeyError:
        parquet_dir = "data/candles"
    # Resolve relative paths against the project root, not the CWD
    path = Path(parquet_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return ParquetStore(str(path))


def _load_watchlist_json() -> list[dict[str, Any]] | None:
    """Load watchlist from data/watchlist.json, or None if absent."""
    if _WATCHLIST_JSON.exists():
        try:
            with open(_WATCHLIST_JSON, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_watchlist_json(items: list[dict[str, Any]]) -> None:
    """Persist the watchlist to data/watchlist.json."""
    _WATCHLIST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_WATCHLIST_JSON, "w") as f:
        json.dump(items, f, indent=2)


def _get_watchlist() -> list[dict[str, Any]]:
    """Return the watchlist, preferring data/watchlist.json over config."""
    wl = _load_watchlist_json()
    if wl is not None:
        return wl
    try:
        return get_config("watchlist") or []
    except KeyError:
        return []


def _available_assets() -> list[str]:
    """Return sorted unique asset symbols that have parquet data."""
    store = _get_store()
    candle_dir = Path(store._parquet_dir)
    if not candle_dir.exists():
        return []
    symbols: set[str] = set()
    for p in candle_dir.glob("*.parquet"):
        # Filename is {SYMBOL}_{TIMEFRAME}.parquet
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2:
            symbols.add(parts[0])
    return sorted(symbols)


def _get_strategy_params() -> dict[str, dict[str, Any]]:
    """Return the strategy params from config."""
    try:
        return get_config("strategies") or {}
    except KeyError:
        return {}


def _get_fee_presets() -> dict[str, Any]:
    """Return the fee presets from config."""
    try:
        return get_config("fee_presets") or {}
    except KeyError:
        return {}


def _format_price(price: float) -> str:
    """Format a price for display."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.6f}"


def _format_timestamp(ts_str: str) -> str:
    """Format a timestamp string for display."""
    try:
        ts = pd.Timestamp(ts_str)
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_str[:16] if len(ts_str) > 16 else ts_str


def _signal_color(signal_str: str) -> str:
    """Return a CSS color class for a signal."""
    if signal_str == "BUY":
        return "text-green-400"
    elif signal_str == "SELL":
        return "text-red-400"
    return "text-gray-400"


def _has_ml_model(asset: str, timeframe: str) -> bool:
    """Check whether a trained ML model exists for this asset/timeframe."""
    safe_asset = asset.replace("/", "-").replace("\\", "-")
    model_path = _PROJECT_ROOT / "models" / f"{safe_asset}_{timeframe}.joblib"
    return model_path.exists()


def _build_candlestick_chart(
    df: pd.DataFrame, symbol: str, sma_periods: list[int] | None = None,
) -> dict[str, Any]:
    """Build a Plotly candlestick chart as a JSON-serialisable dict."""
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"].astype(str).tolist(),
            open=df["open"].tolist(),
            high=df["high"].tolist(),
            low=df["low"].tolist(),
            close=df["close"].tolist(),
            name=symbol,
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )

    # Add SMA overlays
    if sma_periods:
        colors = ["#60a5fa", "#fbbf24", "#a78bfa", "#f472b6"]
        for idx, period in enumerate(sma_periods):
            sma_vals = sma(df["close"], period)
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"].astype(str).tolist(),
                    y=sma_vals.tolist(),
                    mode="lines",
                    name=f"SMA {period}",
                    line=dict(width=1, color=color),
                )
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#d1d5db"),
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="#374151",
        ),
        yaxis=dict(gridcolor="#374151"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=500,
        title=dict(text=f"{symbol} Price Chart", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return json.loads(plotly.io.to_json(fig))


def _build_comparison_equity_chart(
    baseline: pd.Series, filtered: pd.Series,
) -> dict[str, Any]:
    """Build a Plotly chart overlaying baseline and ML-filtered equity curves."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[str(ts) for ts in baseline.index],
            y=baseline.values.tolist(),
            mode="lines",
            name="Baseline",
            line=dict(width=2, color="#94a3b8"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[str(ts) for ts in filtered.index],
            y=filtered.values.tolist(),
            mode="lines",
            name="ML-Filtered",
            line=dict(width=2, color="#a78bfa"),
            fill="tonexty",
            fillcolor="rgba(167, 139, 250, 0.08)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#d1d5db"),
        xaxis=dict(gridcolor="#374151"),
        yaxis=dict(gridcolor="#374151", title="Equity ($)"),
        margin=dict(l=60, r=20, t=40, b=40),
        height=350,
        title=dict(text="Baseline vs ML-Filtered Equity", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return json.loads(plotly.io.to_json(fig))


def _build_equity_chart(equity_curve: pd.Series) -> dict[str, Any]:
    """Build a Plotly line chart for the equity curve."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[str(ts) for ts in equity_curve.index],
            y=equity_curve.values.tolist(),
            mode="lines",
            name="Equity",
            line=dict(width=2, color="#22c55e"),
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.1)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#d1d5db"),
        xaxis=dict(gridcolor="#374151"),
        yaxis=dict(gridcolor="#374151", title="Equity ($)"),
        margin=dict(l=60, r=20, t=40, b=40),
        height=350,
        title=dict(text="Equity Curve", font=dict(size=14)),
    )

    return json.loads(plotly.io.to_json(fig))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Ensure config is loaded
    load_config()

    app = FastAPI(
        title="Market Signal Lab",
        description="Educational trading research and backtesting platform",
        version="0.1.0",
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Templates
    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Add custom template filters
    templates.env.filters["format_price"] = _format_price
    templates.env.filters["format_timestamp"] = _format_timestamp
    templates.env.filters["signal_color"] = _signal_color

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=RedirectResponse)
    async def root():
        """Redirect to watchlist."""
        return RedirectResponse(url="/watchlist", status_code=302)

    @app.get("/watchlist", response_class=HTMLResponse)
    async def watchlist_page(request: Request):
        """Render the watchlist page."""
        watchlist = _get_watchlist()
        store = _get_store()
        engine = SignalEngine()

        assets_data: list[dict[str, Any]] = []

        for item in watchlist:
            symbol = item["asset"]
            timeframes = item.get("timeframes", ["1d"])
            tf = timeframes[0] if timeframes else "1d"

            df = store.load(symbol, tf)

            if df.empty:
                assets_data.append({
                    "symbol": symbol,
                    "price": "N/A",
                    "timestamp": "No data",
                    "signals": [],
                    "dominant_signal": "HOLD",
                    "strength": 0.0,
                })
                continue

            # Prepare df for strategies (needs DatetimeIndex)
            prepared = df.set_index("timestamp") if "timestamp" in df.columns else df

            strategies_config = _get_strategy_params()
            signal_results = engine.run_single(
                prepared, symbol, tf, strategies_config=strategies_config,
            )

            price = float(df["close"].iloc[-1])
            ts = str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else ""

            signals_list = []
            dominant = "HOLD"
            max_strength = 0.0

            for sr in signal_results:
                sig_val = sr.signal.value
                signals_list.append({
                    "strategy": sr.strategy_name,
                    "signal": sig_val,
                    "strength": round(sr.strength, 2),
                })
                if sr.signal != Signal.HOLD and sr.strength > max_strength:
                    max_strength = sr.strength
                    dominant = sig_val

            assets_data.append({
                "symbol": symbol,
                "price": _format_price(price),
                "timestamp": _format_timestamp(ts),
                "signals": signals_list,
                "dominant_signal": dominant,
                "strength": round(max_strength, 2),
            })

        return templates.TemplateResponse("watchlist.html", {
            "request": request,
            "assets": assets_data,
            "available_assets": _available_assets(),
            "watchlist_items": watchlist,
        })

    @app.post("/watchlist/add", response_class=RedirectResponse)
    async def watchlist_add(
        asset: str = Form(...),
        timeframe: str = Form(default="1d"),
        asset_class: str = Form(default="crypto"),
    ):
        """Add an asset to the watchlist."""
        wl = _get_watchlist()
        symbol = asset.upper().strip()

        # Skip if already present.
        if any(item["asset"] == symbol for item in wl):
            return RedirectResponse(url="/watchlist", status_code=303)

        wl.append({
            "asset": symbol,
            "timeframes": [timeframe],
            "asset_class": asset_class,
        })
        _save_watchlist_json(wl)
        return RedirectResponse(url="/watchlist", status_code=303)

    @app.post("/watchlist/remove", response_class=RedirectResponse)
    async def watchlist_remove(asset: str = Form(...)):
        """Remove an asset from the watchlist."""
        wl = _get_watchlist()
        wl = [item for item in wl if item["asset"] != asset]
        _save_watchlist_json(wl)
        return RedirectResponse(url="/watchlist", status_code=303)

    @app.get("/asset/{symbol}", response_class=HTMLResponse)
    async def asset_detail_page(
        request: Request,
        symbol: str,
        tf: str = Query(default="1d"),
    ):
        """Render the asset detail page."""
        store = _get_store()
        df = store.load(symbol, tf)

        if df.empty:
            return templates.TemplateResponse("asset_detail.html", {
                "request": request,
                "symbol": symbol,
                "timeframe": tf,
                "price": "No data",
                "chart_json": "null",
                "signals": [],
                "strategies": list(STRATEGY_REGISTRY.keys()),
                "fee_presets": list(_get_fee_presets().keys()),
            })

        price = float(df["close"].iloc[-1])

        # Build chart
        chart_data = _build_candlestick_chart(df, symbol, sma_periods=[20, 50])
        chart_json = json.dumps(chart_data)

        # Get signals
        engine = SignalEngine()
        prepared = df.set_index("timestamp") if "timestamp" in df.columns else df
        strategies_config = _get_strategy_params()
        signal_results = engine.run_single(
            prepared, symbol, tf, strategies_config=strategies_config,
        )

        signals_list = []
        for sr in signal_results:
            signals_list.append({
                "strategy": sr.strategy_name,
                "signal": sr.signal.value,
                "strength": round(sr.strength, 2),
                "explanation": sr.explanation,
            })

        return templates.TemplateResponse("asset_detail.html", {
            "request": request,
            "symbol": symbol,
            "timeframe": tf,
            "price": _format_price(price),
            "chart_json": chart_json,
            "signals": signals_list,
            "strategies": list(STRATEGY_REGISTRY.keys()),
            "fee_presets": list(_get_fee_presets().keys()),
        })

    @app.get("/asset/{symbol}/chart", response_class=JSONResponse)
    async def asset_chart(
        symbol: str,
        tf: str = Query(default="1d"),
    ):
        """Return Plotly chart JSON for an asset."""
        store = _get_store()
        df = store.load(symbol, tf)

        if df.empty:
            return JSONResponse(content={"error": "No data available"}, status_code=404)

        chart_data = _build_candlestick_chart(df, symbol, sma_periods=[20, 50])
        return JSONResponse(content=chart_data)

    def _bt_template(request: Request) -> str:
        """Pick full page or HTMX partial for backtest responses."""
        if request.headers.get("HX-Request"):
            return "_backtest_results.html"
        return "backtest.html"

    @app.post("/backtest", response_class=HTMLResponse)
    async def run_backtest(
        request: Request,
        asset: str = Form(...),
        strategy: str = Form(...),
        timeframe: str = Form(default="1d"),
        start_date: str = Form(default=""),
        end_date: str = Form(default=""),
        fee_preset: str = Form(default="crypto_major"),
        initial_capital: float = Form(default=10000.0),
        ml_filter: str = Form(default=""),
    ):
        """Run a backtest and return the results page."""
        use_ml = ml_filter == "on"

        store = _get_store()
        df = store.load(asset, timeframe)

        if df.empty:
            return templates.TemplateResponse(_bt_template(request), {
                "request": request,
                "error": f"No data available for {asset} ({timeframe})",
                "strategies": list(STRATEGY_REGISTRY.keys()),
                "fee_presets": list(_get_fee_presets().keys()),
                "available_assets": _available_assets(),
                "result": None,
                "has_ml_model": _has_ml_model(asset, timeframe),
                "ml_filter_on": False,
                "ml_result": None,
                "ml_trained": None,
                "learn_result": None,
                "initial_capital_raw": initial_capital,
            })

        # Parse dates
        start = None
        end = None
        if start_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        if end_date:
            try:
                end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Filter date range (strip tz to match tz-naive parquet timestamps)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start).tz_localize(None)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end).tz_localize(None)]

        df = df.reset_index(drop=True)

        if len(df) < 2:
            return templates.TemplateResponse(_bt_template(request), {
                "request": request,
                "error": "Insufficient data for the selected date range.",
                "strategies": list(STRATEGY_REGISTRY.keys()),
                "fee_presets": list(_get_fee_presets().keys()),
                "available_assets": _available_assets(),
                "result": None,
                "has_ml_model": _has_ml_model(asset, timeframe),
                "ml_filter_on": False,
                "ml_result": None,
                "ml_trained": None,
                "learn_result": None,
                "initial_capital_raw": initial_capital,
            })

        # Get strategy params
        strat_cfg = _get_strategy_params()
        params = strat_cfg.get(strategy, {}).get(timeframe, {})

        # Run backtest
        strat_obj = get_strategy(strategy)
        engine = BacktestEngine(
            initial_capital=initial_capital,
            fee_preset=fee_preset,
        )

        # If ML filter is on, the main result uses ML and we show baseline as comparison
        ml_result_data = None
        if use_ml:
            result: BacktestResult = engine.run(
                df, strat_obj, params, asset=asset, timeframe=timeframe,
                ml_filter=True,
            )
            baseline_bt: BacktestResult = engine.run(
                df, strat_obj, params, asset=asset, timeframe=timeframe,
            )
            baseline_equity_chart = _build_equity_chart(baseline_bt.equity_curve)
            ml_result_data = {
                "label": "Baseline (no ML)",
                "cagr": f"{baseline_bt.cagr * 100:.2f}%",
                "sharpe": f"{baseline_bt.sharpe:.2f}",
                "max_drawdown": f"{baseline_bt.max_drawdown * 100:.2f}%",
                "win_rate": f"{baseline_bt.win_rate * 100:.1f}%",
                "profit_factor": f"{baseline_bt.profit_factor:.2f}" if baseline_bt.profit_factor != float("inf") else "Inf",
                "total_trades": baseline_bt.total_trades,
                "final_equity": f"${baseline_bt.final_equity:,.2f}",
                "equity_chart_json": json.dumps(baseline_equity_chart),
                "trades_blocked": baseline_bt.total_trades - result.total_trades,
            }
        else:
            result: BacktestResult = engine.run(
                df, strat_obj, params, asset=asset, timeframe=timeframe,
            )

        # Build equity chart
        equity_chart = _build_equity_chart(result.equity_curve)
        equity_chart_json = json.dumps(equity_chart)

        # Format trades
        formatted_trades = []
        for t in result.trades:
            formatted_trades.append({
                "entry_time": _format_timestamp(str(t.get("entry_time", ""))),
                "exit_time": _format_timestamp(str(t.get("exit_time", ""))),
                "entry_price": f"${t.get('entry_price', 0):.4f}",
                "exit_price": f"${t.get('exit_price', 0):.4f}",
                "pnl": f"${t.get('pnl', 0):.2f}",
                "pnl_pct": f"{t.get('pnl_pct', 0) * 100:.2f}%",
                "side": t.get("side", "long"),
                "exit_reason": t.get("exit_reason", "signal"),
                "pnl_positive": t.get("pnl", 0) > 0,
            })

        return templates.TemplateResponse(_bt_template(request), {
            "request": request,
            "error": None,
            "strategies": list(STRATEGY_REGISTRY.keys()),
            "fee_presets": list(_get_fee_presets().keys()),
            "available_assets": _available_assets(),
            "has_ml_model": _has_ml_model(asset, timeframe),
            "ml_filter_on": use_ml,
            "ml_result": ml_result_data,
            "ml_trained": None,
            "learn_result": None,
            "initial_capital_raw": initial_capital,
            "result": {
                "asset": asset,
                "strategy": strategy,
                "timeframe": timeframe,
                "fee_preset": fee_preset,
                "cagr": f"{result.cagr * 100:.2f}%",
                "sharpe": f"{result.sharpe:.2f}",
                "max_drawdown": f"{result.max_drawdown * 100:.2f}%",
                "win_rate": f"{result.win_rate * 100:.1f}%",
                "profit_factor": f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "Inf",
                "exposure": f"{result.exposure * 100:.1f}%",
                "total_trades": result.total_trades,
                "total_bars": result.total_bars,
                "initial_capital": f"${result.initial_capital:,.2f}",
                "final_equity": f"${result.final_equity:,.2f}",
                "equity_chart_json": equity_chart_json,
                "trades": formatted_trades,
            },
        })

    @app.post("/backtest/train-ml", response_class=HTMLResponse)
    async def train_ml_model(
        request: Request,
        asset: str = Form(...),
        timeframe: str = Form(default="1d"),
    ):
        """Train an ML model for the given asset/timeframe."""
        store = _get_store()
        df = store.load(asset, timeframe)

        if df.empty or len(df) < 100:
            return templates.TemplateResponse("backtest.html", {
                "request": request,
                "error": f"Need at least 100 bars to train ML. {asset}/{timeframe} has {len(df)}.",
                "strategies": list(STRATEGY_REGISTRY.keys()),
                "fee_presets": list(_get_fee_presets().keys()),
                "available_assets": _available_assets(),
                "result": None,
                "has_ml_model": False,
                "ml_filter_on": False,
                "ml_result": None,
                "ml_trained": None,
                "learn_result": None,
            })

        try:
            from ml.models import train_and_validate

            ml_result = train_and_validate(df, asset=asset, timeframe=timeframe)

            # Save the model
            models_dir = _PROJECT_ROOT / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            safe_asset = asset.replace("/", "-").replace("\\", "-")
            model_path = models_dir / f"{safe_asset}_{timeframe}.joblib"
            ml_result.model.save(model_path)

            train_info = {
                "asset": asset,
                "timeframe": timeframe,
                "auc_scores": [f"{s:.3f}" for s in ml_result.main_aucs],
                "stable": ml_result.is_stable,
                "avg_auc": f"{sum(ml_result.main_aucs) / max(len(ml_result.main_aucs), 1):.3f}",
            }
        except Exception as exc:
            return templates.TemplateResponse("backtest.html", {
                "request": request,
                "error": f"ML training failed: {exc}",
                "strategies": list(STRATEGY_REGISTRY.keys()),
                "fee_presets": list(_get_fee_presets().keys()),
                "available_assets": _available_assets(),
                "result": None,
                "has_ml_model": _has_ml_model(asset, timeframe),
                "ml_filter_on": False,
                "ml_result": None,
                "ml_trained": None,
                "learn_result": None,
            })

        return templates.TemplateResponse("backtest.html", {
            "request": request,
            "error": None,
            "strategies": list(STRATEGY_REGISTRY.keys()),
            "fee_presets": list(_get_fee_presets().keys()),
            "available_assets": _available_assets(),
            "result": None,
            "has_ml_model": True,
            "ml_filter_on": False,
            "ml_result": None,
            "ml_trained": train_info,
            "learn_result": None,
        })

    @app.post("/backtest/learn", response_class=HTMLResponse)
    async def learn_and_improve(
        request: Request,
        asset: str = Form(...),
        strategy: str = Form(...),
        timeframe: str = Form(default="1d"),
        start_date: str = Form(default=""),
        end_date: str = Form(default=""),
        fee_preset: str = Form(default="crypto_major"),
        initial_capital: float = Form(default=10000.0),
    ):
        """Train ML from trade outcomes, then compare baseline vs filtered."""
        _tpl_base = {
            "request": request,
            "strategies": list(STRATEGY_REGISTRY.keys()),
            "fee_presets": list(_get_fee_presets().keys()),
            "available_assets": _available_assets(),
            "ml_filter_on": False,
            "ml_result": None,
            "ml_trained": None,
            "initial_capital_raw": initial_capital,
        }

        store = _get_store()
        df = store.load(asset, timeframe)

        if df.empty:
            return templates.TemplateResponse("backtest.html", {
                **_tpl_base,
                "error": f"No data available for {asset} ({timeframe})",
                "result": None,
                "has_ml_model": _has_ml_model(asset, timeframe),
                "learn_result": None,
            })

        # Parse and apply date filters
        start = None
        end = None
        if start_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        if end_date:
            try:
                end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start).tz_localize(None)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end).tz_localize(None)]
        df = df.reset_index(drop=True)

        if len(df) < 50:
            return templates.TemplateResponse("backtest.html", {
                **_tpl_base,
                "error": "Need at least 50 bars to learn from trades.",
                "result": None,
                "has_ml_model": _has_ml_model(asset, timeframe),
                "learn_result": None,
            })

        strat_cfg = _get_strategy_params()
        params = strat_cfg.get(strategy, {}).get(timeframe, {})
        strat_obj = get_strategy(strategy)

        try:
            from ml.trade_learner import train_from_trades

            learn = train_from_trades(
                df, strat_obj, params,
                asset=asset, timeframe=timeframe,
                fee_preset=fee_preset,
            )

            # Save model to disk
            models_dir = _PROJECT_ROOT / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            safe_asset = asset.replace("/", "-").replace("\\", "-")
            model_path = models_dir / f"{safe_asset}_{timeframe}.joblib"
            learn.model.save(model_path)

        except (ValueError, ImportError) as exc:
            return templates.TemplateResponse("backtest.html", {
                **_tpl_base,
                "error": f"Learn & Improve failed: {exc}",
                "result": None,
                "has_ml_model": _has_ml_model(asset, timeframe),
                "learn_result": None,
            })

        # Run baseline backtest
        engine = BacktestEngine(initial_capital=initial_capital, fee_preset=fee_preset)
        baseline: BacktestResult = engine.run(
            df, strat_obj, params, asset=asset, timeframe=timeframe,
        )
        # Run ML-filtered backtest (model was just saved to disk)
        filtered: BacktestResult = engine.run(
            df, strat_obj, params, asset=asset, timeframe=timeframe,
            ml_filter=True,
        )

        # Build comparison equity chart with both curves
        comparison_chart = _build_comparison_equity_chart(
            baseline.equity_curve, filtered.equity_curve,
        )
        comparison_chart_json = json.dumps(comparison_chart)

        # Format baseline result for the standard display
        baseline_equity_chart = _build_equity_chart(baseline.equity_curve)
        baseline_equity_json = json.dumps(baseline_equity_chart)

        formatted_trades = []
        for t in baseline.trades:
            formatted_trades.append({
                "entry_time": _format_timestamp(str(t.get("entry_time", ""))),
                "exit_time": _format_timestamp(str(t.get("exit_time", ""))),
                "entry_price": f"${t.get('entry_price', 0):.4f}",
                "exit_price": f"${t.get('exit_price', 0):.4f}",
                "pnl": f"${t.get('pnl', 0):.2f}",
                "pnl_pct": f"{t.get('pnl_pct', 0) * 100:.2f}%",
                "side": t.get("side", "long"),
                "exit_reason": t.get("exit_reason", "signal"),
                "pnl_positive": t.get("pnl", 0) > 0,
            })

        # Top feature importances (sorted, top 5)
        top_features = sorted(
            learn.feature_importances.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]

        learn_result = {
            "total_trades": learn.total_trades,
            "winning_trades": learn.winning_trades,
            "losing_trades": learn.losing_trades,
            "train_accuracy": f"{learn.train_accuracy * 100:.1f}%",
            "val_accuracy": f"{learn.val_accuracy * 100:.1f}%",
            "val_auc": f"{learn.val_auc:.3f}",
            "top_features": [
                {"name": name.replace("_", " ").title(), "pct": f"{imp * 100:.1f}%"}
                for name, imp in top_features
            ],
            "comparison_chart_json": comparison_chart_json,
            "baseline": {
                "cagr": f"{baseline.cagr * 100:.2f}%",
                "sharpe": f"{baseline.sharpe:.2f}",
                "max_drawdown": f"{baseline.max_drawdown * 100:.2f}%",
                "win_rate": f"{baseline.win_rate * 100:.1f}%",
                "profit_factor": f"{baseline.profit_factor:.2f}" if baseline.profit_factor != float("inf") else "Inf",
                "total_trades": baseline.total_trades,
                "final_equity": f"${baseline.final_equity:,.2f}",
            },
            "filtered": {
                "cagr": f"{filtered.cagr * 100:.2f}%",
                "sharpe": f"{filtered.sharpe:.2f}",
                "max_drawdown": f"{filtered.max_drawdown * 100:.2f}%",
                "win_rate": f"{filtered.win_rate * 100:.1f}%",
                "profit_factor": f"{filtered.profit_factor:.2f}" if filtered.profit_factor != float("inf") else "Inf",
                "total_trades": filtered.total_trades,
                "final_equity": f"${filtered.final_equity:,.2f}",
            },
            "trades_blocked": baseline.total_trades - filtered.total_trades,
        }

        result_data = {
            "asset": asset,
            "strategy": strategy,
            "timeframe": timeframe,
            "fee_preset": fee_preset,
            "cagr": f"{baseline.cagr * 100:.2f}%",
            "sharpe": f"{baseline.sharpe:.2f}",
            "max_drawdown": f"{baseline.max_drawdown * 100:.2f}%",
            "win_rate": f"{baseline.win_rate * 100:.1f}%",
            "profit_factor": f"{baseline.profit_factor:.2f}" if baseline.profit_factor != float("inf") else "Inf",
            "exposure": f"{baseline.exposure * 100:.1f}%",
            "total_trades": baseline.total_trades,
            "total_bars": baseline.total_bars,
            "initial_capital": f"${baseline.initial_capital:,.2f}",
            "final_equity": f"${baseline.final_equity:,.2f}",
            "equity_chart_json": baseline_equity_json,
            "trades": formatted_trades,
        }

        return templates.TemplateResponse("backtest.html", {
            **_tpl_base,
            "error": None,
            "result": result_data,
            "has_ml_model": True,
            "ml_filter_on": True,
            "learn_result": learn_result,
        })

    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(request: Request):
        """Render the backtest form page."""
        return templates.TemplateResponse("backtest.html", {
            "request": request,
            "error": None,
            "strategies": list(STRATEGY_REGISTRY.keys()),
            "fee_presets": list(_get_fee_presets().keys()),
            "available_assets": _available_assets(),
            "result": None,
            "has_ml_model": False,
            "ml_filter_on": False,
            "ml_result": None,
            "ml_trained": None,
            "learn_result": None,
            "initial_capital_raw": 10000,
        })

    # ------------------------------------------------------------------
    # Simulate — persistent paper-trading scenarios
    # ------------------------------------------------------------------

    def _load_all_scenarios() -> list[dict[str, Any]]:
        """Load all scenario JSON files from data/scenarios/."""
        if not _SCENARIOS_DIR.exists():
            return []
        scenarios = []
        for fp in sorted(_SCENARIOS_DIR.glob("*.json")):
            try:
                with open(fp) as f:
                    scenarios.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        return scenarios

    def _load_scenario(scenario_id: str) -> dict[str, Any] | None:
        """Load a single scenario by ID."""
        fp = _SCENARIOS_DIR / f"{scenario_id}.json"
        if not fp.exists():
            return None
        try:
            with open(fp) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
        """Fetch live data and run the paper trading simulation for a scenario.

        Returns a dict with all the data needed for the detail template.
        """
        asset = scenario["asset"]
        strategy_name = scenario["strategy"]
        timeframe = scenario["timeframe"]
        capital = float(scenario["capital"])
        fee_preset = scenario["fee_preset"]
        created_at = datetime.fromisoformat(scenario["created_at"])
        expires_at = datetime.fromisoformat(scenario["expires_at"])

        now = datetime.now(tz=timezone.utc)
        is_expired = now >= expires_at.replace(tzinfo=timezone.utc)
        end_dt = expires_at.replace(tzinfo=timezone.utc) if is_expired else now

        days_elapsed = (end_dt - created_at.replace(tzinfo=timezone.utc)).days
        days_total = int(scenario["duration_days"])
        days_remaining = max(0, days_total - days_elapsed)

        # Fetch live data with warmup
        warmup_days = 90
        start_dt = created_at.replace(tzinfo=timezone.utc) - timedelta(days=warmup_days)

        provider = YFinanceProvider()
        df = provider.fetch_ohlcv(asset, timeframe, start_dt, end_dt)
        if not df.empty and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        if df.empty or len(df) < 2:
            return {"error": f"No market data available for {asset}"}

        # Get strategy + params
        strat_obj = get_strategy(strategy_name)
        strat_cfg = _get_strategy_params()
        params = strat_cfg.get(strategy_name, {}).get(timeframe, {})
        optimized = ParameterOptimizer.load_optimized(strategy_name, asset, timeframe)
        if optimized is not None:
            params = {**params, **optimized}

        # Compute signals for all bars
        prepared = df.set_index("timestamp") if "timestamp" in df.columns else df.copy()
        if not isinstance(prepared.index, pd.DatetimeIndex):
            prepared.index = pd.to_datetime(prepared.index)
        all_signals = strat_obj.compute(prepared, asset, timeframe, params)

        # Find where simulation period starts (after warmup).
        # Use midnight on the creation date so daily candles (timestamped at
        # midnight) aren't skipped because of the intra-day creation time.
        # Default to len(df) so NO simulation runs if no bar falls on/after
        # the creation date (e.g. created on weekend, after market close).
        created_date = pd.Timestamp(created_at.date())
        sim_start_idx = len(df)
        for idx_i in range(len(df)):
            if df.iloc[idx_i]["timestamp"] >= created_date:
                sim_start_idx = idx_i
                break

        # Run paper trading simulator from sim_start_idx onward
        sim = PaperTradingSimulator(
            initial_capital=capital,
            fee_preset=fee_preset,
            position_size_pct=1.0,
        )

        equity_values: list[float] = []
        equity_dates: list[Any] = []

        for i in range(sim_start_idx, len(df)):
            row = df.iloc[i]
            candle = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "timestamp": str(row["timestamp"]),
            }
            step_signals: dict[str, Any] = {}
            if i < len(all_signals):
                step_signals = {asset: all_signals[i]}
            sim.step(candle, step_signals)
            equity = sim.get_portfolio_value({asset: float(row["close"])})
            equity_values.append(equity)
            equity_dates.append(row["timestamp"])

        # Equity chart (may be empty if no bars since creation)
        if equity_values:
            equity_series = pd.Series(
                equity_values, index=pd.to_datetime(equity_dates),
            )
            equity_chart = _build_equity_chart(equity_series)
            equity_chart_json = json.dumps(equity_chart)
        else:
            equity_chart_json = "null"

        # Portfolio summary
        current_price = float(df["close"].iloc[-1])
        final_equity = equity_values[-1] if equity_values else capital
        total_pnl = final_equity - capital
        total_pnl_pct = (total_pnl / capital) * 100 if capital > 0 else 0.0

        # Current signal
        current_signal_str = "HOLD"
        current_strength = 0.0
        current_explanation = ""
        if all_signals:
            last_sig = all_signals[-1]
            current_signal_str = last_sig.signal.value
            current_strength = last_sig.strength
            current_explanation = last_sig.explanation or ""

        # Format completed trades
        trades = sim.get_trade_ledger()
        formatted_trades = []
        for t in trades:
            formatted_trades.append({
                "entry_time": _format_timestamp(str(t.get("entry_time", ""))),
                "exit_time": _format_timestamp(str(t.get("timestamp", ""))),
                "entry_price": f"${t.get('entry_price', 0):.4f}",
                "exit_price": f"${t.get('exit_price', 0):.4f}",
                "pnl": f"${t.get('pnl', 0):.2f}",
                "pnl_pct": f"{t.get('pnl_pct', 0) * 100:.2f}%",
                "pnl_positive": t.get("pnl", 0) > 0,
            })

        # Format open positions
        positions = sim.get_positions()
        formatted_positions = []
        for pos_asset, pos_data in positions.items():
            unrealized_pnl = (
                (current_price - pos_data["entry_price"]) * pos_data["shares"]
            )
            unrealized_pct = (
                ((current_price / pos_data["entry_price"]) - 1.0) * 100
                if pos_data["entry_price"] > 0 else 0.0
            )
            formatted_positions.append({
                "asset": pos_asset,
                "shares": f"{pos_data['shares']:.6f}",
                "entry_price": f"${pos_data['entry_price']:.4f}",
                "current_price": _format_price(current_price),
                "unrealized_pnl": f"${unrealized_pnl:.2f}",
                "unrealized_pct": f"{unrealized_pct:.2f}%",
                "pnl_positive": unrealized_pnl >= 0,
                "entry_time": _format_timestamp(
                    str(pos_data.get("entry_time", "")),
                ),
            })

        return {
            "scenario": scenario,
            "is_expired": is_expired,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "days_total": days_total,
            "current_price": _format_price(current_price),
            "data_as_of": _format_timestamp(str(df["timestamp"].iloc[-1])),
            "current_signal": current_signal_str,
            "current_strength": f"{current_strength:.2f}",
            "explanation": current_explanation,
            "capital": f"${capital:,.2f}",
            "final_equity": f"${final_equity:,.2f}",
            "total_pnl": f"${total_pnl:,.2f}",
            "total_pnl_pct": f"{total_pnl_pct:.2f}%",
            "pnl_positive": total_pnl >= 0,
            "total_trades": len(trades),
            "equity_chart_json": equity_chart_json,
            "trades": formatted_trades,
            "positions": formatted_positions,
        }

    @app.get("/simulate", response_class=HTMLResponse)
    async def simulate_page(request: Request):
        """List all scenarios with a form to create new ones."""
        scenarios = _load_all_scenarios()

        now = datetime.now(tz=timezone.utc)
        formatted = []
        for sc in scenarios:
            expires = datetime.fromisoformat(sc["expires_at"])
            is_expired = now >= expires.replace(tzinfo=timezone.utc)
            created = datetime.fromisoformat(sc["created_at"])
            days_total = int(sc["duration_days"])
            days_elapsed = (now - created.replace(tzinfo=timezone.utc)).days
            days_remaining = max(0, days_total - days_elapsed)
            formatted.append({
                **sc,
                "is_expired": is_expired,
                "days_remaining": days_remaining,
                "created_fmt": _format_timestamp(sc["created_at"]),
                "strategy_fmt": sc["strategy"].replace("_", " ").title(),
                "capital_fmt": f"${float(sc['capital']):,.0f}",
            })

        return templates.TemplateResponse("simulate.html", {
            "request": request,
            "error": None,
            "scenarios": formatted,
            "strategies": list(STRATEGY_REGISTRY.keys()),
            "fee_presets": list(_get_fee_presets().keys()),
        })

    @app.post("/simulate/create", response_class=RedirectResponse)
    async def create_scenario(
        asset: str = Form(...),
        strategy: str = Form(...),
        timeframe: str = Form(default="1d"),
        capital: float = Form(default=10000.0),
        duration: int = Form(default=30),
        fee_preset: str = Form(default="crypto_major"),
    ):
        """Create a new paper trading scenario and redirect to it."""
        scenario_id = uuid.uuid4().hex[:8]
        now = datetime.now(tz=timezone.utc)
        scenario = {
            "id": scenario_id,
            "asset": asset.upper().strip(),
            "strategy": strategy,
            "timeframe": timeframe,
            "capital": capital,
            "fee_preset": fee_preset,
            "duration_days": duration,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=duration)).isoformat(),
        }

        _SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SCENARIOS_DIR / f"{scenario_id}.json", "w") as f:
            json.dump(scenario, f, indent=2)

        return RedirectResponse(
            url=f"/simulate/{scenario_id}", status_code=303,
        )

    @app.get("/simulate/{scenario_id}", response_class=HTMLResponse)
    async def scenario_detail(request: Request, scenario_id: str):
        """View a scenario with live simulation results."""
        scenario = _load_scenario(scenario_id)
        if scenario is None:
            return RedirectResponse(url="/simulate", status_code=302)

        try:
            result = _run_scenario(scenario)
        except Exception as exc:
            logger.exception("Error running scenario %s", scenario_id)
            result = {"error": str(exc)}

        return templates.TemplateResponse("simulate_detail.html", {
            "request": request,
            "error": result.get("error"),
            "result": result if "error" not in result else None,
            "scenario": scenario,
        })

    @app.post("/simulate/{scenario_id}/delete", response_class=RedirectResponse)
    async def delete_scenario(scenario_id: str):
        """Delete a scenario and redirect to the list."""
        fp = _SCENARIOS_DIR / f"{scenario_id}.json"
        if fp.exists():
            fp.unlink()
        return RedirectResponse(url="/simulate", status_code=303)

    @app.get("/screener", response_class=HTMLResponse)
    async def screener_page(
        request: Request,
        universe: str = Query(default="crypto"),
        tf: str = Query(default="1d"),
    ):
        """Render the screener results page."""
        screener = Screener()
        results = screener.scan(universe_name=universe, timeframe=tf)

        formatted_results = []
        for rank, r in enumerate(results, start=1):
            # Determine dominant signal
            buy_signals = [s for s in r.signals if s.signal == Signal.BUY]
            sell_signals = [s for s in r.signals if s.signal == Signal.SELL]

            if buy_signals:
                dominant = "BUY"
                strength = max(s.strength for s in buy_signals)
            elif sell_signals:
                dominant = "SELL"
                strength = max(s.strength for s in sell_signals)
            else:
                dominant = "HOLD"
                strength = 0.0

            formatted_results.append({
                "rank": rank,
                "asset": r.asset,
                "price": _format_price(r.price),
                "volume": f"{r.volume:,.0f}",
                "signal": dominant,
                "strength": f"{strength:.2f}",
                "composite_score": f"{r.composite_score:.3f}",
            })

        return templates.TemplateResponse("screener.html", {
            "request": request,
            "universe": universe,
            "timeframe": tf,
            "results": formatted_results,
        })

    @app.get("/api/signals", response_class=JSONResponse)
    async def api_signals(
        tf: str = Query(default="1d"),
    ):
        """JSON endpoint returning latest signals for all watchlist assets."""
        watchlist = _get_watchlist()
        store = _get_store()
        engine = SignalEngine()
        strategies_config = _get_strategy_params()

        output: list[dict[str, Any]] = []

        for item in watchlist:
            symbol = item["asset"]
            df = store.load(symbol, tf)

            if df.empty:
                output.append({"symbol": symbol, "signals": []})
                continue

            prepared = df.set_index("timestamp") if "timestamp" in df.columns else df
            signal_results = engine.run_single(
                prepared, symbol, tf, strategies_config=strategies_config,
            )

            output.append({
                "symbol": symbol,
                "price": float(df["close"].iloc[-1]),
                "timestamp": str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else "",
                "signals": [
                    {
                        "strategy": sr.strategy_name,
                        "signal": sr.signal.value,
                        "strength": round(sr.strength, 4),
                    }
                    for sr in signal_results
                ],
            })

        return JSONResponse(content={"timeframe": tf, "assets": output})

    @app.get("/api/health", response_class=JSONResponse)
    async def health_check():
        """Health check endpoint."""
        return JSONResponse(content={
            "status": "ok",
            "service": "market-signal-lab",
            "strategies": list(STRATEGY_REGISTRY.keys()),
        })

    return app
