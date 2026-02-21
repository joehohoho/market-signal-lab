"""Click CLI for Market Signal Lab.

Entry point: ``msl`` (installed via pyproject.toml) or ``python -m app.cli``.
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from app.config import get_config
from app.logging import get_logger

logger = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD string into a datetime at midnight."""
    return datetime.strptime(value, "%Y-%m-%d")


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()


def _error(message: str) -> None:
    """Print a styled error message and exit."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="market-signal-lab")
def cli() -> None:
    """Market Signal Lab -- local-first trading research toolkit."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--asset", required=True, help="Asset symbol (e.g. BTC-USD, AAPL).")
@click.option("--tf", required=True, help="Timeframe (e.g. 1d, 15m).")
@click.option(
    "--start", required=True, help="Start date YYYY-MM-DD."
)
@click.option(
    "--end", default=None, help="End date YYYY-MM-DD (default: today)."
)
@click.option(
    "--provider", default=None, help="Data provider name (default: from config)."
)
def ingest(
    asset: str,
    tf: str,
    start: str,
    end: Optional[str],
    provider: Optional[str],
) -> None:
    """Download OHLCV candle data for an asset."""
    from data.ingest import ingest_asset  # lazy import

    end_str = end or _today_str()

    console.print(
        f"[bold cyan]Ingesting[/bold cyan] {asset} | {tf} | "
        f"{start} -> {end_str}"
        + (f" | provider={provider}" if provider else "")
    )

    try:
        start_dt = _parse_date(start)
        end_dt = _parse_date(end_str)
    except ValueError:
        _error("Dates must be in YYYY-MM-DD format.")

    logger.info("ingest: %s %s %s->%s provider=%s", asset, tf, start, end_str, provider)

    try:
        with console.status("[bold green]Fetching data..."):
            kwargs: dict = dict(
                symbol=asset,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
            )
            if provider is not None:
                kwargs["provider_name"] = provider
            result = ingest_asset(**kwargs)

        console.print(f"[green]Done.[/green] Ingested {result} rows for {asset}/{tf}.")
    except Exception as exc:
        logger.exception("ingest failed")
        _error(str(exc))


# ---------------------------------------------------------------------------
# import-local
# ---------------------------------------------------------------------------

@cli.command("import-local")
@click.option(
    "--dir",
    "directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Directory containing CSV files to import.",
)
@click.option(
    "--base",
    default="BTC,XBT,ETH",
    show_default=True,
    help="Comma-separated base currencies to import.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Scan and show what would be imported without writing.",
)
def import_local(directory: str, base: str, dry_run: bool) -> None:
    """Bulk-import local CSV files (Kraken OHLCV format) into the data store."""
    from data.import_local import import_local_csv, scan_csv_directory  # lazy

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    dir_path = Path(directory)
    base_filter = {b.strip().upper() for b in base.split(",")}

    console.print(
        f"[bold cyan]Import Local[/bold cyan] "
        f"dir={dir_path} | bases={','.join(sorted(base_filter))} "
        + ("[yellow](DRY RUN)[/yellow]" if dry_run else "")
    )

    if dry_run:
        files = scan_csv_directory(dir_path, base_filter=base_filter)
        if not files:
            console.print("[yellow]No matching files found.[/yellow]")
            return

        table = Table(title=f"Files to Import ({len(files)})")
        table.add_column("Source File", style="dim")
        table.add_column("Symbol", style="bold")
        table.add_column("Timeframe")

        for f in files:
            table.add_row(f.path.name, f.app_symbol, f.app_timeframe)

        console.print(table)
        console.print(f"[yellow]Dry run complete. {len(files)} files would be imported.[/yellow]")
        return

    # Real import with progress bar.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Importing...", total=None)

        def on_progress(parsed, idx, total):
            progress.update(
                task_id,
                total=total,
                completed=idx,
                description=f"Importing {parsed.path.name}",
            )

        results = import_local_csv(
            directory=dir_path,
            base_filter=base_filter,
            dry_run=False,
            progress_callback=on_progress,
        )

        if progress.tasks[task_id].total:
            progress.update(task_id, completed=progress.tasks[task_id].total)

    if not results:
        console.print("[yellow]No files were imported.[/yellow]")
        return

    # Summary table.
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    table = Table(title="Import Summary")
    table.add_column("Symbol", style="bold")
    table.add_column("Timeframe")
    table.add_column("Source File", style="dim")
    table.add_column("Rows Read", justify="right")
    table.add_column("Total Rows", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "[green]OK[/green]" if r.success else f"[red]FAIL: {r.error}[/red]"
        table.add_row(
            r.app_symbol,
            r.app_timeframe,
            r.source_file,
            f"{r.rows_imported:,}",
            f"{r.total_rows:,}",
            status,
        )

    console.print(table)

    total_rows = sum(r.rows_imported for r in successes)
    summary = (
        f"[green]Done.[/green] "
        f"{len(successes)} files imported ({total_rows:,} total rows). "
    )
    if failures:
        summary += f"[red]{len(failures)} failed.[/red]"
    console.print(summary)


# ---------------------------------------------------------------------------
# run-signals
# ---------------------------------------------------------------------------

@cli.command("run-signals")
@click.option(
    "--watchlist",
    default=None,
    help="Comma-separated assets to scan (default: use config watchlist).",
)
@click.option("--alert", is_flag=True, default=False, help="Send alerts for signals.")
def run_signals(watchlist: Optional[str], alert: bool) -> None:
    """Run strategy signals across a watchlist."""
    from signals.engine import SignalEngine  # lazy import

    cfg = get_config()
    wl_config = cfg.get("watchlist", [])
    strategies_config = cfg.get("strategies", {})

    # If the user supplied an explicit list, filter the config watchlist.
    if watchlist:
        requested = {s.strip() for s in watchlist.split(",")}
        wl_config = [w for w in wl_config if w.get("asset") in requested]
        if not wl_config:
            _error(f"None of the requested assets found in config: {requested}")

    console.print(
        f"[bold cyan]Running signals[/bold cyan] on "
        f"{len(wl_config)} watchlist item(s)..."
    )

    try:
        engine = SignalEngine()
        with console.status("[bold green]Evaluating strategies..."):
            results = engine.run_watchlist(wl_config, strategies_config)
    except Exception as exc:
        logger.exception("run-signals failed")
        _error(str(exc))

    if not results:
        console.print("[yellow]No signals generated.[/yellow]")
        return

    # Build rich table
    table = Table(title="Signal Results", show_lines=True)
    table.add_column("Asset", style="bold")
    table.add_column("TF")
    table.add_column("Strategy")
    table.add_column("Signal", justify="center")
    table.add_column("Strength", justify="right")

    for r in results:
        signal_val = r.signal.value if hasattr(r.signal, 'value') else str(r.signal)
        # Color-code signal
        if signal_val == "BUY":
            signal_display = "[green]BUY[/green]"
        elif signal_val == "SELL":
            signal_display = "[red]SELL[/red]"
        else:
            signal_display = f"[dim]{signal_val}[/dim]"

        strength = r.strength
        table.add_row(
            r.asset,
            r.timeframe,
            r.strategy_name,
            signal_display,
            f"{strength:.2f}",
        )

    console.print(table)

    # Optionally send alerts
    if alert:
        try:
            from alerts.manager import AlertManager

            mgr = AlertManager()
            with console.status("[bold green]Sending alerts..."):
                mgr.check_and_alert(results)
            console.print("[green]Alerts sent.[/green]")
        except Exception as exc:
            logger.exception("alert dispatch failed")
            _error(str(exc))


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--asset", required=True, help="Asset symbol (e.g. AAPL, BTC-USD).")
@click.option("--strategy", required=True, help="Strategy name (e.g. sma_crossover).")
@click.option("--tf", required=True, help="Timeframe (e.g. 1d).")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.option(
    "--fee-preset",
    default=None,
    help="Fee preset from config (e.g. liquid_stock, crypto_major).",
)
def backtest(
    asset: str,
    strategy: str,
    tf: str,
    start: str,
    end: str,
    fee_preset: Optional[str],
) -> None:
    """Run a backtest for a strategy on historical data."""
    from data.storage.parquet_store import ParquetStore  # lazy imports
    from backtest.engine import BacktestEngine
    from strategies import get_strategy

    try:
        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
    except ValueError:
        _error("Dates must be in YYYY-MM-DD format.")

    cfg = get_config()

    # Resolve fee preset
    params: dict = {}
    if fee_preset:
        presets = cfg.get("fee_presets", {})
        if fee_preset not in presets:
            available = ", ".join(presets.keys()) if presets else "(none)"
            _error(f"Unknown fee preset '{fee_preset}'. Available: {available}")
        params.update(presets[fee_preset])

    # Load strategy
    try:
        strat = get_strategy(strategy)
    except (KeyError, ValueError) as exc:
        _error(f"Strategy not found: {exc}")

    # Resolve strategy params from config
    strategy_params = cfg.get("strategies", {}).get(strategy, {}).get(tf, {})
    params.update(strategy_params)

    console.print(
        f"[bold cyan]Backtesting[/bold cyan] {strategy} on {asset}/{tf} | "
        f"{start} -> {end}"
    )

    # Load data
    store = ParquetStore(cfg.get("storage", {}).get("parquet_dir", "data/candles"))
    df = store.load(asset, tf, start=start_dt, end=end_dt)

    if df.empty:
        _error(
            f"No data for {asset}/{tf} in [{start}, {end}]. "
            "Run `msl ingest` first."
        )

    console.print(f"  Loaded {len(df):,} candles.")

    # Run backtest
    try:
        engine = BacktestEngine(**{k: v for k, v in params.items() if k in (
            "fee_pct", "slippage_pct", "spread_pct",
        )})
        with console.status("[bold green]Running backtest..."):
            result = engine.run(
                df=df,
                strategy=strat,
                params=params,
                asset=asset,
                timeframe=tf,
            )
    except Exception as exc:
        logger.exception("backtest failed")
        _error(str(exc))

    # Print metrics
    metrics = result.metrics if hasattr(result, "metrics") else {}
    if metrics:
        mtable = Table(title="Backtest Metrics", show_lines=True)
        mtable.add_column("Metric", style="bold")
        mtable.add_column("Value", justify="right")

        for key, value in metrics.items():
            if isinstance(value, float):
                mtable.add_row(key, f"{value:.4f}")
            else:
                mtable.add_row(key, str(value))

        console.print(mtable)

    # Print trade summary
    trades = result.trades if hasattr(result, "trades") else []
    if trades:
        ttable = Table(title=f"Trade Summary ({len(trades)} trades)", show_lines=True)
        ttable.add_column("#", justify="right", style="dim")
        ttable.add_column("Entry Date")
        ttable.add_column("Exit Date")
        ttable.add_column("Side")
        ttable.add_column("PnL %", justify="right")

        for i, trade in enumerate(trades, 1):
            pnl = trade.get("pnl_pct", 0.0)
            pnl_style = "green" if pnl >= 0 else "red"
            ttable.add_row(
                str(i),
                str(trade.get("entry_date", "")),
                str(trade.get("exit_date", "")),
                trade.get("side", ""),
                f"[{pnl_style}]{pnl:+.2f}%[/{pnl_style}]",
            )

        console.print(ttable)
    else:
        console.print("[yellow]No trades generated.[/yellow]")


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--universe",
    required=True,
    help="Universe name from config (e.g. crypto, stocks_daily).",
)
@click.option("--tf", required=True, help="Timeframe (e.g. 1d).")
@click.option("--top", default=20, show_default=True, help="Number of top results.")
def screen(universe: str, tf: str, top: int) -> None:
    """Screen a universe of assets and rank by signal strength."""
    from screener.scanner import Screener  # lazy import

    console.print(
        f"[bold cyan]Screening[/bold cyan] universe={universe} | tf={tf} | top={top}"
    )

    try:
        screener = Screener()
        with console.status("[bold green]Scanning universe..."):
            results = screener.scan(universe, tf)
    except Exception as exc:
        logger.exception("screen failed")
        _error(str(exc))

    if not results:
        console.print("[yellow]No results from screener.[/yellow]")
        return

    # Sort by strength descending and take top N
    ranked = sorted(results, key=lambda r: r.get("strength", 0.0), reverse=True)[:top]

    table = Table(title=f"Screener Results -- Top {top}", show_lines=True)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Asset", style="bold")
    table.add_column("Signal", justify="center")
    table.add_column("Strength", justify="right")
    table.add_column("Strategy")

    for rank, r in enumerate(ranked, 1):
        signal_val = r.get("signal", "")
        if signal_val == "BUY":
            signal_display = "[green]BUY[/green]"
        elif signal_val == "SELL":
            signal_display = "[red]SELL[/red]"
        else:
            signal_display = f"[dim]{signal_val}[/dim]"

        table.add_row(
            str(rank),
            r.get("asset", ""),
            signal_display,
            f"{r.get('strength', 0.0):.2f}",
            r.get("strategy", ""),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--host", default=None, help="Bind host (default: from config or 0.0.0.0).")
@click.option("--port", default=None, type=int, help="Bind port (default: from config or 8000).")
def serve(host: Optional[str], port: Optional[int]) -> None:
    """Start the FastAPI web server."""
    import uvicorn  # lazy import
    from api.routes import create_app  # lazy import

    cfg = get_config()
    server_cfg = cfg.get("server", {})

    resolved_host = host or server_cfg.get("host", "0.0.0.0")
    resolved_port = port or server_cfg.get("port", 8000)

    app = create_app()

    console.print(
        f"[bold cyan]Starting server[/bold cyan] at "
        f"http://{resolved_host}:{resolved_port}"
    )
    logger.info("serve: host=%s port=%d", resolved_host, resolved_port)

    uvicorn.run(app, host=resolved_host, port=resolved_port)


# ---------------------------------------------------------------------------
# train-ml (optional)
# ---------------------------------------------------------------------------

@cli.command("train-ml")
@click.option("--asset", required=True, help="Asset symbol (e.g. AAPL).")
@click.option("--tf", required=True, help="Timeframe (e.g. 1d).")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
def train_ml(asset: str, tf: str, start: str, end: str) -> None:
    """Train and validate an ML model on historical data."""
    try:
        from ml.models import train_and_validate  # lazy import
    except ImportError:
        _error(
            "ML dependencies not installed. "
            "Install with: pip install market-signal-lab[ml]"
        )

    from data.storage.parquet_store import ParquetStore

    try:
        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
    except ValueError:
        _error("Dates must be in YYYY-MM-DD format.")

    cfg = get_config()
    ml_cfg = cfg.get("ml", {})

    if not ml_cfg.get("enabled", False):
        console.print(
            "[yellow]Warning:[/yellow] ML is disabled in config. "
            "Proceeding anyway..."
        )

    console.print(
        f"[bold cyan]Training ML model[/bold cyan] for {asset}/{tf} | "
        f"{start} -> {end}"
    )

    # Load data
    store = ParquetStore(cfg.get("storage", {}).get("parquet_dir", "data/candles"))
    df = store.load(asset, tf, start=start_dt, end=end_dt)

    if df.empty:
        _error(
            f"No data for {asset}/{tf} in [{start}, {end}]. "
            "Run `msl ingest` first."
        )

    console.print(f"  Loaded {len(df):,} candles for training.")

    try:
        with console.status("[bold green]Training model..."):
            result = train_and_validate(df=df, asset=asset, timeframe=tf, config=ml_cfg)
    except Exception as exc:
        logger.exception("train-ml failed")
        _error(str(exc))

    # Print validation metrics
    metrics = result if isinstance(result, dict) else getattr(result, "metrics", {})
    if metrics:
        table = Table(title="ML Validation Metrics", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        console.print(table)
    else:
        console.print("[yellow]No metrics returned from training.[/yellow]")


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--asset", required=True, help="Asset symbol (e.g. BTC-USD).")
@click.option("--strategy", required=True, help="Strategy name (e.g. sma_crossover).")
@click.option("--tf", required=True, help="Timeframe (e.g. 1d).")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.option("--top", default=5, show_default=True, help="Number of top results to show.")
@click.option("--fee-preset", default=None, help="Fee preset (default: crypto_major).")
def optimize(
    asset: str,
    strategy: str,
    tf: str,
    start: str,
    end: str,
    top: int,
    fee_preset: Optional[str],
) -> None:
    """Grid-search strategy parameters and rank by Sharpe ratio."""
    from data.storage.parquet_store import ParquetStore  # lazy imports
    from backtest.optimizer import ParameterOptimizer

    try:
        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
    except ValueError:
        _error("Dates must be in YYYY-MM-DD format.")

    resolved_fee_preset = fee_preset or "crypto_major"

    console.print(
        f"[bold cyan]Optimizing[/bold cyan] {strategy} on {asset}/{tf} | "
        f"{start} -> {end} | fee_preset={resolved_fee_preset}"
    )

    # Load data
    cfg = get_config()
    store = ParquetStore(cfg.get("storage", {}).get("parquet_dir", "data/candles"))
    df = store.load(asset, tf, start=start_dt, end=end_dt)

    if df.empty:
        _error(
            f"No data for {asset}/{tf} in [{start}, {end}]. "
            "Run `msl ingest` first."
        )

    console.print(f"  Loaded {len(df):,} candles.")

    # Run optimizer
    try:
        optimizer = ParameterOptimizer(
            strategy_name=strategy,
            asset=asset,
            timeframe=tf,
            df=df,
            fee_preset=resolved_fee_preset,
            top_n=top,
        )
        with console.status("[bold green]Running parameter grid search..."):
            results = optimizer.run()
    except Exception as exc:
        logger.exception("optimize failed")
        _error(str(exc))

    if not results:
        console.print("[yellow]No viable parameter sets found (all had < 3 trades).[/yellow]")
        return

    # Build rich table
    table = Table(title=f"Top {len(results)} Parameter Sets", show_lines=True)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Params")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("WinRate", justify="right")
    table.add_column("Trades", justify="right")

    for rank, r in enumerate(results, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        cagr_pct = r["cagr"] * 100
        dd_pct = r["max_drawdown"] * 100
        wr_pct = r["win_rate"] * 100

        table.add_row(
            str(rank),
            params_str,
            f"{r['sharpe']:.2f}",
            f"{cagr_pct:+.2f}%",
            f"{dd_pct:.2f}%",
            f"{wr_pct:.1f}%",
            str(r["total_trades"]),
        )

    console.print(table)

    # Save best params
    saved_path = optimizer.save_best()
    if saved_path:
        console.print(
            f"[green]Best params saved to:[/green] {saved_path}"
        )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
