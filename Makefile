.PHONY: install dev test serve ingest demo clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,ml,ai]"

test:
	pytest tests/ -v --tb=short

serve:
	python -m app.cli serve

ingest-demo:
	python -m app.cli ingest --asset BTC-USD --tf 1d --start 2023-01-01
	python -m app.cli ingest --asset AAPL --tf 1d --start 2023-01-01

demo: ingest-demo
	python -m app.cli run-signals --watchlist
	python -m app.cli backtest --asset BTC-USD --strategy sma_crossover --tf 1d --start 2023-01-01 --end 2024-01-01
	python -m app.cli screen --universe crypto --tf 1d

clean:
	rm -rf data/*.parquet data/*.duckdb __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
