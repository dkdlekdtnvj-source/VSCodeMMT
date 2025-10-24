# Optimisation Module Layout

This note summarises the modules involved in the optimisation pipeline after
removing the experimental alternative engine. Every code path now routes through
the native Python backtester.

## Core runtime

- `optimize/strategy_model.py` - deterministic Python port of the TradingView
  strategy. Handles signal generation, order sizing, guard rails, and trade
  settlement.
- `optimize/main_loop.py` - CLI/orchestration layer. Seeds studies, schedules
  Optuna trials, manages dataset execution, and gathers reports.
- `optimize/metrics.py` - aggregates trades and equity curves, evaluates
  objectives, and flags anomalies.
- `optimize/search_spaces.py` - builds Optuna search spaces and offers helpers
  for mutation, random sampling, and Top-K walk-forward promotion.
- `optimize/report.py` - writes CSV/JSON reports plus dataset rankings.

## Supporting modules

- `optimize/indicators.py` - vectorised indicators shared between the backtester
  and analytics utilities.
- `optimize/state.py` - lightweight state containers for equity and filter
  context during a run.
- `optimize/wf.py` - walk-forward and purged k-fold evaluation helpers.
- `optimize/regime.py` - market regime tagging used in reports and LLM
  summaries.
- `optimize/llm.py` - Gemini integration for suggestion generation (optional).

## Data & configuration

- `config/params.yaml` - single profile configuration; toggles modules, defines
  Optuna objectives, and lists the search space.
- `config/backtest.yaml` - dataset definitions, symbol aliases, and timeframe
  batches.
- `datafeed/binance_client.py` / `datafeed/cache.py` - download and cache market
  data.
- `reports/` / `studies/` - generated artefacts (ignored by git).

## Testing pointers

- `tests/test_strategy_model.py` - asserts parity between Python and Pine for
  curated scenarios.
- `tests/test_metrics.py` - validates metrics and guard rails.
- `tests/test_run_helpers.py` - exercises dataset serialisation and executor
  plumbing.
- `tests/test_compare_pine.py` - CLI to diff Pine CSV exports with Python
  metrics/trade tables.

Use `tools/compare_pine_csv.py` to generate quick parity checks whenever the
strategy logic changes.
