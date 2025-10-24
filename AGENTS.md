# Repository Guidelines

## Onboarding Checklist
- Always read `README.md` and `docs/structure.md` before starting a new task so you are aligned with the current project layout and expectations.

## Project Structure & Module Organization
Strategy logic lives in `strategy/strategy.pine`, mirrored by the Python backtest in `optimize/strategy_model.py`. CLI orchestration and Optuna utilities reside under `optimize/`, with metrics, reporting, walk-forward analysis, and LLM helpers colocated in that package. Runtime knobs are centralised in `config/params.yaml` and sweep definitions in `config/backtest.yaml`. Market data acquisition is handled by `datafeed/binance_client.py` with cached CSVs managed by `datafeed/cache.py`. Keep ad-hoc notebooks and reports out of source control; generated assets default to `reports/` and `studies/`.

## Build, Test, and Development Commands
Set up a virtual environment before hacking:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run a full optimisation cycle with curated profiles:

```bash
python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
```

Append `--no-interactive` for unattended sweeps, or `--top-k 10` to trigger walk-forward re-ranking. Exercise the test suite via `pytest`, and scope to a module with commands such as `pytest tests/test_metrics.py::TestMetrics`.

## Coding Style & Naming Conventions
Target Python 3.12, using 4-space indentation, explicit imports, and type hints for public functions (see `optimize/run.py`). Keep module-level constants in `UPPER_SNAKE_CASE`, CLI entrypoints in `optimize/cli.py`, and auxiliary scripts under `tools/`. YAML keys follow lower-hyphen (`search.top_k`) conventions; mirror that pattern when extending the config. Document non-obvious heuristics with concise English comments even when code includes Korean log messages to aid cross-team reviews.

## Testing Guidelines
Pytest drives validation with fixtures inside `tests/test_common.py` and sample OHLCV data in `tests/tests_data/sample_ohlcv.csv`. Name new cases `test_<feature>.py` and prefer parametrised tests for search-space combinations. Maintain parity between Pine logic and Python emulation by extending `tests/test_compare_pine.py` alongside any strategy tweak. Aim to keep regression tests deterministic; seed Optuna samplers where randomness is introduced.

## Commit & Pull Request Guidelines
Initial snapshots lack Git metadata; adopt Conventional Commits (`feat:`, `fix:`, `refactor:`) when you initialise version control so downstream agents can filter history quickly. Keep subject lines under 72 characters and include context about impacted configs. Pull requests should describe the scenario exercised, mention relevant CLI flags, and attach or link the resulting `reports/<timestamp>/` artifacts when they inform reviewers.

## Security & Configuration Tips
Never commit raw API keys; store Gemini credentials in `.env` and reference them through `llm.api_key_env`. When sharing configs, strip personal Binance symbols or leverage settings from `config/backtest.yaml`. Large Optuna studies write to SQLite under `studies/`; archive or purge those files before distribution to avoid leaking experimentation history.
