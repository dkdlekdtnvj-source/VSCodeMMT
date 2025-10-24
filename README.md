# Pine Strategy Optimisation Toolkit

This repository contains a reproducible toolchain for tuning a Pine Script
strategy with Binance market data.  It ships with a reference Pine script,
the mirrored Python backtester, Optuna-based optimisation utilities,
reporting helpers, and walk-forward evaluation workflows.

## Project Layout

```
strategy/strategy.pine         # Pine reference strategy (optimiser-ready inputs)
config/params.yaml             # Single profile configuration
config/backtest.yaml           # Batch/sweep configuration
optimize/run.py                # CLI entry point
optimize/strategy_model.py     # Python equivalent of the Pine logic
optimize/metrics.py            # Performance metrics and scoring helpers
optimize/search_spaces.py      # Optuna search-space builders
optimize/wf.py                 # Walk-forward analysis
optimize/report.py             # CSV/JSON/heatmap reporting
optimize/llm.py                # Gemini-assisted candidate generator
datafeed/binance_client.py     # Binance downloader with retries
datafeed/cache.py              # CSV cache layer
reports/                       # Optimisation output (ignored by git)
studies/                       # Optuna SQLite storage (auto-created)
```

## Environment & Encoding

This codebase is UTF-8 only.  The following guard rails are in place:

- `.editorconfig` enforces UTF-8 with LF line endings for all tracked text
  files.
- `sitecustomize.py` reconfigures Python standard streams to UTF-8 and exports
  `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` for child processes.
- A dedicated guide is available in `docs/encoding.md` with platform-specific
  tips (e.g. PowerShell profile updates).

If you add scripts that read or write files, open them explicitly with
`encoding="utf-8"` to avoid regressions.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Optimisations

The primary entry point is `python -m optimize.run`:

```bash
python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
```

Useful flags:

- `--no-interactive` - skip CLI prompts (headless batch mode).
- `--top-k 10` - re-rank the best trials via walk-forward evaluation.
- `--symbol`, `--timeframe`, `--start`, `--end` - override dataset defaults.
- `--n-trials`, `--n-jobs` - control Optuna trial count and parallel workers.
- `--enable name1,name2`, `--disable name3` - flip feature gates.

The optimiser emits:

- `reports/<timestamp>/trials/` - JSONL + CSV trial archives.
- `reports/<timestamp>/results.csv` - aggregated metric summary.
- `studies/<slug>/` - Optuna storage (SQLite by default).

## Configuration Files

- `config/params.yaml` - feature toggles, Optuna objectives, search space,
  LLM settings, and risk controls.
- `config/backtest.yaml` - dataset specifications, symbol aliases, timeframes,
  and run tags.

Maintain parity between Pine and Python logic by updating
`strategy/strategy.pine`, `optimize/strategy_model.py`, and
`tests/test_compare_pine.py` together.

## Testing & Tooling

- Run the full suite with `pytest`.
- `tools/compare_pine_csv.py` compares TradingView CSV exports with Python
  backtest outputs.
- `docs/optimize_structure.md` summarises all optimiser modules after the
  alternative engine removal.

## Coding Guidelines

- Target Python 3.12 with 4-space indentation and explicit imports.
- Keep module-level constants in `UPPER_SNAKE_CASE`.
- Follow lower-hyphen conventions for YAML keys (e.g. `search.top_k`).
- Document non-obvious heuristics with concise English comments; keep log
  messages bilingual only when required by stakeholders.

## Security Tips

- Store API keys in `.env` and reference them with `llm.api_key_env`.
- Do not commit personal Binance symbols or Optuna study databases.
- Review `docs/encoding.md` for permanent UTF-8 environment tweaks on Windows.
