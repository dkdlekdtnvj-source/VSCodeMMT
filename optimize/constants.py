"""Repository-wide configuration defaults and shared constants."""
from __future__ import annotations

import multiprocessing
from importlib.util import find_spec
from pathlib import Path

# Numba availability ---------------------------------------------------------
NUMBA_AVAILABLE: bool = find_spec("numba") is not None

# CPU cores / worker defaults -------------------------------------------------
CPU_COUNT: int = multiprocessing.cpu_count() or 1
DEFAULT_OPTUNA_JOBS: int = max(1, CPU_COUNT)
DEFAULT_DATASET_JOBS: int = max(1, CPU_COUNT)
SQLITE_SAFE_OPTUNA_JOBS: int = max(1, CPU_COUNT // 2)
SQLITE_SAFE_DATASET_JOBS: int = max(1, CPU_COUNT // 2)

# Storage defaults ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(r"D:\OneDrive - usk.ac.kr\문서\backtest")
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_REPORT_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_STORAGE_ROOT = DEFAULT_OUTPUT_ROOT / "storage"
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "logs"
STUDY_ROOT = DEFAULT_OUTPUT_ROOT / "studies"

DEFAULT_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
STUDY_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_STORAGE_ENV_KEY: str = "OPTUNA_STORAGE"
DEFAULT_SQLITE_STORAGE_URL = f"sqlite:///{(STUDY_ROOT / 'optuna_default.db').as_posix()}"
POSTGRES_PREFIXES = ("postgresql://", "postgresql+psycopg://")

# Optimisation safeguards -----------------------------------------------------
NON_FINITE_PENALTY = -1e12
MIN_VOLUME_THRESHOLD = 100.0
MIN_TRADES_ENFORCED = 0

TRIAL_PROGRESS_FIELDS = [
    "number",
    "total_assets",
    "leverage",
    "timeframe",
    "timeframe_origin",
    "effective_timeframe",
    "htf_timeframe",
    "use_fixed_stop",
    "fixed_stop_pct",
    "use_band_exit",
    "band_exit_min_bars",
    "band_exit_mode",
    "band_exit_mult",
    "band_exit_summary",
    "score",
    "score_raw",
    "value",
    "value_raw",
    "state",
    "trades",
    "wins",
    "win_rate",
    "max_dd",
    "liquidations",
    "valid",
    "pruned",
    "anomaly_reason",
    "params",
    "skipped_datasets",
    "datetime_complete",
]

__all__ = [
    "NUMBA_AVAILABLE",
    "CPU_COUNT",
    "DEFAULT_OPTUNA_JOBS",
    "DEFAULT_DATASET_JOBS",
    "SQLITE_SAFE_OPTUNA_JOBS",
    "SQLITE_SAFE_DATASET_JOBS",
    "DEFAULT_STORAGE_ENV_KEY",
    "DEFAULT_SQLITE_STORAGE_URL",
    "POSTGRES_PREFIXES",
    "REPO_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_STORAGE_ROOT",
    "DEFAULT_REPORT_ROOT",
    "DEFAULT_LOG_ROOT",
    "STUDY_ROOT",
    "NON_FINITE_PENALTY",
    "MIN_VOLUME_THRESHOLD",
    "MIN_TRADES_ENFORCED",
    "TRIAL_PROGRESS_FIELDS",
]
