"""전역 상수와 환경 관련 기본값을 한 곳에서 관리합니다."""
from __future__ import annotations

import multiprocessing
from importlib.util import find_spec
from pathlib import Path

# Numba 사용 가능 여부 ------------------------------------------------------------
NUMBA_AVAILABLE: bool = find_spec("numba") is not None

# 시스템 자원/병렬 처리 기본값 -----------------------------------------------------
CPU_COUNT: int = multiprocessing.cpu_count() or 1
DEFAULT_OPTUNA_JOBS: int = max(1, CPU_COUNT)
DEFAULT_DATASET_JOBS: int = max(1, CPU_COUNT)
SQLITE_SAFE_OPTUNA_JOBS: int = max(1, CPU_COUNT // 2)
SQLITE_SAFE_DATASET_JOBS: int = max(1, CPU_COUNT // 2)

# Optuna 스토리지 관련 -----------------------------------------------------------
DEFAULT_STORAGE_ENV_KEY: str = "OPTUNA_STORAGE"
DEFAULT_POSTGRES_STORAGE_URL: str = "postgresql://postgres:5432@127.0.0.1:5432/optuna"
POSTGRES_PREFIXES = ("postgresql://", "postgresql+psycopg://")

# 기본 출력 디렉터리 -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_ROOT = Path(r"D:\OneDrive - usk.ac.kr\문서\backtest")
DEFAULT_REPORT_ROOT = DEFAULT_STORAGE_ROOT / "백테스트"
DEFAULT_LOG_ROOT = DEFAULT_STORAGE_ROOT / "로그"
STUDY_ROOT = DEFAULT_STORAGE_ROOT / "스터디"

# 공통 규칙 및 페널티 -------------------------------------------------------------
NON_FINITE_PENALTY = -1e12
MIN_VOLUME_THRESHOLD = 100.0
MIN_TRADES_ENFORCED = 0

TRIAL_PROGRESS_FIELDS = [
    "number",
    "total_assets",
    "leverage",
    "chart_tf",
    "entry_tf",
    "use_htf",
    "htf_tf",
    "use_fixed_stop",
    "use_atr_stop",
    "use_atr_trail",
    "use_channel_stop",
    "stop_channel_type",
    "stop_channel_mult",
    "fixed_stop_pct",
    "atr_stop_len",
    "atr_stop_mult",
    "atr_trail_len",
    "atr_trail_mult",
    "score",
    "value",
    "state",
    "trades",
    "wins",
    "win_rate",
    "total_vol",
    "avg_vol",
    "max_dd",
    "liquidations",
    "valid",
    "timeframe",
    "pruned",
    "use_chandelier_exit",
    "chandelier_len",
    "chandelier_mult",
    "use_sar_exit",
    "sar_start",
    "sar_increment",
    "sar_maximum",
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
    "DEFAULT_POSTGRES_STORAGE_URL",
    "POSTGRES_PREFIXES",
    "REPO_ROOT",
    "DEFAULT_STORAGE_ROOT",
    "DEFAULT_REPORT_ROOT",
    "DEFAULT_LOG_ROOT",
    "STUDY_ROOT",
    "NON_FINITE_PENALTY",
    "MIN_VOLUME_THRESHOLD",
    "MIN_TRADES_ENFORCED",
    "TRIAL_PROGRESS_FIELDS",
]
