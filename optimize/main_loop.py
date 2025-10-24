"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import OrderedDict, deque
from collections.abc import Sequence as AbcSequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Literal

import numpy as np
import optuna
import optuna.storages
import pandas as pd
import yaml
import multiprocessing
import ccxt
import sqlalchemy
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import make_url
from optuna.trial import TrialState


from datafeed.cache import DataCache
from optimize.common import _resolve_symbol_entry, normalize_tf
from optimize.metrics import (
    EPS,
    ObjectiveSpec,
    Trade,
    aggregate_metrics,
    equity_curve_from_returns,
    evaluate_objective_values,
    normalise_objectives,
)
from optimize.constants import (
    CPU_COUNT,
    DEFAULT_DATASET_JOBS,
    DEFAULT_LOG_ROOT,
    DEFAULT_OPTUNA_JOBS,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SQLITE_STORAGE_URL,
    DEFAULT_STORAGE_ENV_KEY,
    DEFAULT_STORAGE_ROOT,
    MIN_TRADES_ENFORCED,
    MIN_VOLUME_THRESHOLD,
    NON_FINITE_PENALTY,
    POSTGRES_PREFIXES,
    REPO_ROOT,
    SQLITE_SAFE_DATASET_JOBS,
    SQLITE_SAFE_OPTUNA_JOBS,
    STUDY_ROOT,
    TRIAL_PROGRESS_FIELDS,
)
from optimize.study_setup import (
    _mask_storage_url,
    _make_rdb_storage,
    _make_sqlite_storage,
)
from optimize.report import generate_reports, write_bank_file, write_trials_dataframe
from optimize.search_spaces import (
    build_space,
    grid_choices,
    mutate_around,
    random_parameters,
    sample_parameters,
)
from optimize.strategy_model import run_backtest as _native_run_backtest
from optimize.wf import run_purged_kfold, run_walk_forward
from optimize.regime import detect_regime_label, summarise_regime_performance
from optimize.llm import LLMSuggestions, generate_llm_candidates as _native_generate_llm_candidates
from optuna.exceptions import StorageInternalError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import shutil
import msvcrt
import time


HTF_ENABLED = False

# Preset label for the combined 1m/3m/5m optimisation surface.
MERGED_TIMEFRAME_LABEL = "1,3,5merged"
MERGED_TIMEFRAME_COMPONENTS = ["1m", "3m", "5m"]
MERGED_TIMEFRAME_MAP = {MERGED_TIMEFRAME_LABEL: MERGED_TIMEFRAME_COMPONENTS}

# Global configuration populated by ``optimize.run`` before invocation.
backtest_cfg: Dict[str, object] = {}

# Allows test suites to monkeypatch ``run_backtest`` through ``optimize.run``.
run_backtest = _native_run_backtest


def _resolve_llm_generator():
    run_module = sys.modules.get("optimize.run")
    candidate = getattr(run_module, "generate_llm_candidates", None) if run_module else None
    if callable(candidate):
        return candidate
    return _native_generate_llm_candidates


generate_llm_candidates = _native_generate_llm_candidates


GLOBAL_STOP_EVENT = Event()
STOP_INSTRUCTION_SHOWN = False


class _TripleBacktickStopper:
    """Watch stdin for a triple backtick pattern and trigger a stop."""

    def __init__(self, required_presses: int = 3) -> None:
        self.required_presses = max(1, required_presses)
        self._event = Event()
        self._thread: Optional[Thread] = None
        self._lock = Lock()
        self._notified = False

    def start(self) -> None:
        if self._thread is not None:
            return
        thread = Thread(target=self._monitor_input, name="stop-listener", daemon=True)
        self._thread = thread
        thread.start()

    def _monitor_input(self) -> None:
        consecutive = 0
        while not self._event.is_set() and not GLOBAL_STOP_EVENT.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            token = line.strip()
            if not token:
                consecutive = 0
                continue
            if all(ch == "`" for ch in token):
                consecutive += token.count("`")
            else:
                consecutive = 0
            if consecutive >= self.required_presses:
                GLOBAL_STOP_EVENT.set()
                self._event.set()
                break

    @property
    def triggered(self) -> bool:
        return self._event.is_set() or GLOBAL_STOP_EVENT.is_set()

    def callback(self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        if not self.triggered:
            return
        with self._lock:
            if not self._notified:
                self._notified = True
            try:
                study.stop()
            except Exception:
                pass


def _create_triple_backtick_stopper() -> Optional[_TripleBacktickStopper]:
    """Prepare the triple-backtick stopper when running interactively."""

    if GLOBAL_STOP_EVENT.is_set():
        return None
    try:
        is_tty = sys.stdin.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return None

    stopper = _TripleBacktickStopper()
    stopper.start()

    global STOP_INSTRUCTION_SHOWN
    if not STOP_INSTRUCTION_SHOWN:
        LOGGER.info("중단하려면 백틱(`)을 세 번 입력한 뒤 Enter 키를 누르세요.")
        STOP_INSTRUCTION_SHOWN = True

    return stopper

def fetch_top_usdt_perp_symbols(
    limit: int = 50,
    exclude_symbols: Optional[Sequence[str]] = None,
    exclude_keywords: Optional[Sequence[str]] = None,
    min_price: Optional[float] = None,
) -> List[str]:
    """Return the most liquid Binance USDT-M perpetual symbols by 24h quote volume."""

    ex = ccxt.binanceusdm(
        {
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        }
    )
    ex.load_markets()
    tickers = ex.fetch_tickers()

    exclude_symbols_set = set(exclude_symbols or [])
    exclude_keywords = list(exclude_keywords or [])
    keyword_pattern = (
        re.compile("|".join(re.escape(k) for k in exclude_keywords))
        if exclude_keywords
        else None
    )

    rows: List[Tuple[str, float]] = []
    for sym, ticker in tickers.items():
        market = ex.market(sym)
        if not market.get("swap", False):
            continue
        if market.get("quote") != "USDT":
            continue

        unified = market.get("id", "")
        if unified in exclude_symbols_set:
            continue
        if keyword_pattern and keyword_pattern.search(unified):
            continue

        last = ticker.get("last")
        if min_price is not None:
            if last is None or float(last) < float(min_price):
                continue

        quote_volume = ticker.get("quoteVolume")
        try:
            if quote_volume is None:
                base_volume = ticker.get("baseVolume") or 0
                last_price = ticker.get("last") or 0
                quote_volume = base_volume * last_price
            rows.append((unified, float(quote_volume)))
        except (TypeError, ValueError):
            continue

    rows.sort(key=lambda item: item[1], reverse=True)
    return [f"BINANCE:{symbol}" for symbol, _ in rows[:limit]]


LOGGER = logging.getLogger("optimize")

# ---------------------------------------------------------------------------
# Warning management
#
# Optuna exposes a number of experimental features (e.g. heartbeat_interval,
# multivariate sampling) that trigger `ExperimentalWarning` on import or
# configuration.  These warnings are not actionable for end users of this
# framework and clutter the console.  Filter them globally so they do not
# interfere with logging output.  Likewise, other libraries may raise
# `UserWarning` when a supplied option has no effect; these should be handled
import warnings
try:
    from optuna.exceptions import ExperimentalWarning  # type: ignore
except Exception:
    ExperimentalWarning = None  # type: ignore

if ExperimentalWarning is not None:
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Set the number of threads for OpenMP/BLAS libraries to the number of
# available CPU cores.  This improves performance of underlying numerical
# routines by utilising multiple cores during heavy vectorized operations.
_threads = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _threads)
os.environ.setdefault("MKL_NUM_THREADS", _threads)

TRIAL_LOG_WRITE_LOCK = Lock()








@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=3),
    retry=(
        retry_if_exception_type(sqlalchemy.exc.OperationalError)
        | retry_if_exception_type(StorageInternalError)
    ),
)
def _safe_sample_parameters(
    trial: optuna.trial.Trial, space: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Sample parameters with retries to mitigate transient SQLite locks."""

    return sample_parameters(trial, space)


# Flag toggled by CLI switches to reduce metric output for quick runs.
simple_metrics_enabled: bool = False


_INITIAL_BALANCE_KEYS = (
    "InitialCapital",
    "InitialEquity",
    "InitialBalance",
    "StartingBalance",
)


# Keys used when `--basic-factors-only` is enabled.  This set should mirror
# the parameters defined in `config/params.yaml` so that only the core
# oscillator, volatility channel, flux and exit parameters are swept.  If you
# modify `params.yaml` or add new tuneable inputs, update this set
# accordingly.  Removing unused or unsupported names from this set prevents
# accidental sampling of unrelated stop/trail/slippage options.
BASIC_FACTOR_KEYS = {
    # Oscillator & signal lengths
    "oscLen",
    "signalLen",
    # Bollinger/Keltner channels
    "bbLen",
    "bbMult",
    "kcLen",
    "kcMult",
    # Directional flux
    "fluxLen",
    "fluxSmoothLen",
    "fluxDeadzone",
    "useFluxHeikin",
    "useModFlux",
    # Squeeze momentum style selection (KC/AVG/Deluxe/Mod)
    "basisStyle",
    "momStyle",
    "useNormClip",
    "normClipLimit",
    # Dynamic threshold & gates
    "useDynamicThresh",
    "useSymThreshold",
    "statThreshold",
    "buyThreshold",
    "sellThreshold",
    "dynLen",
    "dynMult",
    "maType",
    # Exit logic
    "exitOpposite",
    "useMomFade",
    "momFadeMinAbs",
    "useBandExit",
    "bandExitMinBars",
    # Risk & capital controls that must remain active even in basic profile mode
    "leverage",
    "fixedStopPct",
    "usePyramiding",
}


TRIAL_STATE_LABELS = {
    "COMPLETE": "COMPLETE",
    "PRUNED": "PRUNED",
    "FAIL": "FAIL",
    "RUNNING": "RUNNING",
    "WAITING": "WAITING",
}


def _format_trial_state(state: Optional[TrialState]) -> str:
    """Render an Optuna TrialState value as a human-friendly label."""

    if isinstance(state, TrialState):
        key = state.name
    else:
        text = "" if state is None else str(state)
        key = text.split(".")[-1] if text else ""
    key_upper = key.upper()
    if not key_upper:
        return "UNKNOWN"
    return TRIAL_STATE_LABELS.get(key_upper, key_upper)


def _utcnow_isoformat() -> str:
    """Return the current UTC timestamp formatted with an ISO8601 Z suffix."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _register_study_reference(
    study_storage: Optional[Path],
    *,
    storage_meta: Dict[str, object],
    study_name: Optional[str] = None,
) -> None:
    """Persist study storage metadata for later reuse."""

    if study_storage is None:
        return

    backend = str(storage_meta.get("backend") or "none").lower()
    if backend in {"", "none"}:
        return

    registry_dir = _study_registry_dir(study_storage)
    registry_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = registry_dir / "storage.json"

    payload: Dict[str, object] = {
        "updated_at": _utcnow_isoformat(),
        "backend": backend,
        "study_name": study_name,
        "storage_url_env": storage_meta.get("env_key"),
        "env_value_present": storage_meta.get("env_value_present"),
    }

    url_value = storage_meta.get("url")
    if isinstance(url_value, str) and url_value:
        payload["storage_url"] = url_value
        try:
            payload["storage_url_masked"] = make_url(url_value).render_as_string(
                hide_password=True
            )
        except Exception:
            payload["storage_url_masked"] = url_value

    if backend == "sqlite":
        payload["sqlite_path"] = storage_meta.get("path") or str(study_storage)
        payload["allow_parallel"] = storage_meta.get("allow_parallel")
    else:
        pool_meta = storage_meta.get("pool")
        if isinstance(pool_meta, dict) and pool_meta:
            payload["pool"] = pool_meta
        if storage_meta.get("connect_timeout") is not None:
            payload["connect_timeout"] = storage_meta.get("connect_timeout")
        if storage_meta.get("isolation_level"):
            payload["isolation_level"] = storage_meta.get("isolation_level")
        if storage_meta.get("statement_timeout_ms") is not None:
            payload["statement_timeout_ms"] = storage_meta.get("statement_timeout_ms")

    pointer_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _sanitise_storage_meta(storage_meta: Dict[str, object]) -> Dict[str, object]:
    if not storage_meta:
        return {}

    cleaned = copy.deepcopy(storage_meta)
    url_value = cleaned.get("url")
    if isinstance(url_value, str) and url_value:
        try:
            cleaned["url"] = make_url(url_value).render_as_string(hide_password=True)
        except Exception:
            cleaned["url"] = "***invalid-url***"
    return cleaned


def _slugify_symbol(symbol: str) -> str:
    text = symbol.split(":")[-1]
    return text.replace("/", "").replace(" ", "")


def _slugify_timeframe(timeframe: Optional[str]) -> str:
    if not timeframe:
        return ""
    return str(timeframe).replace("/", "_").replace(" ", "")


def _space_hash(space: Dict[str, object]) -> str:
    payload = json.dumps(space or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _restrict_to_basic_factors(
    space: Dict[str, Dict[str, object]], *, enabled: bool = True
) -> Dict[str, Dict[str, object]]:
    """Return a filtered space containing only the basic-factor parameters."""

    if not space:
        return {}

    if not enabled:
        return {name: dict(spec) for name, spec in space.items()}

    filtered: Dict[str, Dict[str, object]] = {}
    for name, spec in space.items():
        if name in BASIC_FACTOR_KEYS:
            filtered[name] = dict(spec)
    return filtered


def _collect_timeframe_choices(datasets: Sequence[DatasetSpec]) -> List[str]:
    """Return sorted unique LTF values available across datasets."""

    choices: Set[str] = set()
    for dataset in datasets:
        if dataset.timeframe:
            choices.add(str(dataset.timeframe))
    return sorted(choices)


def _ensure_timeframe_param(
    space: Dict[str, Dict[str, object]],
    datasets: Sequence[DatasetSpec],
    search_cfg: Mapping[str, object],
) -> Tuple[Dict[str, Dict[str, object]], bool]:
    return space, False


def _filter_basic_factor_params(
    params: Dict[str, object], *, enabled: bool = True
) -> Dict[str, object]:
    """Filter parameter dict so only basic-factor keys remain when enabled."""

    if not params:
        return {}
    if not enabled:
        return dict(params)
    return {key: value for key, value in params.items() if key in BASIC_FACTOR_KEYS}


def _space_value(space: Optional[Mapping[str, Dict[str, object]]], name: str, key: str) -> Optional[object]:
    if not isinstance(space, Mapping):
        return None
    spec = space.get(name)
    if not isinstance(spec, Mapping):
        return None
    value = spec.get(key)
    return value if value is not None else None


def _ensure_channel_params(
    params: Dict[str, object],
    space: Optional[Mapping[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Ensure Bollinger/Keltner configuration fields have defaults."""

    if not isinstance(params, dict):
        return {}

    def _coerce_int(value: object, default: Optional[int]) -> Optional[int]:
        try:
            if value is None:
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _coerce_float(value: object, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    patched = dict(params)
    kc_len = _coerce_int(patched.get("kcLen"), None)
    kc_mult = None
    try:
        kc_mult = float(patched.get("kcMult"))
    except (TypeError, ValueError):
        kc_mult = None

    bb_len_default = kc_len
    if bb_len_default is None:
        bb_len_default = _coerce_int(_space_value(space, "bbLen", "default"), None)
    if bb_len_default is None:
        bb_len_default = _coerce_int(_space_value(space, "bbLen", "min"), 20)
    if bb_len_default is None:
        bb_len_default = 20

    bb_mult_default = kc_mult
    if bb_mult_default is None:
        raw_default = _space_value(space, "bbMult", "default")
        if raw_default is None:
            raw_default = _space_value(space, "bbMult", "min")
        bb_mult_default = _coerce_float(raw_default, 1.4)
    bb_mult_default = _coerce_float(bb_mult_default, 1.4)

    if patched.get("bbLen") in (None, ""):
        patched["bbLen"] = bb_len_default
    else:
        patched["bbLen"] = _coerce_int(patched.get("bbLen"), bb_len_default)

    if patched.get("bbMult") in (None, ""):
        patched["bbMult"] = bb_mult_default
    else:
        patched["bbMult"] = _coerce_float(patched.get("bbMult"), bb_mult_default)

    return patched


def _band_exit_summary(metrics: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """Extract a compact band-exit summary from metric payloads, choosing BB or KC only."""

    if not isinstance(metrics, Mapping):
        return None

    def _float(key: str, default: float = 0.0) -> float:
        try:
            value = metrics.get(key, default)
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(numeric):
            return default
        return float(numeric)

    enabled_score = _float("BandExitEnabled", 0.0)
    total = _float("BandExitCount", 0.0)
    bb_count = _float("BandExitBBCount", 0.0)
    kc_count = _float("BandExitKCCount", 0.0)
    bb_mult = _float("BandExitBBMult", 0.0)
    kc_mult = _float("BandExitKCMult", 0.0)

    enabled = enabled_score >= 0.5 or total > 0.0 or bb_count > 0.0 or kc_count > 0.0
    if (
        not enabled
        and bb_mult == 0.0
        and kc_mult == 0.0
        and total == 0.0
        and bb_count == 0.0
        and kc_count == 0.0
    ):
        return None

    # Decide strictly between BB or KC when touches occurred; otherwise NONE.
    if total > 0.0:
        mode = "BB" if bb_count >= kc_count else "KC"
    else:
        mode = "NONE"

    mult = bb_mult if mode == "BB" else kc_mult if mode == "KC" else 0.0

    return {
        "enabled": bool(enabled),
        "total": total,
        "bb_count": bb_count,
        "kc_count": kc_count,
        "bb_mult": bb_mult,
        "kc_mult": kc_mult,
        "mode": mode,
        "mult": mult,
    }


def _format_band_exit_summary(summary: Mapping[str, object]) -> str:
    """Render a human-readable band exit summary string (BB/KC only)."""

    if not isinstance(summary, Mapping):
        return ""

    enabled = bool(summary.get("enabled"))
    if not enabled:
        return "Band exit disabled"

    mode = str(summary.get("mode") or "").upper()
    mult_value = summary.get("mult")
    try:
        mult = float(mult_value)
    except (TypeError, ValueError):
        mult = 0.0

    if mode in {"BB", "KC"}:
        return f"{mode} × {mult:.3f}"
    if mode == "NONE":
        return "Band exit enabled (no touches)"
    # Fallback, should not happen with strict BB/KC/NONE
    return "Band exit enabled"


def _order_mapping(
    payload: Mapping[str, object],
    preferred_order: Optional[Sequence[str]] = None,
    *,
    priority: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Return a dictionary ordered by priority and preferred key sequences."""

    if not isinstance(payload, Mapping):
        return {}

    ordered: "OrderedDict[str, object]" = OrderedDict()

    for key in priority or ():
        if key in payload and key not in ordered:
            ordered[key] = payload[key]

    if preferred_order:
        for key in preferred_order:
            if key in payload and key not in ordered:
                ordered[key] = payload[key]

    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value

    return dict(ordered)


def _git_revision() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.parent / f"{path.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)

    run_identifier = log_dir.parent.name or "run"
    central_dir = DEFAULT_LOG_ROOT / run_identifier
    central_dir.mkdir(parents=True, exist_ok=True)

    target_paths = [log_dir / "run.log", central_dir / "run.log"]

    for handler in list(LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                existing_path = Path(handler.baseFilename)
            except Exception:
                continue
            if existing_path in target_paths:
                LOGGER.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for path in target_paths:
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _build_run_tag(
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[str, str, str, str]:
    symbol = params_cfg.get("symbol") or (datasets[0].symbol if datasets else "unknown")
    timeframe = (
        params_cfg.get("timeframe")
        or (datasets[0].timeframe if datasets else "multi")
    )
    htf = None
    if HTF_ENABLED:
        htf = (
            params_cfg.get("htf_timeframe")
            or params_cfg.get("htf")
            or (datasets[0].htf_timeframe if datasets and datasets[0].htf_timeframe else "nohtf")
        )
        if not htf:
            htf = "nohtf"
    symbol_slug = _slugify_symbol(str(symbol))
    timeframe_slug = str(timeframe).replace("/", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    parts = [timestamp, symbol_slug, timeframe_slug]
    if run_tag:
        parts.append(run_tag)
    return timestamp, symbol_slug, timeframe_slug, "_".join(filter(None, parts))


def _coerce_min_trades_value(value: object) -> Optional[int]:
    """Convert ``value`` to a non-negative integer if possible."""

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return None
        value = text

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric):
        return None

    return max(0, int(round(numeric)))


def _coerce_config_int(value: object, *, minimum: int, name: str) -> Optional[int]:
    """Attempt to coerce a config value to an int respecting a minimum bound."""

    if value is None:
        return None

    raw_value = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        raw_value = text

    try:
        numeric = int(float(raw_value))
    except (TypeError, ValueError):
        return None

    if numeric < minimum:
        return None

    return numeric


def _timeframe_lookup_keys(timeframe: Optional[str], htf: Optional[str]) -> List[str]:
    """Return candidate keys for matching timeframe-specific constraints."""

    keys: List[str] = []

    def _normalise(token: str) -> List[str]:
        variants = [token]
        variants.append(token.lower())
        variants.append(token.upper())
        compact = token.replace("/", "").replace(" ", "")
        if compact and compact not in variants:
            variants.append(compact)
            variants.append(compact.lower())
            variants.append(compact.upper())
        return list(dict.fromkeys(variants))

    if timeframe:
        keys.extend(_normalise(timeframe))

    if timeframe and htf:
        keys.extend(_normalise(f"{timeframe}@{htf}"))

    return list(dict.fromkeys(keys))


    """Extract ``min_trades`` requirement from an arbitrary mapping entry."""

    if entry is None:
        return None

    if isinstance(entry, (int, float, str)):
        return _coerce_min_trades_value(entry)

    if not isinstance(entry, dict):
        return None

    priority_keys = [
        "min_trades_test",
        "minTradesTest",
        "min_trades",
        "minTrades",
        "oos",
        "OOS",
        "test",
        "Test",
        "value",
        "Value",
        "default",
        "Default",
    ]

    for key in priority_keys:
        if key not in entry:
            continue
        candidate = entry[key]
        if isinstance(candidate, dict):
            resolved = _extract_min_trades_from_mapping(candidate)
        else:
            resolved = _coerce_min_trades_value(candidate)
        if resolved is not None:
            return resolved

    # Fallback: inspect nested mappings for a usable value.
    for candidate in entry.values():
        resolved = _extract_min_trades_from_mapping(candidate)
        if resolved is not None:
            return resolved

    return None


def _resolve_dataset_min_trades(
    dataset: "DatasetSpec",
    *,
    constraints: Optional[Dict[str, object]] = None,
    risk: Optional[Dict[str, object]] = None,
    explicit: Optional[object] = None,
) -> Optional[int]:
    """Resolve the minimum trade requirement for a dataset."""

    constraints = constraints or {}
    candidates: List[Optional[int]] = []

    candidates.append(_coerce_min_trades_value(explicit))

    lookup_keys = _timeframe_lookup_keys(dataset.timeframe, dataset.htf_timeframe)
    timeframe_rule_keys = [
        "timeframes",
        "per_timeframe",
        "perTimeframe",
        "timeframe_rules",
        "timeframeRules",
        "min_trades_by_timeframe",
        "minTradesByTimeframe",
    ]

    for container_key in timeframe_rule_keys:
        rules = constraints.get(container_key)
        if not isinstance(rules, dict):
            continue
        for key in lookup_keys:
            if key in rules:
                candidates.append(_extract_min_trades_from_mapping(rules[key]))
                break

    candidates.append(_coerce_min_trades_value(constraints.get("min_trades_test")))

    if isinstance(risk, dict):
        candidates.append(_coerce_min_trades_value(risk.get("min_trades")))

    for candidate in candidates:
        if candidate is not None:
            return candidate

    return None


def _run_dataset_backtest_task(
    dataset_ref: object,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """Execute ``run_backtest`` for a single dataset.

    ``dataset_ref`` may be a :class:`DatasetSpec` when running in a thread
    executor, or a dataset identifier string when invoked inside a process
    pool.  For process mode the identifier is resolved back to the cached
    ``DatasetSpec`` via ``_process_pool_initializer``.
    """

    dataset = _resolve_dataset_reference(dataset_ref)

    run_module = sys.modules.get("optimize.run")
    run_backtest_fn = getattr(run_module, "run_backtest", None) if run_module else None
    if not callable(run_backtest_fn):
        run_backtest_fn = run_backtest

    return run_backtest_fn(
        dataset.df,
        params,
        fees,
        risk,
        htf_df=dataset.htf,
        min_trades=min_trades,
        chart_df=dataset.chart_df,
    )


def _resolve_output_directory(
    base: Optional[Path],
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[Path, Dict[str, str]]:
    ts, symbol_slug, timeframe_slug, tag = _build_run_tag(datasets, params_cfg, run_tag)
    
    now = datetime.now()
    date_str = now.strftime("%y%m%d")
    time_str = now.strftime("%H%M")
    
    timeframe = timeframe_slug.split('_')[0]
    
    symbol_name = symbol_slug.replace("USDT", "").capitalize()

    output_dir_name = f"{timeframe},{symbol_name}"
    
    if base is None:
        root = Path("D:\\OneDrive - usk.ac.kr\\\\backtest")
        output = root / date_str / time_str / output_dir_name
    else:
        output = base

    output = _next_available_dir(output)
    output.mkdir(parents=True, exist_ok=False)
    
    manifest = {
        "timestamp": ts,
        "symbol": symbol_slug,
        "timeframe": timeframe_slug,
        "tag": tag,
    }
    if HTF_ENABLED:
        manifest["htf_timeframe"] = _slugify_timeframe(_extract_primary_htf(params_cfg, datasets))
    return output, manifest


def _write_manifest(
    output_dir: Path,
    *,
    manifest: Dict[str, object],
) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _load_json(path: Path) -> Dict[str, object]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _parse_timeframe_grid(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    combos: List[str] = []
    text = str(raw).replace("\n", ",").replace(";", ",")
    for token in text.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if "@" in candidate:
            ltf, _ = candidate.split("@", 1)
        elif ":" in candidate:
            ltf, _ = candidate.split(":", 1)
        else:
            ltf = candidate
        ltf = ltf.strip()
        if not ltf:
            continue
        combos.append(ltf)
    return combos


def _normalise_timeframe_mix_argument(args: argparse.Namespace) -> List[str]:
    """Normalise the CLI ``timeframe_mix`` option into a canonical form."""

    raw = getattr(args, "timeframe_mix", None)
    mix_values = [token for token in _parse_ltf_choice_value(raw) if token]
    if len(mix_values) <= 1:
        if len(mix_values) == 1 and not getattr(args, "timeframe", None):
            args.timeframe = mix_values[0]
        if hasattr(args, "timeframe_mix"):
            setattr(args, "timeframe_mix", None)
        return []

    canonical = ",".join(mix_values)
    if set(mix_values) == set(MERGED_TIMEFRAME_COMPONENTS):
        canonical = MERGED_TIMEFRAME_LABEL
    setattr(args, "timeframe_mix", canonical)
    setattr(args, "timeframe", canonical)
    return mix_values


def _format_batch_value(
    template: Optional[str],
    base: Optional[str],
    suffix: str,
    context: Dict[str, object],
) -> Optional[str]:
    if template:
        try:
            return template.format(**context)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"Unknown placeholder '{missing}' in template {template!r}") from exc
    if base:
        return f"{base}_{suffix}" if suffix else base
    return suffix or None


def _resolve_study_storage(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
) -> Optional[Path]:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    _, symbol_slug, timeframe_slug, _ = _build_run_tag(datasets, params_cfg, None)
    return STUDY_ROOT / f"{symbol_slug}_{timeframe_slug}.db"


def _study_registry_dir(storage_path: Path) -> Path:
    """Return the directory that holds study registry metadata."""

    if storage_path.suffix:
        return storage_path.with_suffix("")
    return storage_path


def _study_registry_payload_path(storage_path: Path) -> Path:
    return _study_registry_dir(storage_path) / "storage.json"


def _load_study_registry(
    study_storage: Optional[Path],
) -> Tuple[Dict[str, object], Optional[Path]]:
    if study_storage is None:
        return {}, None

    pointer_path = _study_registry_payload_path(study_storage)
    if not pointer_path.exists():
        return {}, pointer_path

    return _load_json(pointer_path), pointer_path


def _apply_study_registry_defaults(
    search_cfg: Dict[str, object], study_storage: Optional[Path]
) -> None:
    """Apply stored storage settings when explicit configuration is missing."""

    payload, pointer_path = _load_study_registry(study_storage)
    if not payload:
        return

    backend = str(payload.get("backend") or "none").lower()
    if backend in {"", "none"}:
        return

    applied: List[str] = []

    if backend == "sqlite":
        stored_url = payload.get("storage_url") or payload.get("sqlite_url")
        if stored_url and not search_cfg.get("storage_url"):
            search_cfg["storage_url"] = stored_url
            applied.append("storage_url")
    else:
        env_key = payload.get("storage_url_env")
        if env_key and not search_cfg.get("storage_url_env"):
            search_cfg["storage_url_env"] = env_key
            applied.append("storage_url_env")
        stored_url = payload.get("storage_url")
        if stored_url and not search_cfg.get("storage_url"):
            search_cfg["storage_url"] = stored_url
            applied.append("storage_url")

    if applied and pointer_path is not None:
        LOGGER.info(
            "Study registry %s supplied defaults for: %s",
            pointer_path,
            ", ".join(applied),
        )


def _extract_primary_htf(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
) -> Optional[str]:
    if not HTF_ENABLED:
        return None
    raw = params_cfg.get("htf_timeframes")
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return str(raw[0])
    direct = params_cfg.get("htf_timeframe") or params_cfg.get("htf")
    if direct:
        return str(direct)
    if datasets and getattr(datasets[0], "htf_timeframe", None):
        return str(datasets[0].htf_timeframe)
    return None


def _default_study_name(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
    space_hash: Optional[str] = None,
) -> str:
    _, symbol_slug, timeframe_slug, _ = _build_run_tag(datasets, params_cfg, None)
    suffix = f"_{space_hash[:6]}" if space_hash else ""
    return f"{symbol_slug}_{timeframe_slug}{suffix}"


def _discover_bank_path(
    current_output: Path,
    tag_info: Dict[str, str],
    space_hash: str,
) -> Optional[Path]:
    root = current_output.parent
    if not root.exists():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p != current_output],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        bank_path = candidate / "bank.json"
        if not bank_path.exists():
            continue
        payload = _load_json(bank_path)
        metadata = payload.get("metadata", {})
        if payload.get("space_hash") != space_hash:
            continue
        if metadata.get("symbol") != tag_info.get("symbol"):
            continue
        if metadata.get("timeframe") != tag_info.get("timeframe"):
            continue
        metadata_htf = metadata.get("htf_timeframe") or "nohtf"
        target_htf = tag_info.get("htf_timeframe") or "nohtf"
        if metadata_htf != target_htf:
            continue
        return bank_path
    return None


def _load_seed_trials(
    bank_path: Optional[Path],
    space: Dict[str, object],
    space_hash: str,
    regime_label: Optional[str] = None,
    max_seeds: int = 20,
    *,
    basic_filter_enabled: bool = True,
) -> List[Dict[str, object]]:
    if bank_path is None:
        return []
    payload = _load_json(bank_path)
    if not payload or payload.get("space_hash") != space_hash:
        return []

    entries = payload.get("entries", [])
    if regime_label:
        filtered = [entry for entry in entries if entry.get("regime", {}).get("label") == regime_label]
        if filtered:
            entries = filtered

    seeds: List[Dict[str, object]] = []
    rng = np.random.default_rng()
    for entry in entries[:max_seeds]:
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        filtered_params = _filter_basic_factor_params(
            dict(params), enabled=basic_filter_enabled
        )
        filtered_params = _ensure_channel_params(filtered_params, space)
        if not filtered_params:
            continue
        filtered_params = _enforce_exit_guards(filtered_params, context="seed trial")
        seeds.append(filtered_params)
        mutated = mutate_around(
            filtered_params,
            space,
            scale=float(payload.get("mutation_scale", 0.1)),
            rng=rng,
        )
        mutated_filtered = _filter_basic_factor_params(
            mutated, enabled=basic_filter_enabled
        )
        mutated_filtered = _ensure_channel_params(mutated_filtered, space)
        if mutated_filtered:
            mutated_filtered = _enforce_exit_guards(mutated_filtered, context="seed mutation")
            seeds.append(mutated_filtered)
    return seeds


def _build_bank_payload(
    *,
    tag_info: Dict[str, str],
    space_hash: str,
    entries: List[Dict[str, object]],
    regime_summary,
    mutation_scale: float = 0.1,
) -> Dict[str, object]:
    payload_entries: List[Dict[str, object]] = []
    for entry in entries:
        regime_info = summarise_regime_performance(entry, regime_summary)
        payload_entries.append({**entry, "regime": regime_info})

    return {
        "created_at": _utcnow_isoformat(),
        "metadata": {
            "symbol": tag_info.get("symbol"),
            "timeframe": tag_info.get("timeframe"),
            "htf_timeframe": tag_info.get("htf_timeframe"),
            "tag": tag_info.get("tag"),
        },
        "space_hash": space_hash,
        "mutation_scale": mutation_scale,
        "entries": payload_entries,
    }


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _prompt_choice(label: str, choices: List[str], default: Optional[str] = None) -> Optional[str]:
    if not choices:
        return default
    while True:
        print(f"\n{label}:")
        for idx, value in enumerate(choices, start=1):
            marker = " (default)" if default == value else ""
            print(f"  {idx}. {value}{marker}")
        raw = input("Select option (press Enter for default): ").strip()
        if not raw:
            return default or (choices[0] if choices else None)
        if raw.isdigit():
            sel = int(raw)
            if 1 <= sel <= len(choices):
                return choices[sel - 1]
        print("Invalid selection. Please try again.")


def _prompt_bool(label: str, default: Optional[bool] = None) -> Optional[bool]:
    suffix = " [y/n]" if default is None else (" [Y/n]" if default else " [y/N]")
    while True:
        raw = input(f"{label}{suffix}: ").strip().lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if not raw:
            return default
        print("Please answer 'y' or 'n'.")


@dataclass(frozen=True)
class LTFPromptResult:
    timeframe: Optional[str] = None
    use_all: bool = False
    mix: Optional[List[str]] = None
    label: Optional[str] = None


@dataclass
class ExecutionBatch:
    args: argparse.Namespace
    context: Optional[Dict[str, object]] = None


@dataclass
class TimeframeExecutionPlan:
    mix_values: List[str]
    combos: List[str]
    batches: List[ExecutionBatch]


def _prompt_ltf_selection() -> LTFPromptResult:
    """Interactively ask the user which lower timeframe (LTF) to run."""

    options = {"1": "1m", "3": "3m", "5": "5m"}
    if not sys.stdin or not sys.stdin.isatty():
        LOGGER.info("표준 입력을 확인할 수 없어 %s 조합을 사용합니다.", MERGED_TIMEFRAME_LABEL)
        return LTFPromptResult(
            timeframe=MERGED_TIMEFRAME_LABEL,
            mix=list(MERGED_TIMEFRAME_COMPONENTS),
            label=MERGED_TIMEFRAME_LABEL,
        )

    while True:
        print("\n사용할 LTF를 선택하세요.")
        print("  1) 1m")
        print("  3) 3m")
        print("  5) 5m")
        print(f"  8) {MERGED_TIMEFRAME_LABEL} (세 타임프레임 모두 사용)")
        print(
            f"선택 (1/3/5/8, 10초 후 자동으로 {MERGED_TIMEFRAME_LABEL} 선택): ",
            end="",
            flush=True,
        )

        start_time = time.time()
        raw = ""
        while time.time() - start_time < 10:
            if msvcrt.kbhit():
                raw = sys.stdin.readline().strip()
                break
            time.sleep(0.1)

        if not raw:
            print(f"\n입력이 없어 {MERGED_TIMEFRAME_LABEL} 조합을 사용합니다.")
            return LTFPromptResult(
                timeframe=MERGED_TIMEFRAME_LABEL,
                mix=list(MERGED_TIMEFRAME_COMPONENTS),
                label=MERGED_TIMEFRAME_LABEL,
            )

        if raw in options:
            selection = options[raw]
            print(f"{raw}을(를) 선택했습니다.")
            return LTFPromptResult(selection)
        if raw == "8":
            print(f"{MERGED_TIMEFRAME_LABEL} 조합을 사용합니다.")
            return LTFPromptResult(
                timeframe=MERGED_TIMEFRAME_LABEL,
                mix=list(MERGED_TIMEFRAME_COMPONENTS),
                label=MERGED_TIMEFRAME_LABEL,
            )
        print("잘못 입력했습니다. 1, 3, 5, 8 중 하나를 선택하세요.")


def _apply_ltf_override_to_datasets(backtest_cfg: Dict[str, object], timeframe: str) -> None:
    entries = backtest_cfg.get("datasets")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry["ltf"] = [timeframe]
        entry["ltfs"] = [timeframe]
        entry["timeframes"] = [timeframe]


def _enforce_forced_timeframe_constraints(
    params_cfg: Dict[str, object], search_cfg: Dict[str, object], timeframe: str
) -> None:
    """Force all timeframe/LTF parameters to the specified value."""

    if not timeframe:
        return

    space_cfg = params_cfg.get("space")
    if isinstance(space_cfg, Mapping):
        updated_space: Dict[str, Dict[str, object]] = {}
        for name, spec in space_cfg.items():
            if not isinstance(spec, Mapping):
                if "timeframe" in str(name).lower() or str(name).lower() == "ltf":
                    continue
                updated_space[name] = spec  # type: ignore[assignment]
                continue

            needs_restriction = False
            key_lower = str(name).lower()
            if key_lower == "ltf" or "timeframe" in key_lower:
                needs_restriction = True

            new_spec = dict(spec)
            if needs_restriction:
                new_spec["type"] = "choice"
                new_spec["values"] = [timeframe]
                new_spec["choices"] = [timeframe]
                new_spec["options"] = [timeframe]
                new_spec["default"] = timeframe
                for obsolete_key in ("min", "max", "step", "log", "log_base"):
                    new_spec.pop(obsolete_key, None)
            updated_space[name] = new_spec

        params_cfg["space"] = updated_space

    diversify_cfg: Dict[str, object]
    raw_diversify = search_cfg.get("diversify")
    if isinstance(raw_diversify, Mapping):
        diversify_cfg = dict(raw_diversify)
    else:
        diversify_cfg = {}
    if diversify_cfg.get("timeframe_cycle"):
        diversify_cfg["timeframe_cycle"] = []
    else:
        diversify_cfg.setdefault("timeframe_cycle", [])
    if "htf_timeframe" in diversify_cfg:
        diversify_cfg["htf_timeframe"] = None
    if "htf_timeframes" in diversify_cfg and diversify_cfg["htf_timeframes"]:
        diversify_cfg["htf_timeframes"] = []
    search_cfg["diversify"] = diversify_cfg


def _prepare_timeframe_execution_plan(
    args: argparse.Namespace,
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    *,
    symbol_text: str,
) -> TimeframeExecutionPlan:
    """Assemble execution batches based on CLI timeframe prompts and mixes."""

    cleared_for_mix = False

    def _clear_timeframe_for_mix() -> None:
        nonlocal cleared_for_mix
        if cleared_for_mix:
            return
        if getattr(args, "timeframe", None):
            LOGGER.info(
                "--timeframe=%s 옵션을 해제합니다.",
                args.timeframe,
            )
            args.timeframe = None
        if getattr(args, "timeframe_grid", None):
            LOGGER.info(
                "--timeframe-grid=%s 옵션을 해제합니다.",
                args.timeframe_grid,
            )
            args.timeframe_grid = None
        cleared_for_mix = True

    mix_values = _normalise_timeframe_mix_argument(args)
    if mix_values:
        _clear_timeframe_for_mix()

    ltf_prompt = getattr(args, "_ltf_prompt_selection", None)
    should_prompt = getattr(args, "interactive", False)
    if (
        should_prompt
        and ltf_prompt is None
        and not getattr(args, "timeframe", None)
        and not getattr(args, "timeframe_grid", None)
        and not getattr(args, "timeframe_mix", None)
    ):
        ltf_prompt = _prompt_ltf_selection()
        setattr(args, "_ltf_prompt_selection", ltf_prompt)
    elif (
        not should_prompt
        and ltf_prompt is None
        and not getattr(args, "timeframe", None)
        and not getattr(args, "timeframe_grid", None)
        and not getattr(args, "timeframe_mix", None)
    ):
        candidates = _collect_ltf_candidates(backtest_cfg, params_cfg)
        default_timeframe = candidates[0] if candidates else "1m"
        if default_timeframe:
            LOGGER.info(
                "상호작용 모드가 아니어서 LTF %s 값을 사용합니다.",
                default_timeframe,
            )
            args.timeframe = default_timeframe

    if ltf_prompt:
        if ltf_prompt.mix:
            args.timeframe = None
            args.timeframe_grid = None
            args.timeframe_mix = ",".join(ltf_prompt.mix)
        elif ltf_prompt.use_all:
            args.timeframe = None
            if getattr(args, "timeframe_grid", None):
                LOGGER.info(
                    "기존 timeframe_grid=%s 설정을 유지합니다.",
                    args.timeframe_grid,
                )
            else:
                ltf_candidates = _collect_ltf_candidates(backtest_cfg, params_cfg)
                if not ltf_candidates:
                    ltf_candidates = ["1m", "3m", "5m"]
                args.timeframe_grid = ",".join(ltf_candidates)
                LOGGER.info(
                    "사용 가능한 LTF 전체를 실행합니다: %s",
                    ", ".join(ltf_candidates),
                )
        elif ltf_prompt.timeframe and not getattr(args, "timeframe", None):
            args.timeframe = ltf_prompt.timeframe

    mix_values = _normalise_timeframe_mix_argument(args)
    if mix_values:
        _clear_timeframe_for_mix()
        args.timeframe_mix = ",".join(mix_values)

    active_mix = [
        token
        for token in _parse_ltf_choice_value(getattr(args, "timeframe_mix", None))
        if token
    ]
    combos = [] if len(active_mix) > 1 else _parse_timeframe_grid(getattr(args, "timeframe_grid", None))

    if not combos:
        return TimeframeExecutionPlan(mix_values, combos, [ExecutionBatch(args=args)])

    symbol_slug = _slugify_symbol(symbol_text) if symbol_text else "study"
    total = len(combos)
    combo_summary = ", ".join(combos)
    LOGGER.info("총 %d개의 타임프레임 조합을 실행합니다: %s", total, combo_summary)

    batches: List[ExecutionBatch] = []
    for index, ltf in enumerate(combos, start=1):
        batch_args = argparse.Namespace(**vars(args))
        batch_args.timeframe = ltf
        suffix = _slugify_timeframe(ltf)
        context = {
            "index": index,
            "total": total,
            "ltf": ltf,
            "htf": None,
            "ltf_slug": _slugify_timeframe(ltf),
            "htf_slug": "",
            "symbol": symbol_text,
            "symbol_slug": symbol_slug,
            "suffix": suffix,
            "base_run_tag": getattr(args, "run_tag", None),
            "base_study_name": getattr(args, "study_name", None),
            "run_tag_template": getattr(args, "run_tag_template", None),
            "study_template": getattr(args, "study_template", None),
        }
        batches.append(ExecutionBatch(args=batch_args, context=context))

    return TimeframeExecutionPlan(mix_values, combos, batches)


def _coerce_bool_or_none(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan"}:
            return None
        if text in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "f", "0", "no", "n", "off"}:
            return False
    return None


def _normalise_channel_type(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "nan", "0"}:
        return None
    if lowered in {"bb", "bollinger", "bollingerband", "bollinger_band"}:
        return "BB"
    if lowered in {"kc", "keltner", "keltnerchannel", "keltner_channel"}:
        return "KC"
    return None


def _enforce_exit_guards(
    params: Mapping[str, object], *, context: Optional[str] = None
) -> Dict[str, object]:
    """Ensure at least one exit mechanism remains enabled.

    The strategy exposes two quick exit toggles: ``exitOpposite`` and
    ``useMomFade``. Historical runs assumed ``exitOpposite`` defaults to
    ``True`` while ``useMomFade`` defaults to ``False``. If a configuration
    disables both switches simultaneously we flip ``exitOpposite`` back to
    ``True`` to avoid leaving the strategy without a hard exit.

    Args:
        params: Parameter dictionary provided by the optimiser.
        context: Optional string describing where the guard is applied.

    Returns:
        Sanitised parameter dictionary with consistent exit toggles.
    """

    if not isinstance(params, Mapping):
        return {}

    patched = dict(params)
    exit_flag = _coerce_bool_or_none(patched.get("exitOpposite"))
    mom_fade_flag = _coerce_bool_or_none(patched.get("useMomFade"))

    if exit_flag is None:
        exit_flag = True
    if mom_fade_flag is None:
        mom_fade_flag = False

    def _positive_number(value: object) -> Optional[float]:
        if value in {None, ""}:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0:
            return None
        return float(numeric)

    fixed_stop_enabled = _positive_number(patched.get("fixedStopPct")) is not None
    sar_enabled = _coerce_bool_or_none(patched.get("useSarExit")) is True
    stop_channel_type = _normalise_channel_type(patched.get("stopChannelType"))
    stop_channel_mult = _positive_number(patched.get("stopChannelMult"))
    channel_stop_enabled = stop_channel_type in {"BB", "KC"} and stop_channel_mult is not None

    has_alternative_exit = (
        fixed_stop_enabled
        or sar_enabled
        or channel_stop_enabled
    )

    if exit_flag is False and mom_fade_flag is False:
        context_text = f" ({context})" if context else ""
        if has_alternative_exit:
            LOGGER.debug(
                "Both quick-exit toggles are False%s but alternative exits remain active; leaving as-is.",
                context_text,
            )
        else:
            exit_flag = True

    patched["exitOpposite"] = exit_flag
    patched["useMomFade"] = mom_fade_flag
    return patched


class _TimeframeCycler:
    """Rotate through predefined timeframe/HTF mixes for diversification."""

    def __init__(self, entries: Sequence[Mapping[str, object]]):
        self._entries: List[Dict[str, object]] = []
        for entry in entries:
            timeframe = str(entry.get("timeframe") or entry.get("tf") or "").strip()
            if not timeframe:
                continue
            htf_value = entry.get("htf") or entry.get("htf_timeframe")
            htf = str(htf_value).strip() if htf_value not in {None, ""} else None
            repeat_raw = entry.get("repeat") or entry.get("count") or entry.get("times") or 1
            try:
                repeat = max(1, int(repeat_raw))
            except (TypeError, ValueError):
                repeat = 1
            self._entries.append({
                "timeframe": timeframe,
                "htf_timeframe": htf,
                "repeat": repeat,
            })
        self._index = 0
        self._remaining = self._entries[0]["repeat"] if self._entries else 0

    def next(self) -> Dict[str, Optional[str]]:
        if not self._entries:
            return {}
        current = self._entries[self._index]
        payload = {
            "timeframe": current.get("timeframe"),
            "htf_timeframe": current.get("htf_timeframe"),
        }
        self._remaining -= 1
        if self._remaining <= 0:
            self._index = (self._index + 1) % len(self._entries)
            self._remaining = self._entries[self._index]["repeat"]
        return payload

    def empty(self) -> bool:
        return not self._entries


class TrialDiversifier:
    """Promote exploration by mutating and rotating sampled trial parameters."""

    def __init__(
        self,
        space: Dict[str, Dict[str, object]],
        config: Mapping[str, object],
        *,
        forced_params: Optional[Mapping[str, object]] = None,
        param_order: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.space = space
        self.enabled = True
        self.forced_params = dict(forced_params or {})
        self.param_order = list(param_order) if param_order else list(space.keys())
        self.similarity_threshold = float(config.get("similarity_threshold", 0.9))
        self.float_tolerance = float(config.get("float_tolerance", 1e-3))
        self.max_consecutive = max(2, int(config.get("max_consecutive", 6)))
        cooldown_source = config.get("cooldown", self.max_consecutive)
        try:
            cooldown_default = max(0, int(cooldown_source))
        except (TypeError, ValueError):
            cooldown_default = self.max_consecutive
        self.cooldown = cooldown_default
        self.cooldown_remaining = 0
        self.jump_trials = max(1, int(config.get("jump_trials", 2)))
        self.jump_scale = float(config.get("jump_scale", 0.75))
        try:
            history_bias = float(config.get("history_bias", 0.5))
        except (TypeError, ValueError):
            history_bias = 0.5
        self.history_bias = min(max(history_bias, 0.0), 1.0)
        history_source = config.get("history", 20)
        try:
            history_limit = int(history_source)
        except (TypeError, ValueError):
            history_limit = 20
        self.history_limit = max(self.max_consecutive + 1, history_limit)
        try:
            length_refresh_prob = float(config.get("length_refresh_prob", 0.75))
        except (TypeError, ValueError):
            length_refresh_prob = 0.75
        self.length_refresh_prob = min(max(length_refresh_prob, 0.0), 1.0)
        ignored = config.get("ignore_keys", [])
        self.ignored_keys = {str(key) for key in ignored if str(key)}
        rng_seed = seed if seed is not None else config.get("seed")
        try:
            self.rng = np.random.default_rng(None if rng_seed in {None, ""} else int(rng_seed))
        except (TypeError, ValueError):
            self.rng = np.random.default_rng()
        timeframe_entries: List[Mapping[str, object]] = []
        raw_cycle = config.get("timeframe_cycle")
        if isinstance(raw_cycle, Mapping):
            entries: List[Mapping[str, object]] = []
            for key, repeat in raw_cycle.items():
                entries.append({"timeframe": str(key), "repeat": repeat})
            timeframe_entries = entries
        elif isinstance(raw_cycle, Sequence) and not isinstance(raw_cycle, (str, bytes, bytearray)):
            processed: List[Mapping[str, object]] = []
            for entry in raw_cycle:
                if isinstance(entry, Mapping):
                    processed.append(dict(entry))
                elif isinstance(entry, str):
                    processed.append({"timeframe": str(entry)})
            timeframe_entries = processed
        else:
            timeframe_entries = []
        self.timeframe_cycler = _TimeframeCycler(timeframe_entries)
        self.history: deque[Dict[str, object]] = deque(maxlen=self.history_limit)
        self.last_params: Optional[Dict[str, object]] = None
        self.similar_streak = 0
        length_keys_cfg = config.get("length_keys")
        if isinstance(length_keys_cfg, Sequence) and not isinstance(length_keys_cfg, (str, bytes, bytearray)):
            self.length_keys = [str(key) for key in length_keys_cfg]
        else:
            self.length_keys = [
                name
                for name, spec in space.items()
                if isinstance(spec, Mapping)
                and spec.get("type") == "int"
                and "len" in name.lower()
                and name not in self.ignored_keys
            ]
        self.total_enqueued = 0

    def __call__(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        if not self.enabled or trial.state != TrialState.COMPLETE:
            return
        params = dict(trial.params)
        for ignored_key in self.ignored_keys:
            params.pop(ignored_key, None)
        self.history.append(params)
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.last_params = params
            return
        if self.last_params is None:
            self.last_params = params
            self.similar_streak = 1
            return
        similarity = self._similarity(self.last_params, params)
        if similarity >= self.similarity_threshold:
            self.similar_streak += 1
        else:
            self.similar_streak = 1
        self.last_params = params
        if self.similar_streak < self.max_consecutive:
            return
        self.similar_streak = 0
        self.cooldown_remaining = self.cooldown
        injections = self._plan_injections()
        if not injections:
            return
        accepted = 0
        for candidate in injections:
            enriched = dict(candidate)
            enriched.update(self.forced_params)
            enriched = _enforce_exit_guards(enriched, context="diversifier injection")
            try:
                study.enqueue_trial(enriched, skip_if_exists=True)
            except Exception as exc:
                LOGGER.debug("Diversifier enqueue skipped (%s): %s", enriched, exc)
                continue
            accepted += 1
        if accepted:
            self.total_enqueued += accepted
            LOGGER.info("Diversifier가 %d개의 시드 트라이얼을 대기열에 추가했습니다.", accepted)

    def _similarity(self, base: Mapping[str, object], other: Mapping[str, object]) -> float:
        keys = [key for key in self.param_order if key not in self.ignored_keys]
        if not keys:
            return 0.0
        matches = 0
        total = 0
        for key in keys:
            a = base.get(key)
            b = other.get(key)
            if a is None and b is None:
                continue
            total += 1
            if self._values_equal(a, b):
                matches += 1
        if not total:
            return 0.0
        return matches / total

    def _values_equal(self, a: object, b: object) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= self.float_tolerance
        return a == b

    def _plan_injections(self) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        history_list = list(self.history)
        for index in range(self.jump_trials):
            use_history = (
                index > 0
                and bool(history_list)
                and self.rng.random() < self.history_bias
            )
            if use_history:
                base = dict(self.rng.choice(history_list))
            else:
                base = random_parameters(self.space, rng=self.rng)
            if self.length_keys and (not use_history or self.rng.random() < self.length_refresh_prob):
                reference = random_parameters(self.space, rng=self.rng)
                for key in self.length_keys:
                    if key in reference:
                        base[key] = reference[key]
            scale = self.jump_scale
            if scale > 0:
                base = mutate_around(base, self.space, scale=scale, rng=self.rng)
            base = _enforce_exit_guards(base, context="diversifier ?")
            base = self._apply_timeframe_cycle(base)
            candidates.append(base)
        return candidates

    def _apply_timeframe_cycle(self, params: Dict[str, object]) -> Dict[str, object]:
        if self.timeframe_cycler.empty():
            return params
        override = self.timeframe_cycler.next()
        if not override:
            return params
        updated = dict(params)
        timeframe = override.get("timeframe")
        htf_value = override.get("htf_timeframe")
        if timeframe is not None:
            for key in ("timeframe", "ltf"):
                if key in self.space:
                    updated[key] = timeframe
                    break
        if htf_value is not None:
            for key in ("htf", "htf_timeframe"):
                if key in self.space:
                    updated[key] = htf_value
                    break
        return updated


class LLMCandidateRefresher:
    """Periodically refresh Optuna candidate seeds using LLM insights."""

    def __init__(
        self,
        space: Mapping[str, Mapping[str, object]],
        config: Mapping[str, object],
        *,
        forced_params: Optional[Mapping[str, object]] = None,
        use_basic_factors: bool = False,
    ) -> None:
        self.space = space
        self.config = dict(config)
        self.forced_params = dict(forced_params or {})
        self.use_basic_factors = bool(use_basic_factors)

        interval_source = self.config.get("refresh_trials")
        if interval_source in {None, ""}:
            interval_source = self.config.get("refresh_interval")
        try:
            refresh_interval = int(interval_source) if interval_source not in {None, ""} else 0
        except (TypeError, ValueError):
            refresh_interval = 0

        count_source = self.config.get("refresh_count")
        if count_source in {None, ""}:
            count_source = self.config.get("count")
        try:
            refresh_count = int(count_source) if count_source not in {None, ""} else 0
        except (TypeError, ValueError):
            refresh_count = 0

        self.refresh_interval = max(1, refresh_interval) if refresh_interval > 0 else 0
        self.refresh_count = max(1, refresh_count) if refresh_count > 0 else 0

        enabled_flag = _coerce_bool_or_none(self.config.get("enabled"))
        base_enabled = bool(self.refresh_interval and self.refresh_count)
        if enabled_flag is None:
            self.enabled = base_enabled and bool(self.config.get("enabled", True))
        else:
            self.enabled = base_enabled and enabled_flag

        try:
            base_top_n = int(self.config.get("top_n", 10))
        except (TypeError, ValueError):
            base_top_n = 10

        try:
            base_bottom_n = int(self.config.get("bottom_n", base_top_n))
        except (TypeError, ValueError):
            base_bottom_n = base_top_n
        self.base_top_n = max(base_top_n, 1)
        self.base_bottom_n = max(base_bottom_n, 1)

        self._lock = Lock()
        self._completed_since_refresh = 0
        self._refresh_requests = 0
        self.total_enqueued = 0
        self.total_refreshes = 0
        self.collected_insights: List[str] = []
        self.last_refresh_trial: Optional[int] = None

    def __call__(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        if not self.enabled or trial.state != TrialState.COMPLETE:
            return

        should_refresh = False
        refresh_request_index = 0
        with self._lock:
            self._completed_since_refresh += 1
            if self._completed_since_refresh >= self.refresh_interval:
                self._completed_since_refresh = 0
                self._refresh_requests += 1
                refresh_request_index = self._refresh_requests
                should_refresh = True

        if not should_refresh:
            return

        try:
            request_position = max(refresh_request_index, 1)
            growth_steps = request_position - 1
            dynamic_config = dict(self.config)
            dynamic_config["top_n"] = self.base_top_n + growth_steps * 10
            dynamic_config["bottom_n"] = self.base_bottom_n + growth_steps * 10
            generator = _resolve_llm_generator()
            llm_bundle: LLMSuggestions = generator(self.space, study.trials, dynamic_config)
        except Exception as exc:
            LOGGER.debug("LLM ? ? ?: %s", exc)
            return

        insights = [
            entry.strip()
            for entry in llm_bundle.insights
            if isinstance(entry, str) and entry.strip()
        ]
        if insights:
            with self._lock:
                for insight in insights:
                    if insight not in self.collected_insights:
                        self.collected_insights.append(insight)

        accepted = 0
        for candidate in llm_bundle.candidates[: self.refresh_count]:
            trial_params = _filter_basic_factor_params(
                dict(candidate), enabled=self.use_basic_factors
            )
            trial_params = _ensure_channel_params(trial_params, self.space)
            if not trial_params:
                continue
            trial_params.update(self.forced_params)
            trial_params = _enforce_exit_guards(trial_params, context="LLM refresh")
            try:
                study.enqueue_trial(trial_params, skip_if_exists=True)
            except Exception as exc:
                LOGGER.debug("Gemini 후보 %s 대기열 추가 실패: %s", candidate, exc)
                continue
            accepted += 1

        if accepted:
            self.total_enqueued += accepted
            self.total_refreshes += 1
            self.last_refresh_trial = trial.number
            LOGGER.info(
                "Gemini가 %d개의 후보를 대기열에 추가했습니다 (trial=%d).",
                accepted,
                trial.number,
            )

def _build_trial_diversifier(
    space: Dict[str, Dict[str, object]],
    search_cfg: Mapping[str, object],
    *,
    forced_params: Optional[Mapping[str, object]] = None,
    param_order: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Optional[TrialDiversifier]:
    raw_cfg = search_cfg.get("diversify")
    if isinstance(raw_cfg, Mapping):
        config = dict(raw_cfg)
    elif _coerce_bool_or_none(raw_cfg) is True:
        config = {}
    else:
        return None
    enabled_flag = _coerce_bool_or_none(config.get("enabled"))
    if enabled_flag is False:
        return None
    LOGGER.info("Trial diversifier를 활성화합니다.")
    return TrialDiversifier(
        space,
        config,
        forced_params=forced_params,
        param_order=param_order,
        seed=seed,
    )


def _build_llm_refresher(
    space: Mapping[str, Mapping[str, object]],
    llm_cfg: Mapping[str, object],
    *,
    forced_params: Optional[Mapping[str, object]] = None,
    use_basic_factors: bool = False,
) -> Optional[LLMCandidateRefresher]:
    if not isinstance(llm_cfg, Mapping):
        return None
    refresher = LLMCandidateRefresher(
        space,
        llm_cfg,
        forced_params=forced_params,
        use_basic_factors=use_basic_factors,
    )
    if not refresher.enabled:
        return None
    LOGGER.info(
        "Gemini 후보 새로고침 주기: %d 트라이얼마다 %d개 추가",
        refresher.refresh_interval,
        refresher.refresh_count,
    )
    return refresher


def _collect_tokens(items: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for item in items:
        if not item:
            continue
        for token in item.split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _collect_ltf_candidates(*configs: Mapping[str, object]) -> List[str]:
    seen: "OrderedDict[str, None]" = OrderedDict()

    def _register(value: object) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        seen.setdefault(text, None)

    for cfg in configs:
        if not isinstance(cfg, Mapping):
            continue
        datasets = cfg.get("datasets")
        if isinstance(datasets, list):
            for entry in datasets:
                if not isinstance(entry, Mapping):
                    continue
                for key in ("ltf", "ltfs", "timeframes"):
                    raw = entry.get(key)
                    if isinstance(raw, (list, tuple)):
                        for item in raw:
                            _register(item)
                    elif raw is not None:
                        _register(raw)

        timeframes = cfg.get("timeframes")
        if isinstance(timeframes, (list, tuple)):
            for tf in timeframes:
                _register(tf)
        elif timeframes is not None:
            _register(timeframes)

    return list(seen.keys())


def _ensure_dict(root: Dict[str, object], key: str) -> Dict[str, object]:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


@dataclass(frozen=True)
class DatasetCacheInfo:
    root: Path
    futures: bool = False

    def serialise(self) -> Dict[str, object]:
        return {"root": str(self.root), "futures": self.futures}


@dataclass
class DatasetSpec:
    symbol: str
    timeframe: str
    start: str
    end: str
    df: pd.DataFrame
    htf: Optional[pd.DataFrame]
    htf_timeframe: Optional[str] = None
    source_symbol: Optional[str] = None
    cache_info: Optional[DatasetCacheInfo] = None
    total_volume: Optional[float] = None
    chart_timeframe: Optional[str] = None
    chart_df: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        parts = [self.symbol, self.timeframe]
        if self.htf_timeframe:
            parts.append(f"htf{self.htf_timeframe}")
        parts.extend([self.start, self.end])
        return "_".join(parts)

    @property
    def meta(self) -> Dict[str, str]:
        return {
            "symbol": self.symbol,
            "source_symbol": self.source_symbol or self.symbol,
            "timeframe": self.timeframe,
            "from": self.start,
            "to": self.end,
            "htf_timeframe": self.htf_timeframe or "",
            "chart_timeframe": self.chart_timeframe or "",
        }


_PROCESS_DATASET_REGISTRY: Dict[str, Dict[str, object]] = {}
_PROCESS_DATASET_OBJECTS: Dict[str, DatasetSpec] = {}
_PROCESS_DATASET_CACHES: Dict[Tuple[str, bool], DataCache] = {}
_PROCESS_DATASET_LOCK = Lock()


def _register_process_datasets(handles: Sequence[Dict[str, object]]) -> None:
    global _PROCESS_DATASET_REGISTRY, _PROCESS_DATASET_OBJECTS, _PROCESS_DATASET_CACHES
    _PROCESS_DATASET_REGISTRY = {entry["id"]: entry for entry in handles}
    _PROCESS_DATASET_OBJECTS = {}
    _PROCESS_DATASET_CACHES = {}


def _process_pool_initializer(handles: Sequence[Dict[str, object]]) -> None:
    _register_process_datasets(handles)


def _resolve_process_cache(root: str, futures: bool) -> DataCache:
    key = (root, futures)
    cache = _PROCESS_DATASET_CACHES.get(key)
    if cache is not None:
        return cache

    with _PROCESS_DATASET_LOCK:
        cache = _PROCESS_DATASET_CACHES.get(key)
        if cache is not None:
            return cache
        cache = DataCache(Path(root), futures=futures)
        _PROCESS_DATASET_CACHES[key] = cache
        return cache


def _load_process_dataset(dataset_id: str) -> DatasetSpec:
    handle = _PROCESS_DATASET_REGISTRY.get(dataset_id)
    if handle is None:
        raise KeyError(f"Unknown dataset identifier: {dataset_id}")

    cache_root = handle.get("cache_root")
    if not cache_root:
        raise RuntimeError("Process executor dataset metadata is missing cache_root.")
    futures_flag = bool(handle.get("cache_futures", False))
    cache = _resolve_process_cache(str(cache_root), futures_flag)

    source_symbol = str(handle.get("source_symbol"))
    timeframe = str(handle.get("timeframe"))
    start = str(handle.get("start"))
    end = str(handle.get("end"))
    df = cache.get(source_symbol, timeframe, start, end)

    htf_timeframe = handle.get("htf_timeframe") or None
    htf_df: Optional[pd.DataFrame] = None
    if htf_timeframe:
        htf_df = cache.get(source_symbol, str(htf_timeframe), start, end)
        chart_tf = timeframe
        chart_df = df
    else:
        chart_tf = handle.get("chart_timeframe") or None
        chart_df = handle.get("chart_df") or None

    dataset = DatasetSpec(
        symbol=str(handle.get("symbol")),
        timeframe=timeframe,
        start=start,
        end=end,
        df=df,
        htf=htf_df,
        htf_timeframe=htf_timeframe,
        source_symbol=source_symbol,
        cache_info=DatasetCacheInfo(Path(str(cache_root)), futures=futures_flag),
        total_volume=_compute_total_volume(df),
        chart_timeframe=str(chart_tf) if chart_tf else None,
        chart_df=chart_df,
    )
    _PROCESS_DATASET_OBJECTS[dataset_id] = dataset
    return dataset


def _resolve_dataset_reference(dataset_ref: object) -> DatasetSpec:
    if isinstance(dataset_ref, DatasetSpec):
        return dataset_ref
    if isinstance(dataset_ref, str):
        cached = _PROCESS_DATASET_OBJECTS.get(dataset_ref)
        if cached is not None:
            return cached
        with _PROCESS_DATASET_LOCK:
            cached = _PROCESS_DATASET_OBJECTS.get(dataset_ref)
            if cached is not None:
                return cached
            return _load_process_dataset(dataset_ref)
    raise TypeError(f"Unsupported dataset_ref type: {type(dataset_ref)!r}")


def _serialise_datasets_for_process(datasets: Sequence[DatasetSpec]) -> List[Dict[str, object]]:
    handles: List[Dict[str, object]] = []
    for dataset in datasets:
        if not dataset.cache_info:
            raise RuntimeError(
                "Process executor requires DatasetSpec.cache_info entries."
            )
        cache_data = dataset.cache_info.serialise()
        handles.append(
            {
                "id": dataset.name,
                "symbol": dataset.symbol,
                "source_symbol": dataset.source_symbol or dataset.symbol,
                "timeframe": dataset.timeframe,
                "start": dataset.start,
                "end": dataset.end,
                "htf_timeframe": dataset.htf_timeframe,
                "cache_root": cache_data["root"],
                "cache_futures": cache_data["futures"],
                "chart_timeframe": dataset.chart_timeframe or dataset.timeframe,
            }
        )
    return handles


def _compute_total_volume(frame: Optional[pd.DataFrame]) -> float:
    if frame is None or "volume" not in frame.columns:
        return 0.0

    volume_series = pd.to_numeric(frame["volume"], errors="coerce")
    total = float(np.nansum(volume_series.to_numpy(dtype=float)))
    if not np.isfinite(total):
        return 0.0
    return total


def _dataset_total_volume(dataset: "DatasetSpec") -> float:
    """Return a finite total volume for the given dataset."""

    if dataset.total_volume is not None:
        return dataset.total_volume

    total = _compute_total_volume(dataset.df)
    dataset.total_volume = total
    return total


def _has_sufficient_volume(dataset: "DatasetSpec", threshold: float) -> Tuple[bool, float]:
    """Return whether the dataset meets the minimum volume requirement."""

    total = _dataset_total_volume(dataset)
    return total >= threshold, total


def _normalise_timeframe_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_htf_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na", "off", "0"}:
        return None
    return text


def _parse_ltf_choice_value(value: Optional[object]) -> List[str]:
    """Parse CLI or config values that encode mixed LTF selections."""

    if value is None:
        return []

    tokens: List[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == MERGED_TIMEFRAME_LABEL.lower():
            return list(MERGED_TIMEFRAME_COMPONENTS)
        cleaned = value.replace("\n", ",").replace(";", ",")
        for token in cleaned.split(","):
            candidate = token.strip()
            if candidate:
                if candidate.lower() == MERGED_TIMEFRAME_LABEL.lower():
                    tokens.extend(MERGED_TIMEFRAME_COMPONENTS)
                else:
                    tokens.append(candidate)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.extend(_parse_ltf_choice_value(item))
    else:
        text = str(value).strip()
        if text:
            tokens.append(text)

    deduped: "OrderedDict[str, None]" = OrderedDict()
    for token in tokens:
        deduped.setdefault(token, None)
    return list(deduped.keys())


def _group_datasets(
    datasets: Sequence[DatasetSpec],
) -> Tuple[Dict[Tuple[str, Optional[str]], List[DatasetSpec]], Dict[str, List[DatasetSpec]], Tuple[str, Optional[str]]]:
    groups: Dict[Tuple[str, Optional[str]], List[DatasetSpec]] = {}
    timeframe_groups: Dict[str, List[DatasetSpec]] = {}
    for dataset in datasets:
        key = (dataset.timeframe, dataset.htf_timeframe or None)
        groups.setdefault(key, []).append(dataset)
        timeframe_groups.setdefault(dataset.timeframe, []).append(dataset)

    if not groups:
        raise RuntimeError("No datasets available for optimisation")

    default_key = next(iter(groups))
    return groups, timeframe_groups, default_key


def _configure_parallel_workers(
    search_cfg: Dict[str, object],
    dataset_groups: Mapping[Tuple[str, Optional[str]], List[DatasetSpec]],
    *,
    available_cpu: int,
    n_jobs: int,
) -> Tuple[int, int, str, Optional[str]]:
    dataset_executor = str(search_cfg.get("dataset_executor", "thread") or "thread").lower()
    if dataset_executor not in {"thread", "process"}:
        dataset_executor = "thread"

    dataset_start_method_raw = search_cfg.get("dataset_start_method")
    dataset_start_method = (
        str(dataset_start_method_raw).lower() if dataset_start_method_raw else None
    )

    max_parallel_datasets = max((len(group) for group in dataset_groups.values()), default=1)
    auto_dataset_jobs = min(max_parallel_datasets, max(1, available_cpu))

    legacy_dataset_jobs = search_cfg.get("dataset_n_jobs")
    if legacy_dataset_jobs is not None and "dataset_jobs" not in search_cfg:
        search_cfg["dataset_jobs"] = legacy_dataset_jobs

    search_cfg.setdefault("dataset_jobs", auto_dataset_jobs)

    raw_dataset_jobs = search_cfg.get("dataset_jobs")
    try:
        dataset_jobs = max(1, int(raw_dataset_jobs))
    except (TypeError, ValueError):
        dataset_jobs = auto_dataset_jobs

    dataset_jobs = min(dataset_jobs, max(1, available_cpu))

    dataset_parallel_capable = max_parallel_datasets > 1
    if not dataset_parallel_capable:
        if dataset_jobs != 1:
            LOGGER.info(
                "데이터셋 병렬화가 불가능하므로 dataset_jobs %d → 1 로 강제합니다.",
                dataset_jobs,
            )
        dataset_jobs = 1
    else:
        if dataset_jobs <= 1 and auto_dataset_jobs > 1:
            dataset_jobs = auto_dataset_jobs
            LOGGER.info(
                "데이터셋 worker를 %d개로 자동 조정합니다 (사용 가능 CPU=%d, 그룹=%d).",
                dataset_jobs,
                available_cpu,
                max_parallel_datasets,
            )
        elif dataset_jobs > auto_dataset_jobs:
            LOGGER.info(
                "요청한 dataset_jobs=%d 가 자동 한도 %d 를 초과하여 상한으로 조정합니다 (그룹=%d).",
                dataset_jobs,
                auto_dataset_jobs,
                max_parallel_datasets,
            )
            dataset_jobs = auto_dataset_jobs

    dataset_jobs = max(1, dataset_jobs)
    search_cfg["dataset_jobs"] = dataset_jobs

    if dataset_parallel_capable and dataset_jobs > 1:
        optuna_budget = max(1, available_cpu // dataset_jobs)
        if optuna_budget < n_jobs:
            LOGGER.info(
                "데이터셋 worker %d개 때문에 Optuna worker를 %d개로 제한합니다 (이전=%d).",
                dataset_jobs,
                optuna_budget,
                n_jobs,
            )
            n_jobs = optuna_budget
            search_cfg["n_jobs"] = n_jobs
        LOGGER.info(
            "데이터셋 worker %d개를 %s 실행기로 사용합니다.",
            dataset_jobs,
            dataset_executor,
        )
        if dataset_executor == "process" and dataset_start_method:
            LOGGER.info("process start method=%s", dataset_start_method)
    elif not dataset_parallel_capable:
        LOGGER.info(
            "데이터셋 병렬화가 없어 Optuna worker %d개를 유지합니다.",
            n_jobs,
        )
        dataset_jobs = 1
    else:
        LOGGER.info(
            "dataset_jobs=1 상태이므로 Optuna worker %d개를 유지합니다.",
            n_jobs,
        )
        dataset_jobs = 1

    LOGGER.info(
        "최종 병렬 전략: Optuna worker=%d, dataset worker=%d (%s executor)",
        n_jobs,
        dataset_jobs,
        dataset_executor,
    )

    search_cfg["dataset_jobs"] = dataset_jobs

    return n_jobs, dataset_jobs, dataset_executor, dataset_start_method


def _select_datasets_for_params(
    params_cfg: Dict[str, object],
    dataset_groups: Dict[Tuple[str, Optional[str]], List[DatasetSpec]],
    timeframe_groups: Dict[str, List[DatasetSpec]],
    default_key: Tuple[str, Optional[str]],
    params: Dict[str, object],
) -> Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]:
    def _match(tf: str, htf: Optional[str]) -> Optional[Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]]:
        tf_lower = tf.lower()
        htf_lower = (htf or "").lower()
        for key, group in dataset_groups.items():
            key_tf, key_htf = key
            if key_tf.lower() != tf_lower:
                continue
            key_htf_lower = (key_htf or "").lower()
            if key_htf_lower == htf_lower:
                return key, group
        return None

    timeframe_value = (
        _normalise_timeframe_value(params.get("timeframe"))
        or _normalise_timeframe_value(params_cfg.get("timeframe"))
    )

    htf_value = None

    multi_timeframes: List[str] = []
    components_source = params.get("timeframe_components") or params_cfg.get("timeframe_components")
    if isinstance(components_source, (list, tuple, set)):
        multi_timeframes = [str(token).strip() for token in components_source if str(token).strip()]
    if not multi_timeframes and timeframe_value:
        preset = MERGED_TIMEFRAME_MAP.get(timeframe_value)
        if preset:
            multi_timeframes = list(preset)
        elif "," in timeframe_value:
            multi_timeframes = [token.strip() for token in timeframe_value.split(",") if token.strip()]

    if timeframe_value is None:
        if multi_timeframes:
            timeframe_value = multi_timeframes[0]
        else:
            timeframe_value = default_key[0]

    def _resolve_single(tf: Optional[str], htf: Optional[str]) -> Optional[Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]]:
        if not tf:
            return None
        selected_local = _match(tf, htf)
        if selected_local is None and htf is not None:
            selected_local = _match(tf, None)
        if selected_local is None:
            for key, group in dataset_groups.items():
                if key[0].lower() == tf.lower():
                    selected_local = (key, group)
                    break
        if selected_local is None:
            for tf_candidate, group in timeframe_groups.items():
                if tf_candidate.lower() == tf.lower():
                    key = (group[0].timeframe, group[0].htf_timeframe or None)
                    selected_local = (key, group)
                    break
        return selected_local

    if len(multi_timeframes) > 1:
        aggregated: "OrderedDict[str, DatasetSpec]" = OrderedDict()
        combined_key: Optional[Tuple[str, Optional[str]]] = None
        for tf in multi_timeframes:
            resolved = _resolve_single(tf, htf_value)
            if resolved is None:
                continue
            key_candidate, group = resolved
            if combined_key is None:
                label = timeframe_value or ",".join(multi_timeframes)
                base_tokens = {token.lower() for token in multi_timeframes}
                if label and label.lower() in base_tokens:
                    label = ",".join(multi_timeframes)
                combined_key = (label, key_candidate[1])
            for dataset in group:
                aggregated.setdefault(dataset.name, dataset)
        if aggregated:
            label = timeframe_value or ",".join(multi_timeframes)
            base_tokens = {token.lower() for token in multi_timeframes}
            if label and label.lower() in base_tokens:
                label = ",".join(multi_timeframes)
            return combined_key or (label, htf_value), list(aggregated.values())

    selected = _resolve_single(timeframe_value, htf_value)

    if selected is None:
        selected = (default_key, dataset_groups[default_key])

    return selected


def _pick_primary_dataset(datasets: Sequence[DatasetSpec]) -> DatasetSpec:
    return max(datasets, key=lambda item: len(item.df))


    """Normalise a symbol entry to a display name and a Binance fetch symbol."""

    if isinstance(entry, dict):
        alias = entry.get("alias") or entry.get("name") or entry.get("symbol") or entry.get("id") or ""
        resolved = entry.get("symbol") or entry.get("id") or alias
        alias = str(alias) if alias else str(resolved)
        resolved = str(resolved) if resolved else alias
    else:
        alias = str(entry)
        resolved = alias

    resolved = alias_map.get(alias, alias_map.get(resolved, resolved))
    if not alias:
        alias = resolved
    if not resolved:
        resolved = alias
    return alias, resolved


def _normalise_periods(
    periods_cfg: Optional[Iterable[Dict[str, object]]],
    base_period: Dict[str, object],
) -> List[Dict[str, str]]:
    periods: List[Dict[str, str]] = []
    if periods_cfg:
        for idx, raw in enumerate(periods_cfg):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Period entry #{idx + 1} must be a mapping with 'from'/'to' keys, got {type(raw).__name__}."
                )
            start = raw.get("from")
            end = raw.get("to")
            if not start or not end:
                raise ValueError(
                    f"Period entry #{idx + 1} is missing required 'from'/'to' values: {raw}."
                )
            periods.append({"from": str(start), "to": str(end)})

    if not periods:
        start = base_period.get("from") if isinstance(base_period, dict) else None
        end = base_period.get("to") if isinstance(base_period, dict) else None
        if start and end:
            periods.append({"from": str(start), "to": str(end)})

    return periods


def prepare_datasets(
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    data_dir: Path,
) -> List[DatasetSpec]:
    def _collect_forced_timeframes() -> List[str]:
        sources: List[object] = []
        raw_timeframe = params_cfg.get("timeframe")
        if raw_timeframe is not None:
            sources.append(raw_timeframe)
        components = params_cfg.get("timeframe_components")
        if components is not None:
            sources.append(components)
        pool_values = params_cfg.get("timeframe_pool")
        if pool_values is not None:
            sources.append(pool_values)
        overrides_cfg = params_cfg.get("overrides")
        if isinstance(overrides_cfg, Mapping) and overrides_cfg.get("timeframe") is not None:
            sources.append(overrides_cfg.get("timeframe"))

        ordered: "OrderedDict[str, None]" = OrderedDict()
        for source in sources:
            for token in _parse_ltf_choice_value(source):
                normalised = _normalise_timeframe_value(token)
                if normalised:
                    ordered.setdefault(normalised, None)
        return list(ordered.keys())

    data_cfg = backtest_cfg.get("data") if isinstance(backtest_cfg.get("data"), dict) else {}
    cache_root = Path(data_dir).expanduser()
    futures_flag = bool(backtest_cfg.get("futures", False))
    if data_cfg:
        market_text = str(data_cfg.get("market", "")).lower()
        if market_text == "futures":
            futures_flag = True
        elif market_text == "spot":
            futures_flag = False
        if "futures" in data_cfg:
            futures_flag = bool(data_cfg.get("futures"))
        cache_override = data_cfg.get("cache_dir")
        if cache_override:
            cache_root = Path(cache_override).expanduser()
    cache = DataCache(cache_root, futures=futures_flag)
    cache_info = DatasetCacheInfo(root=cache_root, futures=futures_flag)

    base_symbol = str(params_cfg.get("symbol")) if params_cfg.get("symbol") else ""
    base_timeframe = str(params_cfg.get("timeframe")) if params_cfg.get("timeframe") else ""
    base_period = params_cfg.get("backtest", {}) or {}

    alias_map: Dict[str, str] = {}
    for source in (backtest_cfg.get("symbol_aliases"), params_cfg.get("symbol_aliases")):
        if isinstance(source, dict):
            for key, value in source.items():
                if key and value:
                    alias_map[str(key)] = str(value)

    def _to_list(value: Optional[object]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v]
        text = str(value)
        return [text] if text else []

    dataset_entries = backtest_cfg.get("datasets")
    if isinstance(dataset_entries, list) and dataset_entries:
        datasets: List[DatasetSpec] = []
        forced_timeframes = _collect_forced_timeframes()
        forced_lookup = {token.lower(): token for token in forced_timeframes}
        for entry in dataset_entries:
            if not isinstance(entry, dict):
                continue
            symbol_value = (
                entry.get("symbol")
                or entry.get("name")
                or entry.get("id")
                or entry.get("ticker")
            )
            if not symbol_value:
                raise ValueError("Each dataset entry in backtest.datasets must declare a symbol.")
            display_symbol, source_symbol = _resolve_symbol_entry(str(symbol_value), alias_map)

            ltf_candidates = _to_list(entry.get("ltf") or entry.get("ltfs") or entry.get("timeframes"))
            if not ltf_candidates:
                raise ValueError(f"{symbol_value}에는 최소 하나의 LTF/timeframe 값이 필요합니다.")

            start_value = entry.get("start") or entry.get("from") or base_period.get("from")
            end_value = entry.get("end") or entry.get("to") or base_period.get("to")
            if not start_value or not end_value:
                raise ValueError(f"{symbol_value}에는 start/end 기간이 모두 필요합니다.")
            start = str(start_value)
            end = str(end_value)

            symbol_log = (
                display_symbol if display_symbol == source_symbol else f"{display_symbol} ({source_symbol})"
            )
            selected_timeframes = list(OrderedDict((tf, None) for tf in ltf_candidates).keys())
            if forced_lookup:
                filtered: List[str] = []
                for timeframe in selected_timeframes:
                    normalised_tf = _normalise_timeframe_value(timeframe)
                    if normalised_tf and normalised_tf.lower() in forced_lookup:
                        filtered.append(timeframe)
                if not filtered:
                    requested = ", ".join(forced_timeframes) if forced_timeframes else "unknown"
                    raise ValueError(
                        f"요청된 타임프레임({requested})이 {symbol_value}에 존재하지 않습니다."
                    )
                selected_timeframes = filtered

            for timeframe in selected_timeframes:
                timeframe_text = str(timeframe)
                LOGGER.info(
                    "Preparing dataset %s %s %s->%s (single timeframe)",
                    symbol_log,
                    timeframe_text,
                    start,
                    end,
                )
                df = cache.get(source_symbol, timeframe_text, start, end)
                chart_tf_effective = timeframe_text
                chart_df = df
                total_volume = _compute_total_volume(df)
                datasets.append(
                    DatasetSpec(
                        symbol=display_symbol,
                        timeframe=timeframe_text,
                        start=start,
                        end=end,
                        df=df,
                        htf=None,
                        htf_timeframe=None,
                        source_symbol=source_symbol,
                        cache_info=cache_info,
                        total_volume=total_volume,
                        chart_timeframe=chart_tf_effective,
                        chart_df=chart_df,
                    )
                )
        if not datasets:
            raise ValueError("backtest.datasets 구성이 유효한 데이터셋을 생성하지 못했습니다.")
        return datasets

    symbols = backtest_cfg.get("symbols") or ([base_symbol] if base_symbol else [])
    timeframes = backtest_cfg.get("timeframes") or ([base_timeframe] if base_timeframe else [])
    if timeframes:

        def _tf_priority(tf: str) -> Tuple[int, float]:
            text = str(tf).strip().lower()
            if text == "1m":
                return (0, 1.0)
            if text.endswith("m"):
                try:
                    minutes = float(text[:-1])
                except ValueError:
                    minutes = float("inf")
                return (1, minutes)
            return (2, float("inf"))

        timeframes = sorted(dict.fromkeys(timeframes), key=_tf_priority)
    periods = _normalise_periods(backtest_cfg.get("periods"), base_period)

    if not symbols or not timeframes or not periods:
        raise ValueError(
            "Backtest configuration must specify symbol(s), timeframe(s), and at least one period with 'from'/'to' dates."
        )

    symbol_pairs = [_resolve_symbol_entry(symbol, alias_map) for symbol in symbols]

    datasets: List[DatasetSpec] = []
    for (display_symbol, source_symbol), timeframe, period in product(
        symbol_pairs, timeframes, periods
    ):
        start = str(period["from"])
        end = str(period["to"])
        symbol_log = (
            display_symbol if display_symbol == source_symbol else f"{display_symbol} ({source_symbol})"
        )
        LOGGER.info(
            "Preparing dataset %s %s %s->%s (single timeframe)",
            symbol_log,
            timeframe,
            start,
            end,
        )
        df = cache.get(source_symbol, timeframe, start, end)
        chart_tf_effective = timeframe
        chart_df = df
        total_volume = _compute_total_volume(df)
        datasets.append(
            DatasetSpec(
                symbol=display_symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                df=df,
                htf=None,
                htf_timeframe=None,
                source_symbol=source_symbol,
                cache_info=cache_info,
                total_volume=total_volume,
                chart_timeframe=chart_tf_effective,
                chart_df=chart_df,
            )
        )
    return datasets


def combine_metrics(
    metric_list: List[Dict[str, float]], *, simple_override: Optional[bool] = None
) -> Dict[str, float]:
    """Merge per-timeframe metrics into a single trial summary.

    When ``timeframe_mix`` is active the optimiser evaluates the same
    parameter vector across multiple datasets or lower timeframes. This
    helper collapses the individual metric dictionaries produced by those
    runs into one consolidated payload:

    - ``Returns`` series are concatenated and re-weighted.
    - Capital fields (``TotalAssets``, ``AvailableCapital``, ``Savings``,
      ``Withdrawable``) are averaged with volume-based weights.
    - ``Liquidations`` and band-exit counters are summed.
    - ``TradesList`` entries are merged to preserve chronological order.
    - ``aggregate_metrics`` is recomputed so derived scores (e.g. NetProfit,
      WinRate) reflect the combined performance.

    The resulting structure underpins ``results.csv`` and
    ``results_timeframe_summary.csv`` so that composite runs such as
    ``1m,3m,5m`` behave like a single trial in downstream reports.
    """
    if not metric_list:
        return {}

    simple_mode = bool(simple_override)
    if simple_override is None:
        for metrics in metric_list:
            if bool(metrics.get("SimpleMetricsOnly")):
                simple_mode = True
                break

    combined_returns: List[pd.Series] = []
    combined_trades: List[Trade] = [] if not simple_mode else []
    total_assets_values: List[Optional[float]] = []
    final_equity_values: List[Optional[float]] = []
    net_profit_abs_values: List[Optional[float]] = []
    available_values: List[Optional[float]] = []
    savings_values: List[Optional[float]] = []
    withdrawable_values: List[Optional[float]] = []
    band_exit_enabled_flags: List[Optional[float]] = []
    band_exit_bb_mult_values: List[Optional[float]] = []
    band_exit_kc_mult_values: List[Optional[float]] = []
    weight_factors: List[float] = []
    total_liquidations = 0.0
    band_exit_total = 0.0
    band_exit_bb_total = 0.0
    band_exit_kc_total = 0.0
    drawdown_candidates: List[float] = []
    ruin_detected = False
    valid_flag = True

    def _coerce_float(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return float(number)

    for metrics in metric_list:
        returns = metrics.get("Returns")
        if isinstance(returns, pd.Series):
            combined_returns.append(returns)

        valid_flag = valid_flag and bool(metrics.get("Valid", True))

        total_assets = _coerce_float(metrics.get("TotalAssets"))
        total_assets_values.append(total_assets)

        final_equity = _coerce_float(metrics.get("FinalEquity"))
        final_equity_values.append(final_equity)

        net_profit_abs = _coerce_float(metrics.get("NetProfitAbs"))
        net_profit_abs_values.append(net_profit_abs)

        available = _coerce_float(metrics.get("AvailableCapital"))
        available_values.append(available)

        savings = _coerce_float(metrics.get("Savings"))
        savings_values.append(savings)
        withdrawable = _coerce_float(metrics.get("Withdrawable"))
        withdrawable_values.append(withdrawable)

        trades_value = _coerce_float(metrics.get("Trades"))

        band_exit_enabled = _coerce_float(metrics.get("BandExitEnabled"))
        band_exit_enabled_flags.append(band_exit_enabled)
        band_exit_bb_mult = _coerce_float(metrics.get("BandExitBBMult"))
        band_exit_bb_mult_values.append(band_exit_bb_mult)
        band_exit_kc_mult = _coerce_float(metrics.get("BandExitKCMult"))
        band_exit_kc_mult_values.append(band_exit_kc_mult)

        dd_raw = metrics.get("MaxDD")
        if dd_raw is None:
            dd_raw = metrics.get("MaxDrawdown")
        drawdown_value = _coerce_float(dd_raw)
        if drawdown_value is not None:
            dd_abs = abs(drawdown_value)
            drawdown_candidates.append(dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs)
        else:
            dd_pct_candidate = _coerce_float(metrics.get("DrawdownPct"))
            if dd_pct_candidate is not None:
                drawdown_candidates.append(float(dd_pct_candidate))

        def _weight_component(value: Optional[float]) -> float:
            if value is None or not np.isfinite(value):
                return 0.0
            return max(float(value), 0.0)

        trades_component = _weight_component(trades_value)
        assets_component = _weight_component(total_assets)
        drawdown_component = abs(drawdown_value) if drawdown_value is not None else 0.0

        trade_scale = 1.0 + float(np.log1p(trades_component))
        asset_scale = 1.0 + float(np.log1p(assets_component))
        drawdown_scale = 1.0 + float(drawdown_component)
        weight = (trade_scale * asset_scale) / drawdown_scale if drawdown_scale else 0.0
        if not np.isfinite(weight) or weight <= 0.0:
            weight = 1.0
        weight_factors.append(weight)

        liquidation_value = _coerce_float(metrics.get("Liquidations"))
        if liquidation_value is not None:
            total_liquidations += liquidation_value

        band_exit_value = _coerce_float(metrics.get("BandExitCount"))
        if band_exit_value is not None:
            band_exit_total += band_exit_value
        band_exit_bb_value = _coerce_float(metrics.get("BandExitBBCount"))
        if band_exit_bb_value is not None:
            band_exit_bb_total += band_exit_bb_value
        band_exit_kc_value = _coerce_float(metrics.get("BandExitKCCount"))
        if band_exit_kc_value is not None:
            band_exit_kc_total += band_exit_kc_value

        ruin_detected = ruin_detected or bool(metrics.get("Ruin"))

        if not simple_mode:
            trades = metrics.get("TradesList")
            if isinstance(trades, list):
                combined_trades.extend(trades)

    if not weight_factors:
        weight_factors = [1.0] * len(metric_list)

    def _weighted_mean(values: Sequence[Optional[float]]) -> Optional[float]:
        total_weight = 0.0
        total_value = 0.0
        for value, weight in zip(values, weight_factors):
            if value is None or not np.isfinite(value):
                continue
            if weight <= 0.0 or not np.isfinite(weight):
                continue
            total_weight += weight
            total_value += float(value) * weight
        if total_weight > 0.0:
            return float(total_value / total_weight)
        fallback = [float(v) for v in values if v is not None and np.isfinite(v)]
        if fallback:
            return float(np.mean(fallback))
        return None

    def _fallback_mean(values: Sequence[Optional[float]]) -> Optional[float]:
        cleaned = [float(value) for value in values if value is not None and np.isfinite(value)]
        if cleaned:
            return float(np.mean(cleaned))
        return None

    merged_returns = (
        pd.concat(combined_returns, axis=0).sort_index() if combined_returns else pd.Series(dtype=float)
    )

    if simple_mode:
        returns_clean = merged_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if returns_clean.empty:
            net_profit = 0.0
        else:
            equity = equity_curve_from_returns(returns_clean, initial=1.0)
            net_profit = (
                float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0])
                if len(equity) > 1
                else 0.0
            )
        if returns_clean.empty and not net_profit:
            fallback_candidates = []
            for metrics in metric_list:
                candidate = metrics.get("NetProfit")
                if candidate is None:
                    candidate = metrics.get("TotalReturn")
                fallback_candidates.append(_coerce_float(candidate))
            fallback_weighted = _weighted_mean(fallback_candidates)
            if fallback_weighted is None:
                fallback_weighted = _fallback_mean(fallback_candidates)
            if fallback_weighted is not None:
                net_profit = float(fallback_weighted)

        aggregated: Dict[str, float] = {
            "NetProfit": net_profit,
            "TotalReturn": net_profit,
            "Trades": float(sum(_coerce_float(m.get("Trades")) or 0.0 for m in metric_list)),
            "Wins": float(sum(_coerce_float(m.get("Wins")) or 0.0 for m in metric_list)),
            "Losses": float(sum(_coerce_float(m.get("Losses")) or 0.0 for m in metric_list)),
            "GrossProfit": float(sum(_coerce_float(m.get("GrossProfit")) or 0.0 for m in metric_list)),
            "GrossLoss": float(sum(_coerce_float(m.get("GrossLoss")) or 0.0 for m in metric_list)),
        }
        aggregated["BandExitCount"] = float(sum(_coerce_float(m.get("BandExitCount")) or 0.0 for m in metric_list))
        aggregated["BandExitBBCount"] = float(sum(_coerce_float(m.get("BandExitBBCount")) or 0.0 for m in metric_list))
        aggregated["BandExitKCCount"] = float(sum(_coerce_float(m.get("BandExitKCCount")) or 0.0 for m in metric_list))
        band_exit_enabled_vals = [_coerce_float(m.get("BandExitEnabled")) for m in metric_list]
        enabled_valid = [val for val in band_exit_enabled_vals if val is not None]
        aggregated["BandExitEnabled"] = float(np.mean(enabled_valid)) if enabled_valid else 0.0
        aggregated["BandExitBBMult"] = float(np.mean([_coerce_float(m.get("BandExitBBMult")) or 0.0 for m in metric_list]))
        aggregated["BandExitKCMult"] = float(np.mean([_coerce_float(m.get("BandExitKCMult")) or 0.0 for m in metric_list]))
        avg_holds = [
            val for val in (_coerce_float(m.get("AvgHoldBars")) for m in metric_list) if val is not None
        ]
        aggregated["AvgHoldBars"] = float(np.mean(avg_holds)) if avg_holds else 0.0
        trades_total = aggregated.get("Trades", 0.0)
        wins_total = aggregated.get("Wins", 0.0)
        aggregated["WinRate"] = float(wins_total / trades_total) if trades_total else 0.0
        max_losses = [
            int(round(_coerce_float(m.get("MaxConsecutiveLosses")) or 0.0))
            for m in metric_list
        ]
        aggregated["MaxConsecutiveLosses"] = float(max(max_losses)) if max_losses else 0.0
        aggregated["SimpleMetricsOnly"] = True
    else:
        combined_trades.sort(
            key=lambda trade: (
                getattr(trade, "entry_time", None),
                getattr(trade, "exit_time", None),
            )
        )
        aggregated = aggregate_metrics(combined_trades, merged_returns, simple=False)
        aggregated["TradesList"] = combined_trades

    dd_existing = _coerce_float(aggregated.get("DrawdownPct"))
    if dd_existing is None:
        aggregated["DrawdownPct"] = float(max(drawdown_candidates)) if drawdown_candidates else 0.0

    aggregated["Returns"] = merged_returns
    aggregated["Valid"] = bool(valid_flag)
    total_assets_weighted = _weighted_mean(total_assets_values)
    if total_assets_weighted is None:
        total_assets_weighted = _fallback_mean(total_assets_values)
    if total_assets_weighted is not None:
        aggregated["TotalAssets"] = float(total_assets_weighted)
    else:
        aggregated["TotalAssets"] = float(aggregated.get("TotalAssets", 0.0))

    final_equity_weighted = _weighted_mean(final_equity_values)
    if final_equity_weighted is None:
        final_equity_weighted = _fallback_mean(final_equity_values)
    if final_equity_weighted is not None:
        aggregated["FinalEquity"] = float(final_equity_weighted)
    else:
        aggregated["FinalEquity"] = float(aggregated.get("FinalEquity", aggregated.get("TotalAssets", 0.0)))

    available_weighted = _weighted_mean(available_values)
    if available_weighted is None:
        available_weighted = _fallback_mean(available_values)
    if available_weighted is not None:
        aggregated["AvailableCapital"] = float(available_weighted)

    savings_weighted = _weighted_mean(savings_values)
    if savings_weighted is None:
        savings_weighted = _fallback_mean(savings_values)
    if savings_weighted is not None:
        aggregated["Savings"] = float(savings_weighted)
    withdrawable_weighted = _weighted_mean(withdrawable_values)
    if withdrawable_weighted is None:
        withdrawable_weighted = _fallback_mean(withdrawable_values)
    if withdrawable_weighted is not None:
        aggregated["Withdrawable"] = float(withdrawable_weighted)
    band_exit_enabled_weighted = _weighted_mean(band_exit_enabled_flags)
    if band_exit_enabled_weighted is None:
        band_exit_enabled_weighted = _fallback_mean(band_exit_enabled_flags)
    if band_exit_enabled_weighted is not None:
        aggregated["BandExitEnabled"] = float(band_exit_enabled_weighted)
    else:
        aggregated["BandExitEnabled"] = float(aggregated.get("BandExitEnabled", 0.0))
    band_exit_bb_mult_weighted = _weighted_mean(band_exit_bb_mult_values)
    if band_exit_bb_mult_weighted is None:
        band_exit_bb_mult_weighted = _fallback_mean(band_exit_bb_mult_values)
    if band_exit_bb_mult_weighted is not None:
        aggregated["BandExitBBMult"] = float(band_exit_bb_mult_weighted)
    else:
        aggregated["BandExitBBMult"] = float(aggregated.get("BandExitBBMult", 0.0))
    band_exit_kc_mult_weighted = _weighted_mean(band_exit_kc_mult_values)
    if band_exit_kc_mult_weighted is None:
        band_exit_kc_mult_weighted = _fallback_mean(band_exit_kc_mult_values)
    if band_exit_kc_mult_weighted is not None:
        aggregated["BandExitKCMult"] = float(band_exit_kc_mult_weighted)
    else:
        aggregated["BandExitKCMult"] = float(aggregated.get("BandExitKCMult", 0.0))
    net_profit_abs_weighted = _weighted_mean(net_profit_abs_values)
    if net_profit_abs_weighted is None:
        net_profit_abs_weighted = _fallback_mean(net_profit_abs_values)
    if net_profit_abs_weighted is not None:
        aggregated["NetProfitAbs"] = float(net_profit_abs_weighted)
    else:
        aggregated["NetProfitAbs"] = float(aggregated.get("NetProfitAbs", 0.0))
    aggregated["Liquidations"] = float(total_liquidations)
    aggregated["BandExitCount"] = float(band_exit_total)
    aggregated["BandExitBBCount"] = float(band_exit_bb_total)
    aggregated["BandExitKCCount"] = float(band_exit_kc_total)
    trades_aggregate = _coerce_float(aggregated.get("Trades"))
    if trades_aggregate is not None and trades_aggregate > 0:
        aggregated["BandExitRatio"] = float(band_exit_total / trades_aggregate)
    else:
        aggregated["BandExitRatio"] = 0.0

    band_exit_summary = _band_exit_summary(aggregated)
    if band_exit_summary:
        aggregated["BandExitMode"] = str(band_exit_summary.get("mode") or "NONE").upper()
        try:
            aggregated["BandExitMult"] = float(band_exit_summary.get("mult", 0.0))
        except (TypeError, ValueError):
            aggregated["BandExitMult"] = 0.0
        aggregated["BandExitSummary"] = _format_band_exit_summary(band_exit_summary)
    else:
        aggregated["BandExitMode"] = "NONE"
        aggregated["BandExitMult"] = 0.0
        aggregated["BandExitSummary"] = "Band exit disabled"
    aggregated["Ruin"] = float(ruin_detected)

    penalty_defaults = {
        "MinTrades": 0.0,
        "MinHoldBars": 0.0,
        "MaxConsecutiveLossLimit": 0.0,
        "TradePenalty": 0.0,
        "HoldPenalty": 0.0,
        "ConsecutiveLossPenalty": 0.0,
    }

    for key, default in penalty_defaults.items():
        selected: Optional[float] = None
        for metrics in metric_list:
            value = _coerce_float(metrics.get(key))
            if value is not None:
                selected = value
        if selected is None:
            selected = default
        aggregated[key] = float(selected)

    return aggregated


def compute_total_asset_score(
    metrics: Dict[str, object], constraints: Optional[Dict[str, object]] = None
) -> float:
    """Compute a deterministic score from total-assets metrics and constraints."""

    constraints = constraints or {}

    def _as_float(value: object, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(number):
            return default
        return number

    if bool(metrics.get("Ruin")):
        return 0.0

    total_assets = metrics.get("TotalAssets")
    total_assets_value = _as_float(total_assets, 0.0)
    score = _as_float(total_assets, _as_float(metrics.get("FinalEquity"), 0.0))

    trades = int(round(_as_float(metrics.get("Trades"), 0.0)))
    min_trades = int(round(_as_float(constraints.get("min_trades_test"), 12.0)))
    if trades < min_trades:
        return 0.0

    dd_pct_value = _as_float(metrics.get("DrawdownPct"), float("nan"))
    if not np.isfinite(dd_pct_value):
        dd_raw = metrics.get("MaxDD")
        if dd_raw is None:
            dd_raw = metrics.get("MaxDrawdown")
        dd_value = abs(_as_float(dd_raw, 0.0))
        dd_pct_value = dd_value * 100.0 if dd_value <= 1.0 else dd_value
    dd_pct = dd_pct_value
    max_dd_limit = _as_float(constraints.get("max_dd_pct"), 70.0)
    if total_assets_value <= 50.0:
        return 0.0

    dd_ruin_limit = 80.0
    if dd_pct >= dd_ruin_limit:
        return 0.0

    if dd_pct > max_dd_limit and score > 0:
        excess = min(dd_pct - max_dd_limit, 100.0)
        score *= max(0.0, 1.0 - excess / 100.0)

    liquidations = _as_float(metrics.get("Liquidations"), 0.0)
    if liquidations > 0:
        score -= 25.0 * liquidations

    dd_penalty_ratio = min(max(dd_pct, 0.0), 100.0) / 100.0
    if dd_penalty_ratio > 0.0:
        if score >= 0.0:
            score *= max(0.0, 1.0 - dd_penalty_ratio)
        else:
            score *= 1.0 + dd_penalty_ratio

    return max(score, 0.0)


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(item, (int, float, bool, str)) for item in value):
                clean[key] = ", ".join(str(item) for item in value)
    return clean


def _create_pruner(name: str, params: Dict[str, object]) -> optuna.pruners.BasePruner:
    name = (name or "asha").lower()
    params = params or {}
    if name in {"none", "nop", "off"}:
        return optuna.pruners.NopPruner()
    if name in {"median", "medianpruner"}:
        return optuna.pruners.MedianPruner(**params)
    if name in {"hyperband"}:
        return optuna.pruners.HyperbandPruner(**params)
    if name in {"threshold", "thresholdpruner"}:
        return optuna.pruners.ThresholdPruner(**params)
    if name in {"patient", "patientpruner"}:
        patience = int(params.get("patience", 10))
        wrapped = _create_pruner(params.get("wrapped", "nop"), params.get("wrapped_params", {}))
        return optuna.pruners.PatientPruner(wrapped, patience=patience)
    if name in {"wilcoxon", "wilcoxonpruner"}:
        return optuna.pruners.WilcoxonPruner(**params)
    # Default to ASHA / successive halving
    return optuna.pruners.SuccessiveHalvingPruner(**params)


def optimisation_loop(
    datasets: List[DatasetSpec],
    params_cfg: Dict[str, object],
    objectives: Iterable[object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    constraints_cfg: Dict[str, object],
    search_cfg: Dict[str, object],
    forced_params: Optional[Dict[str, object]] = None,
    *,
    study_storage: Optional[Path] = None,
    space_hash: Optional[str] = None,
    seed_trials: Optional[List[Dict[str, object]]] = None,
    log_dir: Optional[Path] = None,
    force_sqlite_serial: bool = False,
) -> Dict[str, object]:
    search_cfg = params_cfg.get("search", {})
    objective_specs: List[ObjectiveSpec] = normalise_objectives(objectives)
    if not objective_specs:
        objective_specs = [ObjectiveSpec(name="NetProfit")]
    multi_objective = bool(search_cfg.get("multi_objective", False)) and len(objective_specs) > 1
    directions = [spec.direction for spec in objective_specs]
    primary_direction = objective_specs[0].direction if objective_specs else "maximize"
    original_space = build_space(params_cfg.get("space", {}))

    basic_profile_flag = _coerce_bool_or_none(search_cfg.get("basic_factor_profile"))
    if basic_profile_flag is None:
        basic_profile_flag = _coerce_bool_or_none(search_cfg.get("use_basic_factors"))
    use_basic_factors = True if basic_profile_flag is None else basic_profile_flag

    allow_sqlite_parallel_flag = _coerce_bool_or_none(
        search_cfg.get("allow_sqlite_parallel")
    )
    allow_sqlite_parallel = (
        bool(allow_sqlite_parallel_flag)
        if allow_sqlite_parallel_flag is not None
        else False
    )

    space = _restrict_to_basic_factors(original_space, enabled=use_basic_factors)
    param_order = list(space.keys())
    if use_basic_factors:
        if len(space) != len(original_space):
            LOGGER.info(
                "Basic-factor profile trimmed search space from %d to %d parameters",
                len(original_space),
                len(space),
            )
            if not space:
                LOGGER.warning("Basic factor filter removed all parameters; restoring original space")
                space = original_space
    else:
        LOGGER.info(
            "Full optimisation space enabled with %d parameters",
            len(space),
        )

    space, timeframe_added = _ensure_timeframe_param(space, datasets, search_cfg)
    if timeframe_added:
        LOGGER.info(
            "Added timeframe parameter with values: %s",
            ", ".join(space["timeframe"].get("values", [])),
        )
        param_order = list(space.keys())

    params_cfg["space"] = space

    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    available_cpu = max(1, multiprocessing.cpu_count())

    raw_n_jobs = search_cfg.get("n_jobs", 1)
    try:
        n_jobs = max(1, int(raw_n_jobs))
    except (TypeError, ValueError):
        n_jobs = 1
    if force_sqlite_serial and n_jobs != 1:
        LOGGER.info("SQLite serial mode enforced; forcing Optuna n_jobs=%d.", n_jobs)
        n_jobs = 1
        search_cfg["n_jobs"] = n_jobs
    if n_trials := int(search_cfg.get("n_trials", 0) or 0):
        auto_jobs = max(1, min(available_cpu, n_trials))
    else:
        auto_jobs = max(1, available_cpu)
    if not force_sqlite_serial and n_jobs <= 1 and auto_jobs > n_jobs:
        n_jobs = auto_jobs
        search_cfg["n_jobs"] = n_jobs
        LOGGER.info("Optuna n_jobs auto-adjusted to %d.", n_jobs)

    if n_jobs > 1:
        LOGGER.info("Optuna worker count set to %d.", n_jobs)
    (
        n_jobs,
        dataset_jobs,
        dataset_executor,
        dataset_start_method,
    ) = _configure_parallel_workers(
        search_cfg,
        dataset_groups,
        available_cpu=available_cpu,
        n_jobs=n_jobs,
    )

    process_handles: Optional[List[Dict[str, object]]] = None
    if dataset_executor == "process":
        process_handles = _serialise_datasets_for_process(datasets)

    algo_raw = search_cfg.get("algo", "bayes")
    algo = str(algo_raw or "bayes").lower()
    seed = search_cfg.get("seed")
    n_trials = int(search_cfg.get("n_trials", 50))
    forced_params = forced_params or {}
    log_dir_path: Optional[Path] = Path(log_dir) if log_dir else None
    trial_log_path: Optional[Path] = None
    best_yaml_path: Optional[Path] = None
    final_csv_path: Optional[Path] = None
    trial_csv_path: Optional[Path] = None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        trial_log_path = log_dir_path / "trials.jsonl"
        best_yaml_path = log_dir_path / "best.yaml"
        final_csv_path = log_dir_path / "trials_final.csv"
        trial_csv_path = log_dir_path / "trials_progress.csv"
        for candidate in (trial_log_path, best_yaml_path, final_csv_path, trial_csv_path):
            if candidate.exists():
                candidate.unlink()
    non_finite_penalty = float(search_cfg.get("non_finite_penalty", NON_FINITE_PENALTY))
    constraints_raw = params_cfg.get("constraints")
    constraints_cfg = dict(constraints_raw) if isinstance(constraints_raw, dict) else {}
    if not constraints_cfg:
        backtest_constraints = backtest_cfg.get("constraints")
        if isinstance(backtest_constraints, dict):
            constraints_cfg = dict(backtest_constraints)
    llm_cfg = params_cfg.get("llm", {}) if isinstance(params_cfg.get("llm"), dict) else {}

    nsga_params_cfg = search_cfg.get("nsga_params") or {}
    nsga_kwargs: Dict[str, object] = {}
    population_override = nsga_params_cfg.get("population_size") or search_cfg.get("nsga_population")
    if population_override is not None:
        try:
            nsga_kwargs["population_size"] = int(population_override)
        except (TypeError, ValueError):
            pass
    if nsga_params_cfg.get("mutation_prob") is not None:
        try:
            nsga_kwargs["mutation_prob"] = float(nsga_params_cfg["mutation_prob"])
        except (TypeError, ValueError):
            pass
    if nsga_params_cfg.get("crossover_prob") is not None:
        try:
            nsga_kwargs["crossover_prob"] = float(nsga_params_cfg["crossover_prob"])
        except (TypeError, ValueError):
            pass
    if nsga_params_cfg.get("swapping_prob") is not None:
        try:
            nsga_kwargs["swapping_prob"] = float(nsga_params_cfg["swapping_prob"])
        except (TypeError, ValueError):
            pass

    use_nsga = algo in {"nsga", "nsga2", "nsgaii"}
    if use_nsga:
        sampler = optuna.samplers.NSGAIISampler(**nsga_kwargs)
    elif algo == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif algo in {"cmaes", "cma-es", "cma"}:
        # 선택형(choice/bool) 파라미터는 TPE(Bayes)가 맡도록 설정
        categorical_sampler = optuna.samplers.TPESampler(seed=seed)
        
        sampler = optuna.samplers.CmaEsSampler(
            seed=seed,
            consider_pruned_trials=True,
            independent_sampler=categorical_sampler  # <- ✨ 이 한 줄이 핵심이야!
        )
        LOGGER.info("CMA-ES (숫자형) + TPE (선택형) 하이브리드 샘플러를 활성화합니다.")
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
    pruner_cfg = str(search_cfg.get("pruner", "asha"))
    pruner_params = search_cfg.get("pruner_params", {})
    pruner = _create_pruner(pruner_cfg, pruner_params or {})

    storage_cfg = search_cfg.get("storage_url")
    storage_env_key = search_cfg.get("storage_url_env")
    storage_env_value = os.getenv(str(storage_env_key)) if storage_env_key else None

    storage_url = None
    if storage_env_value:
        storage_url = str(storage_env_value)
    elif storage_cfg:
        storage_url = str(storage_cfg)
    elif study_storage is not None:
        study_storage.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{study_storage}"

    study_name = search_cfg.get("study_name") or (space_hash[:12] if space_hash else None)

    storage: Optional[optuna.storages.RDBStorage]
    storage = None
    storage_meta = {
        "backend": None,
        "url": None,
        "path": None,
        "env_key": storage_env_key,
        "env_value_present": bool(storage_env_value),
    }
    if storage_url:
        # When creating storages, avoid setting heartbeat_interval or grace_period
        # to prevent Optuna experimental warnings.  Leave these as None so that
        # Optuna uses its defaults without emitting ExperimentalWarning.
        heartbeat_interval = None
        heartbeat_grace = None
        if storage_url.startswith("sqlite:///"):
            timeout_raw = search_cfg.get("sqlite_timeout", 120)
            try:
                sqlite_timeout = max(1, int(timeout_raw))
            except (TypeError, ValueError):
                sqlite_timeout = 120
            storage = _make_sqlite_storage(
                storage_url,
                timeout_sec=sqlite_timeout,
                heartbeat_interval=None,
                grace_period=None,
            )
            storage_meta["backend"] = "sqlite"
            storage_meta["url"] = storage_url
            storage_meta["allow_parallel"] = allow_sqlite_parallel
            try:
                storage_path = make_url(storage_url).database
            except Exception:
                storage_path = None
            if storage_path:
                storage_meta["path"] = storage_path
            elif study_storage is not None:
                storage_meta["path"] = str(study_storage)
            if n_jobs > 1:
                if allow_sqlite_parallel:
                    LOGGER.info("SQLite storage parallel mode enabled (n_jobs=%d)", n_jobs)
                else:
                    LOGGER.info("SQLite storage running in serial mode; forcing n_jobs=1")
                    search_cfg["n_jobs"] = 1
        else:
            pool_size = _coerce_config_int(
                search_cfg.get("storage_pool_size"),
                minimum=1,
                name="storage_pool_size",
            )
            max_overflow = _coerce_config_int(
                search_cfg.get("storage_max_overflow"),
                minimum=0,
                name="storage_max_overflow",
            )
            pool_timeout = _coerce_config_int(
                search_cfg.get("storage_pool_timeout"),
                minimum=0,
                name="storage_pool_timeout",
            )
            pool_recycle = _coerce_config_int(
                search_cfg.get("storage_pool_recycle"),
                minimum=0,
                name="storage_pool_recycle",
            )
            connect_timeout = _coerce_config_int(
                search_cfg.get("storage_connect_timeout"),
                minimum=1,
                name="storage_connect_timeout",
            )
            statement_timeout_ms = _coerce_config_int(
                search_cfg.get("storage_statement_timeout_ms"),
                minimum=1,
                name="storage_statement_timeout_ms",
            )

            isolation_level_raw = search_cfg.get("storage_isolation_level")
            isolation_level = None
            if isinstance(isolation_level_raw, str):
                isolation_level = isolation_level_raw.strip() or None
            elif isolation_level_raw is not None:
                isolation_level = str(isolation_level_raw).strip() or None

            rdb_storage: Optional[optuna.storages.RDBStorage] = None
            rdb_error: Optional[BaseException] = None
            try:
                candidate_storage = _make_rdb_storage(
                    storage_url,
                    heartbeat_interval=None,
                    grace_period=None,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    pool_recycle=pool_recycle,
                    isolation_level=isolation_level,
                    connect_timeout=connect_timeout,
                    statement_timeout_ms=statement_timeout_ms,
                )
                # Force an eager connection to validate credentials and reachability.
                with candidate_storage.engine.connect() as connection:  # type: ignore[
                    connection.execute(text("SELECT 1"))
                rdb_storage = candidate_storage
            except (SQLAlchemyError, ModuleNotFoundError, OSError) as exc:
                rdb_error = exc

            if rdb_error is None and rdb_storage is not None:
                storage = rdb_storage
                storage_meta["backend"] = "rdb"
                storage_meta["url"] = storage_url
                pool_meta = {}
                if pool_size is not None:
                    pool_meta["size"] = pool_size
                if max_overflow is not None:
                    pool_meta["max_overflow"] = max_overflow
                if pool_timeout is not None:
                    pool_meta["timeout"] = pool_timeout
                if pool_recycle is not None:
                    pool_meta["recycle"] = pool_recycle
                if pool_meta:
                    storage_meta["pool"] = pool_meta
                if connect_timeout is not None:
                    storage_meta["connect_timeout"] = connect_timeout
                if isolation_level:
                    storage_meta["isolation_level"] = isolation_level
                if statement_timeout_ms is not None:
                    storage_meta["statement_timeout_ms"] = statement_timeout_ms
            else:
                fallback_path = study_storage or (STUDY_ROOT / "fallback_optuna.sqlite3")
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                fallback_url = f"sqlite:///{fallback_path}"
                sqlite_timeout_raw = search_cfg.get("sqlite_timeout", 120)
                try:
                    sqlite_timeout = max(1, int(sqlite_timeout_raw))
                except (TypeError, ValueError):
                    sqlite_timeout = 120

                LOGGER.warning(
                    "Configured Optuna storage %s 연결에 실패하여 SQLite (%s)로 전환합니다: %s",
                    _mask_storage_url(storage_url),
                    fallback_url,
                    rdb_error or "unknown error",
                )

                search_cfg["storage_url"] = fallback_url
                storage_url = fallback_url

                storage = _make_sqlite_storage(
                    fallback_url,
                    timeout_sec=sqlite_timeout,
                    heartbeat_interval=None,
                    grace_period=None,
                )
                storage_meta["backend"] = "sqlite"
                storage_meta["url"] = fallback_url
                storage_meta["allow_parallel"] = allow_sqlite_parallel
                storage_meta["fallback_reason"] = repr(rdb_error) if rdb_error else "unknown"
                try:
                    storage_meta["path"] = make_url(fallback_url).database
                except Exception:
                    storage_meta["path"] = str(fallback_path)

                if n_jobs > 1 and not allow_sqlite_parallel:
                    LOGGER.info("SQLite fallback에서는 Optuna n_jobs=1만 지원합니다. 값을 1로 강제합니다.")
                    search_cfg["n_jobs"] = 1
                    n_jobs = 1
    else:
        storage_meta["backend"] = "none"
    storage_arg = storage if storage is not None else storage_url

    # Always initialise a fresh study rather than reusing an existing one.
    # Setting ``load_if_exists`` to False forces Optuna to create a new study
    # even when a storage backend is configured.  Without this override,
    # Optuna will resume a previously stored study which can cause trial
    # numbers to continue from where a prior run left off (e.g., starting at
    # trial 228) despite clearing cached data.  Users who wish to resume
    # studies can explicitly set ``study_name`` and adjust this behaviour.
    study_kwargs = dict(
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_arg,
        load_if_exists=True,
    )
    # Create a new study.  When load_if_exists=False but a study with the
    # same name already exists in the configured storage, Optuna raises
    # DuplicatedStudyError.  To avoid this error and start a fresh study,
    # append a timestamp to the name and retry.  This generates a unique
    # study for each run without requiring the user to supply --study-name.
    try:
        if multi_objective:
            study = optuna.create_study(directions=directions, **study_kwargs)
        else:
            study = optuna.create_study(direction=primary_direction, **study_kwargs)
    except optuna.exceptions.DuplicatedStudyError:
        base_name = study_kwargs.get("study_name") or "study"
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_kwargs["study_name"] = f"{base_name}_{suffix}"
        if multi_objective:
            study = optuna.create_study(directions=directions, **study_kwargs)
        else:
            study = optuna.create_study(direction=primary_direction, **study_kwargs)
    if space_hash:
        study.set_user_attr("space_hash", space_hash)

    for params in seed_trials or []:
        trial_params = dict(params)
        trial_params.update(forced_params)
        trial_params = _enforce_exit_guards(trial_params, context="seed enqueue")
        try:
            study.enqueue_trial(trial_params, skip_if_exists=True)
        except Exception:
            continue

    results: List[Dict[str, object]] = []
    results_lock = Lock()
    dataset_executor_pool: Optional[Executor] = None
    if dataset_jobs > 1:
        if dataset_executor == "process":
            try:
                ctx = (
                    multiprocessing.get_context(dataset_start_method)
                    if dataset_start_method
                    else multiprocessing.get_context("spawn")
                )
            except ValueError:
                ctx = multiprocessing.get_context("spawn")
            init_handles = process_handles or []
            dataset_executor_pool = ProcessPoolExecutor(
                max_workers=dataset_jobs,
                mp_context=ctx,
                initializer=_process_pool_initializer,
                initargs=(init_handles,),
            )
        else:
            dataset_executor_pool = ThreadPoolExecutor(max_workers=dataset_jobs)

    def _to_native(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _summarise_trades_payload(raw: object) -> Dict[str, object]:
        summary = {
            "available": False,
            "trades": 0,
            "wins": 0,
            "winrate": 0.0,
        }
        if not isinstance(raw, list):
            return summary

        def _coerce_float_local(value: object) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        wins = 0
        trades_count = 0

        for trade in raw:
            profit_value: Optional[float] = None
            is_win_value: Optional[bool] = None
            if isinstance(trade, Trade):
                profit_value = _coerce_float_local(getattr(trade, "profit", None))
                if profit_value is not None:
                    is_win_value = profit_value > 0
            elif isinstance(trade, Mapping):
                profit_value = _coerce_float_local(trade.get("profit"))
                is_win_raw = trade.get("is_win") or trade.get("isWin") or trade.get("win")
                if isinstance(is_win_raw, bool):
                    is_win_value = is_win_raw
                elif isinstance(is_win_raw, (int, float)):
                    is_win_value = bool(is_win_raw)
                elif isinstance(is_win_raw, str):
                    lowered = is_win_raw.strip().lower()
                    if lowered in {"true", "1", "win", "yes"}:
                        is_win_value = True
                    elif lowered in {"false", "0", "loss", "no"}:
                        is_win_value = False
                if profit_value is not None and is_win_value is None:
                    is_win_value = profit_value > 0
            else:
                continue

            if profit_value is None and is_win_value is None:
                continue

            trades_count += 1
            if is_win_value:
                wins += 1

        summary["available"] = True
        summary["trades"] = trades_count
        summary["wins"] = wins
        summary["winrate"] = float(wins / trades_count) if trades_count else 0.0
        return summary

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        with TRIAL_LOG_WRITE_LOCK:
            def _normalise_value(value: object) -> Optional[object]:
                if value is None:
                    return None
                if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
                    normalised: List[float] = []
                    for item in value:
                        try:
                            normalised.append(float(item))
                        except Exception:
                            return None
                    return normalised
                try:
                    return float(value)
                except Exception:
                    return None

            trial_value = _normalise_value(trial.value)
            state_label = _format_trial_state(trial.state)
            record = {
                "number": trial.number,
                "value": trial_value,
                "state": state_label,
                "state_raw": str(trial.state),
                "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else None,
            }

            effective_params_attr = trial.user_attrs.get("effective_params")
            if isinstance(effective_params_attr, dict):
                record["params"] = {key: _to_native(val) for key, val in effective_params_attr.items()}
            else:
                record["params"] = {key: _to_native(val) for key, val in trial.params.items()}

            if trial_log_path is not None:
                trial_log_path.parent.mkdir(parents=True, exist_ok=True)
                with trial_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            metrics_attr = trial.user_attrs.get("metrics")
            metrics = metrics_attr if isinstance(metrics_attr, dict) else {}
            trade_summary = _summarise_trades_payload(metrics.get("TradesList"))

            def _metric_value(name: str) -> Optional[object]:
                if name not in metrics:
                    return None
                return _to_native(metrics.get(name))

            dataset_key = trial.user_attrs.get("dataset_key")
            dataset_meta = dataset_key if isinstance(dataset_key, dict) else {}
            skipped_attr = trial.user_attrs.get("skipped_datasets")
            if isinstance(skipped_attr, list):
                skipped_serialisable = skipped_attr
            else:
                skipped_serialisable = [skipped_attr] if skipped_attr else []

            max_dd_value = _metric_value("MaxDD")
            if max_dd_value is None:
                max_dd_value = _metric_value("MaxDrawdown")

            value_field: object
            if isinstance(trial_value, list):
                value_field = json.dumps(trial_value, ensure_ascii=False)
            else:
                value_field = trial_value

            params_payload = record.get("params") if isinstance(record.get("params"), dict) else {}

            def _param_value(name: str) -> Optional[object]:
                if not isinstance(params_payload, dict):
                    return None
                if name not in params_payload:
                    return None
                return _to_native(params_payload.get(name))

            def _csv_value(value: Optional[object]) -> object:
                return "" if value is None else value

            def _positive_number(value: Optional[object]) -> Optional[float]:
                if value is None:
                    return None
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(numeric) or numeric <= 0:
                    return None
                return float(numeric)

            def _coerce_int_value(value: object) -> Optional[int]:
                if value is None:
                    return None
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(numeric):
                    return None
                return int(round(numeric))

            def _finite_float(value: object) -> Optional[float]:
                if value is None:
                    return None
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(numeric):
                    return None
                return float(numeric)

            # SAR exit removed
            use_sar_exit = None
            sar_start = None
            sar_increment = None
            sar_maximum = None

            fixed_stop_pct_raw = _param_value("fixedStopPct")
            fixed_stop_pct_val = _positive_number(fixed_stop_pct_raw)
            use_fixed_stop = True if fixed_stop_pct_val is not None else None

            use_channel_stop_flag = bool(trial.user_attrs.get("use_channel_stop"))
            stop_channel_type_attr = trial.user_attrs.get("stop_channel_type")
            stop_channel_mult_attr = trial.user_attrs.get("stop_channel_mult")
            if stop_channel_type_attr is None and use_channel_stop_flag:
                stop_channel_type_attr = _param_value("stopChannelType")
            if stop_channel_mult_attr is None and use_channel_stop_flag:
                stop_channel_mult_attr = _param_value("stopChannelMult")
            if isinstance(stop_channel_type_attr, str):
                stop_channel_type_attr = _normalise_channel_type(stop_channel_type_attr)
            if stop_channel_mult_attr not in (None, ""):
                try:
                    stop_channel_mult_attr = float(stop_channel_mult_attr)
                except (TypeError, ValueError):
                    pass

            use_band_exit_raw = trial.user_attrs.get("use_band_exit")
            if use_band_exit_raw is None:
                use_band_exit_raw = _param_value("useBandExit")
            use_band_exit_flag = _coerce_bool_or_none(use_band_exit_raw)
            band_exit_min_raw = trial.user_attrs.get("band_exit_min_bars")
            if band_exit_min_raw is None:
                band_exit_min_raw = _param_value("bandExitMinBars")
            band_exit_min_numeric = _positive_number(band_exit_min_raw)
            band_exit_min_bars_val: Optional[int] = None
            if band_exit_min_numeric is not None:
                candidate = int(math.floor(band_exit_min_numeric))
                if candidate >= 0:
                    band_exit_min_bars_val = candidate
            if use_band_exit_flag is False:
                band_exit_min_bars_val = None

            trial.set_user_attr("band_exit_min_bars", band_exit_min_bars_val)

            band_exit_mode_raw = trial.user_attrs.get("band_exit_mode")
            band_exit_mode = str(band_exit_mode_raw or "NONE").upper()
            band_exit_mult_attr = trial.user_attrs.get("band_exit_mult")
            try:
                band_exit_mult = float(band_exit_mult_attr)
            except (TypeError, ValueError):
                band_exit_mult = 0.0
            band_exit_summary_text = trial.user_attrs.get("band_exit_summary_text")
            if not band_exit_summary_text:
                if use_band_exit_flag:
                    if band_exit_mode in {"BB", "KC"} and band_exit_mult:
                        band_exit_summary_text = f"{band_exit_mode} × {band_exit_mult:.3f}"
                    else:
                        band_exit_summary_text = "Band exit enabled"
                else:
                    band_exit_summary_text = "Band exit disabled"

            band_exit_mode_display = band_exit_mode if use_band_exit_flag else "NONE"
            band_exit_mult_display = band_exit_mult if use_band_exit_flag else None

            params_json = json.dumps(record["params"], ensure_ascii=False, sort_keys=False)
            skipped_json = (
                json.dumps(skipped_serialisable, ensure_ascii=False)
                if skipped_serialisable
                else ""
            )

            total_assets_val = _metric_value("TotalAssets")
            if total_assets_val is None:
                total_assets_val = trial.user_attrs.get("total_assets")
            raw_total_assets = (
                _to_native(total_assets_val)
                if total_assets_val not in {None, ""}
                else None
            )
            if raw_total_assets is None:
                fallback_assets = trial.user_attrs.get("total_assets")
                if fallback_assets not in {None, ""}:
                    raw_total_assets = _to_native(fallback_assets)
            liquidations_val = _metric_value("Liquidations")
            if liquidations_val is None:
                liquidations_val = trial.user_attrs.get("liquidations")
            leverage_param = trial.params.get("leverage") if hasattr(trial, "params") else None
            leverage_val = _to_native(leverage_param) if leverage_param is not None else trial.user_attrs.get("leverage")

            if trade_summary["available"]:
                trades_value_row: Optional[int] = int(trade_summary["trades"])
                wins_value_row: Optional[int] = int(trade_summary["wins"])
                win_rate_value_row: Optional[float] = float(trade_summary["winrate"])
            else:
                trades_value_row = _coerce_int_value(_metric_value("Trades"))
                wins_value_row = _coerce_int_value(_metric_value("Wins"))
                win_rate_metric = _metric_value("WinRate")
                win_rate_value_row = _finite_float(win_rate_metric)
            if trades_value_row is None:
                trades_value_row = _coerce_int_value(_metric_value("TotalTrades"))
            if win_rate_value_row is not None:
                win_rate_value_row = round(float(win_rate_value_row), 4)

            timeframe_label = (
                _normalise_timeframe_value(params_payload.get("timeframe"))
                or dataset_meta.get("timeframe")
                or dataset_meta.get("effective_timeframe")
            )

            timeframe_origin_label = dataset_meta.get("timeframe_origin") or params_payload.get(
                "timeframe_origin"
            )
            timeframe_origin_label = _normalise_timeframe_value(timeframe_origin_label)
            if timeframe_origin_label == timeframe_label:
                timeframe_origin_label = None

            effective_timeframe_label = dataset_meta.get("effective_timeframe")
            if effective_timeframe_label:
                normalised_effective = _normalise_timeframe_value(effective_timeframe_label)
                effective_timeframe_label = normalised_effective or str(effective_timeframe_label)
            else:
                effective_timeframe_label = timeframe_label
            htf_timeframe_label = (
                dataset_meta.get("htf_timeframe")
                or params_payload.get("htf_timeframe")
                or params_payload.get("htf")
            )
            if htf_timeframe_label:
                normalised_htf = _normalise_timeframe_value(htf_timeframe_label)
                htf_timeframe_label = normalised_htf or str(htf_timeframe_label)
            else:
                htf_timeframe_label = ""

            anomaly_reason_val = metrics.get("AnomalyReason")
            anomaly_info = trial.user_attrs.get("anomaly_info")
            if anomaly_reason_val in (None, "") and isinstance(anomaly_info, dict):
                anomaly_reason_val = anomaly_info.get("type")
            row = {
                "number": trial.number,
                "total_assets": raw_total_assets,
                "leverage": leverage_val,
                "timeframe": timeframe_label,
                "timeframe_origin": _csv_value(timeframe_origin_label),
                "effective_timeframe": effective_timeframe_label,
                "htf_timeframe": htf_timeframe_label,
                "use_fixed_stop": _csv_value(True if use_fixed_stop else None),
                "fixed_stop_pct": _csv_value(fixed_stop_pct_val),
                "use_band_exit": _csv_value(use_band_exit_flag if use_band_exit_flag is not None else None),
                "band_exit_min_bars": _csv_value(band_exit_min_bars_val),
                "band_exit_mode": _csv_value(band_exit_mode_display),
                "band_exit_mult": _csv_value(
                    f"{band_exit_mult_display:.4f}" if band_exit_mult_display not in (None, "") else None
                ),
                "band_exit_summary": band_exit_summary_text,
                "score": trial.user_attrs.get("score"),
                "score_raw": raw_total_assets,
                "value": value_field,
                "value_raw": raw_total_assets,
                "state": state_label,
                "trades": trades_value_row,
                "wins": wins_value_row,
                "win_rate": win_rate_value_row,
                "max_dd": max_dd_value,
                "liquidations": liquidations_val,
                "valid": trial.user_attrs.get("valid"),
                "pruned": trial.user_attrs.get("pruned"),
                "anomaly_reason": _csv_value(anomaly_reason_val),
                # SAR columns removed
                "params": params_json,
                "skipped_datasets": skipped_json,
                "datetime_complete": record["datetime_complete"],
            }

            if trial_csv_path is not None:
                file_exists = trial_csv_path.exists()
                trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with trial_csv_path.open("a", encoding="utf-8", newline="") as csv_handle:
                    writer = csv.DictWriter(
                        csv_handle,
                        fieldnames=TRIAL_PROGRESS_FIELDS,
                        lineterminator="\n",
                    )
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

                try:
                    trials_snapshot = study.get_trials(deepcopy=False)
                except TypeError:
                    trials_snapshot = study.trials
            completed = sum(
                1
                for item in trials_snapshot
                if item.state in {TrialState.COMPLETE, TrialState.PRUNED}
            )
            total_display: object = n_trials if n_trials else len(trials_snapshot)
            total_assets_display = row.get("total_assets")
            if total_assets_display in {None, ""}:
                total_assets_display = trial.user_attrs.get("total_assets", 0.0)
            if total_assets_display in {None, ""}:
                total_assets_display = "-"
            liquidations_display = row.get("liquidations")
            if liquidations_display in {None, ""}:
                liquidations_display = trial.user_attrs.get("liquidations", 0)
            if liquidations_display in {None, ""}:
                liquidations_display = 0
            trades_display = row.get("trades", "-") if row.get("trades") not in {None, ""} else "-"
            wins_display = row.get("wins", "-") if row.get("wins") not in {None, ""} else "-"
            score_display = row.get("score", "-") if row.get("score") not in {None, ""} else "-"
            win_rate_display = row.get("win_rate")
            if win_rate_display in {None, ""}:
                win_rate_display = _metric_value("WinRate") or 0.0
            dd_display = row.get("max_dd")
            if dd_display in {None, ""}:
                dd_display = max_dd_value or 0.0

            def _format_metric_display(value: object) -> object:
                if value in {None, ""}:
                    return "-"
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return value
                if not np.isfinite(numeric):
                    return "-"
                return f"{numeric:.2f}"

            def _format_int_display(value: object) -> object:
                if value in {None, ""}:
                    return "-"
                try:
                    numeric = int(round(float(value)))
                except (TypeError, ValueError):
                    return value
                return str(numeric)

            def _truncate_two_places(value: float) -> float:
                return math.trunc(value * 100.0) / 100.0

            try:
                total_assets_numeric = float(total_assets_display)
            except (TypeError, ValueError):
                total_assets_numeric = None
            if total_assets_numeric is not None and np.isfinite(total_assets_numeric):
                total_assets_display = f"{_truncate_two_places(total_assets_numeric):.2f}"

            trades_display = _format_int_display(trades_display)
            wins_display = _format_int_display(wins_display)
            win_rate_display = _format_metric_display(win_rate_display)

            dd_percent_value: Optional[float] = None
            try:
                dd_numeric = float(dd_display)
            except (TypeError, ValueError):
                dd_numeric = None
            if dd_numeric is not None and np.isfinite(dd_numeric):
                dd_percent_value = dd_numeric
                dd_display = dd_percent_value
            dd_display = _format_metric_display(dd_display)
            if dd_percent_value is not None and dd_display != "-":
                dd_display = f"{dd_display}%"

            if band_exit_min_bars_val is not None and use_band_exit_flag:
                band_exit_display = f"{band_exit_summary_text} (min={band_exit_min_bars_val})"
            else:
                band_exit_display = band_exit_summary_text

            LOGGER.info(
                "Trial %d/%s (#%d %s) TotalAssets=%s, Liquidations=%s, Score=%s, Trades=%s, Wins=%s, WinRate=%s, MaxDD=%s, BandExit=%s",
                completed,
                total_display,
                trial.number,
                row.get("state"),
                total_assets_display,
                liquidations_display,
                score_display,
                trades_display,
                wins_display,
                win_rate_display,
                dd_display,
                band_exit_display,
            )

            if best_yaml_path is None:
                return
            best_yaml_path.parent.mkdir(parents=True, exist_ok=True)

            selected_trial: Optional[optuna.trial.FrozenTrial]
            if multi_objective:
                try:
                    pareto_trials = list(study.best_trials)
                except ValueError:
                    return
                if not pareto_trials:
                    return
                selected_trial = next(
                    (best_trial for best_trial in pareto_trials if best_trial.number == trial.number),
                    None,
                )
                if selected_trial is None:
                    return
            else:
                try:
                    selected_trial = study.best_trial
                except ValueError:
                    return
                if selected_trial.number != trial.number:
                    return

            best_value = _normalise_value(selected_trial.value)
            best_params_full = {key: _to_native(val) for key, val in selected_trial.params.items()}
            best_band_exit_summary = selected_trial.user_attrs.get("band_exit_summary_text") or "Band exit disabled"
            best_band_exit_mode_raw = selected_trial.user_attrs.get("band_exit_mode")
            best_band_exit_mode = str(best_band_exit_mode_raw or "NONE").upper()
            best_band_exit_mult_attr = selected_trial.user_attrs.get("band_exit_mult")
            try:
                best_band_exit_mult = float(best_band_exit_mult_attr)
            except (TypeError, ValueError):
                best_band_exit_mult = 0.0
            best_band_exit_min = selected_trial.user_attrs.get("band_exit_min_bars")
            if best_band_exit_min is None:
                best_band_exit_min = _param_value("bandExitMinBars")
            snapshot_band_exit = {
                "mode": best_band_exit_mode,
                "mult": best_band_exit_mult,
                "summary": best_band_exit_summary,
            }
            if best_band_exit_min not in (None, ""):
                snapshot_band_exit["min_bars"] = _to_native(best_band_exit_min)
            snapshot = {
                "best_value": best_value,
                "best_params": best_params_full,
                "band_exit": snapshot_band_exit,
            }
            if use_basic_factors:
                snapshot["basic_params"] = {
                    key: value for key, value in best_params_full.items() if key in BASIC_FACTOR_KEYS
                }
            else:
                snapshot["basic_params"] = dict(best_params_full)
            with best_yaml_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    callbacks: List = [_log_trial]

    stopper = _create_triple_backtick_stopper()
    if stopper is not None:
        callbacks.append(stopper.callback)

    diversifier = _build_trial_diversifier(
        space,
        search_cfg,
        forced_params=forced_params,
        param_order=param_order,
        seed=seed,
    )
    if diversifier is not None:
        callbacks.append(diversifier)

    llm_refresher = _build_llm_refresher(
        space,
        llm_cfg,
        forced_params=forced_params,
        use_basic_factors=use_basic_factors,
    )
    if llm_refresher is not None:
        callbacks.append(llm_refresher)

    def objective(trial: optuna.Trial) -> float:
        params = _safe_sample_parameters(trial, space)
        params.update(forced_params)
        params = _enforce_exit_guards(params, context=f"trial #{trial.number}")
        params = normalize_tf(params)
        timeframe_value = (
            _normalise_timeframe_value(params.get("timeframe"))
            or _normalise_timeframe_value(params_cfg.get("timeframe"))
        )

        pool_source = params.get("timeframe_pool")
        if not pool_source:
            pool_source = params_cfg.get("timeframe_pool")
        timeframe_pool: List[str] = []
        if isinstance(pool_source, (list, tuple, set)):
            for token in pool_source:
                normalised = _normalise_timeframe_value(token)
                if normalised:
                    timeframe_pool.append(normalised)
        if not timeframe_pool and timeframe_value == MERGED_TIMEFRAME_LABEL:
            timeframe_pool = list(MERGED_TIMEFRAME_COMPONENTS)



        def _positive_number(value: object) -> Optional[float]:
            if value is None:
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric) or numeric <= 0:
                return None
            return float(numeric)

        fixed_stop_pct_val = _positive_number(params.get("fixedStopPct"))
        use_fixed_stop_flag = fixed_stop_pct_val is not None
        channel_type_label = _normalise_channel_type(params.get("stopChannelType"))
        stop_channel_mult_val = _positive_number(params.get("stopChannelMult"))
        use_channel_stop_flag = (
            channel_type_label in {"BB", "KC"} and stop_channel_mult_val is not None
        )

        key, selected_datasets = _select_datasets_for_params(
            params_cfg, dataset_groups, timeframe_groups, default_key, params
        )
        effective_dataset = _pick_primary_dataset(selected_datasets) if selected_datasets else None
        effective_timeframe = (
            effective_dataset.timeframe if effective_dataset is not None else key[0]
        )
        components_source: Optional[object] = None
        if not timeframe_pool:
            components_source = params.get("timeframe_components")
            if not components_source:
                components_source = params_cfg.get("timeframe_components")
        timeframe_components: List[str] = []
        if isinstance(components_source, (list, tuple, set)):
            timeframe_components = [
                str(token).strip()
                for token in components_source
                if str(token).strip()
            ]
        if not timeframe_components and not timeframe_pool:
            preset_components = MERGED_TIMEFRAME_MAP.get(
                _normalise_timeframe_value(params.get("timeframe"))
                or _normalise_timeframe_value(params_cfg.get("timeframe"))
                or ""
            )
            if preset_components:
                timeframe_components = list(preset_components)

        dataset_key_payload = {
            "timeframe": key[0],
            "effective_timeframe": effective_timeframe,
        }
        if key[0] and key[0] != effective_timeframe:
            dataset_key_payload["timeframe_origin"] = key[0]
        if key[1]:
            dataset_key_payload["htf_timeframe"] = key[1]
        if timeframe_components:
            dataset_key_payload["components"] = timeframe_components

        trial.set_user_attr("dataset_key", dataset_key_payload)
        trial.set_user_attr("ltf_primary", effective_timeframe)
        if timeframe_components:
            trial.set_user_attr("timeframe_components", timeframe_components)
        trial.set_user_attr("use_fixed_stop", use_fixed_stop_flag)
        trial.set_user_attr("use_channel_stop", use_channel_stop_flag)
        trial.set_user_attr("stop_channel_type", channel_type_label)
        trial.set_user_attr("stop_channel_mult", float(stop_channel_mult_val) if use_channel_stop_flag else None)

        params_for_record = dict(params)
        raw_timeframe_label = _normalise_timeframe_value(params_for_record.get("timeframe"))
        params_for_record["timeframe"] = effective_timeframe
        if raw_timeframe_label and raw_timeframe_label != effective_timeframe:
            params_for_record["timeframe_origin"] = raw_timeframe_label
        params_for_record.pop("ltf", None)
        params_for_record.pop("ltfChoice", None)
        params_for_record.pop("ltf_choices", None)
        params_for_record.pop("timeframe_pool", None)
        if timeframe_components:
            params_for_record["timeframe_components"] = timeframe_components
        if channel_type_label is not None:
            params_for_record["stopChannelType"] = channel_type_label
        elif "stopChannelType" in params_for_record:
            params_for_record["stopChannelType"] = None
        if use_channel_stop_flag and stop_channel_mult_val is not None:
            params_for_record["stopChannelMult"] = float(stop_channel_mult_val)
        elif "stopChannelMult" in params_for_record:
            params_for_record["stopChannelMult"] = None
        trial.set_user_attr("effective_params", params_for_record)
        dataset_metrics: List[Dict[str, object]] = []
        numeric_metrics: List[Dict[str, float]] = []
        dataset_scores: List[float] = []
        skipped_dataset_records: List[Dict[str, object]] = []
        aggregate_trade_payload: List[Dict[str, object]] = []

        def _safe_float(value: object) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        def _coerce_int(value: object) -> Optional[int]:
            numeric = _safe_float(value)
            if numeric is None:
                return None
            return int(round(numeric))

        def _serialise_trades_list(raw: object) -> List[Dict[str, object]]:
            serialised: List[Dict[str, object]] = []
            if not isinstance(raw, list):
                return serialised

            def _maybe_float(candidate: object) -> Optional[float]:
                value = _safe_float(candidate)
                return value if value is not None else None

            def _maybe_timestamp(value: object) -> Optional[str]:
                if value is None:
                    return None
                if isinstance(value, pd.Timestamp):
                    ts = value
                else:
                    try:
                        ts = pd.Timestamp(value)
                    except Exception:
                        return None
                if ts.tzinfo is None:
                    ts = ts.tz_localize(UTC)
                else:
                    ts = ts.tz_convert(UTC)
                return ts.isoformat()

            for trade in raw:
                profit = None
                return_pct = None
                bars = None
                entry_ts: Optional[str] = None
                exit_ts: Optional[str] = None
                direction: Optional[str] = None
                qty: Optional[float] = None
                entry_price: Optional[float] = None
                exit_price: Optional[float] = None
                if isinstance(trade, Trade):
                    profit = _maybe_float(getattr(trade, "profit", None))
                    return_pct = _maybe_float(getattr(trade, "return_pct", None))
                    bars = _coerce_int(getattr(trade, "bars_held", None))
                    entry_ts = _maybe_timestamp(getattr(trade, "entry_time", None))
                    exit_ts = _maybe_timestamp(getattr(trade, "exit_time", None))
                    direction = getattr(trade, "direction", None)
                    qty = _maybe_float(getattr(trade, "size", None))
                    entry_price = _maybe_float(getattr(trade, "entry_price", None))
                    exit_price = _maybe_float(getattr(trade, "exit_price", None))
                elif isinstance(trade, Mapping):
                    profit = _maybe_float(trade.get("profit"))
                    return_pct = _maybe_float(trade.get("return_pct"))
                    bars = _coerce_int(trade.get("bars_held"))
                    entry_ts = _maybe_timestamp(
                        trade.get("entry_time")
                        or trade.get("entryTime")
                        or trade.get("entry_ts")
                    )
                    exit_ts = _maybe_timestamp(
                        trade.get("exit_time") or trade.get("exitTime") or trade.get("exit_ts")
                    )
                    direction_raw = trade.get("direction") or trade.get("side")
                    if isinstance(direction_raw, str):
                        direction = direction_raw
                    qty = _maybe_float(trade.get("qty") or trade.get("size"))
                    entry_price = _maybe_float(trade.get("entry_price") or trade.get("entryPrice"))
                    exit_price = _maybe_float(trade.get("exit_price") or trade.get("exitPrice"))
                else:
                    continue

                payload: Dict[str, object] = {}
                if profit is not None:
                    payload["profit"] = float(profit)
                if return_pct is not None:
                    payload["return_pct"] = float(return_pct)
                if bars is not None:
                    payload["bars_held"] = int(bars)
                if entry_ts is not None:
                    payload["entry_time"] = entry_ts
                if exit_ts is not None:
                    payload["exit_time"] = exit_ts
                if direction is not None:
                    payload["side"] = direction
                if qty is not None:
                    payload["qty"] = float(qty)
                if entry_price is not None:
                    payload["entry_price"] = float(entry_price)
                if exit_price is not None:
                    payload["exit_price"] = float(exit_price)
                if profit is not None:
                    payload["is_win"] = bool(profit > 0)
                quote_notional: Optional[float] = None
                if entry_price is not None and qty is not None:
                    quote_notional = abs(entry_price * qty)
                elif isinstance(trade, Mapping):
                    quote_notional = _maybe_float(
                        trade.get("quote_notional")
                        or trade.get("quoteNotional")
                        or trade.get("notional")
                    )
                if quote_notional is not None:
                    payload["quote_notional"] = float(abs(quote_notional))
                if payload:
                    serialised.append(payload)

            return serialised

        def _resolve_min_volume_threshold() -> float:
            candidates: List[object] = [
                params.get("min_volume"),
                params.get("minVolume"),
            ]
            if isinstance(risk, dict):
                candidates.extend([risk.get("min_volume"), risk.get("minVolume")])
            candidates.append(constraints_cfg.get("min_volume"))

            for candidate in candidates:
                numeric = _safe_float(candidate)
                if numeric is None or numeric < 0:
                    continue
                return float(numeric)
            return MIN_VOLUME_THRESHOLD

        def _sanitise(value: float, stage: str) -> float:
            try:
                numeric = float(value)
            except Exception:
                numeric = non_finite_penalty
            if not np.isfinite(numeric):
                return non_finite_penalty
            return numeric

        def _penalise_by_drawdown(value: float, metrics_map: Mapping[str, object]) -> float:
            dd_pct_value = _safe_float(metrics_map.get("DrawdownPct"))
            if dd_pct_value is None:
                dd_candidate = _safe_float(metrics_map.get("MaxDD", metrics_map.get("MaxDrawdown")))
                if dd_candidate is None:
                    return value
                dd_abs = abs(dd_candidate)
                dd_pct_value = dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs
            dd_pct = dd_pct_value
            dd_ratio = min(max(dd_pct, 0.0), 100.0) / 100.0
            if dd_ratio <= 0.0:
                return value
            if value >= 0.0:
                return value * max(0.0, 1.0 - dd_ratio)
            return value * (1.0 + dd_ratio)

        def _consume_dataset(
            idx: int,
            dataset: DatasetSpec,
            metrics: Dict[str, float],
            *,
            simple_override: bool = False,
        ) -> None:
            cleaned_metrics = _clean_metrics(metrics)
            band_exit_detail = _band_exit_summary(metrics)
            if band_exit_detail:
                band_exit_summary = _format_band_exit_summary(band_exit_detail)
                band_exit_mode = str(band_exit_detail.get("mode") or "NONE").upper()
                try:
                    band_exit_mult = float(band_exit_detail.get("mult", 0.0))
                except (TypeError, ValueError):
                    band_exit_mult = 0.0
            else:
                band_exit_summary = "Band exit disabled"
                band_exit_mode = "NONE"
                band_exit_mult = 0.0
            cleaned_metrics["BandExitSummary"] = band_exit_summary
            cleaned_metrics["BandExitMode"] = band_exit_mode
            cleaned_metrics["BandExitMult"] = band_exit_mult
            for key in (
                "BandExitBBCount",
                "BandExitKCCount",
                "BandExitBBRatio",
                "BandExitKCRatio",
                "BandExitBBMult",
                "BandExitKCMult",
            ):
                cleaned_metrics.pop(key, None)
            record = {
                "name": dataset.name,
                "meta": dataset.meta,
                "metrics": cleaned_metrics,
            }
            record["band_exit_summary"] = band_exit_summary
            record["band_exit_mode"] = band_exit_mode
            record["band_exit_mult"] = band_exit_mult
            if band_exit_detail:
                record["band_exit"] = band_exit_detail

            trades_serialised = _serialise_trades_list(metrics.get("TradesList"))
            if trades_serialised:
                record["trades"] = trades_serialised
                cleaned_metrics["TradesList"] = trades_serialised
                aggregate_trade_payload.extend(trades_serialised)

            trades_value = _coerce_int(metrics.get("Trades") or metrics.get("TotalTrades"))
            if trades_value is None:
                trades_value = _coerce_int(cleaned_metrics.get("Trades"))
            if trades_value is not None:
                cleaned_metrics["Trades"] = trades_value
            if trades_value is not None and trades_value < MIN_TRADES_ENFORCED:
                record["skipped"] = True
                record["skip_reason"] = "trades_threshold"
                record["skip_metric"] = trades_value
                record["skip_threshold"] = MIN_TRADES_ENFORCED
                dataset_metrics.append(record)
                skipped_dataset_records.append(
                    {
                        "name": dataset.name,
                        "timeframe": dataset.timeframe,
                        "htf_timeframe": dataset.htf_timeframe,
                        "trades": trades_value,
                        "min_trades": MIN_TRADES_ENFORCED,
                    }
                )
                return

            numeric_metrics.append(metrics)
            dataset_metrics.append(record)

            dataset_score = compute_total_asset_score(metrics, constraints_cfg)
            dataset_score = _sanitise(dataset_score, f"dataset@{idx}")
            dataset_scores.append(dataset_score)

            partial_metrics = combine_metrics(
                numeric_metrics, simple_override=simple_override
            )
            partial_score_raw = sum(dataset_scores) / max(1, len(dataset_scores))
            partial_score = _penalise_by_drawdown(partial_score_raw, partial_metrics)
            partial_score = _sanitise(partial_score, f"partial@{idx}")
            partial_objectives: Optional[Tuple[float, ...]] = (
                evaluate_objective_values(partial_metrics, objective_specs, non_finite_penalty)
                if multi_objective
                else None
            )
            trial.report(partial_score, step=idx)
            if trial.should_prune():
                cleaned_partial = _clean_metrics(partial_metrics)
                pruned_record = {
                    "trial": trial.number,
                    "params": params_for_record,
                    "metrics": cleaned_partial,
                    "datasets": dataset_metrics,
                    "score": partial_score,
                    "valid": cleaned_partial.get("Valid", True),
                    "dataset_key": dict(dataset_key_payload),
                    "pruned": True,
                    "skipped_datasets": list(skipped_dataset_records),
                }
                if partial_objectives is not None:
                    pruned_record["objective_values"] = list(partial_objectives)
                with results_lock:
                    results.append(pruned_record)
                trial.set_user_attr("score", float(partial_score))
                trial.set_user_attr("metrics", cleaned_partial)
                trial.set_user_attr(
                    "total_assets",
                    _safe_float(cleaned_partial.get("TotalAssets")),
                )
                trial.set_user_attr(
                    "liquidations",
                    _safe_float(cleaned_partial.get("Liquidations")),
                )
                if "Leverage" in cleaned_partial:
                    trial.set_user_attr("leverage", _safe_float(cleaned_partial.get("Leverage")))
                trial.set_user_attr("trades", _coerce_int(cleaned_partial.get("Trades")))
                trial.set_user_attr("valid", bool(cleaned_partial.get("Valid", True)))
                trial.set_user_attr("pruned", True)
                trial.set_user_attr("skipped_datasets", list(skipped_dataset_records))
                raise optuna.TrialPruned()

        min_volume_threshold = _resolve_min_volume_threshold()
        eligible_datasets: List[DatasetSpec] = []
        for dataset in selected_datasets:
            meets_volume, total_volume = _has_sufficient_volume(dataset, min_volume_threshold)
            if meets_volume:
                eligible_datasets.append(dataset)
                continue
            # Below volume threshold; record as skipped
            record = {
                "name": dataset.name,
                "meta": dataset.meta,
                "metrics": {},
                "skipped": True,
                "skip_reason": "volume_threshold",
                "skip_metric": total_volume,
                "skip_threshold": min_volume_threshold,
            }
            dataset_metrics.append(record)
            skipped_dataset_records.append(
                {
                    "name": dataset.name,
                    "timeframe": dataset.timeframe,
                    "htf_timeframe": dataset.htf_timeframe,
                    "total_volume": total_volume,
                    "min_volume": min_volume_threshold,
                }
            )
            continue

        selected_datasets = eligible_datasets

        parallel_enabled = dataset_jobs > 1 and len(selected_datasets) > 1
        if parallel_enabled and dataset_executor_pool is not None:
            if dataset_executor == "process":
                dataset_refs: Sequence[object] = [dataset.name for dataset in selected_datasets]
            else:
                dataset_refs = list(selected_datasets)

            futures = []
            for dataset, dataset_ref in zip(selected_datasets, dataset_refs):
                min_trades_requirement = _resolve_dataset_min_trades(
                    dataset,
                    constraints=constraints_cfg,
                    risk=risk,
                )
                futures.append(
                    dataset_executor_pool.submit(
                        _run_dataset_backtest_task,
                        dataset_ref,
                        params,
                        fees,
                        risk,
                        min_trades_requirement,
                    )
                )

            for idx, (dataset, future) in enumerate(zip(selected_datasets, futures), start=1):
                try:
                    metrics = future.result()
                except Exception:
                    for pending in futures[idx:]:
                        pending.cancel()
                    LOGGER.exception(
                        "Dataset execution failed (dataset=%s, timeframe=%s, htf=%s)",
                        dataset.name,
                        dataset.timeframe,
                        dataset.htf_timeframe,
                    )
                    raise
                    _consume_dataset(
                        idx, dataset, metrics, simple_override=simple_metrics_enabled
                    )
                except optuna.TrialPruned:
                    for pending in futures[idx:]:
                        pending.cancel()
                    raise
        else:
            for idx, dataset in enumerate(selected_datasets, start=1):
                try:
                    min_trades_requirement = _resolve_dataset_min_trades(
                        dataset,
                        constraints=constraints_cfg,
                        risk=risk,
                    )
                    # Delegate backtest to the helper which supports alternative engines
                    metrics = _run_dataset_backtest_task(
                        dataset,
                        params,
                        fees,
                        risk,
                        min_trades=min_trades_requirement,
                    )
                except Exception:
                    LOGGER.exception(
                        "Dataset execution failed (dataset=%s, timeframe=%s, htf=%s)",
                        dataset.name,
                        dataset.timeframe,
                        dataset.htf_timeframe,
                    )
                    raise
                _consume_dataset(
                    idx, dataset, metrics, simple_override=simple_metrics_enabled
                )

        aggregated = combine_metrics(
            numeric_metrics, simple_override=simple_metrics_enabled
        )
        if not aggregated:
            aggregated = {"Valid": False}
        if dataset_scores:
            score_raw = sum(dataset_scores) / len(dataset_scores)
        else:
            score_raw = non_finite_penalty
        score = _penalise_by_drawdown(score_raw, aggregated)
        score = _sanitise(score, "final")
        objective_values = (
            evaluate_objective_values(aggregated, objective_specs, non_finite_penalty)
            if multi_objective
            else None
        )

        cleaned_aggregated = _clean_metrics(aggregated)
        band_exit_detail = _band_exit_summary(aggregated)
        if band_exit_detail:
            band_exit_summary_text = _format_band_exit_summary(band_exit_detail)
            band_exit_mode = str(band_exit_detail.get("mode") or "NONE").upper()
            try:
                band_exit_mult = float(band_exit_detail.get("mult", 0.0))
            except (TypeError, ValueError):
                band_exit_mult = 0.0
        else:
            band_exit_detail = None
            band_exit_summary_text = "Band exit disabled"
            band_exit_mode = "NONE"
            band_exit_mult = 0.0

        cleaned_aggregated["BandExitSummary"] = band_exit_summary_text
        cleaned_aggregated["BandExitMode"] = band_exit_mode
        cleaned_aggregated["BandExitMult"] = float(band_exit_mult)
        for key in (
            "BandExitBBCount",
            "BandExitKCCount",
            "BandExitBBRatio",
            "BandExitKCRatio",
            "BandExitBBMult",
            "BandExitKCMult",
        ):
            cleaned_aggregated.pop(key, None)
        trial.set_user_attr("band_exit_summary", band_exit_detail)
        trial.set_user_attr("band_exit_summary_text", band_exit_summary_text)
        trial.set_user_attr("band_exit_mode", band_exit_mode)
        trial.set_user_attr("band_exit_mult", float(band_exit_mult))
        LOGGER.debug("Trial %s band exit: %s", trial.number, band_exit_summary_text)
        if aggregate_trade_payload:
            cleaned_aggregated["TradesList"] = list(aggregate_trade_payload)
        valid_status = bool(aggregated.get("Valid", bool(numeric_metrics)))

        anomaly_info: Optional[Dict[str, object]] = None

        trades_total = _coerce_int(cleaned_aggregated.get("Trades"))
        trade_anomaly = False
        if trades_total is not None:
            cleaned_aggregated["Trades"] = trades_total
            if trades_total < MIN_TRADES_ENFORCED:
                trade_anomaly = True
                score = non_finite_penalty
                valid_status = False
                cleaned_aggregated["Valid"] = False
                if multi_objective and objective_values is not None:
                    objective_values = tuple(non_finite_penalty for _ in objective_values)
                trade_info = {
                    "type": "trades_threshold",
                    "value": trades_total,
                    "threshold": MIN_TRADES_ENFORCED,
                }
                if anomaly_info is None:
                    anomaly_info = trade_info
                elif isinstance(anomaly_info, dict):
                    related = anomaly_info.setdefault("related", [])
                    if isinstance(related, list):
                        related.append(trade_info)

        record = {
            "trial": trial.number,
            "params": params_for_record,
            "metrics": cleaned_aggregated,
            "datasets": dataset_metrics,
            "score": score,
            "valid": valid_status,
            "dataset_key": dict(dataset_key_payload),
            "pruned": False,
            "skipped_datasets": list(skipped_dataset_records),
        }
        if anomaly_info is not None:
            record["anomaly"] = anomaly_info
        if objective_values is not None:
            record["objective_values"] = list(objective_values)
        with results_lock:
            results.append(record)
        trial.set_user_attr("score", float(score))
        trial.set_user_attr("metrics", cleaned_aggregated)
        trial.set_user_attr(
            "total_assets",
            _safe_float(cleaned_aggregated.get("TotalAssets")),
        )
        trial.set_user_attr(
            "liquidations",
            _safe_float(cleaned_aggregated.get("Liquidations")),
        )
        leverage_attr = cleaned_aggregated.get("Leverage")
        if leverage_attr is not None:
            trial.set_user_attr("leverage", _safe_float(leverage_attr))
        elif "leverage" in trial.params:
            trial.set_user_attr("leverage", _to_native(trial.params.get("leverage")))
        trial.set_user_attr("trades", _coerce_int(cleaned_aggregated.get("Trades")))
        trial.set_user_attr("valid", valid_status)
        trial.set_user_attr("pruned", False)
        trial.set_user_attr("skipped_datasets", list(skipped_dataset_records))
        trial.set_user_attr("min_trades_enforced", trade_anomaly)
        if multi_objective and objective_values is not None:
            return objective_values
        return score

    def _run_optuna(batch: int) -> None:
        if batch <= 0:
            return
        if GLOBAL_STOP_EVENT.is_set():
            LOGGER.info("Global stop flag detected; halting remaining Optuna trials.")
            return
        study.optimize(
            objective,
            n_trials=batch,
            n_jobs=n_jobs,
            show_progress_bar=False,
            callbacks=callbacks,
            gc_after_trial=True,
            catch=(sqlalchemy.exc.OperationalError, StorageInternalError),
        )

    use_llm = bool(llm_cfg.get("enabled"))
    llm_count = int(llm_cfg.get("count", 0)) if use_llm else 0
    llm_initial = int(llm_cfg.get("initial_trials", max(10, n_trials // 2))) if use_llm else 0
    llm_initial = max(0, min(llm_initial, n_trials))
    llm_insights: List[str] = []

    try:
        if use_llm and llm_count > 0 and 0 < llm_initial < n_trials:
            _run_optuna(llm_initial)
            try:
                generator = _resolve_llm_generator()
                llm_bundle: LLMSuggestions = generator(space, study.trials, llm_cfg)
                llm_insights = list(llm_bundle.insights)
                for candidate in llm_bundle.candidates[:llm_count]:
                    trial_params = _filter_basic_factor_params(
                        dict(candidate), enabled=use_basic_factors
                    )
                    trial_params = _ensure_channel_params(trial_params, space)
                    if not trial_params:
                        continue
                    trial_params.update(forced_params)
                    try:
                        study.enqueue_trial(trial_params, skip_if_exists=True)
                    except Exception as exc:
                        LOGGER.debug("Failed to enqueue LLM candidate %s: %s", candidate, exc)
            except Exception as exc:
                LOGGER.debug("LLM  ? ? ?: %s", exc)
            remaining = n_trials - llm_initial
            _run_optuna(remaining)
        else:
            _run_optuna(n_trials)
    finally:
        if dataset_executor_pool is not None:
            dataset_executor_pool.shutdown(wait=True, cancel_futures=True)
        if final_csv_path is not None:
            try:
                df = study.trials_dataframe()
            except Exception:
                df = None
            if df is not None:
                final_csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(final_csv_path, index=False)

    if llm_refresher is not None and llm_refresher.collected_insights:
        seen_insights = set(llm_insights)
        for insight in llm_refresher.collected_insights:
            if insight not in seen_insights:
                llm_insights.append(insight)
                seen_insights.add(insight)

    if not results:
        raise RuntimeError("No completed trials were produced during optimisation.")

    def _total_assets_key(record: Dict[str, object]) -> float:
        metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
        if not record.get("valid", True):
            return float("-inf")
        try:
            value = float(metrics.get("TotalAssets", float("-inf")))
        except (TypeError, ValueError):
            value = float("-inf")
        if not np.isfinite(value):
            return float("-inf")
        return value

    if multi_objective:
        best_record = max(results, key=lambda res: res.get("score", float("-inf")))
    else:
        best_record = max(results, key=_total_assets_key)
        if not np.isfinite(_total_assets_key(best_record)):
            best_trial = study.best_trial.number
            best_record = next(res for res in results if res["trial"] == best_trial)
    return {
        "study": study,
        "results": results,
        "best": best_record,
        "multi_objective": multi_objective,
        "storage": storage_meta,
        "basic_factor_profile": use_basic_factors,
        "llm_insights": llm_insights,
        "param_order": param_order,
    }


def merge_dicts(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(primary)
    merged.update({k: v for k, v in secondary.items() if v is not None})
    return merged










def _execute_single(
    args: argparse.Namespace,
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    risk_cfg: Dict[str, object],
    backtest_risk: Dict[str, object],
    argv: Optional[Sequence[str]] = None,
) -> None:
    if GLOBAL_STOP_EVENT.is_set():
        LOGGER.info("Global stop flag detected; skipping remaining execution.")
        return
    params_cfg = copy.deepcopy(params_cfg)
    backtest_cfg = copy.deepcopy(backtest_cfg)
    params_cfg.setdefault("space", {})
    backtest_cfg.setdefault("symbols", backtest_cfg.get("symbols", []))
    backtest_cfg.setdefault("timeframes", backtest_cfg.get("timeframes", []))

    search_cfg = _ensure_dict(params_cfg, "search")

    search_cfg.setdefault("dataset_executor", "thread")
    search_cfg.setdefault("allow_sqlite_parallel", False)

    batch_ctx = getattr(args, "_batch_context", None)
    if batch_ctx:
        suffix = batch_ctx.get("suffix") or batch_ctx.get("ltf_slug") or ""
        try:
            args.run_tag = _format_batch_value(
                batch_ctx.get("run_tag_template"),
                batch_ctx.get("base_run_tag"),
                suffix,
                batch_ctx,
            )
            base_study = batch_ctx.get("base_study_name")
            study_template = batch_ctx.get("study_template")
            if base_study or study_template:
                args.study_name = _format_batch_value(
                    study_template,
                    base_study,
                    suffix,
                    batch_ctx,
                )
        except ValueError as exc:
            raise ValueError(f"Batch templating failed: {exc}") from exc

    if args.n_trials is not None:
        search_cfg["n_trials"] = int(args.n_trials)

    if args.n_jobs is not None:
        try:
            search_cfg["n_jobs"] = max(1, int(args.n_jobs))
        except (TypeError, ValueError):
            search_cfg["n_jobs"] = DEFAULT_OPTUNA_JOBS

    if args.dataset_jobs is not None:
        try:
            search_cfg["dataset_jobs"] = max(1, int(args.dataset_jobs))
        except (TypeError, ValueError):
            search_cfg["dataset_jobs"] = 1

    if args.dataset_executor:
        search_cfg["dataset_executor"] = args.dataset_executor

    if args.dataset_start_method:
        search_cfg["dataset_start_method"] = args.dataset_start_method

    if getattr(args, "full_space", False):
        search_cfg["basic_factor_profile"] = False
    elif getattr(args, "basic_factors_only", False):
        search_cfg["basic_factor_profile"] = True

    if args.study_name:
        search_cfg["study_name"] = args.study_name

    if args.storage_url:
        search_cfg["storage_url"] = args.storage_url

    if args.storage_url_env:
        search_cfg["storage_url_env"] = args.storage_url_env

    if getattr(args, "allow_sqlite_parallel", False):
        search_cfg["allow_sqlite_parallel"] = True

    if getattr(args, "force_sqlite_serial", False):
        search_cfg["allow_sqlite_parallel"] = False
        search_cfg["force_sqlite_serial"] = True
        LOGGER.info("CLI flag --force-sqlite-serial detected; Optuna n_jobs will be fixed at 1.")

    if args.pruner:
        search_cfg["pruner"] = args.pruner

    overrides_cfg = params_cfg.get("overrides")
    if overrides_cfg is None:
        forced_params: Dict[str, object] = {}
    elif isinstance(overrides_cfg, Mapping):
        forced_params = dict(overrides_cfg)
    else:
        forced_params = {}
    for name in _collect_tokens(args.enable):
        forced_params[name] = True
    for name in _collect_tokens(args.disable):
        forced_params[name] = False

    symbol_choices = list(dict.fromkeys(backtest_cfg.get("symbols") or ([params_cfg.get("symbol")] if params_cfg.get("symbol") else [])))

    selected_symbol = args.symbol or params_cfg.get("symbol") or (symbol_choices[0] if symbol_choices else None)
    selected_timeframe: Optional[str] = args.timeframe
    timeframe_overridden = args.timeframe is not None
    all_timeframes_requested = False
    mix_values = [token for token in _parse_ltf_choice_value(getattr(args, "timeframe_mix", None)) if token]

    if (
        not timeframe_overridden
        and not getattr(args, "timeframe_grid", None)
        and batch_ctx is None
        and not mix_values
    ):
        prompt_selection = _prompt_ltf_selection()
        if prompt_selection.mix:
            args.timeframe_mix = prompt_selection.label or MERGED_TIMEFRAME_LABEL
            mix_values = _normalise_timeframe_mix_argument(args)
            selected_timeframe = getattr(prompt_selection, "timeframe", None) or getattr(args, "timeframe", None)
            timeframe_overridden = True
        elif prompt_selection.use_all:
            all_timeframes_requested = True
            selected_timeframe = None
            timeframe_overridden = False
        else:
            selected_timeframe = prompt_selection.timeframe
            timeframe_overridden = True

    if selected_symbol:
        params_cfg["symbol"] = selected_symbol
        backtest_cfg["symbols"] = [selected_symbol]
    params_cfg.pop("timeframe_components", None)
    params_cfg.pop("timeframe_pool", None)
    if timeframe_overridden and selected_timeframe and not mix_values:
        params_cfg["timeframe"] = selected_timeframe
        backtest_cfg["timeframes"] = [selected_timeframe]
        forced_params["timeframe"] = selected_timeframe
        _apply_ltf_override_to_datasets(backtest_cfg, selected_timeframe)
        _enforce_forced_timeframe_constraints(params_cfg, search_cfg, selected_timeframe)
    elif mix_values:
        mix_set = set(mix_values)
        if mix_set == set(MERGED_TIMEFRAME_COMPONENTS):
            params_cfg["timeframe"] = MERGED_TIMEFRAME_LABEL
            params_cfg["timeframe_pool"] = mix_values
            forced_params.pop("timeframe", None)
            LOGGER.info("Randomly sampling one timeframe from merged pool: %s", ", ".join(mix_values))
        else:
            mix_label = ",".join(mix_values)
            params_cfg["timeframe"] = mix_label
            params_cfg["timeframe_components"] = mix_values
            forced_params["timeframe"] = mix_label
            LOGGER.info("%s mix forced for this run.", mix_label)
            _enforce_forced_timeframe_constraints(params_cfg, search_cfg, mix_label)

    backtest_periods = backtest_cfg.get("periods") or []
    params_backtest = _ensure_dict(params_cfg, "backtest")
    if args.start or args.end:
        start = args.start or params_backtest.get("from") or (backtest_periods[0]["from"] if backtest_periods else None)
        end = args.end or params_backtest.get("to") or (backtest_periods[0]["to"] if backtest_periods else None)
        if start and end:
            params_backtest["from"] = start
            params_backtest["to"] = end
            backtest_cfg["periods"] = [{"from": start, "to": end}]
    if params_backtest.get("from") and params_backtest.get("to"):
        backtest_cfg["periods"] = [{"from": params_backtest["from"], "to": params_backtest["to"]}]

    llm_cfg = _ensure_dict(params_cfg, "llm")
    if args.llm is not None:
        llm_enabled = args.llm

    if llm_cfg.get("enabled"):
        api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))
        if not llm_cfg.get("api_key") and not os.environ.get(api_key_env):
            LOGGER.info(
                "LLM integration enabled; populate %s, llm.api_key, or llm.api_key_file before optimisation to activate Gemini support.",
                api_key_env,
            )

    constraints_raw = params_cfg.get("constraints")
    if isinstance(constraints_raw, Mapping):
        constraints_cfg: Dict[str, object] = dict(constraints_raw)
    else:
        constraints_cfg = {}
    if not constraints_cfg:
        backtest_constraints = backtest_cfg.get("constraints")
        if isinstance(backtest_constraints, Mapping):
            constraints_cfg = dict(backtest_constraints)

    def _resolve_simple_metrics_flag() -> bool:
        for candidate in (
            forced_params.get("simpleMetricsOnly"),
            forced_params.get("simpleProfitOnly"),
            risk_cfg.get("simpleMetricsOnly"),
            risk_cfg.get("simpleProfitOnly"),
            backtest_risk.get("simpleMetricsOnly"),
            backtest_risk.get("simpleProfitOnly"),
        ):
            coerced = _coerce_bool_or_none(candidate)
            if coerced is not None:
                return coerced
        return False

    simple_metrics_state = _resolve_simple_metrics_flag()
    if simple_metrics_state:
        LOGGER.info("Simple metrics mode enabled; limiting reports to headline columns.")
    else:
        for key in ("simpleMetricsOnly", "simpleProfitOnly"):
            forced_params.pop(key, None)
            risk_cfg.pop(key, None)
            backtest_risk.pop(key, None)
        LOGGER.info("Full metrics mode enabled.")

    global simple_metrics_enabled
    simple_metrics_enabled = simple_metrics_state

    if "useBandExit" in forced_params:
        band_exit_flag = _coerce_bool_or_none(forced_params.get("useBandExit"))
        band_exit_min = forced_params.get("bandExitMinBars")
        if band_exit_flag is True:
            if band_exit_min is not None:
                LOGGER.info(
                    "BB/KC band exit ON (bandExitMinBars=%s).",
                    band_exit_min,
                )
            else:
                LOGGER.info("BB/KC band exit ON.")
        elif band_exit_flag is False:
            LOGGER.info("BB/KC band exit OFF.")
        elif band_exit_min is not None:
            LOGGER.info(
                "bandExitMinBars=%s supplied but useBandExit is not True; ignoring override.",
                band_exit_min,
            )

    params_cfg["overrides"] = forced_params

    datasets = prepare_datasets(params_cfg, backtest_cfg, args.data)
    if not datasets:
        raise RuntimeError("No datasets prepared for optimisation")

    auto_workers = getattr(args, "auto_workers", False)
    if auto_workers:
        available_cpu = max(multiprocessing.cpu_count(), 1)
        search_cfg = _ensure_dict(params_cfg, "search")
        current_n_jobs = int(search_cfg.get("n_jobs", 1) or 1)
        if current_n_jobs <= 1 and available_cpu > 1:
            recommended_trials = min(available_cpu, max(2, available_cpu // 2))
            if recommended_trials > 1:
                search_cfg["n_jobs"] = recommended_trials
                LOGGER.info(
                    "Auto workers: Optuna n_jobs=%d (available CPU=%d)",
                    recommended_trials,
                    available_cpu,
                )
        dataset_jobs_current = int(search_cfg.get("dataset_jobs", 1) or 1)
        if len(datasets) > 1 and dataset_jobs_current <= 1:
            max_parallel = min(len(datasets), max(1, available_cpu))
            if max_parallel > 1:
                search_cfg["dataset_jobs"] = max_parallel
                LOGGER.info(
                    "Auto workers: dataset_jobs=%d (datasets=%d, CPU=%d)",
                    max_parallel,
                    len(datasets),
                    available_cpu,
                )
                adjusted_n_jobs = max(1, available_cpu // max_parallel)
                if adjusted_n_jobs < current_n_jobs:
                    search_cfg["n_jobs"] = adjusted_n_jobs
                    LOGGER.info(
                        "Auto workers: Optuna n_jobs=%d (dataset  )",
                        adjusted_n_jobs,
                    )


    output_dir, tag_info = _resolve_output_directory(args.output, datasets, params_cfg, args.run_tag)
    _configure_logging(output_dir / "logs")
    LOGGER.info("Writing outputs to %s", output_dir)

    fees = merge_dicts(params_cfg.get("fees", {}), backtest_cfg.get("fees", {}))
    risk = merge_dicts(params_cfg.get("risk", {}), backtest_cfg.get("risk", {}))

    objectives_raw = params_cfg.get("objectives")
    if not objectives_raw:
        objectives_raw = params_cfg.get("objective")
    if objectives_raw is None:
        objectives_raw = []
    if isinstance(objectives_raw, (list, tuple)):
        objectives_config: List[object] = list(objectives_raw)
    elif objectives_raw:
        objectives_config = [objectives_raw]
    else:
        objectives_config = []
    objective_specs = normalise_objectives(objectives_config)
    space_hash = _space_hash(params_cfg.get("space", {}))
    search_cfg = _ensure_dict(params_cfg, "search")
    if not search_cfg.get("study_name"):
        search_cfg["study_name"] = _default_study_name(params_cfg, datasets, space_hash)
    primary_for_regime = _pick_primary_dataset(datasets)
    regime_summary = detect_regime_label(primary_for_regime.df)

    resume_bank_path = args.resume_from
    if resume_bank_path is None:
        resume_bank_path = _discover_bank_path(output_dir, tag_info, space_hash)

    search_cfg_effective = params_cfg.get("search", {})
    basic_flag = _coerce_bool_or_none(search_cfg_effective.get("basic_factor_profile"))
    if basic_flag is None:
        basic_flag = _coerce_bool_or_none(search_cfg_effective.get("use_basic_factors"))
    basic_filter_enabled = True if basic_flag is None else basic_flag

    seed_trials = _load_seed_trials(
        resume_bank_path,
        params_cfg.get("space", {}),
        space_hash,
        regime_label=regime_summary.label,
        basic_filter_enabled=basic_filter_enabled,
    )

    study_storage = _resolve_study_storage(params_cfg, datasets)
    _apply_study_registry_defaults(search_cfg, study_storage)

    storage_env_key = str(search_cfg.get("storage_url_env") or DEFAULT_STORAGE_ENV_KEY)
    if not storage_env_key:
        storage_env_key = DEFAULT_STORAGE_ENV_KEY
    search_cfg["storage_url_env"] = storage_env_key

    if not search_cfg.get("storage_url"):
        search_cfg["storage_url"] = DEFAULT_SQLITE_STORAGE_URL

    storage_env_value = os.getenv(storage_env_key) if storage_env_key else None
    effective_storage_url = str(
        storage_env_value or search_cfg.get("storage_url") or ""
    )
    using_sqlite = effective_storage_url.startswith("sqlite:///")
    is_postgres = effective_storage_url.startswith(POSTGRES_PREFIXES)
    masked_storage_url = _mask_storage_url(effective_storage_url) if effective_storage_url else ""
    if storage_env_value:
        storage_source = f"env:{storage_env_key}"
    elif effective_storage_url:
        storage_source = "config"
    else:
        storage_source = "default"
    backend_label = (
        "PostgreSQL"
        if is_postgres
        else "SQLite"
        if using_sqlite
        else "External RDB"
        if effective_storage_url
        else "SQLite (default)"
    )
    LOGGER.info(
        "Optuna storage backend %s (source=%s, url=%s)",
        backend_label,
        storage_source,
        masked_storage_url or "<none>",
    )

    default_optuna_jobs = (
        SQLITE_SAFE_OPTUNA_JOBS if using_sqlite else DEFAULT_OPTUNA_JOBS
    )
    default_dataset_jobs = (
        SQLITE_SAFE_DATASET_JOBS if using_sqlite else DEFAULT_DATASET_JOBS
    )

    if not search_cfg.get("n_jobs"):
        search_cfg["n_jobs"] = default_optuna_jobs
    if not search_cfg.get("dataset_jobs"):
        search_cfg["dataset_jobs"] = default_dataset_jobs

    if using_sqlite and not search_cfg.get("allow_sqlite_parallel"):
        search_cfg.setdefault("force_sqlite_serial", True)

    trials_log_dir = output_dir / "trials"

    optimisation = optimisation_loop(
        datasets,
        params_cfg,
        objective_specs,
        fees,
        risk,
        constraints_cfg,
        search_cfg,
        forced_params,
        study_storage=study_storage,
        space_hash=space_hash,
        seed_trials=seed_trials,
        log_dir=trials_log_dir,
        force_sqlite_serial=getattr(args, "force_sqlite_serial", False),
    )

    llm_insights_logged: List[str] = []
    raw_llm_insights = optimisation.get("llm_insights")
    if isinstance(raw_llm_insights, (list, tuple)):
        llm_insights_logged = [str(item) for item in raw_llm_insights if str(item).strip()]
    if llm_insights_logged:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        insight_file = logs_dir / "gemini_insights.md"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
        with insight_file.open("a", encoding="utf-8") as handle:
            handle.write(f"## {timestamp}\n")
            for insight in llm_insights_logged:
                handle.write(f"- {insight}\n")
            handle.write("\n")
        for insight in llm_insights_logged:
            LOGGER.info("[Gemini Insight] %s", insight)

    wf_cfg = (
        params_cfg.get("walk_forward")
        or backtest_cfg.get("walk_forward")
        or {"train_bars": 5000, "test_bars": 2000, "step": 2000}
    )

    study = optimisation.get("study")
    if study is not None:
        write_trials_dataframe(
            study,
            output_dir,
            param_order=optimisation.get("param_order"),
        )
    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    def _resolve_record_dataset(record: Dict[str, object]) -> Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]:
        key_info = record.get("dataset_key") if isinstance(record, dict) else None
        if isinstance(key_info, dict):
            candidate_key = (key_info.get("timeframe"), key_info.get("htf_timeframe"))
            if candidate_key in dataset_groups:
                return candidate_key, dataset_groups[candidate_key]
        return _select_datasets_for_params(
            params_cfg,
            dataset_groups,
            timeframe_groups,
            default_key,
            record.get("params", {}),
        )

    best_record = optimisation["best"]
    best_key, best_group = _resolve_record_dataset(best_record)
    primary_dataset = _pick_primary_dataset(best_group)

    wf_min_trades_override = _coerce_min_trades_value(wf_cfg.get("min_trades"))

    def _min_trades_for_dataset(dataset: DatasetSpec) -> Optional[int]:
        return _resolve_dataset_min_trades(
            dataset,
            constraints=constraints_cfg,
            risk=risk,
            explicit=wf_min_trades_override,
        )

    primary_min_trades = _min_trades_for_dataset(primary_dataset)
    raw_space = params_cfg.get("space", {})
    space = raw_space if isinstance(raw_space, Mapping) else {}
    param_order = optimisation.get("param_order")

    def _ordered_params_view(raw_params: object) -> Dict[str, object]:
        if isinstance(raw_params, Mapping):
            patched = _ensure_channel_params(dict(raw_params), space)
            return _order_mapping(patched, param_order)
        return {}

    wf_summary = run_walk_forward(
        primary_dataset.df,
        best_record["params"],
        fees,
        risk,
        train_bars=int(wf_cfg.get("train_bars", 5000)),
        test_bars=int(wf_cfg.get("test_bars", 2000)),
        step=int(wf_cfg.get("step", 2000)),
        htf_df=primary_dataset.htf,
        min_trades=primary_min_trades,
    )

    cv_summary = None
    cv_manifest: Dict[str, object] = {}
    cv_choice = (args.cv or str(params_cfg.get("validation", {}).get("type", ""))).lower()
    if cv_choice == "purged-kfold":
        cv_k = args.cv_k or int(params_cfg.get("validation", {}).get("k", 5))
        cv_embargo = args.cv_embargo or float(params_cfg.get("validation", {}).get("embargo", 0.01))
        cv_summary = run_purged_kfold(
            primary_dataset.df,
            best_record["params"],
            fees,
            risk,
            k=cv_k,
            embargo=cv_embargo,
            htf_df=primary_dataset.htf,
            min_trades=primary_min_trades,
        )
        wf_summary["purged_kfold"] = cv_summary
        cv_manifest = {"type": "purged-kfold", "k": cv_k, "embargo": cv_embargo}
    elif cv_choice and cv_choice != "none":
        cv_manifest = {"type": cv_choice}

    def _total_assets_value(record: Dict[str, object]) -> float:
        metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
        if not record.get("valid", True):
            return float("-inf")
        try:
            value = float(metrics.get("TotalAssets", float("-inf")))
        except (TypeError, ValueError):
            value = float("-inf")
        if not np.isfinite(value):
            return float("-inf")
        return value

    best_band_exit = _band_exit_summary(best_record.get("metrics", {}))
    best_band_exit_text = (
        _format_band_exit_summary(best_band_exit) if best_band_exit else "Band exit disabled"
    )
    best_band_exit_mode = (
        str(best_band_exit.get("mode") or "NONE").upper() if best_band_exit else "NONE"
    )
    try:
        best_band_exit_mult = float(best_band_exit.get("mult", 0.0)) if best_band_exit else 0.0
    except (TypeError, ValueError):
        best_band_exit_mult = 0.0
    candidate_summaries = [
        {
            "trial": best_record["trial"],
            "score": best_record.get("score"),
            "oos_mean": wf_summary.get("oos_mean"),
            "params": _ordered_params_view(best_record.get("params")),
            "timeframe": best_key[0],
            "htf_timeframe": best_key[1],
            "band_exit_summary": best_band_exit_text,
            "band_exit_mode": best_band_exit_mode,
            "band_exit_mult": best_band_exit_mult,
            **(
                {
                    "band_exit": best_band_exit,
                }
                if best_band_exit
                else {}
            ),
        }
    ]

    top_k = args.top_k or int(params_cfg.get("search", {}).get("top_k", 0))
    if top_k > 0:
        sorted_results = sorted(optimisation["results"], key=_total_assets_value, reverse=True)
        best_oos = wf_summary.get("oos_mean", float("-inf"))
        wf_cache = {best_record["trial"]: wf_summary}
        for record in sorted_results[:top_k]:
            if record["trial"] == best_record["trial"]:
                continue
            candidate_key, candidate_group = _resolve_record_dataset(record)
            candidate_dataset = _pick_primary_dataset(candidate_group)
            candidate_min_trades = _min_trades_for_dataset(candidate_dataset)

            candidate_wf = run_walk_forward(
                candidate_dataset.df,
                record["params"],
                fees,
                risk,
                train_bars=int(wf_cfg.get("train_bars", 5000)),
                test_bars=int(wf_cfg.get("test_bars", 2000)),
                step=int(wf_cfg.get("step", 2000)),
                htf_df=candidate_dataset.htf,
                min_trades=candidate_min_trades,
            )
            wf_cache[record["trial"]] = candidate_wf
            candidate_band_exit = _band_exit_summary(record.get("metrics", {}))
            candidate_payload = {
                "trial": record["trial"],
                "score": record.get("score"),
                "oos_mean": candidate_wf.get("oos_mean"),
                "params": _ordered_params_view(record.get("params")),
                "timeframe": candidate_key[0],
                "htf_timeframe": candidate_key[1],
            }
            if candidate_band_exit:
                candidate_payload["band_exit"] = candidate_band_exit
                candidate_payload["band_exit_summary"] = _format_band_exit_summary(candidate_band_exit)
                candidate_payload["band_exit_mode"] = str(
                    candidate_band_exit.get("mode") or "NONE"
                ).upper()
                try:
                    candidate_payload["band_exit_mult"] = float(candidate_band_exit.get("mult", 0.0))
                except (TypeError, ValueError):
                    candidate_payload["band_exit_mult"] = 0.0
            else:
                candidate_payload["band_exit_summary"] = "Band exit disabled"
                candidate_payload["band_exit"] = None
                candidate_payload["band_exit_mode"] = "NONE"
                candidate_payload["band_exit_mult"] = 0.0
            candidate_summaries.append(candidate_payload)
            candidate_oos = candidate_wf.get("oos_mean", float("-inf"))
            if candidate_oos > best_oos or (
                candidate_oos == best_oos
                and _total_assets_value(record) > _total_assets_value(best_record)
            ):
                best_oos = candidate_oos
                best_record = record
                best_key = candidate_key
                best_group = candidate_group
                primary_dataset = candidate_dataset
                wf_summary = candidate_wf
        optimisation["best"] = best_record

    best_band_exit = _band_exit_summary(best_record.get("metrics", {}))
    best_band_exit_text = (
        _format_band_exit_summary(best_band_exit) if best_band_exit else "Band exit disabled"
    )
    best_band_exit_mode = (
        str(best_band_exit.get("mode") or "NONE").upper() if best_band_exit else "NONE"
    )
    try:
        best_band_exit_mult = float(best_band_exit.get("mult", 0.0)) if best_band_exit else 0.0
    except (TypeError, ValueError):
        best_band_exit_mult = 0.0
    LOGGER.info("Best trial %s band exit: %s", best_record["trial"], best_band_exit_text)

    candidate_summaries[0] = {
        "trial": best_record["trial"],
        "score": best_record.get("score"),
        "oos_mean": wf_summary.get("oos_mean"),
        "params": _ordered_params_view(best_record.get("params")),
        "timeframe": best_key[0],
        "htf_timeframe": best_key[1],
        "band_exit_summary": best_band_exit_text,
        "band_exit_mode": best_band_exit_mode,
        "band_exit_mult": best_band_exit_mult,
        "band_exit": best_band_exit,
    }

    wf_summary["candidates"] = candidate_summaries

    trial_index = {record["trial"]: record for record in optimisation["results"]}
    bank_entries: List[Dict[str, object]] = []
    for item in candidate_summaries:
        trial_record = trial_index.get(item["trial"], {})
        filtered_params = _filter_basic_factor_params(
            item.get("params") or {}, enabled=optimisation.get("basic_factor_profile", True)
        )
        filtered_params = _ensure_channel_params(filtered_params, space)
        ordered_params = _order_mapping(filtered_params, param_order)
        raw_metrics = trial_record.get("metrics") if isinstance(trial_record, dict) else {}
        if isinstance(raw_metrics, Mapping):
            metrics_payload: object = _order_mapping(
                raw_metrics,
                None,
                priority=("TotalAssets", "MaxDD"),
            )
        else:
            metrics_payload = raw_metrics
        band_exit_info = (
            _band_exit_summary(raw_metrics) if isinstance(raw_metrics, Mapping) else None
        )
        entry = {
            "trial": item["trial"],
            "score": item.get("score"),
            "oos_mean": item.get("oos_mean"),
            "params": ordered_params,
            "metrics": metrics_payload,
            "timeframe": item.get("timeframe"),
            "htf_timeframe": item.get("htf_timeframe"),
            "band_exit_summary": item.get("band_exit_summary"),
            "band_exit_mode": item.get("band_exit_mode"),
            "band_exit_mult": item.get("band_exit_mult"),
        }
        if band_exit_info:
            entry["band_exit"] = band_exit_info
            entry.setdefault("band_exit_mode", str(band_exit_info.get("mode") or "NONE").upper())
            try:
                entry.setdefault("band_exit_mult", float(band_exit_info.get("mult", 0.0)))
            except (TypeError, ValueError):
                entry.setdefault("band_exit_mult", 0.0)
        if cv_summary:
            entry["cv_mean"] = cv_summary.get("mean")
        bank_entries.append(entry)

    bank_payload = _build_bank_payload(
        tag_info=tag_info,
        space_hash=space_hash,
        entries=bank_entries,
        regime_summary=regime_summary,
    )

    validation_manifest = dict(cv_manifest)
    if cv_summary:
        validation_manifest["summary"] = cv_summary

    storage_meta = optimisation.get("storage", {}) or {}
    _register_study_reference(
        study_storage,
        storage_meta=storage_meta,
        study_name=str(search_cfg.get("study_name")) if search_cfg.get("study_name") else None,
    )
    sanitised_storage_meta = _sanitise_storage_meta(storage_meta)
    registry_dir = _study_registry_dir(study_storage) if study_storage else None
    registry_dir_str = (
        str(registry_dir) if registry_dir and registry_dir.exists() else None
    )
    search_manifest = copy.deepcopy(params_cfg.get("search", {}))
    if "storage_url" in search_manifest:
        url_value = search_manifest.get("storage_url")
        if url_value and not str(url_value).startswith("sqlite:///"):
            search_manifest["storage_url"] = "***redacted***"

    manifest = {
        "created_at": _utcnow_isoformat(),
        "run": tag_info,
        "space_hash": space_hash,
        "symbol": params_cfg.get("symbol"),
        "fees": fees,
        "risk": risk,
        "objectives": [spec.__dict__ for spec in objective_specs],
        "search": search_manifest,
        "basic_factor_profile": optimisation.get("basic_factor_profile", True),
        "resume_bank": str(resume_bank_path) if resume_bank_path else None,
        "study_storage": storage_meta.get("path") if storage_meta.get("backend") == "sqlite" else None,
        "study_registry": registry_dir_str,
        "regime": regime_summary.__dict__,
        "cli": list(argv or []),
    }
    if sanitised_storage_meta:
        manifest["storage"] = sanitised_storage_meta
    if validation_manifest:
        manifest["validation"] = validation_manifest

    _write_manifest(output_dir, manifest=manifest)
    write_bank_file(output_dir, bank_payload)
    (output_dir / "seed.yaml").write_text(
        yaml.safe_dump(
            {
                "params": params_cfg,
                "backtest": backtest_cfg,
                "forced_params": forced_params,
            },
            sort_keys=False,
        )
    )

    generate_reports(
        optimisation["results"],
        optimisation["best"],
        wf_summary,
        objective_specs,
        output_dir,
        param_order=optimisation.get("param_order"),
    )

    LOGGER.info("Run complete. Outputs saved to %s", output_dir)


def execute(args: argparse.Namespace, argv: Optional[Sequence[str]] = None) -> None:
    """Execute one or more optimisation runs based on CLI arguments."""

    GLOBAL_STOP_EVENT.clear()

    global backtest_cfg
    params_cfg = normalize_tf(load_yaml(args.params))
    backtest_cfg = normalize_tf(load_yaml(args.backtest))

    risk_cfg = _ensure_dict(params_cfg, "risk")
    backtest_risk = _ensure_dict(backtest_cfg, "risk")
    if args.leverage is not None:
        risk_cfg["leverage"] = args.leverage
        backtest_risk["leverage"] = args.leverage
    if args.qty_pct is not None:
        risk_cfg["qty_pct"] = args.qty_pct
        backtest_risk["qty_pct"] = args.qty_pct

    auto_list: List[str] = []

    def _load_top_list() -> List[str]:
        return fetch_top_usdt_perp_symbols(
            limit=50,
            exclude_symbols=["BUSDUSDT", "USDCUSDT"],
            exclude_keywords=["UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "5L", "5S"],
            min_price=0.002,
        )

    if args.list_top50:
        auto_list = _load_top_list()
        import csv

        reports_dir = DEFAULT_REPORT_ROOT
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "top50_usdt_perp.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "symbol"])
            for index, symbol in enumerate(auto_list, start=1):
                writer.writerow([index, symbol])
        print("Saved: reports/top50_usdt_perp.csv")
        print("\n== USDT-Perp 24h ??? 50 ==")
        for index, symbol in enumerate(auto_list, start=1):
            print(f"{index:2d}. {symbol}")
        print("\n?? 7??:  python -m optimize.run --pick-top50 7")
        print("     ??  python -m optimize.run --pick-symbol BINANCE:ETHUSDT")
        return

    selected_symbol = ""
    if args.pick_symbol:
        selected_symbol = args.pick_symbol.strip()
    elif args.pick_top50:
        auto_list = auto_list or _load_top_list()
        if 1 <= args.pick_top50 <= len(auto_list):
            selected_symbol = auto_list[args.pick_top50 - 1]
        else:
            print("\n[ERROR] --pick-top50 ??? ??? (1~50).")
            return
    elif args.symbol:
        selected_symbol = args.symbol.strip()
    else:
        print("\n[ERROR] ???? ????")
        print("   ?? ?50 :       python -m optimize.run --list-top50")
        print("       7??(??:      python -m optimize.run --pick-top50 7")
        print("        ??         python -m optimize.run --pick-symbol BINANCE:ETHUSDT")
        return

    print(f"[INFO] 선택된 심볼: {selected_symbol}")

    args.symbol = selected_symbol
    params_cfg["symbol"] = selected_symbol
    backtest_cfg["symbols"] = [selected_symbol]

    datasets = backtest_cfg.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            if isinstance(dataset, dict):
                dataset["symbol"] = selected_symbol

    base_symbol: Optional[str]
    if args.symbol:
        base_symbol = args.symbol
    else:
        base_symbol = params_cfg.get("symbol") if params_cfg else None
        if not base_symbol:
            symbols = backtest_cfg.get("symbols") if isinstance(backtest_cfg, dict) else None
            if isinstance(symbols, list) and symbols:
                first = symbols[0]
                if isinstance(first, dict):
                    base_symbol = (
                        first.get("alias")
                        or first.get("name")
                        or first.get("symbol")
                        or first.get("id")
                    )
                else:
                    base_symbol = str(first)

    symbol_text = str(base_symbol) if base_symbol else "study"
    plan = _prepare_timeframe_execution_plan(
        args,
        params_cfg,
        backtest_cfg,
        symbol_text=symbol_text,
    )

    if plan.mix_values:
        LOGGER.info("? LTF ? ????: %s", ", ".join(plan.mix_values))

    if not plan.combos and plan.batches:
        batch = plan.batches[0]
        if batch.context:
            setattr(batch.args, "_batch_context", batch.context)
        _execute_single(batch.args, params_cfg, backtest_cfg, risk_cfg, backtest_risk, argv)
        return

    total = len(plan.combos) or len(plan.batches)

    for idx, batch in enumerate(plan.batches, start=1):
        context = batch.context or {}
        ltf = context.get("ltf") or getattr(batch.args, "timeframe", None)
        index = context.get("index", idx)
        if GLOBAL_STOP_EVENT.is_set():
            LOGGER.info("???? ?? ?? ??? ??????")
            break
        batch_args = batch.args
        if batch.context:
            batch_args._batch_context = batch.context  # type: ignore[attr-defined]
        LOGGER.info(
            "(%d/%d) LTF=%s  ???",
            index,
            total,
            ltf,
        )
        _execute_single(batch_args, params_cfg, backtest_cfg, risk_cfg, backtest_risk, argv)
        if GLOBAL_STOP_EVENT.is_set():
            LOGGER.info("???? ?? ??? ????????")
            break

    _export_for_github(REPO_ROOT)




def _export_for_github(project_root: Path):
    """Exports the project to a new directory for GitHub, excluding specified folders."""
    export_dir_name = "github_export"
    export_path = project_root / export_dir_name
    
    # directories and files to ignore
    ignore_patterns = (
        ".venv",
        "data",
        "reports",
        "studies",
        "__pycache__",
        "backtest",
        export_dir_name,
        ".git",
        "*.pyc",
        "*.log",
        "*.db",
        "*.jsonl",
    )

    if export_path.exists():
        shutil.rmtree(export_path)
        
    shutil.copytree(project_root, export_path, ignore=shutil.ignore_patterns(*ignore_patterns))

    # Add the export directory to the main .gitignore
    gitignore_path = project_root / ".gitignore"
    with open(gitignore_path, "r+", encoding="utf-8") as f:
        content = f.read()
        if export_dir_name not in content:
            f.write(f"\n# Ignore github export directory\n/{export_dir_name}/\n")

    LOGGER.info(f"Project exported for GitHub to {export_path}")




if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()




