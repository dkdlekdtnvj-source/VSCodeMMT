"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
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
from sqlalchemy import event
from sqlalchemy.engine import make_url
from optuna.trial import TrialState


from datafeed.cache import DataCache
from optimize.common import normalize_tf
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
    DEFAULT_POSTGRES_STORAGE_URL,
    DEFAULT_REPORT_ROOT,
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


HTF_ENABLED = False

# 테스트나 다른 모듈에서 ``optimize.run.backtest_cfg`` 를 직접 주입할 수 있도록
# 모듈 전역 백테스트 설정 딕셔너리를 초기화합니다. ``execute`` 함수는 실제 실행
# 시점에 해당 값을 최신 설정으로 갱신합니다.
backtest_cfg: Dict[str, object] = {}

# ``run_backtest`` 는 전략 엔진의 기본 구현을 가리키지만, ``optimize.run`` 모듈에서
# monkeypatch 로 대체된 함수가 있다면 이를 따라가도록 별도 참조를 유지합니다.
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
    """콘솔에서 백틱(`) 키 입력을 감지해 최적화 중단을 요청합니다."""

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
                LOGGER.warning(
                    "사용자 중지 요청을 감지했습니다. 현재 트라이얼 종료 후 최적화를 중단하고 보고서를 생성합니다."
                )
                self._notified = True
        try:
            study.stop()
        except Exception:
            pass


def _create_triple_backtick_stopper() -> Optional[_TripleBacktickStopper]:
    """필요 시 작업 중지 리스너를 생성하고 활성화합니다."""

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
        LOGGER.info(
            "작업중지 키가 활성화되었습니다. 콘솔에서 백틱(`) 키를 빠르게 세 번 입력 후 Enter를 누르면 현재 작업을 중단하고 즉시 보고서를 생성합니다."
        )
        STOP_INSTRUCTION_SHOWN = True

    return stopper

def fetch_top_usdt_perp_symbols(
    limit: int = 50,
    exclude_symbols: Optional[Sequence[str]] = None,
    exclude_keywords: Optional[Sequence[str]] = None,
    min_price: Optional[float] = None,
) -> List[str]:
    """Binance USDT-M Perp 선물에서 24h quote volume 상위 심볼을 반환합니다."""

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
        if quote_volume is None:
            base_volume = ticker.get("baseVolume") or 0
            last_price = ticker.get("last") or 0
            quote_volume = base_volume * last_price

        try:
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
# at the call site (see optimize/alternative_engine.py).
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
    """SQLite 잠금 오류 발생 시 짧게 재시도하며 파라미터를 샘플링합니다."""

    return sample_parameters(trial, space)


# 단순 메트릭 계산 경로 사용 여부 (CLI 인자/설정으로 갱신됩니다).
simple_metrics_enabled: bool = False


_INITIAL_BALANCE_KEYS = (
    "InitialCapital",
    "InitialEquity",
    "InitialBalance",
    "StartingBalance",
)


# 기본 팩터 최적화에 사용할 파라미터 키 집합입니다.
# 복잡한 보호 장치·부가 필터 대신 핵심 진입 로직과 직접 관련된 항목만 남겨
# 탐색 공간을 크게 줄이고 수렴 속도를 높입니다.
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
    "compatMode",
    "autoThresholdScale",
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
    "useChandelierExit",
    "chandelierLen",
    "chandelierMult",
    "useSarExit",
    "sarStart",
    "sarIncrement",
    "sarMaximum",
    # Risk & capital controls that must remain active even in basic profile mode
    "leverage",
    "chart_tf",
    "entry_tf",
    "use_htf",
    "htf_tf",
    "fixedStopPct",
    "atrStopLen",
    "atrStopMult",
    "usePyramiding",
}


TRIAL_STATE_LABELS = {
    "COMPLETE": "완료",
    "PRUNED": "중단",
    "FAIL": "실패",
    "RUNNING": "실행중",
    "WAITING": "대기",
}


def _format_trial_state(state: Optional[TrialState]) -> str:
    """Optuna TrialState 값을 한국어 라벨로 변환합니다."""

    if isinstance(state, TrialState):
        key = state.name
    else:
        text = "" if state is None else str(state)
        key = text.split(".")[-1] if text else ""
    key_upper = key.upper()
    if not key_upper:
        return "알수없음"
    return TRIAL_STATE_LABELS.get(key_upper, key_upper)


def _utcnow_isoformat() -> str:
    """현재 UTC 시각을 ISO8601 ``Z`` 표기 문자열로 반환합니다."""

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
    """기본 팩터만 남긴 탐색 공간 사본을 반환합니다."""

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
    """
    Ensure the optimisation space includes both a timeframe and an ``ltf`` choice.

    When multiple lower‑timeframe (LTF) datasets exist or a timeframe cycle is
    configured, the optimiser needs explicit categorical parameters to
    select between them.  Historically only the ``timeframe`` key was
    injected, but downstream code also consults the ``ltf`` parameter when
    matching datasets.  If neither key is present in the search space
    we create both entries pointing to the same set of allowable values.
    This helps avoid situations where only one timeframe (typically the
    first) is ever used because a missing ``ltf`` parameter prevents the
    diversifier or dataset matcher from overriding the default choice.

    Parameters
    ----------
    space : dict
        Existing Optuna search space.  May already define ``timeframe`` or
        ``ltf`` keys, in which case this helper does nothing.
    datasets : sequence of DatasetSpec
        Resolved data sets for the backtest run.  Each entry has a
        ``timeframe`` attribute.  The list of unique timeframes forms
        the candidate values for the injected parameters.
    search_cfg : mapping
        The optimiser configuration.  If a ``diversify.timeframe_cycle`` is
        present we inject the selector even if only one dataset exists so
        that the diversifier can rotate through the specified cycle.

    Returns
    -------
    tuple
        A tuple ``(updated_space, added)`` where ``updated_space`` is the
        potentially modified space and ``added`` is ``True`` when a
        new parameter was injected.
    """

    timeframe_present = "timeframe" in space
    ltf_present = "ltf" in space
    if timeframe_present and ltf_present:
        return space, False

    # Gather distinct LTF values from the datasets.  If no datasets are
    # provided there is nothing to inject.
    choices = _collect_timeframe_choices(datasets)
    if not choices:
        return space, False

    # Determine whether a timeframe cycle exists.  When configured we
    # inject the selector even if only one dataset is available so that
    # the diversifier can cycle through the specified timeframes.
    raw_diversify_cfg = search_cfg.get("diversify") if isinstance(search_cfg, Mapping) else {}
    diversify_cfg = raw_diversify_cfg if isinstance(raw_diversify_cfg, Mapping) else {}
    cycle_defined = bool(diversify_cfg.get("timeframe_cycle"))
    if not cycle_defined and len(choices) <= 1 and not (timeframe_present ^ ltf_present):
        return space, False

    updated = dict(space)
    added = False
    if not timeframe_present:
        updated["timeframe"] = {"type": "choice", "values": choices}
        added = True
    if not ltf_present:
        updated["ltf"] = {"type": "choice", "values": choices}
        added = True
    return updated, added


def _filter_basic_factor_params(
    params: Dict[str, object], *, enabled: bool = True
) -> Dict[str, object]:
    """기본 팩터 키만 남겨 파라미터 딕셔너리를 정리합니다."""

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
    """과거 기록 호환을 위해 BB 파라미터가 없으면 KC 값을 기반으로 채웁니다."""

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


def _order_mapping(
    payload: Mapping[str, object],
    preferred_order: Optional[Sequence[str]] = None,
    *,
    priority: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """주어진 참조 순서에 맞춰 딕셔너리 순서를 재정렬합니다."""

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
    """설정값을 정수로 강제 변환하며 하한선을 검증합니다."""

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
        LOGGER.warning(
            "%s 값 '%s' 을(를) 정수로 변환할 수 없어 무시합니다.",
            name,
            raw_value,
        )
        return None

    if numeric < minimum:
        LOGGER.warning(
            "%s 값 %d 이(가) %d 보다 작아 무시합니다.",
            name,
            numeric,
            minimum,
        )
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


def _extract_min_trades_from_mapping(entry: object) -> Optional[int]:
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

    ``dataset_ref`` may be a :class:`DatasetSpec` (thread executor) or 문자열 ID
    (process executor). When a string ID is provided, 워커 초기화 시 등록된 전역
    캐시를 통해 DataFrame을 최초 한 번만 로드합니다. 이 함수는 모듈 레벨에
    존재해야 ``ProcessPoolExecutor``에서 피클링할 수 있습니다.
    """

    # Determine if an alternative engine is requested.  The ``engine``
    # parameter can be provided in the ``params`` dict as ``altEngine``
    # (e.g. "vectorbt" or "pybroker").  If specified and an alternative
    # engine is available, attempt to delegate the backtest accordingly.  In
    # the event of missing dependencies or unimplemented integration, fall
    # back to the native run_backtest implementation.
    dataset = _resolve_dataset_reference(dataset_ref)

    engine = None
    try:
        engine = params.get("altEngine") or params.get("engine")
    except Exception:
        engine = None
    if engine:
        try:
            from .alternative_engine import run_backtest_alternative
            return run_backtest_alternative(
                dataset.df,
                params,
                fees,
                risk,
                htf_df=dataset.htf,
                min_trades=min_trades,
                engine=str(engine),
            )
        except Exception as exc:
            # Log but continue with the default implementation
            LOGGER.warning(
                "Alternative engine '%s' failed (%s); falling back to native backtest.",
                engine,
                exc,
            )

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
    )


def _resolve_output_directory(
    base: Optional[Path],
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[Path, Dict[str, str]]:
    ts, symbol_slug, timeframe_slug, tag = _build_run_tag(datasets, params_cfg, run_tag)
    if base is None:
        root = DEFAULT_REPORT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        output = root / tag
    else:
        output = base
        output.parent.mkdir(parents=True, exist_ok=True)
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
    """CLI 인자의 timeframe_mix 값을 정규화하고 Canonical 문자열로 보존합니다."""

    raw = getattr(args, "timeframe_mix", None)
    mix_values = [token for token in _parse_ltf_choice_value(raw) if token]
    if len(mix_values) <= 1:
        if len(mix_values) == 1 and not getattr(args, "timeframe", None):
            args.timeframe = mix_values[0]
        if hasattr(args, "timeframe_mix"):
            setattr(args, "timeframe_mix", None)
        return []

    canonical = ",".join(mix_values)
    setattr(args, "timeframe_mix", canonical)
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
            "스터디 레지스트리(%s)에서 %s 설정을 불러왔습니다.",
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
    """사용자가 선호하는 LTF 조합(1, 3, 5분봉 또는 전체)을 선택하도록 안내합니다."""

    options = {"1": "1m", "3": "3m", "5": "5m"}
    if not sys.stdin or not sys.stdin.isatty():
        LOGGER.info("비대화형 환경이 감지되어 기본 1m LTF를 사용합니다.")
        return LTFPromptResult("1m")

    while True:
        print("\n작업을 시작 하기 전에 LTF를 선택해주세요.")
        print("  1) 1분봉")
        print("  3) 3분봉")
        print("  5) 5분봉")
        print("  7) 1/3/5 전체 (순차 실행)")
        print("  8) 1/3/5 혼합 (단일 리포트)")
        raw = input("선택 (1/3/5/7/8): ").strip()
        if raw in options:
            selection = options[raw]
            print(f"{raw}분봉을 선택했습니다.")
            return LTFPromptResult(selection)
        if raw == "7":
            print("1, 3, 5분봉을 모두 활용해 순차 실행합니다.")
            return LTFPromptResult(None, use_all=True)
        if raw == "8":
            print("1, 3, 5분봉을 한 번의 실행으로 혼합합니다 (단일 시트).")
            return LTFPromptResult(None, mix=["1m", "3m", "5m"])
        print("잘못된 입력입니다. 1, 3, 5, 7, 8 중 하나를 입력해주세요.")


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
    """Restrict LTF 관련 탐색 요소를 강제된 타임프레임으로 고정합니다."""

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
    """시간 프레임 관련 CLI 입력을 정규화해 실행 배치 계획을 생성합니다."""

    cleared_for_mix = False

    def _clear_timeframe_for_mix() -> None:
        nonlocal cleared_for_mix
        if cleared_for_mix:
            return
        if getattr(args, "timeframe", None):
            LOGGER.info(
                "타임프레임 혼합 실행이 지정되어 --timeframe=%s 설정을 무시합니다.",
                args.timeframe,
            )
            args.timeframe = None
        if getattr(args, "timeframe_grid", None):
            LOGGER.info(
                "타임프레임 혼합 실행이 지정되어 --timeframe-grid=%s 설정을 무시합니다.",
                args.timeframe_grid,
            )
            args.timeframe_grid = None
        cleared_for_mix = True

    mix_values = _normalise_timeframe_mix_argument(args)
    if mix_values:
        _clear_timeframe_for_mix()

    ltf_prompt = getattr(args, "_ltf_prompt_selection", None)
    if (
        ltf_prompt is None
        and not getattr(args, "timeframe", None)
        and not getattr(args, "timeframe_grid", None)
        and not getattr(args, "timeframe_mix", None)
    ):
        ltf_prompt = _prompt_ltf_selection()
        setattr(args, "_ltf_prompt_selection", ltf_prompt)

    if ltf_prompt:
        if ltf_prompt.mix:
            args.timeframe = None
            args.timeframe_grid = None
            args.timeframe_mix = ",".join(ltf_prompt.mix)
        elif ltf_prompt.use_all:
            args.timeframe = None
            if getattr(args, "timeframe_grid", None):
                LOGGER.info(
                    "사용자가 이미 타임프레임 그리드를 지정해 혼합 실행 요청을 유지합니다: %s",
                    args.timeframe_grid,
                )
            else:
                ltf_candidates = _collect_ltf_candidates(backtest_cfg, params_cfg)
                if not ltf_candidates:
                    ltf_candidates = ["1m", "3m", "5m"]
                args.timeframe_grid = ",".join(ltf_candidates)
                LOGGER.info(
                    "혼합 LTF 실행을 위해 타임프레임 그리드를 자동 구성했습니다: %s",
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
    LOGGER.info("타임프레임 그리드 %d건 실행: %s", total, combo_summary)

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

    기본 전략 사양에서는 반대 신호 청산(`exitOpposite`)과 모멘텀 페이드
    (`useMomFade`) 가운데 최소 하나가 켜져 있어야 합니다.  최적화 탐색이나
    Seed 불러오기 과정에서 두 옵션이 모두 꺼져 있으면 손절·익절 장치가
    사라져 전략이 비정상적으로 동작합니다.  이 헬퍼는 파라미터 집합을
    받아 논리를 보정하고, 두 옵션이 모두 ``False``일 때 반대 신호 청산을
    기본값으로 강제합니다.

    Args:
        params: 검사할 파라미터 매핑.
        context: 로그 메시지에 추가할 식별자(예: ``"trial #5"``).

    Returns:
        두 출구 플래그가 최소 하나는 ``True``가 되도록 정규화한 사본.
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
    chandelier_enabled = _coerce_bool_or_none(patched.get("useChandelierExit")) is True
    sar_enabled = _coerce_bool_or_none(patched.get("useSarExit")) is True
    atr_stop_len = _positive_number(patched.get("atrStopLen"))
    atr_stop_mult = _positive_number(patched.get("atrStopMult"))
    atr_stop_enabled = atr_stop_len is not None and atr_stop_mult is not None
    stop_channel_type = _normalise_channel_type(patched.get("stopChannelType"))
    stop_channel_mult = _positive_number(patched.get("stopChannelMult"))
    channel_stop_enabled = stop_channel_type in {"BB", "KC"} and stop_channel_mult is not None

    has_alternative_exit = (
        fixed_stop_enabled
        or chandelier_enabled
        or sar_enabled
        or atr_stop_enabled
        or channel_stop_enabled
    )

    if exit_flag is False and mom_fade_flag is False:
        context_text = f" ({context})" if context else ""
        if has_alternative_exit:
            LOGGER.debug(
                "출구 안전장치%s: exitOpposite/useMomFade가 모두 False지만 다른 출구 옵션이 활성화되어 강제를 건너뜁니다.",
                context_text,
            )
        else:
            LOGGER.warning(
                "출구 안전장치%s: exitOpposite과 useMomFade가 모두 비활성화되고 다른 출구 옵션이 없어 exitOpposite을 True로 강제합니다.",
                context_text,
            )
            exit_flag = True

    patched["exitOpposite"] = exit_flag
    patched["useMomFade"] = mom_fade_flag
    return patched


class _TimeframeCycler:
    """Rotate through pre-defined timeframe/HTF 조합."""

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
    """감시 콜백으로 동작하며, 탐색이 단조로울 때 강제 점프 시도를 주입합니다."""

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
            enriched = _enforce_exit_guards(enriched, context="diversifier 큐")
            try:
                study.enqueue_trial(enriched, skip_if_exists=True)
            except Exception as exc:
                LOGGER.debug("분산 탐색 후보 등록 실패(%s): %s", enriched, exc)
                continue
            accepted += 1
        if accepted:
            self.total_enqueued += accepted
            LOGGER.info("단조 탐색 감지 – 강제 점프 %d건을 큐에 추가했습니다.", accepted)

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
            base = _enforce_exit_guards(base, context="diversifier 후보")
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
    """LLM 후보를 주기적으로 큐에 주입하는 Optuna 콜백."""

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
            LOGGER.warning("Gemini 후보 주기 값 '%s' 을(를) 해석할 수 없습니다.", interval_source)
            refresh_interval = 0
        self.refresh_interval = refresh_interval if refresh_interval > 0 else 0

        count_source = self.config.get("refresh_count")
        if count_source in {None, ""}:
            count_source = self.config.get("count")
        try:
            refresh_count = int(count_source) if count_source not in {None, ""} else 0
        except (TypeError, ValueError):
            LOGGER.warning("Gemini 후보 수 '%s' 를 해석할 수 없습니다.", count_source)
            refresh_count = 0
        self.refresh_count = refresh_count if refresh_count > 0 else 0

        enabled_flag = _coerce_bool_or_none(self.config.get("enabled"))
        base_enabled = bool(self.refresh_interval and self.refresh_count)
        if enabled_flag is None:
            self.enabled = base_enabled and bool(self.config.get("enabled", True))
        else:
            self.enabled = base_enabled and enabled_flag

        try:
            base_top_n = int(self.config.get("top_n", 10))
        except (TypeError, ValueError):
            LOGGER.warning("Gemini 상위 참조 수 '%s' 을(를) 해석할 수 없습니다.", self.config.get("top_n"))
            base_top_n = 10
        self.base_top_n = max(base_top_n, 1)

        try:
            base_bottom_n = int(self.config.get("bottom_n", base_top_n))
        except (TypeError, ValueError):
            LOGGER.warning(
                "Gemini 하위 참조 수 '%s' 을(를) 해석할 수 없습니다.",
                self.config.get("bottom_n"),
            )
            base_bottom_n = base_top_n
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
        except Exception as exc:  # pragma: no cover - 방어적 로깅
            LOGGER.warning("Gemini 후보 생성 중 예외가 발생했습니다: %s", exc)
            return

        if llm_bundle.insights:
            with self._lock:
                for insight in llm_bundle.insights:
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
            trial_params = _enforce_exit_guards(trial_params, context="LLM 후보")
            try:
                study.enqueue_trial(trial_params, skip_if_exists=True)
            except Exception as exc:  # pragma: no cover - Optuna 저장소 오류 시
                LOGGER.debug("Gemini 후보 %s 큐 등록 실패: %s", candidate, exc)
                continue
            accepted += 1

        if accepted:
            self.total_enqueued += accepted
            self.total_refreshes += 1
            self.last_refresh_trial = trial.number
            LOGGER.info(
                "Gemini 제안 %d건을 큐에 추가했습니다. (트라이얼 %d 기준)",
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
    LOGGER.info("탐색 다변화 보조 기능을 활성화합니다.")
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
        "Gemini 보조 탐색을 활성화합니다: 완료된 트라이얼 %d회마다 최대 %d건 후보 요청",
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
        raise KeyError(f"등록되지 않은 데이터셋 ID: {dataset_id}")

    cache_root = handle.get("cache_root")
    if not cache_root:
        raise RuntimeError("process executor에서 데이터셋을 로드할 캐시 경로가 설정되지 않았습니다.")
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
        try:
            htf_df = cache.get(source_symbol, str(htf_timeframe), start, end)
        except Exception as exc:
            LOGGER.warning("HTF 데이터 로드 실패(%s): %s", dataset_id, exc)
            htf_df = None

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
    raise TypeError(f"dataset_ref 타입을 처리할 수 없습니다: {type(dataset_ref)!r}")


def _serialise_datasets_for_process(datasets: Sequence[DatasetSpec]) -> List[Dict[str, object]]:
    handles: List[Dict[str, object]] = []
    for dataset in datasets:
        if not dataset.cache_info:
            raise RuntimeError(
                "process executor를 사용하려면 DatasetSpec.cache_info 가 필요합니다."
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
    """쉼표/세미콜론/개행 등으로 구분된 LTF 선택값을 정규화합니다."""

    if value is None:
        return []

    tokens: List[str] = []
    if isinstance(value, str):
        cleaned = value.replace("\n", ",").replace(";", ",")
        for token in cleaned.split(","):
            candidate = token.strip()
            if candidate:
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
        LOGGER.warning(
            "알 수 없는 dataset_executor '%s' 가 지정되어 thread 모드로 대체합니다.",
            dataset_executor,
        )
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
        LOGGER.warning(
            "search.dataset_jobs 값 '%s' 을 해석할 수 없어 %d로 대체합니다.",
            raw_dataset_jobs,
            auto_dataset_jobs,
        )
        dataset_jobs = auto_dataset_jobs

    dataset_jobs = min(dataset_jobs, max(1, available_cpu))

    dataset_parallel_capable = max_parallel_datasets > 1
    if not dataset_parallel_capable:
        if dataset_jobs != 1:
            LOGGER.info("단일 티커 구성으로 dataset_jobs %d→1로 비활성화합니다.", dataset_jobs)
        dataset_jobs = 1
    else:
        if dataset_jobs <= 1 and auto_dataset_jobs > 1:
            dataset_jobs = auto_dataset_jobs
            LOGGER.info(
                "데이터셋 병렬 worker %d개 자동 설정 (가용 CPU=%d, 최대 병렬=%d)",
                dataset_jobs,
                available_cpu,
                max_parallel_datasets,
            )
        elif dataset_jobs > auto_dataset_jobs:
            LOGGER.info(
                "데이터셋 병렬 worker 수를 %d→%d로 제한합니다. (최대 병렬=%d)",
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
                "데이터셋 병렬(%d worker) 활성화로 Optuna worker %d→%d 조정",
                dataset_jobs,
                n_jobs,
                optuna_budget,
            )
            n_jobs = optuna_budget
            search_cfg["n_jobs"] = n_jobs
        LOGGER.info(
            "보조 데이터셋 병렬 worker %d개 (%s) 사용",
            dataset_jobs,
            dataset_executor,
        )
        if dataset_executor == "process" and dataset_start_method:
            LOGGER.info("프로세스 start method=%s", dataset_start_method)
    elif not dataset_parallel_capable:
        LOGGER.info(
            "단일 티커/데이터셋 구성이라 데이터셋 병렬화를 비활성화하고 Optuna worker %d개를 유지합니다.",
            n_jobs,
        )
        dataset_jobs = 1
    else:
        LOGGER.info(
            "설정상 데이터셋 병렬 worker 1개라 Optuna 병렬(worker=%d)만 사용합니다.",
            n_jobs,
        )
        dataset_jobs = 1

    LOGGER.info(
        "최종 병렬 전략: Optuna worker=%d (우선), 데이터셋 worker=%d (%s, 보조)",
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
        _normalise_timeframe_value(params.get("entry_tf"))
        or _normalise_timeframe_value(params.get("timeframe"))
        or _normalise_timeframe_value(params.get("ltf"))
        or _normalise_timeframe_value(params_cfg.get("entry_tf"))
        or _normalise_timeframe_value(params_cfg.get("timeframe"))
    )

    htf_value = (
        _normalise_htf_value(params.get("htf"))
        or _normalise_htf_value(params.get("htf_timeframe"))
    )

    if htf_value is None:
        cfg_htf = params_cfg.get("htf_timeframe")
        if cfg_htf:
            htf_value = _normalise_htf_value(cfg_htf)
        elif isinstance(params_cfg.get("htf_timeframes"), list) and len(params_cfg["htf_timeframes"]) == 1:
            htf_value = _normalise_htf_value(params_cfg["htf_timeframes"][0])

    multi_timeframes: List[str] = []
    for key in ("entry_tf", "ltfChoice", "ltf_choices"):
        multi_timeframes = _parse_ltf_choice_value(params.get(key))
        if multi_timeframes:
            break

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
                combined_key = (",".join(multi_timeframes), key_candidate[1])
            for dataset in group:
                aggregated.setdefault(dataset.name, dataset)
        if aggregated:
            return combined_key or (",".join(multi_timeframes), htf_value), list(aggregated.values())

    selected = _resolve_single(timeframe_value, htf_value)

    if selected is None:
        selected = (default_key, dataset_groups[default_key])

    return selected


def _pick_primary_dataset(datasets: Sequence[DatasetSpec]) -> DatasetSpec:
    return max(datasets, key=lambda item: len(item.df))


def _resolve_symbol_entry(entry: object, alias_map: Dict[str, str]) -> Tuple[str, str]:
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
                raise ValueError("datasets 항목에 symbol 키가 필요합니다.")
            display_symbol, source_symbol = _resolve_symbol_entry(str(symbol_value), alias_map)

            ltf_candidates = _to_list(entry.get("ltf") or entry.get("ltfs") or entry.get("timeframes"))
            if not ltf_candidates:
                raise ValueError(f"{symbol_value} 데이터셋에 최소 하나의 ltf/timeframe 이 필요합니다.")

            start_value = entry.get("start") or entry.get("from") or base_period.get("from")
            end_value = entry.get("end") or entry.get("to") or base_period.get("to")
            if not start_value or not end_value:
                raise ValueError(f"{symbol_value} 데이터셋에 start/end 구간이 필요합니다.")
            start = str(start_value)
            end = str(end_value)

            symbol_log = (
                display_symbol if display_symbol == source_symbol else f"{display_symbol}→{source_symbol}"
            )
            for timeframe in ltf_candidates:
                timeframe_text = str(timeframe)
                LOGGER.info(
                    "Preparing dataset %s %s %s→%s (LTF only)",
                    symbol_log,
                    timeframe_text,
                    start,
                    end,
                )
                df = cache.get(source_symbol, timeframe_text, start, end)
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
                    )
                )
        if not datasets:
            raise ValueError("backtest.datasets 설정에서 어떤 데이터셋도 생성되지 않았습니다.")
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
            display_symbol if display_symbol == source_symbol else f"{display_symbol}→{source_symbol}"
        )
        LOGGER.info(
            "Preparing dataset %s %s %s→%s (LTF only)",
            symbol_log,
            timeframe,
            start,
            end,
        )
        df = cache.get(source_symbol, timeframe, start, end)
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
            )
        )
    return datasets


def combine_metrics(
    metric_list: List[Dict[str, float]], *, simple_override: Optional[bool] = None
) -> Dict[str, float]:
    """여러 데이터셋의 백테스트 지표를 단일 Trial 결과로 합산합니다.

    다중 LTF 혼합 실행(`timeframe_mix` 혹은 `ltfChoice`)처럼 하나의 트라이얼에서
    복수의 하위 타임프레임을 동시에 평가할 때 각 데이터셋의 산출물을 받아 평균·합계를
    구한 뒤 대표 지표를 만들어 반환합니다. 핵심 계산 원칙은 다음과 같습니다.

    - ``Returns`` 시리즈는 인덱스를 기준으로 이어 붙인 뒤 시간순으로 정렬합니다.
    - ``TotalAssets`` · ``AvailableCapital`` · ``Savings`` 는 거래 수와 총자산, 최대
      드로우다운을 동시에 고려한 가중 평균으로 결합해 실전에서 더 안정적인
      타임프레임의 기여도를 높입니다.
    - ``Liquidations`` 은 전체 청산 발생 횟수를 보존하기 위해 단순 합산합니다.
    - ``TradesList`` 가 존재하면 모든 거래를 결합해 후속 리포트/JSON에 기록합니다.
    - 단순 지표 모드가 아니면 ``aggregate_metrics`` 로 세부 통계를 다시 집계해
      ``NetProfit``·``WinRate`` 등 파생 지표도 동일한 규칙으로 재평가합니다.

    따라서 ``results.csv`` 나 ``results_timeframe_summary.csv`` 에서 `1m,3m,5m` 같은
    혼합 LTF 라벨을 확인할 경우, 위 방식으로 결합된 값이라는 점을 전제로 해석해야
    합니다.
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
    available_values: List[Optional[float]] = []
    savings_values: List[Optional[float]] = []
    weight_factors: List[float] = []
    total_liquidations = 0.0
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

        available = _coerce_float(metrics.get("AvailableCapital"))
        available_values.append(available)

        savings = _coerce_float(metrics.get("Savings"))
        savings_values.append(savings)

        trades_value = _coerce_float(metrics.get("Trades"))

        dd_raw = metrics.get("MaxDD")
        if dd_raw is None:
            dd_raw = metrics.get("MaxDrawdown")
        drawdown_value = _coerce_float(dd_raw)

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

    aggregated["Returns"] = merged_returns
    aggregated["Valid"] = bool(valid_flag)
    total_assets_weighted = _weighted_mean(total_assets_values)
    if total_assets_weighted is None:
        total_assets_weighted = _fallback_mean(total_assets_values)
    if total_assets_weighted is not None:
        aggregated["TotalAssets"] = float(total_assets_weighted)
    else:
        aggregated["TotalAssets"] = float(aggregated.get("TotalAssets", 0.0))

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
    aggregated["Liquidations"] = float(total_liquidations)
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
    """총 자산 기반 점수를 계산합니다."""

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
    score = _as_float(total_assets, _as_float(metrics.get("FinalEquity"), 0.0))

    trades = int(round(_as_float(metrics.get("Trades"), 0.0)))
    min_trades = int(round(_as_float(constraints.get("min_trades_test"), 12.0)))
    if trades < min_trades:
        return 0.0

    dd_raw = metrics.get("MaxDD")
    if dd_raw is None:
        dd_raw = metrics.get("MaxDrawdown")
    dd_value = abs(_as_float(dd_raw, 0.0))
    dd_pct = dd_value * 100.0 if dd_value <= 1.0 else dd_value
    max_dd_limit = _as_float(constraints.get("max_dd_pct"), 70.0)
    if dd_pct > max_dd_limit and score > 0:
        excess = min(dd_pct - max_dd_limit, 100.0)
        score *= max(0.0, 1.0 - excess / 100.0)

    liquidations = _as_float(metrics.get("Liquidations"), 0.0)
    if liquidations > 0:
        score -= 25.0 * liquidations

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
    forced_params: Optional[Dict[str, object]] = None,
    *,
    study_storage: Optional[Path] = None,
    space_hash: Optional[str] = None,
    seed_trials: Optional[List[Dict[str, object]]] = None,
    log_dir: Optional[Path] = None,
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
                "기본 팩터 프로파일: %d→%d개 파라미터로 탐색 공간을 축소합니다.",
                len(original_space),
                len(space),
            )
            if not space:
                LOGGER.warning(
                    "기본 팩터 집합에 해당하는 항목이 없어 탐색 공간이 비었습니다."
                    " space 설정을 점검하세요."
                )
    else:
        LOGGER.info(
            "기본 팩터 프로파일 비활성화: 전체 %d개 파라미터 탐색", len(space)
        )

    space, timeframe_added = _ensure_timeframe_param(space, datasets, search_cfg)
    if timeframe_added:
        LOGGER.info(
            "탐색 공간에 타임프레임 파라미터(timeframe=%s)를 자동 추가했습니다.",
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
        LOGGER.warning("search.n_jobs 값 '%s' 을 해석할 수 없어 1로 대체합니다.", raw_n_jobs)
        n_jobs = 1
    force_sqlite_serial = bool(search_cfg.get("force_sqlite_serial"))
    if force_sqlite_serial and n_jobs != 1:
        LOGGER.info("SQLite 직렬 강제 옵션으로 Optuna worker %d→1개 조정", n_jobs)
        n_jobs = 1
        search_cfg["n_jobs"] = n_jobs
    if n_trials := int(search_cfg.get("n_trials", 0) or 0):
        auto_jobs = max(1, min(available_cpu, n_trials))
    else:
        auto_jobs = max(1, available_cpu)
    if not force_sqlite_serial and n_jobs <= 1 and auto_jobs > n_jobs:
        n_jobs = auto_jobs
        search_cfg["n_jobs"] = n_jobs
        LOGGER.info("가용 자원에 맞춰 Optuna worker %d개를 자동 할당했습니다.", n_jobs)

    if n_jobs > 1:
        LOGGER.info("Optuna 병렬 worker %d개를 사용합니다.", n_jobs)
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
            LOGGER.warning("Invalid nsga population size '%s'; using Optuna default", population_override)
    elif multi_objective:
        space_size = len(space) if hasattr(space, "__len__") else 0
        nsga_kwargs["population_size"] = max(64, (space_size or 0) * 2 or 64)
    if nsga_params_cfg.get("mutation_prob") is not None:
        try:
            nsga_kwargs["mutation_prob"] = float(nsga_params_cfg["mutation_prob"])
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga mutation_prob '%s'; ignoring", nsga_params_cfg["mutation_prob"])
    if nsga_params_cfg.get("crossover_prob") is not None:
        try:
            nsga_kwargs["crossover_prob"] = float(nsga_params_cfg["crossover_prob"])
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga crossover_prob '%s'; ignoring", nsga_params_cfg["crossover_prob"])
    if nsga_params_cfg.get("swap_step") is not None:
        try:
            nsga_kwargs["swap_step"] = int(nsga_params_cfg["swap_step"])
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga swap_step '%s'; ignoring", nsga_params_cfg["swap_step"])
    if seed is not None:
        try:
            nsga_kwargs["seed"] = int(seed)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid seed '%s'; ignoring for NSGA-II", seed)

    use_nsga = algo in {"nsga", "nsga2", "nsgaii"}
    if multi_objective and not use_nsga and algo in {"bayes", "tpe", "default", "auto"}:
        use_nsga = True

    if algo == "grid":
        sampler = optuna.samplers.GridSampler(grid_choices(space))
    elif algo == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif algo in {"cmaes", "cma-es", "cma"}:
        sampler = optuna.samplers.CmaEsSampler(seed=seed, consider_pruned_trials=True)
    elif use_nsga:
        sampler = optuna.samplers.NSGAIISampler(**nsga_kwargs)
    else:
        # Instantiate TPESampler without experimental arguments.  Passing
        # multivariate=True or group=True triggers ExperimentalWarning and may
        # change algorithm behaviour.  Use default settings to avoid
        # experimental features and suppress related warnings.
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
                LOGGER.warning(
                    "sqlite_timeout 값 '%s' 을 정수로 변환할 수 없어 120초로 대체합니다.",
                    timeout_raw,
                )
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
                    LOGGER.warning(
                        "SQLite 병렬 허용 옵션이 활성화되었습니다. Optuna worker %d개를 유지하지만 잠금"
                        " 충돌이 발생할 수 있습니다.",
                        n_jobs,
                    )
                else:
                    LOGGER.warning(
                        "SQLite 스토리지에서 Optuna worker %d개를 병렬로 실행합니다. 잠금 충돌 시 자동 재시도로 복구를 시도하니 모니터링하세요.",
                        n_jobs,
                    )
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

            storage = _make_rdb_storage(
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
        load_if_exists=False,
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
        trial_params = _enforce_exit_guards(trial_params, context="seed 큐")
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
                LOGGER.warning(
                    "dataset_start_method '%s' 을 사용할 수 없어 기본 spawn 을 사용합니다.",
                    dataset_start_method,
                )
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
            "total_vol": float("nan"),
            "avg_vol": float("nan"),
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

        volumes: List[float] = []
        wins = 0
        trades_count = 0

        for trade in raw:
            profit_value: Optional[float] = None
            volume_value: Optional[float] = None
            is_win_value: Optional[bool] = None
            if isinstance(trade, Trade):
                profit_value = _coerce_float_local(getattr(trade, "profit", None))
                qty_value = _coerce_float_local(getattr(trade, "size", None))
                entry_price_value = _coerce_float_local(getattr(trade, "entry_price", None))
                if qty_value is not None and entry_price_value is not None:
                    volume_value = abs(qty_value * entry_price_value)
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
                qty_value = _coerce_float_local(trade.get("qty") or trade.get("size"))
                entry_price_value = _coerce_float_local(
                    trade.get("entry_price") or trade.get("entryPrice")
                )
                volume_candidate = trade.get("quote_notional") or trade.get("quoteNotional")
                if volume_candidate is None and qty_value is not None and entry_price_value is not None:
                    volume_candidate = abs(qty_value * entry_price_value)
                volume_value = _coerce_float_local(volume_candidate)
                if profit_value is not None and is_win_value is None:
                    is_win_value = profit_value > 0
            else:
                continue

            if profit_value is None and is_win_value is None and volume_value is None:
                continue

            trades_count += 1
            if is_win_value:
                wins += 1
            if volume_value is not None:
                volumes.append(volume_value)

        summary["available"] = True
        summary["trades"] = trades_count
        summary["wins"] = wins
        summary["winrate"] = float(wins / trades_count) if trades_count else 0.0
        if volumes:
            volumes_array = np.asarray(volumes, dtype=np.float64)
            finite = volumes_array[np.isfinite(volumes_array)]
            if finite.size:
                summary["total_vol"] = float(np.sum(finite))
                summary["avg_vol"] = float(np.mean(finite))
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

            use_chandelier_exit_raw = _param_value("useChandelierExit")
            use_chandelier_exit = _coerce_bool_or_none(use_chandelier_exit_raw)
            chandelier_len = _param_value("chandelierLen") if use_chandelier_exit else None
            chandelier_mult = _param_value("chandelierMult") if use_chandelier_exit else None

            use_sar_exit_raw = _param_value("useSarExit")
            use_sar_exit = _coerce_bool_or_none(use_sar_exit_raw)
            sar_start = _param_value("sarStart") if use_sar_exit else None
            sar_increment = _param_value("sarIncrement") if use_sar_exit else None
            sar_maximum = _param_value("sarMaximum") if use_sar_exit else None

            fixed_stop_pct_raw = _param_value("fixedStopPct")
            fixed_stop_pct_val = _positive_number(fixed_stop_pct_raw)
            use_fixed_stop = True if fixed_stop_pct_val is not None else None

            atr_stop_len_raw = _param_value("atrStopLen")
            atr_stop_len_val = _positive_number(atr_stop_len_raw)
            atr_stop_mult_raw = _param_value("atrStopMult")
            atr_stop_mult_val = (
                _positive_number(atr_stop_mult_raw) if atr_stop_len_val is not None else None
            )
            use_atr_stop = True if atr_stop_len_val is not None and atr_stop_mult_val is not None else None

            use_atr_trail_raw = _param_value("useAtrTrail")
            use_atr_trail = _coerce_bool_or_none(use_atr_trail_raw)
            atr_trail_len_val: Optional[int] = None
            if use_atr_trail:
                atr_trail_len_candidate = _positive_number(_param_value("atrTrailLen"))
                if atr_trail_len_candidate is not None:
                    atr_trail_len_val = int(round(atr_trail_len_candidate))
            atr_trail_mult_val = (
                _positive_number(_param_value("atrTrailMult")) if use_atr_trail else None
            )

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

            params_json = json.dumps(record["params"], ensure_ascii=False, sort_keys=False)
            skipped_json = (
                json.dumps(skipped_serialisable, ensure_ascii=False)
                if skipped_serialisable
                else ""
            )

            total_assets_val = _metric_value("TotalAssets")
            if total_assets_val is None:
                total_assets_val = trial.user_attrs.get("total_assets")
            liquidations_val = _metric_value("Liquidations")
            if liquidations_val is None:
                liquidations_val = trial.user_attrs.get("liquidations")
            leverage_param = trial.params.get("leverage") if hasattr(trial, "params") else None
            leverage_val = _to_native(leverage_param) if leverage_param is not None else trial.user_attrs.get("leverage")
            chart_tf_val = trial.user_attrs.get("chart_tf") or _param_value("chart_tf")
            if not chart_tf_val:
                chart_tf_val = dataset_meta.get("chart_tf") or dataset_meta.get("timeframe")
            if chart_tf_val is not None:
                chart_tf_val = _to_native(chart_tf_val)

            entry_tf_val = trial.user_attrs.get("entry_tf") or _param_value("entry_tf")
            if not entry_tf_val:
                entry_tf_val = dataset_meta.get("entry_tf") or dataset_meta.get("timeframe")
            if entry_tf_val is not None:
                entry_tf_val = _to_native(entry_tf_val)

            use_htf_val = trial.user_attrs.get("use_htf")
            if use_htf_val is None:
                use_htf_val = _param_value("use_htf")
            use_htf_val = bool(use_htf_val) if use_htf_val is not None else False

            htf_tf_val = trial.user_attrs.get("htf_tf") or dataset_meta.get("htf_timeframe")
            if not use_htf_val or not htf_tf_val:
                htf_tf_val = "NA"
            htf_tf_val = _to_native(htf_tf_val)

            if trade_summary["available"]:
                trades_value_row: Optional[int] = int(trade_summary["trades"])
                wins_value_row: Optional[int] = int(trade_summary["wins"])
                win_rate_value_row: Optional[float] = float(trade_summary["winrate"])
                total_vol_row = _finite_float(trade_summary.get("total_vol"))
                avg_vol_row = _finite_float(trade_summary.get("avg_vol"))
            else:
                trades_value_row = _coerce_int_value(_metric_value("Trades"))
                wins_value_row = _coerce_int_value(_metric_value("Wins"))
                win_rate_metric = _metric_value("WinRate")
                win_rate_value_row = _finite_float(win_rate_metric)
                total_vol_row = None
                avg_vol_row = None
            if trades_value_row is None:
                trades_value_row = _coerce_int_value(_metric_value("TotalTrades"))
            if win_rate_value_row is not None:
                win_rate_value_row = round(float(win_rate_value_row), 4)
            if total_vol_row is not None:
                total_vol_row = round(float(total_vol_row), 2)
            if avg_vol_row is not None:
                avg_vol_row = round(float(avg_vol_row), 2)

            row = {
                "number": trial.number,
                "state": state_label,
                "value": value_field,
                "score": trial.user_attrs.get("score"),
                "total_assets": total_assets_val,
                "leverage": leverage_val,
                "chart_tf": chart_tf_val,
                "entry_tf": entry_tf_val,
                "use_htf": use_htf_val,
                "htf_tf": htf_tf_val,
                "use_fixed_stop": _csv_value(True if use_fixed_stop else None),
                "use_atr_stop": _csv_value(True if use_atr_stop else None),
                "use_atr_trail": _csv_value(True if use_atr_trail else None),
                "use_channel_stop": _csv_value(True if use_channel_stop_flag else None),
                "stop_channel_type": _csv_value(stop_channel_type_attr if stop_channel_type_attr else None),
                "stop_channel_mult": _csv_value(stop_channel_mult_attr if stop_channel_mult_attr is not None else None),
                "trades": trades_value_row,
                "wins": wins_value_row,
                "win_rate": win_rate_value_row,
                "total_vol": total_vol_row,
                "avg_vol": avg_vol_row,
                "max_dd": max_dd_value,
                "liquidations": liquidations_val,
                "valid": trial.user_attrs.get("valid"),
                "timeframe": (
                    dataset_meta.get("timeframe")
                    if dataset_meta.get("timeframe")
                    else dataset_meta.get("effective_timeframe")
                ),
                "htf_timeframe": dataset_meta.get("htf_timeframe"),
                "pruned": trial.user_attrs.get("pruned"),
                "use_chandelier_exit": _csv_value(use_chandelier_exit),
                "chandelier_len": _csv_value(chandelier_len),
                "chandelier_mult": _csv_value(chandelier_mult),
                "use_sar_exit": _csv_value(use_sar_exit),
                "sar_start": _csv_value(sar_start),
                "sar_increment": _csv_value(sar_increment),
                "sar_maximum": _csv_value(sar_maximum),
                "fixed_stop_pct": _csv_value(fixed_stop_pct_val),
                "atr_stop_len": _csv_value(int(round(atr_stop_len_val)) if atr_stop_len_val is not None else None),
                "atr_stop_mult": _csv_value(atr_stop_mult_val),
                "atr_trail_len": _csv_value(atr_trail_len_val),
                "atr_trail_mult": _csv_value(atr_trail_mult_val),
                "params": params_json,
                "skipped_datasets": skipped_json,
                "datetime_complete": record["datetime_complete"],
            }

            if trial_csv_path is not None:
                file_exists = trial_csv_path.exists()
                trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with trial_csv_path.open("a", encoding="utf-8", newline="") as csv_handle:
                    writer = csv.DictWriter(csv_handle, fieldnames=TRIAL_PROGRESS_FIELDS)
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
                total_assets_display = trial.user_attrs.get("total_assets")
            if total_assets_display in {None, ""}:
                total_assets_display = "-"
            liquidations_display = row.get("liquidations")
            if liquidations_display in {None, ""}:
                liquidations_display = trial.user_attrs.get("liquidations")
            if liquidations_display in {None, ""}:
                liquidations_display = 0
            trades_display = row.get("trades") if row.get("trades") not in {None, ""} else "-"
            wins_display = row.get("wins") if row.get("wins") not in {None, ""} else "-"
            score_display = row.get("score") if row.get("score") not in {None, ""} else "-"
            win_rate_display = row.get("win_rate")
            if win_rate_display in {None, ""}:
                win_rate_display = _metric_value("WinRate")
            total_vol_display = row.get("total_vol")
            avg_vol_display = row.get("avg_vol")
            dd_display = row.get("max_dd")
            if dd_display in {None, ""}:
                dd_display = max_dd_value

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

            trades_display = _format_int_display(trades_display)
            wins_display = _format_int_display(wins_display)
            win_rate_display = _format_metric_display(win_rate_display)
            total_vol_display = _format_metric_display(total_vol_display)
            avg_vol_display = _format_metric_display(avg_vol_display)
            dd_display = _format_metric_display(dd_display)

            LOGGER.info(
                "작업 진행상황 ＝＝＝＝＝＝ %d/%s (Trial %d %s) TotalAssets=%s, Liquidations=%s, Score=%s, Trades=%s, Wins=%s, WinRate=%s, TotVol=%s, AvgVol=%s, MaxDD=%s",
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
                total_vol_display,
                avg_vol_display,
                dd_display,
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
            snapshot = {
                "best_value": best_value,
                "best_params": best_params_full,
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
        atr_stop_len_val = _positive_number(params.get("atrStopLen"))
        atr_stop_mult_val = (
            _positive_number(params.get("atrStopMult")) if atr_stop_len_val is not None else None
        )
        use_fixed_stop_flag = fixed_stop_pct_val is not None
        use_atr_stop_flag = atr_stop_len_val is not None and atr_stop_mult_val is not None
        use_atr_trail_flag = _coerce_bool_or_none(params.get("useAtrTrail")) is True
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
        chart_tf_value = params.get("chart_tf")
        entry_tf_value = params.get("entry_tf")
        use_htf_flag = bool(params.get("use_htf"))
        htf_tf_value = params.get("htf_tf") if use_htf_flag else "NA"

        dataset_key_payload = {
            "timeframe": key[0],
            "htf_timeframe": key[1],
            "effective_timeframe": effective_timeframe,
            "chart_tf": chart_tf_value,
            "entry_tf": entry_tf_value,
        }
        ltf_selection = entry_tf_value or dataset_key_payload.get("timeframe")
        trial.set_user_attr("dataset_key", dataset_key_payload)
        trial.set_user_attr("chart_tf", chart_tf_value)
        trial.set_user_attr("entry_tf", entry_tf_value)
        trial.set_user_attr("use_htf", use_htf_flag)
        trial.set_user_attr("htf_tf", htf_tf_value)
        trial.set_user_attr("ltf_primary", effective_timeframe)
        trial.set_user_attr("use_fixed_stop", use_fixed_stop_flag)
        trial.set_user_attr("use_atr_stop", use_atr_stop_flag)
        trial.set_user_attr("use_atr_trail", use_atr_trail_flag)
        trial.set_user_attr("use_channel_stop", use_channel_stop_flag)
        trial.set_user_attr("stop_channel_type", channel_type_label)
        trial.set_user_attr("stop_channel_mult", float(stop_channel_mult_val) if use_channel_stop_flag else None)

        params_for_record = dict(params)
        params_for_record.pop("ltf", None)
        params_for_record.pop("ltfChoice", None)
        params_for_record.setdefault("chart_tf", chart_tf_value)
        params_for_record.setdefault("entry_tf", entry_tf_value)
        params_for_record.setdefault("use_htf", use_htf_flag)
        params_for_record["htf_tf"] = htf_tf_value
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
                LOGGER.warning(
                    "Non-finite %s score detected for trial %s; applying penalty %.0e",
                    stage,
                    trial.number,
                    non_finite_penalty,
                )
                return non_finite_penalty
            return numeric

        def _consume_dataset(
            idx: int,
            dataset: DatasetSpec,
            metrics: Dict[str, float],
            *,
            simple_override: bool = False,
        ) -> None:
            cleaned_metrics = _clean_metrics(metrics)
            dataset_entry: Dict[str, object] = {
                "name": dataset.name,
                "meta": dataset.meta,
                "metrics": cleaned_metrics,
            }

            trades_serialised = _serialise_trades_list(metrics.get("TradesList"))
            if trades_serialised:
                dataset_entry["trades"] = trades_serialised
                cleaned_metrics["TradesList"] = trades_serialised
                aggregate_trade_payload.extend(trades_serialised)

            trades_value = _coerce_int(metrics.get("Trades") or metrics.get("TotalTrades"))
            if trades_value is None:
                trades_value = _coerce_int(cleaned_metrics.get("Trades"))
            if trades_value is not None:
                cleaned_metrics["Trades"] = trades_value
            if trades_value is not None and trades_value < MIN_TRADES_ENFORCED:
                dataset_entry["skipped"] = True
                dataset_entry["skip_reason"] = "trades_threshold"
                dataset_entry["skip_metric"] = trades_value
                dataset_entry["skip_threshold"] = MIN_TRADES_ENFORCED
                dataset_metrics.append(dataset_entry)
                skipped_dataset_records.append(
                    {
                        "name": dataset.name,
                        "timeframe": dataset.timeframe,
                        "htf_timeframe": dataset.htf_timeframe,
                        "trades": trades_value,
                        "min_trades": MIN_TRADES_ENFORCED,
                    }
                )
                LOGGER.warning(
                    "데이터셋 %s 의 트레이드 수 %d 가 최소 요구치 %d 미만이라 제외합니다.",
                    dataset.name,
                    trades_value,
                    MIN_TRADES_ENFORCED,
                )
                return

            numeric_metrics.append(metrics)
            dataset_metrics.append(dataset_entry)

            dataset_score = compute_total_asset_score(metrics, constraints_cfg)
            dataset_score = _sanitise(dataset_score, f"dataset@{idx}")
            dataset_scores.append(dataset_score)

            partial_score = sum(dataset_scores) / max(1, len(dataset_scores))
            partial_score = _sanitise(partial_score, f"partial@{idx}")

            partial_metrics = combine_metrics(
                numeric_metrics, simple_override=simple_override
            )
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
                trial.set_user_attr("entry_tf", entry_tf_value)
                trial.set_user_attr("chart_tf", chart_tf_value)
                trial.set_user_attr("use_htf", use_htf_flag)
                trial.set_user_attr("htf_tf", htf_tf_value)
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

            dataset_entry: Dict[str, object] = {
                "name": dataset.name,
                "meta": dataset.meta,
                "metrics": {},
                "skipped": True,
                "skip_reason": "volume_threshold",
                "skip_metric": total_volume,
                "skip_threshold": min_volume_threshold,
            }
            dataset_metrics.append(dataset_entry)
            skipped_dataset_records.append(
                {
                    "name": dataset.name,
                    "timeframe": dataset.timeframe,
                    "htf_timeframe": dataset.htf_timeframe,
                    "total_volume": total_volume,
                    "min_volume": min_volume_threshold,
                }
            )
            LOGGER.warning(
                "데이터셋 %s 의 총 거래량 %.2f 이 최소 요구치 %.2f 미만이라 제외합니다.",
                dataset.name,
                total_volume,
                min_volume_threshold,
            )

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
                        "백테스트 실행 중 오류 발생 (dataset=%s, timeframe=%s, htf=%s)",
                        dataset.name,
                        dataset.timeframe,
                        dataset.htf_timeframe,
                    )
                    raise
                try:
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
                        "백테스트 실행 중 오류 발생 (dataset=%s, timeframe=%s, htf=%s)",
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
            score = sum(dataset_scores) / len(dataset_scores)
        else:
            score = non_finite_penalty
        score = _sanitise(score, "final")
        objective_values = (
            evaluate_objective_values(aggregated, objective_specs, non_finite_penalty)
            if multi_objective
            else None
        )

        cleaned_aggregated = _clean_metrics(aggregated)
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
                LOGGER.warning(
                    "트라이얼 %d 의 총 트레이드 수 %d 가 최소 요구치 %d 미만이라 결과를 무효 처리합니다.",
                    trial.number,
                    trades_total,
                    MIN_TRADES_ENFORCED,
                )
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
        trial.set_user_attr("entry_tf", entry_tf_value)
        trial.set_user_attr("chart_tf", chart_tf_value)
        trial.set_user_attr("use_htf", use_htf_flag)
        trial.set_user_attr("htf_tf", htf_tf_value)
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
            LOGGER.info("사용자 중지 요청으로 남은 Optuna 배치를 생략합니다.")
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


def merge_dicts(primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
    merged = dict(primary)
    merged.update({k: v for k, v in secondary.items() if v is not None})
    return merged










def _execute_single(
    args: argparse.Namespace,
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    argv: Optional[Sequence[str]] = None,
) -> None:
    if GLOBAL_STOP_EVENT.is_set():
        LOGGER.info("사용자 중지 요청이 이미 감지되어 해당 실행을 건너뜁니다.")
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
            raise ValueError(f"배치 템플릿 해석에 실패했습니다: {exc}") from exc

    if args.n_trials is not None:
        search_cfg["n_trials"] = int(args.n_trials)

    if args.n_jobs is not None:
        try:
            search_cfg["n_jobs"] = max(1, int(args.n_jobs))
        except (TypeError, ValueError):
            LOGGER.warning("--n-jobs 값 '%s' 이 올바르지 않아 1로 설정합니다.", args.n_jobs)
            search_cfg["n_jobs"] = 1

    if args.optuna_jobs is not None:
        try:
            search_cfg["n_jobs"] = max(1, int(args.optuna_jobs))
        except (TypeError, ValueError):
            LOGGER.warning(
                "--optuna-jobs 값 '%s' 이 올바르지 않아 %d로 설정합니다.",
                args.optuna_jobs,
                DEFAULT_OPTUNA_JOBS,
            )
            search_cfg["n_jobs"] = DEFAULT_OPTUNA_JOBS

    if args.dataset_jobs is not None:
        try:
            search_cfg["dataset_jobs"] = max(1, int(args.dataset_jobs))
        except (TypeError, ValueError):
            LOGGER.warning(
                "--dataset-jobs 값 '%s' 이 올바르지 않아 1로 설정합니다.",
                args.dataset_jobs,
            )
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
        LOGGER.info("CLI --force-sqlite-serial 지정: Optuna worker를 1개로 강제합니다.")

    if args.pruner:
        search_cfg["pruner"] = args.pruner

    forced_params: Dict[str, object] = dict(params_cfg.get("overrides", {}))
    auto_workers = bool(getattr(args, "auto_workers", False))
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
            args.timeframe_mix = ",".join(prompt_selection.mix)
            mix_values = _normalise_timeframe_mix_argument(args)
            selected_timeframe = None
            timeframe_overridden = False
        elif prompt_selection.use_all:
            all_timeframes_requested = True
            selected_timeframe = None
            timeframe_overridden = False
        else:
            selected_timeframe = prompt_selection.timeframe
            timeframe_overridden = True

    if args.interactive and symbol_choices:
        selected_symbol = _prompt_choice("Select symbol", symbol_choices, selected_symbol)

    if selected_symbol:
        params_cfg["symbol"] = selected_symbol
        backtest_cfg["symbols"] = [selected_symbol]
    if timeframe_overridden and selected_timeframe:
        params_cfg["timeframe"] = selected_timeframe
        backtest_cfg["timeframes"] = [selected_timeframe]
        _apply_ltf_override_to_datasets(backtest_cfg, selected_timeframe)
        forced_params["timeframe"] = selected_timeframe
        forced_params.setdefault("ltf", selected_timeframe)
        _enforce_forced_timeframe_constraints(params_cfg, search_cfg, selected_timeframe)
    elif mix_values:
        mix_label = ",".join(mix_values)
        forced_params.pop("timeframe", None)
        forced_params.pop("ltf", None)
        forced_params["ltfChoice"] = mix_label
        forced_params.setdefault("ltf_choices", mix_label)
        space_cfg = params_cfg.get("space")
        if isinstance(space_cfg, Mapping):
            spec_key = "ltfChoice"
            if "ltfChoice" not in space_cfg and "ltf_choices" in space_cfg:
                spec_key = "ltf_choices"
            spec = space_cfg.get(spec_key)
            if not isinstance(spec, dict):
                spec = {"type": "choice", "values": [mix_label]}
                space_cfg[spec_key] = spec
            else:
                updated = False
                for field in ("values", "choices", "options"):
                    seq = spec.get(field)
                    if isinstance(seq, list):
                        if mix_label not in seq:
                            seq.append(mix_label)
                        updated = True
                if not updated:
                    spec["values"] = [mix_label]
                if "type" not in spec:
                    spec["type"] = "choice"
        LOGGER.info("혼합 LTF 조합을 강제로 사용합니다: %s", mix_label)
    for key in ("htf", "htf_timeframe", "htf_timeframes"):
        params_cfg.pop(key, None)
        backtest_cfg.pop(key, None)

    backtest_periods = backtest_cfg.get("periods") or []
    params_backtest = _ensure_dict(params_cfg, "backtest")
    if args.start or args.end:
        start = args.start or params_backtest.get("from") or (backtest_periods[0]["from"] if backtest_periods else None)
        end = args.end or params_backtest.get("to") or (backtest_periods[0]["to"] if backtest_periods else None)
        if start and end:
            params_backtest["from"] = start
            params_backtest["to"] = end
            backtest_cfg["periods"] = [{"from": start, "to": end}]
    elif args.interactive and backtest_periods:
        labels = [f"{p['from']} → {p['to']}" for p in backtest_periods]
        default_label = f"{params_backtest.get('from')} → {params_backtest.get('to')}" if params_backtest.get("from") and params_backtest.get("to") else labels[0]
        choice = _prompt_choice("Select backtest period", labels, default_label)
        if choice and choice in labels:
            selected = dict(backtest_periods[labels.index(choice)])
            params_backtest["from"] = selected["from"]
            params_backtest["to"] = selected["to"]
            backtest_cfg["periods"] = [selected]
    elif params_backtest.get("from") and params_backtest.get("to"):
        backtest_cfg["periods"] = [{"from": params_backtest["from"], "to": params_backtest["to"]}]

    risk_cfg = _ensure_dict(params_cfg, "risk")
    backtest_risk = _ensure_dict(backtest_cfg, "risk")
    if args.leverage is not None:
        risk_cfg["leverage"] = args.leverage
        backtest_risk["leverage"] = args.leverage
    if args.qty_pct is not None:
        risk_cfg["qty_pct"] = args.qty_pct
        backtest_risk["qty_pct"] = args.qty_pct

    if args.interactive:
        leverage_default = risk_cfg.get("leverage")
        qty_default = risk_cfg.get("qty_pct")
        lev_input = input(f"Leverage [{leverage_default}]: ").strip()
        if lev_input:
            try:
                leverage_val = float(lev_input)
                risk_cfg["leverage"] = leverage_val
                backtest_risk["leverage"] = leverage_val
            except ValueError:
                print("Invalid leverage value, keeping previous setting.")
        qty_input = input(f"Position size %% [{qty_default}]: ").strip()
        if qty_input:
            try:
                qty_val = float(qty_input)
                risk_cfg["qty_pct"] = qty_val
                backtest_risk["qty_pct"] = qty_val
            except ValueError:
                print("Invalid quantity percentage, keeping previous setting.")

        bool_candidates = [name for name, spec in params_cfg.get("space", {}).items() if spec.get("type") == "bool"]
        for name in bool_candidates:
            default_val = forced_params.get(name)
            decision = _prompt_bool(f"Enable {name}", default_val)
            if decision is not None:
                forced_params[name] = decision

    llm_cfg = _ensure_dict(params_cfg, "llm")
    if args.llm is not None:
        llm_cfg["enabled"] = bool(args.llm)
    elif args.interactive:
        llm_default = _coerce_bool_or_none(llm_cfg.get("enabled"))
        llm_choice = _prompt_bool(
            "Gemini 후보/전략 인사이트를 사용할까요?", llm_default
        )
        if llm_choice is not None:
            llm_cfg["enabled"] = llm_choice

    if llm_cfg.get("enabled"):
        api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))
        if not llm_cfg.get("api_key") and not os.environ.get(api_key_env):
            LOGGER.warning(
                "Gemini 활성화 상태지만 API 키가 설정되지 않았습니다. %s 환경 변수를 확인하세요.",
                api_key_env,
            )

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
        LOGGER.info("단순화된 메트릭 계산 경로가 구성에서 활성화되어 있습니다.")
    else:
        for key in ("simpleMetricsOnly", "simpleProfitOnly"):
            forced_params.pop(key, None)
            risk_cfg.pop(key, None)
            backtest_risk.pop(key, None)
        LOGGER.info("전체 지표 계산 경로를 사용합니다.")

    global simple_metrics_enabled
    simple_metrics_enabled = simple_metrics_state

    params_cfg["overrides"] = forced_params

    datasets = prepare_datasets(params_cfg, backtest_cfg, args.data)
    if not datasets:
        raise RuntimeError("No datasets prepared for optimisation")

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
                        "Auto workers: Optuna n_jobs=%d (dataset 병렬 보정)",
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
        search_cfg["storage_url"] = DEFAULT_POSTGRES_STORAGE_URL

    storage_env_value = os.getenv(storage_env_key) if storage_env_key else None
    effective_storage_url = str(
        storage_env_value or search_cfg.get("storage_url") or ""
    )
    using_sqlite = effective_storage_url.startswith("sqlite:///")
    is_postgres = effective_storage_url.startswith(POSTGRES_PREFIXES)
    masked_storage_url = _mask_storage_url(effective_storage_url) if effective_storage_url else ""
    if storage_env_value:
        storage_source = f"환경변수 {storage_env_key}"
    elif effective_storage_url:
        storage_source = "설정값"
    else:
        storage_source = "기본값"
    backend_label = (
        "PostgreSQL"
        if is_postgres
        else "SQLite"
        if using_sqlite
        else "기타 RDB"
        if effective_storage_url
        else "비활성"
    )
    LOGGER.info(
        "Optuna 스토리지 백엔드: %s (%s, URL=%s)",
        backend_label,
        storage_source,
        masked_storage_url or "(없음)",
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
        forced_params,
        study_storage=study_storage,
        space_hash=space_hash,
        seed_trials=seed_trials,
        log_dir=trials_log_dir,
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

    study = optimisation.get("study")
    if study is not None:
        write_trials_dataframe(
            study,
            output_dir,
            param_order=optimisation.get("param_order"),
        )
    else:
        LOGGER.warning("No Optuna study returned; skipping trials export")

    walk_cfg = (
        params_cfg.get("walk_forward")
        or backtest_cfg.get("walk_forward")
        or {"train_bars": 5000, "test_bars": 2000, "step": 2000}
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

    wf_min_trades_override = _coerce_min_trades_value(walk_cfg.get("min_trades"))

    def _min_trades_for_dataset(dataset: DatasetSpec) -> Optional[int]:
        return _resolve_dataset_min_trades(
            dataset,
            constraints=constraints_cfg,
            risk=risk,
            explicit=wf_min_trades_override,
        )

    primary_min_trades = _min_trades_for_dataset(primary_dataset)
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
        train_bars=int(walk_cfg.get("train_bars", 5000)),
        test_bars=int(walk_cfg.get("test_bars", 2000)),
        step=int(walk_cfg.get("step", 2000)),
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

    candidate_summaries = [
        {
            "trial": best_record["trial"],
            "score": best_record.get("score"),
            "oos_mean": wf_summary.get("oos_mean"),
            "params": _ordered_params_view(best_record.get("params")),
            "timeframe": best_key[0],
            "htf_timeframe": best_key[1],
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
                train_bars=int(walk_cfg.get("train_bars", 5000)),
                test_bars=int(walk_cfg.get("test_bars", 2000)),
                step=int(walk_cfg.get("step", 2000)),
                htf_df=candidate_dataset.htf,
                min_trades=candidate_min_trades,
            )
            wf_cache[record["trial"]] = candidate_wf
            candidate_summaries.append(
                {
                    "trial": record["trial"],
                    "score": record.get("score"),
                    "oos_mean": candidate_wf.get("oos_mean"),
                    "params": _ordered_params_view(record.get("params")),
                    "timeframe": candidate_key[0],
                    "htf_timeframe": candidate_key[1],
                }
            )
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

    candidate_summaries[0] = {
        "trial": best_record["trial"],
        "score": best_record.get("score"),
        "oos_mean": wf_summary.get("oos_mean"),
        "params": _ordered_params_view(best_record.get("params")),
        "timeframe": best_key[0],
        "htf_timeframe": best_key[1],
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
        entry = {
            "trial": item["trial"],
            "score": item.get("score"),
            "oos_mean": item.get("oos_mean"),
            "params": ordered_params,
            "metrics": metrics_payload,
            "timeframe": item.get("timeframe"),
            "htf_timeframe": item.get("htf_timeframe"),
        }
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
        print("\n== USDT-Perp 24h 거래대금 상위 50 ==")
        for index, symbol in enumerate(auto_list, start=1):
            print(f"{index:2d}. {symbol}")
        print("\n예) 7번 선택:  python -m optimize.run --pick-top50 7")
        print("    직접 지정:  python -m optimize.run --pick-symbol BINANCE:ETHUSDT")
        return

    selected_symbol = ""
    if args.pick_symbol:
        selected_symbol = args.pick_symbol.strip()
    elif args.pick_top50:
        auto_list = auto_list or _load_top_list()
        if 1 <= args.pick_top50 <= len(auto_list):
            selected_symbol = auto_list[args.pick_top50 - 1]
        else:
            print("\n[ERROR] --pick-top50 인덱스가 범위를 벗어났습니다 (1~50).")
            return
    elif args.symbol:
        selected_symbol = args.symbol.strip()
    else:
        print("\n[ERROR] 심볼이 지정되지 않았습니다.")
        print("   예) 상위50 출력:       python -m optimize.run --list-top50")
        print("       7번 선택(예):      python -m optimize.run --pick-top50 7")
        print("       직접 지정:         python -m optimize.run --pick-symbol BINANCE:ETHUSDT")
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
        LOGGER.info("다중 LTF 혼합 실행을 준비합니다: %s", ", ".join(plan.mix_values))

    if not plan.combos and plan.batches:
        batch = plan.batches[0]
        if batch.context:
            setattr(batch.args, "_batch_context", batch.context)
        _execute_single(batch.args, params_cfg, backtest_cfg, argv)
        return

    total = len(plan.combos) or len(plan.batches)

    for idx, batch in enumerate(plan.batches, start=1):
        context = batch.context or {}
        ltf = context.get("ltf") or getattr(batch.args, "timeframe", None)
        index = context.get("index", idx)
        if GLOBAL_STOP_EVENT.is_set():
            LOGGER.info("사용자 중지 요청으로 남은 타임프레임 실행을 취소합니다.")
            break
        batch_args = batch.args
        if batch.context:
            batch_args._batch_context = batch.context  # type: ignore[attr-defined]
        LOGGER.info(
            "(%d/%d) LTF=%s 조합 최적화 시작",
            index,
            total,
            ltf,
        )
        _execute_single(batch_args, params_cfg, backtest_cfg, argv)
        if GLOBAL_STOP_EVENT.is_set():
            LOGGER.info("사용자 중지 요청으로 타임프레임 그리드 실행을 종료합니다.")
            break




if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
