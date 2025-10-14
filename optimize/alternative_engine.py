"""고속 외부 백테스트 엔진 호환 레이어.

이 모듈은 :mod:`optimize.strategy_model` 의 ``run_backtest`` 함수를
대체할 수 있는 호환 레이어를 제공한다. 현재는 **vectorbt** 기반 실행
경로를 우선 지원하며, 동일한 규칙(모멘텀 교차, 동적 임계값, `exitOpposite`
등)을 그대로 적용해 결과를 집계한다. 외부 엔진이 준비되지 않았거나
필수 기능을 아직 매핑하지 못했다면 :class:`NotImplementedError` 를
발생시켜 기본 파이썬 구현으로 안전하게 폴백하도록 설계했다.

PyBroker 통합은 향후 추가 예정이므로, 해당 엔진이 선택되면 현재는
명시적으로 ``NotImplementedError`` 를 발생시킨다.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimize.common import resolve_leverage
from inspect import Parameter, signature
from optimize.metrics import Trade, aggregate_metrics, equity_curve_from_returns
from optimize.indicators import (
    _atr,
    _bars_since_mask,
    _chandelier_levels,
    _compute_dynamic_thresholds,
    _compute_flux_block,
    _compute_momentum_block,
    _ema,
    _heikin_ashi,
    _hma,
    _parabolic_sar,
    _sma,
    _std,
)
from optimize.strategy_model import run_backtest as _run_backtest_reference

LOGGER = logging.getLogger(__name__)

# Note: Previous versions of this module suppressed warnings from vectorbt
# when both long and short signals were passed along with a direction argument.
# To avoid unnecessary warnings, the `direction` argument has been removed
# from the call to `Portfolio.from_signals`.  Any warnings related to
# experimental features should be addressed by disabling those options in
# upstream configuration rather than filtering at runtime.


# vectorbt/pybroker 는 선택적 의존성이므로 "필요할 때" 불러온다.
try:  # pragma: no cover - 런타임 환경에 따라 달라짐
    import vectorbt  # type: ignore  # noqa: F401

    _VBT_MODULE = vectorbt
    VECTORBT_AVAILABLE = True
except Exception:  # pragma: no cover - 미설치 시 자동 폴백
    _VBT_MODULE = None
    VECTORBT_AVAILABLE = False

try:  # pragma: no cover - 선택적 모듈
    import pybroker  # type: ignore  # noqa: F401

    PYBROKER_AVAILABLE = True
except Exception:  # pragma: no cover - 미설치 시 False
    PYBROKER_AVAILABLE = False


_SUPPORTED_ENGINES = {"vectorbt", "vectorbtpro", "vbt"}
_PYBROKER_ENGINES = {"pybroker", "pb"}


@dataclass
class _ParsedInputs:
    """전처리된 입력 및 파생 설정 값 컨테이너."""

    df: pd.DataFrame
    htf_df: Optional[pd.DataFrame]
    start_ts: pd.Timestamp
    commission_pct: float
    slippage_ticks: float
    leverage: float
    initial_capital: float
    capital_pct: float
    allow_long: bool
    allow_short: bool
    require_cross: bool
    exit_opposite: bool
    min_trades: int
    min_hold_bars: int
    max_consecutive_losses: int


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan"}:
            return default
        if text in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "f", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _coerce_str(value: object, default: str, *, name: str) -> str:
    """문자열 파라미터를 엄격하게 변환한다."""

    if value is None:
        return str(default)

    if isinstance(value, (list, tuple)):
        if not value:
            return str(default)
        if len(value) == 1:
            value = value[0]
        else:
            raise TypeError(
                f"'{name}' 파라미터에 여러 문자열이 전달되었습니다: {value!r}"
            )

    text = str(value)
    return text if text else str(default)


def _ensure_datetime_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """OHLCV 프레임이 UTC 기반 DatetimeIndex 를 갖도록 강제한다."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" in frame.columns:
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            mask = ts.notna()
            if not mask.any():
                raise TypeError(
                    f"{label} 데이터프레임에 유효한 timestamp 컬럼이 없습니다."
                )
            frame = frame.loc[mask].copy()
            frame.index = ts[mask]
            frame = frame.drop(columns=["timestamp"])
        else:
            raise TypeError(
                f"{label} 데이터프레임은 DatetimeIndex 혹은 timestamp 컬럼이 필요합니다."
            )

    if frame.index.tz is None:
        frame = frame.tz_localize("UTC")
    else:
        frame = frame.tz_convert("UTC")
    return frame


def _normalise_ohlcv(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        dup = int(frame.index.duplicated(keep="last").sum())
        if dup:
            LOGGER.warning("%s 데이터에서 중복 인덱스 %d개를 제거합니다.", label, dup)
        frame = frame[~frame.index.duplicated(keep="last")]

    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in frame.columns:
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = coerced

    before = len(frame)
    frame = frame.dropna(subset=[col for col in required if col in frame.columns])
    dropped = before - len(frame)
    if dropped:
        LOGGER.warning("%s 데이터에서 결측 OHLCV 행 %d개를 제거했습니다.", label, dropped)
    if len(frame) < 2:
        raise ValueError(f"{label} 데이터가 부족하여 백테스트를 수행할 수 없습니다.")
    return frame


def _parse_core_settings(
    df: pd.DataFrame,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    *,
    min_trades: Optional[int],
    htf_df: Optional[pd.DataFrame],
) -> _ParsedInputs:
    df = _ensure_datetime_index(df, "가격")
    df = _normalise_ohlcv(df, "가격")
    if htf_df is not None:
        htf_df = _normalise_ohlcv(_ensure_datetime_index(htf_df, "HTF"), "HTF")

    start_raw = params.get("startDate")
    try:
        start_ts = pd.to_datetime(start_raw, utc=True) if start_raw else df.index[0]
    except Exception:
        start_ts = df.index[0]
    if start_ts < df.index[0]:
        start_ts = df.index[0]

    commission_pct = _coerce_float(
        fees.get("commission_pct", params.get("commission_value", 0.0005)), 0.0005
    )
    slippage_ticks = _coerce_float(fees.get("slippage_ticks", params.get("slipTicks")), 0.0)
    leverage = resolve_leverage(params if isinstance(params, dict) else None, risk if isinstance(risk, dict) else None)
    initial_capital = _coerce_float(
        risk.get("initial_capital", params.get("initial_capital")), 500.0
    )
    base_qty_pct = _coerce_float(params.get("baseQtyPercent"), 30.0) / 100.0

    allow_long = _coerce_bool(params.get("allowLongEntry"), True)
    allow_short = _coerce_bool(params.get("allowShortEntry"), True)
    # 전략 요구사항: 모멘텀 교차 필터는 항상 활성화한다.
    require_cross = True
    exit_opposite = _coerce_bool(params.get("exitOpposite"), True)

    if min_trades is not None:
        min_trades_value = max(int(min_trades), 0)
    else:
        min_trades_value = max(
            _coerce_int(
                params.get("minTrades", risk.get("min_trades", 0)),
                0,
            ),
            0,
        )

    min_hold_bars = max(_coerce_int(params.get("minHoldBars"), 0), 0)
    max_consecutive_losses = max(
        _coerce_int(params.get("maxConsecutiveLosses"), 3),
        0,
    )

    return _ParsedInputs(
        df=df,
        htf_df=htf_df,
        start_ts=start_ts,
        commission_pct=float(abs(commission_pct)),
        slippage_ticks=float(abs(slippage_ticks)),
        leverage=float(abs(leverage)) if leverage else 1.0,
        initial_capital=float(abs(initial_capital)) if initial_capital else 1.0,
        capital_pct=float(max(base_qty_pct, 0.0)),
        allow_long=allow_long,
        allow_short=allow_short,
        require_cross=require_cross,
        exit_opposite=exit_opposite,
        min_trades=min_trades_value,
        min_hold_bars=min_hold_bars,
        max_consecutive_losses=max_consecutive_losses,
    )


def _validate_feature_flags(params: Dict[str, object]) -> None:
    unsupported = [
        "useBreakevenStop",
        "usePivotStop",
        "useAtrProfit",
        "useDynVol",
        "useStopDistanceGuard",
        "useTimeStop",
        "useKASA",
        "useBETiers",
        "useShock",
        "useReversal",
        "useSqzGate",
        "useStructureGate",
        "useSizingOverride",
        "useDrawdownScaling",
        "usePerfAdaptiveRisk",
        "useDailyStopLoss",
        "useSessionFilter",
        "useDayFilter",
        "useAdx",
        "useEma",
        "useBbFilter",
        "useStochRsi",
        "useObv",
        "useAtrDiff",
        "useHtfTrend",
        "useHmaFilter",
        "useRangeFilter",
        "useEventFilter",
        "useSlopeFilter",
        "useDistanceGuard",
    ]
    enabled = [name for name in unsupported if _coerce_bool(params.get(name), False)]
    if enabled:
        raise NotImplementedError(
            "다음 옵션은 vectorbt 호환 레이어에서 아직 지원되지 않습니다: "
            + ", ".join(sorted(enabled))
        )


def _compute_indicators(
    df: pd.DataFrame,
    params: Dict[str, object],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    osc_len = max(_coerce_int(params.get("oscLen"), 20), 1)
    sig_len = max(_coerce_int(params.get("signalLen"), 3), 1)
    use_same_len = _coerce_bool(params.get("useSameLen"), False)
    kc_len = osc_len if use_same_len else max(_coerce_int(params.get("kcLen"), 18), 1)
    kc_mult = _coerce_float(params.get("kcMult"), 1.0)
    bb_len = osc_len if use_same_len else max(_coerce_int(params.get("bbLen"), kc_len), 1)
    bb_mult = _coerce_float(params.get("bbMult"), kc_mult)

    basis_style_raw = _coerce_str(params.get("basisStyle"), "Deluxe", name="basisStyle")
    if not basis_style_raw:
        basis_style_raw = _coerce_str(params.get("momStyle"), "Deluxe", name="momStyle")
    style_tmp = str(basis_style_raw).strip().lower()
    if style_tmp in {"kc", "basic", "원본", "기본"}:
        mom_style = "kc"
    elif style_tmp in {"avg", "average", "avgstyle", "average_style"}:
        mom_style = "avg"
    elif style_tmp in {"deluxe", "dx", "delux", "디럭스"}:
        mom_style = "deluxe"
    elif style_tmp in {"mod", "modified", "modsqueeze", "모디파이드", "modded"}:
        mom_style = "mod"
    else:
        mom_style = "deluxe"

    raw_ma_type = _coerce_str(params.get("maType"), "SMA", name="maType")
    mt_tmp = str(raw_ma_type).strip().lower() if raw_ma_type is not None else "sma"
    if mt_tmp in {"기본", "basic", "sma", "default"}:
        ma_type = "sma"
    elif mt_tmp == "ema":
        ma_type = "ema"
    elif mt_tmp == "hma":
        ma_type = "hma"
    else:
        ma_type = "sma"

    flux_len = max(_coerce_int(params.get("fluxLen"), 14), 1)
    flux_smooth_len = max(_coerce_int(params.get("fluxSmoothLen"), 1), 1)
    flux_deadzone = _coerce_float(params.get("fluxDeadzone"), 25.0)
    flux_use_ha = _coerce_bool(params.get("useFluxHeikin"), True)
    use_mod_flux = _coerce_bool(params.get("useModFlux"), False)
    compat_mode = _coerce_bool(params.get("compatMode"), True)
    auto_scale = _coerce_bool(params.get("autoThresholdScale"), True)
    use_norm_clip = _coerce_bool(params.get("useNormClip"), False)
    norm_clip_limit = _coerce_float(params.get("normClipLimit"), 350.0)

    momentum_block = _compute_momentum_block(
        df,
        osc_len,
        sig_len,
        style=mom_style,
        ma_type=ma_type,
        clip_enabled=use_norm_clip,
        clip_limit=norm_clip_limit,
    )
    momentum = momentum_block["momentum"]
    mom_signal = momentum_block["signal"]
    tr1_series = momentum_block["tr1_safe"]

    threshold_scale = pd.Series(1.0, index=df.index)

    prev_mom = momentum.shift(1).fillna(momentum)
    prev_sig = mom_signal.shift(1).fillna(mom_signal)
    cross_up = (prev_mom <= prev_sig) & (momentum > mom_signal)
    cross_down = (prev_mom >= prev_sig) & (momentum < mom_signal)

    flux_components = _compute_flux_block(
        df,
        flux_len,
        flux_smooth_len,
        flux_deadzone,
        use_heikin=flux_use_ha,
        use_mod_flux=use_mod_flux,
    )
    flux_hist = flux_components["cut"]
    flux_gate = flux_components["gate"]

    return (
        momentum,
        mom_signal,
        cross_up.astype(bool),
        cross_down.astype(bool),
        flux_hist,
        flux_gate,
        threshold_scale,
    )


def _resolve_thresholds(
    momentum: pd.Series,
    params: Dict[str, object],
    threshold_scale: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    use_dynamic = _coerce_bool(params.get("useDynamicThresh"), True)
    use_sym = _coerce_bool(params.get("useSymThreshold"), False)
    stat_threshold = _coerce_float(params.get("statThreshold"), 38.0)
    buy_threshold = _coerce_float(params.get("buyThreshold"), 36.0)
    sell_threshold = _coerce_float(params.get("sellThreshold"), 36.0)
    dyn_len = _coerce_int(params.get("dynLen"), 21)
    dyn_mult = _coerce_float(params.get("dynMult"), 1.1)
    buy_series, sell_series = _compute_dynamic_thresholds(
        momentum,
        use_dynamic=use_dynamic,
        use_sym_threshold=use_sym,
        stat_threshold=stat_threshold,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        dyn_len=dyn_len,
        dyn_mult=dyn_mult,
    )
    compat_mode = _coerce_bool(params.get("compatMode"), True)
    auto_scale = _coerce_bool(params.get("autoThresholdScale"), True)
    if compat_mode and auto_scale and not use_dynamic:
        buy_series = buy_series.multiply(threshold_scale)
        sell_series = sell_series.multiply(threshold_scale)
    return buy_series, sell_series


def _build_signals(
    df: pd.DataFrame,
    params: Dict[str, object],
    parsed: _ParsedInputs,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    (
        momentum,
        mom_signal,
        cross_up,
        cross_down,
        flux_hist,
        flux_gate,
        threshold_scale,
    ) = _compute_indicators(df, params)
    buy_thresh, sell_thresh = _resolve_thresholds(momentum, params, threshold_scale)

    base_long = (momentum < buy_thresh) & (flux_gate > 0)
    base_short = (momentum > sell_thresh) & (flux_gate < 0)

    if parsed.require_cross:
        base_long &= cross_up
        base_short &= cross_down

    if not parsed.allow_long:
        base_long = pd.Series(False, index=df.index)
    if not parsed.allow_short:
        base_short = pd.Series(False, index=df.index)

    long_exits = cross_down.copy()
    short_exits = cross_up.copy()
    if parsed.exit_opposite:
        long_exits |= base_short
        short_exits |= base_long

    base_long &= df.index >= parsed.start_ts
    base_short &= df.index >= parsed.start_ts
    long_exits &= df.index >= parsed.start_ts
    short_exits &= df.index >= parsed.start_ts

    return (
        base_long.astype(bool),
        long_exits.astype(bool),
        base_short.astype(bool),
        short_exits.astype(bool),
    )


def _apply_exit_overrides(
    parsed: _ParsedInputs,
    params: Dict[str, object],
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """스탑 및 모멘텀 페이드 조건을 반영해 청산 신호를 조정한다."""

    use_stop_loss = _coerce_bool(params.get("useStopLoss"), False)
    use_atr_trail = _coerce_bool(params.get("useAtrTrail"), False)
    use_mom_fade = _coerce_bool(params.get("useMomFade"), False)
    use_chandelier_exit = _coerce_bool(params.get("useChandelierExit"), False)
    use_sar_exit = _coerce_bool(params.get("useSarExit"), False)

    if not any((use_stop_loss, use_atr_trail, use_mom_fade, use_chandelier_exit, use_sar_exit)):
        return long_entries, long_exits, short_entries, short_exits

    df = parsed.df
    index = df.index
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    def _max_ignore(current: float, candidate: float) -> float:
        if np.isnan(candidate):
            return current
        if np.isnan(current):
            return candidate
        return max(current, candidate)

    def _min_ignore(current: float, candidate: float) -> float:
        if np.isnan(candidate):
            return current
        if np.isnan(current):
            return candidate
        return min(current, candidate)

    atr_values = None
    atr_trail_mult = 0.0
    if use_atr_trail:
        atr_len = max(_coerce_int(params.get("atrTrailLen"), 7), 1)
        atr_trail_mult = _coerce_float(params.get("atrTrailMult"), 2.5)
        atr_values = _atr(df, atr_len).to_numpy(dtype=float)

    swing_low = None
    swing_high = None
    if use_stop_loss:
        lookback = max(_coerce_int(params.get("stopLookback"), 5), 1)
        swing_low = df["low"].rolling(lookback, min_periods=lookback).min().to_numpy(dtype=float)
        swing_high = df["high"].rolling(lookback, min_periods=lookback).max().to_numpy(dtype=float)

    if use_chandelier_exit:
        chandelier_len = max(_coerce_int(params.get("chandelierLen"), 22), 1)
        chandelier_mult = _coerce_float(params.get("chandelierMult"), 3.0)
        chand_high, chand_low, chand_atr = _chandelier_levels(df, chandelier_len)
        chandelier_high = chand_high.to_numpy(dtype=float)
        chandelier_low = chand_low.to_numpy(dtype=float)
        chandelier_atr = chand_atr.to_numpy(dtype=float)
        chandelier_long_candidates = chandelier_high - chandelier_atr * chandelier_mult
        chandelier_short_candidates = chandelier_low + chandelier_atr * chandelier_mult
    else:
        chandelier_mult = 0.0
        chandelier_long_candidates = chandelier_short_candidates = None

    sar_values = None
    if use_sar_exit:
        sar_start = _coerce_float(params.get("sarStart"), 0.02)
        sar_increment = _coerce_float(params.get("sarIncrement"), 0.02)
        sar_maximum = _coerce_float(params.get("sarMaximum"), 0.2)
        sar_values = _parabolic_sar(df, sar_start, sar_increment, sar_maximum).to_numpy(dtype=float)

    # Compute simplified momentum fade context on the fly.  Rather than using
    # Bollinger/Keltner based histograms, we recompute momentum, its signal and
    # track where the momentum crosses the zero line.  We derive boolean
    # arrays indicating cross‑over/-under events and bar counts since
    # momentum was <= 0 or >= 0.  These arrays drive the fade exit logic.
    if use_mom_fade:
        # Recompute core indicators; this mirrors _compute_indicators but
        # returns momentum, signal and cross arrays.  We ignore the returned
        # flux histogram here.
        try:
            momentum, mom_signal, cross_up_series, cross_down_series, _ = _compute_indicators(df, params)
        except Exception:
            # If indicator computation fails, disable momentum fade to avoid
            # runtime errors.
            use_mom_fade = False
            cross_up_arr = cross_down_arr = None
            mom_abs_arr = mom_since_le_zero_arr = mom_since_ge_zero_arr = None
        else:
            mom_abs_series = momentum.abs()
            mom_le_zero = momentum <= 0
            mom_ge_zero = momentum >= 0
            mom_since_le_zero_series = _bars_since_mask(mom_le_zero)
            mom_since_ge_zero_series = _bars_since_mask(mom_ge_zero)
            cross_up_arr = cross_up_series.to_numpy(dtype=bool)
            cross_down_arr = cross_down_series.to_numpy(dtype=bool)
            mom_abs_arr = mom_abs_series.to_numpy(dtype=float)
            mom_since_le_zero_arr = mom_since_le_zero_series.to_numpy(dtype=float)
            mom_since_ge_zero_arr = mom_since_ge_zero_series.to_numpy(dtype=float)
        mom_fade_min_abs = max(0.0, _coerce_float(params.get("momFadeMinAbs"), 0.0))
    else:
        cross_up_arr = cross_down_arr = None
        mom_abs_arr = mom_since_le_zero_arr = mom_since_ge_zero_arr = None
        mom_fade_min_abs = 0.0

    long_entries_arr = long_entries.to_numpy(dtype=bool)
    short_entries_arr = short_entries.to_numpy(dtype=bool)
    long_exits_arr = long_exits.to_numpy(dtype=bool)
    short_exits_arr = short_exits.to_numpy(dtype=bool)

    position_dir = 0
    bars_held = 0
    long_stop_price = np.nan
    short_stop_price = np.nan
    chandelier_long_stop = np.nan
    chandelier_short_stop = np.nan

    for i in range(n):
        skip_entry = False

        if position_dir != 0:
            long_entries_arr[i] = False
            short_entries_arr[i] = False

        if position_dir > 0:
            bars_held += 1
            exit_long = bool(long_exits_arr[i])

            # ATR 트레일링 스탑
            if use_atr_trail and atr_values is not None:
                atr_val = atr_values[i]
                if np.isfinite(atr_val):
                    candidate = close[i] - atr_val * atr_trail_mult
                    if np.isfinite(candidate):
                        if np.isnan(long_stop_price):
                            long_stop_price = candidate
                        else:
                            long_stop_price = max(long_stop_price, candidate)

            # 스윙 로우 기반 스탑
            if use_stop_loss and swing_low is not None:
                ref = swing_low[i]
                if np.isfinite(ref):
                    if np.isnan(long_stop_price):
                        long_stop_price = ref
                    else:
                        long_stop_price = max(long_stop_price, ref)

            if use_chandelier_exit and chandelier_long_candidates is not None:
                base = chandelier_long_candidates[i]
                if np.isfinite(base):
                    if np.isnan(chandelier_long_stop):
                        chandelier_long_stop = base
                    else:
                        chandelier_long_stop = max(chandelier_long_stop, base)
                    long_stop_price = _max_ignore(long_stop_price, chandelier_long_stop)

            if use_sar_exit and sar_values is not None:
                sar_val = sar_values[i]
                if np.isfinite(sar_val):
                    long_stop_price = _max_ignore(long_stop_price, sar_val)

            if np.isfinite(long_stop_price) and low[i] <= long_stop_price:
                exit_long = True

            # Simplified momentum fade exit (long).  Trigger when momentum
            # crosses below its signal after being above zero and meets the
            # minimum absolute momentum threshold.
            if use_mom_fade and cross_down_arr is not None:
                if (
                    cross_down_arr[i]
                    and mom_since_le_zero_arr[i] > 0.0
                    and (mom_fade_min_abs <= 0.0 or mom_abs_arr[i] >= mom_fade_min_abs)
                ):
                    exit_long = True

            long_exits_arr[i] = exit_long
            if exit_long:
                position_dir = 0
                bars_held = 0
                long_stop_price = np.nan
                chandelier_long_stop = np.nan
                skip_entry = True
                long_entries_arr[i] = False
                short_entries_arr[i] = False

        elif position_dir < 0:
            bars_held += 1
            exit_short = bool(short_exits_arr[i])

            if use_atr_trail and atr_values is not None:
                atr_val = atr_values[i]
                if np.isfinite(atr_val):
                    candidate = close[i] + atr_val * atr_trail_mult
                    if np.isfinite(candidate):
                        if np.isnan(short_stop_price):
                            short_stop_price = candidate
                        else:
                            short_stop_price = min(short_stop_price, candidate)

            if use_stop_loss and swing_high is not None:
                ref = swing_high[i]
                if np.isfinite(ref):
                    if np.isnan(short_stop_price):
                        short_stop_price = ref
                    else:
                        short_stop_price = min(short_stop_price, ref)

            if use_chandelier_exit and chandelier_short_candidates is not None:
                base = chandelier_short_candidates[i]
                if np.isfinite(base):
                    if np.isnan(chandelier_short_stop):
                        chandelier_short_stop = base
                    else:
                        chandelier_short_stop = min(chandelier_short_stop, base)
                    short_stop_price = _min_ignore(short_stop_price, chandelier_short_stop)

            if use_sar_exit and sar_values is not None:
                sar_val = sar_values[i]
                if np.isfinite(sar_val):
                    short_stop_price = _min_ignore(short_stop_price, sar_val)

            if np.isfinite(short_stop_price) and high[i] >= short_stop_price:
                exit_short = True

            # Simplified momentum fade exit (short).  Trigger when momentum
            # crosses above its signal after being below zero and meets the
            # minimum absolute momentum threshold.
            if use_mom_fade and cross_up_arr is not None:
                if (
                    cross_up_arr[i]
                    and mom_since_ge_zero_arr[i] > 0.0
                    and (mom_fade_min_abs <= 0.0 or mom_abs_arr[i] >= mom_fade_min_abs)
                ):
                    exit_short = True

            short_exits_arr[i] = exit_short
            if exit_short:
                position_dir = 0
                bars_held = 0
                short_stop_price = np.nan
                chandelier_short_stop = np.nan
                skip_entry = True
                long_entries_arr[i] = False
                short_entries_arr[i] = False

        else:
            bars_held = 0
            long_stop_price = np.nan
            short_stop_price = np.nan
            chandelier_long_stop = np.nan
            chandelier_short_stop = np.nan

        if position_dir == 0 and not skip_entry:
            long_entry = long_entries_arr[i]
            short_entry = short_entries_arr[i]
            if long_entry and not short_entry:
                position_dir = 1
                bars_held = 0
                long_stop_price = np.nan
                chandelier_short_stop = np.nan
                if use_atr_trail and atr_values is not None:
                    atr_val = atr_values[i]
                    if np.isfinite(atr_val):
                        candidate = close[i] - atr_val * atr_trail_mult
                        if np.isfinite(candidate):
                            long_stop_price = candidate
                if use_stop_loss and swing_low is not None:
                    ref = swing_low[i]
                    if np.isfinite(ref):
                        long_stop_price = ref if np.isnan(long_stop_price) else max(long_stop_price, ref)
                if use_chandelier_exit and chandelier_long_candidates is not None:
                    base = chandelier_long_candidates[i]
                    if np.isfinite(base):
                        chandelier_long_stop = base
                        long_stop_price = _max_ignore(long_stop_price, base)
                else:
                    chandelier_long_stop = np.nan
                if use_sar_exit and sar_values is not None:
                    sar_val = sar_values[i]
                    if np.isfinite(sar_val):
                        long_stop_price = _max_ignore(long_stop_price, sar_val)
            elif short_entry and not long_entry:
                position_dir = -1
                bars_held = 0
                short_stop_price = np.nan
                chandelier_long_stop = np.nan
                if use_atr_trail and atr_values is not None:
                    atr_val = atr_values[i]
                    if np.isfinite(atr_val):
                        candidate = close[i] + atr_val * atr_trail_mult
                        if np.isfinite(candidate):
                            short_stop_price = candidate
                if use_stop_loss and swing_high is not None:
                    ref = swing_high[i]
                    if np.isfinite(ref):
                        short_stop_price = ref if np.isnan(short_stop_price) else min(short_stop_price, ref)
                if use_chandelier_exit and chandelier_short_candidates is not None:
                    base = chandelier_short_candidates[i]
                    if np.isfinite(base):
                        chandelier_short_stop = base
                        short_stop_price = _min_ignore(short_stop_price, base)
                else:
                    chandelier_short_stop = np.nan
                if use_sar_exit and sar_values is not None:
                    sar_val = sar_values[i]
                    if np.isfinite(sar_val):
                        short_stop_price = _min_ignore(short_stop_price, sar_val)

    return (
        pd.Series(long_entries_arr, index=index, dtype=bool),
        pd.Series(long_exits_arr, index=index, dtype=bool),
        pd.Series(short_entries_arr, index=index, dtype=bool),
        pd.Series(short_exits_arr, index=index, dtype=bool),
    )


def _bars_between(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int:
    try:
        start_loc = index.get_loc(start)
    except KeyError:
        start_loc = index.get_indexer([start])[0]
    try:
        end_loc = index.get_loc(end)
    except KeyError:
        end_loc = index.get_indexer([end])[0]
    return max(int(end_loc - start_loc), 1)


def _vectorbt_backtest(
    parsed: _ParsedInputs,
    params: Dict[str, object],
) -> Dict[str, float]:
    if not VECTORBT_AVAILABLE or _VBT_MODULE is None:  # pragma: no cover - 환경 의존
        raise ImportError("vectorbt 가 설치되어 있지 않습니다.")

    _validate_feature_flags(params)

    long_entries, long_exits, short_entries, short_exits = _build_signals(
        parsed.df, params, parsed
    )

    long_entries, long_exits, short_entries, short_exits = _apply_exit_overrides(
        parsed,
        params,
        long_entries,
        long_exits,
        short_entries,
        short_exits,
    )

    trade_value = parsed.initial_capital * parsed.capital_pct * parsed.leverage
    if trade_value <= 0:
        trade_value = parsed.initial_capital * parsed.leverage

    # Construct the portfolio keyword arguments without specifying a direction.
    # When both long/short entry and exit arrays are provided, vectorbt's
    # ``direction`` argument has no effect and would trigger a warning.  By
    # omitting it we allow vectorbt to infer the direction from the provided
    # signals automatically.  We conditionally include the ``execute_on_close``
    # flag only if the installed version of vectorbt advertises support for
    # it (checked via inspect.signature).  This avoids a ``TypeError`` on
    # older versions while preserving explicit close execution semantics when
    # available.
    pf_kwargs = dict(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=parsed.commission_pct,
        size=trade_value,
        size_type="value",
        # rely on vectorbt defaults for direction and opposite entry handling
        upon_opposite_entry="close",
    )
    # Conditionally enable execute_on_close if supported by the vectorbt API.
    try:
        vbt_from_signals_sig = signature(_VBT_MODULE.Portfolio.from_signals)
        accepts_kwargs = any(
            param.kind is Parameter.VAR_KEYWORD
            for param in vbt_from_signals_sig.parameters.values()
        )
        if "execute_on_close" in vbt_from_signals_sig.parameters or accepts_kwargs:
            pf_kwargs["execute_on_close"] = True
    except Exception:
        # Fail silently; earlier versions default to close execution.
        pass

    # Build the portfolio.  If a TypeError occurs and it does not mention
    # execute_on_close then propagate the error.  Otherwise retry without
    # the execute_on_close flag.  Because we only set execute_on_close when
    # supported, this branch should rarely trigger.
    try:
        pf = _VBT_MODULE.Portfolio.from_signals(parsed.df["close"], **pf_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "execute_on_close" not in message:
            raise
        warnings.warn(
            "vectorbt 포트폴리오가 execute_on_close 인자를 지원하지 않아 기본 동작으로 재시도합니다.",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback_kwargs = dict(pf_kwargs)
        fallback_kwargs.pop("execute_on_close", None)
        pf = _VBT_MODULE.Portfolio.from_signals(parsed.df["close"], **fallback_kwargs)

    # Collapse the portfolio if `close` is a callable method.  In some versions
    # of vectorbt, `close` is a method that returns a collapsed portfolio, while
    # in others it may be a property.  Avoid calling a Series as a function.
    close_attr = getattr(pf, "close", None)
    if callable(close_attr):
        pf = close_attr()

    # Access returns safely: vectorbt versions may expose `returns` as either a
    # property returning a Series or as a callable method.  Avoid calling a
    # Series as a function by checking if the attribute is callable.
    raw_returns_attr = getattr(pf, "returns")
    if callable(raw_returns_attr):
        raw_returns = raw_returns_attr()
    else:
        raw_returns = raw_returns_attr
    returns = pd.Series(
        np.asarray(raw_returns, dtype=float), index=parsed.df.index, dtype=float
    )
    returns.name = "returns"

    # -------------------------------------------------------------------
    # RUIN threshold enforcement
    #
    # Approximate the native Python backtest behaviour by computing an
    # equity curve from the returns series.  Starting from the initial
    # capital, we compound each period's return.  If the equity falls
    # below a fixed threshold of 50 units at any point, mark the run as
    # ruined.  This flag is later used in scoring to reject parameter
    # combinations with unacceptable drawdowns.
    ruin_threshold = 50.0
    try:
        equity_curve = equity_curve_from_returns(
            returns.fillna(0.0), initial=float(parsed.initial_capital)
        )
        min_equity = float(equity_curve.min()) if not equity_curve.empty else float(parsed.initial_capital)
        ruin_flag = min_equity < ruin_threshold
    except Exception:
        ruin_flag = False

    records = pf.trades.records_readable
    trades: List[Trade] = []
    if not records.empty:
        for row in records.to_dict("records"):
            entry_ts = pd.Timestamp(row.get("Entry Timestamp"))
            exit_ts = pd.Timestamp(row.get("Exit Timestamp"))
            if pd.isna(exit_ts):
                continue
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            else:
                entry_ts = entry_ts.tz_convert("UTC")
            if exit_ts.tz is None:
                exit_ts = exit_ts.tz_localize("UTC")
            else:
                exit_ts = exit_ts.tz_convert("UTC")
            direction = str(row.get("Direction", "long")).strip().lower()
            trade = Trade(
                entry_time=entry_ts,
                exit_time=exit_ts,
                direction="long" if direction.startswith("long") else "short",
                size=float(row.get("Size", 0.0) or 0.0),
                entry_price=float(row.get("Avg Entry Price", 0.0) or 0.0),
                exit_price=float(row.get("Avg Exit Price", 0.0) or 0.0),
                profit=float(row.get("PnL", 0.0) or 0.0),
                return_pct=float(row.get("Return", 0.0) or 0.0),
                mfe=float("nan"),
                mae=float("nan"),
                bars_held=_bars_between(parsed.df.index, entry_ts, exit_ts),
                reason="vectorbt",
            )
            trades.append(trade)

    metrics = aggregate_metrics(trades, returns, simple=False)
    metrics["TradesList"] = trades
    metrics["Returns"] = returns
    metrics.setdefault("InitialCapital", parsed.initial_capital)
    metrics.setdefault("Leverage", parsed.leverage)
    metrics.setdefault("Commission", parsed.commission_pct)
    metrics.setdefault("SlippageTicks", parsed.slippage_ticks)
    # 기본 엔진과 동일하게 전체 자산/가용 자본/청산 지표를 기록한다. vectorbt는
    # 동적 자본 흐름을 구현하지 않으므로, 최종 자산은 초기 자본에 순이익률을
    # 곱한 값으로 근사하고 청산·저축 지표는 0으로 채운다.
    try:
        net_profit_ratio = float(metrics.get("NetProfit", 0.0))
    except (TypeError, ValueError):
        net_profit_ratio = 0.0
    try:
        initial_capital = float(metrics.get("InitialCapital", parsed.initial_capital))
    except (TypeError, ValueError):
        initial_capital = float(parsed.initial_capital)
    final_equity = initial_capital * (1.0 + net_profit_ratio)
    metrics.setdefault("TotalAssets", final_equity)
    metrics.setdefault("AvailableCapital", final_equity)
    metrics.setdefault("Savings", 0.0)
    metrics.setdefault("Liquidations", 0.0)
    # Propagate the ruin flag calculated above.  When True this causes
    # downstream scoring functions to assign a zero score, effectively
    # skipping the parameter combination.
    metrics["Ruin"] = float(ruin_flag)
    metrics["Engine"] = "vectorbt"
    metrics["MinTrades"] = float(parsed.min_trades)
    metrics["MinHoldBars"] = float(parsed.min_hold_bars)
    metrics["MaxConsecutiveLossLimit"] = float(parsed.max_consecutive_losses)

    valid = (
        metrics.get("Trades", 0.0) >= parsed.min_trades
        and metrics.get("AvgHoldBars", 0.0) >= parsed.min_hold_bars
        and metrics.get("MaxConsecutiveLosses", 0.0) <= parsed.max_consecutive_losses
    )
    metrics["Valid"] = bool(valid)
    return metrics


def run_backtest_alternative(
    df: pd.DataFrame,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
    *,
    engine: str = "vectorbt",
) -> Dict[str, float]:
    """외부 엔진으로 백테스트를 실행한다."""

    # Normalise engine name to lower case; default to vectorbt when unspecified.
    engine_name = str(engine or "vectorbt").strip().lower()

    # 기본 동작은 항상 파이썬 기준 엔진을 사용하도록 한다. ``forceAltEngine``이
    # 명시적으로 참인 경우에만 대안 엔진을 시도한다. 이렇게 하면 실수로 벡터 엔진을
    # 활성화하여 결과가 변동되는 상황을 방지하면서도, 필요 시 수동으로 강제 실행할
    # 수 있다.
    force_alt_engine = _coerce_bool(params.get("forceAltEngine"), False)

    if not force_alt_engine:
        return _run_backtest_reference(
            df,
            params,
            fees,
            risk,
            htf_df=htf_df,
            min_trades=min_trades,
        )

    # 강제로 대안 엔진을 사용하더라도, vectorbt 래퍼는 아직 확장 손절 옵션이나
    # 레버리지 스윕을 완벽하게 반영하지 못한다. 이러한 파라미터가 포함되면 기준
    # 엔진으로 다시 되돌린다.
    extended_keys = {
        "fixedStopPct",
        "atrStopLen",
        "atrStopMult",
        "leverage",
        "chart_tf",
        "entry_tf",
        "use_htf",
        "htf_tf",
    }
    has_extended = any(k in params for k in extended_keys)

    # Wallet 기반 자본 관리 옵션은 아직 vectorbt 엔진에 매핑되지 않았다.
    # useWallet, profitReservePct, applyReserveToSizing 등이 활성화되면
    # 기본 파이썬 구현으로 폴백해 동일한 결과를 보장한다.
    wallet_enabled = _coerce_bool(params.get("useWallet"), False)
    reserve_pct = 0.0
    if "profitReservePct" in params:
        reserve_pct = _coerce_float(params.get("profitReservePct"), 0.0)
    reserve_applied = False
    if "applyReserveToSizing" in params:
        reserve_applied = _coerce_bool(params.get("applyReserveToSizing"), False)
    if wallet_enabled or reserve_pct > 0.0 or reserve_applied:
        # TODO(대안 엔진): vectorbt 포트폴리오가 월렛/적립금 기반 자본 조정
        #                  로직을 지원하도록 확장하면 폴백 조건을 재검토한다.
        return _run_backtest_reference(
            df,
            params,
            fees,
            risk,
            htf_df=htf_df,
            min_trades=min_trades,
        )

    # When extended parameters are present or when the engine is not vectorbt,
    # delegate the backtest to the reference Python implementation.  This
    # preserves dynamic capital, stop‑loss, and leverage behaviour across all
    # backtesting runs at the cost of additional computation time.
    if has_extended or engine_name not in _SUPPORTED_ENGINES:
        return _run_backtest_reference(df, params, fees, risk, htf_df=htf_df, min_trades=min_trades)

    # Otherwise, run the vectorbt wrapper.  The parsed inputs include the
    # standard capital percentage and leverage extracted from the risk config.
    parsed = _parse_core_settings(
        df,
        params,
        fees,
        risk,
        min_trades=min_trades,
        htf_df=htf_df,
    )
    return _vectorbt_backtest(parsed, params)

