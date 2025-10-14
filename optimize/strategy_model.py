"""Python 백테스트 엔진 – TradingView `매직1분VN` 최종본을 재현합니다."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import resolve_leverage
from .constants import NUMBA_AVAILABLE
from .indicators import (
    _atr,
    _bars_since_mask,
    _chandelier_levels,
    _compute_dynamic_thresholds,
    _compute_flux_block,
    _compute_momentum_block,
    _cross_over,
    _cross_under,
    _dmi,
    _ema,
    _ensure_series,
    _estimate_tick,
    _heikin_ashi,
    _hma,
    _linreg,
    _max_ignore_nan,
    _min_ignore_nan,
    _obv_slope,
    _parabolic_sar,
    _pivot_series,
    _resample_ohlcv,
    _rolling_rma_last,
    _rma,
    _rsi,
    _security_series,
    _seeded_ewma,
    _sma,
    _squeeze_momentum_norm,
    _std,
    _stoch_rsi,
    _timeframe_to_offset,
    _true_range,
    _wma,
)
from .metrics import (
    Trade,
    finalise_metrics_result,
)
from .state import EquityState, FilterContext, FilterSettings, Position
from .utils import njit, prange


@njit  # type: ignore[misc]
def _compute_cross_series_numba(momentum_array: np.ndarray, signal_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-over and cross-under boolean arrays using a Numba-accelerated
    loop.  The function accepts NumPy arrays of momentum and signal values
    (derived from Pandas Series via ``.to_numpy()``) and returns two
    boolean arrays indicating where a momentum cross-over (momentum rises
    above the signal) and a cross-under (momentum falls below the signal)
    occur.  A cross-over is defined when the previous momentum is less than
    or equal to the previous signal and the current momentum exceeds the
    current signal.  Similarly, a cross-under is defined when the previous
    momentum is greater than or equal to the previous signal and the current
    momentum is below the current signal.

    Parameters
    ----------
    momentum_array : np.ndarray
        The array of momentum values.
    signal_array : np.ndarray
        The array of corresponding signal values.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two boolean arrays (cross_up, cross_down).

    Notes
    -----
    The computation proceeds sequentially over the array using a standard
    ``for`` loop rather than ``prange``.  Using ``prange`` for this logic is
    incorrect because cross-over detection relies on the momentum and signal
    values from the immediately preceding bar.  ``prange`` allows iterations
    to run out of order or in parallel, which would break this dependency
    and lead to spurious results or no detected crosses at all.  By using a
    sequential loop, we ensure that ``prev_mom`` and ``prev_sig`` always
    reflect the values from the prior index, yielding correct cross
    detection.
    """
    n = len(momentum_array)
    cross_up = np.zeros(n, dtype=np.bool_)
    cross_down = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return cross_up, cross_down
    prev_mom = momentum_array[0]
    prev_sig = signal_array[0]
    for i in range(n):
        m = momentum_array[i]
        s = signal_array[i]
        # cross-over: previously below or equal and now above
        cross_up[i] = (prev_mom <= prev_sig) and (m > s)
        # cross-under: previously above or equal and now below
        cross_down[i] = (prev_mom >= prev_sig) and (m < s)
        prev_mom = m
        prev_sig = s
    return cross_up, cross_down


LOGGER = logging.getLogger(__name__)


_FASTPATH_REASON_GENERAL = 0
_FASTPATH_REASON_OPPOSITE = 1
_FASTPATH_REASON_TIME = 2


if NUMBA_AVAILABLE:

    @njit  # type: ignore[misc]
    def _fastpath_calc_order_size(
        price: float,
        tradable_capital: float,
        base_qty_percent: float,
        leverage: float,
    ) -> float:
        if price <= 0.0:
            return 0.0
        pct = max(base_qty_percent, 0.0)
        if pct <= 0.0 or leverage <= 0.0:
            return 0.0
        capital_portion = tradable_capital * pct / 100.0
        qty = (capital_portion * leverage) / price
        if not np.isfinite(qty) or qty <= 0.0:
            return 0.0
        return qty


    @njit  # type: ignore[misc]
    def _run_backtest_numba_fastpath(
        close: np.ndarray,
        base_long_trigger: np.ndarray,
        base_short_trigger: np.ndarray,
        allow_long_entry: bool,
        allow_short_entry: bool,
        debug_force_long: bool,
        debug_force_short: bool,
        exit_opposite: bool,
        use_time_stop: bool,
        max_hold_bars: int,
        min_hold_bars: int,
        reentry_bars: int,
        pyramiding_enabled: bool,
        commission_pct: float,
        slip_value: float,
        leverage: float,
        base_qty_percent: float,
        initial_capital: float,
        start_idx: int,
        min_tradable_capital: float,
        day_codes: np.ndarray,
        min_equity_floor: float,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        bool,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        n = close.shape[0]
        returns = np.zeros(n, dtype=np.float64)
        trade_entry_idx = np.empty(n, dtype=np.int64)
        trade_exit_idx = np.empty(n, dtype=np.int64)
        trade_direction = np.empty(n, dtype=np.int64)
        trade_qty = np.empty(n, dtype=np.float64)
        trade_entry_price = np.empty(n, dtype=np.float64)
        trade_exit_price = np.empty(n, dtype=np.float64)
        trade_pnl = np.empty(n, dtype=np.float64)
        trade_reason = np.empty(n, dtype=np.int64)
        trade_bars_held = np.empty(n, dtype=np.int64)
        trade_count = 0

        if start_idx < 0:
            start_idx = 0
        if start_idx >= n:
            return (
                initial_capital,
                0.0,
                initial_capital,
                max(initial_capital, min_equity_floor),
                False,
                returns,
                trade_entry_idx[:0],
                trade_exit_idx[:0],
                trade_direction[:0],
                trade_qty[:0],
                trade_entry_price[:0],
                trade_exit_price[:0],
                trade_pnl[:0],
                trade_reason[:0],
                trade_bars_held[:0],
            )

        equity = initial_capital
        net_profit = 0.0
        peak_equity = initial_capital
        tradable_capital = max(initial_capital, min_equity_floor)
        guard_frozen = False
        reentry_countdown = 0
        position_dir = 0
        position_qty = 0.0
        position_avg_price = 0.0
        position_entry_idx = -1
        bars_held = 0
        pyramid_count = 0
        base_entry_qty = 0.0
        last_day = day_codes[start_idx]

        for idx in range(start_idx, n):
            day_val = day_codes[idx]
            if day_val != last_day:
                guard_frozen = False
                last_day = day_val

            tradable_capital = equity
            if tradable_capital < min_equity_floor:
                tradable_capital = min_equity_floor

            if position_dir != 0:
                bars_held += 1

            if position_dir == 0 and reentry_countdown > 0:
                reentry_countdown -= 1

            if min_tradable_capital > 0.0 and tradable_capital < min_tradable_capital:
                guard_frozen = True

            can_trade = not guard_frozen

            if can_trade and position_dir == 0 and reentry_countdown == 0:
                enter_long = False
                if allow_long_entry and (debug_force_long or base_long_trigger[idx]):
                    enter_long = True
                enter_short = False
                if allow_short_entry and (debug_force_short or base_short_trigger[idx]):
                    enter_short = True

                if enter_long:
                    qty = _fastpath_calc_order_size(close[idx], tradable_capital, base_qty_percent, leverage)
                    if qty > 0.0:
                        position_dir = 1
                        position_qty = qty
                        position_avg_price = close[idx]
                        position_entry_idx = idx
                        bars_held = 0
                        pyramid_count = 0
                        base_entry_qty = qty
                        continue
                if enter_short:
                    qty = _fastpath_calc_order_size(close[idx], tradable_capital, base_qty_percent, leverage)
                    if qty > 0.0:
                        position_dir = -1
                        position_qty = qty
                        position_avg_price = close[idx]
                        position_entry_idx = idx
                        bars_held = 0
                        pyramid_count = 0
                        base_entry_qty = qty
                        continue

            if (
                pyramiding_enabled
                and position_dir != 0
                and pyramid_count == 0
                and can_trade
                and reentry_countdown == 0
            ):
                if position_dir > 0 and (debug_force_long or base_long_trigger[idx]):
                    if base_entry_qty > 0.0:
                        new_qty = base_entry_qty
                        new_total = position_qty + new_qty
                        if new_total > 0.0:
                            position_avg_price = (
                                position_avg_price * position_qty + close[idx] * new_qty
                            ) / new_total
                            position_qty = new_total
                            pyramid_count = 1
                elif position_dir < 0 and (debug_force_short or base_short_trigger[idx]):
                    if base_entry_qty > 0.0:
                        new_qty = base_entry_qty
                        new_total = position_qty + new_qty
                        if new_total > 0.0:
                            position_avg_price = (
                                position_avg_price * position_qty + close[idx] * new_qty
                            ) / new_total
                            position_qty = new_total
                            pyramid_count = 1

            exit_triggered = False
            exit_reason = _FASTPATH_REASON_GENERAL

            if position_dir > 0:
                if exit_opposite and (debug_force_short or base_short_trigger[idx]) and bars_held >= min_hold_bars:
                    exit_triggered = True
                    exit_reason = _FASTPATH_REASON_OPPOSITE
                if use_time_stop and max_hold_bars > 0 and bars_held >= max_hold_bars:
                    exit_triggered = True
                    if exit_reason == _FASTPATH_REASON_GENERAL:
                        exit_reason = _FASTPATH_REASON_TIME
            elif position_dir < 0:
                if exit_opposite and (debug_force_long or base_long_trigger[idx]) and bars_held >= min_hold_bars:
                    exit_triggered = True
                    exit_reason = _FASTPATH_REASON_OPPOSITE
                if use_time_stop and max_hold_bars > 0 and bars_held >= max_hold_bars:
                    exit_triggered = True
                    if exit_reason == _FASTPATH_REASON_GENERAL:
                        exit_reason = _FASTPATH_REASON_TIME

            if exit_triggered and position_dir != 0:
                direction = position_dir
                exit_price = close[idx] - slip_value if direction > 0 else close[idx] + slip_value
                pnl = (exit_price - position_avg_price) * direction * position_qty
                fees = (position_avg_price + exit_price) * position_qty * commission_pct
                pnl -= fees
                equity += pnl
                net_profit += pnl
                if equity > peak_equity:
                    peak_equity = equity
                tradable_capital = equity
                if tradable_capital < min_equity_floor:
                    tradable_capital = min_equity_floor
                if initial_capital > 0.0:
                    returns[idx] += pnl / initial_capital
                trade_entry_idx[trade_count] = position_entry_idx
                trade_exit_idx[trade_count] = idx
                trade_direction[trade_count] = direction
                trade_qty[trade_count] = position_qty
                trade_entry_price[trade_count] = position_avg_price
                trade_exit_price[trade_count] = exit_price
                trade_pnl[trade_count] = pnl
                trade_reason[trade_count] = exit_reason
                trade_bars_held[trade_count] = bars_held
                trade_count += 1
                position_dir = 0
                position_qty = 0.0
                position_avg_price = 0.0
                position_entry_idx = -1
                bars_held = 0
                reentry_countdown = reentry_bars
                pyramid_count = 0
                base_entry_qty = 0.0
                continue

        if position_dir != 0:
            direction = position_dir
            exit_price = close[n - 1] - slip_value if direction > 0 else close[n - 1] + slip_value
            pnl = (exit_price - position_avg_price) * direction * position_qty
            fees = (position_avg_price + exit_price) * position_qty * commission_pct
            pnl -= fees
            equity += pnl
            net_profit += pnl
            if equity > peak_equity:
                peak_equity = equity
            tradable_capital = equity
            if tradable_capital < min_equity_floor:
                tradable_capital = min_equity_floor
            if initial_capital > 0.0:
                returns[n - 1] += pnl / initial_capital
            trade_entry_idx[trade_count] = position_entry_idx
            trade_exit_idx[trade_count] = n - 1
            trade_direction[trade_count] = direction
            trade_qty[trade_count] = position_qty
            trade_entry_price[trade_count] = position_avg_price
            trade_exit_price[trade_count] = exit_price
            trade_pnl[trade_count] = pnl
            trade_reason[trade_count] = _FASTPATH_REASON_GENERAL
            trade_bars_held[trade_count] = bars_held
            trade_count += 1

        return (
            equity,
            net_profit,
            peak_equity,
            tradable_capital,
            guard_frozen,
            returns,
            trade_entry_idx[:trade_count],
            trade_exit_idx[:trade_count],
            trade_direction[:trade_count],
            trade_qty[:trade_count],
            trade_entry_price[:trade_count],
            trade_exit_price[:trade_count],
            trade_pnl[:trade_count],
            trade_reason[:trade_count],
            trade_bars_held[:trade_count],
        )

else:  # NUMBA_AVAILABLE

    def _run_backtest_numba_fastpath(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("Numba is not available")


# =====================================================================================
# === 보조 계산 함수들 ===============================================================
# =====================================================================================









































































def _prepare_filter_context(df: pd.DataFrame, settings: FilterSettings) -> FilterContext:
    """각종 진입/필터 시그널을 미리 계산해 ``run_backtest`` 본문을 단순화합니다."""

    index = df.index

    # 변동성 가드 ATR 비율 ---------------------------------------------------------
    if settings.use_volatility_guard:
        atr_window = max(settings.volatility_lookback, 1)
        tr_values = _true_range(df).to_numpy(dtype=float)
        atr_series_values = _rolling_rma_last(tr_values, atr_window)
        close_values = df["close"].to_numpy(dtype=float)
        atr_pct_array = np.zeros(len(df), dtype=float)
        for pos in range(len(df)):
            if pos >= atr_window:
                atr_val = atr_series_values[pos]
                close_val = close_values[pos]
                if not np.isnan(atr_val) and close_val != 0.0:
                    atr_pct_array[pos] = atr_val / close_val * 100.0
        vol_guard_atr_pct = pd.Series(atr_pct_array, index=index)
    else:
        vol_guard_atr_pct = pd.Series(0.0, index=index)

    # ADX / ATR 차이 --------------------------------------------------------------
    if settings.use_adx or settings.use_atr_diff:
        adx_df = (
            _resample_ohlcv(df, settings.adx_atr_tf)
            if settings.adx_atr_tf not in {"", "0"}
            else df
        )
        _, _, adx_raw = _dmi(adx_df, settings.adx_len)
        adx_series = adx_raw.reindex(index, method="ffill").fillna(0.0)
        atr_htf = _atr(adx_df, settings.adx_len)
        atr_diff = (atr_htf - _sma(atr_htf, settings.adx_len)).reindex(index, method="ffill").fillna(0.0)
    else:
        adx_series = pd.Series(0.0, index=index)
        atr_diff = pd.Series(0.0, index=index)

    # EMA 기반 추세 ---------------------------------------------------------------
    if settings.use_ema:
        ema_fast = _ema(df["close"], settings.ema_fast_len)
        ema_slow = _ema(df["close"], settings.ema_slow_len)
    else:
        ema_fast = df["close"]
        ema_slow = df["close"]

    # 볼린저/밴드 필터 -------------------------------------------------------------
    if settings.use_bb_filter:
        bb_filter_basis = _sma(df["close"], settings.bb_filter_len)
        bb_filter_dev = _std(df["close"], settings.bb_filter_len)
        bb_filter_upper = bb_filter_basis + bb_filter_dev * settings.bb_filter_mult
        bb_filter_lower = bb_filter_basis - bb_filter_dev * settings.bb_filter_mult
    else:
        bb_filter_basis = df["close"]
        bb_filter_upper = df["close"]
        bb_filter_lower = df["close"]

    stoch_rsi = (
        _stoch_rsi(df["close"], settings.stoch_len)
        if settings.use_stoch_rsi
        else pd.Series(50.0, index=index)
    )
    obv_slope = (
        _obv_slope(df["close"], df["volume"], settings.obv_smooth_len)
        if settings.use_obv
        else pd.Series(0.0, index=index)
    )

    if settings.use_htf_trend:
        htf_ma = _security_series(
            df, settings.htf_trend_tf, lambda data: _ema(data["close"], settings.htf_ma_len)
        )
        htf_trend_up = df["close"] > htf_ma
        htf_trend_down = df["close"] < htf_ma
    else:
        htf_trend_up = pd.Series(True, index=index)
        htf_trend_down = pd.Series(True, index=index)

    hma_value = (
        _ema(df["close"], settings.hma_len)
        if settings.use_hma_filter
        else df["close"]
    )

    if settings.use_range_filter:
        range_high = _security_series(
            df,
            settings.range_tf,
            lambda data: data["high"].rolling(settings.range_bars).max(),
        )
        range_low = _security_series(
            df,
            settings.range_tf,
            lambda data: data["low"].rolling(settings.range_bars).min(),
        )
        range_perc = (range_high - range_low) / range_low.replace(0.0, np.nan) * 100.0
        in_range_box = range_perc <= settings.range_percent
    else:
        in_range_box = pd.Series(False, index=index)

    event_mask = pd.Series(False, index=index)
    if settings.use_event_filter and settings.event_windows:
        for segment in settings.event_windows.split(","):
            if "/" not in segment:
                continue
            start_str, end_str = segment.split("/", 1)
            try:
                start = pd.to_datetime(start_str.strip(), utc=True)
                end = pd.to_datetime(end_str.strip(), utc=True)
            except ValueError:
                continue
            if pd.isna(start) or pd.isna(end):
                continue
            if end < start:
                start, end = end, start
            event_mask |= (index >= start) & (index <= end)

    if settings.use_slope_filter:
        slope_basis = _ema(df["close"], settings.slope_lookback)
        slope_prev = slope_basis.shift(settings.slope_lookback).fillna(slope_basis)
        slope_pct = np.where(
            slope_basis != 0,
            (slope_basis - slope_prev) / slope_basis * 100.0,
            0.0,
        )
        slope_pct = pd.Series(slope_pct, index=index)
        slope_ok_long = slope_pct >= settings.slope_min_pct
        slope_ok_short = slope_pct <= -settings.slope_min_pct
    else:
        slope_ok_long = pd.Series(True, index=index)
        slope_ok_short = pd.Series(True, index=index)

    if settings.use_distance_guard:
        distance_atr = _atr(df, settings.distance_atr_len)
        vwap = df["close"].rolling(settings.distance_atr_len, min_periods=1).mean()
        vw_distance = (df["close"] - vwap).abs() / distance_atr.replace(0.0, np.nan)
        trend_ma = _ema(df["close"], settings.distance_trend_len)
        trend_distance = (df["close"] - trend_ma).abs() / distance_atr.replace(0.0, np.nan)
        distance_ok = (vw_distance <= settings.distance_max_atr) & (
            trend_distance <= settings.distance_max_atr
        )
    else:
        distance_ok = pd.Series(True, index=index)

    kasa_rsi = (
        _rsi(df["close"], settings.kasa_rsi_len)
        if settings.use_kasa
        else pd.Series(50.0, index=index)
    )

    if settings.use_regime_filter:
        ctx_close = _security_series(df, settings.ctx_htf_tf, lambda data: data["close"])
        ctx_ema = _security_series(
            df, settings.ctx_htf_tf, lambda data: _ema(data["close"], settings.ctx_htf_ema_len)
        )
        ctx_adx = _security_series(
            df, settings.ctx_htf_tf, lambda data: _dmi(data, settings.ctx_htf_adx_len)[2]
        )
        regime_long_ok = (ctx_close > ctx_ema) & (ctx_adx > settings.ctx_htf_adx_th)
        regime_short_ok = (ctx_close < ctx_ema) & (ctx_adx > settings.ctx_htf_adx_th)
    else:
        regime_long_ok = pd.Series(True, index=index)
        regime_short_ok = pd.Series(True, index=index)

    if settings.use_structure_gate:
        bos_high = _security_series(
            df,
            settings.bos_tf,
            lambda data: data["high"].rolling(
                settings.bos_lookback, min_periods=settings.bos_lookback
            ).max(),
        )
        bos_low = _security_series(
            df,
            settings.bos_tf,
            lambda data: data["low"].rolling(
                settings.bos_lookback, min_periods=settings.bos_lookback
            ).min(),
        )
        bos_high_ref = bos_high.shift()
        bos_low_ref = bos_low.shift()
        if settings.use_bos:
            bos_long_event = (df["close"] > bos_high_ref).where(~bos_high_ref.isna(), True)
            bos_short_event = (df["close"] < bos_low_ref).where(~bos_low_ref.isna(), True)
            bos_long_state = (
                bos_long_event.rolling(settings.bos_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
            bos_short_state = (
                bos_short_event.rolling(settings.bos_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
        else:
            bos_long_state = pd.Series(True, index=index)
            bos_short_state = pd.Series(True, index=index)

        pivot_high_ctx = _security_series(
            df,
            settings.bos_tf,
            lambda data: _pivot_series(
                data["high"], settings.pivot_left, settings.pivot_right, True
            ),
        )
        pivot_low_ctx = _security_series(
            df,
            settings.bos_tf,
            lambda data: _pivot_series(
                data["low"], settings.pivot_left, settings.pivot_right, False
            ),
        )
        if settings.use_choch:
            choch_long_event = (df["close"] > pivot_high_ctx).where(~pivot_high_ctx.isna(), True)
            choch_short_event = (df["close"] < pivot_low_ctx).where(~pivot_low_ctx.isna(), True)
            choch_long_state = (
                choch_long_event.rolling(settings.choch_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
            choch_short_state = (
                choch_short_event.rolling(settings.choch_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
        else:
            choch_long_state = pd.Series(True, index=index)
            choch_short_state = pd.Series(True, index=index)
    else:
        bos_long_state = pd.Series(True, index=index)
        bos_short_state = pd.Series(True, index=index)
        choch_long_state = pd.Series(True, index=index)
        choch_short_state = pd.Series(True, index=index)

    if settings.use_shock:
        atr_fast = _atr(df, settings.atr_fast_len)
        atr_slow = _atr(df, settings.atr_slow_len)
        shock_series = atr_fast > atr_slow * settings.shock_mult
    else:
        shock_series = pd.Series(False, index=index)

    return FilterContext(
        vol_guard_atr_pct=vol_guard_atr_pct,
        adx_series=adx_series,
        atr_diff=atr_diff,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        bb_filter_basis=bb_filter_basis,
        bb_filter_upper=bb_filter_upper,
        bb_filter_lower=bb_filter_lower,
        stoch_rsi=stoch_rsi,
        obv_slope=obv_slope,
        htf_trend_up=htf_trend_up,
        htf_trend_down=htf_trend_down,
        hma_value=hma_value,
        in_range_box=in_range_box,
        event_mask=event_mask,
        slope_ok_long=slope_ok_long,
        slope_ok_short=slope_ok_short,
        distance_ok=distance_ok,
        kasa_rsi=kasa_rsi,
        regime_long_ok=regime_long_ok,
        regime_short_ok=regime_short_ok,
        bos_long_state=bos_long_state,
        bos_short_state=bos_short_state,
        choch_long_state=choch_long_state,
        choch_short_state=choch_short_state,
        shock_series=shock_series,
    )


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float | bool],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """TradingView `매직1분VN` 최종본과 동등한 파이썬 백테스트."""

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain OHLCV columns")

    def _ensure_datetime_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if isinstance(frame.index, pd.DatetimeIndex):
            idx = frame.index
            if idx.tz is None:
                frame = frame.copy()
                frame.index = idx.tz_localize("UTC")
            return frame

        frame = frame.copy()
        if "timestamp" in frame.columns:
            converted = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            valid_mask = converted.notna()
            invalid = (~valid_mask).sum()
            if invalid:
                LOGGER.warning(
                    "%s 데이터프레임에서 timestamp 컬럼의 %d개 행을 제거했습니다.",
                    label,
                    int(invalid),
                )
            if valid_mask.any():
                frame = frame.loc[valid_mask].copy()
                frame.index = converted[valid_mask]
                frame.drop(columns=["timestamp"], inplace=True)
                return frame
        raise TypeError(
            f"{label} 데이터프레임은 DatetimeIndex 를 가져야 합니다. "
            "timestamp 컬럼이 있다면 UTC 로 변환한 뒤 다시 실행해주세요."
        )

    df = _ensure_datetime_index(df, "가격")
    if htf_df is not None:
        htf_df = _ensure_datetime_index(htf_df, "HTF")

    def _normalise_ohlcv(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        frame = frame.copy()
        frame.sort_index(inplace=True)

        if frame.index.has_duplicates:
            dup_count = int(frame.index.duplicated(keep="last").sum())
            if dup_count:
                LOGGER.warning("%s 데이터프레임에서 중복 인덱스 %d개를 제거합니다.", label, dup_count)
            frame = frame[~frame.index.duplicated(keep="last")]

        for column in required_cols:
            if column not in frame.columns:
                continue
            coerced = pd.to_numeric(frame[column], errors="coerce")
            invalid_count = int((coerced.isna() & frame[column].notna()).sum())
            if invalid_count:
                LOGGER.warning(
                    "%s 데이터프레임의 %s 열에서 비수치 값 %d개를 NaN 으로 치환했습니다.",
                    label,
                    column,
                    invalid_count,
                )
            frame[column] = coerced

        before = len(frame)
        frame = frame.dropna(subset=list(required_cols))
        dropped = before - len(frame)
        if dropped:
            LOGGER.warning("%s 데이터프레임에서 결측 OHLCV 행 %d개를 제거했습니다.", label, int(dropped))

        if len(frame) < 2:
            raise ValueError(f"{label} 데이터가 부족하여 백테스트를 진행할 수 없습니다.")

        return frame

    # Determine the user-supplied start date if present. We don't want to
    # blindly default to a date far in the future relative to the data
    # because doing so causes the backtest loop to skip all bars and
    # produce zero trades. Instead, we parse the supplied value here so that it
    # exists before we adjust it against the dataset start timestamp below.
    _start_raw = params.get("startDate")
    try:
        start_ts = pd.to_datetime(_start_raw, utc=True) if _start_raw else pd.NaT
    except Exception:
        start_ts = pd.NaT
    df = _normalise_ohlcv(df, "가격")
    # Ensure that the start date is not in the future relative to the data.
    # If no start date was provided or the provided date is earlier than the
    # first bar, align it to the first timestamp of the dataset to avoid skipping
    # the entire dataset and returning zero trades.
    if not isinstance(start_ts, pd.Timestamp) or pd.isna(start_ts) or start_ts < df.index[0]:
        start_ts = df.index[0]
    if htf_df is not None:
        htf_df = _normalise_ohlcv(htf_df, "HTF")

    def _coerce_bool(value: object, default: bool) -> bool:
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

    def bool_param(name: str, default: bool, *, enabled: bool = True) -> bool:
        if not enabled:
            return default
        return _coerce_bool(params.get(name, default), default)

    def int_param(name: str, default: int, *, enabled: bool = True) -> int:
        if not enabled:
            return int(default)
        value = params.get(name, default)
        if isinstance(value, bool):
            return int(value)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    def float_param(name: str, default: float, *, enabled: bool = True) -> float:
        if not enabled:
            return float(default)
        value = params.get(name, default)
        if isinstance(value, bool):
            return float(int(value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _resolve_float_param(
        names: Iterable[str],
        default: float,
        *,
        enabled: bool = True,
    ) -> float:
        if not enabled:
            return float(default)
        fallback = _coerce_float_value(default, 0.0)
        if not np.isfinite(fallback):
            fallback = 0.0
        for source in (params, risk):
            if not isinstance(source, dict):
                continue
            for name in names:
                if name not in source or source[name] is None:
                    continue
                candidate = _coerce_float_value(source[name], fallback)
                if np.isfinite(candidate):
                    return float(candidate)
        return float(fallback)

    def _coerce_float_value(value: object, default: float) -> float:
        """임의의 값을 ``float`` 로 강제 변환합니다."""

        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(int(value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _resolve_penalty_value(*names: str, default: float) -> float:
        fallback = float(default)
        if not np.isfinite(fallback) or fallback < 0:
            fallback = 1.0
        for source in (risk, params):
            if not isinstance(source, dict):
                continue
            for name in names:
                if name not in source or source[name] is None:
                    continue
                candidate = _coerce_float_value(source[name], fallback)
                if not np.isfinite(candidate):
                    continue
                return float(abs(candidate))
        return abs(fallback)

    def _resolve_requirement_value(*names: str, default: float) -> float:
        fallback = float(default)
        if not np.isfinite(fallback):
            fallback = 0.0
        for source in (risk, params):
            if not isinstance(source, dict):
                continue
            for name in names:
                if name not in source or source[name] is None:
                    continue
                candidate = _coerce_float_value(source[name], fallback)
                if np.isfinite(candidate):
                    return float(candidate)
        return fallback


    def str_param(name: str, default: str, *, enabled: bool = True) -> str:
        """문자열 파라미터를 안전하게 변환한다.

        Pine/Optuna 설정이 잘못 직렬화되면 문자열 입력이 리스트/튜플로
        들어오는 경우가 있는데, 이때는 첫 번째 요소만 허용하거나 명시적으로
        오류를 발생시켜 잘못된 설정이 그대로 진행되지 않도록 한다.
        """

        if not enabled:
            return str(default)

        value = params.get(name, default)
        if isinstance(value, (list, tuple)):
            if not value:
                value = default
            elif len(value) == 1:
                value = value[0]
            else:
                raise TypeError(
                    f"'{name}' 파라미터에 여러 개의 문자열이 전달되었습니다: {value!r}"
                )

        return str(value) if value is not None else str(default)

    # Pine 입력 매핑 -----------------------------------------------------------------
    osc_len = int_param("oscLen", 20)
    sig_len = int_param("signalLen", 3)
    use_same_len = bool_param("useSameLen", False)
    kc_len = osc_len if use_same_len else int_param("kcLen", 18)
    kc_mult = float_param("kcMult", 1.0)
    if use_same_len:
        bb_len = osc_len
    else:
        bb_len = int_param("bbLen", kc_len)
    bb_mult = float_param("bbMult", kc_mult)

    flux_len = int_param("fluxLen", 14)
    flux_smooth_len = int_param("fluxSmoothLen", 1)
    flux_deadzone = float_param("fluxDeadzone", 25.0)
    flux_use_ha = bool_param("useFluxHeikin", True)

    # 모디파이드 플럭스 및 스퀴즈 사용 여부와 신호선 타입
    use_mod_flux = bool_param("useModFlux", False)
    compat_mode = bool_param("compatMode", True)
    auto_threshold_scale = bool_param("autoThresholdScale", True)
    use_norm_clip = bool_param("useNormClip", False)
    norm_clip_limit = float_param("normClipLimit", 350.0)
    # 선택할 모멘텀 계산 스타일. pine 파라미터 이름은 basisStyle 로 변경되었으나
    # 기존 YAML 호환을 위해 momStyle도 함께 확인합니다.
    basis_style_raw = str_param("basisStyle", None)
    if basis_style_raw in {None, ""}:
        basis_style_raw = str_param("momStyle", "Deluxe")
    mom_style_raw = basis_style_raw
    # 정규화: 영문/한글 입력을 모두 처리하도록 소문자로 변환 후 매핑합니다. 기본값은 "deluxe" 스타일입니다.
    mom_style_tmp = str(mom_style_raw).strip().lower()
    if mom_style_tmp in {"kc", "basic", "원본", "기본"}:
        mom_style = "kc"
    elif mom_style_tmp in {"avg", "average", "avgstyle", "average_style"}:
        mom_style = "avg"
    elif mom_style_tmp in {"deluxe", "dx", "delux", "디럭스"}:
        mom_style = "deluxe"
    elif mom_style_tmp in {"mod", "modified", "modsqueeze", "모디파이드", "modded"}:
        mom_style = "mod"
    else:
        mom_style = "deluxe"
    # -------------------------------------------------------------------------
    # Momentum signal MA type.  Accept both the English identifiers (SMA, EMA,
    # HMA) and the Korean label used in the Pine script ("기본").  Some users
    # reported that the maType parameter was not being honoured because the
    # default value in the YAML profile is "SMA" while the Pine script uses
    # "기본" (meaning "basic").  To ensure both strings resolve to the same
    # behaviour we normalise the value here.  If the parameter is unset or
    # cannot be parsed, fall back to SMA.
    ma_type = str_param("maType", "SMA")
    if ma_type is None:
        ma_type = "SMA"
    # Normalise to lower‑case and map Korean/English synonyms to SMA
    mt_normalised = str(ma_type).strip().lower()
    if mt_normalised in {"기본", "basic", "sma", "default"}:
        ma_type = "SMA"
    elif mt_normalised in {"ema"}:
        ma_type = "EMA"
    elif mt_normalised in {"hma"}:
        ma_type = "HMA"
    else:
        # Unknown value – default to SMA
        ma_type = "SMA"

    use_dynamic_thresh = bool_param("useDynamicThresh", True)
    use_sym_threshold = bool_param("useSymThreshold", False)
    stat_threshold = float_param("statThreshold", 38.0)
    buy_threshold = float_param("buyThreshold", 36.0)
    sell_threshold = float_param("sellThreshold", 36.0)
    dyn_len = int_param("dynLen", 21, enabled=use_dynamic_thresh)
    dyn_mult = float_param("dynMult", 1.1, enabled=use_dynamic_thresh)
    # 전략 명세에 따라 모멘텀 교차 필터는 항상 사용한다.
    require_momentum_cross = True
    # Optional toggle: enable Numba-accelerated cross calculations.  When
    # ``useNumba`` is True and Numba is installed in the environment,
    # momentum/signal cross-over arrays will be computed using a
    # Numba-jitted loop for improved performance.  Otherwise, a pure
    # Pandas vectorised approach will be used.  Defaults to False.
    # 기본값을 True 로 변경하여 Numba 가 사용 가능한 경우 교차 계산을 가속화합니다.
    use_numba = bool_param("useNumba", True)

    # -------------------------------------------------------------------------
    # 추가 손절 및 레버리지 파라미터 불러오기
    # fixedStopPct: 고정 퍼센트 손절. 0이면 비활성화됩니다.
    fixed_stop_pct_val = float_param("fixedStopPct", 0.0)
    # atrStopLen: ATR 손절 길이. 0이면 ATR 손절을 비활성화합니다.
    atr_stop_len_val = int_param("atrStopLen", 0, enabled=True)
    # atrStopMult: ATR 손절 배수.
    atr_stop_mult_val = float_param("atrStopMult", 1.0)
    stop_channel_type_raw = str_param("stopChannelType", "None")
    stop_channel_mult_val = float_param("stopChannelMult", 1.0)

    def _normalise_channel_mode(value: str) -> Optional[str]:
        text = (value or "").strip().lower()
        if not text or text in {"none", "null", "nan"}:
            return None
        if text in {"bb", "bollinger", "bollingerband", "bollinger_band"}:
            return "BB"
        if text in {"kc", "keltner", "keltnerchannel", "keltner_channel"}:
            return "KC"
        return None

    stop_channel_mode = _normalise_channel_mode(stop_channel_type_raw)
    use_channel_stop = bool(stop_channel_mode) and stop_channel_mult_val > 0.0
    # salvagePct: 레버리지 대비 청산 보정치(10% = 0.1). 현재 고정값으로 사용합니다.
    salvage_pct = 0.10
    # profit_deposit_pct: 수익의 몇 퍼센트를 저축 계좌에 적립할지 (20% = 0.20)
    profit_deposit_pct = 0.20
    # RUIN 임계값. 전체 자산(available_capital + savings)이 이 값 미만이면 백테스트를 중단합니다.
    ruin_threshold = 50.0

    # (start_ts will be initialised prior to data normalisation below)

    leverage = resolve_leverage(params, risk)
    commission_pct = float(fees.get("commission_pct", params.get("commission_value", 0.0005)))
    slippage_ticks = float(fees.get("slippage_ticks", params.get("slipTicks", 1)))
    initial_capital = float(risk.get("initial_capital", params.get("initial_capital", 500.0)))

    allow_long_entry = bool_param("allowLongEntry", True)
    allow_short_entry = bool_param("allowShortEntry", True)
    debug_force_long = bool_param("debugForceLong", False)
    debug_force_short = bool_param("debugForceShort", False)
    reentry_bars = int_param("reentryBars", 0)

    if bool_param("useSessionFilter", False):
        LOGGER.warning("세션 필터 기능은 안정성 문제로 인해 현재 비활성화됩니다.")
    if bool_param("useDayFilter", False):
        LOGGER.warning("요일 필터 기능은 안정성 문제로 인해 현재 비활성화됩니다.")
    if bool_param("useEventFilter", False):
        LOGGER.warning("이벤트 필터 기능은 안정성 문제로 인해 현재 비활성화됩니다.")

    # 리스크/지갑 --------------------------------------------------------------------
    base_qty_percent = float_param("baseQtyPercent", 30.0)
    use_pyramiding = bool_param("usePyramiding", False)
    qty_override = risk.get("qty_pct")
    if qty_override is None:
        qty_override = risk.get("qtyPercent")
    if qty_override is not None:
        resolved_qty = _coerce_float_value(qty_override, base_qty_percent)
        if np.isfinite(resolved_qty):
            base_qty_percent = resolved_qty
    use_sizing_override = bool_param("useSizingOverride", False)
    sizing_mode = str_param("sizingMode", "자본 비율")
    advanced_percent = float_param("advancedPercent", 25.0, enabled=use_sizing_override)
    fixed_usd_amount = float_param("fixedUsdAmount", 100.0, enabled=use_sizing_override)
    fixed_contract_size = float_param("fixedContractSize", 1.0, enabled=use_sizing_override)
    risk_sizing_type = str_param("riskSizingType", "손절 기반 %", enabled=use_sizing_override)
    base_risk_pct = float_param("baseRiskPct", 0.6)
    risk_contract_size = float_param("riskContractSize", 1.0, enabled=use_sizing_override)
    use_wallet = bool_param("useWallet", False)
    profit_reserve_pct = (
        float_param("profitReservePct", 20.0, enabled=use_wallet) / 100.0 if use_wallet else 0.0
    )
    apply_reserve_to_sizing = bool_param("applyReserveToSizing", True, enabled=use_wallet)
    min_tradable_capital = float_param("minTradableCapital", 250.0)
    use_drawdown_scaling = bool_param("useDrawdownScaling", False)
    drawdown_trigger_pct = float_param("drawdownTriggerPct", 7.0, enabled=use_drawdown_scaling)
    drawdown_risk_scale = float_param("drawdownRiskScale", 0.5, enabled=use_drawdown_scaling)

    use_perf_adaptive_risk = bool_param("usePerfAdaptiveRisk", False)
    par_lookback = int_param("parLookback", 6, enabled=use_perf_adaptive_risk)
    par_min_trades = int_param("parMinTrades", 3, enabled=use_perf_adaptive_risk)
    par_hot_win_rate = float_param("parHotWinRate", 65.0, enabled=use_perf_adaptive_risk)
    par_cold_win_rate = float_param("parColdWinRate", 35.0, enabled=use_perf_adaptive_risk)
    par_hot_mult = _resolve_float_param(
        ("parHotMult", "parHotRiskMult"),
        1.25,
        enabled=use_perf_adaptive_risk,
    )
    par_cold_mult = _resolve_float_param(
        ("parColdMult", "parColdRiskMult"),
        0.35,
        enabled=use_perf_adaptive_risk,
    )
    par_pause_on_cold = bool_param("parPauseOnCold", True, enabled=use_perf_adaptive_risk)

    if min_trades is not None:
        min_trades_default = _coerce_float_value(min_trades, 0.0)
    else:
        min_trades_default = _coerce_float_value(params.get("minTrades"), 0.0)
    if not np.isfinite(min_trades_default):
        min_trades_default = 0.0
    if min_trades is not None:
        min_trades_value = min_trades_default
    else:
        min_trades_value = _resolve_requirement_value(
            "min_trades",
            "minTrades",
            "minTradesReq",
            default=min_trades_default,
        )
    try:
        min_trades_req = max(0, int(float(min_trades_value)))
    except (TypeError, ValueError, OverflowError):
        min_trades_req = max(0, int(min_trades_default))
    try:
        min_trades_value_float = float(min_trades_value)
    except (TypeError, ValueError, OverflowError):
        min_trades_value_float = float(min_trades_default)
    if not np.isfinite(min_trades_value_float):
        min_trades_value_float = float(min_trades_default)

    use_daily_loss_guard = bool_param("useDailyLossGuard", False)
    daily_loss_limit = float_param("dailyLossLimit", 80.0)
    use_daily_profit_lock = bool_param("useDailyProfitLock", False)
    daily_profit_target = float_param("dailyProfitTarget", 120.0)
    use_weekly_profit_lock = bool_param("useWeeklyProfitLock", False)
    weekly_profit_target = float_param("weeklyProfitTarget", 250.0)
    use_loss_streak_guard = bool_param("useLossStreakGuard", False)
    max_consecutive_loss_default = int_param("maxConsecutiveLosses", 3)
    max_consecutive_value = _resolve_requirement_value(
        "max_consecutive_losses",
        "maxConsecutiveLosses",
        "maxLossStreak",
        default=float(max_consecutive_loss_default),
    )
    try:
        max_consecutive_losses = max(0, int(float(max_consecutive_value)))
    except (TypeError, ValueError, OverflowError):
        max_consecutive_losses = max(0, int(max_consecutive_loss_default))
    try:
        max_consecutive_value_float = float(max_consecutive_value)
    except (TypeError, ValueError, OverflowError):
        max_consecutive_value_float = float(max_consecutive_loss_default)
    if not np.isfinite(max_consecutive_value_float):
        max_consecutive_value_float = float(max_consecutive_loss_default)

    trade_penalty_value = _resolve_penalty_value(
        "penalty_trade",
        "penaltyTrade",
        "tradePenalty",
        default=0.0,
    )
    hold_penalty_value = _resolve_penalty_value(
        "penalty_hold",
        "penaltyHold",
        "holdPenalty",
        default=0.0,
    )
    consecutive_penalty_value = _resolve_penalty_value(
        "penalty_consecutive_loss",
        "penaltyConsecutiveLoss",
        "consecutiveLossPenalty",
        default=0.0,
    )

    min_hold_default = _coerce_float_value(params.get("minHoldBars"), 0.0)
    if not np.isfinite(min_hold_default):
        min_hold_default = 0.0
    min_hold_value = _resolve_requirement_value(
        "min_hold_bars",
        "minHoldBars",
        "minHold",
        default=min_hold_default,
    )
    try:
        min_hold_bars_param = max(0, int(float(min_hold_value)))
    except (TypeError, ValueError, OverflowError):
        min_hold_bars_param = max(0, int(min_hold_default))
    try:
        min_hold_value_float = float(min_hold_value)
    except (TypeError, ValueError, OverflowError):
        min_hold_value_float = float(min_hold_default)
    if not np.isfinite(min_hold_value_float):
        min_hold_value_float = float(min_hold_default)

    penalty_config = {
        "min_trades_value": min_trades_value_float,
        "min_hold_value": min_hold_value_float,
        "max_loss_streak": max_consecutive_value_float,
        "trade_penalty": trade_penalty_value,
        "hold_penalty": hold_penalty_value,
        "consecutive_loss_penalty": consecutive_penalty_value,
    }

    validity_thresholds = {
        "min_trades": float(min_trades_req),
        "min_hold_bars": float(min_hold_bars_param),
        "max_consecutive_losses": float(max_consecutive_losses),
    }
    use_capital_guard = bool_param("useCapitalGuard", False)
    capital_guard_pct = float_param("capitalGuardPct", 20.0)
    max_daily_losses = int_param("maxDailyLosses", 0)
    max_weekly_dd = float_param("maxWeeklyDD", 0.0)
    max_guard_fires = int_param("maxGuardFires", 0)
    use_guard_exit = bool_param("useGuardExit", False)
    maintenance_margin_pct = float_param("maintenanceMarginPct", 0.5)
    preempt_ticks = int_param("preemptTicks", 8)
    liq_buffer_raw = _coerce_float_value(risk.get("liq_buffer_pct"), 0.0)
    if not np.isfinite(liq_buffer_raw):
        liq_buffer_raw = 0.0
    liq_buffer_pct = max(liq_buffer_raw, 0.0)

    simple_metrics_only = (
        bool_param("simpleMetricsOnly", False)
        or bool_param("simpleProfitOnly", False)
        or _coerce_bool(risk.get("simpleMetricsOnly"), False)
        or _coerce_bool(risk.get("simpleProfitOnly"), False)
    )

    use_volatility_guard = bool_param("useVolatilityGuard", False)
    volatility_lookback = int_param("volatilityLookback", 50, enabled=use_volatility_guard)
    volatility_lower_pct = float_param("volatilityLowerPct", 0.15, enabled=use_volatility_guard)
    volatility_upper_pct = float_param("volatilityUpperPct", 2.5, enabled=use_volatility_guard)

    # 필터 옵션 ---------------------------------------------------------------------
    use_adx = bool_param("useAdx", False)
    use_atr_diff = bool_param("useAtrDiff", False)
    adx_len = int_param("adxLen", 10, enabled=use_adx or use_atr_diff)
    adx_thresh = float_param("adxThresh", 15.0, enabled=use_adx)
    use_ema = bool_param("useEma", False)
    ema_fast_len = int_param("emaFastLen", 8, enabled=use_ema)
    ema_slow_len = int_param("emaSlowLen", 20, enabled=use_ema)
    ema_mode = str_param("emaMode", "Trend", enabled=use_ema)
    use_bb_filter = bool_param("useBb", False)
    bb_filter_len = int_param("bbLenFilter", 20, enabled=use_bb_filter)
    bb_filter_mult = float_param("bbMultFilter", 2.0, enabled=use_bb_filter)
    use_stoch_rsi = bool_param("useStochRsi", False)
    stoch_len = int_param("stochLen", 14, enabled=use_stoch_rsi)
    stoch_ob = float_param("stochOB", 80.0, enabled=use_stoch_rsi)
    stoch_os = float_param("stochOS", 20.0, enabled=use_stoch_rsi)
    use_obv = bool_param("useObv", False)
    obv_smooth_len = int_param("obvSmoothLen", 3, enabled=use_obv)
    adx_atr_tf = str_param("adxAtrTf", "5", enabled=use_adx or use_atr_diff)
    use_htf_trend = bool_param("useHtfTrend", False)
    htf_trend_tf = str_param("htfTrendTf", "240", enabled=use_htf_trend)
    htf_ma_len = int_param("htfMaLen", 20, enabled=use_htf_trend)
    use_hma_filter = bool_param("useHmaFilter", False)
    hma_len = int_param("hmaLen", 20, enabled=use_hma_filter)
    use_range_filter = bool_param("useRangeFilter", False)
    range_tf = str_param("rangeTf", "5", enabled=use_range_filter)
    range_bars = int_param("rangeBars", 20, enabled=use_range_filter)
    range_percent = float_param("rangePercent", 1.0, enabled=use_range_filter)
    use_event_filter = False  # 이벤트 필터는 안정성 문제로 비활성화
    event_windows_raw = str_param("eventWindows", "", enabled=use_event_filter)

    use_regime_filter = bool_param("useRegimeFilter", False)
    ctx_htf_tf = str_param("ctxHtfTf", "240", enabled=use_regime_filter)
    ctx_htf_ema_len = int_param("ctxHtfEmaLen", 120, enabled=use_regime_filter)
    ctx_htf_adx_len = int_param("ctxHtfAdxLen", 14, enabled=use_regime_filter)
    ctx_htf_adx_th = float_param("ctxHtfAdxTh", 22.0, enabled=use_regime_filter)
    use_slope_filter = bool_param("useSlopeFilter", False)
    slope_lookback = int_param("slopeLookback", 8, enabled=use_slope_filter)
    slope_min_pct = float_param("slopeMinPct", 0.06, enabled=use_slope_filter)
    use_distance_guard = bool_param("useDistanceGuard", False)
    distance_atr_len = int_param("distanceAtrLen", 21, enabled=use_distance_guard)
    distance_trend_len = int_param("distanceTrendLen", 55, enabled=use_distance_guard)
    distance_max_atr = float_param("distanceMaxAtr", 2.4, enabled=use_distance_guard)
    use_equity_slope_filter = bool_param("useEquitySlopeFilter", False)
    eq_slope_len = int_param("eqSlopeLen", 120, enabled=use_equity_slope_filter)

    use_sqz_gate = bool_param("useSqzGate", False)
    sqz_release_bars = int_param("sqzReleaseBars", 5, enabled=use_sqz_gate)
    use_structure_gate = bool_param("useStructureGate", False)
    structure_gate_mode = str_param("structureGateMode", "어느 하나 충족", enabled=use_structure_gate)
    use_bos = bool_param("useBOS", False, enabled=use_structure_gate)
    use_choch = bool_param("useCHOCH", False, enabled=use_structure_gate)
    bos_state_bars = int_param("bos_stateBars", 5, enabled=use_bos)
    choch_state_bars = int_param("choch_stateBars", 5, enabled=use_choch)
    bos_tf = str_param("bosTf", "15", enabled=use_bos or use_choch)
    bos_lookback = int_param("bosLookback", 50, enabled=use_bos)
    pivot_left = int_param("pivotLeft_vn", 5, enabled=use_bos or use_choch)
    pivot_right = int_param("pivotRight_vn", 5, enabled=use_bos or use_choch)

    use_reversal = bool_param("useReversal", False)
    reversal_delay_sec = float_param("reversalDelaySec", 0.0, enabled=use_reversal)

    # 출구 옵션 ---------------------------------------------------------------------
    exit_opposite = bool_param("exitOpposite", True)
    use_mom_fade = bool_param("useMomFade", False)
    mom_fade_min_abs = float_param("momFadeMinAbs", 0.0, enabled=use_mom_fade)

    use_stop_loss = bool_param("useStopLoss", False)
    stop_lookback = int_param("stopLookback", 5, enabled=use_stop_loss)
    use_atr_trail = bool_param("useAtrTrail", False)
    atr_trail_len = int_param("atrTrailLen", 7, enabled=use_atr_trail)
    atr_trail_mult = float_param("atrTrailMult", 2.5, enabled=use_atr_trail)
    use_chandelier_exit = bool_param("useChandelierExit", False)
    chandelier_len = int_param("chandelierLen", 22, enabled=use_chandelier_exit)
    chandelier_mult = float_param("chandelierMult", 3.0, enabled=use_chandelier_exit)
    use_sar_exit = bool_param("useSarExit", False)
    sar_start = float_param("sarStart", 0.02, enabled=use_sar_exit)
    sar_increment = float_param("sarIncrement", 0.02, enabled=use_sar_exit)
    sar_maximum = float_param("sarMaximum", 0.2, enabled=use_sar_exit)
    use_breakeven_stop = bool_param("useBreakevenStop", False)
    breakeven_mult = float_param("breakevenMult", 1.0, enabled=use_breakeven_stop)
    use_pivot_stop = bool_param("usePivotStop", False)
    pivot_len = int_param("pivotLen", 5, enabled=use_pivot_stop)
    use_pivot_htf = bool_param("usePivotHtf", False, enabled=use_pivot_stop)
    pivot_tf = str_param("pivotTf", "5", enabled=use_pivot_stop)
    use_atr_profit = bool_param("useAtrProfit", False)
    atr_profit_mult = float_param("atrProfitMult", 2.0, enabled=use_atr_profit)
    use_dyn_vol = bool_param("useDynVol", False)
    use_stop_distance_guard = bool_param("useStopDistanceGuard", False)
    max_stop_atr_mult = float_param("maxStopAtrMult", 2.8, enabled=use_stop_distance_guard)
    use_time_stop = bool_param("useTimeStop", False)
    max_hold_bars = int_param("maxHoldBars", 45, enabled=use_time_stop)
    use_kasa = bool_param("useKASA", False)
    kasa_rsi_len = int_param("kasa_rsiLen", 14, enabled=use_kasa)
    kasa_rsi_ob = float_param("kasa_rsiOB", 72.0, enabled=use_kasa)
    kasa_rsi_os = float_param("kasa_rsiOS", 28.0, enabled=use_kasa)
    use_be_tiers = bool_param("useBETiers", False)

    use_shock = bool_param("useShock", False)
    atr_fast_len = int_param("atrFastLen", 5, enabled=use_shock)
    atr_slow_len = int_param("atrSlowLen", 20, enabled=use_shock)
    shock_mult = float_param("shockMult", 2.5, enabled=use_shock)
    shock_action = str_param("shockAction", "손절 타이트닝", enabled=use_shock)

    # RUIN 발생 여부를 추적합니다. 청산 및 손절 후 전체 자산이 ruin_threshold 미만이면 True 로 설정
    ruin_hit: bool = False


    def _attempt_fastpath() -> Optional[Dict[str, float]]:
        if not (use_numba and NUMBA_AVAILABLE):
            return None
        blockers: List[str] = []
        feature_flags = {
            "wallet": use_wallet,
            "drawdown_scaling": use_drawdown_scaling,
            "performance_risk": use_perf_adaptive_risk,
            "daily_loss_guard": use_daily_loss_guard,
            "daily_profit_lock": use_daily_profit_lock,
            "weekly_profit_lock": use_weekly_profit_lock,
            "loss_streak_guard": use_loss_streak_guard,
            "capital_guard": use_capital_guard,
            "max_daily_losses": max_daily_losses > 0,
            "max_guard_fires": max_guard_fires > 0,
            "guard_exit": use_guard_exit,
            "volatility_guard": use_volatility_guard,
            "mom_fade": use_mom_fade,
            "stop_loss": use_stop_loss,
            "atr_trail": use_atr_trail,
            "breakeven_stop": use_breakeven_stop,
            "pivot_stop": use_pivot_stop,
            "atr_profit": use_atr_profit,
            "dynamic_vol": use_dyn_vol,
            "stop_distance_guard": use_stop_distance_guard,
            "kasa_exit": use_kasa,
            "be_tiers": use_be_tiers,
            "shock_guard": use_shock,
            "reversal": use_reversal,
            "equity_slope_filter": use_equity_slope_filter,
            "sqz_gate": use_sqz_gate,
            "structure_gate": use_structure_gate,
            "event_filter": use_event_filter,
            "distance_guard": use_distance_guard,
            "regime_filter": use_regime_filter,
            "adx_filter": use_adx,
            "atr_diff": use_atr_diff,
            "ema_filter": use_ema,
            "bb_filter": use_bb_filter,
            "stoch_rsi": use_stoch_rsi,
            "obv_filter": use_obv,
            "htf_trend": use_htf_trend,
            "hma_filter": use_hma_filter,
            "range_filter": use_range_filter,
            "sizing_override": use_sizing_override,
            "capital_locking": True,
        }
        for name, enabled in feature_flags.items():
            if enabled:
                blockers.append(name)
        if blockers:
            LOGGER.debug("Numba 패스트패스를 사용할 수 없습니다. 비활성화된 기능: %s", ", ".join(sorted(blockers)))
            return None
        try:
            close_array = df["close"].to_numpy(dtype=np.float64)
            momentum_array = momentum.to_numpy(dtype=np.float64)
            buy_thresh_array = buy_thresh_series.to_numpy(dtype=np.float64)
            sell_thresh_array = sell_thresh_series.to_numpy(dtype=np.float64)
            flux_array = flux_hist.to_numpy(dtype=np.float64)
            cross_up_array = _cross_up_series.to_numpy(dtype=np.bool_)
            cross_down_array = _cross_dn_series.to_numpy(dtype=np.bool_)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Numba 패스트패스 준비 중 예외 발생: %s", exc)
            return None
        long_cross_ok = cross_up_array
        short_cross_ok = cross_down_array
        base_long_trigger = long_cross_ok & (momentum_array < buy_thresh_array) & (flux_array > 0.0)
        base_short_trigger = short_cross_ok & (momentum_array > sell_thresh_array) & (flux_array < 0.0)
        base_long_trigger = np.ascontiguousarray(base_long_trigger, dtype=np.bool_)
        base_short_trigger = np.ascontiguousarray(base_short_trigger, dtype=np.bool_)
        index_ns = df.index.asi8
        start_value = int(start_ts.value)
        start_idx = int(np.searchsorted(index_ns, start_value, side="left")) if len(index_ns) else 0
        day_codes = np.ascontiguousarray(index_ns // 86_400_000_000_000, dtype=np.int64)
        min_equity_floor = float(initial_capital) * 0.01
        try:
            (
                equity_val,
                net_profit_val,
                peak_equity_val,
                tradable_capital_val,
                guard_flag,
                returns_array,
                entry_idx_arr,
                exit_idx_arr,
                direction_arr,
                qty_arr,
                entry_price_arr,
                exit_price_arr,
                pnl_arr,
                reason_arr,
                bars_arr,
            ) = _run_backtest_numba_fastpath(
                np.ascontiguousarray(close_array, dtype=np.float64),
                base_long_trigger,
                base_short_trigger,
                bool(allow_long_entry),
                bool(allow_short_entry),
                bool(debug_force_long),
                bool(debug_force_short),
                bool(exit_opposite),
                bool(use_time_stop),
                int(max_hold_bars),
                int(min_hold_bars_param),
                int(reentry_bars),
                bool(use_pyramiding),
                float(commission_pct),
                float(slip_value),
                float(leverage),
                float(base_qty_percent),
                float(initial_capital),
                start_idx,
                float(min_tradable_capital),
                day_codes,
                float(min_equity_floor),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Numba 패스트패스 실행 중 예외 발생: %s", exc)
            return None

        returns_series = pd.Series(returns_array, index=df.index)
        trades: List[Trade] = []
        total_trades = len(entry_idx_arr)
        for i in range(total_trades):
            entry_idx = int(entry_idx_arr[i])
            exit_idx = int(exit_idx_arr[i])
            direction = int(direction_arr[i])
            qty = float(qty_arr[i])
            entry_price = float(entry_price_arr[i])
            exit_price = float(exit_price_arr[i])
            pnl = float(pnl_arr[i])
            bars = int(bars_arr[i])
            reason_code = int(reason_arr[i])
            if direction > 0:
                reason_text = "Exit Long"
            else:
                reason_text = "Exit Short"
            if reason_code == _FASTPATH_REASON_OPPOSITE:
                reason_text = "opposite_signal"
            elif reason_code == _FASTPATH_REASON_TIME:
                reason_text = "time_stop"
            trades.append(
                Trade(
                    entry_time=df.index[entry_idx],
                    exit_time=df.index[exit_idx],
                    direction="long" if direction > 0 else "short",
                    size=qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit=pnl,
                    return_pct=pnl / initial_capital if initial_capital else 0.0,
                    mfe=np.nan,
                    mae=np.nan,
                    bars_held=bars,
                    reason=reason_text,
                )
            )

        state = EquityState(
            initial_capital=initial_capital,
            equity=float(equity_val),
            net_profit=float(net_profit_val),
            withdrawable=0.0,
            tradable_capital=float(tradable_capital_val),
            peak_equity=float(peak_equity_val),
            daily_start_capital=initial_capital,
            daily_peak_capital=max(initial_capital, float(peak_equity_val)),
            week_start_equity=initial_capital,
            week_peak_equity=float(peak_equity_val),
            available_capital=float(tradable_capital_val),
            savings=0.0,
            liquidations=0,
            locked_capital=0.0,
        )
        return finalise_metrics_result(
            state,
            trades,
            returns_series,
            guard_flag=guard_flag,
            simple_metrics_only=simple_metrics_only,
            penalty_config=penalty_config,
            validity_thresholds=validity_thresholds,
            ruin_hit=ruin_hit,
            leverage=leverage,
        )

    # =================================================================================
    # === 인디케이터 선계산 ===========================================================
    # =================================================================================

    tick_size = _estimate_tick(df["close"])
    slip_value = tick_size * slippage_ticks

    # ATR 계산: 오실레이터 길이와 KC 길이 기반으로 각각 계산합니다.
    atr_len_series = _atr(df, osc_len)
    atr_primary = _atr(df, kc_len)
    # Compute the unsmoothed True Range (TR1) for momentum normalisation.
    tr1_series = _true_range(df)
    # === 스퀴즈 모멘텀 입력값 계산 ===
    # 다양한 모멘텀 스타일(KC/AVG/Deluxe/Mod)에 따라 기준선을 구성하고 ATR 정규화 여부를 결정합니다.
    # 공통 선계산
    hl2 = (df["high"] + df["low"]) / 2.0
    # --- KC 스타일을 위한 최고/최저 기반 중앙값 ---
    highest_high = df["high"].rolling(kc_len).max()
    lowest_low = df["low"].rolling(kc_len).min()
    mean_kc = (highest_high + lowest_low) / 2.0
    # --- BB 및 KC 선계산 ---
    # Bollinger Band 중간선은 종가의 단순 이동평균으로 정의합니다.
    bb_basis_close = _sma(df["close"], bb_len)
    # Keltner Channel 기반 중심선 계산을 위해 hl2 를 사용합니다.
    kc_basis = _sma(hl2, kc_len)
    kc_range_series = atr_primary * float(kc_mult)
    kc_upper = kc_basis + kc_range_series
    kc_lower = kc_basis - kc_range_series
    kc_average = (kc_upper + kc_lower) / 2.0
    midline = (hl2 + kc_average) / 2.0
    if use_channel_stop:
        if stop_channel_mode == "BB":
            bb_std_for_stop = _std(df["close"], bb_len) * float(stop_channel_mult_val)
            channel_stop_lower_series = bb_basis_close - bb_std_for_stop
            channel_stop_upper_series = bb_basis_close + bb_std_for_stop
        else:
            kc_stop_range = atr_primary * float(stop_channel_mult_val)
            channel_stop_lower_series = kc_basis - kc_stop_range
            channel_stop_upper_series = kc_basis + kc_stop_range
    else:
        channel_stop_lower_series = pd.Series(np.nan, index=df.index)
        channel_stop_upper_series = pd.Series(np.nan, index=df.index)
    # --- AVG 스타일: BB 중간선과 KC 중앙값의 평균선 ---
    avg_line_avg = (bb_basis_close + mean_kc) / 2.0
    # --- Deluxe 스타일: 최고/최저 기반 중앙값과 hl2 기반 중간선의 평균선 ---
    bb_mid_hl2 = _sma(hl2, bb_len)
    kc_hl2 = mean_kc
    avg_line_deluxe = (kc_hl2 + bb_mid_hl2) / 2.0
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
    threshold_scale_series = pd.Series(1.0, index=df.index)
    # -------------------------------------------------------------------------
    # Precompute momentum cross-over/-under booleans once.
    # Calculating cross overs inside the main bar loop can be expensive and
    # unnecessarily repetitive. We compute the previous momentum and signal
    # values here and derive boolean series indicating where a momentum
    # cross-over or cross-under occurs. These are then referenced by index
    # inside the loop to determine entry triggers.
    # Compute cross-over/-under boolean series.  If Numba is available and
    # the user has enabled ``useNumba``, leverage a JIT-compiled loop to
    # build the arrays; otherwise, fall back to a vectorised Pandas approach.
    if use_numba and NUMBA_AVAILABLE:
        try:
            cu, cd = _compute_cross_series_numba(momentum.to_numpy(), mom_signal.to_numpy())
        except Exception:
            # If JIT compilation fails at runtime, revert to vectorised method
            _prev_mom = momentum.shift(1).fillna(momentum)
            _prev_sig = mom_signal.shift(1).fillna(mom_signal)
            _cross_up_series = (_prev_mom <= _prev_sig) & (momentum > mom_signal)
            _cross_dn_series = (_prev_mom >= _prev_sig) & (momentum < mom_signal)
        else:
            _cross_up_series = pd.Series(cu, index=df.index)
            _cross_dn_series = pd.Series(cd, index=df.index)
    else:
        _prev_mom = momentum.shift(1).fillna(momentum)
        _prev_sig = mom_signal.shift(1).fillna(mom_signal)
        _cross_up_series = (_prev_mom <= _prev_sig) & (momentum > mom_signal)
        _cross_dn_series = (_prev_mom >= _prev_sig) & (momentum < mom_signal)

    flux_components = _compute_flux_block(
        df,
        flux_len,
        flux_smooth_len,
        flux_deadzone,
        use_heikin=flux_use_ha,
        use_mod_flux=use_mod_flux,
    )
    flux_raw_series = flux_components["raw"]
    flux_cut_series = flux_components["cut"]
    flux_gate_series = flux_components["gate"]
    flux_hist = flux_cut_series

    # -------------------------------------------------------------------------
    # Momentum fade context (simplified)
    #
    # The original implementation computed a complex fade histogram based on
    # Bollinger Bands and Keltner Channels.  To match the TradingView
    # implementation more closely, we collapse the fade logic into a simple
    # momentum‑based exit: if the momentum crosses under its signal after
    # having been above zero for at least one bar (for longs), or crosses
    # over its signal after being below zero (for shorts), we trigger a
    # fade exit provided the absolute momentum exceeds a minimum threshold.
    #
    # We therefore compute the absolute value of momentum and bar counts
    # tracking how many bars have passed since momentum was <= 0 (for long
    # fades) or >= 0 (for short fades).  These series are later converted
    # to NumPy arrays and referenced inside the main bar loop.
    mom_abs = momentum.abs()
    mom_le_zero = momentum.le(0.0)
    mom_ge_zero = momentum.ge(0.0)
    mom_since_le_zero = _bars_since_mask(mom_le_zero)
    mom_since_ge_zero = _bars_since_mask(mom_ge_zero)

    bb_dev = _std(df["close"], bb_len) * bb_mult
    kc_range = atr_primary * kc_mult
    gate_sq_on = (bb_dev < kc_range).fillna(False).astype(bool)
    gate_sq_prev = gate_sq_on.shift(fill_value=False)
    gate_sq_rel = gate_sq_prev & np.logical_not(gate_sq_on)
    gate_rel_idx = gate_sq_rel.cumsum()
    gate_rel_idx = gate_rel_idx.where(gate_sq_rel, np.nan).ffill()
    gate_bars_since_release = (df.index.to_series().groupby(gate_rel_idx).cumcount()).fillna(np.inf)

    buy_thresh_series, sell_thresh_series = _compute_dynamic_thresholds(
        momentum,
        use_dynamic=use_dynamic_thresh,
        use_sym_threshold=use_sym_threshold,
        stat_threshold=stat_threshold,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        dyn_len=dyn_len,
        dyn_mult=dyn_mult,
    )
    if compat_mode and auto_threshold_scale and not use_dynamic_thresh:
        buy_thresh_series = buy_thresh_series.multiply(threshold_scale_series)
        sell_thresh_series = sell_thresh_series.multiply(threshold_scale_series)

    fastpath_metrics = _attempt_fastpath()
    if fastpath_metrics is not None:
        return fastpath_metrics

    filter_settings = FilterSettings(
        use_volatility_guard=use_volatility_guard,
        volatility_lookback=volatility_lookback,
        use_adx=use_adx,
        use_atr_diff=use_atr_diff,
        adx_atr_tf=adx_atr_tf,
        adx_len=adx_len,
        use_ema=use_ema,
        ema_fast_len=ema_fast_len,
        ema_slow_len=ema_slow_len,
        use_bb_filter=use_bb_filter,
        bb_filter_len=bb_filter_len,
        bb_filter_mult=bb_filter_mult,
        use_stoch_rsi=use_stoch_rsi,
        stoch_len=stoch_len,
        use_obv=use_obv,
        obv_smooth_len=obv_smooth_len,
        use_htf_trend=use_htf_trend,
        htf_trend_tf=htf_trend_tf,
        htf_ma_len=htf_ma_len,
        use_hma_filter=use_hma_filter,
        hma_len=hma_len,
        use_range_filter=use_range_filter,
        range_tf=range_tf,
        range_bars=range_bars,
        range_percent=range_percent,
        use_event_filter=use_event_filter,
        event_windows=event_windows_raw,
        use_slope_filter=use_slope_filter,
        slope_lookback=slope_lookback,
        slope_min_pct=slope_min_pct,
        use_distance_guard=use_distance_guard,
        distance_atr_len=distance_atr_len,
        distance_trend_len=distance_trend_len,
        distance_max_atr=distance_max_atr,
        use_kasa=use_kasa,
        kasa_rsi_len=kasa_rsi_len,
        use_regime_filter=use_regime_filter,
        ctx_htf_tf=ctx_htf_tf,
        ctx_htf_ema_len=ctx_htf_ema_len,
        ctx_htf_adx_len=ctx_htf_adx_len,
        ctx_htf_adx_th=ctx_htf_adx_th,
        use_structure_gate=use_structure_gate,
        use_bos=use_bos,
        use_choch=use_choch,
        bos_tf=bos_tf,
        bos_lookback=bos_lookback,
        bos_state_bars=bos_state_bars,
        choch_state_bars=choch_state_bars,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        use_shock=use_shock,
        atr_fast_len=atr_fast_len,
        atr_slow_len=atr_slow_len,
        shock_mult=shock_mult,
    )

    filter_ctx = _prepare_filter_context(df, filter_settings)
    vol_guard_atr_pct = filter_ctx.vol_guard_atr_pct
    adx_series = filter_ctx.adx_series
    atr_diff = filter_ctx.atr_diff
    ema_fast = filter_ctx.ema_fast
    ema_slow = filter_ctx.ema_slow
    bb_filter_basis = filter_ctx.bb_filter_basis
    bb_filter_upper = filter_ctx.bb_filter_upper
    bb_filter_lower = filter_ctx.bb_filter_lower
    stoch_rsi_val = filter_ctx.stoch_rsi
    obv_slope = filter_ctx.obv_slope
    htf_trend_up = filter_ctx.htf_trend_up
    htf_trend_down = filter_ctx.htf_trend_down
    hma_value = filter_ctx.hma_value
    in_range_box = filter_ctx.in_range_box
    event_mask = filter_ctx.event_mask
    slope_ok_long = filter_ctx.slope_ok_long
    slope_ok_short = filter_ctx.slope_ok_short
    distance_ok = filter_ctx.distance_ok
    kasa_rsi = filter_ctx.kasa_rsi
    regime_long_ok = filter_ctx.regime_long_ok
    regime_short_ok = filter_ctx.regime_short_ok
    bos_long_state = filter_ctx.bos_long_state
    bos_short_state = filter_ctx.bos_short_state
    choch_long_state = filter_ctx.choch_long_state
    choch_short_state = filter_ctx.choch_short_state
    shock_series = filter_ctx.shock_series

    # 스탑 계산용 시리즈 -------------------------------------------------------
    atr_trail_series = (
        _atr(df, atr_trail_len)
        if (use_atr_trail or use_breakeven_stop or use_be_tiers or use_dyn_vol or use_atr_profit)
        else pd.Series(np.nan, index=df.index)
    )
    pivot_low_series = (
        _pivot_series(df["low"], pivot_len, pivot_len, False) if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    pivot_high_series = (
        _pivot_series(df["high"], pivot_len, pivot_len, True) if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    swing_low_series = (
        df["low"].rolling(stop_lookback).min() if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    swing_high_series = (
        df["high"].rolling(stop_lookback).max() if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    if use_chandelier_exit:
        chandelier_high_series, chandelier_low_series, chandelier_atr_series = _chandelier_levels(
            df, chandelier_len
        )
    else:
        chandelier_high_series = pd.Series(np.nan, index=df.index)
        chandelier_low_series = pd.Series(np.nan, index=df.index)
        chandelier_atr_series = pd.Series(np.nan, index=df.index)
    sar_series = _parabolic_sar(df, sar_start, sar_increment, sar_maximum) if use_sar_exit else pd.Series(
        np.nan, index=df.index
    )
    if use_pivot_stop and use_pivot_htf:
        pivot_low_htf = _security_series(
            df, pivot_tf, lambda data: _pivot_series(data["low"], pivot_len, pivot_len, False)
        )
        pivot_high_htf = _security_series(
            df, pivot_tf, lambda data: _pivot_series(data["high"], pivot_len, pivot_len, True)
        )
    else:
        pivot_low_htf = pd.Series(np.nan, index=df.index)
        pivot_high_htf = pd.Series(np.nan, index=df.index)

    if use_shock:
        atr_fast = _atr(df, atr_fast_len)
        atr_slow = _sma(atr_fast, atr_slow_len)
        shock_series = atr_fast > atr_slow * shock_mult
    else:
        shock_series = pd.Series(False, index=df.index)

    if use_dyn_vol:
        atr_ratio = atr_trail_series / df["close"]
        bb_dev20 = _std(df["close"], 20) * 2.0
        bb_width = (bb_dev20 * 2.0) / df["close"]
        ma50 = _sma(df["close"], 50)
        ma_dist = (df["close"] - ma50).abs() / df["close"]
        dyn_metric = (atr_ratio.fillna(0.0) + bb_width.fillna(0.0) + ma_dist.fillna(0.0)) / 3.0
        dyn_factor_series = dyn_metric + 1.0
        dyn_factor_series = dyn_factor_series.clip(lower=0.5, upper=3.0)
    else:
        dyn_factor_series = pd.Series(1.0, index=df.index)

    # =================================================================================
    # === 상태 초기화 =================================================================
    # =================================================================================

    state = EquityState(
        initial_capital=initial_capital,
        equity=initial_capital,
        tradable_capital=initial_capital,
        peak_equity=initial_capital,
        daily_start_capital=initial_capital,
        daily_peak_capital=initial_capital,
        week_start_equity=initial_capital,
        week_peak_equity=initial_capital,
        available_capital=initial_capital,
        savings=0.0,
        liquidations=0,
        locked_capital=0.0,
    )
    position = Position()
    trades: List[Trade] = []

    recent_trade_results: List[float] = []
    guard_frozen = False
    guard_fired_total = 0
    loss_streak = 0
    daily_losses = 0
    reentry_countdown = 0
    reversal_countdown = 0
    last_position_dir = 0
    highest_since_entry = np.nan
    lowest_since_entry = np.nan
    pos_bars = 0

    equity_trace: List[float] = [state.equity]
    returns_series = pd.Series(0.0, index=df.index)

    # -------------------------------------------------------------------------
    # 손절 및 청산 관련 변수 초기화
    # ATR 계산: atr_stop_len_val > 0 인 경우에만 계산하고, 아니면 0 배열을 사용합니다.
    if atr_stop_len_val > 0:
        try:
            atr_series_for_stop = _atr(df, atr_stop_len_val)
            atr_values_for_stop = atr_series_for_stop.to_numpy(dtype=float)
        except Exception:
            # ATR 계산 실패 시 0 배열로 대체
            atr_values_for_stop = np.zeros(len(df), dtype=float)
    else:
        atr_values_for_stop = np.zeros(len(df), dtype=float)
    # 현재 포지션의 손절 및 청산 가격을 저장하는 변수
    position_stop_price: Optional[float] = None
    position_liq_price: Optional[float] = None
    chandelier_long_trail = np.nan
    chandelier_short_trail = np.nan
    def sync_equity(update_trace: bool = False) -> None:
        state.equity = state.available_capital + state.savings + state.locked_capital
        state.peak_equity = max(state.peak_equity, state.equity)
        if update_trace:
            equity_trace.append(state.equity)

    def record_trade_profit(pnl: float) -> None:
        nonlocal loss_streak, daily_losses
        state.net_profit += pnl
        if pnl < 0:
            loss_streak += 1
            daily_losses += 1
        elif pnl > 0:
            loss_streak = 0
        sync_equity(update_trace=True)

    def calc_order_size(close_price: float, stop_distance: float, risk_mult: float) -> Tuple[float, float]:
        if close_price <= 0:
            return 0.0, 0.0
        effective_scale = base_risk_pct
        if use_drawdown_scaling and state.peak_equity > 0:
            dd = (state.peak_equity - state.equity) / state.peak_equity * 100.0
            if dd > drawdown_trigger_pct:
                effective_scale *= drawdown_risk_scale
        if use_perf_adaptive_risk and recent_trade_results:
            wins = sum(1 for x in recent_trade_results if x > 0)
            win_rate = wins / len(recent_trade_results) * 100.0
            if len(recent_trade_results) >= par_min_trades:
                if win_rate >= par_hot_win_rate:
                    effective_scale *= par_hot_mult
                elif win_rate <= par_cold_win_rate:
                    effective_scale *= par_cold_mult
        mult = max(risk_mult, 0.0)
        margin_from_qty = lambda qty: (qty * close_price) / leverage if leverage > 0 else 0.0
        if not use_sizing_override:
            pct_to_use = max(
                base_qty_percent
                * mult
                * ((effective_scale / base_risk_pct) if base_risk_pct > 0 else 1.0),
                0.0,
            )
            capital_portion = state.available_capital * pct_to_use / 100.0
            if capital_portion <= 0.0:
                return 0.0, 0.0
            qty = (capital_portion * leverage) / close_price
            if not np.isfinite(qty) or qty <= 0.0:
                return 0.0, 0.0
            return qty, capital_portion
        if sizing_mode == "자본 비율":
            pct_to_use = max(advanced_percent * mult, 0.0)
            capital_portion = state.available_capital * pct_to_use / 100.0
            if capital_portion <= 0.0:
                return 0.0, 0.0
            qty = (capital_portion * leverage) / close_price
            if not np.isfinite(qty) or qty <= 0.0:
                return 0.0, 0.0
            return qty, capital_portion
        if sizing_mode == "고정 금액 (USD)":
            usd_to_use = max(fixed_usd_amount * mult, 0.0)
            if usd_to_use <= 0.0:
                return 0.0, 0.0
            qty = (usd_to_use * leverage) / close_price
            if not np.isfinite(qty) or qty <= 0.0:
                return 0.0, 0.0
            return qty, usd_to_use
        if sizing_mode == "고정 계약":
            qty = max(fixed_contract_size * mult, 0.0)
            if qty <= 0.0 or not np.isfinite(qty):
                return 0.0, 0.0
            return qty, margin_from_qty(qty)
        if sizing_mode == "리스크 기반":
            if risk_sizing_type == "고정 계약":
                qty = max(risk_contract_size * mult, 0.0)
                if qty <= 0.0 or not np.isfinite(qty):
                    return 0.0, 0.0
                return qty, margin_from_qty(qty)
            if stop_distance <= 0 or np.isnan(stop_distance):
                return 0.0, 0.0
            risk_pct = max(effective_scale * mult, 0.0)
            risk_capital = state.available_capital * risk_pct / 100.0
            if risk_capital <= 0.0:
                return 0.0, 0.0
            denom = stop_distance + slip_value
            if denom <= 0 or not np.isfinite(denom):
                return 0.0, 0.0
            qty = risk_capital / denom
            if qty <= 0.0 or not np.isfinite(qty):
                return 0.0, 0.0
            return qty, margin_from_qty(qty)
        return 0.0, 0.0

    def close_position(ts: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal position, highest_since_entry, lowest_since_entry, pos_bars, guard_frozen, guard_fired_total, last_position_dir, ruin_hit, chandelier_long_trail, chandelier_short_trail
        if position.direction == 0 or position.entry_time is None:
            return
        qty = position.qty
        direction = position.direction
        exit_price = price - slip_value if direction > 0 else price + slip_value
        pnl = (exit_price - position.avg_price) * direction * qty
        fees_paid = (position.avg_price + exit_price) * qty * commission_pct
        pnl -= fees_paid
        state.release_locked_capital(position.capital_used)
        state.apply_pnl(pnl)
        returns_series.loc[ts] += pnl / state.initial_capital if state.initial_capital else 0.0
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                direction="long" if direction > 0 else "short",
                size=qty,
                entry_price=position.avg_price,
                exit_price=exit_price,
                profit=pnl,
                return_pct=pnl / state.initial_capital if state.initial_capital else 0.0,
                mfe=np.nan,
                mae=np.nan,
                bars_held=position.bars_held,
                reason=reason,
            )
        )
        ruin_hit = state.handle_trade_settlement(
            pnl,
            deposit_pct=profit_deposit_pct,
            reason=reason,
            ruin_threshold=ruin_threshold,
        ) or ruin_hit
        record_trade_profit(pnl)
        last_position_dir = direction
        position = Position()
        highest_since_entry = np.nan
        lowest_since_entry = np.nan
        pos_bars = 0
        chandelier_long_trail = np.nan
        chandelier_short_trail = np.nan
        if use_perf_adaptive_risk:
            recent_trade_results.append(pnl)
            if len(recent_trade_results) > par_lookback:
                recent_trade_results.pop(0)
        reentry_countdown = reentry_bars

    prev_guard_state = guard_frozen

    close_vals = df["close"].to_numpy()
    high_vals = df["high"].to_numpy()
    low_vals = df["low"].to_numpy()
    momentum_vals = momentum.to_numpy()
    mom_signal_vals = mom_signal.to_numpy()
    flux_hist_vals = flux_hist.to_numpy()
    flux_gate_vals = flux_gate_series.to_numpy()
    buy_thresh_vals = buy_thresh_series.to_numpy()
    sell_thresh_vals = sell_thresh_series.to_numpy()
    cross_up_vals = _cross_up_series.to_numpy()
    cross_down_vals = _cross_dn_series.to_numpy()
    adx_vals = adx_series.to_numpy() if use_adx else None
    ema_fast_vals = ema_fast.to_numpy() if use_ema else None
    ema_slow_vals = ema_slow.to_numpy() if use_ema else None
    bb_filter_basis_vals = bb_filter_basis.to_numpy() if use_bb_filter else None
    bb_filter_lower_vals = bb_filter_lower.to_numpy() if use_bb_filter else None
    bb_filter_upper_vals = bb_filter_upper.to_numpy() if use_bb_filter else None
    stoch_rsi_vals = stoch_rsi_val.to_numpy() if use_stoch_rsi else None
    obv_slope_vals = obv_slope.to_numpy() if use_obv else None
    atr_diff_vals = atr_diff.to_numpy() if use_atr_diff else None
    htf_trend_up_vals = htf_trend_up.to_numpy() if use_htf_trend else None
    htf_trend_down_vals = htf_trend_down.to_numpy() if use_htf_trend else None
    hma_vals = hma_value.to_numpy() if use_hma_filter else None
    in_range_box_vals = in_range_box.to_numpy() if use_range_filter else None
    event_mask_vals = event_mask.to_numpy() if use_event_filter else None
    slope_ok_long_vals = slope_ok_long.to_numpy() if use_slope_filter else None
    slope_ok_short_vals = slope_ok_short.to_numpy() if use_slope_filter else None
    distance_ok_vals = distance_ok.to_numpy() if use_distance_guard else None
    regime_long_ok_vals = regime_long_ok.to_numpy()
    regime_short_ok_vals = regime_short_ok.to_numpy()
    gate_bars_since_release_vals = gate_bars_since_release.to_numpy()
    gate_sq_on_vals = gate_sq_on.to_numpy()
    bos_long_state_vals = bos_long_state.to_numpy()
    bos_short_state_vals = bos_short_state.to_numpy()
    choch_long_state_vals = choch_long_state.to_numpy()
    choch_short_state_vals = choch_short_state.to_numpy()
    # Simplified momentum fade context arrays
    mom_abs_vals = mom_abs.to_numpy()
    mom_since_le_zero_vals = mom_since_le_zero.to_numpy()
    if use_channel_stop:
        channel_stop_lower_vals = channel_stop_lower_series.to_numpy()
        channel_stop_upper_vals = channel_stop_upper_series.to_numpy()
    else:
        channel_stop_lower_vals = np.full(len(df), np.nan)
        channel_stop_upper_vals = np.full(len(df), np.nan)
    mom_since_ge_zero_vals = mom_since_ge_zero.to_numpy()
    # Cross arrays converted to NumPy for fade logic; these were precomputed
    cross_up_vals = _cross_up_series.to_numpy()
    cross_dn_vals = _cross_dn_series.to_numpy()
    kasa_rsi_vals = kasa_rsi.to_numpy() if use_kasa else None
    atr_trail_vals = atr_trail_series.to_numpy()
    dyn_factor_vals = dyn_factor_series.to_numpy()
    swing_low_vals = swing_low_series.to_numpy()
    swing_high_vals = swing_high_series.to_numpy()
    pivot_low_htf_vals = pivot_low_htf.to_numpy()
    pivot_high_htf_vals = pivot_high_htf.to_numpy()
    pivot_low_vals = pivot_low_series.to_numpy()
    pivot_high_vals = pivot_high_series.to_numpy()
    chandelier_high_vals = (
        chandelier_high_series.to_numpy() if use_chandelier_exit else None
    )
    chandelier_low_vals = (
        chandelier_low_series.to_numpy() if use_chandelier_exit else None
    )
    chandelier_atr_vals = (
        chandelier_atr_series.to_numpy() if use_chandelier_exit else None
    )
    sar_vals = sar_series.to_numpy() if use_sar_exit else None
    atr_len_vals = atr_len_series.to_numpy()
    shock_vals = shock_series.to_numpy() if use_shock else None
    vol_guard_vals = vol_guard_atr_pct.to_numpy() if use_volatility_guard else None

    for idx, ts in enumerate(df.index):
        if ts < start_ts:
            continue

        if ruin_hit:
            break

        if idx > 0:
            prev_day = df.index[idx - 1].date()
            if ts.date() != prev_day:
                state.daily_start_capital = state.tradable_capital
                state.daily_peak_capital = state.tradable_capital
                daily_losses = 0
                guard_frozen = False
            prev_week = df.index[idx - 1].isocalendar()[1]
            if ts.isocalendar()[1] != prev_week:
                state.week_start_equity = state.tradable_capital
                state.week_peak_equity = state.tradable_capital

        if use_wallet and state.net_profit > 0:
            state.withdrawable += state.net_profit * profit_reserve_pct
        available_for_sizing = state.available_capital
        if use_wallet and apply_reserve_to_sizing:
            available_for_sizing = max(available_for_sizing - state.withdrawable, 0.0)
        state.tradable_capital = max(available_for_sizing, state.initial_capital * 0.01)
        state.peak_equity = max(state.peak_equity, state.equity)
        state.daily_peak_capital = max(state.daily_peak_capital, state.tradable_capital)
        state.week_peak_equity = max(state.week_peak_equity, state.tradable_capital)

        daily_pnl = state.tradable_capital - state.daily_start_capital
        weekly_pnl = state.tradable_capital - state.week_start_equity
        weekly_dd = (
            (state.week_peak_equity - state.tradable_capital) / state.week_peak_equity * 100.0
            if state.week_peak_equity > 0
            else 0.0
        )

        daily_loss_breached = use_daily_loss_guard and daily_pnl <= -abs(daily_loss_limit)
        daily_profit_reached = use_daily_profit_lock and daily_pnl >= abs(daily_profit_target)
        weekly_profit_reached = use_weekly_profit_lock and weekly_pnl >= abs(weekly_profit_target)
        loss_streak_breached = use_loss_streak_guard and loss_streak >= max_consecutive_losses
        capital_breached = use_capital_guard and state.equity <= state.initial_capital * (1 - capital_guard_pct / 100.0)
        weekly_dd_breached = max_weekly_dd > 0 and weekly_dd >= max_weekly_dd
        loss_count_breached = max_daily_losses > 0 and daily_losses >= max_daily_losses
        guard_fire_limit = max_guard_fires > 0 and guard_fired_total >= max_guard_fires

        close_price = close_vals[idx]
        high_price = high_vals[idx]
        low_price = low_vals[idx]
        atr_pct_val = vol_guard_vals[idx] if use_volatility_guard and vol_guard_vals is not None else 0.0
        is_vol_ok = (not use_volatility_guard) or (
            volatility_lower_pct <= atr_pct_val <= volatility_upper_pct
        )

        performance_pause = False
        if use_perf_adaptive_risk and recent_trade_results:
            wins = sum(1 for x in recent_trade_results if x > 0)
            win_rate = wins / len(recent_trade_results) * 100.0
            if len(recent_trade_results) >= par_min_trades and win_rate <= par_cold_win_rate and par_pause_on_cold:
                performance_pause = True

        should_freeze = (
            daily_loss_breached
            or daily_profit_reached
            or weekly_profit_reached
            or loss_streak_breached
            or capital_breached
            or weekly_dd_breached
            or loss_count_breached
            or guard_fire_limit
            or performance_pause
            or state.tradable_capital < min_tradable_capital
        )
        if should_freeze:
            guard_frozen = True

        guard_activated = guard_frozen and not prev_guard_state
        prev_guard_state = guard_frozen

        if guard_activated and position.direction != 0:
            close_position(ts, close_price, "Guard Halt")
            guard_fired_total += 1
            if ruin_hit:
                break

        if use_guard_exit and position.direction != 0 and not guard_activated:
            qty = abs(position.qty)
            if qty > 0:
                entry_price = position.avg_price
                initial_margin = (qty * entry_price) / leverage
                maint_margin = (qty * entry_price) * (maintenance_margin_pct / 100.0)
                offset = (initial_margin - maint_margin) / qty if qty > 0 else 0.0
                liq_price = entry_price - offset if position.direction > 0 else entry_price + offset
                if liq_buffer_pct > 0:
                    buffer = entry_price * (liq_buffer_pct / 100.0)
                    if position.direction > 0:
                        liq_price -= buffer
                    else:
                        liq_price += buffer
                preempt_price = liq_price + preempt_ticks * tick_size if position.direction > 0 else liq_price - preempt_ticks * tick_size
                hit_guard = low_price <= preempt_price if position.direction > 0 else high_price >= preempt_price
                if hit_guard:
                    close_position(ts, close_price, "Guard Exit")
                    guard_frozen = True
                    guard_fired_total += 1
                    if ruin_hit:
                        break

        can_trade = (not guard_frozen) and is_vol_ok

        if reentry_countdown > 0 and position.direction == 0:
            reentry_countdown -= 1
        if reversal_countdown > 0 and position.direction == 0:
            reversal_countdown -= 1

        if position.direction != 0:
            pos_bars += 1
            if position.direction > 0:
                highest_since_entry = high_price if np.isnan(highest_since_entry) else max(highest_since_entry, high_price)
                lowest_since_entry = low_price if np.isnan(lowest_since_entry) else min(lowest_since_entry, low_price)
            else:
                lowest_since_entry = low_price if np.isnan(lowest_since_entry) else min(lowest_since_entry, low_price)
                highest_since_entry = high_price if np.isnan(highest_since_entry) else max(highest_since_entry, high_price)
            position.bars_held += 1

            if use_channel_stop:
                if position.direction > 0 and np.isfinite(channel_stop_lower_val):
                    candidate = float(channel_stop_lower_val)
                    if position_stop_price is None:
                        position_stop_price = candidate
                    else:
                        position_stop_price = max(position_stop_price, candidate)
                elif position.direction < 0 and np.isfinite(channel_stop_upper_val):
                    candidate = float(channel_stop_upper_val)
                    if position_stop_price is None:
                        position_stop_price = candidate
                    else:
                        position_stop_price = min(position_stop_price, candidate)

            # --- 청산 및 손절 체크 ---
            # 포지션이 열려 있을 때, 우선 청산 기준을 확인합니다. 레버리지 기반 청산은
            # entry_price 대비 지정된 비율(liq_drop_pct) 만큼 손실이 나면 발생합니다.
            if position_liq_price is not None:
                if position.direction > 0 and low_price <= position_liq_price:
                    # 롱 포지션 청산
                    close_position(ts, position_liq_price, "Liquidation")
                    # 포지션 리셋
                    position_liq_price = None
                    position_stop_price = None
                    # RUIN 발생 시 루프 종료
                    if ruin_hit:
                        break
                    # 다음 바로 이동
                    continue
                if position.direction < 0 and high_price >= position_liq_price:
                    # 숏 포지션 청산
                    close_position(ts, position_liq_price, "Liquidation")
                    position_liq_price = None
                    position_stop_price = None
                    if ruin_hit:
                        break
                    continue
            # 손절(stop) 조건: ATR 또는 고정 % 기준. 설정된 stop_price 를 확인하고
            # 해당 가격에 도달하면 즉시 포지션을 종료합니다.
            if position_stop_price is not None:
                if position.direction > 0 and low_price <= position_stop_price:
                    close_position(ts, position_stop_price, "Stop")
                    position_stop_price = None
                    position_liq_price = None
                    if ruin_hit:
                        break
                    continue
                if position.direction < 0 and high_price >= position_stop_price:
                    close_position(ts, position_stop_price, "Stop")
                    position_stop_price = None
                    position_liq_price = None
                    if ruin_hit:
                        break
                    continue

        prev_idx = max(idx - 1, 0)
        prev_momentum = momentum_vals[prev_idx]
        prev_signal = mom_signal_vals[prev_idx]
        mom_val = momentum_vals[idx]
        sig_val = mom_signal_vals[idx]
        flux_gate_val = flux_gate_vals[idx]
        buy_thresh_val = buy_thresh_vals[idx]
        sell_thresh_val = sell_thresh_vals[idx]

        # Determine cross-over and cross-under signals from the precomputed series.
        cross_up = bool(cross_up_vals[idx])
        cross_down = bool(cross_down_vals[idx])

        atr_trail_val = atr_trail_vals[idx]
        dyn_factor_val = dyn_factor_vals[idx]
        atr_len_val = atr_len_vals[idx]
        swing_low_val = swing_low_vals[idx]
        swing_high_val = swing_high_vals[idx]
        pivot_low_val = pivot_low_vals[idx]
        pivot_high_val = pivot_high_vals[idx]
        pivot_low_htf_val = pivot_low_htf_vals[idx]
        pivot_high_htf_val = pivot_high_htf_vals[idx]
        channel_stop_lower_val = channel_stop_lower_vals[idx] if use_channel_stop else np.nan
        channel_stop_upper_val = channel_stop_upper_vals[idx] if use_channel_stop else np.nan

        long_cross_ok = cross_up or not require_momentum_cross
        short_cross_ok = cross_down or not require_momentum_cross
        base_long_trigger = long_cross_ok and mom_val < buy_thresh_val and flux_gate_val > 0
        base_short_trigger = short_cross_ok and mom_val > sell_thresh_val and flux_gate_val < 0
        base_long_signal = debug_force_long or base_long_trigger
        base_short_signal = debug_force_short or base_short_trigger

        long_ok = True
        short_ok = True

        if use_adx and adx_vals is not None:
            adx_val = adx_vals[idx]
            long_ok &= adx_val > adx_thresh
            short_ok &= adx_val > adx_thresh
        if use_ema and ema_slow_vals is not None:
            if ema_mode == "Crossover" and ema_fast_vals is not None:
                long_ok &= ema_fast_vals[idx] > ema_slow_vals[idx]
                short_ok &= ema_fast_vals[idx] < ema_slow_vals[idx]
            else:
                long_ok &= close_price > ema_slow_vals[idx]
                short_ok &= close_price < ema_slow_vals[idx]
        if use_bb_filter and bb_filter_basis_vals is not None:
            long_ok &= (close_price <= bb_filter_basis_vals[idx]) or (close_price < bb_filter_lower_vals[idx])
            short_ok &= (close_price >= bb_filter_basis_vals[idx]) or (close_price > bb_filter_upper_vals[idx])
        if use_stoch_rsi and stoch_rsi_vals is not None:
            stoch_val = stoch_rsi_vals[idx]
            long_ok &= stoch_val <= stoch_os
            short_ok &= stoch_val >= stoch_ob
        if use_obv and obv_slope_vals is not None:
            obv_val = obv_slope_vals[idx]
            long_ok &= obv_val > 0
            short_ok &= obv_val < 0
        if use_atr_diff and atr_diff_vals is not None:
            atr_diff_val = atr_diff_vals[idx]
            long_ok &= atr_diff_val > 0
            short_ok &= atr_diff_val > 0
        if use_htf_trend and htf_trend_up_vals is not None and htf_trend_down_vals is not None:
            long_ok &= bool(htf_trend_up_vals[idx])
            short_ok &= bool(htf_trend_down_vals[idx])
        if use_hma_filter and hma_vals is not None:
            hma_val = hma_vals[idx]
            long_ok &= close_price > hma_val
            short_ok &= close_price < hma_val
        if use_range_filter and in_range_box_vals is not None:
            in_range = bool(in_range_box_vals[idx])
            long_ok &= not in_range
            short_ok &= not in_range
        if use_event_filter and event_mask_vals is not None:
            event_flag = bool(event_mask_vals[idx])
            long_ok &= not event_flag
            short_ok &= not event_flag
        if use_slope_filter and slope_ok_long_vals is not None and slope_ok_short_vals is not None:
            long_ok &= bool(slope_ok_long_vals[idx])
            short_ok &= bool(slope_ok_short_vals[idx])
        if use_distance_guard and distance_ok_vals is not None:
            distance_flag = bool(distance_ok_vals[idx])
            long_ok &= distance_flag
            short_ok &= distance_flag
        if use_equity_slope_filter and len(equity_trace) >= eq_slope_len:
            equity_window = pd.Series(equity_trace[-eq_slope_len:])
            eq_slope = _linreg(equity_window, min(eq_slope_len, len(equity_window))).iloc[-1]
            long_ok &= eq_slope >= 0
            short_ok &= eq_slope <= 0
        long_ok &= bool(regime_long_ok_vals[idx])
        short_ok &= bool(regime_short_ok_vals[idx])

        structure_require_all = structure_gate_mode == "모두 충족"
        structure_long_pass = True
        structure_short_pass = True
        if use_structure_gate:
            if structure_require_all:
                structure_long_pass = (not use_bos or bool(bos_long_state_vals[idx])) and (
                    not use_choch or bool(choch_long_state_vals[idx])
                )
                structure_short_pass = (not use_bos or bool(bos_short_state_vals[idx])) and (
                    not use_choch or bool(choch_short_state_vals[idx])
                )
            else:
                structure_long_pass = (
                    (use_bos and bool(bos_long_state_vals[idx]))
                    or (use_choch and bool(choch_long_state_vals[idx]))
                    or (not use_bos and not use_choch)
                )
                structure_short_pass = (
                    (use_bos and bool(bos_short_state_vals[idx]))
                    or (use_choch and bool(choch_short_state_vals[idx]))
                    or (not use_bos and not use_choch)
                )
            long_ok &= structure_long_pass
            short_ok &= structure_short_pass

        gate_release_seen = gate_bars_since_release_vals[idx] != np.inf
        gate_sq_valid = (
            gate_release_seen
            and gate_bars_since_release_vals[idx] <= sqz_release_bars
            and not bool(gate_sq_on_vals[idx])
        )
        if use_sqz_gate:
            long_ok &= gate_sq_valid
            short_ok &= gate_sq_valid

        if use_structure_gate:
            base_long_signal = base_long_signal and structure_long_pass
            base_short_signal = base_short_signal and structure_short_pass

        long_entry_signal = (
            allow_long_entry
            and can_trade
            and base_long_signal
            and long_ok
            and reentry_countdown == 0
        )
        short_entry_signal = (
            allow_short_entry
            and can_trade
            and base_short_signal
            and short_ok
            and reentry_countdown == 0
        )

        if use_reversal and reversal_countdown == 0 and position.direction == 0 and last_position_dir != 0 and can_trade:
            if last_position_dir == 1:
                short_entry_signal = True
            elif last_position_dir == -1:
                long_entry_signal = True
            last_position_dir = 0

        enter_long = position.direction == 0 and long_entry_signal
        enter_short = position.direction == 0 and short_entry_signal

        if position.direction == 0:
            if use_chandelier_exit:
                chandelier_long_trail = np.nan
                chandelier_short_trail = np.nan
            if enter_long:
                stop_hint = atr_len_val
                if use_stop_loss:
                    if not np.isnan(swing_low_val):
                        stop_hint = max(stop_hint, close_price - swing_low_val) if not np.isnan(stop_hint) else close_price - swing_low_val
                    if use_pivot_stop:
                        pivot_ref = pivot_low_htf_val if use_pivot_htf else pivot_low_val
                        if not np.isnan(pivot_ref):
                            dist_pivot = close_price - pivot_ref
                            stop_hint = max(stop_hint, dist_pivot) if not np.isnan(stop_hint) else dist_pivot
                if use_atr_trail and not np.isnan(atr_trail_val):
                    atr_dist = atr_trail_val * atr_trail_mult
                    stop_hint = max(stop_hint, atr_dist) if not np.isnan(stop_hint) else atr_dist
                if use_channel_stop and np.isfinite(channel_stop_lower_val):
                    channel_dist = close_price - float(channel_stop_lower_val)
                    if channel_dist > 0:
                        if np.isnan(stop_hint) or stop_hint <= 0:
                            stop_hint = channel_dist
                        else:
                            stop_hint = max(stop_hint, channel_dist)
                if np.isnan(stop_hint) or stop_hint <= 0:
                    stop_hint = tick_size
                stop_for_size = max(stop_hint, tick_size)
                guard_ok = (
                    (not use_stop_distance_guard)
                    or np.isnan(atr_len_val)
                    or stop_for_size <= atr_len_val * max_stop_atr_mult
                )
                qty, capital_used = calc_order_size(close_price, stop_for_size, 1.0)
                if guard_ok and qty > 0 and capital_used > 0:
                    if capital_used > state.available_capital + 1e-9:
                        LOGGER.debug(
                            "가용 자본 부족으로 롱 진입을 건너뜀: 필요 %.2f, 보유 %.2f",
                            capital_used,
                            state.available_capital,
                        )
                        qty = 0.0
                    else:
                        state.available_capital -= capital_used
                        state.locked_capital += capital_used
                        state.tradable_capital = max(state.available_capital, state.initial_capital * 0.01)
                        sync_equity()
                    # 새 포지션을 진입합니다. 손절 및 청산 가격을 계산해 저장합니다.
                        position = Position(direction=1, qty=qty, avg_price=close_price, entry_time=ts)
                        position.capital_used = capital_used
                        position.base_qty = qty
                        position.base_capital_used = capital_used
                        position.pyramid_adds = 0
                        highest_since_entry = high_price
                        lowest_since_entry = low_price
                        pos_bars = 0
                        reversal_countdown = int(reversal_delay_sec // 60) if reversal_delay_sec > 0 else 0
                        liq_drop = (1.0 / leverage) * (1.0 - salvage_pct)
                        position_liq_price = close_price * (1.0 - liq_drop)
                        fixed_dist = (fixed_stop_pct_val / 100.0) * close_price if fixed_stop_pct_val > 0 else 0.0
                        atr_dist = atr_values_for_stop[idx] * atr_stop_mult_val if atr_stop_len_val > 0 else 0.0
                        stop_dist = max(fixed_dist, atr_dist)
                        position_stop_price = close_price - stop_dist if stop_dist > 0 else None
                        if use_channel_stop and np.isfinite(channel_stop_lower_val):
                            channel_candidate = float(channel_stop_lower_val)
                            if position_stop_price is None:
                                position_stop_price = channel_candidate
                            else:
                                position_stop_price = max(position_stop_price, channel_candidate)
                        continue
            elif enter_short:
                stop_hint = atr_len_val
                if use_stop_loss:
                    if not np.isnan(swing_high_val):
                        stop_hint = max(stop_hint, swing_high_val - close_price) if not np.isnan(stop_hint) else swing_high_val - close_price
                    if use_pivot_stop:
                        pivot_ref = pivot_high_htf_val if use_pivot_htf else pivot_high_val
                        if not np.isnan(pivot_ref):
                            dist_pivot = pivot_ref - close_price
                            stop_hint = max(stop_hint, dist_pivot) if not np.isnan(stop_hint) else dist_pivot
                if use_atr_trail and not np.isnan(atr_trail_val):
                    atr_dist = atr_trail_val * atr_trail_mult
                    stop_hint = max(stop_hint, atr_dist) if not np.isnan(stop_hint) else atr_dist
                if use_channel_stop and np.isfinite(channel_stop_upper_val):
                    channel_dist = float(channel_stop_upper_val) - close_price
                    if channel_dist > 0:
                        if np.isnan(stop_hint) or stop_hint <= 0:
                            stop_hint = channel_dist
                        else:
                            stop_hint = max(stop_hint, channel_dist)
                if np.isnan(stop_hint) or stop_hint <= 0:
                    stop_hint = tick_size
                stop_for_size = max(stop_hint, tick_size)
                guard_ok = (
                    (not use_stop_distance_guard)
                    or np.isnan(atr_len_val)
                    or stop_for_size <= atr_len_val * max_stop_atr_mult
                )
                qty, capital_used = calc_order_size(close_price, stop_for_size, 1.0)
                if guard_ok and qty > 0 and capital_used > 0:
                    if capital_used > state.available_capital + 1e-9:
                        LOGGER.debug(
                            "가용 자본 부족으로 숏 진입을 건너뜀: 필요 %.2f, 보유 %.2f",
                            capital_used,
                            state.available_capital,
                        )
                        qty = 0.0
                    else:
                        state.available_capital -= capital_used
                        state.locked_capital += capital_used
                        state.tradable_capital = max(state.available_capital, state.initial_capital * 0.01)
                        sync_equity()
                    # 새 숏 포지션을 진입합니다. 손절 및 청산 가격을 계산해 저장합니다.
                        position = Position(direction=-1, qty=qty, avg_price=close_price, entry_time=ts)
                        position.capital_used = capital_used
                        position.base_qty = qty
                        position.base_capital_used = capital_used
                        position.pyramid_adds = 0
                        highest_since_entry = high_price
                        lowest_since_entry = low_price
                        pos_bars = 0
                        reversal_countdown = int(reversal_delay_sec // 60) if reversal_delay_sec > 0 else 0
                        liq_drop = (1.0 / leverage) * (1.0 - salvage_pct)
                        position_liq_price = close_price * (1.0 + liq_drop)
                        fixed_dist = (fixed_stop_pct_val / 100.0) * close_price if fixed_stop_pct_val > 0 else 0.0
                        atr_dist = atr_values_for_stop[idx] * atr_stop_mult_val if atr_stop_len_val > 0 else 0.0
                        stop_dist = max(fixed_dist, atr_dist)
                        position_stop_price = close_price + stop_dist if stop_dist > 0 else None
                        if use_channel_stop and np.isfinite(channel_stop_upper_val):
                            channel_candidate = float(channel_stop_upper_val)
                            if position_stop_price is None:
                                position_stop_price = channel_candidate
                            else:
                                position_stop_price = min(position_stop_price, channel_candidate)
                        continue
        elif (
            use_pyramiding
            and position.pyramid_adds == 0
            and can_trade
            and reentry_countdown == 0
        ):
            if position.direction > 0 and long_entry_signal:
                stop_hint = atr_len_val
                if use_stop_loss:
                    if not np.isnan(swing_low_val):
                        stop_hint = max(stop_hint, close_price - swing_low_val) if not np.isnan(stop_hint) else close_price - swing_low_val
                    if use_pivot_stop:
                        pivot_ref = pivot_low_htf_val if use_pivot_htf else pivot_low_val
                        if not np.isnan(pivot_ref):
                            dist_pivot = close_price - pivot_ref
                            stop_hint = max(stop_hint, dist_pivot) if not np.isnan(stop_hint) else dist_pivot
                if use_atr_trail and not np.isnan(atr_trail_val):
                    atr_dist = atr_trail_val * atr_trail_mult
                    stop_hint = max(stop_hint, atr_dist) if not np.isnan(stop_hint) else atr_dist
                if use_channel_stop and np.isfinite(channel_stop_lower_val):
                    channel_dist = close_price - float(channel_stop_lower_val)
                    if channel_dist > 0:
                        if np.isnan(stop_hint) or stop_hint <= 0:
                            stop_hint = channel_dist
                        else:
                            stop_hint = max(stop_hint, channel_dist)
                if np.isnan(stop_hint) or stop_hint <= 0:
                    stop_hint = tick_size
                stop_for_size = max(stop_hint, tick_size)
                guard_ok = (
                    (not use_stop_distance_guard)
                    or np.isnan(atr_len_val)
                    or stop_for_size <= atr_len_val * max_stop_atr_mult
                )
                add_qty = position.base_qty if position.base_qty > 0 else 0.0
                add_capital = position.base_capital_used if position.base_capital_used > 0 else 0.0
                if guard_ok and add_qty > 0 and add_capital > 0:
                    if add_capital > state.available_capital + 1e-9:
                        LOGGER.debug(
                            "가용 자본 부족으로 롱 피라미딩을 건너뜀: 필요 %.2f, 보유 %.2f",
                            add_capital,
                            state.available_capital,
                        )
                    else:
                        state.available_capital -= add_capital
                        state.locked_capital += add_capital
                        state.tradable_capital = max(state.available_capital, state.initial_capital * 0.01)
                        sync_equity()
                        prev_qty = position.qty
                        new_qty = prev_qty + add_qty
                        position.avg_price = (
                            position.avg_price * prev_qty + close_price * add_qty
                        ) / new_qty
                        position.qty = new_qty
                        position.capital_used += add_capital
                        position.pyramid_adds = 1
                        highest_since_entry = max(highest_since_entry, high_price) if not np.isnan(highest_since_entry) else high_price
                        lowest_since_entry = min(lowest_since_entry, low_price) if not np.isnan(lowest_since_entry) else low_price
                        liq_drop = (1.0 / leverage) * (1.0 - salvage_pct)
                        position_liq_price = position.avg_price * (1.0 - liq_drop)
                        if use_stop_loss or use_atr_trail or use_pivot_stop:
                            new_stop = None
                            if use_stop_loss:
                                fixed_dist = (fixed_stop_pct_val / 100.0) * close_price if fixed_stop_pct_val > 0 else 0.0
                                atr_dist = (
                                    atr_values_for_stop[idx] * atr_stop_mult_val if atr_stop_len_val > 0 else 0.0
                                )
                                stop_dist = max(fixed_dist, atr_dist)
                                if stop_dist > 0:
                                    new_stop = close_price - stop_dist
                            if new_stop is not None:
                                if position_stop_price is None:
                                    position_stop_price = new_stop
                                else:
                                    position_stop_price = max(position_stop_price, new_stop)
                        if use_channel_stop and np.isfinite(channel_stop_lower_val):
                            channel_candidate = float(channel_stop_lower_val)
                            if position_stop_price is None:
                                position_stop_price = channel_candidate
                            else:
                                position_stop_price = max(position_stop_price, channel_candidate)
                        continue
            elif position.direction < 0 and short_entry_signal:
                stop_hint = atr_len_val
                if use_stop_loss:
                    if not np.isnan(swing_high_val):
                        stop_hint = max(stop_hint, swing_high_val - close_price) if not np.isnan(stop_hint) else swing_high_val - close_price
                    if use_pivot_stop:
                        pivot_ref = pivot_high_htf_val if use_pivot_htf else pivot_high_val
                        if not np.isnan(pivot_ref):
                            dist_pivot = pivot_ref - close_price
                            stop_hint = max(stop_hint, dist_pivot) if not np.isnan(stop_hint) else dist_pivot
                if use_atr_trail and not np.isnan(atr_trail_val):
                    atr_dist = atr_trail_val * atr_trail_mult
                    stop_hint = max(stop_hint, atr_dist) if not np.isnan(stop_hint) else atr_dist
                if use_channel_stop and np.isfinite(channel_stop_upper_val):
                    channel_dist = float(channel_stop_upper_val) - close_price
                    if channel_dist > 0:
                        if np.isnan(stop_hint) or stop_hint <= 0:
                            stop_hint = channel_dist
                        else:
                            stop_hint = max(stop_hint, channel_dist)
                if np.isnan(stop_hint) or stop_hint <= 0:
                    stop_hint = tick_size
                stop_for_size = max(stop_hint, tick_size)
                guard_ok = (
                    (not use_stop_distance_guard)
                    or np.isnan(atr_len_val)
                    or stop_for_size <= atr_len_val * max_stop_atr_mult
                )
                add_qty = position.base_qty if position.base_qty > 0 else 0.0
                add_capital = position.base_capital_used if position.base_capital_used > 0 else 0.0
                if guard_ok and add_qty > 0 and add_capital > 0:
                    if add_capital > state.available_capital + 1e-9:
                        LOGGER.debug(
                            "가용 자본 부족으로 숏 피라미딩을 건너뜀: 필요 %.2f, 보유 %.2f",
                            add_capital,
                            state.available_capital,
                        )
                    else:
                        state.available_capital -= add_capital
                        state.locked_capital += add_capital
                        state.tradable_capital = max(state.available_capital, state.initial_capital * 0.01)
                        sync_equity()
                        prev_qty = position.qty
                        new_qty = prev_qty + add_qty
                        position.avg_price = (
                            position.avg_price * prev_qty + close_price * add_qty
                        ) / new_qty
                        position.qty = new_qty
                        position.capital_used += add_capital
                        position.pyramid_adds = 1
                        highest_since_entry = max(highest_since_entry, high_price) if not np.isnan(highest_since_entry) else high_price
                        lowest_since_entry = min(lowest_since_entry, low_price) if not np.isnan(lowest_since_entry) else low_price
                        liq_drop = (1.0 / leverage) * (1.0 - salvage_pct)
                        position_liq_price = position.avg_price * (1.0 + liq_drop)
                        if use_stop_loss or use_atr_trail or use_pivot_stop:
                            new_stop = None
                            if use_stop_loss:
                                fixed_dist = (fixed_stop_pct_val / 100.0) * close_price if fixed_stop_pct_val > 0 else 0.0
                                atr_dist = (
                                    atr_values_for_stop[idx] * atr_stop_mult_val if atr_stop_len_val > 0 else 0.0
                                )
                                stop_dist = max(fixed_dist, atr_dist)
                                if stop_dist > 0:
                                    new_stop = close_price + stop_dist
                            if new_stop is not None:
                                if position_stop_price is None:
                                    position_stop_price = new_stop
                                else:
                                    position_stop_price = min(position_stop_price, new_stop)
                        if use_channel_stop and np.isfinite(channel_stop_upper_val):
                            channel_candidate = float(channel_stop_upper_val)
                            if position_stop_price is None:
                                position_stop_price = channel_candidate
                            else:
                                position_stop_price = min(position_stop_price, channel_candidate)
                        continue

        exit_long = False
        exit_short = False
        exit_long_reason: Optional[str] = None
        exit_short_reason: Optional[str] = None

        if position.direction > 0:
            if exit_opposite and base_short_signal and position.bars_held >= min_hold_bars_param:
                exit_long = True
                exit_long_reason = exit_long_reason or "opposite_signal"
            # Simplified momentum fade exit for long positions
            # Trigger a fade exit when momentum crosses below its signal
            # after being above zero for at least one bar and the absolute
            # momentum exceeds the configured minimum.
            if use_mom_fade:
                if (
                    cross_dn_vals[idx]
                    and mom_since_le_zero_vals[idx] > 0
                    and (mom_fade_min_abs <= 0 or mom_abs_vals[idx] >= mom_fade_min_abs)
                ):
                    exit_long = True
                    exit_long_reason = exit_long_reason or "mom_fade"
            if use_time_stop and max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                exit_long = True
                exit_long_reason = exit_long_reason or "time_stop"
            if (
                use_kasa
                and kasa_rsi_vals is not None
                and kasa_rsi_vals[idx] < kasa_rsi_ob
                and kasa_rsi_vals[prev_idx] >= kasa_rsi_ob
            ):
                exit_long = True
                exit_long_reason = exit_long_reason or "kasa_exit"
        elif position.direction < 0:
            if exit_opposite and base_long_signal and position.bars_held >= min_hold_bars_param:
                exit_short = True
                exit_short_reason = exit_short_reason or "opposite_signal"
            # Simplified momentum fade exit for short positions
            # Trigger a fade exit when momentum crosses above its signal
            # after being below zero for at least one bar and the absolute
            # momentum exceeds the configured minimum.
            if use_mom_fade:
                if (
                    cross_up_vals[idx]
                    and mom_since_ge_zero_vals[idx] > 0
                    and (mom_fade_min_abs <= 0 or mom_abs_vals[idx] >= mom_fade_min_abs)
                ):
                    exit_short = True
                    exit_short_reason = exit_short_reason or "mom_fade"
            if use_time_stop and max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                exit_short = True
                exit_short_reason = exit_short_reason or "time_stop"
            if (
                use_kasa
                and kasa_rsi_vals is not None
                and kasa_rsi_vals[idx] > kasa_rsi_os
                and kasa_rsi_vals[prev_idx] <= kasa_rsi_os
            ):
                exit_short = True
                exit_short_reason = exit_short_reason or "kasa_exit"

        is_shock = use_shock and shock_vals is not None and bool(shock_vals[idx])
        if position.direction > 0 and is_shock and shock_action == "즉시 청산":
            close_position(ts, close_price, "Volatility Shock")
            if ruin_hit:
                break
            continue
        if position.direction < 0 and is_shock and shock_action == "즉시 청산":
            close_position(ts, close_price, "Volatility Shock")
            if ruin_hit:
                break
            continue

        if position.direction > 0 and (exit_long or (is_shock and shock_action == "손절 타이트닝")):
            if exit_long:
                close_position(ts, close_price, exit_long_reason or "Exit Long")
                if ruin_hit:
                    break
                continue
        if position.direction < 0 and (exit_short or (is_shock and shock_action == "손절 타이트닝")):
            if exit_short:
                close_position(ts, close_price, exit_short_reason or "Exit Short")
                if ruin_hit:
                    break
                continue

        if position.direction > 0:
            stop_long = np.nan
            chandelier_stop_val = np.nan
            if (
                use_chandelier_exit
                and chandelier_high_vals is not None
                and chandelier_atr_vals is not None
            ):
                base = chandelier_high_vals[idx] - chandelier_atr_vals[idx] * chandelier_mult
                if np.isfinite(base):
                    if np.isnan(chandelier_long_trail):
                        chandelier_long_trail = base
                    else:
                        chandelier_long_trail = max(chandelier_long_trail, base)
                    chandelier_stop_val = chandelier_long_trail
            if use_atr_trail and not np.isnan(atr_trail_val):
                stop_long = close_price - atr_trail_val * atr_trail_mult * dyn_factor_val
            if use_stop_loss:
                stop_long = _max_ignore_nan(stop_long, swing_low_val)
                if use_pivot_stop:
                    pivot_ref = pivot_low_htf_val if use_pivot_htf else pivot_low_val
                    stop_long = _max_ignore_nan(stop_long, pivot_ref)
            if use_chandelier_exit:
                stop_long = _max_ignore_nan(stop_long, chandelier_stop_val)
            if use_sar_exit and sar_vals is not None:
                stop_long = _max_ignore_nan(stop_long, sar_vals[idx])
            if use_breakeven_stop and not np.isnan(highest_since_entry) and not np.isnan(atr_trail_val):
                move = highest_since_entry - position.avg_price
                trigger = atr_trail_val * breakeven_mult * dyn_factor_val
                if move >= trigger:
                    stop_long = _max_ignore_nan(stop_long, position.avg_price)
            if use_be_tiers and not np.isnan(highest_since_entry):
                atr_seed = atr_len_val
                if atr_seed > 0 and (highest_since_entry - position.avg_price) >= atr_seed:
                    stop_long = _max_ignore_nan(stop_long, position.avg_price)
            if not np.isnan(stop_long) and low_price <= stop_long:
                close_position(ts, stop_long, "Stop Long")
                if ruin_hit:
                    break
                continue
            if use_atr_profit and not np.isnan(atr_trail_val):
                target = position.avg_price + atr_trail_val * atr_profit_mult * dyn_factor_val
                if high_price >= target:
                    close_position(ts, target, "ATR Profit Long")
                    if ruin_hit:
                        break
                    continue
        elif position.direction < 0:
            stop_short = np.nan
            chandelier_stop_val = np.nan
            if (
                use_chandelier_exit
                and chandelier_low_vals is not None
                and chandelier_atr_vals is not None
            ):
                base = chandelier_low_vals[idx] + chandelier_atr_vals[idx] * chandelier_mult
                if np.isfinite(base):
                    if np.isnan(chandelier_short_trail):
                        chandelier_short_trail = base
                    else:
                        chandelier_short_trail = min(chandelier_short_trail, base)
                    chandelier_stop_val = chandelier_short_trail
            if use_atr_trail and not np.isnan(atr_trail_val):
                stop_short = close_price + atr_trail_val * atr_trail_mult * dyn_factor_val
            if use_stop_loss:
                stop_short = _min_ignore_nan(stop_short, swing_high_val)
                if use_pivot_stop:
                    pivot_ref = pivot_high_htf_val if use_pivot_htf else pivot_high_val
                    stop_short = _min_ignore_nan(stop_short, pivot_ref)
            if use_chandelier_exit:
                stop_short = _min_ignore_nan(stop_short, chandelier_stop_val)
            if use_sar_exit and sar_vals is not None:
                stop_short = _min_ignore_nan(stop_short, sar_vals[idx])
            if use_breakeven_stop and not np.isnan(lowest_since_entry) and not np.isnan(atr_trail_val):
                move = position.avg_price - lowest_since_entry
                trigger = atr_trail_val * breakeven_mult * dyn_factor_val
                if move >= trigger:
                    stop_short = _min_ignore_nan(stop_short, position.avg_price)
            if use_be_tiers and not np.isnan(lowest_since_entry):
                atr_seed = atr_len_val
                if atr_seed > 0 and (position.avg_price - lowest_since_entry) >= atr_seed:
                    stop_short = _min_ignore_nan(stop_short, position.avg_price)
            if not np.isnan(stop_short) and high_price >= stop_short:
                close_position(ts, stop_short, "Stop Short")
                if ruin_hit:
                    break
                continue
            if use_atr_profit and not np.isnan(atr_trail_val):
                target = position.avg_price - atr_trail_val * atr_profit_mult * dyn_factor_val
                if low_price <= target:
                    close_position(ts, target, "ATR Profit Short")
                    if ruin_hit:
                        break
                    continue

    if not ruin_hit and position.direction != 0 and position.entry_time is not None:
        close_position(df.index[-1], close_vals[-1], "EndOfData")

    return finalise_metrics_result(
        state,
        trades,
        returns_series,
        guard_flag=guard_frozen,
        simple_metrics_only=simple_metrics_only,
        penalty_config=penalty_config,
        validity_thresholds=validity_thresholds,
        ruin_hit=ruin_hit,
        leverage=leverage,
    )







