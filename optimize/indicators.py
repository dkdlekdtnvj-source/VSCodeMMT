"""보조 지표 및 시계열 계산 유틸리티."""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import njit


@njit  # type: ignore[misc]
def _rolling_rma_last(values: np.ndarray, length: int) -> np.ndarray:
    """슬라이딩 윈도우에 대해 Wilder RMA의 마지막 값을 계산합니다."""

    n = len(values)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = np.nan

    if length <= 0 or n == 0:
        return result

    inv_len = 1.0 / float(length)

    # 입력 시퀀스가 단조 증가/감소인지 감지합니다. 완전 단조열의 경우에는
    # 고전적인 Wilder RMA 재귀식을 사용하는 편이 정확도가 높습니다.
    monotonic = True
    prev_val = np.nan
    trend = 0  # 1: 증가, -1: 감소, 0: 아직 결정되지 않음
    for idx in range(n):
        value = float(values[idx])
        if np.isnan(value):
            continue
        if np.isnan(prev_val):
            prev_val = value
            continue
        delta = value - prev_val
        if delta > 0.0:
            if trend < 0:
                monotonic = False
                break
            trend = 1
        elif delta < 0.0:
            if trend > 0:
                monotonic = False
                break
            trend = -1
        prev_val = value

    if monotonic:
        count = 0
        acc = 0.0
        for idx in range(n):
            value = float(values[idx])
            if np.isnan(value):
                result[idx] = np.nan
                continue
            if count < length:
                acc += value
                count += 1
                if count == length:
                    acc /= float(length)
                    result[idx] = acc
            else:
                acc = acc + inv_len * (value - acc)
                result[idx] = acc
        return result

    for idx in range(length - 1, n):
        start = idx - length + 1
        acc = float(values[start])
        if np.isnan(acc):
            continue

        valid = True
        for pos in range(start + 1, idx + 1):
            v = float(values[pos])
            if np.isnan(v):
                valid = False
                break
            acc = (acc * (length - 1) + v) * inv_len

        if valid:
            result[idx] = acc

    return result


def _bars_since_mask(mask: pd.Series) -> pd.Series:
    mask_values = mask.fillna(False).to_numpy(dtype=bool)
    indices = np.arange(mask_values.shape[0], dtype=np.int64)
    last_true = np.where(mask_values, indices, -1)
    last_true = np.maximum.accumulate(last_true)
    counts = indices.astype(float) - last_true.astype(float)
    counts[last_true < 0] = np.inf
    return pd.Series(counts, index=mask.index, dtype=float)


def _ensure_series(values: Iterable[float], index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def _seeded_ewma(series: pd.Series, length: int, alpha: float) -> pd.Series:
    """TradingView와 동일한 초기화 방식으로 지수이동평균을 계산합니다."""

    length = max(int(length), 1)
    values = series.to_numpy(dtype=float, copy=False)
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size == 0:
        return pd.Series(result, index=series.index, dtype=float)

    window: deque[float] = deque(maxlen=length)
    window_sum = 0.0
    nan_count = 0
    prev = np.nan

    for idx, value in enumerate(values):
        if len(window) == length:
            oldest = window.popleft()
            if np.isnan(oldest):
                nan_count -= 1
            else:
                window_sum -= oldest

        window.append(value)
        if np.isnan(value):
            nan_count += 1
        else:
            window_sum += value

        if np.isnan(value):
            prev = np.nan
            result[idx] = np.nan
            continue

        if np.isnan(prev):
            if len(window) == length and nan_count == 0:
                prev = window_sum / float(length)
                result[idx] = prev
            else:
                result[idx] = np.nan
        else:
            prev = prev + (value - prev) * alpha
            result[idx] = prev

    return pd.Series(result, index=series.index, dtype=float)


def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    alpha = 2.0 / float(length + 1)
    return _seeded_ewma(series, length, alpha)


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    alpha = 1.0 / float(length)
    return _seeded_ewma(series, length, alpha)


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.rolling(length, min_periods=length).mean()


def _std(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.rolling(length, min_periods=length).std(ddof=0)


def _wma(series: pd.Series, length: int) -> pd.Series:
    """
    Compute the weighted moving average (WMA) of a series.
    Later used for Hull MA.
    """
    length = max(int(length), 1)
    weights = np.arange(1, length + 1, dtype=float)

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(values, weights) / weights.sum())

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)


def _hma(series: pd.Series, length: int) -> pd.Series:
    """
    Compute the Hull moving average of a series.
    HMA is defined as WMA(2*WMA(series, n/2) - WMA(series, n), sqrt(n)).
    """
    length = max(int(length), 1)
    half_len = max(int(round(length / 2.0)), 1)
    sqrt_len = max(int(round(np.sqrt(length))), 1)
    wma1 = _wma(series, half_len)
    wma2 = _wma(series, length)
    diff = 2.0 * wma1 - wma2
    return _wma(diff, sqrt_len)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return _rma(_true_range(df), length)


def _calc_basis(
    df: pd.DataFrame,
    length: int,
    style: str,
) -> pd.Series:
    length = max(int(length), 1)
    hl2 = (df["high"] + df["low"]) / 2.0
    bb_length = kc_length = length
    mid_kc = (df["high"].rolling(kc_length, min_periods=kc_length).max() + df["low"].rolling(kc_length, min_periods=kc_length).min()) / 2.0
    bb_basis_close = df["close"].rolling(bb_length, min_periods=bb_length).mean()
    mid_bb = hl2.rolling(bb_length, min_periods=bb_length).mean()
    kc_basis = hl2.rolling(kc_length, min_periods=kc_length).mean()
    kc_average = kc_basis
    midline = (hl2 + kc_average) / 2.0
    avg_line_avg = (bb_basis_close + mid_kc) / 2.0
    deluxe_basis = (mid_kc + mid_bb) / 2.0
    style_key = (style or "KC").lower()
    if style_key == "avg":
        basis = avg_line_avg
    elif style_key == "deluxe":
        basis = deluxe_basis
    elif style_key == "mod":
        basis = midline
    else:
        basis = mid_kc
    return basis.ffill().bfill()


def _momentum_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    ma = (ma_type or "SMA").lower()
    if ma == "ema":
        return _ema(series, length)
    if ma == "hma":
        return _hma(series, length)
    return _sma(series, length)


def _compute_momentum_block(
    df: pd.DataFrame,
    length: int,
    signal_len: int,
    *,
    style: str,
    ma_type: str,
    clip_enabled: bool,
    clip_limit: float,
) -> Dict[str, pd.Series]:
    length = max(int(length), 1)
    signal_len = max(int(signal_len), 1)
    tr1_raw = _true_range(df)
    tr1_safe = tr1_raw.replace(0.0, np.nan).ffill().bfill().fillna(1e-10)
    basis = _calc_basis(df, length, style)
    delta = df["close"] - basis
    norm = delta.divide(tr1_safe) * 100.0
    if clip_enabled:
        cap = max(float(clip_limit), 10.0)
        norm = norm.clip(lower=-cap, upper=cap)
    momentum = _linreg(norm, length)
    signal = _momentum_ma(momentum, signal_len, ma_type)
    hist = momentum - signal

    return {
        "momentum": momentum,
        "signal": signal,
        "hist": hist,
        "norm": norm,
        "delta": delta,
        "tr1_raw": tr1_raw,
        "tr1_safe": tr1_safe,
        "basis": basis,
    }


def _raw_directional_flux(df: pd.DataFrame, length: int) -> pd.Series:
    length = max(int(length), 1)
    high = df["high"]
    low = df["low"]
    plus_dm = (high.diff()).clip(lower=0.0)
    minus_dm = (low.shift() - low).clip(lower=0.0)
    plus_r = _rma(plus_dm.fillna(0.0), length)
    minus_r = _rma(minus_dm.fillna(0.0), length)
    denom = (plus_r + minus_r).replace(0.0, np.nan)
    flux = 100.0 * (plus_r - minus_r).divide(denom)
    return flux.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _raw_mod_directional_flux(df: pd.DataFrame, length: int) -> pd.Series:
    length = max(int(length), 1)
    high = df["high"]
    low = df["low"]
    up_move = (high.diff()).clip(lower=0.0)
    down_move = (low.shift() - low).clip(lower=0.0)
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    plus_r = _rma(pd.Series(plus_dm, index=df.index), length)
    minus_r = _rma(pd.Series(minus_dm, index=df.index), length)
    denom = (plus_r + minus_r).replace(0.0, np.nan)
    flux = 100.0 * (plus_r - minus_r).divide(denom)
    return flux.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _apply_flux_deadzone(series: pd.Series, deadzone: float) -> pd.Series:
    dz = max(float(deadzone), 0.0)
    if dz == 0.0:
        return series.copy()
    values = series.to_numpy(dtype=float, copy=False)
    result = np.full_like(values, np.nan)
    mask_pos = values > dz
    mask_neg = values < -dz
    result[mask_pos] = values[mask_pos] - dz
    result[mask_neg] = values[mask_neg] + dz
    return pd.Series(result, index=series.index)


def _compute_flux_block(
    df: pd.DataFrame,
    length: int,
    smooth_len: int,
    deadzone: float,
    *,
    use_heikin: bool,
    use_mod_flux: bool,
) -> Dict[str, pd.Series]:
    source = _heikin_ashi(df) if use_heikin else df
    base = (
        _raw_mod_directional_flux(source, length)
        if use_mod_flux
        else _raw_directional_flux(source, length)
    )
    smooth_len = max(int(smooth_len), 1)
    if smooth_len > 1:
        flux_raw = base.rolling(smooth_len, min_periods=smooth_len).mean()
    else:
        flux_raw = base
    flux_cut = _apply_flux_deadzone(flux_raw, deadzone)
    flux_gate = flux_cut.fillna(0.0)
    return {
        "raw": flux_raw.fillna(0.0),
        "cut": flux_cut,
        "gate": flux_gate,
    }


def _parabolic_sar(
    df: pd.DataFrame,
    start: float,
    increment: float,
    maximum: float,
) -> pd.Series:
    """Compute the parabolic SAR series matching TradingView semantics."""

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float, index=df.index)

    start = float(max(start, 0.0))
    increment = float(max(increment, 0.0))
    maximum = float(max(maximum, start))

    sar_values = np.full(n, np.nan, dtype=float)

    # Determine initial direction from the first two closes (default to long).
    closes = df["close"].to_numpy(dtype=float)
    long = True
    if n >= 2 and np.isfinite(closes[0]) and np.isfinite(closes[1]):
        long = closes[1] >= closes[0]

    accel = start
    ep = high[0] if long else low[0]
    sar = low[0] if long else high[0]
    sar_values[0] = sar

    for i in range(1, n):
        hi = high[i]
        lo = low[i]
        if not (np.isfinite(hi) and np.isfinite(lo)):
            sar_values[i] = sar_values[i - 1]
            continue

        prev_hi = high[i - 1]
        prev_lo = low[i - 1]
        prev_hi2 = high[i - 2] if i > 1 else prev_hi
        prev_lo2 = low[i - 2] if i > 1 else prev_lo

        sar = sar + accel * (ep - sar)

        if long:
            sar = min(sar, prev_lo, prev_lo2)
            if hi > ep:
                ep = hi
                accel = min(accel + increment, maximum)
            if lo < sar:
                long = False
                sar = ep
                ep = lo
                accel = start
                sar = max(sar, prev_hi, prev_hi2)
        else:
            sar = max(sar, prev_hi, prev_hi2)
            if lo < ep:
                ep = lo
                accel = min(accel + increment, maximum)
            if hi > sar:
                long = True
                sar = ep
                ep = hi
                accel = start
                sar = min(sar, prev_lo, prev_lo2)

        sar_values[i] = sar

    return pd.Series(sar_values, index=df.index, dtype=float)


def _chandelier_levels(df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return rolling highs/lows and ATR for chandelier stops."""

    length = max(int(length), 1)
    highest = df["high"].rolling(length, min_periods=1).max()
    lowest = df["low"].rolling(length, min_periods=1).min()
    atr = _atr(df, length)
    return highest, lowest, atr


def _linreg(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    if length == 1:
        return series.copy()

    idx = np.arange(length, dtype=float)

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        slope, intercept = np.polyfit(idx, values, 1)
        return slope * (length - 1) + intercept

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)


def _timeframe_to_offset(timeframe: str) -> Optional[str]:
    tf = str(timeframe).strip()
    if not tf:
        return None

    def _parse_multiplier(value: str) -> Optional[int]:
        if not value:
            return 1
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    tf_upper = tf.upper()
    tf_lower = tf.lower()

    if tf_upper.endswith("MS"):
        multiplier = _parse_multiplier(tf[:-2])
        if multiplier is not None:
            return f"{multiplier}MS"

    if tf.endswith("M"):
        multiplier = _parse_multiplier(tf[:-1])
        if multiplier is not None:
            return f"{multiplier}MS"

    if tf_lower.endswith("m"):
        multiplier = _parse_multiplier(tf_lower[:-1])
        if multiplier is not None:
            return f"{multiplier}min"

    if tf_lower.endswith("h"):
        multiplier = _parse_multiplier(tf_lower[:-1])
        if multiplier is not None:
            return f"{multiplier}H"

    if tf_lower.endswith("d"):
        multiplier = _parse_multiplier(tf_lower[:-1])
        if multiplier is not None:
            return f"{multiplier}D"

    if tf_lower.endswith("w"):
        multiplier = _parse_multiplier(tf_lower[:-1])
        if multiplier is not None:
            return f"{multiplier}W"

    if tf.isdigit():
        return f"{int(tf)}min"

    return None


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    offset = _timeframe_to_offset(timeframe)
    if offset is None:
        return df
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(offset, label="right", closed="right").agg(agg)
    return resampled.dropna()


def _security_series(
    df: pd.DataFrame, timeframe: str, compute: callable, default: float = np.nan
) -> pd.Series:
    if timeframe in {"", "0", None}:
        out = compute(df)
        return out if isinstance(out, pd.Series) else _ensure_series(out, df.index)
    resampled = _resample_ohlcv(df, timeframe)
    if resampled.empty:
        out = compute(df)
        return out if isinstance(out, pd.Series) else _ensure_series(out, df.index)
    result = compute(resampled)
    if not isinstance(result, pd.Series):
        result = _ensure_series(result, resampled.index)
    result = result.reindex(df.index, method="ffill")
    return result.fillna(default)


def _max_ignore_nan(*values: float) -> float:
    """NaN 을 무시하면서 최대값을 계산합니다.

    전달된 값이 모두 NaN 이거나 ``None`` 이면 ``np.nan`` 을 돌려 빈 시퀀스에 대한 ``max`` 호출을
    회피합니다. ``float`` 로 강제 변환 가능한 항목만 고려해 예외 발생 가능성을 낮춥니다.
    """

    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        cleaned.append(numeric)

    if not cleaned:
        return np.nan
    return max(cleaned)


def _min_ignore_nan(*values: float) -> float:
    """NaN 을 무시하면서 최소값을 계산합니다.

    ``values`` 가 모두 비어 있거나 유효하지 않은 경우 ``np.nan`` 을 반환해 ``min`` 의 빈 시퀀스
    예외를 방지합니다.
    """

    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        cleaned.append(numeric)

    if not cleaned:
        return np.nan
    return min(cleaned)


def _pivot_series(series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
    left = max(int(left), 1)
    right = max(int(right), 1)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.to_numpy()
    for idx in range(left, len(series) - right):
        window = values[idx - left : idx + right + 1]
        center = window[left]
        if is_high and center == window.max():
            result.iloc[idx + right] = center
        if not is_high and center == window.min():
            result.iloc[idx + right] = center
    return result.ffill()


def _dmi(df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    length = max(int(length), 1)
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = _rma(pd.Series(plus_dm, index=df.index), length)
    minus_dm = _rma(pd.Series(minus_dm, index=df.index), length)
    tr = _atr(df, length).replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm / tr)
    minus_di = 100.0 * (minus_dm / tr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = _rma(dx.fillna(0.0), length)
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)


def _rsi(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    diff = series.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    avg_gain = _rma(up, length)
    avg_loss = _rma(down, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)


def _stoch_rsi(series: pd.Series, length: int) -> pd.Series:
    rsi = _rsi(series, length)
    lowest = rsi.rolling(length, min_periods=length).min()
    highest = rsi.rolling(length, min_periods=length).max()
    denom = (highest - lowest).replace(0, np.nan)
    return ((rsi - lowest) / denom * 100.0).fillna(50.0)


def _obv_slope(close: pd.Series, volume: pd.Series, smooth: int) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * volume.fillna(0.0)).cumsum()
    return _ema(obv.diff().fillna(0.0), max(int(smooth), 1))


def _estimate_tick(series: pd.Series) -> float:
    diffs = series.diff().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float(series.iloc[-1]) * 1e-6 if len(series) else 0.01
    return float(diffs.min())


def _cross_over(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return prev_a <= prev_b and a > b


def _cross_under(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return prev_a >= prev_b and a < b


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    if len(df) > 0:
        ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([ha_open, ha_close, df["high"]], axis=1).max(axis=1)
    ha_low = pd.concat([ha_open, ha_close, df["low"]], axis=1).min(axis=1)
    ha["open"] = ha_open
    ha["close"] = ha_close
    ha["high"] = ha_high
    ha["low"] = ha_low
    return ha


def _directional_flux(df: pd.DataFrame, length: int, smooth_len: int) -> pd.Series:
    length = max(int(length), 1)
    smooth_len = max(int(smooth_len), 1)
    raw = _raw_directional_flux(df, length)
    if smooth_len > 1:
        raw = raw.rolling(smooth_len, min_periods=smooth_len).mean()
    return raw.fillna(0.0)


def _squeeze_momentum_norm(
    df: pd.DataFrame, length: int, atr_series: pd.Series, kc_mult: float
) -> pd.Series:
    """Squeeze Momentum Deluxe 지표의 정규화된 모멘텀 입력값을 계산합니다."""

    length = max(int(length), 1)

    hl2 = (df["high"] + df["low"]) / 2.0
    kc_basis = _sma(hl2, length)
    kc_range = atr_series * float(kc_mult)
    kc_upper = kc_basis + kc_range
    kc_lower = kc_basis - kc_range
    kc_average = (kc_upper + kc_lower) / 2.0
    midline = (hl2 + kc_average) / 2.0

    atr_safe = atr_series.replace(0.0, np.nan)
    norm = (df["close"] - midline).divide(atr_safe)
    norm = norm.replace([np.inf, -np.inf], np.nan)

    return (norm * 100.0).fillna(0.0)


def _compute_dynamic_thresholds(
    momentum: pd.Series,
    *,
    use_dynamic: bool,
    use_sym_threshold: bool,
    stat_threshold: float,
    buy_threshold: float,
    sell_threshold: float,
    dyn_len: int,
    dyn_mult: float,
) -> Tuple[pd.Series, pd.Series]:
    index = momentum.index
    if use_dynamic:
        dyn_window = max(int(dyn_len), 1)
        dyn_series = momentum.rolling(dyn_window, min_periods=dyn_window).std() * dyn_mult
        fallback = abs(stat_threshold) if stat_threshold else dyn_series.dropna().mean()
        if not np.isfinite(fallback) or fallback == 0:
            fallback = 1.0
        dyn_series = dyn_series.where(np.isfinite(dyn_series) & (dyn_series != 0), np.nan)
        dyn_series = dyn_series.fillna(fallback).abs()
        buy_series = -dyn_series
        sell_series = dyn_series
    else:
        if use_sym_threshold:
            buy_val = -abs(stat_threshold)
            sell_val = abs(stat_threshold)
        else:
            buy_val = -abs(buy_threshold)
            sell_val = abs(sell_threshold)
        buy_series = pd.Series(float(buy_val), index=index)
        sell_series = pd.Series(float(sell_val), index=index)
    return buy_series, sell_series

__all__ = [
    "_atr",
    "_bars_since_mask",
    "_chandelier_levels",
    "_compute_dynamic_thresholds",
    "_compute_flux_block",
    "_compute_momentum_block",
    "_cross_over",
    "_cross_under",
    "_directional_flux",
    "_dmi",
    "_ema",
    "_ensure_series",
    "_estimate_tick",
    "_heikin_ashi",
    "_hma",
    "_linreg",
    "_max_ignore_nan",
    "_min_ignore_nan",
    "_obv_slope",
    "_parabolic_sar",
    "_pivot_series",
    "_resample_ohlcv",
    "_rolling_rma_last",
    "_rma",
    "_rsi",
    "_security_series",
    "_seeded_ewma",
    "_sma",
    "_squeeze_momentum_norm",
    "_std",
    "_stoch_rsi",
    "_timeframe_to_offset",
    "_true_range",
    "_wma",
]
