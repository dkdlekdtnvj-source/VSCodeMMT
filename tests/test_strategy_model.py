import math
from typing import List

import numpy as np

import pandas as pd
import pytest

import optimize.strategy_model as strategy_model
import optimize.indicators as indicators_mod

from optimize.strategy_model import (
    _atr,
    _bars_since_mask,
    _compute_dynamic_thresholds,
    _ema,
    _hma,
    _linreg,
    _rma,
    _rolling_rma_last,
    _security_series,
    _sma,
    _std,
    _true_range,
    _wma,
    run_backtest,
)


def _pine_rma(values, length):
    length = max(int(length), 1)
    result = []
    prev = math.nan
    for idx, value in enumerate(values):
        if math.isnan(value):
            result.append(math.nan)
            prev = math.nan
            continue
        if math.isnan(prev):
            if idx + 1 >= length:
                window = values[idx - length + 1 : idx + 1]
                if any(math.isnan(v) for v in window):
                    result.append(math.nan)
                else:
                    prev = sum(window) / float(length)
                    result.append(prev)
            else:
                result.append(math.nan)
        else:
            prev = (prev * (length - 1) + value) / float(length)
            result.append(prev)
    return result


def _pine_ema(values, length):
    length = max(int(length), 1)
    alpha = 2.0 / float(length + 1)
    result = []
    prev = math.nan
    for idx, value in enumerate(values):
        if math.isnan(value):
            result.append(math.nan)
            prev = math.nan
            continue
        if math.isnan(prev):
            if idx + 1 >= length:
                window = values[idx - length + 1 : idx + 1]
                if any(math.isnan(v) for v in window):
                    result.append(math.nan)
                else:
                    prev = sum(window) / float(length)
                    result.append(prev)
            else:
                result.append(math.nan)
        else:
            prev = prev + (value - prev) * alpha
            result.append(prev)
    return result


def _pine_wma(values, length):
    length = max(int(length), 1)
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()
    result = []
    for idx in range(len(values)):
        if idx + 1 < length:
            result.append(math.nan)
            continue
        window = values[idx - length + 1 : idx + 1]
        if any(math.isnan(v) for v in window):
            result.append(math.nan)
            continue
        arr = np.array(window, dtype=float)
        result.append(float(np.dot(arr, weights) / denom))
    return result


def _pine_hma(values, length):
    length = max(int(length), 1)
    half_len = max(int(round(length / 2.0)), 1)
    sqrt_len = max(int(round(math.sqrt(length))), 1)
    wma1 = _pine_wma(values, half_len)
    wma2 = _pine_wma(values, length)
    diff = []
    for a, b in zip(wma1, wma2):
        if math.isnan(a) or math.isnan(b):
            diff.append(math.nan)
        else:
            diff.append(2.0 * a - b)
    return _pine_wma(diff, sqrt_len)


def test_rma_matches_tradingview_reference():
    data = pd.Series([math.nan, 1.0, 2.0, 3.0, 4.0, math.nan, 5.0, 6.0, 7.0])
    expected = pd.Series(_pine_rma(data.tolist(), 3), index=data.index, dtype=float)
    result = _rma(data, 3)
    pd.testing.assert_series_equal(result, expected)


def test_ema_matches_tradingview_reference():
    data = pd.Series([1.0, 2.0, 3.0, math.nan, 4.0, 5.0, 6.0, 7.0])
    expected = pd.Series(_pine_ema(data.tolist(), 4), index=data.index, dtype=float)
    result = _ema(data, 4)
    pd.testing.assert_series_equal(result, expected)


def test_wma_matches_tradingview_reference():
    data = pd.Series([1.0, 2.0, 3.0, 4.0, math.nan, 5.0, 6.0])
    expected = pd.Series(_pine_wma(data.tolist(), 3), index=data.index, dtype=float)
    result = _wma(data, 3)
    pd.testing.assert_series_equal(result, expected)


def test_hma_matches_tradingview_reference():
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, math.nan, 6.0, 7.0, 8.0, 9.0])
    expected = pd.Series(_pine_hma(data.tolist(), 5), index=data.index, dtype=float)
    result = _hma(data, 5)
    pd.testing.assert_series_equal(result, expected)


def test_std_matches_tradingview_reference():
    length = 4
    data = pd.Series([1.0, 2.0, 3.0, 4.0, math.nan, 5.0, 6.0, 7.0])
    expected_values = []
    for idx in range(len(data)):
        if idx + 1 < length:
            expected_values.append(math.nan)
            continue
        window = data.iloc[idx - length + 1 : idx + 1]
        if window.isna().any():
            expected_values.append(math.nan)
            continue
        mean = window.mean()
        variance = ((window - mean) ** 2).sum() / float(length)
        expected_values.append(float(math.sqrt(variance)))
    expected = pd.Series(expected_values, index=data.index, dtype=float)
    result = _std(data, length)
    pd.testing.assert_series_equal(result, expected)


def _make_ohlcv(prices):
    index = pd.date_range("2025-07-01", periods=len(prices), freq="1min", tz="UTC")
    close = pd.Series(prices, index=index)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )
    return df


def _base_params(**overrides):
    params = {
        "oscLen": 3,
        "signalLen": 1,
        "fluxLen": 3,
        "useFluxHeikin": False,
        "fluxDeadzone": 25.0,
        "useDynamicThresh": False,
        "useSymThreshold": True,
        "statThreshold": 0.0,
        "basisStyle": "Deluxe",
        "compatMode": True,
        "autoThresholdScale": True,
        "useNormClip": False,
        "normClipLimit": 350.0,
        "startDate": "2025-07-01T00:00:00",
        "allowLongEntry": True,
        "allowShortEntry": False,
        "debugForceLong": True,
    }
    params.update(overrides)
    return params


FEES = {"commission_pct": 0.0, "slippage_ticks": 0.0}
RISK = {"initial_capital": 1000.0, "min_trades": 0, "min_hold_bars": 0, "max_consecutive_losses": 10}


def test_debug_force_long_creates_trade():
    df = _make_ohlcv([100, 101, 102, 103, 104, 105])
    params = _base_params(useTimeStop=True, maxHoldBars=1)

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1


def test_daily_loss_guard_freezes_after_loss():
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        useDailyLossGuard=True,
        dailyLossLimit=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 1
    assert metrics["GuardFrozen"] == 1.0


def test_min_trades_argument_marks_invalid_when_threshold_not_met():
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        useDailyLossGuard=True,
        dailyLossLimit=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK, min_trades=2)

    assert metrics["Trades"] == pytest.approx(1.0)
    assert metrics["MinTrades"] == pytest.approx(2.0)
    assert not metrics["Valid"]


def test_squeeze_gate_blocks_without_release():
    df = _make_ohlcv([100] * 20)
    params = _base_params(
        useSqzGate=True,
        sqzReleaseBars=0,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 0


def test_stop_distance_guard_prevents_entry():
    prices = [100, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0]
    df = _make_ohlcv(prices)
    params = _base_params(
        useStopDistanceGuard=True,
        maxStopAtrMult=0.5,
        startDate=df.index[2].isoformat(),
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 0


@pytest.mark.parametrize(
    ("param_leverage", "risk_leverage", "expected"),
    [
        (25.0, 5.0, 25.0),
        (None, 7.5, 7.5),
        (None, None, 10.0),
    ],
)
def test_leverage_resolution_prefers_params(param_leverage, risk_leverage, expected):
    df = _make_ohlcv([100, 101, 102, 103])
    params = _base_params(useTimeStop=True, maxHoldBars=1)
    if param_leverage is not None:
        params["leverage"] = param_leverage
    risk = dict(RISK)
    if risk_leverage is not None:
        risk["leverage"] = risk_leverage
    metrics = run_backtest(df, params, FEES, risk)
    assert metrics["Leverage"] == pytest.approx(expected)


def test_timestamp_column_with_invalid_rows_is_cleaned():
    prices = [100, 101, 102, 103, 104, 105, 106]
    df = _make_ohlcv(prices)
    raw = df.reset_index().rename(columns={"index": "timestamp"})

    raw.loc[2, "timestamp"] = None  # invalid timestamp row -> should be dropped
    raw["close"] = raw["close"].astype(object)
    raw.loc[3, "close"] = "bad"  # non-numeric OHLC value -> should be coerced then dropped
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw.loc[len(raw) - 1, "timestamp"] = raw.loc[1, "timestamp"]  # duplicate timestamp

    # Pandas 3.x는 object 컬럼에 문자열/숫자가 혼재하면 향후 dtype 호환성 경고를 발생시킨다.
    # 청소 로직은 NaN 값이 있는 행을 제거하므로, 테스트에서도 명시적으로 수치형으로 변환해
    # 경고 없이 동일한 경로를 밟도록 맞춘다.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    raw[numeric_cols] = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")

    params = _base_params(useTimeStop=True, maxHoldBars=1)

    metrics = run_backtest(raw, params, FEES, RISK)

    returns = metrics["Returns"]
    assert isinstance(returns, pd.Series)
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert returns.index.tz is not None
    assert 0 < len(returns) < len(raw)


def test_short_stop_handles_missing_candidates():
    prices = [110, 109, 108, 107, 106, 105, 104, 103]
    df = _make_ohlcv(prices)
    params = _base_params(
        allowLongEntry=False,
        allowShortEntry=True,
        debugForceLong=False,
        debugForceShort=True,
        useStructureGate=True,
        useBOS=True,
        useCHOCH=True,
        useStopLoss=True,
        stopLookback=3,
        usePivotStop=True,
        useTimeStop=True,
        maxHoldBars=1,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1


def test_security_series_resamples_monthly_timeframe():
    index = pd.date_range("2025-01-01", periods=65, freq="D", tz="UTC")
    close = pd.Series(range(len(index)), index=index)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
        }
    )

    captured = {}

    def _compute(resampled: pd.DataFrame) -> pd.Series:
        captured["index"] = resampled.index
        return resampled["close"]

    result = _security_series(df, "1M", _compute)

    assert "index" in captured
    assert captured["index"].freqstr in {"MS", "M"}

    period_index = result.index.tz_localize(None).to_period("M")
    unique_per_month = result.groupby(period_index).nunique()
    assert (unique_per_month == 1).all()


def test_mom_fade_bars_since_vectorised_matches_reference():
    hist = pd.Series(
        [0.5, -0.2, -0.1, 0.3, 0.2, -0.4, 0.1, 0.0, 0.6, -0.3],
        index=pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC"),
    )

    nonpos_mask = hist.le(0)
    nonneg_mask = hist.ge(0)

    vector_nonpos = _bars_since_mask(nonpos_mask)
    vector_nonneg = _bars_since_mask(nonneg_mask)

    def _reference(mask: pd.Series) -> pd.Series:
        results = []
        for idx, _ in enumerate(mask):
            count = 0
            found = False
            for lookback in range(idx, -1, -1):
                if mask.iloc[lookback]:
                    results.append(float(count))
                    found = True
                    break
                count += 1
            if not found:
                results.append(float("inf"))
        return pd.Series(results, index=mask.index, dtype=float)

    ref_nonpos = _reference(nonpos_mask)
    ref_nonneg = _reference(nonneg_mask)

    for left, right in zip(vector_nonpos, ref_nonpos):
        if math.isinf(right):
            assert math.isinf(left)
        else:
            assert left == pytest.approx(right)

    for left, right in zip(vector_nonneg, ref_nonneg):
        if math.isinf(right):
            assert math.isinf(left)
        else:
            assert left == pytest.approx(right)


def test_volatility_guard_atr_matches_reference():
    prices = [100, 101, 100.5, 102, 101.5, 103, 102.5, 104, 103.5, 105]
    df = _make_ohlcv(prices)
    window = 3

    tr_series = _true_range(df)
    atr_values = _rolling_rma_last(tr_series.to_numpy(dtype=float), window)

    computed = []
    for idx, close_value in enumerate(df["close"].to_numpy(dtype=float)):
        if idx >= window and not math.isnan(atr_values[idx]) and close_value != 0.0:
            computed.append(atr_values[idx] / close_value * 100.0)
        else:
            computed.append(0.0)

    expected = []
    for idx in range(len(df)):
        if idx >= window:
            window_tr = tr_series.iloc[idx - window + 1 : idx + 1].to_numpy(dtype=float)
            acc = window_tr[0]
            for value in window_tr[1:]:
                acc = (acc * (window - 1) + value) / window
            expected.append(acc / df["close"].iloc[idx] * 100.0)
        else:
            expected.append(0.0)

    assert computed == pytest.approx(expected)


def test_rolling_rma_last_matches_recursive_formula():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    result = _rolling_rma_last(values, 3)

    assert np.isnan(result[0])
    assert np.isnan(result[1])
    expected_tail = [2.0, 2.6666666667, 3.4444444444]
    assert result[2:] == pytest.approx(expected_tail)

    zero_length = _rolling_rma_last(values, 0)
    assert np.isnan(zero_length).all()


def _run_par_scenario(prices, **overrides):
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        usePerfAdaptiveRisk=True,
        parMinTrades=1,
        parLookback=10,
        parHotWinRate=10.0,
        parColdWinRate=90.0,
        parPauseOnCold=False,
        baseQtyPercent=10.0,
    )
    params.update(overrides)
    metrics = run_backtest(df, params, FEES, RISK)
    return metrics["TradesList"]


def _run_par_metrics(prices, **overrides):
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        usePerfAdaptiveRisk=True,
        parMinTrades=1,
        parLookback=10,
        parHotWinRate=10.0,
        parColdWinRate=90.0,
        parPauseOnCold=False,
        baseQtyPercent=10.0,
    )
    params.update(overrides)
    return run_backtest(df, params, FEES, RISK)


@pytest.mark.parametrize(
    (
        "prices",
        "primary_overrides",
        "fallback_overrides",
        "expected_ratio",
        "rel_tol",
    ),
    [
        (
            [100, 101, 102, 103, 104, 105],
            {"parHotMult": 2.0, "parHotRiskMult": 0.5},
            {"parHotRiskMult": 0.5},
            4.0,
            0.15,
        ),
        (
            [105, 104, 103, 102, 101, 100],
            {"parColdMult": 0.2, "parColdRiskMult": 0.5},
            {"parColdRiskMult": 0.5},
            0.4,
            0.2,
        ),
    ],
)
def test_perf_adaptive_risk_prefers_primary_multiplier_keys(
    prices,
    primary_overrides,
    fallback_overrides,
    expected_ratio,
    rel_tol,
):
    primary_trades = _run_par_scenario(prices, **primary_overrides)
    fallback_trades = _run_par_scenario(prices, **fallback_overrides)

    assert len(primary_trades) >= 2
    assert len(fallback_trades) >= 2

    ratio = primary_trades[1].size / fallback_trades[1].size
    assert ratio == pytest.approx(expected_ratio, rel=rel_tol)


def test_perf_adaptive_risk_scaling_impacts_realized_pnl():
    prices = [100, 101, 102, 103, 104, 105]

    hot_metrics = _run_par_metrics(
        prices,
        parHotMult=2.0,
        parHotRiskMult=0.5,
    )
    fallback_metrics = _run_par_metrics(
        prices,
        parHotRiskMult=0.5,
    )

    hot_trades = hot_metrics["TradesList"]
    fallback_trades = fallback_metrics["TradesList"]

    assert len(hot_trades) >= 3
    assert len(fallback_trades) == len(hot_trades)

    base_profit = float(fallback_trades[0].profit)
    assert float(hot_trades[0].profit) == pytest.approx(base_profit)

    expected_hot_gross = float(hot_trades[0].profit)
    expected_fallback_gross = base_profit

    for hot_trade, fallback_trade in zip(hot_trades[1:], fallback_trades[1:]):
        ratio = float(hot_trade.size / fallback_trade.size)
        assert ratio > 1.0
        assert float(hot_trade.profit) == pytest.approx(float(fallback_trade.profit) * ratio, rel=1e-6)
        expected_hot_gross += float(fallback_trade.profit) * ratio
        expected_fallback_gross += float(fallback_trade.profit)

    assert fallback_metrics["GrossProfit"] == pytest.approx(expected_fallback_gross, rel=1e-6)
    assert hot_metrics["GrossProfit"] == pytest.approx(expected_hot_gross, rel=1e-6)

    initial_capital = RISK["initial_capital"]
    assert fallback_metrics["NetProfit"] == pytest.approx(
        fallback_metrics["GrossProfit"] / initial_capital, rel=0.02
    )
    assert hot_metrics["NetProfit"] == pytest.approx(
        hot_metrics["GrossProfit"] / initial_capital, rel=0.02
    )

    assert hot_metrics["NetProfit"] > fallback_metrics["NetProfit"]
    gross_ratio = hot_metrics["GrossProfit"] / fallback_metrics["GrossProfit"]
    net_ratio = hot_metrics["NetProfit"] / fallback_metrics["NetProfit"]
    assert net_ratio == pytest.approx(gross_ratio, rel=0.05)
def test_perf_adaptive_risk_prefers_primary_multiplier_keys():
    hot_trades = _run_par_scenario(
        [100, 101, 102, 103, 104, 105],
        parHotMult=2.0,
        parHotRiskMult=0.5,
    )
    fallback_trades = _run_par_scenario(
        [100, 101, 102, 103, 104, 105],
        parHotRiskMult=0.5,
    )

    assert len(hot_trades) >= 2
    assert len(fallback_trades) >= 2
    ratio = hot_trades[1].size / fallback_trades[1].size
    assert ratio == pytest.approx(4.0, rel=0.15)


def test_perf_adaptive_risk_prefers_primary_cold_multiplier():
    cold_trades = _run_par_scenario(
        [105, 104, 103, 102, 101, 100],
        parColdMult=0.2,
        parColdRiskMult=0.5,
    )
    fallback_trades = _run_par_scenario(
        [105, 104, 103, 102, 101, 100],
        parColdRiskMult=0.5,
    )

    assert len(cold_trades) >= 2
    assert len(fallback_trades) >= 2
    ratio = cold_trades[1].size / fallback_trades[1].size
    assert ratio == pytest.approx(0.4, rel=0.2)


def test_zero_atr_norm_uses_unity_and_remains_finite():
    df = _make_ohlcv([100.0] * 30)
    kc_len = 5
    bb_len = 5
    osc_len = 5

    atr_primary = _atr(df, kc_len)
    atr_norm = atr_primary.replace(0.0, np.nan).fillna(1.0)
    highest_high = df["high"].rolling(kc_len).max()
    lowest_low = df["low"].rolling(kc_len).min()
    mean_kc = (highest_high + lowest_low) / 2.0
    bb_basis_close = _sma(df["close"], bb_len)
    avg_line_avg = (bb_basis_close + mean_kc) / 2.0
    norm_raw = (df["close"] - avg_line_avg).divide(atr_norm)

    assert norm_raw.iloc[kc_len - 1 :].isna().sum() == 0

    norm_series = (norm_raw * 100.0).fillna(0.0)
    momentum = _linreg(norm_series, osc_len)
    assert np.isfinite(momentum.iloc[osc_len - 1 :]).all()


def test_kc_and_deluxe_styles_use_tr1_normalisation(monkeypatch):
    prices = [100, 102, 101, 105, 108, 112, 118, 121]
    df = _make_ohlcv(prices)
    kc_len = 3
    bb_len = 3
    osc_len = 3
    kc_mult = 1.4

    captured: List[pd.Series] = []

    def fake_linreg(series: pd.Series, length: int) -> pd.Series:
        captured.append(series.copy())
        return pd.Series(0.0, index=series.index)

    monkeypatch.setattr(strategy_model, "_linreg", fake_linreg)
    monkeypatch.setattr(indicators_mod, "_linreg", fake_linreg)

    common_kwargs = dict(
        useTimeStop=True,
        maxHoldBars=1,
        kcLen=kc_len,
        kcMult=kc_mult,
        bbLen=bb_len,
        bbMult=1.8,
        oscLen=osc_len,
        signalLen=1,
    )

    tr1 = indicators_mod._true_range(df)
    tr1_norm = tr1.replace(0.0, np.nan).ffill().bfill().fillna(1e-10)

    basis_kc = indicators_mod._calc_basis(df, osc_len, "kc")
    basis_deluxe = indicators_mod._calc_basis(df, osc_len, "deluxe")
    expected_kc = (df["close"] - basis_kc).divide(tr1_norm) * 100.0
    expected_kc = expected_kc.fillna(0.0)
    expected_deluxe = (df["close"] - basis_deluxe).divide(tr1_norm) * 100.0
    expected_deluxe = expected_deluxe.fillna(0.0)

    for style_name, expected in (("KC", expected_kc), ("Deluxe", expected_deluxe)):
        captured.clear()
        params = _base_params(basisStyle=style_name, compatMode=True, **common_kwargs)
        run_backtest(df, params, FEES, RISK)
        assert len(captured) == 1
        pd.testing.assert_series_equal(captured[0], expected)

        captured.clear()
        params = _base_params(basisStyle=style_name, compatMode=False, **common_kwargs)
        run_backtest(df, params, FEES, RISK)
        assert len(captured) == 1
        pd.testing.assert_series_equal(captured[0], expected)


def test_dynamic_thresholds_replace_zero_with_fallback():
    index = pd.date_range("2025-01-01", periods=30, freq="1min", tz="UTC")
    momentum = pd.Series(0.0, index=index)
    buy, sell = _compute_dynamic_thresholds(
        momentum,
        use_dynamic=True,
        use_sym_threshold=True,
        stat_threshold=12.0,
        buy_threshold=8.0,
        sell_threshold=8.0,
        dyn_len=5,
        dyn_mult=1.0,
    )

    assert (buy == -12.0).all()
    assert (sell == 12.0).all()


def test_capital_is_locked_and_restored():
    df = _make_ohlcv([100, 110])
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        baseQtyPercent=50,
        minTradableCapital=0,
    )
    risk = {**RISK, "initial_capital": 1000.0, "leverage": 1.0}

    metrics = run_backtest(df, params, FEES, risk)

    assert metrics["Trades"] == 1
    total_unlocked = metrics["AvailableCapital"] + metrics["Savings"]
    assert total_unlocked == pytest.approx(1050.0)
    assert metrics["LockedCapital"] == pytest.approx(0.0)
    assert metrics["TotalAssets"] == pytest.approx(total_unlocked)


def test_fixed_usd_entry_respects_available_capital():
    df = _make_ohlcv([100, 101, 102])
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        useSizingOverride=True,
        sizingMode="고정 금액 (USD)",
        fixedUsdAmount=2000.0,
        minTradableCapital=0,
    )
    risk = {**RISK, "initial_capital": 1000.0, "leverage": 1.0}

    metrics = run_backtest(df, params, FEES, risk)

    assert metrics["Trades"] == 0
    assert metrics["AvailableCapital"] == pytest.approx(1000.0)
    assert metrics["LockedCapital"] == pytest.approx(0.0)


def test_ruin_halts_additional_trades():
    df = _make_ohlcv([100, 0, 200, 300, 400])
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        baseQtyPercent=100,
        minTradableCapital=0,
    )
    risk = {**RISK, "initial_capital": 100.0, "leverage": 1.0}

    metrics = run_backtest(df, params, FEES, risk)

    assert metrics["Trades"] == 1
    assert metrics["TradesList"][0].reason == "Liquidation"
    assert metrics["Ruin"] == 1.0
    assert metrics["AvailableCapital"] == pytest.approx(10.0)
    assert metrics["TotalAssets"] == pytest.approx(10.0)


def test_chandelier_exit_triggers_stop():
    df = _make_ohlcv([100, 102, 104, 103, 95, 94])
    params = _base_params(
        useStopLoss=False,
        useAtrTrail=False,
        useChandelierExit=True,
        chandelierLen=1,
        chandelierMult=0.5,
        useSarExit=False,
        exitOpposite=False,
        useMomFade=False,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1
    assert any(trade.reason == "Stop Long" for trade in metrics["TradesList"])


def test_parabolic_sar_exit_triggers_stop():
    df = _make_ohlcv([100, 99, 98, 97, 96, 95, 94, 110, 120])
    params = _base_params(
        allowLongEntry=False,
        allowShortEntry=True,
        debugForceLong=False,
        debugForceShort=True,
        useStopLoss=False,
        useAtrTrail=False,
        useChandelierExit=False,
        useSarExit=True,
        sarStart=0.05,
        sarIncrement=0.05,
        sarMaximum=0.2,
        exitOpposite=False,
        useMomFade=False,
        leverage=1,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1
    assert any(trade.reason == "Stop Short" for trade in metrics["TradesList"])


def test_single_pyramiding_adds_one_extra_leg():
    prices = [100, 101, 102, 103, 104]
    df = _make_ohlcv(prices)
    base_kwargs = dict(
        useTimeStop=False,
        maxHoldBars=0,
        baseQtyPercent=50,
        minTradableCapital=0,
        reentryBars=5,
        fixedStopPct=0.0,
        atrStopLen=0,
        atrStopMult=0.0,
        useStopLoss=False,
        useAtrTrail=False,
        exitOpposite=False,
        useMomFade=False,
    )

    params_no_pyramid = _base_params(usePyramiding=False, **base_kwargs)
    params_with_pyramid = _base_params(usePyramiding=True, **base_kwargs)
    risk = {**RISK, "initial_capital": 1000.0, "leverage": 1.0}

    metrics_no = run_backtest(df, params_no_pyramid, FEES, risk)
    metrics_yes = run_backtest(df, params_with_pyramid, FEES, risk)

    assert metrics_no["Trades"] == pytest.approx(1.0)
    assert metrics_yes["Trades"] == pytest.approx(1.0)

    trade_no = metrics_no["TradesList"][0]
    trade_yes = metrics_yes["TradesList"][0]

    assert trade_no.size == pytest.approx(5.0)
    assert trade_yes.size == pytest.approx(10.0)
