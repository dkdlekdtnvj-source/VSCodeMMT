from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from optimize import alternative_engine as alt
from optimize import indicators
from optimize.run import combine_metrics


class DummyRecords:
    def __init__(self, records: List[Dict[str, object]]):
        self.records_readable = pd.DataFrame(records)


class DummyTrades:
    def __init__(self, records: List[Dict[str, object]]):
        self.records_readable = pd.DataFrame(records)


class DummyPortfolio:
    def __init__(self, close_series: pd.Series, returns: np.ndarray, records: List[Dict[str, object]]):
        self._close = close_series
        self._returns = returns
        self.trades = DummyTrades(records)

    @property
    def close(self) -> pd.Series:
        return self._close

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @staticmethod
    def from_signals(
        close: pd.Series,
        entries: np.ndarray,
        exits: np.ndarray,
        short_entries: np.ndarray,
        short_exits: np.ndarray,
        fees: float,
        size: float,
        size_type: str,
        upon_opposite_entry: str,
        **kwargs,
    ) -> "DummyPortfolio":
        returns = np.array([0.02, -0.01, 0.03], dtype=float)
        records = [
            {
                "Entry Timestamp": close.index[0],
                "Exit Timestamp": close.index[1],
                "Direction": "Long",
                "Size": size,
                "Avg Entry Price": float(close.iloc[0]),
                "Avg Exit Price": float(close.iloc[1]),
                "PnL": 10.0,
                "Return": 0.1,
            }
        ]
        return DummyPortfolio(close, returns, records)


class LegacyPortfolio(DummyPortfolio):
    call_count = 0

    @staticmethod
    def from_signals(
        close: pd.Series,
        entries: np.ndarray,
        exits: np.ndarray,
        short_entries: np.ndarray,
        short_exits: np.ndarray,
        fees: float,
        size: float,
        size_type: str,
        upon_opposite_entry: str,
        **kwargs,
    ) -> "DummyPortfolio":
        LegacyPortfolio.call_count += 1
        if "execute_on_close" in kwargs:
            raise TypeError("execute_on_close")
        return DummyPortfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            fees=fees,
            size=size,
            size_type=size_type,
            upon_opposite_entry=upon_opposite_entry,
            **kwargs,
        )


def _base_params(**overrides: object) -> Dict[str, object]:
    params: Dict[str, object] = {
        "oscLen": 3,
        "signalLen": 1,
        "kcLen": 3,
        "kcMult": 1.4,
        "bbLen": 3,
        "bbMult": 1.8,
        "fluxLen": 3,
        "fluxSmoothLen": 1,
        "useFluxHeikin": False,
        "useModFlux": False,
        "fluxDeadzone": 25.0,
        "basisStyle": "Deluxe",
        "compatMode": True,
        "autoThresholdScale": True,
        "useNormClip": False,
        "normClipLimit": 350.0,
    }
    params.update(overrides)
    return params


def test_alt_engine_kc_and_deluxe_styles_use_tr1_normalisation(monkeypatch):
    index = pd.date_range("2025-07-01", periods=8, freq="1min", tz="UTC")
    close = pd.Series([100, 102, 101, 105, 108, 112, 118, 121], index=index)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )

    captured: List[pd.Series] = []

    def fake_linreg(series: pd.Series, length: int) -> pd.Series:
        captured.append(series.copy())
        return pd.Series(0.0, index=series.index)

    monkeypatch.setattr(indicators, "_linreg", fake_linreg)

    tr1 = indicators._true_range(df)
    tr1_norm = tr1.replace(0.0, np.nan).ffill().bfill().fillna(1e-10)

    for style_name in ("KC", "Deluxe"):
        base_params = _base_params(basisStyle=style_name, compatMode=True)
        basis = indicators._calc_basis(df, base_params["oscLen"], style_name)
        expected = (df["close"] - basis).divide(tr1_norm) * 100.0
        expected = expected.fillna(0.0)

        captured.clear()
        alt._compute_indicators(df, base_params)
        assert len(captured) == 1
        pdt.assert_series_equal(captured[0], expected)

        captured.clear()
        alt._compute_indicators(df, _base_params(basisStyle=style_name, compatMode=False))
        assert len(captured) == 1
        pdt.assert_series_equal(captured[0], expected)


def test_vectorbt_backtest_returns_trades_and_returns(monkeypatch):
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )

    parsed = alt._ParsedInputs(
        df=df,
        htf_df=None,
        start_ts=index[0],
        commission_pct=0.0005,
        slippage_ticks=0.0,
        leverage=1.0,
        initial_capital=1000.0,
        capital_pct=1.0,
        allow_long=True,
        allow_short=False,
        require_cross=False,
        exit_opposite=False,
        min_trades=0,
        min_hold_bars=0,
        max_consecutive_losses=10,
    )

    def fake_validate(params: Dict[str, object]) -> None:
        return None

    def fake_signals(df: pd.DataFrame, params: Dict[str, object], parsed_inputs: alt._ParsedInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        entries = np.array([True, False, False])
        exits = np.array([False, True, False])
        shorts = np.zeros_like(entries, dtype=bool)
        return entries, exits, shorts, shorts

    def fake_apply(parsed_inputs, params, long_entries, long_exits, short_entries, short_exits):
        return long_entries, long_exits, short_entries, short_exits

    monkeypatch.setattr(alt, "_validate_feature_flags", fake_validate)
    monkeypatch.setattr(alt, "_build_signals", fake_signals)
    monkeypatch.setattr(alt, "_apply_exit_overrides", fake_apply)
    monkeypatch.setattr(alt, "VECTORBT_AVAILABLE", True)
    monkeypatch.setattr(alt, "_VBT_MODULE", type("DummyModule", (), {"Portfolio": DummyPortfolio}))

    metrics = alt._vectorbt_backtest(parsed, params={})

    assert "TradesList" in metrics
    assert "Returns" in metrics
    assert isinstance(metrics["TradesList"], list)
    assert isinstance(metrics["Returns"], pd.Series)
    assert len(metrics["Returns"]) == len(df)

    combined = combine_metrics([metrics])
    assert combined["Trades"] == len(metrics["TradesList"])
    assert combined["NetProfit"] != 0.0


def test_vectorbt_backtest_legacy_execute_on_close_fallback(monkeypatch):
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )

    parsed = alt._ParsedInputs(
        df=df,
        htf_df=None,
        start_ts=index[0],
        commission_pct=0.0005,
        slippage_ticks=0.0,
        leverage=1.0,
        initial_capital=1000.0,
        capital_pct=1.0,
        allow_long=True,
        allow_short=False,
        require_cross=False,
        exit_opposite=False,
        min_trades=0,
        min_hold_bars=0,
        max_consecutive_losses=10,
    )

    def fake_validate(params: Dict[str, object]) -> None:
        return None

    def fake_signals(df: pd.DataFrame, params: Dict[str, object], parsed_inputs: alt._ParsedInputs):
        entries = np.array([True, False, False])
        exits = np.array([False, True, False])
        shorts = np.zeros_like(entries, dtype=bool)
        return entries, exits, shorts, shorts

    def fake_apply(parsed_inputs, params, long_entries, long_exits, short_entries, short_exits):
        return long_entries, long_exits, short_entries, short_exits

    monkeypatch.setattr(alt, "_validate_feature_flags", fake_validate)
    monkeypatch.setattr(alt, "_build_signals", fake_signals)
    monkeypatch.setattr(alt, "_apply_exit_overrides", fake_apply)
    monkeypatch.setattr(alt, "VECTORBT_AVAILABLE", True)
    LegacyPortfolio.call_count = 0
    monkeypatch.setattr(alt, "_VBT_MODULE", type("LegacyModule", (), {"Portfolio": LegacyPortfolio}))

    with pytest.warns(RuntimeWarning) as record:
        metrics = alt._vectorbt_backtest(parsed, params={})

    assert len(record) == 1
    assert "execute_on_close" in str(record[0].message)
    assert LegacyPortfolio.call_count == 2
    assert metrics["Engine"] == "vectorbt"


def test_parse_core_settings_leverage_priority():
    index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [100.2, 101.2, 102.2, 103.2],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    base_params: Dict[str, object] = {"startDate": index[0].isoformat()}
    risk = {"leverage": 3.0}

    parsed_with_param = alt._parse_core_settings(
        df,
        {**base_params, "leverage": 17},
        fees={},
        risk=risk,
        min_trades=None,
        htf_df=None,
    )
    assert parsed_with_param.leverage == pytest.approx(17.0)

    parsed_with_risk = alt._parse_core_settings(
        df,
        dict(base_params),
        fees={},
        risk=risk,
        min_trades=None,
        htf_df=None,
    )
    assert parsed_with_risk.leverage == pytest.approx(3.0)


@pytest.mark.parametrize(
    "wallet_params",
    [
        {"useWallet": True},
        {"profitReservePct": 15.0},
        {"applyReserveToSizing": True},
    ],
)
def test_run_backtest_alternative_wallet_flags_force_reference(monkeypatch, wallet_params):
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )

    called = {}

    def fake_reference(df_arg, params_arg, fees_arg, risk_arg, *, htf_df=None, min_trades=None):
        called["args"] = (df_arg, params_arg, fees_arg, risk_arg, htf_df, min_trades)
        return {"Engine": "reference", "TradesList": []}

    monkeypatch.setattr(alt, "_run_backtest_reference", fake_reference)

    result = alt.run_backtest_alternative(
        df,
        params=dict(wallet_params),
        fees={},
        risk={},
        htf_df=None,
        min_trades=None,
        engine="vectorbt",
    )

    assert "args" in called
    assert result["Engine"] == "reference"


def test_run_backtest_alternative_defaults_to_reference(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )

    called = {}

    def fake_reference(df_arg, params_arg, fees_arg, risk_arg, *, htf_df=None, min_trades=None):
        called["args"] = (df_arg, params_arg, fees_arg, risk_arg, htf_df, min_trades)
        return {"Engine": "reference"}

    monkeypatch.setattr(alt, "_run_backtest_reference", fake_reference)

    result = alt.run_backtest_alternative(
        df,
        params={},
        fees={},
        risk={},
        htf_df=None,
        min_trades=None,
        engine="vectorbt",
    )

    assert "args" in called
    assert result["Engine"] == "reference"


def test_run_backtest_alternative_force_flag_allows_vector_engine(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )

    parsed_marker = object()

    def fake_parse(df_arg, params_arg, fees_arg, risk_arg, min_trades=None, htf_df=None):
        assert params_arg["forceAltEngine"] is True
        return parsed_marker

    vector_called = {}

    def fake_vector(parsed_arg, params_arg):
        vector_called["args"] = (parsed_arg, params_arg)
        return {"Engine": "vectorbt"}

    def fail_reference(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("reference engine should not run when forceAltEngine is True")

    monkeypatch.setattr(alt, "_parse_core_settings", fake_parse)
    monkeypatch.setattr(alt, "_vectorbt_backtest", fake_vector)
    monkeypatch.setattr(alt, "_run_backtest_reference", fail_reference)

    result = alt.run_backtest_alternative(
        df,
        params={"forceAltEngine": True},
        fees={},
        risk={},
        htf_df=None,
        min_trades=None,
        engine="vectorbt",
    )

    assert vector_called["args"] == (parsed_marker, {"forceAltEngine": True})
    assert result["Engine"] == "vectorbt"


@pytest.mark.parametrize("use_flux_heikin", [False, True])
def test_compute_indicators_with_mod_flux_matches_manual_calculation(use_flux_heikin):
    index = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    base = np.linspace(100.0, 110.0, num=len(index))
    df = pd.DataFrame(
        {
            "open": base + np.random.default_rng(42).normal(0, 0.5, size=len(index)),
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + np.random.default_rng(24).normal(0, 0.3, size=len(index)),
            "volume": np.linspace(1.0, 2.0, num=len(index)),
        },
        index=index,
    )

    params = {
        "oscLen": 10,
        "signalLen": 3,
        "fluxLen": 8,
        "fluxSmoothLen": 3,
        "useFluxHeikin": use_flux_heikin,
        "useModFlux": True,
        "fluxDeadzone": 0.0,
        "kcLen": 10,
        "kcMult": 1.0,
        "bbLen": 12,
        "bbMult": 1.2,
        "maType": "SMA",
        "basisStyle": "Deluxe",
        "compatMode": True,
        "autoThresholdScale": True,
        "useNormClip": False,
        "normClipLimit": 350.0,
    }

    (
        _momentum,
        _signal,
        _cross_up,
        _cross_down,
        flux_hist,
        flux_gate,
        _threshold_scale,
    ) = alt._compute_indicators(df, params)

    source_df = indicators._heikin_ashi(df) if use_flux_heikin else df
    mod_flux_base = indicators._raw_mod_directional_flux(source_df, params["fluxLen"])
    if params["fluxSmoothLen"] > 1:
        expected_raw = mod_flux_base.rolling(
            params["fluxSmoothLen"], min_periods=params["fluxSmoothLen"]
        ).mean()
    else:
        expected_raw = mod_flux_base
    expected_cut = indicators._apply_flux_deadzone(expected_raw, params["fluxDeadzone"])
    expected_gate = expected_cut.fillna(0.0)

    pdt.assert_series_equal(
        flux_hist, expected_cut, check_names=False, check_exact=False, rtol=1e-12, atol=1e-12
    )
    pdt.assert_series_equal(
        flux_gate, expected_gate, check_names=False, check_exact=False, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("use_flux_heikin", [False, True])
def test_compute_indicators_with_directional_flux_applies_additional_smoothing(use_flux_heikin):
    index = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    base = np.linspace(120.0, 135.0, num=len(index))
    df = pd.DataFrame(
        {
            "open": base + np.random.default_rng(11).normal(0, 0.4, size=len(index)),
            "high": base + 1.2,
            "low": base - 0.8,
            "close": base + np.random.default_rng(7).normal(0, 0.25, size=len(index)),
            "volume": np.linspace(1.0, 3.0, num=len(index)),
        },
        index=index,
    )

    params = {
        "oscLen": 12,
        "signalLen": 4,
        "fluxLen": 9,
        "fluxSmoothLen": 4,
        "useFluxHeikin": use_flux_heikin,
        "useModFlux": False,
        "fluxDeadzone": 25.0,
        "kcLen": 15,
        "kcMult": 1.1,
        "bbLen": 18,
        "bbMult": 1.3,
        "maType": "EMA",
        "basisStyle": "Deluxe",
        "compatMode": True,
        "autoThresholdScale": True,
        "useNormClip": False,
        "normClipLimit": 350.0,
    }

    (
        _momentum,
        _signal,
        _cross_up,
        _cross_down,
        flux_hist,
        flux_gate,
        _threshold_scale,
    ) = alt._compute_indicators(df, params)

    source_df = indicators._heikin_ashi(df) if use_flux_heikin else df
    flux_base = indicators._raw_directional_flux(source_df, params["fluxLen"])
    if params["fluxSmoothLen"] > 1:
        expected_raw = flux_base.rolling(
            params["fluxSmoothLen"], min_periods=params["fluxSmoothLen"]
        ).mean()
    else:
        expected_raw = flux_base
    expected_cut = indicators._apply_flux_deadzone(expected_raw, params["fluxDeadzone"])
    expected_gate = expected_cut.fillna(0.0)

    pdt.assert_series_equal(
        flux_hist, expected_cut, check_names=False, check_exact=False, rtol=1e-12, atol=1e-12
    )
    pdt.assert_series_equal(
        flux_gate, expected_gate, check_names=False, check_exact=False, rtol=1e-12, atol=1e-12
    )


def test_compute_indicators_handles_zero_atr_norm():
    index = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(len(index), 100.0),
            "high": np.full(len(index), 100.0),
            "low": np.full(len(index), 100.0),
            "close": np.full(len(index), 100.0),
            "volume": np.ones(len(index)),
        },
        index=index,
    )

    params = {
        "oscLen": 5,
        "signalLen": 3,
        "fluxLen": 3,
        "fluxSmoothLen": 1,
        "useFluxHeikin": False,
        "useModFlux": False,
        "kcLen": 5,
        "kcMult": 1.0,
        "bbLen": 5,
        "bbMult": 1.0,
        "maType": "SMA",
        "basisStyle": "avg",
        "compatMode": True,
        "autoThresholdScale": True,
        "fluxDeadzone": 25.0,
        "useNormClip": False,
        "normClipLimit": 350.0,
    }

    momentum, mom_signal, *_ = alt._compute_indicators(df, params)

    assert np.isfinite(momentum.dropna()).all()
    assert np.isfinite(mom_signal.dropna()).all()
    assert (momentum.dropna() == 0).all()
    assert (mom_signal.dropna() == 0).all()


def test_resolve_thresholds_uses_fallback_for_zero_std():
    index = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    momentum = pd.Series(0.0, index=index)
    params = {
        "useDynamicThresh": True,
        "useSymThreshold": True,
        "statThreshold": 15.0,
        "dynLen": 4,
        "dynMult": 1.0,
        "buyThreshold": 10.0,
        "sellThreshold": 10.0,
        "compatMode": True,
        "autoThresholdScale": True,
    }

    scale = pd.Series(1.0, index=index)
    buy, sell = alt._resolve_thresholds(momentum, params, scale)

    assert (buy == -15.0).all()
    assert (sell == 15.0).all()


def test_apply_exit_overrides_chandelier_stop_triggers_exit():
    index = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 103.0, 101.0],
            "high": [101.0, 103.0, 104.0, 102.0],
            "low": [99.0, 101.0, 102.0, 90.0],
            "close": [100.5, 102.5, 103.5, 91.0],
            "volume": 1.0,
        },
        index=index,
    )
    parsed = alt._ParsedInputs(
        df=df,
        htf_df=None,
        start_ts=index[0],
        commission_pct=0.0,
        slippage_ticks=0.0,
        leverage=1.0,
        initial_capital=1000.0,
        capital_pct=1.0,
        allow_long=True,
        allow_short=False,
        require_cross=False,
        exit_opposite=False,
        min_trades=0,
        min_hold_bars=0,
        max_consecutive_losses=10,
    )

    long_entries = pd.Series([True, False, False, False], index=index)
    long_exits = pd.Series(False, index=index)
    short_entries = pd.Series(False, index=index)
    short_exits = pd.Series(False, index=index)

    params = {
        "useStopLoss": False,
        "useAtrTrail": False,
        "useMomFade": False,
        "useChandelierExit": True,
        "chandelierLen": 1,
        "chandelierMult": 0.5,
    }

    _, adjusted_long_exits, _, _ = alt._apply_exit_overrides(
        parsed, params, long_entries, long_exits, short_entries, short_exits
    )

    assert adjusted_long_exits.any()


def test_apply_exit_overrides_parabolic_sar_triggers_exit():
    index = pd.date_range("2025-02-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 99.0, 98.0, 97.0, 98.0],
            "high": [101.0, 100.0, 99.0, 98.0, 120.0],
            "low": [99.0, 98.0, 97.0, 96.0, 97.0],
            "close": [100.0, 99.0, 98.0, 97.0, 119.0],
            "volume": 1.0,
        },
        index=index,
    )
    parsed = alt._ParsedInputs(
        df=df,
        htf_df=None,
        start_ts=index[0],
        commission_pct=0.0,
        slippage_ticks=0.0,
        leverage=1.0,
        initial_capital=1000.0,
        capital_pct=1.0,
        allow_long=False,
        allow_short=True,
        require_cross=False,
        exit_opposite=False,
        min_trades=0,
        min_hold_bars=0,
        max_consecutive_losses=10,
    )

    long_entries = pd.Series(False, index=index)
    long_exits = pd.Series(False, index=index)
    short_entries = pd.Series([True, False, False, False, False], index=index)
    short_exits = pd.Series(False, index=index)

    params = {
        "useStopLoss": False,
        "useAtrTrail": False,
        "useMomFade": False,
        "useSarExit": True,
        "sarStart": 0.05,
        "sarIncrement": 0.05,
        "sarMaximum": 0.2,
    }

    _, _, _, adjusted_short_exits = alt._apply_exit_overrides(
        parsed, params, long_entries, long_exits, short_entries, short_exits
    )

    assert adjusted_short_exits.any()
