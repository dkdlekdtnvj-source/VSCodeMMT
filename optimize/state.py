"""백테스트 상태 추적용 데이터 클래스 모음."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    direction: int = 0
    qty: float = 0.0
    avg_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_held: int = 0
    highest: float = np.nan
    lowest: float = np.nan
    capital_used: float = 0.0
    base_qty: float = 0.0
    base_capital_used: float = 0.0
    pyramid_adds: int = 0


@dataclass
class EquityState:
    initial_capital: float
    equity: float
    net_profit: float = 0.0
    withdrawable: float = 0.0
    tradable_capital: float = 0.0
    peak_equity: float = 0.0
    daily_start_capital: float = 0.0
    daily_peak_capital: float = 0.0
    week_start_equity: float = 0.0
    week_peak_equity: float = 0.0
    # 추가: 사용 가능한 자본과 저축 자본. 각 거래는 available_capital의 일정 비율만 사용하고,
    # 수익금 일부는 savings 로 따로 적립합니다. 청산 시 savings 의 절반을 available_capital 로 이동합니다.
    available_capital: float = 0.0
    savings: float = 0.0
    # 청산 횟수 기록용
    liquidations: int = 0
    locked_capital: float = 0.0
    use_wallet: bool = False
    reserve_for_sizing: bool = False

    def release_locked_capital(self, amount: float) -> None:
        """포지션 청산 시 잠겨 있던 증거금을 가용 자본으로 되돌립니다."""

        if amount <= 0.0:
            return
        self.available_capital += amount
        self.locked_capital = max(self.locked_capital - amount, 0.0)

    def apply_pnl(self, pnl: float) -> None:
        """손익을 가용 자본에 반영합니다."""

        self.available_capital += pnl

    def _equity_base(self) -> float:
        total = self.available_capital + self.locked_capital
        if not self.use_wallet:
            total += self.savings
        return total

    def update_tradable_capital(self) -> None:
        """현재 상태를 기준으로 재거래 가능 자본을 갱신합니다."""

        total = self._equity_base()
        reserve = max(self.withdrawable, 0.0) if (self.use_wallet and self.reserve_for_sizing) else 0.0
        effective = max(total - reserve, 0.0)
        self.tradable_capital = max(effective, self.initial_capital * 0.01)
        self.equity = self._equity_base()
        if self.peak_equity < self.equity:
            self.peak_equity = self.equity

    def handle_trade_settlement(
        self,
        pnl: float,
        *,
        deposit_pct: float,
        reason: str,
        ruin_threshold: float,
        drawdown_ruin_pct: float,
    ) -> bool:
        """수익 분배, 저축 이동, 파산 임계값 확인까지 한 번에 처리합니다.

        반환값은 전체 자산이 ``ruin_threshold`` 미만인지 여부입니다.
        """

        if pnl > 0.0 and deposit_pct > 0.0:
            deposit_amt = pnl * deposit_pct
            if deposit_amt > 0.0:
                if self.use_wallet:
                    self.withdrawable += deposit_amt
                else:
                    deposit_amt = min(deposit_amt, self.available_capital)
                    self.available_capital -= deposit_amt
                    self.savings += deposit_amt

        if reason == "Liquidation":
            withdraw = self.savings * 0.5
            if withdraw > 0.0:
                self.available_capital += withdraw
                self.savings -= withdraw
            self.liquidations += 1

        self.update_tradable_capital()

        current_equity = self.equity
        if current_equity < ruin_threshold:
            return True

        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - current_equity) / self.peak_equity * 100.0
            if dd_pct >= max(drawdown_ruin_pct, 0.0):
                return True

        return False

    def total_assets(self) -> float:
        """가용 자본, 저축, 잠긴 자본을 모두 더한 총 자산을 반환합니다."""

        return self._equity_base()


@dataclass(frozen=True)
class FilterSettings:
    use_volatility_guard: bool
    volatility_lookback: int
    use_adx: bool
    use_atr_diff: bool
    adx_atr_tf: str
    adx_len: int
    use_ema: bool
    ema_fast_len: int
    ema_slow_len: int
    use_bb_filter: bool
    bb_filter_len: int
    bb_filter_mult: float
    use_stoch_rsi: bool
    stoch_len: int
    use_obv: bool
    obv_smooth_len: int
    use_htf_trend: bool
    htf_trend_tf: str
    htf_ma_len: int
    use_hma_filter: bool
    hma_len: int
    use_range_filter: bool
    range_tf: str
    range_bars: int
    range_percent: float
    use_event_filter: bool
    event_windows: str
    use_slope_filter: bool
    slope_lookback: int
    slope_min_pct: float
    use_distance_guard: bool
    distance_atr_len: int
    distance_trend_len: int
    distance_max_atr: float
    use_kasa: bool
    kasa_rsi_len: int
    use_regime_filter: bool
    ctx_htf_tf: str
    ctx_htf_ema_len: int
    ctx_htf_adx_len: int
    ctx_htf_adx_th: float
    use_structure_gate: bool
    use_bos: bool
    use_choch: bool
    bos_tf: str
    bos_lookback: int
    bos_state_bars: int
    choch_state_bars: int
    pivot_left: int
    pivot_right: int
    use_shock: bool
    atr_fast_len: int
    atr_slow_len: int
    shock_mult: float


@dataclass
class FilterContext:
    vol_guard_atr_pct: pd.Series
    adx_series: pd.Series
    atr_diff: pd.Series
    ema_fast: pd.Series
    ema_slow: pd.Series
    bb_filter_basis: pd.Series
    bb_filter_upper: pd.Series
    bb_filter_lower: pd.Series
    stoch_rsi: pd.Series
    obv_slope: pd.Series
    htf_trend_up: pd.Series
    htf_trend_down: pd.Series
    hma_value: pd.Series
    in_range_box: pd.Series
    event_mask: pd.Series
    slope_ok_long: pd.Series
    slope_ok_short: pd.Series
    distance_ok: pd.Series
    kasa_rsi: pd.Series
    regime_long_ok: pd.Series
    regime_short_ok: pd.Series
    bos_long_state: pd.Series
    bos_short_state: pd.Series
    choch_long_state: pd.Series
    choch_short_state: pd.Series
    shock_series: pd.Series
