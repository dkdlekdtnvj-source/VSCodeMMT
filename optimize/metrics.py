"""Performance metric calculations for optimisation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd


EPS = 1e-12
LOGGER = logging.getLogger(__name__)

_MAX_EXP = 700.0  # np.exp(709) ~= 8.2e307 -> float64 상한 근처에서 여유 확보


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """과도한 값으로 인한 오버플로우를 방지하며 exp를 계산합니다."""

    return np.exp(np.clip(x, -_MAX_EXP, _MAX_EXP))

if TYPE_CHECKING:  # pragma: no cover - 순환 참조 방지용 힌트
    from .state import EquityState


@dataclass
class Trade:
    """Container describing the outcome of a single trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    size: float
    entry_price: float
    exit_price: float
    profit: float
    return_pct: float
    mfe: float
    mae: float
    bars_held: int
    reason: str = ""


@dataclass(frozen=True)
class ObjectiveSpec:
    """Normalised representation of an optimisation objective."""

    name: str
    weight: float = 1.0
    goal: str = "maximize"

    @property
    def direction(self) -> str:
        goal = str(self.goal).lower()
        if goal in {"minimise", "minimize", "min", "lower"}:
            return "minimize"
        return "maximize"

    @property
    def is_minimize(self) -> bool:
        return self.direction == "minimize"


def _normalise_returns_array(
    returns: pd.Series, *, max_clip: float = 5.0
) -> Tuple[pd.Index, np.ndarray]:
    """일관된 float64 배열과 인덱스를 반환하며, 스케일 오인을 자동 교정합니다."""

    if max_clip <= 0:
        max_clip = 5.0

    cleaned = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    array = cleaned.to_numpy(dtype=np.float64, copy=True)
    if array.size == 0:
        return cleaned.index, array

    with np.errstate(invalid="ignore"):
        max_abs = float(np.nanmax(np.abs(array))) if np.size(array) else 0.0

    # returns_series 는 초기 자본 대비 소수 비율(1.0 == 100%)을 기본 전제로 사용한다.
    # 다만 레거시 퍼센트 단위(예: 35 == 35%)가 전달되는 경우를 감지하기 위해 분포를
    # 점검한다. 절대값 최댓값이 20 이상인 경우(전형적인 퍼센트 단위)나, 절대값 중앙값
    # 자체가 1 이상인 경우(값의 절반 이상이 ±100%를 넘는 경우), 혹은 값의 절반 이상이
    # 절대값 1보다 큰 경우에는 퍼센트 단위로 간주하고 100으로 나누어 소수 비율로 환산한다.
    if np.isfinite(max_abs) and max_abs >= 1.0:
        valid = array[np.isfinite(array)]
        if valid.size:
            abs_valid = np.abs(valid)
            median_abs = float(np.nanmedian(abs_valid))
            fraction_gt_one = float(np.mean(abs_valid > 1.0))
        else:  # pragma: no cover - 비어 있는 경우는 앞선 size 체크로 반환됨
            median_abs = 0.0
            fraction_gt_one = 0.0

        needs_scaling = False
        if max_abs >= 20.0:
            needs_scaling = True
        elif median_abs >= 1.0:
            needs_scaling = True
        elif fraction_gt_one >= 0.5:
            needs_scaling = True

        if needs_scaling:
            array = array / 100.0

    lower_bound = -0.95
    upper_bound = max(0.0, float(max_clip))
    array = np.clip(array, lower_bound, upper_bound)
    return cleaned.index, array


def equity_curve_from_returns(
    returns: pd.Series,
    initial: float = 1.0,
    *,
    max_clip: float = 5.0,
    use_direct_product: bool = False,
) -> pd.Series:
    """로그 누적 혹은 직접 누적곱을 이용해 안정적인 자본곡선을 계산합니다."""

    index, values = _normalise_returns_array(returns, max_clip=max_clip)
    if values.size == 0:
        return pd.Series([], index=index, dtype=np.float64)

    initial_float = float(initial)
    if use_direct_product:
        equity_values = equity_from_returns(initial_float, values)
    else:
        log_returns = np.log1p(values)
        log_equity = np.cumsum(log_returns, dtype=np.float64)
        log_equity_clipped = np.clip(log_equity, -_MAX_EXP, _MAX_EXP)
        equity_values = _safe_exp(log_equity_clipped) * initial_float
    return pd.Series(equity_values, index=index, dtype=np.float64)


def equity_from_returns(initial: float, returns: np.ndarray) -> np.ndarray:
    """`(1 + r)`의 누적곱을 사용해 직접 자본곡선을 계산합니다."""

    if returns.size == 0:
        return np.array([], dtype=np.float64)
    r = np.clip(returns, -0.99, 10.0)
    cumulative = np.cumprod(1.0 + r, dtype=np.float64)
    return float(initial) * cumulative


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """자본曲선으로부터 드로우다운 시퀀스를 계산합니다."""

    if equity.empty:
        return pd.Series(dtype=np.float64)

    cleaned = equity.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    values = cleaned.to_numpy(dtype=np.float64, copy=True)
    peaks = np.maximum.accumulate(np.maximum(values, EPS))
    drawdowns = (values - peaks) / peaks
    return pd.Series(drawdowns, index=cleaned.index, dtype=np.float64)


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown as a negative percentage."""

    if equity.empty:
        return 0.0
    dd_series = drawdown_curve(equity)
    return float(dd_series.min()) if not dd_series.empty else 0.0


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return 0.0
    with np.errstate(invalid="ignore"):
        std = cleaned.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((cleaned.mean() - risk_free) / std)


def win_rate(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if trade.profit > 0)
    return wins / len(trades)


def average_rr(trades: Sequence[Trade]) -> float:
    rs = [trade.mfe / abs(trade.mae) for trade in trades if trade.mae < 0]
    return float(np.mean(rs)) if rs else 0.0


def average_hold_time(trades: Sequence[Trade]) -> float:
    holds = [trade.bars_held for trade in trades]
    return float(np.mean(holds)) if holds else 0.0


def _consecutive_losses(trades: Sequence[Trade]) -> int:
    streak = 0
    worst = 0
    for trade in trades:
        if trade.profit < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def _weekly_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    weekly = returns.resample("W").sum()
    return weekly.dropna()


def aggregate_metrics(
    trades: List[Trade], returns: pd.Series, *, simple: bool = False
) -> Dict[str, float]:
    """Aggregate trade-level information into rich performance metrics."""

    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = equity_curve_from_returns(returns, initial=1.0)
    net_profit = float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]) if len(equity) > 1 else 0.0

    gross_profit = float(sum(max(trade.profit, 0.0) for trade in trades))
    gross_loss = float(sum(min(trade.profit, 0.0) for trade in trades))
    wins = sum(1 for trade in trades if trade.profit > 0)
    losses = sum(1 for trade in trades if trade.profit < 0)

    band_exit_bb_count = sum(1 for trade in trades if getattr(trade, "reason", "") == "band_exit_bb")
    band_exit_kc_count = sum(1 for trade in trades if getattr(trade, "reason", "") == "band_exit_kc")
    band_exit_count = band_exit_bb_count + band_exit_kc_count

    if simple:
        metrics: Dict[str, float] = {
            "NetProfit": net_profit,
            "TotalReturn": net_profit,
            "Trades": float(len(trades)),
            "Wins": float(wins),
            "Losses": float(losses),
            "GrossProfit": gross_profit,
            "GrossLoss": gross_loss,
            "AvgHoldBars": float(average_hold_time(trades)),
            "MaxConsecutiveLosses": float(_consecutive_losses(trades)),
            "WinRate": float(win_rate(trades)),
        }
        metrics["BandExitCount"] = float(band_exit_count)
        metrics["BandExitBBCount"] = float(band_exit_bb_count)
        metrics["BandExitKCCount"] = float(band_exit_kc_count)
        if trades:
            trade_count = float(len(trades))
            metrics["BandExitRatio"] = float(band_exit_count / trade_count)
        else:
            metrics["BandExitRatio"] = 0.0
        metrics["DrawdownPct"] = 0.0
        return metrics

    weekly = _weekly_returns(returns)
    weekly_mean = float(weekly.mean()) if not weekly.empty else 0.0
    weekly_std = float(weekly.std(ddof=0)) if len(weekly) > 1 else 0.0

    metrics: Dict[str, float] = {
        "NetProfit": net_profit,
        "TotalReturn": net_profit,
        "MaxDD": float(max_drawdown(equity)),
        "WinRate": float(win_rate(trades)),
        "Sharpe": float(sharpe_ratio(returns)),
        "AvgRR": float(average_rr(trades)),
        "AvgHoldBars": float(average_hold_time(trades)),
        "Trades": float(len(trades)),
        "Wins": float(wins),
        "Losses": float(losses),
        "GrossProfit": gross_profit,
        "GrossLoss": gross_loss,
        "Expectancy": float((gross_profit + gross_loss) / len(trades)) if trades else 0.0,
        "WeeklyNetProfit": weekly_mean,
        "WeeklyReturnStd": weekly_std,
        "MaxConsecutiveLosses": float(_consecutive_losses(trades)),
    }
    metrics["BandExitCount"] = float(band_exit_count)
    metrics["BandExitBBCount"] = float(band_exit_bb_count)
    metrics["BandExitKCCount"] = float(band_exit_kc_count)
    if trades:
        trade_count = float(len(trades))
        metrics["BandExitRatio"] = float(band_exit_count / trade_count)
    else:
        metrics["BandExitRatio"] = 0.0
    max_dd_value = metrics.get("MaxDD")
    if max_dd_value is not None:
        dd_abs = abs(float(max_dd_value))
        metrics["DrawdownPct"] = dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs
    else:
        metrics["DrawdownPct"] = 0.0

    mfe = [trade.mfe for trade in trades]
    mae = [trade.mae for trade in trades]
    metrics["AvgMFE"] = float(np.mean(mfe)) if mfe else 0.0
    metrics["AvgMAE"] = float(np.mean(mae)) if mae else 0.0
    return metrics


def normalise_objectives(objectives: Iterable[object]) -> List[ObjectiveSpec]:
    """Coerce raw objective declarations into :class:`ObjectiveSpec` entries."""

    specs: List[ObjectiveSpec] = []
    for obj in objectives:
        if isinstance(obj, ObjectiveSpec):
            specs.append(obj)
            continue
        if isinstance(obj, str):
            specs.append(ObjectiveSpec(name=obj))
            continue
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("metric")
            if not name:
                continue
            weight = float(obj.get("weight", 1.0))
            if "minimize" in obj:
                goal = "minimize" if bool(obj.get("minimize")) else "maximize"
            elif "maximize" in obj:
                goal = "maximize" if bool(obj.get("maximize")) else "minimize"
            else:
                goal_raw = obj.get("goal") or obj.get("direction") or obj.get("target")
                if goal_raw is None:
                    goal = "maximize"
                else:
                    goal_text = str(goal_raw).lower()
                    if goal_text in {"min", "minimise", "minimize", "lower"}:
                        goal = "minimize"
                    elif goal_text in {"max", "maximise", "maximize", "higher"}:
                        goal = "maximize"
                    else:
                        goal = "maximize"
            specs.append(ObjectiveSpec(name=str(name), weight=weight, goal=goal))
    return specs


def _objective_iterator(objectives: Iterable[object]) -> Iterable[ObjectiveSpec]:
    for spec in normalise_objectives(objectives):
        yield spec


def evaluate_objective_values(
    metrics: Dict[str, float],
    objectives: Sequence[ObjectiveSpec],
    non_finite_penalty: float,
) -> Tuple[float, ...]:
    """Transform metric dict into ordered objective values respecting directions."""

    penalty = abs(float(non_finite_penalty))

    drawdown_pct_value = metrics.get("DrawdownPct")
    if drawdown_pct_value is not None:
        try:
            dd_pct = float(drawdown_pct_value)
        except (TypeError, ValueError):
            dd_pct = 0.0
    else:
        dd_raw = metrics.get("MaxDD", metrics.get("MaxDrawdown"))
        try:
            dd_value = float(dd_raw) if dd_raw is not None else 0.0
        except (TypeError, ValueError):
            dd_value = 0.0
        dd_abs = abs(dd_value)
        dd_pct = dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs
    dd_ratio = min(max(dd_pct, 0.0), 100.0) / 100.0

    values: List[float] = []
    for spec in objectives:
        raw = metrics.get(spec.name)
        # Convert the raw metric to a numeric value.  Certain sentinel
        # strings (e.g. "overfactor" or the ProfitFactor check label) are
        # intentionally ignored during objective evaluation and treated as
        # neutral values.  If the value cannot be coerced to float and is
        # not one of these sentinel values, ``nan`` is used to trigger the
        # non-finite penalty.
        try:
            # If raw is a string sentinel, handle separately
            if isinstance(raw, str):
                raw_str = raw.strip().lower()
                if raw_str in {"overfactor", "체크 필요"}:
                    numeric = 0.0
                else:
                    numeric = float(raw)
            else:
                numeric = float(raw)
        except Exception:
            numeric = float("nan")

        name_lower = spec.name.lower()
        if name_lower in {"maxdd", "maxdrawdown"}:
            numeric = abs(numeric) if spec.is_minimize else -abs(numeric)

        if not np.isfinite(numeric):
            weight = abs(float(spec.weight))
            if weight == 0:
                numeric = 0.0
            else:
                base = penalty if spec.is_minimize else -penalty
                numeric = base * weight
        else:
            numeric *= float(spec.weight)

        values.append(numeric)

    if dd_ratio > 0.0:
        adjusted: List[float] = []
        for numeric in values:
            if numeric >= 0.0:
                adjusted.append(numeric * max(0.0, 1.0 - dd_ratio))
            else:
                adjusted.append(numeric * (1.0 + dd_ratio))
        values = adjusted

    return tuple(values)


def score_metrics(metrics: Dict[str, float], objectives: Iterable[object]) -> float:
    """Score a metric dictionary according to weighted objectives and penalties."""

    score = 0.0
    for spec in _objective_iterator(objectives):
        value = metrics.get(spec.name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        name_lower = spec.name.lower()
        if name_lower in {"maxdd", "maxdrawdown"}:
            contribution = -abs(numeric)
        elif spec.is_minimize:
            contribution = -numeric
        else:
            contribution = numeric
        score += float(spec.weight) * contribution

    trades = float(metrics.get("Trades", 0))
    min_trades = metrics.get("MinTrades")
    if min_trades is not None and trades < float(min_trades):
        penalty = float(metrics.get("TradePenalty", 1.0))
        score -= (float(min_trades) - trades) * penalty

    avg_hold = float(metrics.get("AvgHoldBars", 0.0))
    min_hold = metrics.get("MinHoldBars")
    if min_hold is not None and avg_hold < float(min_hold):
        penalty = float(metrics.get("HoldPenalty", 1.0))
        score -= (float(min_hold) - avg_hold) * penalty

    losses = float(metrics.get("MaxConsecutiveLosses", 0.0))
    loss_cap = metrics.get("MaxConsecutiveLossLimit")
    if loss_cap is not None and losses > float(loss_cap):
        penalty = float(metrics.get("ConsecutiveLossPenalty", 1.0))
        score -= (losses - float(loss_cap)) * penalty

    drawdown_pct_value = metrics.get("DrawdownPct")
    if drawdown_pct_value is not None:
        try:
            dd_pct = float(drawdown_pct_value)
        except (TypeError, ValueError):
            dd_pct = 0.0
    else:
        dd_raw = metrics.get("MaxDD", metrics.get("MaxDrawdown"))
        try:
            dd_value = float(dd_raw) if dd_raw is not None else 0.0
        except (TypeError, ValueError):
            dd_value = 0.0
        dd_abs = abs(dd_value)
        dd_pct = dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs
    dd_ratio = min(max(dd_pct, 0.0), 100.0) / 100.0
    if dd_ratio > 0.0:
        if score >= 0.0:
            score *= max(0.0, 1.0 - dd_ratio)
        else:
            score *= 1.0 + dd_ratio

    return float(score)


def finalise_metrics_result(
    state: "EquityState",
    trades: List[Trade],
    returns_series: pd.Series,
    *,
    guard_flag: bool,
    simple_metrics_only: bool,
    penalty_config: Dict[str, float],
    validity_thresholds: Dict[str, float],
    ruin_hit: bool,
    leverage: float,
) -> Dict[str, float]:
    """최종 백테스트 결과 메트릭을 집계하고 보조 지표를 부여합니다."""

    metrics = aggregate_metrics(trades, returns_series, simple=simple_metrics_only)

    initial_capital_value = float(state.initial_capital)
    metrics["InitialCapital"] = initial_capital_value
    metrics.setdefault("InitialEquity", initial_capital_value)
    metrics.setdefault("InitialBalance", initial_capital_value)
    metrics.setdefault("StartingBalance", initial_capital_value)

    if simple_metrics_only:
        metrics["SimpleMetricsOnly"] = True

    metrics["FinalEquity"] = state.equity
    metrics["NetProfitAbs"] = state.net_profit
    metrics["GuardFrozen"] = float(guard_flag)
    metrics["TradesList"] = trades
    metrics["Returns"] = returns_series
    metrics["Withdrawable"] = state.withdrawable

    min_trades_value = float(max(0.0, penalty_config.get("min_trades_value", 0.0)))
    min_hold_value = float(max(0.0, penalty_config.get("min_hold_value", 0.0)))
    max_loss_streak = float(max(0.0, penalty_config.get("max_loss_streak", 0.0)))

    metrics["MinTrades"] = min_trades_value
    metrics["MinHoldBars"] = min_hold_value
    metrics["MaxConsecutiveLossLimit"] = max_loss_streak

    metrics["TradePenalty"] = float(max(0.0, penalty_config.get("trade_penalty", 0.0)))
    metrics["HoldPenalty"] = float(max(0.0, penalty_config.get("hold_penalty", 0.0)))
    metrics["ConsecutiveLossPenalty"] = float(
        max(0.0, penalty_config.get("consecutive_loss_penalty", 0.0))
    )

    min_trades_req = float(validity_thresholds.get("min_trades", 0.0))
    min_hold_req = float(validity_thresholds.get("min_hold_bars", 0.0))
    max_consecutive_allowed = float(validity_thresholds.get("max_consecutive_losses", float("inf")))

    metrics["Valid"] = (
        metrics.get("Trades", 0.0) >= min_trades_req
        and metrics.get("AvgHoldBars", 0.0) >= min_hold_req
        and metrics.get("MaxConsecutiveLosses", 0.0) <= max_consecutive_allowed
    )

    drawdown_limit_pct = float(abs(validity_thresholds.get("max_drawdown_pct", float("inf"))))
    if np.isfinite(drawdown_limit_pct):
        dd_raw = metrics.get("MaxDD", metrics.get("MaxDrawdown"))
        try:
            dd_value = float(dd_raw)
        except (TypeError, ValueError, OverflowError):
            dd_value = np.nan
        if np.isfinite(dd_value):
            dd_abs = abs(dd_value)
            dd_pct = dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs
            if dd_pct >= drawdown_limit_pct:
                metrics["Valid"] = False
                metrics["DrawdownFiltered"] = True

    total_assets = state.total_assets()
    metrics["TotalAssets"] = total_assets
    metrics["Liquidations"] = float(state.liquidations)
    metrics["AvailableCapital"] = state.available_capital
    metrics["Savings"] = state.savings if not state.use_wallet else 0.0
    metrics["LockedCapital"] = state.locked_capital
    metrics["Ruin"] = float(ruin_hit)
    metrics["Leverage"] = float(leverage)

    if "DrawdownPct" not in metrics or not np.isfinite(metrics.get("DrawdownPct")):
        dd_raw = metrics.get("MaxDD", metrics.get("MaxDrawdown"))
        try:
            dd_value = float(dd_raw) if dd_raw is not None else 0.0
        except (TypeError, ValueError):
            dd_value = 0.0
        dd_abs = abs(dd_value)
        metrics["DrawdownPct"] = dd_abs * 100.0 if dd_abs <= 1.0 else dd_abs

    return metrics


__all__ = [
    "Trade",
    "ObjectiveSpec",
    "evaluate_objective_values",
    "aggregate_metrics",
    "equity_curve_from_returns",
    "max_drawdown",
    "normalise_objectives",
    "finalise_metrics_result",
    "score_metrics",
]
