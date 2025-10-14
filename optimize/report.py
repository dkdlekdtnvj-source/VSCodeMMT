"""Report generation utilities for optimisation runs."""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

TRIAL_STATE_LABELS = {
    "COMPLETE": "완료",
    "PRUNED": "중단",
    "FAIL": "실패",
    "RUNNING": "실행중",
    "WAITING": "대기",
}


def _normalise_state_label(value: object) -> str:
    """TrialState 문자열/객체를 한국어 라벨로 변환합니다."""

    if value in {None, ""}:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text in TRIAL_STATE_LABELS.values():
            return text
        key = text.split(".")[-1]
    else:
        key = str(value).split(".")[-1]
    key_upper = key.upper()
    if not key_upper:
        return ""
    return TRIAL_STATE_LABELS.get(key_upper, key_upper)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


from optimize.metrics import normalise_objectives


LOGGER = logging.getLogger(__name__)

_SEABORN: Optional[object] = None
_SEABORN_IMPORT_ERROR: Optional[Exception] = None


def _get_seaborn():
    """Lazy seaborn importer to avoid hard dependency at module import."""

    global _SEABORN, _SEABORN_IMPORT_ERROR
    if _SEABORN is not None:
        return _SEABORN
    if _SEABORN_IMPORT_ERROR is not None:
        raise ImportError("seaborn import previously failed") from _SEABORN_IMPORT_ERROR
    try:
        import seaborn as sns  # type: ignore
    except Exception as exc:  # pragma: no cover - 환경 의존
        _SEABORN_IMPORT_ERROR = exc
        raise ImportError("seaborn is required for heatmap export") from exc
    _SEABORN = sns
    return sns


def _summarise_distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _collect_trade_profits(best: Dict[str, object]) -> List[float]:
    profits: List[float] = []
    seen: set[Tuple[Optional[float], Optional[float], Optional[int]]] = set()

    def _extract_from_container(container: object) -> None:
        if not isinstance(container, dict):
            return
        trades_payload = container.get("TradesList") or container.get("trades")
        if not isinstance(trades_payload, list):
            return
        for entry in trades_payload:
            candidate = None
            return_value: Optional[float] = None
            bars_candidate: Optional[int] = None
            if isinstance(entry, dict):
                candidate = entry.get("profit")
                raw_return = entry.get("return_pct")
                try:
                    return_value = float(raw_return) if raw_return is not None else None
                except (TypeError, ValueError):
                    return_value = None
                bars_value = entry.get("bars_held")
                try:
                    bars_candidate = int(bars_value) if bars_value is not None else None
                except (TypeError, ValueError):
                    bars_candidate = None
            elif isinstance(entry, (int, float)):
                candidate = entry
            if candidate is None:
                continue
            try:
                numeric = float(candidate)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                key = (numeric, return_value, bars_candidate)
                if key in seen:
                    continue
                seen.add(key)
                profits.append(numeric)

    _extract_from_container(best.get("metrics"))
    datasets = best.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            _extract_from_container(dataset.get("metrics"))
            _extract_from_container(dataset)

    return profits


def _run_trade_monte_carlo(
    profits: Sequence[float],
    *,
    iterations: int = 500,
    drop_ratio: float = 0.1,
) -> Optional[Dict[str, object]]:
    if not profits:
        return None

    array = np.asarray([p for p in profits if np.isfinite(p)], dtype=float)
    if array.size == 0:
        return None

    drop_ratio = max(0.0, min(float(drop_ratio), 0.9))
    rng = np.random.default_rng()
    net_results: List[float] = []
    drawdowns: List[float] = []
    win_rates: List[float] = []
    sample_lengths: List[int] = []

    for _ in range(int(iterations)):
        sample = np.array(array, copy=True)
        rng.shuffle(sample)
        drop_count = int(round(sample.size * drop_ratio))
        if drop_count > 0 and drop_count < sample.size:
            mask = np.ones(sample.size, dtype=bool)
            mask[rng.choice(sample.size, size=drop_count, replace=False)] = False
            sample = sample[mask]
        sample_lengths.append(int(sample.size))
        if sample.size == 0:
            net_results.append(0.0)
            drawdowns.append(0.0)
            win_rates.append(0.0)
            continue
        wins = float(np.sum(sample > 0) / sample.size)
        win_rates.append(wins)
        equity = np.concatenate([[0.0], np.cumsum(sample)])
        net_results.append(float(equity[-1]))
        running_max = np.maximum.accumulate(equity)
        dd_curve = equity - running_max
        drawdowns.append(float(dd_curve.min()) if dd_curve.size else 0.0)

    summary = {
        "iterations": int(iterations),
        "sample_size": int(array.size),
        "drop_ratio": float(drop_ratio),
        "avg_sample_length": float(np.mean(sample_lengths)) if sample_lengths else 0.0,
        "net_profit": _summarise_distribution(net_results),
        "max_drawdown": _summarise_distribution(drawdowns),
        "win_rate": _summarise_distribution(win_rates),
    }
    return summary


def _objective_iterator(objectives: Iterable[object]) -> Iterable[Tuple[str, float]]:
    for spec in normalise_objectives(objectives):
        yield spec.name, float(spec.weight)


def _flatten_results(results: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregated_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []

    for record in results:
        base_row: Dict[str, object] = {
            "trial": record.get("trial"),
            "score": record.get("score"),
            "valid": record.get("valid", True),
        }
        base_row.update(record.get("params", {}))
        for key, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float, bool)):
                base_row[key] = value
        aggregated_rows.append(base_row)

        for dataset in record.get("datasets", []):
            ds_row: Dict[str, object] = {
                "trial": record.get("trial"),
                "score": record.get("score"),
                "valid": dataset.get("metrics", {}).get("Valid", True),
                "dataset": dataset.get("name"),
            }
            ds_row.update(dataset.get("meta", {}))
            ds_row.update(record.get("params", {}))
            for key, value in dataset.get("metrics", {}).items():
                if isinstance(value, (int, float, bool)):
                    ds_row[key] = value
            dataset_rows.append(ds_row)

    return pd.DataFrame(aggregated_rows), pd.DataFrame(dataset_rows)


def _annotate_objectives(df: pd.DataFrame, objectives: Iterable[object]) -> pd.DataFrame:
    if df.empty:
        return df

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    invalid_tokens = {"overfactor", "체크 필요", "체크필요", "check", "needs review"}
    for name, weight in _objective_iterator(objectives):
        if name not in df.columns:
            continue
        series_raw = df[name]
        if isinstance(series_raw, pd.Series):
            cleaned = series_raw.copy()
            cleaned = cleaned.apply(
                lambda value: (
                    np.nan
                    if isinstance(value, str)
                    and value.strip().lower() in invalid_tokens
                    else value
                )
            )
            series = pd.to_numeric(cleaned, errors="coerce")
        else:
            series = pd.to_numeric(pd.Series(series_raw), errors="coerce")
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (series - series.mean()) / std
        df[f"{name}_z"] = z
        composite += weight * z
        total_weight += abs(weight)

    if total_weight:
        df["CompositeScore"] = composite / total_weight
    else:
        df["CompositeScore"] = composite
    return df


def _reorder_table(
    df: pd.DataFrame,
    param_order: Optional[Sequence[str]],
    leading_columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = {"score": "Score", "valid": "Valid"}
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

    front: List[str] = [col for col in leading_columns if col in df.columns]
    ordered_params: List[str] = []
    if param_order:
        ordered_params = [col for col in param_order if col in df.columns]
    remaining = [
        col
        for col in df.columns
        if col not in front
        and col not in ordered_params
    ]
    ordered_cols = front + ordered_params + remaining
    return df.loc[:, ordered_cols]


def export_results(
    results: List[Dict[str, object]],
    objectives: Iterable[object],
    output_dir: Path,
    *,
    param_order: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dir(output_dir)
    agg_df, dataset_df = _flatten_results(results)
    agg_df = _annotate_objectives(agg_df, objectives)
    # 총 자산 관련 지표를 Score 앞에 배치해 결과 자본 흐름을 우선 확인할 수 있도록 정렬한다.
    # 기존 자본/위험 관련 컬럼 역시 상단에 유지해 핵심 지표를 한눈에 파악할 수 있게 한다.
    agg_df = _reorder_table(
        agg_df,
        param_order,
        (
            "TotalAssets",
            "Liquidations",
            "AvailableCapital",
            "Savings",
            "Ruin",
            "Score",
            "CompositeScore",
            "Valid",
            "Trades",
            "WinRate",
            "MaxDD",
            "NetProfit",
            "trial",
        ),
    )
    agg_df.to_csv(output_dir / "results.csv", index=False)
    if not dataset_df.empty:
        if "htf_timeframe" in dataset_df.columns:
            dataset_df = dataset_df.drop(columns=["htf_timeframe"])
        dataset_df = _reorder_table(
            dataset_df,
            param_order,
            (
                "TotalAssets",
                "Liquidations",
                "AvailableCapital",
                "Savings",
                "Ruin",
                "Score",
                "Valid",
                "Trades",
                "WinRate",
                "MaxDD",
                "NetProfit",
                "dataset",
                "timeframe",
                "trial",
            ),
        )
        dataset_df.to_csv(output_dir / "results_datasets.csv", index=False)
    return agg_df, dataset_df


def export_best(best: Dict[str, object], wf_summary: Dict[str, object], output_dir: Path) -> None:
    segments_payload = []
    for seg in wf_summary.get("segments", []):
        segments_payload.append(
            {
                "train": [seg.train_start.isoformat(), seg.train_end.isoformat()],
                "test": [seg.test_start.isoformat(), seg.test_end.isoformat()],
                "train_metrics": seg.train_metrics,
                "test_metrics": seg.test_metrics,
            }
        )

    payload = {
        "params": best.get("params"),
        "metrics": best.get("metrics"),
        "score": best.get("score"),
        "datasets": best.get("datasets", []),
        "walk_forward": {
            "oos_mean": wf_summary.get("oos_mean"),
            "oos_median": wf_summary.get("oos_median"),
            "count": wf_summary.get("count"),
            "segments": segments_payload,
            "candidates": wf_summary.get("candidates", []),
        },
    }
    if "monte_carlo" in best:
        payload["monte_carlo"] = best["monte_carlo"]
    (output_dir / "best.json").write_text(json.dumps(payload, indent=2))


def export_heatmap(metrics_df: pd.DataFrame, params: List[str], metric: str, plots_dir: Path) -> None:
    if len(params) < 2 or metrics_df.empty or metric not in metrics_df.columns:
        return
    x_param, y_param = params[:2]
    if x_param not in metrics_df or y_param not in metrics_df:
        return
    pivot = metrics_df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc="mean")
    if pivot.empty:
        return
    _ensure_dir(plots_dir)
    plt.figure(figsize=(10, 6))
    try:
        sns = _get_seaborn()
    except ImportError as exc:
        LOGGER.warning("seaborn 사용이 불가능하여 heatmap 생성을 건너뜁니다: %s", exc)
        plt.close()
        return
    sns.heatmap(pivot, annot=False, cmap="viridis")
    plt.title(f"{metric} heatmap ({y_param} vs {x_param})")
    plt.tight_layout()
    plt.savefig(plots_dir / "heatmap.png")
    plt.close()


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part))
            for col in df.columns.values
        ]
    return df


def export_timeframe_summary(dataset_df: pd.DataFrame, output_dir: Path) -> None:
    if dataset_df.empty:
        return
    if "timeframe" not in dataset_df.columns:
        return

    df = dataset_df.copy()
    df["timeframe"] = df["timeframe"].astype(str)

    metrics = [
        "TotalAssets",
        "NetProfit",
        "MaxDD",
        "WinRate",
        "WeeklyNetProfit",
        "Trades",
        "Liquidations",
    ]
    present = [metric for metric in metrics if metric in df.columns]
    if not present:
        return

    summary = (
        df.groupby(["timeframe"], dropna=False)[present]
        .agg(["mean", "median", "max"])
        .sort_index()
    )
    if summary.empty:
        return

    summary = summary.round(6).reset_index()
    summary = _flatten_multiindex_columns(summary)
    summary.to_csv(output_dir / "results_timeframe_summary.csv", index=False)

    sort_candidates = [
        "TotalAssets_mean",
        "NetProfit_mean",
        "MaxDD_mean",
    ]
    sort_metric: Optional[str] = next((name for name in sort_candidates if name in summary.columns), None)
    rankings = summary.sort_values(sort_metric, ascending=False) if sort_metric else summary
    rankings.to_csv(output_dir / "results_timeframe_rankings.csv", index=False)


def generate_reports(
    results: List[Dict[str, object]],
    best: Dict[str, object],
    wf_summary: Dict[str, object],
    objectives: Iterable[object],
    output_dir: Path,
    *,
    param_order: Optional[Sequence[str]] = None,
) -> None:
    agg_df, dataset_df = export_results(
        results,
        objectives,
        output_dir,
        param_order=param_order,
    )

    best_payload = deepcopy(best)
    if isinstance(best_payload, dict):
        params_payload = best_payload.get("params")
        if isinstance(params_payload, dict):
            ordered_params = OrderedDict()
            if param_order:
                for key in param_order:
                    if key in params_payload and key not in ordered_params:
                        ordered_params[key] = params_payload[key]
            for key, value in params_payload.items():
                if key not in ordered_params:
                    ordered_params[key] = value
            best_payload["params"] = dict(ordered_params)

        metrics_payload = best_payload.get("metrics")
        if isinstance(metrics_payload, dict):
            # 총 자산과 청산 횟수를 우선 정렬하고 나머지는 기존 순서를 유지합니다.
            ordered_metrics = OrderedDict()
            for key in ("TotalAssets", "Liquidations"):
                if key in metrics_payload and key not in ordered_metrics:
                    ordered_metrics[key] = metrics_payload[key]
            for key, value in metrics_payload.items():
                if key not in ordered_metrics:
                    ordered_metrics[key] = value
            best_payload["metrics"] = dict(ordered_metrics)

    monte_carlo_summary = _run_trade_monte_carlo(_collect_trade_profits(best_payload))
    if monte_carlo_summary:
        best_payload["monte_carlo"] = monte_carlo_summary
        (output_dir / "monte_carlo.json").write_text(
            json.dumps(monte_carlo_summary, indent=2)
        )

    export_best(best_payload, wf_summary, output_dir)
    export_timeframe_summary(dataset_df, output_dir)

    params = list(best.get("params", {}).keys())
    metric_name = next((name for name, _ in _objective_iterator(objectives)), "NetProfit")
    plots_dir = output_dir / "plots"
    export_heatmap(agg_df, params, metric_name, plots_dir)


def write_trials_dataframe(
    study: optuna.study.Study,
    output_dir: Path,
    *,
    param_order: Optional[Sequence[str]] = None,
) -> None:
    _ensure_dir(output_dir)
    try:
        trials_df = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "state",
                "datetime_start",
                "datetime_complete",
                "params",
                "user_attrs",
            )
        )
    except Exception:
        return
    if trials_df.empty:
        return
    attrs_series = trials_df.get("user_attrs") if "user_attrs" in trials_df else None
    if attrs_series is not None:
        attrs_series = attrs_series.apply(
            lambda payload: payload if isinstance(payload, dict) else {}
        )

    def _attr_value(key: str) -> pd.Series:
        if attrs_series is None:
            return pd.Series([None] * len(trials_df))
        return attrs_series.apply(lambda payload: payload.get(key))

    def _metric_value(*names: str) -> pd.Series:
        if attrs_series is None:
            return pd.Series([None] * len(trials_df))

        def _extract(payload: object) -> Optional[object]:
            metrics = None
            if isinstance(payload, dict):
                metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                return None
            for name in names:
                if name in metrics:
                    return metrics.get(name)
            return None

        return attrs_series.apply(_extract)

    def _flag_attr(key: str) -> pd.Series:
        series = _attr_value(key)

        def _coerce(value: object) -> Optional[bool]:
            if isinstance(value, bool):
                return True if value else None
            if value in (None, "", 0):
                return None
            try:
                return True if bool(value) else None
            except Exception:
                return None

        return series.apply(_coerce)

    if attrs_series is not None:
        dataset_meta = attrs_series.apply(
            lambda payload: payload.get("dataset_key")
            if isinstance(payload, dict)
            else {}
        )
    else:
        dataset_meta = pd.Series([{}] * len(trials_df))

    trial_numbers = trials_df["number"] if "number" in trials_df else pd.Series(range(len(trials_df)))
    if "state" in trials_df:
        states = trials_df["state"].apply(_normalise_state_label)
    else:
        states = pd.Series(["" for _ in range(len(trials_df))])
    values = trials_df["value"] if "value" in trials_df else pd.Series([None] * len(trials_df))
    completed = (
        trials_df["datetime_complete"]
        if "datetime_complete" in trials_df
        else pd.Series([None] * len(trials_df))
    )

    # Build the trial summary columns.  총 자산과 자본 관련 지표를 우선 노출합니다.
    # 딕셔너리의 키 순서가 결과 DataFrame의 컬럼 순서를 결정합니다.
    summary_columns: Dict[str, pd.Series] = {
        "TotalAssets": _metric_value("TotalAssets"),
        "Liquidations": _metric_value("Liquidations"),
        "Leverage": _attr_value("leverage"),
        "ChartTF": _attr_value("chart_tf"),
        "EntryTF": _attr_value("entry_tf"),
        "UseHTF": _flag_attr("use_htf"),
        "HtfTF": _attr_value("htf_tf"),
        "UseFixedStop": _flag_attr("use_fixed_stop"),
        "UseAtrStop": _flag_attr("use_atr_stop"),
        "UseAtrTrail": _flag_attr("use_atr_trail"),
        "UseChannelStop": _flag_attr("use_channel_stop"),
        "StopChannelType": _attr_value("stop_channel_type"),
        "StopChannelMult": _attr_value("stop_channel_mult"),
        "Score": _attr_value("score"),
        "Valid": _attr_value("valid"),
        "Trades": _attr_value("trades"),
        "WinRate": _metric_value("WinRate"),
        "MaxDD": _metric_value("MaxDD", "MaxDrawdown"),
        "Trial": trial_numbers,
        "State": states,
        "Value": values,
        "Completed": completed,
        "Timeframe": dataset_meta.apply(
            lambda meta: (
                meta.get("effective_timeframe")
                if isinstance(meta, dict) and meta.get("effective_timeframe")
                else (meta.get("timeframe") if isinstance(meta, dict) else None)
            )
        ),
    }

    summary_df = pd.DataFrame(summary_columns)

    params_series = trials_df.get("params") if "params" in trials_df else None
    if params_series is not None:
        params_df = pd.json_normalize(params_series).replace({pd.NA: None})
    else:
        params_df = pd.DataFrame()
    
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

    if not params_df.empty:
        if "fixedStopPct" in params_df.columns:
            fixed_mask = params_df["fixedStopPct"].apply(_positive_number).notna()
            params_df.loc[~fixed_mask, "fixedStopPct"] = None
        atr_len_mask = None
        if "atrStopLen" in params_df.columns:
            atr_len_mask = params_df["atrStopLen"].apply(_positive_number).notna()
            params_df.loc[~atr_len_mask, "atrStopLen"] = None
        if "atrStopMult" in params_df.columns:
            if atr_len_mask is None and "atrStopLen" in params_df.columns:
                atr_len_mask = params_df["atrStopLen"].apply(_positive_number).notna()
            if atr_len_mask is not None:
                params_df.loc[~atr_len_mask, "atrStopMult"] = None
            else:
                atr_mult_mask = params_df["atrStopMult"].apply(_positive_number).notna()
                params_df.loc[~atr_mult_mask, "atrStopMult"] = None
        if "useAtrTrail" in params_df.columns:
            trail_mask = params_df["useAtrTrail"].apply(lambda value: bool(value))
            params_df.loc[~trail_mask, "useAtrTrail"] = None
            for column in ("atrTrailLen", "atrTrailMult"):
                if column in params_df.columns:
                    params_df.loc[~trail_mask, column] = None
        if "stopChannelType" in params_df.columns:
            params_df.loc[params_df["stopChannelType"].isin([None, "", "None"]), "stopChannelType"] = None
        if "stopChannelMult" in params_df.columns:
            if "stopChannelType" in params_df.columns:
                active_mask = params_df["stopChannelType"].notna()
                params_df.loc[~active_mask, "stopChannelMult"] = None
            else:
                params_df.loc[params_df["stopChannelMult"].apply(_positive_number).isna(), "stopChannelMult"] = None
    if not params_df.empty:
        ordered_params: List[str] = []
        if param_order:
            ordered_params = [col for col in param_order if col in params_df.columns]
        remaining_params = [col for col in params_df.columns if col not in ordered_params]
        params_df = params_df.loc[:, ordered_params + remaining_params]

    combined = pd.concat([summary_df, params_df], axis=1)
    combined = combined.loc[:, [col for col in combined.columns if not combined[col].isna().all()]]
    cleaned_columns: List[object] = []
    for col in combined.columns:
        if isinstance(col, str):
            if col.strip():
                cleaned_columns.append(col)
        elif col is not None:
            cleaned_columns.append(col)
    if cleaned_columns:
        combined = combined.loc[:, cleaned_columns]
    combined.to_csv(output_dir / "trials.csv", index=False)


def write_bank_file(output_dir: Path, payload: Dict[str, object]) -> None:
    (output_dir / "bank.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
