import csv
from datetime import UTC, datetime
import argparse
from typing import Dict, List, Optional

import logging
from pathlib import Path

import pandas as pd
import pytest
from optuna.trial import TrialState

from optimize.run import (
    DatasetCacheInfo,
    DatasetSpec,
    optimisation_loop,
    _configure_parallel_workers,
    _ensure_timeframe_param,
    _enforce_exit_guards,
    _enforce_forced_timeframe_constraints,
    _apply_study_registry_defaults,
    _clean_metrics,
    _dataset_total_volume,
    _filter_basic_factor_params,
    _has_sufficient_volume,
    _group_datasets,
    _normalise_periods,
    _process_pool_initializer,
    _register_study_reference,
    _resolve_symbol_entry,
    _serialise_datasets_for_process,
    _sanitise_storage_meta,
    _study_registry_dir,
    _restrict_to_basic_factors,
    _run_dataset_backtest_task,
    _select_datasets_for_params,
    _normalise_timeframe_mix_argument,
    parse_args,
)


def test_normalise_periods_returns_configured_entries():
    periods_cfg = [{"from": "2023-01-01", "to": "2023-06-30"}]
    base = {"from": "2022-01-01", "to": "2022-12-31"}

    result = _normalise_periods(periods_cfg, base)

    assert result == [{"from": "2023-01-01", "to": "2023-06-30"}]


def test_normalise_periods_falls_back_to_base_when_empty():
    base = {"from": "2022-01-01", "to": "2022-12-31"}

    result = _normalise_periods([], base)

    assert result == [{"from": "2022-01-01", "to": "2022-12-31"}]


def test_normalise_timeframe_mix_argument_canonicalises_namespace():
    args = argparse.Namespace(timeframe_mix="1m, 3m;5m", timeframe=None)

    mix = _normalise_timeframe_mix_argument(args)

    assert mix == ["1m", "3m", "5m"]
    assert args.timeframe_mix == "1m,3m,5m"


def test_normalise_timeframe_mix_argument_degrades_to_single_timeframe():
    args = argparse.Namespace(timeframe_mix="1m", timeframe=None)

    mix = _normalise_timeframe_mix_argument(args)

    assert mix == []
    assert args.timeframe == "1m"
    assert args.timeframe_mix is None


@pytest.mark.parametrize(
    "payload",
    [
        [{"from": "2023-01-01"}],
        [{"to": "2023-06-30"}],
        ["2023-01-01/2023-06-30"],
    ],
)
def test_normalise_periods_raises_for_invalid_entries(payload):
    base = {}

    with pytest.raises(ValueError):
        _normalise_periods(payload, base)


def test_resolve_symbol_entry_uses_alias_map():
    alias, resolved = _resolve_symbol_entry(
        "BINANCE:XPLUSDT", {"BINANCE:XPLUSDT": "BINANCE:XPLAUSDT"}
    )

    assert alias == "BINANCE:XPLUSDT"
    assert resolved == "BINANCE:XPLAUSDT"


def test_resolve_symbol_entry_accepts_mapping_definition():
    entry = {"alias": "BINANCE:ASTERUSDT", "symbol": "BINANCE:ASTRUSDT"}
    alias, resolved = _resolve_symbol_entry(entry, {})

    assert alias == "BINANCE:ASTERUSDT"
    assert resolved == "BINANCE:ASTRUSDT"


def test_parse_args_accepts_trial_override():
    args = parse_args(["--n-trials", "25"])

    assert args.n_trials == 25


def test_parse_args_exposes_space_flags():
    args = parse_args(["--full-space"])

    assert args.full_space is True
    assert args.basic_factors_only is False

    args = parse_args(["--basic-factors-only"])

    assert args.full_space is False
    assert args.basic_factors_only is True


def test_basic_factor_filter_toggle():
    space = {
        "oscLen": {"type": "int"},
        "leverage": {"type": "int"},
        "fixedStopPct": {"type": "float"},
        "useChandelierExit": {"type": "bool"},
        "chandelierLen": {"type": "int"},
        "chandelierMult": {"type": "float"},
        "useSarExit": {"type": "bool"},
        "sarStart": {"type": "float"},
        "sarIncrement": {"type": "float"},
        "sarMaximum": {"type": "float"},
        "custom": {"type": "int"},
    }

    restricted = _restrict_to_basic_factors(space)
    assert restricted == {
        "oscLen": {"type": "int"},
        "leverage": {"type": "int"},
        "fixedStopPct": {"type": "float"},
        "useChandelierExit": {"type": "bool"},
        "chandelierLen": {"type": "int"},
        "chandelierMult": {"type": "float"},
        "useSarExit": {"type": "bool"},
        "sarStart": {"type": "float"},
        "sarIncrement": {"type": "float"},
        "sarMaximum": {"type": "float"},
    }

    restored = _restrict_to_basic_factors(space, enabled=False)
    assert restored == space


def test_basic_factor_param_filter_toggle():
    params = {
        "oscLen": 20,
        "exitOpposite": True,
        "useChandelierExit": True,
        "chandelierMult": 2.5,
        "chandelierLen": 15,
        "useSarExit": False,
        "sarMaximum": 0.2,
        "sarIncrement": 0.02,
        "sarStart": 0.01,
        "leverage": 10,
        "custom": 7,
    }

    filtered = _filter_basic_factor_params(params)
    assert filtered == {
        "oscLen": 20,
        "exitOpposite": True,
        "useChandelierExit": True,
        "chandelierMult": 2.5,
        "chandelierLen": 15,
        "useSarExit": False,
        "sarMaximum": 0.2,
        "sarIncrement": 0.02,
        "sarStart": 0.01,
        "leverage": 10,
    }

    unfiltered = _filter_basic_factor_params(params, enabled=False)
    assert unfiltered == params


def test_enforce_exit_guards_applies_defaults():
    result = _enforce_exit_guards({})

    assert result["exitOpposite"] is True
    assert result["useMomFade"] is False


def test_enforce_exit_guards_forces_fail_safe(caplog):
    with caplog.at_level(logging.WARNING):
        result = _enforce_exit_guards(
            {"exitOpposite": False, "useMomFade": False}, context="trial #7"
        )

    assert result["exitOpposite"] is True
    assert result["useMomFade"] is False
    assert "trial #7" in caplog.text


def test_enforce_exit_guards_handles_string_false_requests():
    result = _enforce_exit_guards({"exitOpposite": "false", "useMomFade": "false"})

    assert result["exitOpposite"] is True
    assert result["useMomFade"] is False


def _make_dataset(timeframe: str, htf: Optional[str]) -> DatasetSpec:
    idx = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10, 11, 12],
        },
        index=idx,
    )
    return DatasetSpec(
        symbol="BINANCE:TESTUSDT",
        timeframe=timeframe,
        start="2024-01-01",
        end="2024-01-02",
        df=df,
        htf=None,
        htf_timeframe=htf,
        source_symbol="BINANCE:TESTUSDT",
    )


def test_ensure_timeframe_param_infers_choices_when_missing():
    datasets = [_make_dataset("1m", None), _make_dataset("3m", None)]
    space: Dict[str, Dict[str, object]] = {}
    search_cfg = {"diversify": {"timeframe_cycle": [{"timeframe": "1m"}, {"timeframe": "3m"}]}}

    updated, added = _ensure_timeframe_param(space, datasets, search_cfg)

    assert added is True
    assert "timeframe" in updated
    assert updated["timeframe"]["values"] == ["1m", "3m"]
    assert "ltf" in updated
    assert updated["ltf"]["values"] == ["1m", "3m"]


def test_ensure_timeframe_param_supplements_missing_ltf():
    datasets = [_make_dataset("1m", None), _make_dataset("3m", None)]
    space: Dict[str, Dict[str, object]] = {"timeframe": {"type": "choice", "values": ["1m", "3m"]}}
    search_cfg: Dict[str, object] = {}

    updated, added = _ensure_timeframe_param(space, datasets, search_cfg)

    assert added is True
    assert updated["timeframe"] == {"type": "choice", "values": ["1m", "3m"]}
    assert updated["ltf"] == {"type": "choice", "values": ["1m", "3m"]}


def test_ensure_timeframe_param_supplements_missing_timeframe():
    datasets = [_make_dataset("1m", None), _make_dataset("3m", None)]
    space: Dict[str, Dict[str, object]] = {"ltf": {"type": "choice", "values": ["1m", "3m"]}}
    search_cfg: Dict[str, object] = {}

    updated, added = _ensure_timeframe_param(space, datasets, search_cfg)

    assert added is True
    assert updated["ltf"] == {"type": "choice", "values": ["1m", "3m"]}
    assert updated["timeframe"] == {"type": "choice", "values": ["1m", "3m"]}


def test_dataset_total_volume_sums_numeric_values():
    dataset = _make_dataset("1m", None)

    assert dataset.total_volume is None
    total = _dataset_total_volume(dataset)

    assert total == pytest.approx(33.0)
    assert dataset.total_volume == pytest.approx(33.0)

    # Subsequent calls should use the cached value even if the frame changes.
    dataset.df.loc[:, "volume"] = [0, 0, 0]
    cached = _dataset_total_volume(dataset)

    assert cached == pytest.approx(33.0)
    assert dataset.total_volume == pytest.approx(33.0)


def test_dataset_total_volume_handles_missing_volume_column():
    dataset = _make_dataset("1m", None)
    dataset.df = dataset.df.drop(columns=["volume"])

    total = _dataset_total_volume(dataset)

    assert total == 0.0
    assert dataset.total_volume == 0.0


def test_has_sufficient_volume_detects_shortfall():
    dataset = _make_dataset("1m", None)

    meets, total = _has_sufficient_volume(dataset, 100.0)

    assert meets is False
    assert total == pytest.approx(33.0)


def test_select_datasets_respects_timeframe_and_htf_choice():
    datasets = [_make_dataset("1m", "15m"), _make_dataset("3m", "1h")]
    groups, timeframe_groups, default_key = _group_datasets(datasets)
    params_cfg = {"timeframe": "1m", "htf_timeframes": ["15m", "1h"]}
    params = {"timeframe": "3m", "htf": "1h"}

    key, selection = _select_datasets_for_params(params_cfg, groups, timeframe_groups, default_key, params)

    assert key == ("3m", "1h")
    assert selection == [datasets[1]]


def test_select_datasets_falls_back_when_htf_disabled():
    datasets = [_make_dataset("1m", "15m"), _make_dataset("1m", None)]
    groups, timeframe_groups, default_key = _group_datasets(datasets)
    params_cfg = {"timeframe": "1m", "htf_timeframes": ["15m"]}
    params = {"timeframe": "1m", "htf": "none"}

    key, selection = _select_datasets_for_params(params_cfg, groups, timeframe_groups, default_key, params)

    assert key[0] == "1m"
    assert all(dataset.timeframe == "1m" for dataset in selection)


def test_select_datasets_supports_multi_timeframe_choice():
    datasets = [_make_dataset("1m", None), _make_dataset("3m", None), _make_dataset("5m", None)]
    groups, timeframe_groups, default_key = _group_datasets(datasets)
    params_cfg: Dict[str, object] = {}
    params = {"entry_tf": "1m,3m,5m"}

    key, selection = _select_datasets_for_params(params_cfg, groups, timeframe_groups, default_key, params)

    assert [dataset.timeframe for dataset in selection] == ["1m", "3m", "5m"]
    assert key[0] == "1m,3m,5m"


def test_forced_timeframe_matches_trial_params_and_dataset_key(monkeypatch):
    forced_tf = "3m"
    params_cfg = {
        "symbol": "BINANCE:TESTUSDT",
        "timeframe": forced_tf,
        "space": {
            "oscLen": {"type": "int", "min": 1, "max": 1, "step": 1},
            "timeframe": {"type": "choice", "values": ["1m", forced_tf]},
            "ltf": {"type": "choice", "values": ["1m", forced_tf]},
        },
        "search": {
            "n_trials": 1,
            "n_jobs": 1,
            "algo": "random",
            "basic_factor_profile": False,
            "diversify": {
                "timeframe_cycle": [{"timeframe": "1m"}, {"timeframe": forced_tf}],
                "htf_timeframe": "1h",
            },
        },
    }

    search_cfg = params_cfg["search"]
    _enforce_forced_timeframe_constraints(params_cfg, search_cfg, forced_tf)

    assert params_cfg["space"]["timeframe"]["values"] == [forced_tf]
    assert params_cfg["space"]["ltf"]["values"] == [forced_tf]
    assert search_cfg["diversify"].get("timeframe_cycle") == []

    datasets = [_make_dataset("1m", None), _make_dataset(forced_tf, None)]

    def fake_run_backtest(*_, **__):
        return {
            "NetProfit": 100.0,
            "TotalAssets": 620.0,
            "Trades": 25,
            "Valid": True,
        }

    monkeypatch.setattr("optimize.run.run_backtest", fake_run_backtest)
    monkeypatch.setattr("optimize.run.backtest_cfg", {}, raising=False)

    optimisation = optimisation_loop(
        datasets,
        params_cfg,
        objectives=[{"name": "NetProfit"}],
        fees={},
        risk={},
        forced_params={"timeframe": forced_tf, "ltf": forced_tf},
    )

    record = optimisation["results"][0]
    dataset_key = record["dataset_key"]
    trial_params = optimisation["study"].best_trial.params

    assert dataset_key["timeframe"] == forced_tf
    assert trial_params.get("timeframe") == forced_tf
    assert trial_params.get("ltf") == forced_tf


def test_optimisation_loop_propagates_minimize_direction(monkeypatch):
    datasets = [_make_dataset("1m", None)]
    params_cfg = {
        "symbol": "BINANCE:TESTUSDT",
        "timeframe": "1m",
        "space": {},
        "search": {
            "algo": "random",
            "n_trials": 1,
            "n_jobs": 1,
            "multi_objective": False,
        },
    }
    captured: Dict[str, object] = {}

    class SentinelStudy(Exception):
        pass

    def fake_create_study(*args, **kwargs):
        captured.update(kwargs)
        raise SentinelStudy()

    monkeypatch.setattr("optimize.run.optuna.create_study", fake_create_study)

    with pytest.raises(SentinelStudy):
        optimisation_loop(
            datasets,
            params_cfg,
            objectives=[{"name": "MaxDD", "goal": "minimize"}],
            fees={},
            risk={},
        )

    assert captured.get("direction") == "minimize"


def test_run_dataset_backtest_task_falls_back_on_missing_alt_engine(monkeypatch):
    dataset = _make_dataset("1m", None)
    sentinel = {"called": False}

    def fake_native(df, params, fees, risk, htf_df=None, min_trades=None):
        sentinel["called"] = True
        return {"Valid": True, "Trades": 0.0}

    def fake_alt(*args, **kwargs):
        raise ImportError("vectorbt 미설치")

    monkeypatch.setattr("optimize.run.run_backtest", fake_native)
    monkeypatch.setattr("optimize.alternative_engine.run_backtest_alternative", fake_alt)

    metrics = _run_dataset_backtest_task(
        dataset,
        {"altEngine": "vectorbt"},
        {"commission_pct": 0.0006},
        {},
    )

    assert metrics["Valid"] is True
    assert sentinel["called"] is True


def test_run_dataset_backtest_task_accepts_dataset_id_via_process_cache(
    monkeypatch, tmp_path
):
    dataset = _make_dataset("1m", None)
    dataset.cache_info = DatasetCacheInfo(root=tmp_path, futures=False)

    handles = _serialise_datasets_for_process([dataset])
    _process_pool_initializer(handles)

    class DummyCache:
        def __init__(self, frame):
            self.frame = frame

        def get(self, *args, **kwargs):
            return self.frame

    dummy_cache = DummyCache(dataset.df)
    monkeypatch.setattr("optimize.run._resolve_process_cache", lambda *args, **kwargs: dummy_cache)

    sentinel = {"called": False}

    def fake_native(df, params, fees, risk, htf_df=None, min_trades=None):
        sentinel["called"] = True
        assert df.equals(dataset.df)
        return {"Valid": True}

    monkeypatch.setattr("optimize.run.run_backtest", fake_native)

    result = _run_dataset_backtest_task(handles[0]["id"], {}, {}, {})

    assert result == {"Valid": True}
    assert sentinel["called"] is True

    _process_pool_initializer([])


def test_optimisation_loop_process_executor_reuses_handles(monkeypatch, tmp_path):
    dataset_a = _make_dataset("1m", None)
    dataset_b = _make_dataset("1m", None)
    dataset_a.cache_info = DatasetCacheInfo(root=tmp_path, futures=False)
    dataset_b.cache_info = DatasetCacheInfo(root=tmp_path, futures=False)
    dataset_b.start = "2024-01-05"
    dataset_b.end = "2024-01-06"

    params_cfg: Dict[str, object] = {
        "space": {},
        "search": {
            "n_trials": 1,
            "dataset_executor": "process",
            "dataset_jobs": 2,
        },
        "llm": {"enabled": False},
        "timeframe": "1m",
    }

    objectives = [{"name": "NetProfit", "goal": "maximize"}]

    serialise_calls: List[List[str]] = []
    original_serialise = _serialise_datasets_for_process

    def fake_serialise(datasets):
        serialise_calls.append([dataset.name for dataset in datasets])
        return original_serialise(datasets)

    monkeypatch.setattr("optimize.run._serialise_datasets_for_process", fake_serialise)

    submissions: List[object] = []

    def fake_run(dataset_ref, params, fees, risk, min_trades=None):
        submissions.append(dataset_ref)
        return {
            "Valid": True,
            "TotalAssets": 580.0,
            "Trades": 20,
            "Wins": 12,
            "Losses": 8,
        }

    monkeypatch.setattr("optimize.run._run_dataset_backtest_task", fake_run)

    class DummyFuture:
        def __init__(self, fn, args, kwargs):
            self._fn = fn
            self._args = args
            self._kwargs = kwargs
            self.cancelled = False

        def result(self):
            return self._fn(*self._args, **self._kwargs)

        def cancel(self):
            self.cancelled = True

    class DummyProcessPoolExecutor:
        instances: List["DummyProcessPoolExecutor"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_called = False
            DummyProcessPoolExecutor.instances.append(self)

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn, args, kwargs)

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_called = True

    monkeypatch.setattr("optimize.run.ProcessPoolExecutor", DummyProcessPoolExecutor)

    class DummyTrial:
        def __init__(self, number: int):
            self.number = number
            self.params: Dict[str, object] = {}
            self.user_attrs: Dict[str, object] = {}
            self.state = TrialState.RUNNING
            self.datetime_complete = None

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

        def report(self, *args, **kwargs) -> None:  # pragma: no cover - behaviourless
            return None

        def should_prune(self) -> bool:
            return False

    class DummyStudy:
        def __init__(self):
            self._trials: List[DummyTrial] = []
            self._attrs: Dict[str, object] = {}

        def set_user_attr(self, key: str, value: object) -> None:
            self._attrs[key] = value

        def enqueue_trial(self, *args, **kwargs) -> None:
            return None

        def get_trials(self, deepcopy: bool = False):
            return list(self._trials)

        @property
        def trials(self):
            return list(self._trials)

        def trials_dataframe(self):
            return pd.DataFrame()

        @property
        def best_trial(self):
            if self._trials:
                return self._trials[0]
            dummy = DummyTrial(0)
            dummy.state = TrialState.COMPLETE
            dummy.value = 0.0
            return dummy

        def optimize(
            self,
            objective,
            n_trials,
            n_jobs,
            show_progress_bar,
            callbacks,
            gc_after_trial,
            catch,
        ):
            trial = DummyTrial(0)
            value = objective(trial)
            trial.state = TrialState.COMPLETE
            trial.value = value
            self._trials.append(trial)
            for callback in callbacks:
                callback(self, trial)
            raise StopIteration

    dummy_study = DummyStudy()
    monkeypatch.setattr("optimize.run.optuna.create_study", lambda *args, **kwargs: dummy_study)
    monkeypatch.setattr("optimize.run.backtest_cfg", {}, raising=False)

    with pytest.raises(StopIteration):
        optimisation_loop(
            [dataset_a, dataset_b],
            params_cfg,
            objectives,
            fees={},
            risk={"min_volume": 0},
        )

    assert len(serialise_calls) == 1
    assert {ref for ref in submissions} == {dataset_a.name, dataset_b.name}
    assert DummyProcessPoolExecutor.instances
    assert DummyProcessPoolExecutor.instances[0].shutdown_called is True


def test_optimisation_loop_thread_executor_uses_objects(monkeypatch):
    dataset_a = _make_dataset("1m", None)
    dataset_b = _make_dataset("1m", None)

    params_cfg: Dict[str, object] = {
        "space": {},
        "search": {
            "n_trials": 1,
            "dataset_executor": "thread",
            "dataset_jobs": 2,
        },
        "llm": {"enabled": False},
        "timeframe": "1m",
    }

    objectives = [{"name": "NetProfit", "goal": "maximize"}]

    submissions: List[object] = []

    def fake_run(dataset_ref, params, fees, risk, min_trades=None):
        submissions.append(dataset_ref)
        return {
            "Valid": True,
            "TotalAssets": 540.0,
            "Trades": 15,
            "Wins": 9,
            "Losses": 6,
        }

    monkeypatch.setattr("optimize.run._run_dataset_backtest_task", fake_run)

    class DummyFuture:
        def __init__(self, fn, args, kwargs):
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def result(self):
            return self._fn(*self._args, **self._kwargs)

        def cancel(self):
            return None

    class DummyThreadPoolExecutor:
        instances: List["DummyThreadPoolExecutor"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_called = False
            DummyThreadPoolExecutor.instances.append(self)

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn, args, kwargs)

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_called = True

    monkeypatch.setattr("optimize.run.ThreadPoolExecutor", DummyThreadPoolExecutor)

    class DummyTrial:
        def __init__(self, number: int):
            self.number = number
            self.params: Dict[str, object] = {}
            self.user_attrs: Dict[str, object] = {}
            self.state = TrialState.RUNNING
            self.datetime_complete = None

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

        def report(self, *args, **kwargs) -> None:  # pragma: no cover - behaviourless
            return None

        def should_prune(self) -> bool:
            return False

    class DummyStudy:
        def __init__(self):
            self._trials: List[DummyTrial] = []

        def set_user_attr(self, key: str, value: object) -> None:
            return None

        def enqueue_trial(self, *args, **kwargs) -> None:
            return None

        def get_trials(self, deepcopy: bool = False):
            return list(self._trials)

        @property
        def trials(self):
            return list(self._trials)

        def trials_dataframe(self):
            return pd.DataFrame()

        @property
        def best_trial(self):
            if self._trials:
                return self._trials[0]
            dummy = DummyTrial(0)
            dummy.state = TrialState.COMPLETE
            dummy.value = 0.0
            return dummy

        def optimize(
            self,
            objective,
            n_trials,
            n_jobs,
            show_progress_bar,
            callbacks,
            gc_after_trial,
            catch,
        ):
            trial = DummyTrial(0)
            value = objective(trial)
            trial.state = TrialState.COMPLETE
            trial.value = value
            self._trials.append(trial)
            for callback in callbacks:
                callback(self, trial)
            raise StopIteration

    dummy_study = DummyStudy()
    monkeypatch.setattr("optimize.run.optuna.create_study", lambda *args, **kwargs: dummy_study)
    monkeypatch.setattr("optimize.run.backtest_cfg", {}, raising=False)

    with pytest.raises(StopIteration):
        optimisation_loop(
            [dataset_a, dataset_b],
            params_cfg,
            objectives,
            fees={},
            risk={"min_volume": 0},
        )

    assert len(submissions) == 2
    assert all(ref is dataset_a or ref is dataset_b for ref in submissions)
    assert DummyThreadPoolExecutor.instances
    assert DummyThreadPoolExecutor.instances[0].shutdown_called is True

def test_study_registry_round_trip_for_rdb(tmp_path):
    study_path = tmp_path / "studies" / "demo.db"
    storage_meta = {
        "backend": "rdb",
        "url": "postgresql://postgres:5432@127.0.0.1:5432/optuna",
        "env_key": "OPTUNA_STORAGE",
        "env_value_present": True,
        "pool": {"size": 8},
    }

    _register_study_reference(study_path, storage_meta=storage_meta, study_name="demo")

    pointer_path = _study_registry_dir(study_path) / "storage.json"
    assert pointer_path.exists()

    search_cfg: Dict[str, object] = {}
    _apply_study_registry_defaults(search_cfg, study_path)

    assert search_cfg["storage_url"] == storage_meta["url"]
    assert search_cfg["storage_url_env"] == "OPTUNA_STORAGE"


def test_sanitise_storage_meta_masks_password():
    raw = {
        "backend": "rdb",
        "url": "postgresql://postgres:5432@127.0.0.1:5432/optuna",
    }

    masked = _sanitise_storage_meta(raw)

    assert masked["url"].startswith("postgresql://postgres:***@127.0.0.1")
    # 원본 딕셔너리는 변경하지 않습니다.
    assert raw["url"].endswith("/optuna")


def test_clean_metrics_preserves_numeric_values() -> None:
    metrics = {
        "TotalAssets": 1520.75,
        "Liquidations": 2,
        "Label": "ok",
    }
    cleaned = _clean_metrics(metrics)
    assert cleaned["TotalAssets"] == 1520.75
    assert cleaned["Liquidations"] == 2
    assert cleaned["Label"] == "ok"


def test_log_trial_preserves_zero_metrics(monkeypatch, tmp_path):
    dataset = _make_dataset("1m", None)

    metrics_payload = {
        "TotalAssets": 500.0,
        "Liquidations": 0.0,
        "Trades": 0.0,
        "WinRate": 0.0,
        "MaxDD": 12.3,
        "Valid": 1.0,
        "Leverage": 4.0,
    }

    class DummyTrial:
        def __init__(self, number: int):
            self.number = number
            self.value = 0.0
            self.state = TrialState.COMPLETE
            self.params: Dict[str, object] = {}
            self.datetime_complete = datetime.now(UTC)
            self.user_attrs: Dict[str, object] = {
                "metrics": metrics_payload,
                "total_assets": metrics_payload["TotalAssets"],
                "liquidations": metrics_payload["Liquidations"],
                "score": 0.0,
                "valid": True,
                "pruned": False,
                "skipped_datasets": [],
            }

    class DummyStudy:
        def __init__(self, trial: DummyTrial):
            self._trial = trial
            self._trials: List[DummyTrial] = []
            self._attrs: Dict[str, object] = {}

        def set_user_attr(self, key: str, value: object) -> None:
            self._attrs[key] = value

        def enqueue_trial(self, *args, **kwargs) -> None:
            return None

        def get_trials(self, deepcopy: bool = False):
            return list(self._trials)

        @property
        def trials(self):
            return list(self._trials)

        def trials_dataframe(self):
            return pd.DataFrame()

        @property
        def best_trial(self):
            return self._trial

        def optimize(
            self,
            objective,
            n_trials,
            n_jobs,
            show_progress_bar,
            callbacks,
            gc_after_trial,
            catch,
        ):
            self._trials.append(self._trial)
            for callback in callbacks:
                callback(self, self._trial)
            raise StopIteration

    dummy_trial = DummyTrial(0)
    dummy_trial.params = {"leverage": 12}
    dummy_trial.user_attrs["leverage"] = 6.0
    dummy_study = DummyStudy(dummy_trial)

    def fake_create_study(*args, **kwargs):
        return dummy_study

    monkeypatch.setattr("optimize.run.optuna.create_study", fake_create_study)

    captured_logs: List[str] = []

    def fake_info(message, *args, **kwargs):
        text = message % args if args else message
        captured_logs.append(text)

    monkeypatch.setattr("optimize.run.LOGGER.info", fake_info)
    monkeypatch.setattr("optimize.run.backtest_cfg", {}, raising=False)

    params_cfg: Dict[str, object] = {"space": {}, "search": {"n_trials": 1}}
    objectives = [
        {"name": "NetProfit", "goal": "maximize"},
        {"name": "TotalAssets", "goal": "maximize", "weight": 3.0},
    ]

    with pytest.raises(StopIteration):
        optimisation_loop(
            [dataset],
            params_cfg,
            objectives,
            fees={},
            risk={},
            log_dir=tmp_path,
        )

    progress_path = tmp_path / "trials_progress.csv"
    assert progress_path.exists()

    with progress_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert pytest.approx(float(row["total_assets"])) == 500.0
    assert pytest.approx(float(row["liquidations"])) == 0.0
    assert pytest.approx(float(row["leverage"])) == 12.0
    assert row["ltf_choice"] == ""
    assert row["state"] == "완료"

    progress_logs = [entry for entry in captured_logs if "작업 진행상황" in entry]
    assert progress_logs, "진행 로그가 기록되어야 합니다."
    latest = progress_logs[-1]
    assert "Liquidations=0.0" in latest
    assert "TotalAssets=500.0" in latest


def test_configure_parallel_workers_single_dataset(caplog):
    dataset = _make_dataset("1m", None)
    dataset_groups, _, _ = _group_datasets([dataset])
    search_cfg: Dict[str, object] = {"n_jobs": 4}

    caplog.set_level(logging.INFO, logger="optimize")
    n_jobs, dataset_jobs, _, _ = _configure_parallel_workers(
        search_cfg,
        dataset_groups,
        available_cpu=8,
        n_jobs=4,
    )

    assert n_jobs == 4
    assert dataset_jobs == 1
    assert search_cfg["dataset_jobs"] == 1
    assert any(
        "단일 티커/데이터셋 구성" in message for message in caplog.messages
    )


def test_configure_parallel_workers_multiple_datasets_reduce_optuna(caplog):
    dataset_a = _make_dataset("1m", None)
    dataset_b = _make_dataset("1m", None)
    dataset_groups, _, _ = _group_datasets([dataset_a, dataset_b])
    search_cfg: Dict[str, object] = {"n_jobs": 6}

    caplog.set_level(logging.INFO, logger="optimize")
    n_jobs, dataset_jobs, _, _ = _configure_parallel_workers(
        search_cfg,
        dataset_groups,
        available_cpu=4,
        n_jobs=6,
    )

    assert dataset_jobs == 2
    assert n_jobs == 2
    assert search_cfg["n_jobs"] == 2
    assert any(
        "보조 데이터셋 병렬 worker" in message for message in caplog.messages
    )
