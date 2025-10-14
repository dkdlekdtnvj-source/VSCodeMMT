import numpy as np
import optuna
import pytest

from optimize.search_spaces import (
    grid_choices,
    mutate_around,
    random_parameters,
    sample_parameters,
)


def test_sample_parameters_skips_requires_when_condition_false():
    space = {
        "useStopLoss": {"type": "bool"},
        "stopLookback": {"type": "int", "min": 2, "max": 10, "step": 2, "requires": "useStopLoss"},
    }
    trial = optuna.trial.FixedTrial({"useStopLoss": False})

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is False
    assert "stopLookback" not in params


def test_sample_parameters_emits_requires_when_condition_true():
    space = {
        "useStopLoss": {"type": "bool"},
        "stopLookback": {"type": "int", "min": 2, "max": 10, "step": 2, "requires": "useStopLoss"},
    }
    trial = optuna.trial.FixedTrial({"useStopLoss": True, "stopLookback": 6})

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is True
    assert params["stopLookback"] == 6


def test_sample_parameters_respects_singleton_boolean_values():
    space = {
        "useStopLoss": {"type": "bool", "values": [True]},
    }
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler({"useStopLoss": [True]}),
    )
    trial = study.ask()

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is True
    assert trial.params["useStopLoss"] is True


def test_sample_parameters_keeps_at_least_one_exit_enabled():
    space = {
        "exitOpposite": {"type": "bool"},
        "useMomFade": {"type": "bool"},
    }
    trial = optuna.trial.FixedTrial({"exitOpposite": False, "useMomFade": True})

    params = sample_parameters(trial, space)

    assert params["exitOpposite"] or params["useMomFade"]


def test_sample_parameters_normalizes_trial_params_when_both_false():
    space = {
        "exitOpposite": {"type": "bool"},
        "useMomFade": {"type": "bool"},
    }
    trial = optuna.trial.FixedTrial({"exitOpposite": False, "useMomFade": False})

    params = sample_parameters(trial, space)

    assert params == {"exitOpposite": True, "useMomFade": False}
    assert trial.params == {"exitOpposite": True, "useMomFade": False}


def test_sample_parameters_forces_single_boolean_choice():
    space = {"useStopLoss": {"type": "bool", "values": [True]}}

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.GridSampler({"useStopLoss": [True]})
    )
    trial = study.ask()

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is True
    assert trial.params["useStopLoss"] is True


def test_sample_parameters_applies_boolean_default():
    space = {"useStopLoss": {"type": "bool", "default": True}}
    trial = optuna.trial.FixedTrial({"useStopLoss": False})

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is True
    assert trial.params["useStopLoss"] is True


def test_sample_parameters_updates_real_trial_params_after_guard_normalization():
    space = {
        "exitOpposite": {"type": "bool"},
        "useMomFade": {"type": "bool"},
    }
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler({"exitOpposite": [False], "useMomFade": [False]}),
    )
    trial = study.ask()

    params = sample_parameters(trial, space)

    assert params == {"exitOpposite": True, "useMomFade": False}
    assert trial.params == {"exitOpposite": True, "useMomFade": False}


def test_mutate_around_respects_integer_steps():
    space = {"foo": {"type": "int", "min": 3, "max": 11, "step": 2}}
    params = {"foo": 7}
    legal_values = {3, 5, 7, 9, 11}

    rng = np.random.default_rng(123)
    for _ in range(200):
        mutated = mutate_around(params, space, scale=0.5, rng=rng)
        assert mutated["foo"] in legal_values


def test_random_parameters_respects_requires_and_steps():
    space = {
        "useStopLoss": {"type": "bool"},
        "stopLookback": {
            "type": "int",
            "min": 2,
            "max": 10,
            "step": 2,
            "requires": "useStopLoss",
        },
        "signal": {"type": "float", "min": 1.0, "max": 2.0, "step": 0.5},
        "mode": {"type": "choice", "values": ["A", "B"]},
    }

    rng = np.random.default_rng(0)
    seen_stop_values = set()
    for _ in range(100):
        params = random_parameters(space, rng=rng)
        assert params["signal"] in {1.0, 1.5, 2.0}
        assert params["mode"] in {"A", "B"}
        if params["useStopLoss"]:
            assert params["stopLookback"] in {2, 4, 6, 8, 10}
            seen_stop_values.add(params["stopLookback"])
        else:
            assert "stopLookback" not in params
    assert seen_stop_values  # 적어도 한 번은 조건부 파라미터가 생성되어야 함


def test_random_parameters_keeps_exit_guard():
    space = {
        "exitOpposite": {"type": "bool"},
        "useMomFade": {"type": "bool"},
    }
    rng = np.random.default_rng(7)

    for _ in range(200):
        params = random_parameters(space, rng=rng)
        assert params["exitOpposite"] or params["useMomFade"]


def test_mutate_around_keeps_exit_guard():
    space = {
        "exitOpposite": {"type": "bool"},
        "useMomFade": {"type": "bool"},
    }
    rng = np.random.default_rng(21)
    params = {"exitOpposite": False, "useMomFade": True}

    for _ in range(100):
        mutated = mutate_around(params, space, scale=1.0, rng=rng)
        assert mutated["exitOpposite"] or mutated["useMomFade"]
        params = mutated


def test_grid_choices_raises_for_non_positive_float_step():
    space = {
        "alpha": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.0},
    }

    with pytest.raises(
        ValueError, match="float 파라미터 'alpha'는 grid 샘플링을 위해 양수 step이 필요합니다."
    ):
        grid_choices(space)


def test_grid_choices_raises_for_non_positive_int_step():
    space = {
        "beta": {"type": "int", "min": 0, "max": 10, "step": -1},
    }

    with pytest.raises(
        ValueError, match="int 파라미터 'beta'는 grid 샘플링을 위해 양수 step이 필요합니다."
    ):
        grid_choices(space)
