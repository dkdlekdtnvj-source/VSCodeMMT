"""Helpers for translating YAML search spaces to Optuna."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import optuna
from optuna.trial import FixedTrial

SpaceSpec = Dict[str, Dict[str, object]]


def build_space(space: SpaceSpec) -> SpaceSpec:
    return space


def _requirements_met(params: Dict[str, object], requirement: Union[str, Sequence[object], Dict[str, object]]) -> bool:
    if requirement in (None, ""):
        return True
    if isinstance(requirement, dict):
        name = requirement.get("name") or requirement.get("param") or requirement.get("key")
        expected = requirement.get("equals")
        if expected is None:
            expected = requirement.get("value", True)
        if not name:
            return True
        if name not in params:
            return False
        return params.get(name) == expected
    if isinstance(requirement, (list, tuple, set)):
        return all(_requirements_met(params, item) for item in requirement)
    # Treat plain strings as boolean flags that must evaluate to truthy
    name = str(requirement)
    if name not in params:
        return False
    return bool(params.get(name))

def _coerce_bool_or_none(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan"}:
            return None
        if text in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "f", "0", "no", "n", "off"}:
            return False
    return None


def _update_trial_param(trial: Optional[optuna.Trial], name: str, value: object) -> None:
    if trial is None:
        return

    if isinstance(trial, FixedTrial):
        if hasattr(trial, "_suggested_params"):
            trial._suggested_params[name] = value
        if hasattr(trial, "_params") and name in trial._params:
            trial._params[name] = value
        return

    frozen = getattr(trial, "_cached_frozen_trial", None)
    storage = getattr(trial, "storage", None)
    trial_id = getattr(trial, "_trial_id", None)
    if frozen is None or storage is None or trial_id is None:
        return

    distribution = frozen.distributions.get(name)
    if distribution is None:
        return

    storage.set_trial_param(
        trial_id,
        name,
        distribution.to_internal_repr(value),
        distribution,
    )
    frozen.params[name] = value


def _normalize_exit_flags(
    params: Dict[str, object], trial: Optional[optuna.Trial] = None
) -> Dict[str, object]:
    patched = dict(params)
    exit_flag = _coerce_bool_or_none(patched.get("exitOpposite"))
    mom_fade_flag = _coerce_bool_or_none(patched.get("useMomFade"))

    if exit_flag is None:
        exit_flag = True
    if mom_fade_flag is None:
        mom_fade_flag = False

    if exit_flag is False and mom_fade_flag is False:
        exit_flag = True

    patched["exitOpposite"] = exit_flag
    patched["useMomFade"] = mom_fade_flag

    _update_trial_param(trial, "exitOpposite", exit_flag)
    _update_trial_param(trial, "useMomFade", mom_fade_flag)

    return patched


def sample_parameters(trial: optuna.Trial, space: SpaceSpec) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for name, spec in space.items():
        requires = spec.get("requires")
        if requires and not _requirements_met(params, requires):
            if "default" in spec:
                params[name] = spec["default"]
            continue
        dtype = spec["type"]
        if dtype == "int":
            params[name] = trial.suggest_int(name, int(spec["min"]), int(spec["max"]), step=int(spec.get("step", 1)))
        elif dtype == "float":
            params[name] = trial.suggest_float(name, float(spec["min"]), float(spec["max"]), step=float(spec.get("step", 0.1)))
        elif dtype == "bool":
            choice_source: Optional[List[object]] = None
            choice_key: Optional[str] = None
            for key in ("values", "options", "choices"):
                candidate = spec.get(key)
                if isinstance(candidate, (list, tuple, set)):
                    choice_source = list(candidate)
                    choice_key = key
                    break

            normalized_choices: List[bool] = []
            if choice_source is not None:
                if not choice_source:
                    raise ValueError(
                        f"Bool 파라미터 '{name}' 의 '{choice_key}' 목록이 비어 있습니다."
                    )
                seen = set()
                for raw_value in choice_source:
                    coerced = _coerce_bool_or_none(raw_value)
                    if coerced is None:
                        raise ValueError(
                            f"Bool 파라미터 '{name}' 의 {choice_key} 항목에서 불리언으로 해석할 수 없는 값이 발견되었습니다."
                        )
                    if coerced not in seen:
                        normalized_choices.append(coerced)
                        seen.add(coerced)

            default_bool: Optional[bool] = None
            if "default" in spec:
                default_bool = _coerce_bool_or_none(spec.get("default"))
                if default_bool is None:
                    raise ValueError(
                        f"Bool 파라미터 '{name}' 의 기본값을 불리언으로 해석할 수 없습니다."
                    )

            if normalized_choices:
                if default_bool is not None and default_bool not in normalized_choices:
                    raise ValueError(
                        f"Bool 파라미터 '{name}' 의 기본값이 허용된 선택지와 일치하지 않습니다."
                    )
            elif default_bool is not None:
                normalized_choices = [default_bool]
            else:
                normalized_choices = [True, False]

            try:
                suggested = trial.suggest_categorical(name, normalized_choices)
            except ValueError:
                if default_bool is None:
                    raise
                suggested = default_bool

            suggested_bool = _coerce_bool_or_none(suggested)
            if suggested_bool is None:
                raise ValueError(
                    f"Bool 파라미터 '{name}' 은 불리언 값만 허용됩니다."
                )

            if default_bool is not None:
                params[name] = default_bool
                _update_trial_param(trial, name, default_bool)
            else:
                params[name] = suggested_bool
        elif dtype in {"choice", "str", "string"}:
            values = spec.get("values") or spec.get("options") or spec.get("choices")
            if not values:
                raise ValueError(f"Choice parameter '{name}' requires a non-empty 'values' list.")
            params[name] = trial.suggest_categorical(name, list(values))
        else:
            raise ValueError(f"Unsupported parameter type: {dtype}")
    return _normalize_exit_flags(params, trial)


def random_parameters(space: SpaceSpec, rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
    """Sample a random parameter set from *space* without an Optuna trial.

    이 함수는 Optuna sampler 문맥 밖에서 탐색 공간을 균일하게 샘플링할 때
    사용됩니다. `requires` 조건을 준수하며, 정수/실수 타입은 `step` 간격을
    고려해 균등 분포로 선택합니다.
    """

    rng = rng or np.random.default_rng()
    params: Dict[str, object] = {}
    for name, spec in space.items():
        requires = spec.get("requires")
        if requires and not _requirements_met(params, requires):
            if "default" in spec:
                params[name] = spec["default"]
            continue

        dtype = spec["type"]
        if dtype == "int":
            low = int(spec["min"])
            high = int(spec["max"])
            step = int(spec.get("step", 1)) or 1
            choices = list(range(low, high + 1, step))
            if not choices:
                raise ValueError(f"정수 파라미터 '{name}' 의 범위가 비어있습니다.")
            params[name] = int(rng.choice(choices))
        elif dtype == "float":
            low = float(spec["min"])
            high = float(spec["max"])
            step = float(spec.get("step", 0.0))
            if step:
                count = int(np.floor((high - low) / step)) + 1
                choices = [low + idx * step for idx in range(count)]
                params[name] = float(rng.choice(choices))
            else:
                params[name] = float(rng.uniform(low, high))
        elif dtype == "bool":
            if name == "useMomFade" and params.get("exitOpposite") is False:
                params[name] = True
            elif name == "exitOpposite" and params.get("useMomFade") is False:
                params[name] = True
            else:
                params[name] = bool(rng.integers(0, 2))
        elif dtype in {"choice", "str", "string"}:
            values = spec.get("values") or spec.get("options") or spec.get("choices")
            if not values:
                raise ValueError(
                    f"Choice 파라미터 '{name}' 은 'values' 목록이 필요합니다."
                )
            params[name] = rng.choice(list(values))
        else:
            raise ValueError(f"Unsupported parameter type: {dtype}")

    return _normalize_exit_flags(params)


def grid_choices(space: SpaceSpec) -> Dict[str, List[object]]:
    grid: Dict[str, List[object]] = {}
    for name, spec in space.items():
        dtype = spec["type"]
        if dtype == "int":
            step = int(spec.get("step", 1))
            if step <= 0:
                raise ValueError(
                    f"int 파라미터 '{name}'는 grid 샘플링을 위해 양수 step이 필요합니다."
                )
            grid[name] = list(range(int(spec["min"]), int(spec["max"]) + 1, step))
        elif dtype == "float":
            step = float(spec.get("step", 0.1))
            if step <= 0:
                raise ValueError(
                    f"float 파라미터 '{name}'는 grid 샘플링을 위해 양수 step이 필요합니다."
                )
            values = np.arange(float(spec["min"]), float(spec["max"]) + 1e-12, step)
            grid[name] = [round(val, 10) for val in values.tolist()]
        elif dtype == "bool":
            grid[name] = [True, False]
        elif dtype in {"choice", "str", "string"}:
            values = spec.get("values") or spec.get("options") or spec.get("choices")
            if not values:
                raise ValueError(f"Choice parameter '{name}' requires a non-empty 'values' list for grid sampling.")
            grid[name] = list(values)
        else:
            raise ValueError(f"Unsupported parameter type for grid: {dtype}")
    return grid


def mutate_around(
    params: Dict[str, object],
    space: SpaceSpec,
    scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    rng = rng or np.random.default_rng()
    mutated = dict(params)
    for name, spec in space.items():
        if name not in params:
            continue
        dtype = spec["type"]
        value = params[name]
        if dtype == "int":
            width = max(int((spec["max"] - spec["min"]) * scale), int(spec.get("step", 1)))
            low = int(spec["min"])
            high = int(spec["max"])
            step = int(spec.get("step", 1)) or 1
            jitter = rng.integers(-width, width + 1)
            candidate = int(value) + jitter
            offset = candidate - low
            candidate = low + int(round(offset / step)) * step
            candidate = max(low, min(high, candidate))
            mutated[name] = candidate
        elif dtype == "float":
            span = float(spec["max"] - spec["min"]) * scale
            low = float(spec["min"])
            high = float(spec["max"])
            step = float(spec.get("step", 0.0))
            jitter = rng.normal(0.0, span or 1e-6)
            candidate = float(value) + jitter
            offset = candidate - low
            if step:
                candidate = low + round(offset / step) * step
            candidate = max(low, min(high, candidate))
            precision = int(spec.get("precision", 0)) if "precision" in spec else None
            if precision is not None and precision >= 0:
                candidate = round(candidate, precision)
                candidate = max(low, min(high, candidate))
            mutated[name] = float(candidate)
        elif dtype in {"bool", "choice", "str", "string"}:
            if rng.random() < 0.2:
                # Flip bool or pick another categorical option.
                if dtype == "bool":
                    mutated[name] = not bool(value)
                else:
                    values = list(
                        spec.get("values") or spec.get("options") or spec.get("choices") or []
                    )
                    if values:
                        choices = [option for option in values if option != value]
                        if choices:
                            mutated[name] = rng.choice(choices)
        else:
            continue
    exit_flag = mutated.get("exitOpposite")
    mom_fade_flag = mutated.get("useMomFade")
    if exit_flag is False and mom_fade_flag is False:
        mutated["exitOpposite"] = True
    return mutated


def get_search_spaces() -> List[Dict[str, object]]:
    """Optuna 탐색 공간 프리셋을 반환합니다."""

    spaces: List[Dict[str, object]] = [
        {
            "key": "sun_deluxe_core",
            "label": "쑨모멘텀 디럭스 – 핵심 파라미터",
            "space": build_space(
                {
                    "oscLen": {"type": "int", "min": 10, "max": 40, "step": 2},
                    "signalLen": {"type": "int", "min": 2, "max": 12, "step": 1},
                    "bbLen": {"type": "int", "min": 10, "max": 40, "step": 2},
                    "bbMult": {"type": "float", "min": 1.0, "max": 3.0, "step": 0.1},
                    "kcLen": {"type": "int", "min": 10, "max": 30, "step": 2},
                    "kcMult": {"type": "float", "min": 1.0, "max": 2.5, "step": 0.1},
                    "fluxLen": {"type": "int", "min": 10, "max": 40, "step": 2},
                    "fluxSmoothLen": {"type": "int", "min": 1, "max": 7, "step": 1},
                    "fluxDeadzone": {"type": "float", "min": 0.0, "max": 40.0, "step": 1.0},
                    "useFluxHeikin": {"type": "bool"},
                    "useModFlux": {"type": "bool"},
                    # 모멘텀 스퀴즈 스타일: KC, AVG, Deluxe, Mod (TR1 정규화)
                    "basisStyle": {
                        "type": "choice",
                        "values": ["KC", "AVG", "Deluxe", "Mod"],
                    },
                    "compatMode": {"type": "bool"},
                    "autoThresholdScale": {"type": "bool"},
                    "useNormClip": {"type": "bool"},
                    "normClipLimit": {
                        "type": "float",
                        "min": 100.0,
                        "max": 600.0,
                        "step": 10.0,
                    },
                    "maType": {
                        "type": "choice",
                        "values": ["SMA", "EMA", "HMA"],
                    },
                    "useDynamicThresh": {"type": "bool"},
                    "useSymThreshold": {
                        "type": "bool",
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "statThreshold": {
                        "type": "float",
                        "min": 20.0,
                        "max": 80.0,
                        "step": 2.0,
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "buyThreshold": {
                        "type": "float",
                        "min": 20.0,
                        "max": 60.0,
                        "step": 2.0,
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "sellThreshold": {
                        "type": "float",
                        "min": 20.0,
                        "max": 60.0,
                        "step": 2.0,
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "dynLen": {
                        "type": "int",
                        "min": 15,
                        "max": 60,
                        "step": 5,
                        "requires": {"name": "useDynamicThresh", "equals": True},
                    },
                    "dynMult": {
                        "type": "float",
                        "min": 0.8,
                        "max": 2.0,
                        "step": 0.1,
                        "requires": {"name": "useDynamicThresh", "equals": True},
                    },
                    "exitOpposite": {"type": "bool"},
                    "useMomFade": {"type": "bool"},
                    "momFadeMinAbs": {
                        "type": "float",
                        "min": 0.0,
                        "max": 80.0,
                        "step": 1.0,
                        "requires": "useMomFade",
                    },
                }
            ),
        }
    ]

    return spaces
