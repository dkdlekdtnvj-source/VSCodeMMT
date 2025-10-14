"""LLM-assisted parameter suggestion helpers."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np
import optuna
import yaml

from optimize.search_spaces import SpaceSpec

LOGGER = logging.getLogger("optimize.llm")

DEFAULT_GEMINI_MODEL_PRIORITY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

try:  # pragma: no cover - optional dependency
    from google import genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None


@dataclass
class LLMSuggestions:
    """Gemini가 반환한 후보 파라미터와 전략 인사이트 번들을 보관합니다."""

    candidates: List[Dict[str, object]]
    insights: List[str]


def _trial_to_payload(trial: optuna.trial.FrozenTrial, priority: float) -> Dict[str, object]:
    """Gemini 프롬프트에 포함할 트라이얼 요약 정보를 생성합니다."""

    entry: Dict[str, object] = {
        "number": trial.number,
        "value": priority if math.isfinite(priority) else None,
        "params": trial.params,
    }
    values = _serialise_objective_values(_trial_objective_values(trial))
    if values is not None:
        entry["values"] = values
    attrs = getattr(trial, "user_attrs", None)
    if isinstance(attrs, dict):
        meta: Dict[str, object] = {}
        total_assets_attr = _coerce_float(attrs.get("total_assets"))
        if total_assets_attr is not None:
            meta["total_assets"] = total_assets_attr
        liquidation_attr = _coerce_float(attrs.get("liquidations"))
        if liquidation_attr is not None:
            meta["liquidations"] = liquidation_attr
        leverage_attr = _coerce_float(attrs.get("leverage"))
        if leverage_attr is not None:
            meta["leverage"] = leverage_attr
        ltf_choice_attr = attrs.get("ltf_choice")
        if isinstance(ltf_choice_attr, (list, tuple)):
            sequence = [str(item) for item in ltf_choice_attr if str(item)]
            if sequence:
                meta["ltf_choice"] = sequence
        elif ltf_choice_attr not in {None, ""}:
            meta["ltf_choice"] = ltf_choice_attr
        metrics_attr = attrs.get("metrics")
        if isinstance(metrics_attr, dict):
            metrics_subset: Dict[str, object] = {}
            for key in ("TotalAssets", "MaxDD", "Trades", "WinRate", "Liquidations"):
                numeric = _coerce_float(metrics_attr.get(key))
                if numeric is not None:
                    metrics_subset[key] = numeric
            if metrics_subset:
                meta["metrics"] = metrics_subset
        if meta:
            entry["meta"] = meta
    return entry


def _extract_text(response: object) -> str:
    if response is None:
        return ""
    parts: List[str] = []
    text = getattr(response, "text", None)
    if isinstance(text, str):
        parts.append(text)
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            content_parts = getattr(content, "parts", None) if content else None
            if content_parts:
                for part in content_parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str):
                        parts.append(part_text)
    return "\n".join(parts).strip()


def _normalise_gemini_object(value: object, depth: int = 0) -> object:
    """Gemini SDK 객체를 JSON 직렬화 가능한 자료형으로 재구성합니다."""

    if depth > 5:
        # 너무 깊은 중첩은 문자열로 요약해 무한 재귀를 방지합니다.
        text = str(value)
        return text[:500] if isinstance(text, str) else text
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _normalise_gemini_object(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [
            _normalise_gemini_object(item, depth + 1)
            for item in value
        ]
    for attr in ("to_dict", "model_dump", "as_dict"):
        converter = getattr(value, attr, None)
        if callable(converter):
            try:
                converted = converter()
            except Exception:
                continue
            return _normalise_gemini_object(converted, depth + 1)
    if hasattr(value, "__dict__"):
        try:
            return {
                key: _normalise_gemini_object(val, depth + 1)
                for key, val in value.__dict__.items()
                if not key.startswith("_")
            }
        except Exception:
            pass
    text = str(value)
    return text[:500] if len(text) > 500 else text


def _extract_payload_from_parts(parts: Sequence[object]) -> Optional[Dict[str, Any]]:
    for part in parts:
        part_dict = _normalise_gemini_object(part)
        if not isinstance(part_dict, dict):
            continue
        for key in ("functionCall", "function_call", "toolCall", "tool_call"):
            call = part_dict.get(key)
            if isinstance(call, dict):
                args = call.get("args") or call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        try:
                            args = yaml.safe_load(args)
                        except Exception:
                            continue
                if isinstance(args, dict) and args:
                    return args
        text_value = part_dict.get("text")
        if isinstance(text_value, str):
            payload = _extract_json_payload(text_value)
            if isinstance(payload, dict):
                return payload
    return None


def _extract_structured_payload(response: object) -> Optional[Dict[str, Any]]:
    """Gemini 응답 객체에서 functionCall/JSON 블록을 우선적으로 추출합니다."""

    normalised = _normalise_gemini_object(response)
    if not isinstance(normalised, dict):
        return None

    candidates = normalised.get("candidates")
    if isinstance(candidates, Sequence):
        for candidate in candidates:
            candidate_dict = _normalise_gemini_object(candidate)
            if not isinstance(candidate_dict, dict):
                continue
            parts: List[object] = []
            content = candidate_dict.get("content")
            if isinstance(content, dict):
                inner_parts = content.get("parts")
                if isinstance(inner_parts, Sequence):
                    parts.extend(inner_parts)
            candidate_parts = candidate_dict.get("parts")
            if isinstance(candidate_parts, Sequence):
                parts.extend(candidate_parts)
            if parts:
                payload = _extract_payload_from_parts(parts)
                if isinstance(payload, dict):
                    return payload
    return None


def _summarise_for_log(value: object, limit: int = 600) -> str:
    """로그용으로 Gemini 응답을 한 줄 요약합니다."""

    normalised = _normalise_gemini_object(value)
    try:
        summary = json.dumps(normalised, ensure_ascii=False)
    except Exception:
        summary = str(normalised)
    summary = summary.replace("\n", " ").strip()
    if len(summary) > limit:
        return summary[:limit] + "…"
    return summary


def _extract_json_payload(raw: str) -> Optional[object]:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop opening fence
        lines = lines[1:]
        # drop closing fence if present
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = None
    end = None
    for token in ("[", "{"):
        idx = cleaned.find(token)
        if idx != -1 and (start is None or idx < start):
            start = idx
    for token in ("]", "}"):
        idx = cleaned.rfind(token)
        if idx != -1 and (end is None or idx > end):
            end = idx
    if start is not None and end is not None and start < end:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            parsed = yaml.safe_load(cleaned)
        except Exception:
            return None
        else:
            return parsed


def _iter_balanced_segments(raw: str, opener: str, closer: str) -> Iterable[str]:
    depth = 0
    start = -1
    for index, char in enumerate(raw):
        if char == opener:
            if depth == 0:
                start = index
            depth += 1
        elif char == closer and depth:
            depth -= 1
            if depth == 0 and start != -1:
                yield raw[start : index + 1]
                start = -1


def _flatten_structured_payload(payload: object, space: SpaceSpec) -> Iterable[Dict[str, object]]:
    if isinstance(payload, dict):
        keys = set(space.keys())
        if keys and keys.issubset(payload.keys()):
            non_mapping_values = sum(
                1 for name in keys if not isinstance(payload.get(name), dict)
            )
            if non_mapping_values:
                yield payload
        for candidate_key in ("candidates", "Candidate", "params", "parameters"):
            nested = payload.get(candidate_key)
            if isinstance(nested, dict):
                yield from _flatten_structured_payload(nested, space)
            elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                for entry in nested:
                    if isinstance(entry, dict):
                        yield from _flatten_structured_payload(entry, space)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for entry in payload:
            if isinstance(entry, dict):
                yield from _flatten_structured_payload(entry, space)


def _looks_like_boundary(segment: str) -> bool:
    if not segment:
        return False
    if ",," in segment:
        return True
    if segment.count("\n") >= 2:
        return True
    if any(token in segment for token in "]}"):
        return True
    if len(segment.strip()) == 0:
        return False
    return len(segment.strip()) > 120


def _scan_key_value_candidates(raw: str, space: SpaceSpec) -> List[Dict[str, str]]:
    if not raw or not space:
        return []
    param_names = sorted(space.keys(), key=len, reverse=True)
    if not param_names:
        return []
    name_pattern = "|".join(re.escape(name) for name in param_names)
    pair_pattern = re.compile(
        rf"(?P<name>{name_pattern})\s*(?:[:=]|=>|->|→)?\s*(?P<value>[^,\n\r;]+)", re.IGNORECASE
    )
    candidates: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    last_end: Optional[int] = None
    for match in pair_pattern.finditer(raw):
        name = match.group("name")
        value = match.group("value").strip()
        value = value.strip("\"' ")
        value = value.rstrip("]})")
        value = value.rstrip(",; ")
        if current and (name in current or (last_end is not None and _looks_like_boundary(raw[last_end: match.start()]))):
            candidates.append(current)
            current = {}
        current[name] = value
        last_end = match.end()
    if current:
        candidates.append(current)
    return candidates


def _recent_parameter_trends(
    trials: Sequence[optuna.trial.FrozenTrial],
    space: SpaceSpec,
    config: Dict[str, object],
) -> List[str]:
    """최근 트라이얼에서 관측된 파라미터 추세 요약을 생성합니다."""

    window_raw = config.get("trend_window", 5)
    try:
        window = max(3, int(window_raw))
    except (TypeError, ValueError):  # pragma: no cover - 잘못된 사용자 입력 대비
        window = 5

    recent_trials = sorted(trials, key=lambda trial: trial.number, reverse=True)[:window]
    if len(recent_trials) < 3:
        return []

    trend_messages: List[str] = []
    for name in space.keys():
        series: List[Tuple[float, float]] = []
        for trial in recent_trials:
            value = trial.params.get(name)
            if not isinstance(value, (int, float)):
                continue
            score = _trial_priority(trial, config)
            if not math.isfinite(score):
                continue
            series.append((float(value), float(score)))
        if len(series) < 3:
            continue
        values = np.array([pair[0] for pair in series], dtype=float)
        scores = np.array([pair[1] for pair in series], dtype=float)
        if np.allclose(values, values[0]) or np.std(scores) == 0.0:
            continue
        with np.errstate(all="ignore"):
            corr = np.corrcoef(values, scores)[0, 1]
        if not np.isfinite(corr):
            continue
        if abs(corr) < 0.3:
            continue
        direction = "increasing" if corr > 0 else "decreasing"
        impact = "improved" if corr > 0 else "weakened"
        trend_messages.append(
            f"{name}: values {direction} tended to yield {impact} scores over the last {len(series)} trials (corr={corr:.2f})."
        )

    return trend_messages


def _extract_candidates_from_text(raw: str, space: SpaceSpec, limit: int) -> List[Dict[str, object]]:
    if not raw or not space:
        return []

    validated: List[Dict[str, object]] = []
    seen: Set[Tuple[Tuple[str, object], ...]] = set()

    def _register(candidate: Dict[str, object]) -> None:
        signature = tuple(sorted(candidate.items()))
        if signature in seen:
            return
        seen.add(signature)
        validated.append(candidate)

    structured_blocks: List[object] = []
    for segment in _iter_balanced_segments(raw, "[", "]"):
        structured_blocks.append(segment)
    for segment in _iter_balanced_segments(raw, "{", "}"):
        structured_blocks.append(segment)

    for block in structured_blocks:
        text_block = block if isinstance(block, str) else str(block)
        text_block = text_block.strip()
        if not text_block:
            continue
        for loader in (json.loads, yaml.safe_load):
            try:
                parsed = loader(text_block)
            except Exception:
                continue
            if parsed is None:
                continue
            for entry in _flatten_structured_payload(parsed, space):
                checked = _validate_candidate(entry, space, require_complete=True)
                if checked:
                    _register(checked)
                    if len(validated) >= limit:
                        return validated
            break

    for kv_candidate in _scan_key_value_candidates(raw, space):
        checked = _validate_candidate(kv_candidate, space, require_complete=True)
        if checked:
            _register(checked)
            if len(validated) >= limit:
                break

    return validated


def _coerce_numeric(value: object, *, to_int: bool = False) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if to_int:
        return float(int(round(numeric)))
    return numeric


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _rankdata(values: Sequence[float]) -> List[float]:
    """Compute rank data with average ranks for ties."""

    enumerated = sorted(((float(v), idx) for idx, v in enumerate(values)), key=lambda item: item[0])
    n = len(enumerated)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        value = enumerated[i][0]
        while j < n and enumerated[j][0] == value:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[enumerated[k][1]] = avg_rank
        i = j
    return ranks


def _spearmanr_safe(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Calculate Spearman correlation with graceful degradation."""

    if len(x) != len(y) or len(x) < 2:
        return None
    rank_x = _rankdata(x)
    rank_y = _rankdata(y)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for rx, ry in zip(rank_x, rank_y):
        dx = rx - mean_x
        dy = ry - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return num / math.sqrt(denom_x * denom_y)


def _compute_param_statistics(
    trials: Sequence[optuna.trial.FrozenTrial],
    scores: Sequence[float],
) -> Dict[str, object]:
    """Derive correlation-style statistics for parameters across trials."""

    if not trials or not scores:
        return {}
    per_param: Dict[str, List[Tuple[object, float]]] = {}
    for trial, score in zip(trials, scores):
        if not math.isfinite(score):
            continue
        for name, value in trial.params.items():
            per_param.setdefault(name, []).append((value, float(score)))
    analytics: Dict[str, object] = {}
    for name, values in per_param.items():
        if len(values) < 3:
            continue
        raw_values = [item[0] for item in values]
        score_values = [item[1] for item in values]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_values):
            numeric_values = [float(v) for v in raw_values]
            correlation = _spearmanr_safe(numeric_values, score_values)
            pair_array = sorted(zip(score_values, numeric_values), key=lambda item: item[0])
            quartile = max(int(len(pair_array) * 0.25), 1)
            bottom_slice = pair_array[:quartile]
            top_slice = pair_array[-quartile:]
            top_scores = [item[0] for item in top_slice]
            analytics[name] = {
                "count": len(pair_array),
                "spearman": None if correlation is None else round(float(correlation), 6),
                "top_quartile_value_mean": round(float(np.mean([item[1] for item in top_slice])), 6),
                "top_quartile_score_mean": round(float(np.mean(top_scores)), 6),
                "bottom_quartile_value_mean": round(float(np.mean([item[1] for item in bottom_slice])), 6),
                "bottom_quartile_score_mean": round(float(np.mean([item[0] for item in bottom_slice])), 6),
            }
        else:
            grouped: Dict[str, List[float]] = {}
            for value, score in values:
                key = str(value)
                grouped.setdefault(key, []).append(score)
            averaged = [
                (label, float(np.mean(scores)), len(scores))
                for label, scores in grouped.items()
            ]
            averaged.sort(key=lambda item: item[1], reverse=True)
            analytics[name] = {
                "count": sum(item[2] for item in averaged),
                "category_scores": [
                    {"choice": label, "mean_score": round(score, 6), "observations": obs}
                    for label, score, obs in averaged[:6]
                ],
            }
    return analytics


def _load_env_file_variables(path: Path) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return variables
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_quotes(value.strip())
        variables[key] = value
    return variables


def _load_gemini_api_key(config: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the Gemini API key from config, env vars or helper files.

    Returns a tuple of (api_key, source_description).
    """

    # 1) Direct inline key in the YAML config.
    direct = config.get("api_key") if config else None
    if isinstance(direct, str) and direct.strip():
        return direct.strip(), "llm.api_key"

    # 2) Optional file path (api_key_file / api_key_path).
    file_field = None
    if isinstance(config, dict):
        file_field = config.get("api_key_file") or config.get("api_key_path")
    if isinstance(file_field, str) and file_field.strip():
        candidate = Path(file_field).expanduser()
        try:
            content = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            LOGGER.debug("Gemini API 키 파일 %s을(를) 열 수 없습니다.", candidate)
        else:
            if content:
                return content, f"file:{candidate}"

    # 3) Environment variable lookup (default GEMINI_API_KEY).
    env_name = str(config.get("api_key_env", "GEMINI_API_KEY")) if config else "GEMINI_API_KEY"
    env_value = os.environ.get(env_name)
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip(), f"env:{env_name}"

    # 4) Look for .env style files in common locations.
    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    env_candidates = [
        cwd / ".env",
        repo_root / ".env",
        repo_root / "config" / ".env",
    ]
    seen: Set[Path] = set()
    for candidate in env_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        variables = _load_env_file_variables(candidate)
        value = variables.get(env_name)
        if value and value.strip():
            return value.strip(), f"env_file:{candidate}"

    return None, None


def _validate_candidate(
    candidate: Dict[str, object], space: SpaceSpec, *, require_complete: bool = False
) -> Optional[Dict[str, object]]:
    validated: Dict[str, object] = {}
    missing: List[str] = []
    for name, spec in space.items():
        if name not in candidate:
            missing.append(name)
            continue
        value = candidate[name]
        dtype = spec["type"]
        if dtype == "int":
            numeric = _coerce_numeric(value, to_int=True)
            if numeric is None:
                return None
            low = int(spec["min"])
            high = int(spec["max"])
            step = int(spec.get("step", 1))
            as_int = int(numeric)
            if as_int < low or as_int > high:
                return None
            if step:
                offset = as_int - low
                as_int = low + round(offset / step) * step
                as_int = max(low, min(high, as_int))
            validated[name] = int(as_int)
        elif dtype == "float":
            numeric = _coerce_numeric(value)
            if numeric is None:
                return None
            low = float(spec["min"])
            high = float(spec["max"])
            if numeric < low or numeric > high:
                return None
            step = float(spec.get("step", 0.0))
            if step:
                offset = numeric - low
                numeric = low + round(offset / step) * step
                numeric = max(low, min(high, numeric))
            validated[name] = float(numeric)
        elif dtype == "bool":
            if isinstance(value, str):
                normalised = value.strip().lower()
                validated[name] = normalised in {"true", "1", "yes", "on"}
            else:
                validated[name] = bool(value)
        elif dtype in {"choice", "str", "string"}:
            values = list(
                spec.get("values") or spec.get("options") or spec.get("choices") or []
            )
            if not values:
                return None
            if value in values:
                validated[name] = value
            else:
                normalised = str(value).strip().lower()
                match = next((option for option in values if str(option).strip().lower() == normalised), None)
                if match is None:
                    return None
                validated[name] = match
        else:
            # Unsupported type for LLM suggestions.
            return None
    if require_complete and missing:
        LOGGER.debug(
            "Gemini 후보에서 YAML 파라미터가 누락되어 제외합니다: %s",
            ", ".join(sorted(missing)),
        )
        return None
    if missing:
        LOGGER.debug(
            "Gemini 후보에 YAML 파라미터 일부가 빠져 있습니다: %s",
            ", ".join(sorted(missing)),
        )
    return validated


def _coerce_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _trial_objective_values(trial: optuna.trial.FrozenTrial) -> Optional[List[object]]:
    raw_values = getattr(trial, "values", None)
    if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
        return list(raw_values)
    raw_value = getattr(trial, "value", None)
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return list(raw_value)
    return None


def _trial_priority(
    trial: optuna.trial.FrozenTrial, config: Dict[str, object]
) -> float:
    direct = _coerce_float(getattr(trial, "value", None))
    if direct is not None:
        return direct

    values = _trial_objective_values(trial)
    if not values:
        return float("-inf")

    # 멀티 목적 최적화 시 우선순위로 사용할 지표 인덱스를 명시적으로 지정할 수 있습니다.
    index = config.get("objective_index")
    if isinstance(index, int) and 0 <= index < len(values):
        indexed = _coerce_float(values[index])
        if indexed is not None:
            return indexed

    for candidate in values:
        numeric = _coerce_float(candidate)
        if numeric is not None:
            return numeric
    return float("-inf")


def _serialise_objective_values(values: Optional[List[object]]) -> Optional[List[object]]:
    if not values:
        return None
    serialised: List[object] = []
    for value in values:
        numeric = _coerce_float(value)
        serialised.append(numeric if numeric is not None else value)
    return serialised


def _has_objective_values(trial: optuna.trial.FrozenTrial) -> bool:
    value = getattr(trial, "value", None)
    if value is not None:
        return True
    values = _trial_objective_values(trial)
    if not values:
        return False
    return any(item is not None for item in values)


def generate_llm_candidates(
    space: SpaceSpec,
    trials: Iterable[optuna.trial.FrozenTrial],
    config: Dict[str, object],
) -> LLMSuggestions:
    if not config or not config.get("enabled"):
        return LLMSuggestions([], [])
    if genai is None:
        LOGGER.warning("google-genai is not installed; skipping Gemini-guided proposals.")
        return LLMSuggestions([], [])

    api_key, api_key_source = _load_gemini_api_key(config)
    if not api_key:
        LOGGER.warning("Gemini API 키가 설정되지 않아 LLM 제안을 건너뜁니다.")
        return LLMSuggestions([], [])
    if api_key_source:
        LOGGER.debug("Gemini API 키를 %s에서 로드했습니다.", api_key_source)

    finished_trials: List[optuna.trial.FrozenTrial] = [
        trial
        for trial in trials
        if trial.state.is_finished() and _has_objective_values(trial)
    ]
    if not finished_trials:
        LOGGER.info("아직 완료된 트라이얼이 없어 LLM 제안을 생략합니다.")
        return LLMSuggestions([], [])

    top_n = max(int(config.get("top_n", 10)), 1)
    count = max(int(config.get("count", 8)), 1)
    sorted_trials = sorted(
        finished_trials,
        key=lambda trial: _trial_priority(trial, config),
        reverse=True,
    )
    top_trials: List[Dict[str, object]] = []
    for trial in sorted_trials[:top_n]:
        priority = _trial_priority(trial, config)
        top_trials.append(_trial_to_payload(trial, priority))

    bottom_trials: List[Dict[str, object]] = []
    bottom_n = max(int(config.get("bottom_n", top_n)), 1)
    if len(sorted_trials) > 1:
        seen_numbers = {entry["number"] for entry in top_trials}
        for trial in reversed(sorted_trials[-bottom_n:]):
            if trial.number in seen_numbers:
                continue
            priority = _trial_priority(trial, config)
            bottom_trials.append(_trial_to_payload(trial, priority))

    param_importances: Dict[str, float] = {}
    if finished_trials:
        representative = finished_trials[0]
        is_multi_objective = bool(getattr(representative, "values", None)) and (
            isinstance(representative.values, Sequence) and len(representative.values) > 1
        )
        if not is_multi_objective:
            try:
                importance_study = optuna.create_study(direction="maximize")
                importance_study.add_trials(finished_trials)
                importances = optuna.importance.get_param_importances(importance_study)
            except Exception as exc:  # pragma: no cover - fallback path
                LOGGER.debug("Gemini 파라미터 중요도 계산 실패: %s", exc)
            else:
                param_importances = {name: float(value) for name, value in importances.items()}

    param_summaries: Dict[str, object] = {}
    summary_trials = sorted_trials[:top_n]
    summary_scores = [_trial_priority(trial, config) for trial in summary_trials]
    for name in space.keys():
        observed = [trial.params.get(name) for trial in summary_trials if name in trial.params]
        if not observed:
            continue
        if all(isinstance(value, bool) for value in observed):
            counts = Counter(bool(value) for value in observed)
            param_summaries[name] = {"most_common": [item for item, _ in counts.most_common(3)]}
            continue
        if all(isinstance(value, (int, float)) for value in observed):
            numeric_values = [float(value) for value in observed]
            param_summaries[name] = {
                "min": round(min(numeric_values), 6),
                "median": round(statistics.median(numeric_values), 6),
                "max": round(max(numeric_values), 6),
            }
            continue
        counts = Counter(observed)
        param_summaries[name] = {"most_common": [item for item, _ in counts.most_common(3)]}

    param_statistics = _compute_param_statistics(summary_trials, summary_scores)
    trend_summaries = _recent_parameter_trends(finished_trials, space, config)

    client = genai.Client(api_key=api_key)
    configured_model = config.get("model")
    if configured_model in {None, ""}:
        model = DEFAULT_GEMINI_MODEL_PRIORITY[0]
    else:
        model = str(configured_model)
    prompt_sections = [
        "You are assisting with hyper-parameter optimisation for a trading strategy.",
        "The search space is defined by the following JSON (types: int, float, bool, choice):",
        json.dumps(space, indent=2),
        "Here are the top completed trials with their objective values (higher is better):",
        json.dumps(top_trials, indent=2),
        "Focus on maximising TotalAssets (available capital + savings) while keeping drawdown and liquidation counts low.",
        "Leverage and timeframe selections are fixed per trial; honour any multi-timeframe combinations when proposing candidates.",
    ]
    if bottom_trials:
        prompt_sections.append(
            "For contrast, these are the weakest-performing trials (avoid repeating their pitfalls):"
        )
        prompt_sections.append(json.dumps(bottom_trials, indent=2))
    if param_importances:
        prompt_sections.append(
            "Optuna parameter importances (descending order, sum≈1.0):"
        )
        prompt_sections.append(json.dumps(param_importances, indent=2))
    if param_summaries:
        prompt_sections.append(
            "Parameter value summaries across the top trials (min/median/max or most common choices):"
        )
        prompt_sections.append(json.dumps(param_summaries, indent=2))
    if param_statistics:
        prompt_sections.append(
            "Parameter-performance analytics highlighting correlations and top/bottom quartiles:"
        )
        prompt_sections.append(json.dumps(param_statistics, indent=2))
    if trend_summaries:
        prompt_sections.append(
            "Recent trial trends hinting at momentum in parameter adjustments (correlation-based):"
        )
        prompt_sections.append("\n".join(f"- {message}" for message in trend_summaries))
    prompt_sections.append(
        f"Propose {count} new parameter sets strictly within the given bounds as a JSON array under the key 'candidates'."
        " Also return 2-3 short tactical insights or strategy adjustments derived from the trials as a string array under the key 'insights'."
        " Respond with a single JSON object containing both 'candidates' and 'insights'."
    )
    prompt = "\n\n".join(prompt_sections)

    # Optional thinking budget / generation config support for Gemini 2.5+ models.
    generation_config = None
    thinking_budget = config.get("thinking_budget")
    if thinking_budget is not None:
        try:
            budget_int = int(thinking_budget)
            try:
                GenerationConfig = getattr(genai, "types", None)
                if GenerationConfig is not None:
                    types_module = GenerationConfig
                    thinking_cls = getattr(types_module, "ThinkingConfig", None)
                    gen_cfg_cls = getattr(types_module, "GenerationConfig", None)
                    if thinking_cls is not None and gen_cfg_cls is not None:
                        generation_config = gen_cfg_cls(
                            thinking_config=thinking_cls(thinking_budget=budget_int)
                        )
            except Exception:
                generation_config = None
        except Exception:
            generation_config = None

    prompt_contents = [{"role": "user", "parts": [{"text": prompt}]}]
    request_kwargs: Dict[str, object] = {"model": model, "contents": prompt_contents}
    if generation_config is not None:
        request_kwargs["generation_config"] = generation_config

    system_instruction = config.get("system_instruction")
    if isinstance(system_instruction, str) and system_instruction.strip():
        request_kwargs["system_instruction"] = system_instruction.strip()

    candidate_count = config.get("candidate_count")
    if candidate_count is not None:
        try:
            request_kwargs["candidate_count"] = max(1, int(candidate_count))
        except (TypeError, ValueError):
            LOGGER.warning(
                "Gemini candidate_count 설정 '%s' 을 정수로 변환하지 못했습니다.",
                candidate_count,
            )

    def _invoke_model(kwargs: Dict[str, object]):
        attempt = dict(kwargs)
        removed: Set[str] = set()
        while True:
            try:
                return client.models.generate_content(**attempt)
            except TypeError as exc:  # pragma: no cover - SDK compatibility path
                message = str(exc)
                stripped = message.strip()
                removed_key = False
                for key in list(attempt.keys()):
                    if key in {"model", "contents"} or key in removed:
                        continue
                    if f"'{key}'" in stripped or f" {key}" in stripped:
                        LOGGER.debug(
                            "Gemini generate_content 인자 '%s' 를 제거하고 재시도합니다: %s",
                            key,
                            exc,
                        )
                        attempt.pop(key, None)
                        removed.add(key)
                        removed_key = True
                if not removed_key:
                    raise

    def _should_switch_model(exc: Exception) -> bool:
        text = str(exc).lower()
        keywords = (
            "permission",
            "denied",
            "forbidden",
            "blocked",
            "not found",
            "does not exist",
            "unsupported",
            "unavailable",
            "quota",
        )
        return any(token in text for token in keywords)

    fallback_models_cfg = config.get("fallback_models")
    fallback_models: List[str] = []
    if isinstance(fallback_models_cfg, Sequence) and not isinstance(
        fallback_models_cfg, (str, bytes, bytearray)
    ):
        for entry in fallback_models_cfg:
            candidate = str(entry or "").strip()
            if candidate:
                fallback_models.append(candidate)

    model_candidates: List[str] = []

    def _register_model(candidate: object) -> None:
        candidate_str = str(candidate or "").strip()
        if candidate_str and candidate_str not in model_candidates:
            model_candidates.append(candidate_str)

    _register_model(model)
    for candidate in fallback_models:
        _register_model(candidate)
    for default_model in DEFAULT_GEMINI_MODEL_PRIORITY:
        _register_model(default_model)

    endpoint = None
    for attr in ("api_endpoint", "_api_endpoint", "_endpoint", "_host"):
        endpoint = getattr(client, attr, None)
        if endpoint:
            break
    endpoint_desc = str(endpoint) if endpoint else client.__class__.__name__

    response = None
    last_error: Optional[Exception] = None
    for idx, candidate_model in enumerate(model_candidates):
        attempt_kwargs = dict(request_kwargs)
        attempt_kwargs["model"] = candidate_model
        try:  # pragma: no cover - network side effects
            LOGGER.debug(
                "Gemini 모델 '%s' 호출 (엔드포인트=%s)",
                candidate_model,
                endpoint_desc,
            )
            response = _invoke_model(attempt_kwargs)
            break
        except Exception as exc:  # pragma: no cover - network side effects
            last_error = exc
            if idx < len(model_candidates) - 1 and _should_switch_model(exc):
                next_model = model_candidates[idx + 1]
                LOGGER.warning(
                    "Gemini 모델 '%s' 호출이 거부되어 '%s'(으)로 재시도합니다: %s",
                    candidate_model,
                    next_model,
                    exc,
                )
                continue
            LOGGER.warning("Gemini 호출에 실패했습니다: %s", exc)
            return LLMSuggestions([], [])

    if response is None:
        LOGGER.warning("Gemini 호출에 실패했습니다: %s", last_error)
        return LLMSuggestions([], [])

    structured_payload = _extract_structured_payload(response)
    if structured_payload:
        LOGGER.debug(
            "Gemini functionCall/도구 응답을 파싱했습니다: %s",
            _summarise_for_log(structured_payload),
        )
    raw_text = _extract_text(response)
    payload = (
        structured_payload if structured_payload is not None else _extract_json_payload(raw_text)
    )

    insights: List[str] = []
    accepted: List[Dict[str, object]] = []
    candidate_payload = payload
    if isinstance(payload, dict):
        candidate_payload = payload.get("candidates")
        raw_insights = payload.get("insights")
        if isinstance(raw_insights, str):
            text = raw_insights.strip()
            if text:
                insights.append(text)
        elif isinstance(raw_insights, Sequence):
            for entry in raw_insights:
                if isinstance(entry, str):
                    text = entry.strip()
                    if text:
                        insights.append(text)
        if not insights:
            nested_insights = payload.get("insights")
            if isinstance(nested_insights, dict):
                for value in nested_insights.values():
                    if isinstance(value, Sequence) and not isinstance(
                        value, (str, bytes, bytearray)
                    ):
                        for entry in value:
                            if isinstance(entry, str):
                                text = entry.strip()
                                if text:
                                    insights.append(text)
        if candidate_payload is None:
            for entry in _flatten_structured_payload(payload, space):
                validated = _validate_candidate(entry, space, require_complete=True)
                if validated:
                    accepted.append(validated)
                    if len(accepted) >= count:
                        break
            if accepted:
                candidate_payload = []  # 이미 accepted에 반영했음을 표시

    if isinstance(candidate_payload, list):
        for entry in candidate_payload:
            if not isinstance(entry, dict):
                continue
            validated = _validate_candidate(entry, space)
            if not validated:
                continue
            accepted.append(validated)
            if len(accepted) >= count:
                break

    if len(accepted) < count:
        fallback_candidates = _extract_candidates_from_text(
            raw_text, space, count - len(accepted)
        )
        if fallback_candidates:
            LOGGER.info(
                "Gemini 자유 형식 응답에서 %d개 후보를 추출했습니다.",
                len(fallback_candidates),
            )
            accepted.extend(fallback_candidates[: count - len(accepted)])
        elif not accepted:
            LOGGER.warning(
                "Gemini 응답에서 후보 파라미터 배열을 찾지 못했습니다. 응답 요약: %s",
                _summarise_for_log(response if structured_payload is None else payload),
            )
            return LLMSuggestions([], insights)

    if accepted:
        LOGGER.info("Gemini가 제안한 %d개의 후보를 큐에 추가합니다.", len(accepted))
    else:
        LOGGER.info("Gemini 제안 중 조건을 만족하는 후보가 없었습니다.")
    if insights:
        LOGGER.info("Gemini 전략 인사이트 %d건 수신", len(insights))
    return LLMSuggestions(accepted, insights)
