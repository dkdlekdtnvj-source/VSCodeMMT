"""공통 최적화 유틸리티."""

from __future__ import annotations

from typing import Mapping, Any, Dict, Optional

import math

import numpy as np


def _coerce_float(value: object) -> float:
    """입력 값을 ``float`` 로 변환합니다."""

    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(int(value))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(result):
        return math.nan
    return result


def resolve_leverage(
    params: Mapping[str, Any] | None,
    risk: Mapping[str, Any] | None,
    *,
    default: float = 10.0,
) -> float:
    """파라미터와 리스크 설정에서 레버리지를 일관되게 해석합니다."""

    candidate = _coerce_float(params.get("leverage") if isinstance(params, Mapping) else None)
    if np.isfinite(candidate):
        return float(candidate)

    fallback = _coerce_float(risk.get("leverage") if isinstance(risk, Mapping) else None)
    if np.isfinite(fallback):
        return float(fallback)

    default_value = _coerce_float(default)
    if np.isfinite(default_value):
        return float(default_value)
    return 10.0


def normalize_tf(params: Mapping[str, Any]) -> Dict[str, Any]:
    """타임프레임 관련 파라미터를 단일 소스로 정규화합니다.

    ``chart_tf`` 는 데이터 로딩·백테스트 기준 타임프레임을, ``entry_tf`` 는
    진입 신호 계산에 사용할 LTF(lower timeframe)를 의미합니다. HTF 사용
    여부(``use_htf``)가 꺼져 있으면 ``htf_tf`` 는 ``"NA"`` 로 고정합니다.

    레거시 키(``ltf``, ``ltfChoice``, ``timeframe`` 등)는 입력으로 전달될 수
    있으므로 가능한 경우 ``entry_tf`` 로 승격하고, 내부 호환성을 위해
    동일한 값으로 다시 채워 줍니다. 호출자는 반환 값을 그대로 사용하거나
    추가 후처리를 수행할 수 있습니다.
    """

    canonical = dict(params)

    chart_tf = canonical.get("chart_tf") or canonical.get("timeframe") or "1m"
    canonical["chart_tf"] = str(chart_tf)

    canonical["use_htf"] = bool(canonical.get("use_htf", False))

    legacy_entry = canonical.pop("ltf", canonical.get("ltfChoice", canonical.get("timeframe")))
    legacy_candidate: Optional[str] = None
    if isinstance(legacy_entry, str):
        tokens = [token.strip() for token in legacy_entry.split(",") if token.strip()]
        for token in tokens:
            if token in {"1m", "3m", "5m"}:
                legacy_candidate = token
                break
    entry_tf = canonical.get("entry_tf")
    if not isinstance(entry_tf, str) or not entry_tf:
        if legacy_candidate is not None:
            entry_tf = legacy_candidate
        else:
            entry_tf = str(chart_tf)
    canonical["entry_tf"] = entry_tf

    if canonical["use_htf"]:
        htf_value = canonical.get("htf_tf") or canonical.get("htf_timeframe") or canonical.get("htf")
        canonical["htf_tf"] = str(htf_value) if isinstance(htf_value, str) and htf_value else "NA"
    else:
        canonical["htf_tf"] = "NA"

    # 레거시 키 유지(내부 호환용)
    canonical["timeframe"] = canonical.get("chart_tf")
    canonical["ltf"] = canonical["entry_tf"]
    canonical["ltfChoice"] = canonical["entry_tf"]

    return canonical
