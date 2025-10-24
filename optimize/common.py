"""Common optimisation helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import math

import numpy as np


def _coerce_float(value: object) -> float:
    """Return a finite float representation or NaN."""

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
    """Resolve leverage preference from params, risk, or a default fallback."""

    candidate = _coerce_float(params.get("leverage") if isinstance(params, Mapping) else None)
    if np.isfinite(candidate):
        return float(np.clip(candidate, 1.0, 30.0))

    fallback = _coerce_float(risk.get("leverage") if isinstance(risk, Mapping) else None)
    if np.isfinite(fallback):
        return float(np.clip(fallback, 1.0, 30.0))

    default_value = _coerce_float(default)
    if np.isfinite(default_value):
        return float(np.clip(default_value, 1.0, 30.0))
    return 10.0


def normalize_tf(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalise timeframe-related parameters.

    Ensures that chart_tf, entry_tf, and other timeframe aliases are all
    set to the primary timeframe for the run, eliminating any possibility of
    using a different chart timeframe for the backtest.
    """
    canonical: Dict[str, Any] = dict(params)

    # Determine the primary timeframe for this run.
    # The 'timeframe' parameter is the most reliable source.
    primary_tf = canonical.get("timeframe")
    if not isinstance(primary_tf, str) or not primary_tf.strip():
        # Fallback to other aliases if 'timeframe' is not present
        tf_candidates = [
            canonical.get("entry_tf"),
            canonical.get("ltf"),
            canonical.get("chart_tf"),
        ]
        primary_tf = next(
            (str(candidate).strip() for candidate in tf_candidates if isinstance(candidate, str) and candidate.strip()),
            "1m",  # Default to 1m if nothing is found
        )

    primary_tf = primary_tf.strip()

    # Set all timeframe-related keys to the same primary timeframe
    canonical["timeframe"] = primary_tf

    # Remove legacy multi-timeframe and HTF knobs to keep a single timeframe surface.
    for key in (
        "entry_tf",
        "chart_tf",
        "ltf",
        "ltfChoice",
        "ltf_choices",
        "use_htf",
        "htf",
        "htf_timeframe",
        "htf_tf",
    ):
        canonical.pop(key, None)

    return canonical


def _resolve_symbol_entry(
    symbol_value: str,
    alias_map: Optional[Dict[str, str]] = None,
) -> tuple[str, str]:
    """Resolve a symbol entry to its display and source names."""
    alias_map = alias_map or {}
    if not symbol_value:
        return "", ""
    if ":" in symbol_value:
        source, symbol = symbol_value.split(":", 1)
        display_symbol = f"{source.upper()}:{symbol.upper()}"
        source_symbol = alias_map.get(display_symbol, display_symbol)
        return display_symbol, source_symbol
    display_symbol = symbol_value.upper()
    source_symbol = alias_map.get(display_symbol, display_symbol)
    return display_symbol, source_symbol
