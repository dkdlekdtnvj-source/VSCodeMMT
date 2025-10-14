import math

import pytest

from optimize.common import resolve_leverage


def test_resolve_leverage_prefers_params():
    result = resolve_leverage({"leverage": 7}, {"leverage": 3})
    assert result == pytest.approx(7.0)


def test_resolve_leverage_falls_back_to_risk():
    result = resolve_leverage({}, {"leverage": 11})
    assert result == pytest.approx(11.0)


def test_resolve_leverage_returns_default_when_missing():
    result = resolve_leverage({}, {})
    assert result == pytest.approx(10.0)


def test_resolve_leverage_ignores_non_finite():
    result = resolve_leverage({"leverage": math.nan}, {"leverage": math.inf})
    assert result == pytest.approx(10.0)


def test_resolve_leverage_handles_bool_and_string():
    result = resolve_leverage({"leverage": "15"}, {"leverage": True})
    assert result == pytest.approx(15.0)
