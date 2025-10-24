"""공용 유틸리티 함수 모음."""
from __future__ import annotations

from typing import Any, Callable

from .constants import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:  # pragma: no cover - numba 사용 시 경로
    from numba import njit, prange  # type: ignore[assignment]
else:  # pragma: no cover - numba 미설치 환경

    def njit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Numba 없이도 동일한 데코레이터 인터페이스를 제공하는 폴백."""

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]  # type: ignore[return-value]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def prange(*args: Any, **kwargs: Any):  # type: ignore[override]
        return range(*args, **kwargs)

__all__ = ["njit", "prange"]
