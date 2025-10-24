"""최적화 실행 모듈 – `optimize.main_loop` 와 CLI 어댑터를 재노출합니다."""
from __future__ import annotations

import importlib.util
from typing import Final

from .cli import app, main, parse_args, TYPER_AVAILABLE

_REQUIRED_MODULES: Final[dict[str, str]] = {
    "numpy": "수치 연산 및 배열 연산",
    "pandas": "시계열 데이터 프레임 처리",
    "optuna": "최적화 스터디 구성",
}

_missing = [name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None]
if _missing:
    summary = ", ".join(f"{name} ({_REQUIRED_MODULES[name]})" for name in _missing)
    raise ModuleNotFoundError(
        "필수 의존성 모듈을 찾을 수 없습니다: "
        f"{summary}.\n"
        "`pip install -r requirements.txt` 명령으로 프로젝트 의존성을 설치한 뒤 다시 실행해 주세요."
    )

from . import main_loop as _main_loop

for _name in dir(_main_loop):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_main_loop, _name)

__all__ = [name for name in globals().keys() if not name.startswith("__")]

if __name__ == "__main__":
    main()
