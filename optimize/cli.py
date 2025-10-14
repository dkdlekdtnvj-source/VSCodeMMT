"""최적화 실행을 위한 CLI 진입점."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .constants import DEFAULT_STORAGE_ROOT
from . import main_loop

# Typer 로 CLI를 제공할 수 있는지 여부를 확인합니다.
typer_spec = importlib.util.find_spec("typer")
if typer_spec is not None:  # pragma: no cover - 선택적 의존성
    import typer
else:  # pragma: no cover - Typer 미설치 환경 폴백
    typer = None  # type: ignore[assignment]

TYPER_AVAILABLE = typer_spec is not None

_ORIGINAL_ARGV: List[str] = []

if TYPER_AVAILABLE:
    app = typer.Typer(help="매직1분VN 전략 최적화 CLI")
else:  # pragma: no cover - Typer 없음
    app = None  # type: ignore[assignment]


def _prepare_cli_tokens(tokens: Sequence[str]) -> List[str]:
    processed: List[str] = []
    replaced_interactive = False
    for token in tokens:
        if token == "시작":
            processed.append("--interactive")
            replaced_interactive = True
        else:
            processed.append(token)
    if replaced_interactive and "--interactive" not in processed:
        processed.append("--interactive")
    return processed


def _build_argparse_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimize.run",
        description="Pine 전략 최적화를 실행하는 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"), help="전략 파라미터 YAML 경로")
    parser.add_argument(
        "--backtest", type=Path, default=Path("config/backtest.yaml"), help="백테스트 설정 YAML 경로"
    )
    parser.add_argument("--output", type=Path, help="결과 저장 디렉터리(기본: 자동 생성)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="데이터 캐시 루트 경로")
    parser.add_argument("--symbol", type=str, help="최적화 대상 심볼 강제 지정")
    parser.add_argument("--list-top50", action="store_true", help="USDT-Perp 거래대금 상위 50 심볼을 출력 후 종료")
    parser.add_argument("--pick-top50", type=int, default=0, help="상위 50 리스트에서 번호로 심볼 선택 (1~50)")
    parser.add_argument("--pick-symbol", type=str, help="직접 심볼 지정 (예: BINANCE:ETHUSDT)")
    parser.add_argument("--timeframe", type=str, help="하위 타임프레임 강제 지정")
    parser.add_argument("--timeframe-grid", type=str, help="쉼표/세미콜론 구분 타임프레임 일괄 실행")
    parser.add_argument("--timeframe-mix", type=str, help="여러 타임프레임 혼합 실행 (쉼표/세미콜론 구분)")
    parser.add_argument("--start", type=str, help="백테스트 시작일 (ISO8601)")
    parser.add_argument("--end", type=str, help="백테스트 종료일 (ISO8601)")
    parser.add_argument("--leverage", type=float, help="레버리지 강제 설정")
    parser.add_argument("--qty-pct", type=float, help="포지션 진입 비율 강제 설정")
    parser.add_argument("--full-space", action="store_true", help="기본 팩터 필터 비활성화")
    parser.add_argument("--basic-factors-only", action="store_true", help="기본 팩터 필터 강제 활성화")
    parser.add_argument("--llm", dest="llm", action="store_true", help="Gemini 기반 LLM 제안 사용")
    parser.add_argument("--no-llm", dest="llm", action="store_false", help="Gemini 기반 LLM 제안 비활성화")
    parser.set_defaults(llm=None)
    parser.add_argument("--interactive", "--시작", action="store_true", help="심볼/옵션을 대화형으로 선택")
    parser.add_argument("--enable", action="append", dest="enable", help="강제로 활성화할 불리언 파라미터")
    parser.add_argument("--disable", action="append", dest="disable", help="강제로 비활성화할 불리언 파라미터")
    parser.add_argument("--top-k", type=int, default=0, help="상위 K개 트라이얼을 워크포워드 성과로 재정렬")
    parser.add_argument("--n-trials", type=int, help="Optuna 트라이얼 수 덮어쓰기")
    parser.add_argument("--n-jobs", type=int, help="Optuna 병렬 worker 수")
    parser.add_argument("--optuna-jobs", type=int, help="Optuna n_jobs 직접 지정")
    parser.add_argument("--dataset-jobs", type=int, help="데이터셋 병렬 worker 수")
    parser.add_argument("--dataset-executor", choices=["thread", "process"], help="데이터셋 병렬 백테스트 실행기 선택")
    parser.add_argument("--dataset-start-method", type=str, help="multiprocessing start method 지정")
    parser.add_argument("--auto-workers", action="store_true", help="가용 CPU 기반 병렬 설정 자동 조정")
    parser.add_argument("--study-name", type=str, help="Optuna 스터디 이름 덮어쓰기")
    parser.add_argument("--study-template", type=str, help="타임프레임 그리드 실행 시 스터디 이름 템플릿")
    parser.add_argument("--storage-url", type=str, help="Optuna 스토리지 URL")
    parser.add_argument("--storage-url-env", type=str, help="스토리지 URL 환경변수 이름")
    parser.add_argument("--allow-sqlite-parallel", action="store_true", help="SQLite 스토리지에서도 병렬 허용")
    parser.add_argument("--force-sqlite-serial", action="store_true", help="SQLite 사용 시 worker 1개로 강제")
    parser.add_argument("--run-tag", type=str, help="출력 디렉터리 추가 태그")
    parser.add_argument("--run-tag-template", type=str, help="타임프레임 그리드 출력 태그 템플릿")
    parser.add_argument("--resume-from", type=Path, help="warm-start 용 bank.json 경로")
    parser.add_argument("--pruner", type=str, help="Optuna 프루너 선택 (asha, median 등)")
    parser.add_argument("--cv", type=str, choices=["purged-kfold", "none"], help="보조 검증 방식")
    parser.add_argument("--cv-k", type=int, help="Purged K-Fold 분할 수")
    parser.add_argument("--cv-embargo", type=float, help="Purged K-Fold embargo 비율")
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
        help="결과 및 로그 기본 저장 경로",
    )
    return parser


if TYPER_AVAILABLE:

    @app.callback(invoke_without_command=True)
    def _typer_entry(
        ctx: typer.Context,
        params: Path = typer.Option(Path("config/params.yaml"), help="전략 파라미터 YAML 경로"),
        backtest: Path = typer.Option(Path("config/backtest.yaml"), help="백테스트 설정 YAML 경로"),
        output: Optional[Path] = typer.Option(None, help="결과 저장 디렉터리"),
        data: Path = typer.Option(Path("data"), help="데이터 캐시 루트 경로"),
        symbol: Optional[str] = typer.Option(None, help="심볼 강제 지정"),
        list_top50: bool = typer.Option(False, "--list-top50", help="상위 50 심볼 출력"),
        pick_top50: int = typer.Option(0, "--pick-top50", help="상위 50 리스트에서 선택"),
        pick_symbol: Optional[str] = typer.Option(None, "--pick-symbol", help="심볼 문자열 직접 지정"),
        timeframe: Optional[str] = typer.Option(None, help="하위 타임프레임 강제 지정"),
        timeframe_grid: Optional[str] = typer.Option(None, help="다중 타임프레임 그리드 실행"),
        timeframe_mix: Optional[str] = typer.Option(None, help="타임프레임 혼합 실행"),
        start: Optional[str] = typer.Option(None, help="백테스트 시작일"),
        end: Optional[str] = typer.Option(None, help="백테스트 종료일"),
        leverage: Optional[float] = typer.Option(None, help="레버리지 강제 설정"),
        qty_pct: Optional[float] = typer.Option(None, help="포지션 진입 비율"),
        full_space: bool = typer.Option(False, "--full-space", help="모든 팩터 사용"),
        basic_factors_only: bool = typer.Option(False, help="기본 팩터만 사용"),
        llm: Optional[bool] = typer.Option(None, help="Gemini LLM 사용 여부"),
        interactive: bool = typer.Option(False, "--interactive", help="대화형 실행"),
        enable: Optional[List[str]] = typer.Option(None, help="강제 활성화 파라미터"),
        disable: Optional[List[str]] = typer.Option(None, help="강제 비활성화 파라미터"),
        top_k: int = typer.Option(0, help="워크포워드 재정렬 상위 K"),
        n_trials: Optional[int] = typer.Option(None, help="Optuna 트라이얼 수"),
        n_jobs: Optional[int] = typer.Option(None, help="Optuna 병렬 worker 수"),
        optuna_jobs: Optional[int] = typer.Option(None, help="Optuna n_jobs 직접 지정"),
        dataset_jobs: Optional[int] = typer.Option(None, help="데이터셋 병렬 worker 수"),
        dataset_executor: Optional[str] = typer.Option(None, help="데이터셋 실행기"),
        dataset_start_method: Optional[str] = typer.Option(None, help="multiprocessing start method"),
        auto_workers: bool = typer.Option(False, help="CPU 기반 worker 자동 조정"),
        study_name: Optional[str] = typer.Option(None, help="Optuna 스터디 이름"),
        study_template: Optional[str] = typer.Option(None, help="타임프레임 그리드 실행 시 스터디 이름 템플릿"),
        storage_url: Optional[str] = typer.Option(None, help="Optuna 스토리지 URL"),
        storage_url_env: Optional[str] = typer.Option(None, help="스토리지 URL 환경변수 이름"),
        allow_sqlite_parallel: bool = typer.Option(False, help="SQLite 병렬 허용"),
        force_sqlite_serial: bool = typer.Option(False, help="SQLite 직렬 강제"),
        run_tag: Optional[str] = typer.Option(None, help="출력 태그"),
        run_tag_template: Optional[str] = typer.Option(None, help="타임프레임 그리드 출력 태그 템플릿"),
        resume_from: Optional[Path] = typer.Option(None, help="warm-start bank.json"),
        pruner: Optional[str] = typer.Option(None, help="Optuna 프루너"),
        cv: Optional[str] = typer.Option(None, help="보조 검증 방식"),
        cv_k: Optional[int] = typer.Option(None, help="Purged K-Fold 분할 수"),
        cv_embargo: Optional[float] = typer.Option(None, help="Purged K-Fold embargo 비율"),
        storage_root: Path = typer.Option(DEFAULT_STORAGE_ROOT, help="결과 및 로그 기본 저장 경로"),
    ) -> argparse.Namespace:
        namespace = argparse.Namespace(
            params=params,
            backtest=backtest,
            output=output,
            data=data,
            symbol=symbol,
            list_top50=list_top50,
            pick_top50=pick_top50,
            pick_symbol=pick_symbol,
            timeframe=timeframe,
            timeframe_grid=timeframe_grid,
            timeframe_mix=timeframe_mix,
            start=start,
            end=end,
            leverage=leverage,
            qty_pct=qty_pct,
            full_space=full_space,
            basic_factors_only=basic_factors_only,
            llm=llm,
            interactive=interactive,
            enable=list(enable or []),
            disable=list(disable or []),
            top_k=top_k,
            n_trials=n_trials,
            n_jobs=n_jobs,
            optuna_jobs=optuna_jobs,
            dataset_jobs=dataset_jobs,
            dataset_executor=dataset_executor,
            dataset_start_method=dataset_start_method,
            auto_workers=auto_workers,
            study_name=study_name,
            study_template=study_template,
            storage_url=storage_url,
            storage_url_env=storage_url_env,
            allow_sqlite_parallel=allow_sqlite_parallel,
            force_sqlite_serial=force_sqlite_serial,
            run_tag=run_tag,
            run_tag_template=run_tag_template,
            resume_from=resume_from,
            pruner=pruner,
            cv=cv,
            cv_k=cv_k,
            cv_embargo=cv_embargo,
            storage_root=storage_root,
        )
        ctx.obj = namespace
        if ctx.resilient_parsing:
            return namespace

        main_loop.simple_metrics_enabled = False
        main_loop.execute(namespace, list(_ORIGINAL_ARGV))
        return namespace

    @app.command("retest", help="저장된 리포트에서 위험 파라미터 리테스트를 실행합니다.")
    def _typer_retest(
        storage_root: Optional[Path] = typer.Option(None, help="결과 저장 루트 커스텀 경로"),
    ) -> None:
        from . import retest as retest_module

        retest_module.run_retest(storage_root=storage_root)
else:  # pragma: no cover - Typer 미사용 환경에서 placeholder 유지
    app = None  # type: ignore[assignment]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI 인자를 파싱해 :class:`argparse.Namespace` 로 반환합니다."""

    tokens = list(argv or [])
    global _ORIGINAL_ARGV
    _ORIGINAL_ARGV = list(tokens)
    processed = _prepare_cli_tokens(tokens)

    if not TYPER_AVAILABLE:
        parser = _build_argparse_parser()
        namespace = parser.parse_args(processed)
        namespace.enable = list(namespace.enable or [])  # type: ignore[attr-defined]
        namespace.disable = list(namespace.disable or [])  # type: ignore[attr-defined]
        main_loop.simple_metrics_enabled = False
        return namespace

    assert typer is not None and app is not None  # for type-checkers
    cmd = typer.main.get_command(app)
    with cmd.make_context("optimize", processed, resilient_parsing=True) as ctx:
        cmd.invoke(ctx)
        namespace = ctx.obj if isinstance(ctx.obj, argparse.Namespace) else argparse.Namespace()
    main_loop.simple_metrics_enabled = False
    return namespace


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``python -m optimize.run``."""

    if argv is None:
        original_argv = list(sys.argv[1:])
    else:
        original_argv = list(argv)

    if original_argv and original_argv[0] == "retest":
        parser = argparse.ArgumentParser(prog="optimize.run retest")
        parser.add_argument("--storage-root", type=Path, help="결과 저장 루트 커스텀 경로")
        args = parser.parse_args(original_argv[1:])
        from . import retest as retest_module

        retest_module.run_retest(storage_root=args.storage_root)
        return

    global _ORIGINAL_ARGV
    _ORIGINAL_ARGV = list(original_argv)
    processed = _prepare_cli_tokens(original_argv)
    if not TYPER_AVAILABLE or app is None:
        namespace = parse_args(original_argv)
        main_loop.execute(namespace, list(_ORIGINAL_ARGV))
        return

    assert typer is not None
    app(args=processed)


__all__ = ["parse_args", "main", "app", "TYPER_AVAILABLE"]
