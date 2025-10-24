"""리포트 기반 리테스트 워크플로우.

이 모듈은 기존 최적화 실행 결과를 다시 불러와 손절·ATR·샹들리에·레버리지
파라미터만 집중적으로 재검증하는 보조 CLI 흐름을 제공합니다. 사용자는
저장된 리포트 목록에서 원본 실행을 고르고, 총자산 임계값을 지정하면 해당
조건을 만족하는 후보를 자동으로 큐에 적재해 10회의 미니 탐색을 수행할 수
있습니다. 폴더명은 사용자가 입력한 임계값을 포함한 "Retest : XXX" 접두사를
붙여 원본 실행과 짝지어 생성됩니다.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from . import main_loop
from . import constants


RetestExecutor = Callable[[argparse.Namespace, Sequence[str]], None]


# 리테스트 시 고정할 기본 로직 파라미터. 사용자가 제시한 값 그대로 overrides에
# 주입해 탐색 대상에서 제외합니다.
BASE_PARAMETER_OVERRIDES: Dict[str, object] = {
    "oscLen": 14,
    "signalLen": 5,
    "bbLen": 29,
    "bbMult": 2.0,
    "kcLen": 33,
    "kcMult": 1.3,
    "fluxLen": 9,
    "fluxSmoothLen": 1,
    "useFluxHeikin": False,
    "useModFlux": True,
    "basisStyle": "Deluxe",
    "maType": "EMA",
    "useDynamicThresh": True,
    "useSymThreshold": False,
    "statThreshold": 64.0,
    "buyThreshold": 50.0,
    "sellThreshold": 70.0,
    "dynLen": 46,
    "dynMult": 1.0,
    "exitOpposite": False,
    "useMomFade": False,
}


# 리테스트에서 탐색할 위험/청산 관련 파라미터 목록입니다.
RETEST_SPACE_KEYS: Sequence[str] = (
    "fixedStopPct",
    "leverage",
    "usePyramiding",
)


DISPLAY_METRICS: Sequence[str] = (
    "TotalAssets",
    "Trades",
    "WinRate",
    "MaxDD",
)


@dataclass(frozen=True)
class RunManifest:
    """리포트 디렉터리와 메타데이터 묶음."""

    path: Path
    created_at: str
    label: str
    manifest: Dict[str, object]


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_runs(root: Path) -> List[RunManifest]:
    runs: List[RunManifest] = []
    if not root.exists():
        return runs

    for entry in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not entry.is_dir():
            continue
        manifest_data = _load_manifest(entry / "manifest.json")
        created = str(
            manifest_data.get("created_at")
            or manifest_data.get("run", {}).get("timestamp")
            or manifest_data.get("timestamp")
            or "unknown"
        )
        label = manifest_data.get("run", {}).get("tag") or entry.name
        runs.append(RunManifest(entry, created, label, manifest_data))
    return runs


def _prompt_run_selection(candidates: Sequence[RunManifest]) -> Optional[RunManifest]:
    if not candidates:
        print("저장된 리포트가 없어 리테스트를 진행할 수 없습니다.")
        return None

    print("\n[리테스트] 기본 기록 폴더에서 실행 기록을 찾았습니다:")
    for idx, info in enumerate(candidates, start=1):
        print(f"  {idx}. {info.label} (생성: {info.created_at})")

    while True:
        raw = input("리테스트할 실행 번호를 선택하세요 (취소하려면 Enter): ").strip()
        if not raw:
            return None
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        print("유효한 번호를 입력해주세요.")


def _prompt_threshold() -> Optional[float]:
    while True:
        raw = input("총자산 몇 이상을 임계로 설정하시겠습니까? ").strip()
        if not raw:
            print("임계값이 입력되지 않아 리테스트를 종료합니다.")
            return None
        try:
            value = float(raw)
        except ValueError:
            print("숫자로 입력해주세요.")
            continue
        if value <= 0:
            print("0보다 큰 값을 입력해주세요.")
            continue
        return value


def _format_threshold_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _load_seed(seed_path: Path) -> Dict[str, object]:
    if not seed_path.exists():
        raise FileNotFoundError(f"seed.yaml 을 찾을 수 없습니다: {seed_path}")
    with seed_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _extract_timeframe_label(df: pd.DataFrame) -> Optional[str]:
    for column in ("timeframe", "Timeframe"):
        if column in df.columns:
            series = df[column].dropna()
            if not series.empty:
                first = series.iloc[0]
                if isinstance(first, str):
                    return first
                return str(first)
    return None


def _normalise_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y"}:
            return True
        if lowered in {"false", "no", "n"}:
            return False
    return None


def _extract_seed_params(row: pd.Series) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for key in RETEST_SPACE_KEYS:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            continue
        if isinstance(value, (float, int)) and not pd.isna(value):
            params[key] = float(value) if isinstance(value, float) else int(value)
            continue
        coerced_bool = _normalise_bool(value)
        if coerced_bool is not None:
            params[key] = coerced_bool
            continue
        params[key] = value
    return params


def _build_bank_entries(df: pd.DataFrame) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        params = _extract_seed_params(row)
        if not params:
            continue
        metrics: Dict[str, object] = {}
        for metric in DISPLAY_METRICS:
            if metric in row.index and not pd.isna(row[metric]):
                metrics[metric] = row[metric]
        entry: Dict[str, object] = {"params": params}
        if metrics:
            entry["metrics"] = metrics
        score_field = None
        for candidate in ("Score", "score", "CompositeScore"):
            if candidate in row.index and not pd.isna(row[candidate]):
                score_field = row[candidate]
                break
        if score_field is not None:
            entry["score"] = score_field
        entries.append(entry)
    return entries


def _prepare_params(seed: Dict[str, object], timeframe_label: Optional[str]) -> Dict[str, object]:
    params_cfg = copy.deepcopy(seed.get("params", {})) if seed else {}
    overrides = dict(params_cfg.get("overrides", {}))
    forced_params = seed.get("forced_params")
    if isinstance(forced_params, dict):
        for key, value in forced_params.items():
            overrides.setdefault(key, value)

    for key in RETEST_SPACE_KEYS:
        overrides.pop(key, None)
    overrides.pop("entry_tf", None)
    overrides.pop("ltfChoice", None)
    overrides.pop("ltf_choices", None)

    overrides.update(BASE_PARAMETER_OVERRIDES)
    if timeframe_label:
        overrides["timeframe"] = timeframe_label

    params_cfg["overrides"] = overrides
    if timeframe_label:
        params_cfg["timeframe"] = timeframe_label

    search_cfg = params_cfg.setdefault("search", {})
    search_cfg["n_trials"] = 10

    llm_cfg = params_cfg.setdefault("llm", {})
    llm_cfg["enabled"] = True

    original_space = params_cfg.get("space") or {}
    narrowed_space: Dict[str, object] = {}
    for key in RETEST_SPACE_KEYS:
        if key in original_space:
            narrowed_space[key] = copy.deepcopy(original_space[key])
    if not narrowed_space:
        raise RuntimeError("리테스트에 사용할 탐색 공간을 구성할 수 없습니다.")
    params_cfg["space"] = narrowed_space
    return params_cfg


def _prepare_backtest(seed: Dict[str, object]) -> Dict[str, object]:
    return copy.deepcopy(seed.get("backtest", {})) if seed else {}


def _make_bank_payload(
    entries: List[Dict[str, object]],
    manifest: Dict[str, object],
    space_hash: str,
) -> Dict[str, object]:
    metadata_source = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    return {
        "created_at": main_loop._utcnow_isoformat(),
        "metadata": {
            "symbol": metadata_source.get("symbol") or manifest.get("symbol"),
            "timeframe": metadata_source.get("timeframe") or manifest.get("timeframe"),
            "tag": metadata_source.get("tag") or manifest.get("tag"),
        },
        "space_hash": space_hash,
        "entries": entries,
    }


def _sanitise_prefix(label: str) -> str:
    if os.name == "nt":
        return label.replace(":", "-")
    return label


def run_retest(
    storage_root: Optional[Path] = None,
    *,
    execute_fn: RetestExecutor = main_loop.execute,
) -> None:
    """인터랙티브 리테스트 실행."""

    root = Path(storage_root) if storage_root else constants.DEFAULT_REPORT_ROOT
    runs = _discover_runs(root)
    selection = _prompt_run_selection(runs)
    if selection is None:
        return

    threshold = _prompt_threshold()
    if threshold is None:
        return

    results_path = selection.path / "results.csv"
    if not results_path.exists():
        print(f"{results_path} 에서 results.csv 를 찾을 수 없습니다.")
        return

    df = pd.read_csv(results_path)
    if "TotalAssets" not in df.columns:
        print("results.csv 에 TotalAssets 컬럼이 없어 필터링을 진행할 수 없습니다.")
        return

    df["_TotalAssets"] = pd.to_numeric(df["TotalAssets"], errors="coerce")
    filtered = df[df["_TotalAssets"] >= threshold].copy()
    filtered = filtered.sort_values(by="_TotalAssets", ascending=False)
    filtered = filtered.drop(columns=["_TotalAssets"], errors="ignore")

    if filtered.empty:
        print("조건을 만족하는 후보가 없습니다.")
        return

    timeframe_label = _extract_timeframe_label(filtered)
    seed = _load_seed(selection.path / "seed.yaml")
    params_cfg = _prepare_params(seed, timeframe_label)
    backtest_cfg = _prepare_backtest(seed)

    entries = _build_bank_entries(filtered)
    if not entries:
        print("리테스트에 사용할 파라미터 후보를 구성하지 못했습니다.")
        return

    trial_budget = max(10, len(entries) * 10)
    search_cfg = params_cfg.setdefault("search", {})
    search_cfg["n_trials"] = trial_budget

    space_hash = main_loop._space_hash(params_cfg.get("space", {}))
    bank_payload = _make_bank_payload(entries, selection.manifest, space_hash)

    threshold_label = _format_threshold_label(threshold)
    display_prefix = f"Retest : {threshold_label}"
    fs_prefix = _sanitise_prefix(display_prefix)
    target_dir = selection.path.parent / f"{fs_prefix} {selection.path.name}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="retest_") as tmpdir:
        tmp_root = Path(tmpdir)
        params_path = tmp_root / "params.yaml"
        backtest_path = tmp_root / "backtest.yaml"
        bank_path = tmp_root / "bank.json"
        params_path.write_text(yaml.safe_dump(params_cfg, sort_keys=False), encoding="utf-8")
        backtest_path.write_text(yaml.safe_dump(backtest_cfg, sort_keys=False), encoding="utf-8")
        bank_path.write_text(json.dumps(bank_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Namespace가 필요로 하는 모든 필드를 수동으로 구성합니다. 실제 CLI와 동일한
        # 기본값을 맞추기 위해 parse_args가 반환하는 항목을 참고했습니다.
        namespace = argparse.Namespace(
            params=params_path,
            backtest=backtest_path,
            output=target_dir,
            data=Path("data"),
            symbol=None,
            list_top50=False,
            pick_top50=0,
            pick_symbol=None,
            timeframe=timeframe_label,
            timeframe_grid=None,
            timeframe_mix=None,
            start=None,
            end=None,
            leverage=None,
            qty_pct=None,
            full_space=False,
            basic_factors_only=False,
            llm=None,
            interactive=False,
            enable=[],
            disable=[],
            top_k=0,
            n_trials=None,
            n_jobs=None,
            optuna_jobs=None,
            dataset_jobs=None,
            dataset_executor=None,
            dataset_start_method=None,
            auto_workers=False,
            study_name=None,
            study_template=None,
            storage_url=None,
            storage_url_env=None,
            allow_sqlite_parallel=False,
            force_sqlite_serial=False,
            run_tag=None,
            run_tag_template=None,
            resume_from=bank_path,
            pruner=None,
            cv=None,
            cv_k=None,
            cv_embargo=None,
            storage_root=constants.DEFAULT_STORAGE_ROOT,
        )

        argv = [
            "--params",
            str(params_path),
            "--backtest",
            str(backtest_path),
            "--output",
            str(target_dir),
            "--resume-from",
            str(bank_path),
        ]

        print(
            f"리테스트 실행을 시작합니다. 출력 경로: {target_dir} (임계값: {threshold_label})"
        )
        execute_fn(namespace, argv)


__all__ = ["run_retest"]
