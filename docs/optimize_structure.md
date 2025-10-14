# 최적화 엔진 구조도

최신 모듈 분리 작업(2025-10-12 기준)을 토대로 `optimize` 패키지의 구성 요소를
묶음 단위로 정리했습니다. 각 묶음은 실사용 경로(활성)과 참고용/향후 통합 예정인
경로(비활성)를 함께 기록합니다. 새로운 기능을 추가하거나 기존 흐름을 수정할 때는
동일 묶음의 파일을 한 번에 검토해 유기적으로 반영해주세요.

> **업데이트 규칙**
> - 각 작업 후 묶음/상태/의존 관계에 변화가 있으면 본 문서의 표와 타임라인을 갱신합니다.
> - 새로운 파일을 추가할 때는 어떤 묶음에 속하는지 명시하고, 활성 여부를 처음부터 기록합니다.
> - 비활성 파일을 다시 사용할 때는 상태를 "활성"으로 바꾸고 이유/사용 경로를 타임라인에 남깁니다.

## 1. 상위 흐름 요약

```mermaid
graph TD
    subgraph CLI & 실행 제어
        run[run.py\n(모듈 재노출/의존성 점검)]
        cli[cli.py\n(Typer/Argparse)]
        main[main_loop.py\n(최적화 엔진/출구키 정규화·실사용 파라미터 기록·목표 방향 동기화·LTF 실행 플랜 통합·chart_tf/entry_tf 정규화·트레이드 요약 로그)]
        quickstart[quickstart.py\n(대화형 가이드)]
        retest[retest.py\n(리테스트 워크플로우)]
    end

    subgraph 전략 백엔드
        strat[strategy_model.py\n(주 전략·피라미딩·임계 스케일)]
        indi[indicators.py\n(TR1 정규화·Deadzone 지표)]
        state[state.py\n(상태 데이터클래스)]
        utils[utils.py\n(Numba 폴백)]
        common[common.py\n(공용 해석기)]
        pine[strategy/strategy.pine\n(TradingView 동기화·TR1 모멘텀 HUD)]
    end

    subgraph 평가 & 보고
        metrics[metrics.py\n(성과 집계·로그 스케일 자본곡선)]
        report[report.py\n(리포트 출력·스탑 플래그 노출)]
    end

    subgraph 실험 옵션
        alt[alternative_engine.py\n(외부 엔진·Flux 게이트·안정 자본곡선)]
        wrapper[wrapper_strategy_model.py\n(HTF 비활성 래퍼)]
        regime[regime.py\n(시장 국면)]
        wf[wf.py\n(워크포워드)]
    end

    subgraph 탐색 & 스토리지
        study[study_setup.py\n(스토리지 초기화)]
        space[search_spaces.py\n(Exit guard 정규화·불리언 choices 강제 적용)]
    end

    subgraph LLM 후보 다양화
        llm[llm.py\n(LLM 후보 생성·Gemini 연동)]
        diversifier[TrialDiversifier\n(main_loop.py)]
    end

    cli --> run
    run --> main
    main --> strat
    main --> metrics
    main --> ltfMix[combine_metrics\n(LTF 혼합 집계)]
    ltfMix --> report
    main --> report
    cli --> retest
    retest --> main
    strat --> indi
    strat --> state
    strat --> utils
    strat --> common
    strat -. Pine 동기화 .-> pine
    main -. 선택적 .-> alt
    main --> regime
    main --> wf
    main --> study
    main -. 큐 제어 .-> diversifier
    study --> space
    quickstart --> run
    wrapper -. 수동 호출 .- strat
    main --> llm
    llm --> diversifier
    diversifier --> study
    llm -. 후보 피드백 .-> main
```

## 2. 묶음별 상세 표

| 묶음 | 구성 파일 | 주요 책임 | 상태 | 주 사용 경로 |
| --- | --- | --- | --- | --- |
| CLI & 실행 제어 | `optimize/run.py`, `optimize/cli.py`, `optimize/main_loop.py`, `optimize/quickstart.py`, `optimize/retest.py` | CLI 파싱, 실행 루프, 인터랙티브 가이드, 모듈 재노출 및 사전 의존성 점검, **BASIC_FACTOR_KEYS 출구 파라미터 동기화**, 단일 목적 최적화 시 Optuna 방향(**minimize/maximize**) 전파, 리포트 기반 리테스트 파이프라인, **실사용 LTF·파라미터 기록 및 CSV/로그 동기화**, **exitOpposite/useMomFade 강제가 다른 손절 옵션 유무를 감지하도록 개선**, **진행 로그에 WinRate·MaxDD 추가 노출**, **LTF 그리드/혼합 실행 플랜을 `_prepare_timeframe_execution_plan`으로 통합**, **chart_tf/entry_tf/use_htf/htf_tf 단일 소스 정규화 및 기록 필드 교체**, **트레이드 요약(승수·승률·총/평균 거래대금) 로그/CSV 동기화** | **활성** | `python -m optimize.run`, Typer 앱, 리테스트 서브커맨드, 테스트: `tests/test_run_helpers.py`, `tests/test_retest.py` |

| 전략 백엔드 코어 | `optimize/strategy_model.py`, `optimize/indicators.py`, `optimize/state.py`, `optimize/utils.py`, `optimize/common.py` | 백테스트 엔진, Numba 지표, 상태 관리, 공용 계산, **피라미딩 제어**, **TR1 기반 모멘텀 정규화 일원화(레거시 ATR 경로 제거)**, **normalize_tf 로 타임프레임 파라미터 단일 소스화** | **활성** | `main_loop.run_backtest`, `alternative_engine` 참조, 전략 관련 테스트 |
| TradingView 전략 스크립트 | `strategy/strategy.pine` | TradingView 전략 동기화, TR1 모멘텀 HUD·데드존 플럭스 표시, **게이트 컷 기본 적용·레거시 ATR 지표 제거** | **활성** | TradingView 전략 차트, Pine 백테스트 |
| 메트릭 & 보고 | `optimize/metrics.py`, `optimize/report.py` | 백테스트 결과 정규화, 목적함수 계산, 보고서/시각화, **CSV 후처리(빈 열 제거)**, **ATR·고정 스탑 플래그 요약**, **로그 스케일 자본곡선/드로우다운 계산 및 수익률 스케일 자동 교정** | **활성** | `main_loop` 후처리, CLI 보고, `tests/test_metrics.py` |
| 실험/보조 모듈 | `optimize/alternative_engine.py`, `optimize/wrapper_strategy_model.py`, `optimize/regime.py`, `optimize/wf.py` | 대체 엔진, HTF 옵션 강제 비활성 래퍼, 시장 국면 감지, 워크포워드 평가, **Flux Deadzone/게이트 적용 및 벡터 엔진 동기화**, **외부 엔진 결과도 로그 스케일 자본곡선·RUIN 검사에 맞춰 안정화** | **부분 활성**<br>(기본 경로 아님) | `main_loop`의 선택적 경로, 특정 실험 스크립트, 테스트: `tests/test_alternative_engine.py` |
| Optuna 스토리지 & 검색 공간 | `optimize/study_setup.py`, `optimize/search_spaces.py` | 스토리지 초기화, 탐색 공간 생성/변형, 파라미터 샘플링, **불리언 values/기본값 우선 반영**, **exitOpposite/useMomFade 안전장치 보정**, **플럭스 게이트 컷 옵션 제거 반영** | **활성** | `main_loop`에서 스터디 구성 및 탐색, 테스트: `tests/test_search_spaces.py`, `tests/test_run_helpers.py` |
| Optuna 스토리지 & 검색 공간 | `optimize/study_setup.py`, `optimize/search_spaces.py` | 스토리지 초기화, 탐색 공간 생성/변형, 파라미터 샘플링, **exitOpposite/useMomFade 안전장치 보정**, **불리언 values/default 일관성 보장** | **활성** | `main_loop`에서 스터디 구성 및 탐색, 테스트: `tests/test_search_spaces.py`, `tests/test_run_helpers.py` |
| Optuna 스토리지 & 검색 공간 | `optimize/study_setup.py`, `optimize/search_spaces.py` | 스토리지 초기화, 탐색 공간 생성/변형, 파라미터 샘플링, **exitOpposite/useMomFade 안전장치 보정** | **활성** | `main_loop`에서 스터디 구성 및 탐색, 테스트: `tests/test_search_spaces.py`, `tests/test_run_helpers.py` |
| LLM 후보 & 다양화 | `optimize/llm.py`, `optimize/main_loop.py::<br>LLMCandidateRefresher`, `optimize/main_loop.py::<br>TrialDiversifier` | Gemini 기반 후보 생성, **LLM 프롬프트 직렬화**, Optuna 큐에 동적 후보 주입, **트라이얼 다양화 시드 관리** | **부분 활성**<br>(API 키 필요) | `main_loop` LLM 큐, CLI 플래그(`--use-llm`), `tests/test_diversifier.py` |
| 설정 & 상수 | `optimize/constants.py` | 전역 상수, 기본 경로, 환경 설정, **트라이얼 진행 로그 필드 관리**, **chart_tf/entry_tf/use_htf/htf_tf 진행 로그 컬럼 추가**, **트레이드 요약용 Wins/TotalVol/AvgVol 컬럼 확장** | **활성** | 여러 모듈에서 공통 참조 |
| LLM 후보 & 다양화 | `optimize/llm.py`, `optimize/main_loop.py::<br>LLMCandidateRefresher`, `optimize/main_loop.py::<br>TrialDiversifier` | Gemini 기반 후보 생성, **LLM 프롬프트 직렬화**, Optuna 큐에 동적 후보 주입, **트라이얼 다양화 시드 관리**, **Gemini 응답 파서 다중 포맷 대응(functionCall/코드블럭/텍스트) 및 실패 요약 로그·모델 호출 경로 추적** | **부분 활성**<br>(API 키 필요) | `main_loop` LLM 큐, CLI 플래그(`--use-llm`), `tests/test_diversifier.py` |
| 설정 & 상수 | `optimize/constants.py` | 전역 상수, 기본 경로, 환경 설정, **트라이얼 진행 로그 필드 관리**, **chart_tf/entry_tf/use_htf/htf_tf 진행 로그 컬럼 추가** | **활성** | 여러 모듈에서 공통 참조 |
| 데이터 피드 연계 | `datafeed/cache.py` 등 | 가격 데이터 캐시 로딩, 백테스트 입력 준비 | **활성** | `main_loop`가 직접 사용 |

## 3. 활성/비활성 세부 메모

### 활성
- **`optimize/main_loop.py`**: 실행 루프의 핵심. 백테스트 설정(`backtest_cfg`)을 모듈 전역으로 보존하고 `run_backtest` 및 LLM 후보 생성기를 동적으로 주입합니다. 최신 작업에서 실사용 파라미터(타임프레임·채널 스탑 포함)를 Optuna 사용자 속성·CSV·진행 로그에 일치시켜 기록하고, 단일 목적 스터디에도 정의한 방향(`minimize`/`maximize`)을 그대로 전달하며, 탐색 공간에 `timeframe` 또는 `ltf` 중 누락된 항목이 있으면 자동으로 보완해 데이터셋 선택이 엇갈리지 않도록 합니다. 또한 exitOpposite/useMomFade 안전장치가 고정 손절·ATR 손절·샹들리에·SAR 출구가 이미 활성화된 경우에는 원래 설정을 존중하도록 보완하고, 진행 로그에 WinRate/MaxDD/승수/총·평균 거래대금을 함께 노출해 확인성을 높였습니다.
- **`combine_metrics`**: 거래 수·총자산·최대 드로우다운을 기반으로 LTF 혼합 자본 지표를 가중 평균으로 합산해, 실전에서 안정적인 타임프레임의 영향력을 확대합니다.
- **`optimize/indicators.py`**: 과거 `strategy_model.py` 내부 함수였던 `_ema`, `_hma`, `_atr`, `_dmi` 등 지표를 모두 모아 Numba JIT 경로/폴백을 일관성 있게 제공하며, TR1 정규화 모멘텀과 Flux Deadzone 게이트 계산을 공통 모듈로 노출합니다.
- **`optimize/metrics.py`**: 퍼포먼스 집계를 담당하며, 수익률 스케일을 자동 교정하고 로그 스케일 자본곡선·드로우다운 곡선을 계산해 overflow·inf를 방지합니다. 누락된 거래 수익률도 float64로 강제해 안정적인 NetProfit/Sharpe 추정치를 제공합니다.
- **`optimize/strategy_model.py` & `optimize/alternative_engine.py`**: KC/Deluxe를 포함한 모든 모멘텀 스타일을 TR1 기반 정규화로 통일하고 레거시 ATR 호환 스케일 경로를 제거했습니다. 채널 기반 손절(BB/KC)은 유지하며 Flux 게이트는 컷 적용이 기본 동작으로 고정되었습니다. vectorbt 대체 엔진 경로 역시 로그 스케일 자본곡선을 사용해 TotalAssets/RUIN 판단이 파이썬 엔진과 동일하게 유지됩니다.
- **`optimize/state.py`**: 포지션 추적(`Position`), 자본 상태(`EquityState`), 필터 컨텍스트 등을 데이터클래스로 정의해 `strategy_model.run_backtest`에서 사용합니다.
- **`optimize/report.py`**: `write_trials_dataframe` 가 빈 파라미터 헤더를 제거해 `trials.csv` 에 공백 열이 나타나지 않도록 후처리를 강화했습니다. ATR/고정 손절·트레일링 활성 여부를 요약 컬럼으로 노출하고 비활성 시 파라미터 값을 비웁니다.
- **`strategy/strategy.pine`**: TradingView 전략 본문. HUD·시각화를 Python 백엔진과 동기화하며, TR1 모멘텀과 Flux Deadzone을 표시합니다. Flux 게이트는 컷 값을 기본으로 사용하고 레거시 ATR 보정 경로를 제거해 Python 엔진과 완전히 동일한 정규화를 유지합니다.
- **`optimize/llm.py` & `TrialDiversifier`**: Gemini API를 통해 정기적으로 후보 파라미터 번들을 생성하고, `LLMCandidateRefresher`가 Optuna 큐에 주입합니다. API 키가 없을 때는 로그 안내만 남기고 비활성화되며, 후보는 `TrialDiversifier`를 통해 `study.enqueue_trial`에 전달됩니다.

### 부분 활성 / 비활성
- **`optimize/alternative_engine.py`**: `run_backtest_alternative`가 준비된 외부 엔진(예: vectorbt)일 때만 호출됩니다. 기본 실행에서는 비활성이며, 테스트에서 모의 객체로 검증 중입니다. 현재 버전은 로그 스케일 자본곡선을 사용해 TotalAssets 및 RUIN 플래그를 계산하므로 메인 엔진과 동일한 안정성을 보장합니다.
- **`optimize/wrapper_strategy_model.py`**: 특정 HTF 옵션을 강제로 끄기 위한 래퍼. 현재 자동 호출 경로는 없으나, 레거시 스크립트/수동 실험에서 사용 가능하도록 유지하고 있습니다.
- **LLM 후보 생성 경로**: `optimize/llm.py`가 Google Gemini 클라이언트(`google-genai`)에 의존하므로, 키 미제공 환경에서는 자동으로 비활성 로그만 남깁니다. CLI에서 `--use-llm`를 활성화하거나 환경 변수를 제공해야 실사용 경로가 열립니다.

## 4. 변경 타임라인 (예시)
타임라인에는 날짜 / 시간 예시) 25-10-13/15:45 식으로 적을 것.
| 날짜 | 변경 내용 | 작업자 | 비고 |
| --- | --- | --- | --- |
| 2025-10-12 | `strategy_model.py` 분리, `indicators.py`/`state.py`/`constants.py` 생성 | @soonback | 구조 재정비 1차 |
| 2025-10-12 | `main_loop.py` 도입, CLI/LLM/메트릭 역할 분리 | @soonback | `run.py`는 재노출 전용으로 축소 |
| 2025-10-13 | `run.py` 실행 시 필수 파이썬 의존성 확인 로직 추가 | @assistant | 누락 모듈 안내 메시지 강화 |
| 2025-10-13 | 사용자 지침에 "TradingView와 동등한 결과 확보" 항목 추가 | @assistant | Pine ↔ 파이썬 계산 일치 강조 |
| 2025-10-12 | `strategy_model.py`/`state.py` 피라미딩 1회 진입 로직 추가, `strategy/strategy.pine` 동기화 | @assistant | 백엔진·파인스크립트 모두 동일 동작 유지 |
| 2025-10-14 | `main_loop.py` BASIC_FACTOR_KEYS에 샹들리에/파라볼릭 SAR 출구 키 추가, 결과 리포트 컬럼 확장 | @assistant | 결과표에 출구 파라미터 노출, `tests/test_run_helpers.py`/`tests/test_report.py` 연동 |
| 2025-10-16 | `retest.py` 임계 통과 후보당 10회 탐색 배분 로직 보강 | @assistant | 후보 개수 × 10회 트라이얼 예산으로 스케일링 |
| 2025-10-15 | `cli.py` Typer 서브커맨드 `retest` 추가, `retest.py` 신설 | @assistant | 저장 리포트 기반 위험 파라미터 10회 리테스트 자동화 |
| 2025-10-15 | `combine_metrics`에 LTF 혼합 집계 규칙 설명 추가, README에 해석 가이드 신설 | @assistant | 혼합 타임프레임 결과 해석 명확화 |
| 2025-10-16 | `combine_metrics` 가중 평균 도입, `write_trials_dataframe` 빈 파라미터 열 제거 | @assistant | 혼합 자본 지표 신뢰도·CSV 가독성 개선 |
| 2025-10-17 | `strategy/strategy.pine` 중복 블록 제거 및 최신 고정 손절 로직 유지 | @assistant | Pine 본문 복제 제거, % 기반 손절 버전만 보존 |
| 2025-10-18/12:00 | KC/Deluxe 모멘텀 스타일 ATR 정규화, Pine 스크립트·메인 엔진·대체 엔진 동기화 | @assistant | 모멘텀 스케일 일관화, 구조 문서 업데이트 |
| 2025-10-19 | `main_loop.py` 실사용 LTF 기록, 스탑 플래그 Optuna 속성/로그 반영 및 `report.py`/`constants.py` 플래그 컬럼 확장 | @assistant | ATR·고정 손절/트레일링 활성화 요약, trials.csv/진행 로그 컬럼 정비 |
| 2025-10-20/11:00 | `strategy/strategy.pine` 모멘텀/신호 교차 원형 마커 추가 및 옵션화 | @assistant | TradingView 시각화 강화, Python 백엔진과 조건 동기화 |
| 2025-10-20/17:30 | `search_spaces.py` Optuna 샘플링 후 출구 플래그 정규화, `trial.params` 동기화 | @assistant | exitOpposite/useMomFade 최소 1개 활성, 회귀 테스트 보강 |
| 2025-10-21/09:30 | `main_loop.py` 단일 목적 스터디 방향 전달, 테스트 보강 (`tests/test_run_helpers.py`) | @assistant | Optuna 최소화 옵션이 정확히 전파되도록 수정 |
| 2025-10-21/14:45 | `search_spaces.py` 불리언 values/default 강제 반영, 테스트 회귀 추가 | @assistant | 단일 선택지·기본값 충돌 방지 |
| 2025-10-20/15:30 | `main_loop.py` 탐색 공간 `timeframe`/`ltf` 자동 보완 로직 분리, 테스트 강화 | @assistant | 파라미터 누락 시 데이터셋 선택 일관성 유지 |
| 2025-10-21/14:10 | `main_loop.py` 출구 안전장치가 대체 손절 활성 여부를 확인하도록 개선, `tests/test_retest.py` 리테스트 케이스 확장 | @assistant | exitOpposite 강제 조건 명시 로그 및 대체 출구 병행 시 설정 유지 |
| 2025-10-21/13:40 | `search_spaces.py` 불리언 `values`/`default` 처리 강화, 회귀 테스트 추가 | @assistant | 단일 선택 강제 및 기본값 모순 방지 |
| 2025-10-22/10:15 | `llm.py` Gemini 후보 생성기, `TrialDiversifier` LLM 큐 연동 | @assistant | API 키 의존성 명시, 구조 문서 업데이트 |
| 2025-10-22/18:30 | `main_loop.py` 실사용 파라미터·로그 동기화, 채널 손절 파이프라인 (`strategy_model.py`, `strategy/strategy.pine`, `report.py`, `constants.py`, `config/params.yaml`) | @assistant | 채널 손절 파라미터 추가, CSV/로그 정합화, 진행 로그 WinRate·MaxDD 표기 |
| 2025-10-22/21:00 | `main_loop.py` LTF 실행 플랜 분기 통합(`_prepare_timeframe_execution_plan` 도입), `docs/optimize_structure.md` 구조 업데이트 | @assistant | CLI 타임프레임 혼합/그리드 흐름을 단일 경로로 정리, 문서 구조 반영 |
| 2025-10-23/09:45 | TR1 모멘텀·Flux Deadzone 스케일 정렬, 호환 토글 및 자동 임계 스케일 옵션 정비 (`strategy/strategy.pine`, `optimize/indicators.py`, `optimize/strategy_model.py`, `optimize/alternative_engine.py`, `config/params.yaml`) | @assistant | 파인·파이썬 임계 스케일 동기화, 대체 엔진 테스트 보강, 구조 문서 업데이트 |
| 2025-10-23/18:00 | 레거시 ATR 기반 모멘텀 경로 제거 및 Flux 게이트 컷 기본 적용 (`optimize/indicators.py`, `optimize/strategy_model.py`, `optimize/alternative_engine.py`, `strategy/strategy.pine`, `config/params.yaml`, `optimize/search_spaces.py`, `optimize/main_loop.py`, `tests/*`) | @assistant | Pine·Python 정규화 완전 통일, 탐색 공간 파라미터 정리 |
| 2025-10-24/10:30 | 타임프레임 정규화 유틸(`normalize_tf`) 추가, chart_tf/entry_tf/use_htf/htf_tf 출력 필드 정비 (`optimize/common.py`, `optimize/main_loop.py`, `optimize/retest.py`, `optimize/constants.py`, `config/params.yaml`, `README.md`, `tests/*`) | @assistant | CSV·로그·리테스트에서 단일 소스 타임프레임 기록 유지 |
| 2025-10-24/16:45 | 로그 스케일 자본곡선 및 트레이드 요약 로그 도입 (`optimize/metrics.py`, `optimize/alternative_engine.py`, `optimize/main_loop.py`, `optimize/constants.py`) | @assistant | overflow·inf 방지, Trials CSV/콘솔에 Wins·Total/AvgVol 노출 |
| 2025-10-24/14:00 | Gemini 응답 파서 다중 포맷 대응(functionCall/코드블럭/텍스트) 및 모델 호출 경로 로깅, 실패 시 응답 요약 로그 추가 (`optimize/llm.py`, `docs/optimize_structure.md`) | @assistant | HTTP 200 성공 응답 포맷 변동에도 후보 추출 내구성 확보 |

> **작성 팁**: 각 작업자가 변동 사항을 반영할 때 위 타임라인과 표에 함께 추가해 주세요. 작업 전에 문서가 최신인지 확인한 뒤, 변경 요약과 상태 변화를 명확히 남기면 후속 작업이 수월해집니다.

## 5. 향후 요청 시 권장 사용자 지침 예시

향후 에이전트에게 작업을 지시할 때 아래 문구를 사용자 맞춤 지침에 포함하면 묶음 단위 유지보수와 문서 갱신을 자동으로 요구할 수 있습니다.

> "`docs/optimize_structure.md`에 정의된 모듈 묶음을 기준으로 변경 사항을 검토·반영하고, 작업 결과에 따라 구조도 문서의 상태 표와 타임라인을 업데이트해 주세요. 묶음 간 의존 관계나 활성/비활성 상태가 바뀌면 반드시 문서에 기록하고 최종 응답에서 어떤 묶음이 영향받았는지 요약해 주세요."
> "모든 코드 변경과 지표·수식 구현은 파이썬 고유 기능에만 의존하지 말고, TradingView(Pine Script)와 **동일한 결과**가 나오도록 구조화해 주세요. Pine 계산 흐름을 우선 정리하고 동일한 로직을 파이썬으로 재현한 뒤, 로그·테스트를 통해 동등성을 확인하는 절차를 권장합니다."

이 지침을 활용하면 매 작업마다 관련 파일을 한 번에 다루도록 유도하고, 문서화 누락을 방지할 수 있습니다.
