# 작업 메모 (2025-10-19)

- `optimize/strategy_model.py`  
  - Pine `useWallet` 처리와 Python 엔진을 동기화했습니다. 이익 비축분은 `Withdrawable` 필드로 추적되며, 포지션 사이징 시 자동으로 제외됩니다.  
  - 총자산 계산이 Pine 스크립트와 동일하게 `TotalAssets == tradableCapital + withdrawable` 관계를 유지하도록 정리했습니다.  
  - 승률 90 % 이상(또는 100 % + 과도한 자산 상승) 시 `AnomalyReason=win_rate_high`/`win_rate_spike`로 실행을 무효화합니다.
- `optimize/main_loop.py`, `optimize/metrics.py`  
  - `Withdrawable` 가중 평균을 리포트에 반영하고, `Savings` 는 legacy 필드로 유지하되 지갑 모드에서는 0으로 고정합니다.
- `tools/compare_pine_csv.py`  
  - Pine CSV에 `withdrawable` 컬럼이 있을 경우 자동으로 매핑해 비교합니다.
- 테스트  
  - `tests/test_strategy_model.py`, `tests/test_metrics.py` 를 새 지갑 로직에 맞추어 갱신했습니다.

# 작업 메모 (2025-03-28)

- `optimize/main_loop.py`  
  - `DatasetSpec`에 `chart_timeframe`·`chart_df` 필드를 추가해 기준 차트 타임프레임을 명시적으로 전달합니다.  
  - 데이터셋 로딩 시 기준 타임프레임을 기록하며, 캐시·병렬 실행 루트를 모두 지원하도록 보강했습니다.  
  - 프로세스 워커 공유 객체에도 동일한 메타 정보를 직렬화합니다.
- `--no-interactive` 플래그로 비대화형 실행을 선택하면 첫 번째 LTF 후보를 자동 선택해 CLI가 입력 대기 상태로 멈추지 않도록 했습니다.

- `optimize/strategy_model.py`  
  - `run_backtest` 가 `chart_df` 파라미터를 수용해 파인 스크립트의 합성 바 로직과 1:1 대응하도록 전체 ATR/스탑 계산 경로를 차트 프레임 기준으로 재정렬했습니다.  
  - 샹들리에, 피벗, 스윙 하이/로우, ATR 스탑 길이 등 모든 리스크/출구 계산이 chart TF에 동기화됩니다.

- `optimize/wrapper_strategy_model.py`  
  - 래퍼와 호환 alias 모두 `chart_df` 를 패스하도록 시그니처 업데이트.

- `optimize/report.py`  
  - `results.csv` 집계 행의 첫 번째 컬럼을 `timeframe`으로 고정해 어떤 분봉 데이터가 사용됐는지 Python 출력만으로도 확인할 수 있습니다.  
  - `results_datasets.csv`와 `best.json`에도 실행에 참여한 분봉 목록을 그대로 보존하고, 혼합 실행 시에는 `1,3,5merged` 라벨을 사용합니다.

- `tools/compare_pine_csv.py` (신규)  
  - 파인 CSV와 Python 리포트를 비교해 핵심 메트릭·종료 사유 일치 여부를 검증하는 스크립트를 추가했습니다.  
  - `tests/test_compare_pine.py`에서 Metric/Reason 비교 로직을 단위 테스트합니다.

- 테스트  
  - `tests/test_strategy_model.py`, `tests/test_report.py`, `tests/test_compare_pine.py` 통과 (Pytest).  
  - 리포트 컬럼 검증은 `Liquidations` 존재 여부를 확인하도록 수정.

## 다음 단계
1. `results_timeframe_summary.csv`/`results_timeframe_rankings.csv`에서 혼합 실행 표기를 더 풍부하게 노출할지 검토.
2. `tools/compare_pine_csv.py`를 CI 혹은 pytest marker와 연동해 파인 CSV가 제공되면 자동으로 비교 리포트를 남기도록 자동화.
3. 실제 파인 CSV 샘플과 Python 리포트를 비교해 샹들리에·피라미딩·수수료/슬리피지 계산이 1:1인지 확인하고, 필요하면 스크립트에 추가 검증 항목(MFE/MAE, 피라미딩 횟수 등)을 확장.
4. README 혹은 별도 가이드에 새 비교 스크립트 사용법과 `timeframe` 컬럼 의미를 문서화.
