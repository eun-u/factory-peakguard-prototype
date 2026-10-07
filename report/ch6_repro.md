# 제6장 코드 구성 및 재현성

> **이 장의 주장.** 평가자는 제출 ZIP과 README만으로 전처리부터 학습·추론·결과 생성까지 명령 한 줄로 다시 실행할 수 있고, 같은 결과를 얻는다.

## 6.1 실행 방법

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -X utf8 run_all.py

macOS·Linux는 README 2절의 같은 명령을 쓴다. 원자료 없이 동작을 점검하려면 `python -m unittest discover -s tests -v`를 실행한다(합성 자료 사용).

## 6.2 실행 환경

| 항목 | 값 |
| :-- | :-- |
| Python | 3.13 |
| 패키지 | pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.6.0, matplotlib 3.10.6, pyyaml 6.0.2 (`requirements.txt`에 버전 고정) |
| 난수 | seed 42 (LightGBM, 부트스트랩, permutation importance) |
| 설정 | 모든 설정값을 `configs/default.yaml` 한 곳에서 읽음 |
| 실행 시간 | 〔R: `run_summary.json`의 `runtime_seconds`〕. 같은 크기의 합성 자료에서 4코어 기준 약 65초 |
| 실행 기록 | `outputs/logs/run_summary.json`: 원자료 SHA-256, 분할 범위, 피크 경계, 선택된 기준선·임계값, 재현 점검, 패키지 버전, 실행 시간 |

## 6.3 코드 구성과 실행 순서

`run_all.py`는 아래 순서로 실행하며, 각 단계의 출력이 보고서의 어느 표·그림이 되는지 정해져 있다.

| 단계 | 모듈 | 내용 | 출력 → 보고서 |
| :-- | :-- | :-- | :-- |
| 1 | `src/data.py` | 원자료 로딩, 해시 확인, 시간 오류 복원, 15분 전개, 품질 점검, 변수 사전 | `t1_*.csv`, 그림 1-1·1-2 → 1장 |
| 2 | `src/features.py`, `src/split.py` | 가용 시점을 지킨 특징, 시간순 분할과 경계 간격(assert) | — |
| 3 | `src/models/` | 기준선 6종, LightGBM 점·분위수·분류기 학습 | — |
| 4 | `src/evaluate.py` | 검증에서 기준선·임계값 선택, 테스트 평가, 에피소드 매칭, 날짜 블록 부트스트랩 | `final_test.csv`, `t2_*.csv`, 그림 2-1~2-3 → 2장 |
| 5 | `src/analysis/horizon_curve.py` | 개발 폴드: rolling-origin, 예측거리 곡선, conformal | `t2_rolling_origin.csv` 등, 그림 2-4 → 2장 |
| 6 | `src/daily_max.py` | 보조 과제 | `t2_daily_max.csv` → 2장 |
| 7 | `src/analysis/errors.py`, `importance.py`, `peak_conditions.py` | 조건별 FN·FP, 저녁 미탐 사례, 영향변수, 피크 조건 | `t3_*.csv`, 그림 3-1~3-4 → 3장 |
| 8 | `src/analysis/alerting.py`, `decision.py`, `simulate_shift.py` | 경보 규칙, REV, 생산량 이동 시뮬레이션 | `t4_*.csv`, 그림 4-1~4-3 → 4장 |
| 9 | `run_all.py` | 예측결과 파일, 실행 기록 | `outputs/predictions/`, `outputs/logs/` → 6장 |

## 6.4 테스트데이터 예측결과 파일

주최측이 별도의 테스트 파일이나 제출 양식을 제공하지 않았으므로(가정 A7), 시간순 마지막 15% 구간의 예측을 두 과제 각각 CSV로 제출한다.

| 파일 | 행 | 열 |
| :-- | --: | :-- |
| `outputs/predictions/test_predictions_1h.csv` | 3,510 | `forecast_origin`, `target_time`, `y`, `seasonal`, `persistence`, `lgb`, `q10`, `q50`, `q90`, `q95`, `classifier_prob`, `quantile_prob`, `lgb_alarm`, `actual_peak` |
| `outputs/predictions/test_predictions_daily_max.csv` | 36 | `forecast_origin_day`, `y`, `naive`, `lgb` |

## 6.5 재현성 검증

| 점검 | 방법 | 결과 |
| :-- | :-- | :-- |
| 사전 검증과의 동일성 | 같은 합성 입력에 사전 검증 코드와 제출 코드를 각각 실행해 예측값·지표·구간·rolling-origin·조건별 집계·익일 최대 126개 항목 비교 | 126개 모두 일치 (2026-10-07) |
| 원자료 재현 | `run_summary.json`의 `reproduction_check`: persistence 27.73, LightGBM 17.77(피크 위치 MAE)과 소수 둘째 자리 일치 | 〔R〕 |
| ZIP 재실행 | 제출 ZIP을 새 폴더에 풀어 다시 실행한 결과를 원래 출력과 비교 | 합성 자료에서 `final_test.csv`·예측 파일 완전 일치. 원자료 〔R〕 |
| 단위 점검 | 15분 전개·시간 복원, 원점 이후 값 변경 시 특징 불변, 생산량 가용 시점, 분할 간격, 에피소드 1:1 매칭, k/n 규칙, conformal 보정량, 합성 자료 전체 실행 (8개) | 통과 |
| 블라인드 점검 | `tools/package_submission.py`가 ZIP에 담길 텍스트 파일에서 이메일·개인 경로·기관명 패턴을 찾으면 ZIP 생성을 중단 | 통과 (원자료 실행 후 재확인 〔R〕) |

## 6.6 제출 ZIP 구성

`python tools/package_submission.py`가 허용 목록의 파일만 담아 `submission/source/task5_source.zip`을 만든다.

| 포함 | 제외 |
| :-- | :-- |
| `README.md`, `requirements.txt`, `run_all.py`, `configs/`, `src/`, `tests/`, 학습용 데이터 `data/raw/task05_power/okm_augumented_2021.csv`, `outputs/`(예측결과·표·그림·실행 기록) | 사전 검증 코드·결과, 화면 시제품, 내부 의사결정 문서, 다른 과제 자료, 보고서 초안 |

## 6.7 판정

주장은 **지지**한다. 합성 자료에서 동일성·ZIP 재실행·단위 점검을 확인했다. 원자료에서의 수치 재현과 실행 시간은 `run_all.py` 실행으로 확정한다 〔R〕.
