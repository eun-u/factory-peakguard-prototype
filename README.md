# 제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤(자원 최적화 AI 데이터셋)의 소스코드다.
명령 한 줄(`python run_all.py`)로 전처리 → 학습 → 평가 → 오류·조건 분석 → 예측결과 파일 → 그림·표를 모두 다시 만든다.

## 1. 환경

| 항목 | 값 |
| :-- | :-- |
| Python | 3.13 |
| 패키지 | `requirements.txt` (pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.6.0, matplotlib 3.10.6, pyyaml 6.0.2) |
| 난수 | seed 42 고정 (`configs/default.yaml`) |
| 딥러닝·GPU | 사용하지 않음 |
| 한글 그림 | Windows는 맑은 고딕을 자동 사용. 다른 OS는 NanumGothic 또는 Noto Sans CJK KR 설치 권장 |

## 2. 설치와 실행

Windows PowerShell (압축을 푼 폴더에서):

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -X utf8 run_all.py

macOS·Linux:

    python3.13 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -X utf8 run_all.py

단위 점검(합성 자료, 원자료 불필요): `python -m unittest discover -s tests -v`

예상 실행 시간: 4코어 노트북 기준 약 1~2분 (부트스트랩 1,000회 포함). 실제 소요 시간은 `outputs/logs/run_summary.json`의 `runtime_seconds`에 기록된다.

## 3. 입력 데이터

| 경로 | 내용 |
| :-- | :-- |
| `data/raw/task05_power/okm_augumented_2021.csv` | 대회 제공 원자료(학습용 데이터). 6,168행 × 18열, 2021-01-01 ~ 2021-09-14. SHA-256 `8f7af2e4…4674830` |

원자료는 읽기만 한다. 시간 복원·15분 전개 결과를 이 폴더에 쓰지 않는다.

## 4. 폴더 구조

    run_all.py                 전체 실행 진입점
    configs/default.yaml       모든 설정값(분할 비율, 시차, LightGBM 하이퍼파라미터, 피크 정의, 부트스트랩 횟수 등)
    src/
      data.py                  로딩, 시간 오류 48행 복원, 15분 전개, 품질 점검, 변수 사전
      features.py              시차·이동통계·완료 생산량·달력 특징 (특징별 가용 시점 표 포함)
      split.py                 시간순 분할, 경계 간격, rolling-origin 폴드
      evaluate.py              지표, 피크 에피소드 1:1 매칭, 날짜 블록 부트스트랩
      daily_max.py             보조 과제: 익일 일간 최대 15분 전력
      viz.py                   그림 (한글 폰트 설정은 이 파일에서만)
      models/                  기준선(계절 나이브·persistence), LightGBM 점·분위수·분류기
      analysis/                rolling-origin·예측거리·conformal, 조건별 FN·FP, 영향변수,
                               피크 발생 조건, 경보 규칙, 비용-손실 REV, 생산량 이동 시뮬레이션
    tests/                     누수·분할·매칭 단위 점검과 합성 자료 전체 실행 점검
    outputs/                   실행 결과 (아래 5절)

## 5. 산출물

| 파일 | 내용 | 보고서 |
| :-- | :-- | :-- |
| `outputs/predictions/test_predictions_1h.csv` | **테스트 구간 예측결과(주 과제)**: 원점, 목표 시각, 실제값, 계절 나이브·persistence·LightGBM 예측, 분위수(10·50·90·95%), 초과확률, 경보 여부 | 2장 |
| `outputs/predictions/test_predictions_daily_max.csv` | **테스트 구간 예측결과(보조 과제)**: 익일 일간 최대 15분 전력 | 2장 |
| `outputs/tables/final_test.csv` | 최종 모델 비교표 (피크 위치·에피소드·날짜 단위 지표와 95% CI, TP/FP/FN) | 2장 |
| `outputs/tables/t1_*.csv` | 변수 사전, 0값 연속 구간, 월·요일·시각 분포, 변수 간 관계 | 1장 |
| `outputs/tables/t2_*.csv` | 기준선 선택, 개선률, 분위수·확률 지표, 피크 정의 민감도, rolling-origin, 예측거리 곡선, conformal, 익일 최대 | 2장 |
| `outputs/tables/t3_*.csv` | 조건별 FN·FP, 저녁 미탐 사례, permutation importance, 부분의존, 피크 발생 조건 | 3장 |
| `outputs/tables/t4_*.csv` | 경보 규칙 KPI, REV, 생산량–전력 기울기, 이동 시뮬레이션 | 4장 |
| `outputs/figures/f*.png` | 보고서 그림 (파일 이름 앞 숫자가 장 번호) | 1~4장 |
| `outputs/logs/run_summary.json` | 분할 범위, 피크 경계, 선택된 기준선·임계값, 재현 점검 결과, 실행 환경·시간 | 6장 |

## 6. 평가 설계 요약

- 주 과제: 원점 t에서 **1시간 후 15분 전력** 예측. 보조 과제: 전날 23:45에 **다음 날 일간 최대 15분 전력** 예측.
- 분할: 시간순 학습 70% / 검증 15% / 테스트 15%. 경계마다 예측거리(4개 원점)만큼 비운다. 랜덤 분할 없음.
- 피크: 테스트 실제값 > 학습 구간 상위 5% 경계. 경보 임계값은 검증 구간 F1 최대로 고정한다.
- 신뢰구간: 테스트 날짜 단위 1,000회 복원추출의 95% 백분위 구간.
- 모델 설정은 사전 검증 때 고정했고, 이후 테스트 결과를 보고 바꾸지 않았다. 예측거리 곡선과 conformal 보정은 테스트 이전 개발 폴드에서만 평가한다.

## 7. 가정

| ID | 가정 |
| :-- | :-- |
| A1 | 주 과제는 1시간 후 15분 전력, 보조 과제는 익일 일간 최대 15분 전력 (출제문의 '지정된 시간구간' 미확정) |
| A3 | 전력 단위(kW·kWh)는 확인되지 않았다. 모든 수치는 원자료 단위로 쓴다 |
| A4 | `15분`·`30분`·`45분`·`60분` 열은 HH:15, HH:30, HH:45, (HH+1):00에 끝나는 연속 15분 구간이다 |
| A5 | 피크는 학습 구간 상위 5% 초과(통계적 사건)이며 계약전력 초과가 아니다. 상위 2.5%·10%도 민감도로 보고한다 |
| A6 | 설비 동시가동·제품 전환·교대 정보가 없어 시각대(00–08·08–16·16–24), 요일, 생산량 구간을 대리변수로 쓴다 |
| A7 | 별도 테스트 파일이 없어 마지막 15% 구간 예측을 두 과제 각각 CSV로 낸다 |
| A8 | 생산량은 그 시간이 끝난 뒤에만 안다. 미래 생산량, 실측 기상, 같은 시간 평균 전력은 입력에서 뺀다 |
| — | 2021-07-13·15의 시간 열 48행은 0~23 범위를 벗어나 날짜별 행 순서로 임시 복원했고, 해당 위치와 이에 의존하는 특징 행은 학습·평가에서 제외했다 |
