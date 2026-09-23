# 01 데이터·파일 인벤토리

이 파일은 실행 시점의 프로젝트 스냅샷이다. `.git`, `.venv`, `__pycache__` 등 실행환경 내부 파일은 제외했다. ZIP 내부 CSV와 풀린 CSV는 별도 관측자료로 합산하지 않는다. 실행 뒤 생성되는 결과 파일은 다음 재실행에서 목록에 나타날 수 있다.

## 전체 파일 목록

파일 | 바이트 | SHA-256 앞 12자리 | 형태
:-- | --: | :-- | :--
.gitignore | 59 | 84704da52f22 | 코드/설정/기타
.streamlit/config.toml | 140 | c64cc6872348 | 코드/설정/기타
3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/outlier_data.csv | 38,823 | 9fad8c238e63 | CSV 표
3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/press_data_normal.csv | 1,211,960 | f2d61cb3b310 | CSV 표
5. 자원 최적화 AI 데이터셋/5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv | 437,447 | 8f7af2e49366 | CSV 표
app.py | 9,644 | b7ca8beae592 | 코드/설정/기타
data/source.zip | 116,833 | 02640b4cc909 | ZIP 자료
forecast.py | 10,123 | 068bbc73b500 | 코드/설정/기타
K제조AI_5번_전력피크예측_의사결정기록_20260922.docx | 62,585 | 35b265b84c94 | 문서
README.md | 4,980 | 3df155626aaf | 문서
requirements.txt | 66 | b9a22db6db69 | 문서
tests/test_forecast.py | 3,320 | 617280a5ce02 | 코드/설정/기타
verification/common.py | 8,477 | e0ad6857fc34 | 코드/설정/기타
verification/README.md | 921 | 2eaac1059869 | 문서
verification/report.py | 19,746 | bff4ce2e2357 | 코드/설정/기타
verification/requirements.txt | 82 | 60ce32477104 | 문서
verification/results/00_decision_criteria.md | 990 | 7742dd938e65 | 문서
verification/results/01_inventory.md | self | self | 문서
verification/results/03_gaps.csv | 2,449,350 | 3751ef8dbf4c | CSV 표
verification/results/03_p1_univariate.csv | 1,757 | 0b5d0adbd66a | CSV 표
verification/results/03_p2_models.csv | 1,520 | 86e9fd0e25eb | CSV 표
verification/results/03_p3_level_relation.png | 43,085 | b0d163e53e42 | 코드/설정/기타
verification/results/03_p3_relation.csv | 491 | bfc1086e7105 | CSV 표
verification/results/03_p4_injections.csv | 1,550 | 029dc33fecc8 | CSV 표
verification/results/03_p5_alarms.csv | 1,535 | fa640da0e918 | CSV 표
verification/results/03_press.json | 22,814 | 933e1ad4a304 | 코드/설정/기타
verification/results/03_press.md | 2,661 | 1e1580c81196 | 문서
verification/results/03_section.md | 2,661 | 1e1580c81196 | 문서
verification/results/03_segments.csv | 109,436 | e29123351263 | CSV 표
verification/results/03_summary.json | 807 | 2a16f0d10f4d | 코드/설정/기타
verification/results/05_daily_test_predictions.csv | 1,557 | dc2f06b1f9fd | CSV 표
verification/results/05_error_conditions.csv | 6,773 | 94547ef8e7b9 | CSV 표
verification/results/05_forecasts.png | 135,377 | f2d67d9ece31 | 코드/설정/기타
verification/results/05_reliability.png | 51,826 | de31878e2b8e | 코드/설정/기타
verification/results/05_rev.png | 37,637 | 8d653a8bf11d | 코드/설정/기타
verification/results/05_rolling_origin.csv | 2,291 | 48b57a5f69b5 | CSV 표
verification/results/05_section.md | 7,079 | 8d618f074620 | 문서
verification/results/05_summary.json | 20,136 | 5848dcca5009 | 코드/설정/기타
verification/results/05_test_predictions.csv | 806,214 | 7042fa678fce | CSV 표
verification/results/report.md | 20,696 | 51d264c81b77 | 문서
verification/run_all.py | 526 | 37c82eee837b | 코드/설정/기타
verification/t3_press.py | 33,321 | 951a90243f76 | 코드/설정/기타
verification/t5_power.py | 40,046 | f7c4a9f755ec | 코드/설정/기타

## 표 파일의 구조

### 3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/outlier_data.csv

- 인코딩: utf-8-sig
- 행×열: 600 × 6
- 시간 범위: 2022-07-17 10:51:07.943000–2022-07-17 10:53:53.540000 (파싱 실패 0)
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  Unnamed: 0 | int64 | 0 | 600
  TimeStamp | object | 0 | 600
  AI0_Vibration | float64 | 0 | 594
  AI1_Vibration | float64 | 0 | 597
  AI2_Current | float64 | 0 | 340
  Equipment_state | int64 | 0 | 1

### 3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/press_data_normal.csv

- 인코딩: utf-8-sig
- 행×열: 20,000 × 6
- 시간 범위: 2022-07-12 00:00:00.019000–2022-07-12 01:16:55.828000 (파싱 실패 0)
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  Unnamed: 0 | int64 | 0 | 20,000
  TimeStamp | object | 0 | 19,999
  AI0_Vibration | float64 | 0 | 19,124
  AI1_Vibration | float64 | 0 | 19,460
  AI2_Current | float64 | 0 | 19,934
  Equipment_state | int64 | 0 | 1

### 5. 자원 최적화 AI 데이터셋/5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv

- 인코딩: utf-8-sig
- 행×열: 6,168 × 18
- 시간 범위: 날짜 2021-01-01 00:00:00–2021-09-14 00:00:00 (파싱 실패 0); 원본 시간 0–188, 0–23 외 48행
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  날짜 | int64 | 0 | 257
  시간 | int64 | 0 | 60
  15분 | int64 | 0 | 166
  30분 | int64 | 0 | 185
  45분 | int64 | 0 | 187
  60분 | int64 | 0 | 183
  평균 | int64 | 0 | 176
  생산량 | int64 | 0 | 1,637
  기온 | float64 | 0 | 439
  풍속 | float64 | 3 | 70
  습도 | int64 | 0 | 90
  강수량 | float64 | 1 | 244
  전기요금(계절) | float64 | 0 | 3
  day | int64 | 0 | 7
  d | int64 | 0 | 31
  m | int64 | 0 | 9
  공장인원 | float64 | 17 | 3,432
  인건비 | float64 | 0 | 2

### data/source.zip::5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv

- ZIP 내부 압축 전 크기: 437,447바이트; 인코딩: utf-8-sig; 동일한 비압축 사본: 5. 자원 최적화 AI 데이터셋/5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv
- 행×열: 6,168 × 18
- 시간 범위: 날짜 2021-01-01 00:00:00–2021-09-14 00:00:00 (파싱 실패 0); 원본 시간 0–188, 0–23 외 48행
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  날짜 | int64 | 0 | 257
  시간 | int64 | 0 | 60
  15분 | int64 | 0 | 166
  30분 | int64 | 0 | 185
  45분 | int64 | 0 | 187
  60분 | int64 | 0 | 183
  평균 | int64 | 0 | 176
  생산량 | int64 | 0 | 1,637
  기온 | float64 | 0 | 439
  풍속 | float64 | 3 | 70
  습도 | int64 | 0 | 90
  강수량 | float64 | 1 | 244
  전기요금(계절) | float64 | 0 | 3
  day | int64 | 0 | 7
  d | int64 | 0 | 31
  m | int64 | 0 | 9
  공장인원 | float64 | 17 | 3,432
  인건비 | float64 | 0 | 2

### verification/results/03_gaps.csv

- 인코딩: utf-8-sig
- 행×열: 20,598 × 3
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 20,012
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  source | object | 0 | 2
  label | int64 | 0 | 2
  gap_seconds | float64 | 0 | 582

### verification/results/03_p1_univariate.csv

- 인코딩: utf-8-sig
- 행×열: 13 × 6
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  feature | object | 0 | 13
  direction | int64 | 0 | 2
  auroc | float64 | 0 | 13
  ci_low | float64 | 0 | 13
  ci_high | float64 | 0 | 13
  warning | object | 0 | 1

### verification/results/03_p2_models.csv

- 인코딩: utf-8-sig
- 행×열: 12 × 8
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  representation | object | 0 | 4
  model | object | 0 | 3
  auroc | float64 | 0 | 11
  auroc_ci_low | float64 | 0 | 12
  auroc_ci_high | float64 | 0 | 7
  pr_auc | float64 | 0 | 12
  pr_auc_ci_low | float64 | 0 | 12
  pr_auc_ci_high | float64 | 0 | 7

### verification/results/03_p3_relation.csv

- 인코딩: utf-8-sig
- 행×열: 4 × 7
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  score | object | 0 | 4
  auroc | float64 | 0 | 4
  auroc_ci_low | float64 | 0 | 4
  auroc_ci_high | float64 | 0 | 2
  pr_auc | float64 | 0 | 4
  pr_auc_ci_low | float64 | 0 | 4
  pr_auc_ci_high | float64 | 0 | 2

### verification/results/03_p4_injections.csv

- 인코딩: utf-8-sig
- 행×열: 16 × 6
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  injection | object | 0 | 8
  score | object | 0 | 2
  auroc | float64 | 0 | 16
  ci_low | float64 | 0 | 16
  ci_high | float64 | 0 | 15
  n_pairs | int64 | 0 | 1

### verification/results/03_p5_alarms.csv

- 인코딩: utf-8-sig
- 행×열: 8 × 14
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  alpha | float64 | 0 | 2
  threshold | float64 | 0 | 2
  rule | object | 0 | 4
  normal_false_alarm_fraction | float64 | 0 | 3
  false_alarm_ci95 | object | 0 | 3
  false_alarm_events | int64 | 0 | 3
  alarms_per_operating_hour | float64 | 0 | 3
  alarms_per_hour_ci95 | object | 0 | 3
  chattering_transitions | int64 | 0 | 3
  abnormal_segment_detection_fraction | float64 | 0 | 4
  detection_ci95 | object | 0 | 4
  delay_segments_from_abnormal_start | int64 | 0 | 4
  delay_seconds_from_abnormal_start | float64 | 0 | 4
  delay_ci95 | object | 0 | 1

### verification/results/03_segments.csv

- 인코딩: utf-8-sig
- 행×열: 620 × 7
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  segment_id | int64 | 0 | 620
  source | object | 0 | 2
  label | int64 | 0 | 2
  start | object | 0 | 620
  end | object | 0 | 620
  n_samples | int64 | 0 | 50
  duration_seconds | float64 | 0 | 50

### verification/results/05_daily_test_predictions.csv

- 인코딩: utf-8-sig
- 행×열: 36 × 4
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  forecast_origin_day | object | 0 | 36
  y | float64 | 0 | 27
  naive | float64 | 0 | 25
  lgb | float64 | 0 | 32

### verification/results/05_error_conditions.csv

- 인코딩: utf-8-sig
- 행×열: 39 × 16
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  factor | object | 0 | 5
  value | object | 0 | 39
  n | int64 | 0 | 15
  mae | float64 | 0 | 39
  mean_residual | float64 | 0 | 39
  events | int64 | 0 | 22
  fn | int64 | 0 | 21
  fp | int64 | 0 | 22
  recall | float64 | 16 | 21
  fp_rate | float64 | 0 | 26
  mae_ci_low | float64 | 0 | 39
  mae_ci_high | float64 | 0 | 39
  recall_ci_low | float64 | 16 | 21
  recall_ci_high | float64 | 16 | 21
  fp_rate_ci_low | float64 | 0 | 23
  fp_rate_ci_high | float64 | 0 | 26

### verification/results/05_rolling_origin.csv

- 인코딩: utf-8-sig
- 행×열: 12 × 14
- 시간 범위: 시간 열 식별 불가
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  fold | int64 | 0 | 3
  model | object | 0 | 4
  training_rows | int64 | 0 | 3
  evaluation_rows | int64 | 0 | 2
  evaluation_first | object | 0 | 3
  evaluation_last | object | 0 | 3
  peak_threshold | float64 | 0 | 3
  peak_events | int64 | 0 | 3
  mae | float64 | 0 | 12
  mae_ci_low | float64 | 0 | 12
  mae_ci_high | float64 | 0 | 12
  peak_mae | float64 | 0 | 12
  peak_mae_ci_low | float64 | 0 | 12
  peak_mae_ci_high | float64 | 0 | 12

### verification/results/05_test_predictions.csv

- 인코딩: utf-8-sig
- 행×열: 3,510 × 16
- 시간 범위: 2021-08-09 10:45:00–2021-09-15 00:00:00 (파싱 실패 0)
- 완전 중복 행: 0
- 컬럼 | dtype | 결측 수 | 고유값 수
  :-- | :-- | --: | --:
  forecast_origin | object | 0 | 3,510
  y | float64 | 0 | 184
  day | float64 | 0 | 186
  week | float64 | 0 | 185
  average | float64 | 0 | 351
  naive | float64 | 0 | 185
  lgb | float64 | 0 | 3,283
  q10 | float64 | 0 | 3,432
  q50 | float64 | 0 | 3,355
  q90 | float64 | 0 | 3,043
  q95 | float64 | 0 | 2,850
  classifier_prob | float64 | 0 | 3,062
  quantile_prob | float64 | 0 | 787
  climate_prob | float64 | 0 | 21
  target_time | object | 0 | 3,510
  production_target | float64 | 0 | 509

## 설명 문서와 변수 정의

### K제조AI_5번_전력피크예측_의사결정기록_20260922.docx

- DOCX 문단 607개를 확인했다. 내부 의사결정·탐색 기록이며 공식 데이터 사전으로 확인되지 않았다.
- ⑤: `날짜`·`시간`은 시간축, `15분`·`30분`·`45분`·`60분`은 시간별 네 전력 위치, `평균`은 그 파생 평균으로 기술한다. `생산량`은 생산 실적/계획 여부 미확정, `기온`·`풍속`·`습도`·`강수량`은 환경 관측, `전기요금(계절)`·`인건비`는 비용·시간 대리 가능성, `day`·`d`·`m`은 달력 파생, `공장인원`은 생성관계 의심이다.
- ⑤의 전력 kW/kWh 단위와 네 15분 열의 구간 시작·종료 의미는 문서도 미확정으로 표기한다. 미래 생산량·미래 실측 기상은 사전에 안다고 볼 근거가 없다.
- ③은 이전 대화 탐색값(정상 20,000행·이상 600행, 파일별 라벨 고정, 짧은 공백 분절)을 기록하지만 재현 코드·원자료가 이 문서에 없다. 공식 변수 정의와 원파형 여부는 확인되지 않았다.
- 이 문서의 ⑤ 선택 결론과 과거 성능 수치는 이번 중립적 타당성 검증의 결론·재현 수치로 채택하지 않는다.

## 과제 판별

- ⑤: `okm_augumented_2021.csv`는 `15분`·`30분`·`45분`·`60분`, `생산량`, `전기요금(계절)`을 포함하고 ⑤ 폴더/ZIP에 위치한다. 동일한 압축·비압축 사본은 중복 입력으로 취급하지 않는다.
- ③: AI0/AI1 진동·AI2 전류 컬럼으로 확인: 3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/outlier_data.csv, 3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/press_data_normal.csv
