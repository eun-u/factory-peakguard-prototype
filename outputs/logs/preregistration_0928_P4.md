# Phase 4 사전 고정 — 오픈소스 기준 모델 보강 (E1·E2·E3·E4·E5)

작성 2026-09-28 (KST). 이 파일은 후보 예측을 한 건도 채점하기 전에 작성했으며, 작성 후 수정하지 않는다. 오탈자·코드 버그 수정이 필요하면 별도 부록 파일에 날짜·사유·전후 수치를 남긴다. 결과를 본 뒤 후보·탐색 범위·채택 문턱을 바꾸지 않는다.

## 0. 목적과 범위

- 목적: 기존 선정(development_selection.json)이 모델 탐색 부족 때문에 약한지, 데이터 한계 때문인지를 오픈소스·대회 관행 수준의 비교군으로 확인한다. 참고: OpenSTEF(LightGBM·분위수 운영), Nixtla statsforecast(MSTL), Taylor(2003) 이중 계절 Holt-Winters, GEFCom 계열 예측 결합, Chronos-2(사전학습 시계열 모델).
- 입력: `src.session_data.load_development_history`의 봉인 개발 prefix만 사용한다. 목표 시각 < 2021-08-09 09:45. 테스트 구간은 읽지 않는다.
- 분할: 기존 `src.split.make_splits`와 `src.training._partition`, 폴드별 τ 산정 절차를 그대로 재사용한다. 각 거리·폴드의 fit/stop/cal/score 원점 집합이 기존 OOF와 일치하는지 먼저 검사하고, 불일치하면 실행을 멈춘다.
- 코드 위치: `research_p4/`와 `scripts/run_p4_*.py`. `src/`, `configs/default.yaml`, 기존 사전등록·채택 기준·평가 프로토콜은 수정하지 않는다(개발 캐시 지문 보존).
- 예측거리: h1(15분), h4(1시간), h16(4시간), h96(24시간). T2(익일 최대)는 범위 밖.

## 1. 기존 선정(incumbent)

| 거리 | 점예측 incumbent | CBL 대표 | 위험 incumbent |
|---|---|---|---|
| h1 | p1_latest | c2a_max_4_5_adjusted | lgbm_quantile_b |
| h4 | lgbm_no_holiday_weight_2 | c2a_max_4_5_adjusted | lgbm_quantile_b |
| h16 | lgbm_residual_cbl | c3_holiday_hybrid | lgbm_quantile_b |
| h96 | c3_holiday_hybrid | c3_holiday_hybrid | lgbm_quantile_b |

incumbent 예측은 `outputs/predictions/p2_combined_oof.csv`에서 읽는다.

## 2. 점예측 후보 (거리마다 6개 + 참고 1개)

모든 후보는 원점 이하 관측만 사용한다. 보정(cal) 구간에서 기존 `_cutoff`로 경보 임계를 정하고, score 구간 행을 기존 OOF 스키마로 저장한다.

- **P4-1 `p4_dshw` (E1a)**: Taylor(2003) 가법 이중 계절 Holt-Winters(일 96, 주 672) + AR(1) 오차 보정. 모수 α,δ,ω∈[0,1], φ∈[0,0.99]를 fit 구간 끝까지의 관측으로 1단계 제곱오차 최소화(L-BFGS-B, 시작값 α=.1,δ=.2,ω=.2,φ=.5). 초기값은 첫 14일. 결측·복원 관측은 갱신하지 않고 1단계 예측으로 대체한다. 원점 t의 h단계 예측은 l_t + d_{t+h−96} + w_{t+h−672} + φ^h·e_t.
- **P4-2 `p4_mstl_daily` (E1b)**: statsforecast MSTL(season_length=[96,672], trend_forecaster=AutoETS(model="ZZN")). 원점당 재적합은 CPU 비용(약 3초/회)으로 불가하여, 매일 00:00 이전 28일 봉인 관측으로 하루 한 번 적합한다. 원점 t 예측 = MSTL_day(t+h) + φ^h·(y_t − MSTL_day(t)). φ는 해당 폴드 fit 구간의 (y − MSTL_day) 1시차 자기상관(0~0.99로 절단). 창 안 결측은 과거 관측만으로 선형 보간한다.
- **P4-3 `p4_blend2` (E3a)**: w·LGBM + (1−w)·CBL대표, w∈{0, .05, …, 1}. cal 구간 전체 MAE 최소 w(동률은 작은 w).
- **P4-4 `p4_blend3` (E3b)**: {LGBM, CBL대표, P4-1} 가중합, 가중치는 합 1·음수 없음·0.1 격자. cal 구간 전체 MAE 최소(동률은 격자 순서 첫 항).
  - P4-3/4의 LGBM 구성요소는 해당 폴드 chosen_params로 재학습한 `lgbm_no_holiday_weight_2`. 기존 OOF와의 최대 절대차를 보고한다.
- **P4-5 `p4_lgbm_tuned` (E2a)**: Optuna TPE(seed 42) 40회, 목적은 stop 구간 MAE. 공간: num_leaves 15~127(정수), min_child_samples 20~200(정수), learning_rate 0.02~0.10(로그), feature_fraction 0.6~1.0, bagging_fraction 0.6~1.0(bagging_freq=1), lambda_l2 0~10. n_estimators 최대 600, 조기종료 60(stop). 휴일 특징 제외, 피크 가중 2. 최종 예측은 최적 설정으로 seed 42~46 다섯 모델의 평균.
- **P4-6 `p4_lgbm_delta` (E2b)**: P4-5와 같은 폴드별 최적 설정·특징·가중·5 seed 평균. 목표는 y − current, 예측은 current + f. 추가 탐색 없음.
- **P4-R `p4_chronos2_ref` (E5, 참고 전용)**: amazon/chronos-2 학습 없이 사용. 입력은 원점 이하 봉인 전력 최근 2048개(결측·복원은 NaN), 단변량, 중앙값의 h번째 단계. 채택 대상이 아니며 Holm 가족에 넣지 않는다. 모델 식별자·패키지 버전·torch 버전을 기록한다.

## 3. 위험 후보 (E4, 거리마다 1개)

- **P4-Q `p4_quantile_dense`**: 기존 분위수 모델과 같은 특징·chosen_params로 LightGBM 분위수 {.05,.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99}. .9 이상에 기존 Mondrian B 보정(같은 경계·같은 cal 전반부)을 적용하고 교차를 제거한다. p_exceed는 기존 `exceedance_from_quantiles`, 경보 임계는 기존 cal 후반부 규칙.

## 4. 채택 규칙

점예측 후보가 incumbent를 대체하려면 아래를 모두 만족해야 한다.

1. 피크 위치 MAE 개선(incumbent − 후보, 같은 유효 행, y>τ)이 양수이고, 날짜 블록 1000회(seed 42) 중심화 단방향 p의 Holm 조정값 < 0.05.
2. 에피소드 F1(통합) ≥ incumbent − 0.02, 위치 오경보 수 ≤ incumbent × 1.20.
3. 마지막 폴드 피크 MAE ≤ incumbent × 1.10, 그리고 모든 폴드 피크 MAE ≤ incumbent × 1.50.
4. 여러 후보가 통과하면 피크 MAE 개선 추정치가 가장 큰 후보. 동률이면 단순한 후보(P4-3 < P4-4 < P4-1 < P4-2 < P4-5 < P4-6 순).

위험 후보는 아래를 모두 만족해야 lgbm_quantile_b를 대체한다.

1. Brier 점수 개선(B − 후보)이 양수이고 Holm 조정 p < 0.05.
2. 상위 구간 q95 커버리지 절대오차 ≤ B + 0.01.
3. 공통 분위수 {.1,.5,.9,.95,.975} 평균 pinball ≤ B × 1.05.
4. 경보 에피소드 F1 ≥ B − 0.02.

Holm 가족: 점예측 24개(6후보×4거리) + 위험 4개 = 28개. 실행하지 못한 검정은 p=1로 포함한다. CI가 0을 포함하면 "미입증"으로 쓴다. 모든 후보는 CBL 대표·persistence 대표 대비 피크 MAE 개선과 CI도 함께 보고한다(선정에는 쓰지 않음).

## 5. 중단·통합

- 모든 후보 결과는 2026-09-30 23:59 KST까지 산출한다. 그때까지 산출하지 못한 후보는 미실행, p=1.
- 채택 후보가 생기면 `src/`에 통합하고 전체 개발 CV를 다시 실행·재봉인한다. 2026-10-01 12:00 KST까지 재현(원 후보 OOF와 1e-6 이내 일치)을 끝내지 못하면 기존 선정을 유지하고 "개발 채택·미통합"으로 기록한다.
- 동결 일시(10/1 18:00 KST)와 사람 승인 절차는 바꾸지 않는다.
- 부정적 결과를 포함해 전 후보의 수치·실행 시간·적합 횟수를 `outputs/logs/p4_summary.json`과 `outputs/tables/p4_*.csv`에 저장한다.

## 6. 분기 경로 집계

- 점예측 후보 6 + 참고 1, 위험 후보 1, 거리 4, 폴드 3.
- Optuna 시행 40회 × 12(거리·폴드) = 480회, 최종 적합 5 seed × 2 후보 × 12 = 120회.
- 결합 가중 격자: P4-3 21점, P4-4 66점.
