# Phase F — 예측 성능 개선 탐색 실험 지시서 (에이전트용 프롬프트) · 2026-10-03 개정판

> 이 문서를 그대로 에이전트(Claude Code 등)에게 전달한다. 에이전트는 이 문서 전체를 먼저 읽고, 0장의 확인 사항을 사람에게 보고한 뒤 실행한다.

---

## 0. 역할과 목표

너는 이 저장소(`factory-peakguard-prototype`)의 연구 엔지니어다. Phase A~E는 끝났고 결과는 잠겨 있다. 이번 **Phase F**의 목표는 하나다.

> **같은 development 데이터와 같은 누수 규칙 안에서, 1~4시간 앞 15분 공장 전력 예측의 절대 오차를 최대한 낮춘다. 피크 구간 오차는 악화시키지 않는다.**

**계산 예산·시간·일정 제약은 없다.** 6장의 모든 실험을 끝까지 수행하고, 각 계열 안에서 유망한 변형이 보이면 스스로 추가 실험을 설계해 확장한다. 단, 3장의 불변 경계는 어떤 실험에서도 깨면 안 된다. 많이 실험하는 만큼 **선택 편향**이 커지므로 5장의 EXPLORE/CONFIRM 분리를 반드시 지킨다.

### 0.1 실행 전에 사람에게 한 번 보고할 것

다음 네 가지를 확인해서 한 번에 보고하고, 사람의 답을 받은 뒤 시작한다.

1. 작업 브랜치: `Phase3_experiments`에서 새 브랜치 `Phase4_performance`를 만든다(확인만 받고 진행).
2. 최종 holdout 동결·평가의 현재 상태: 이미 했는지, 승인 대기인지. `DECISIONS.md`·`PROGRESS.md`·`outputs/logs/run_status.json`으로 확인해 보고한다.
3. 실행 환경: GPU 사용 가능 여부(Phase C 기록상 CUDA 12.8 환경 존재), 원자료 `data/raw/task05_power/` 존재 여부, `.venv-phase-c` 사용 가능 여부.
4. 4장의 "이번 Phase에서 바뀌는 규칙"을 사용자가 승인했는지 확인한다.

---

## 1. 먼저 읽을 파일 (이 순서)

1. `AGENTS.md`, `CLAUDE.md`, `eval_protocol.md`, `DECISIONS.md` 최신 10개 항목, `PROGRESS.md` 최신 3개 항목
2. `phase_c/README.md`, `configs/phase_c.json`, `outputs/phase_c/logs/preregistration_phase_c.md`
3. `phase_c/data.py` (봉인 로더, fold·purge, Phase B feature, D2 mask), `phase_c/statistical.py` (B5 Kalman, MSTL), `phase_c/training.py`, `phase_c/tcn.py`, `phase_c/chronos_reference.py`, `phase_c/evaluation.py`
4. `outputs/phase_c/tables/` 전체 (특히 `model_horizon_fold_metrics.csv`, `model_horizon_pooled_metrics.csv`, `fold_manifest.csv`)
5. `outputs/phase_e/` 코드·표 (Phase E를 재실행할 때 재사용)
6. 구 파이프라인 참고: `src/features.py`, `research_p4/` (lgbm_deep, blends, stat_models, chronos_ref), `outputs/p4/p4_summary.json`, `outputs/p6/P6_summary.md`

구 파이프라인(`src/`, P1~P6)과 새 파이프라인(`phase_c/`, Phase A~E)은 지표 정의·horizon 표기(구: 1/4/16/96 step, 신: 4~16 quarter)·cohort가 다르다. **Phase F는 `phase_c/` 하네스를 기반으로 한다.** 구 파이프라인은 feature·모델 아이디어의 출처로만 쓰고, 수치는 섞어서 비교하지 않는다.

---

## 2. 현재 기준 성능 (깨야 할 숫자)

Phase C 결과 (D1, development score, AUC = 13개 horizon 평균):

| 모델 | AUC-MAE | AUC-PeakMAE | 4h MAE / Peak |
|---|---|---|---|
| B1 Weekly | 19.196 | 19.496 | 19.24 / 19.50 |
| **B5 Kalman (현 M\*)** | **13.037** | **18.481** | **14.86 / 18.94** |
| M1 LightGBM | 12.757 | 20.226 | 14.91 / 22.88 |
| M1-W (peak w=2) | 12.873 | 18.561 | 15.28 / 22.72 |
| M2 TCN (context 96) | 10.051 | 31.129 | 13.85 / 40.30 |
| R1 Chronos-2 (context 2048, zero-shot) | 6.419 | 15.296 | 7.57 / 17.70 |

진단상 이미 알려진 사실 (실험 설계에 반영할 것):

- **정보 예산이 좁다.** Phase B에서 고정한 G1은 `current, lag4, slot1d, slot7d`, G2는 `r4_mean, r4_max, r4_std, r16_max, r96_max`뿐이다. TCN context는 96(1일)이다. 반면 Chronos-2는 2048(약 21일)이다. **이 격차가 성능 차이의 1순위 가설이다.**
- B5는 2~4h에서 피크 오차가 Weekly와 같은 수준이다. 피크 개선이 이번 Phase의 2순위 목표다.
- **Fold 1의 score 2,746행 중 novel profile은 315행뿐이다** (`fold_manifest.csv`). fold 1은 증강 중복이 지배해서 Weekly가 거의 완벽하다. fold 1에서 개선이 안 나오는 것을 모델 실패로 해석하지 말고 D2와 fold별 결과로 따로 본다.
- MSTL(AUC-MAE 113.7)은 구현 오류 가능성이 높다.
- P6 E2 기록상 7월 말 2주는 모든 전략이 붕괴했다. 원인을 조사할 가치가 있다.
- 피크 τ는 fold별 fit의 Q0.95로 약 176~178이다.

---

## 3. 불변 경계 (어떤 실험에서도 위반 금지)

1. **봉인 경계**: `src.session_data.SEALED_BOUNDARY`(2021-08-09 09:45) 이후의 데이터를 읽거나 파싱하지 않는다. `phase_c.data.load_history`만 사용한다. 구 파이프라인의 전체 데이터 로더, `scripts/phase_b_reference/`, `verification/run_all.py`는 실행하지 않는다.
2. **최종 holdout 평가·동결 승인·예약 활성화를 하지 않는다.** `final_test.csv`를 만들지 않는다. `verification/results/`의 과거 final artifact를 열지 않는다(Phase C incident 재발 금지). 열람 여부를 매 실행 로그에 `historical_final_artifact_read=false`로 기록한다.
3. **누수 규칙**: 시간순 분할, 원점 t까지 관측된 값만 입력(`관측 최대 시각 ≤ origin`), 달력은 사전 가용, 미래 실측 기상·동시간 생산·인원 금지. fold별 fit → stop → cal → score 경계와 horizon purge를 그대로 사용한다. 정규화·τ·보정기·임계·앙상블 가중치·stacker는 모두 해당 fold의 fit(또는 fit+stop/cal) 구간에서만 추정한다.
4. **Phase A~E의 기존 산출물·사전등록·코드 봉인 파일을 수정하거나 덮어쓰지 않는다.** 모든 새 코드는 `phase_f/`, 모든 산출물은 `outputs/phase_f/`에 둔다.
5. 실패·기각 결과를 삭제하지 않는다. 실험 레지스트리에 전부 남긴다.
6. 블라인드 규칙(소속·로고·개인 경로 금지)을 코드·주석·그림에서 지킨다.

---

## 4. 이번 Phase에서 바뀌는 규칙 (사용자 승인 사항)

아래는 기존 규칙과 다르므로 `DECISIONS.md`에 "2026-10-03 Phase F 사용자 승인"으로 기록한 뒤 진행한다. 사용자가 승인하지 않은 항목은 실행하지 않는다.

- (a) **딥러닝 학습 허용**: Phase F 탐색 범위에서 신경망 학습(TCN, N-HiTS, PatchTST, DLinear 등)을 허용한다.
- (b) **사전학습 모델의 정식 후보화와 fine-tuning 허용**: Chronos-2 등 foundation model을 reference가 아닌 정식 후보로 비교한다. fit 구간만 사용하는 fine-tuning을 허용한다.
- (c) **입력 정보 확장 허용**: Phase B의 G0+G1+G2 잠금을 Phase F 탐색에 한해 풀고, 과거 전력에서 파생되는 정보(더 긴 lag·profile·context)를 추가한다. 미래 정보·동시간 생산·실측 기상 금지는 그대로다.
- (d) **Phase F는 탐색(exploratory) 단계다.** Phase C/D의 사전등록 선정을 소급해서 바꾸지 않는다. Phase F 결과는 "사후 탐색 + CONFIRM 확인"으로 보고한다.

---

## 5. 실험 하네스와 평가 설계 (모든 실험이 공유)

### 5.0 진행 중 수정 반영 (2026-10-03 개정판) — 지금 실행 중이라면 먼저 수행

이 문서는 실험 도중 개정되었다. 개정 내용은 5.2-B(walk-forward fold), 5.3-B(seed), 5.6(Phase E 자동 평가), 7장 중단 기준, 8장 선정 규칙 보완이다. 이미 실행한 실험은 무효가 아니다. 아래 순서로 반영한다.

1. **CONFIRM 열람 여부 점검**: 지금까지 CONFIRM 행이 집계·출력·조회된 적이 있는지 로그로 확인해 보고한다. 열람한 적이 있으면 숨기지 말고 `DECISIONS.md`에 기록하고 사람에게 알린다.
2. **R1 parity 점검**: `F0-1-R1`을 Phase C와 같은 전체 score cohort로 집계했을 때 AUC-MAE 6.4189, AUC-PeakMAE 15.2963이 재현되는지 확인한다. EXPLORE 전용 표의 7.80과의 차이가 cohort 차이인지 재현 실패인지 밝힌다.
3. **중복 설정 정리**: `F6-1-*-median`과 `F6-1-*-native_mean`이 모든 지표에서 동일하다. Chronos-2가 mean을 실제로 산출하는지, median으로 대체되는지 구현을 확인한다. 동일 출력이면 `duplicate_of`를 기록하고 하나로 합친다.
4. 표에 B5·M1·R1을 **같은 cohort(같은 행)**로 넣고, B5 대비 paired CI 컬럼을 추가한다.
5. 달력 공변량 실험(F6-2)의 공변량이 원자료에서 파생된 값 없이 순수 달력(시간·요일·휴일 목록)만 쓰는지 확인하고 미래 교란 검사 로그를 첨부한다.
6. 5.2-B fold를 잠근 뒤, **이미 실행한 모든 설정을 새 fold에서 다시 계산**한다(실험이 빠르므로 전부 재실행). 이후 선정은 새 기준을 따른다.

### 5.1 코드 구조

```
phase_f/
  harness.py        # phase_c.data.build_contexts 재사용, 실험 1개 = (모델, 설정) → 13 horizon × 3 fold 예측
  features_ext.py   # 확장 feature (원점 가용 시각을 컬럼별 provenance로 기록)
  models/           # 계열별 구현
  ensemble.py
  metrics.py        # 5.3 지표 전부
  registry.py       # 실험 레지스트리 기록
  leakage_tests.py
  run.py            # --exp <ID> --tier <n>
outputs/phase_f/
  registry.csv      # 실험 1행 = 1 설정
  predictions/      # parquet (Git 제외)
  tables/ figures/ logs/
```

예측 산출 형식은 Phase C와 같게 맞춘다(동일 origin·target·fold·horizon 키). 그래야 기존 B1/B5/M1/R1 예측과 paired 비교가 된다.

### 5.2 EXPLORE / CONFIRM 분리 (선택 편향 방지, 가장 중요)

수백 개 설정을 같은 score 구간에서 비교하면 최고 점수는 운으로 부풀려진다. 그래서 실험 시작 **전에** 다음을 고정하고 `outputs/phase_f/logs/split_lock.json`에 해시와 함께 저장한다.

- 각 fold의 **score 날짜를 ISO 주차 기준으로 두 그룹으로 나눈다**: 짝수 주차 = `EXPLORE`, 홀수 주차 = `CONFIRM`. 날짜 단위로 나누고, 같은 날의 행은 같은 그룹이다.
- 모든 튜닝·feature 선택·앙상블 가중치 탐색·모델 선택은 **EXPLORE만** 본다.
- `CONFIRM`은 7장 Stage 4에서 **최종 후보(최대 5개)에 대해 한 번만** 평가한다. CONFIRM 결과를 보고 후보를 바꾸거나 다시 튜닝하지 않는다.
- 학습 데이터는 그대로다(score 날짜는 원래도 학습에 쓰이지 않는다). 분리는 평가 행에만 적용된다.
- 레지스트리에 각 설정의 "EXPLORE 조회 횟수"를 남기고, 최종 보고에 **총 시도 설정 수**를 공개한다.
- CONFIRM은 실험 속도와 무관하게 다시 보지 않는다. "빠르니까 한 번만 더 확인"도 금지다.

### 5.2-B Walk-forward 주간 fold (비교 정밀도 향상)

Phase C의 3 fold는 유지하되(비교 연속성), fold 1은 score 2,746행 중 novel profile이 315행뿐이라 사실상 중복 데이터다. 3개 구간만으로는 0.1 수준의 차이를 구분할 수 없다. 그래서 **주 단위 walk-forward fold를 주 평가 체계로 추가**한다.

- 첫 score 주: 2021-04-12가 속한 ISO 주(또는 fit 이력이 최소 8주 확보되는 첫 주). 마지막 score 주: 목표 시각이 봉인 경계 이전인 마지막 완전한 주.
- 각 주 w에 대해: fit = w 시작 전까지(horizon purge 적용), stop/cal = fit 끝부분에서 Phase C와 같은 비율로 분리, score = 주 w. 모든 통계량(τ, 정규화, 보정기, 앙상블 가중치)은 주마다 fit/cal에서 다시 추정한다.
- 주 w는 ISO 주차의 짝·홀로 EXPLORE/CONFIRM에 배정한다(5.2와 같은 규칙). 즉 walk-forward fold 단위로 EXPLORE fold와 CONFIRM fold가 번갈아 나온다.
- 이 fold 정의와 주 목록을 `outputs/phase_f/logs/walkforward_lock.json`에 해시와 함께 잠근 뒤 사용한다.
- 비용이 큰 모델(fine-tuning 등)은 주마다 재학습 대신 4주마다 재학습하고 그 사이 주에는 같은 모델로 예측할 수 있다. 어느 방식인지 레지스트리에 기록한다.
- **주 평가 지표는 walk-forward EXPLORE fold의 pooled 값**으로 한다. Phase C 3-fold 값은 보조로 함께 보고한다.
- bootstrap은 날짜 블록 단위로, 모든 EXPLORE fold의 score를 합쳐 수행한다. 주별 개선 부호(몇 주에서 이겼는지)도 함께 보고한다.

### 5.3 지표 (모든 실험에서 전부 계산)

점예측 (D1, D2 각각, horizon별·fold별·pooled):

- MAE, RMSE, **nMAE = MAE / fit 구간 평균 전력**, **CV(RMSE) = RMSE / fit 평균**, NMBE(bias), MASE(fit 구간 weekly naive MAE로 정규화), MAPE(0 제외, 보조)
- **Peak-MAE (실제값 > τ 인 행)**: 기존 정의, 비교 연속성 유지용
- **Predicted-Peak MAE (예측값 > τ 인 행)**: 과대예측 편향을 잡는 대칭 지표
- **Daily-max error**: 일별 실제 최대값 vs 같은 날 해당 시각 예측의 최대값 차이, 피크 시각 오차(분)
- AUC-MAE, AUC-PeakMAE (13 horizon 평균), **4h(h16) 값을 따로 표기**
- 학습·추론 시간, 메모리

확률 출력이 있는 모델은 pinball loss(0.1/0.5/0.9/0.95), coverage, Brier(τ 초과)를 추가한다.

통계: 기준 모델 대비 paired 차이, 날짜 블록 bootstrap 1,000회, 95% CI. 기준 모델은 **B5와 M1과 R1(Chronos-2 ctx2048)** 세 개다.

### 5.3-B 반복 seed (무작위성이 있는 모든 모델)

- 딥러닝·fine-tuning·bagging이 있는 GBDT는 **seed 5개(최종 후보는 10개)**로 반복한다.
- 레지스트리에는 seed 평균과 표준편차를 모두 기록한다. 순위는 **seed 평균 예측의 점수**로 정하고, 단일 seed 최고값으로 순위를 매기지 않는다.
- 두 설정의 차이가 seed 표준편차보다 작으면 개선으로 세지 않는다.
- Chronos-2 zero-shot처럼 결정적인 모델은 1회로 충분하되, 결정성을 1회 재실행으로 확인한다.

### 5.4 누수 자동 검사 (모든 새 feature·모델 계열에 필수)

- **미래 교란 불변성 검사**: origin 이후의 전력값을 무작위로 바꿔도 해당 origin의 예측이 바뀌지 않아야 한다(최대 절대 차 0). 계열마다 최소 1회 실행하고 로그를 남긴다.
- feature provenance: 모든 확장 feature의 `latest_observed ≤ origin`을 행 단위로 assert한다.
- 앙상블·stacker는 fold 내부 OOF(cal 구간) 예측으로만 학습한다. score 예측으로 가중치를 정하면 안 된다.

### 5.6 Phase E 자동 평가 (모든 상위 후보에 기본 적용)

점예측이 좋아져도 경보가 그대로일 수 있다. 그래서 EXPLORE AUC-MAE 상위 20위 안에 들어온 모든 설정과 기준 모델(B5, M1, R1)에 대해 **Phase E 위험·경보 층을 자동 실행**한다. `outputs/phase_e/code/`를 재사용하고 Phase E의 고정 설정(Q0.95 τ, CAL-only Platt, standardized one-sided conformal, C/L grid {.01,.05,.10,.20,.30,.50}, 1/1·2/2 episode alert)을 그대로 쓴다.

- 예측분포: quantile 출력이 있는 모델은 quantile 보간으로 P(Y > τ)를 계산한다. 점예측만 있는 모델은 cal 구간 잔차의 hour×day-type별 경험분포(또는 표준화 잔차 Gaussian)로 만든다. 어느 방식인지 기록한다.
- 기록 지표(EXPLORE 기준): Brier, BSS, PR-AUC, calibration intercept/slope, 전체·peak-only conformal coverage, c=.10에서 1/1·2/2 episode precision/recall/F1/FP episode 수, median lead time, miss decomposition(decision/forecast miss).
- 비교 기준은 Phase E B5 결과(c=.10, 2/2: recall 0.234, precision 0.611, FP 21 / peak-only coverage 0.889)다. 단, cohort가 다르므로 같은 EXPLORE 행에서 B5를 다시 계산한 값과 비교한다.
- Phase E 층은 이미 고정된 설정이라 실행 시간이 짧다(원래 약 17초). 후보가 갱신될 때마다 다시 돌린다.
- 경보 층의 C/L 임계나 정책을 Phase F에서 튜닝하지 않는다. 점예측 개선의 하류 효과만 측정한다.

### 5.5 레지스트리 형식 (`outputs/phase_f/registry.csv`)

`exp_id, family, tier, parent_exp, duplicate_of, config_json, config_hash, code_commit, n_seeds, wf_explore_AUC_MAE, wf_explore_AUC_MAE_seed_sd, wf_explore_AUC_PeakMAE, wf_explore_h16_MAE, wf_explore_h16_Peak, wf_explore_PredPeakMAE, wf_weeks_won_vs_B5, D2_wf_explore_AUC_MAE, pc3_explore_AUC_MAE, pc3_fold0/1/2_MAE, ciLow_vs_B5, E_brier, E_prauc, E_peak_coverage, E_c10_22_recall, E_c10_22_precision, E_c10_22_FP, train_sec, infer_sec, leakage_test, status(planned/run/failed/rejected/finalist), note`

(`wf_` = walk-forward fold, `pc3_` = Phase C 3-fold, `E_` = 5.6 Phase E 자동 평가)

---

## 6. 실험 목록 (전부 지시함, 우선순위는 7장)

각 실험 ID는 그대로 레지스트리에 쓴다. 각 계열은 **가장 강한 기존 설정을 parent로 두고 한 번에 한 요소만 바꾸는 ablation**을 기본으로 한다.

### F0. 재현과 기준선 정비

- F0-1: B1, B5, M1, M1-W, M2, R1 예측을 기존 parquet에서 불러와 새 지표로 재계산한다. 재학습 결과와 최대 차 0을 확인한다(가능한 경우).
- F0-2: **MSTL 디버깅**. AUC-MAE 113.7의 원인을 찾는다(외삽 방식, AutoETS 추세 폭주, 결측 처리, 인덱스 정렬). 고친 MSTL을 `F0-2-fixed`로 별도 등록하고, 원래 B3은 그대로 둔다.
- F0-3: 추가 기준선: seasonal naive 변형(최근 k주 같은 slot의 평균·중앙값, k=2,3,4), day-type별 profile(근무일/토/일·휴일), CBL 최선안.
- F0-4: **"오라클 상한"(진단 전용)**: 실제 미래값 일부를 넣었을 때 얼마나 좋아지는지 측정해 이론적 여유를 가늠한다. 이 결과는 절대 후보가 될 수 없고 `oracle`로 표시한다.

### F1. 입력 정보·context 확장 (1순위 가설)

모든 feature는 `features_ext.py`에 provenance와 함께 구현한다. 기본 모델은 M1(LightGBM)과 Ridge 두 개로 평가하고, 효과가 확인된 묶음만 다른 계열로 전파한다.

- F1-1 단기 lag: lag 1~16 전부, 1~8, 1~4
- F1-2 같은 slot 과거값: 목표 시각과 같은 slot의 1~7일 전, 7·14·21·28일 전
- F1-3 같은 slot 요약: 최근 2/3/4주 같은 요일·같은 slot의 평균·중앙값·최대·표준편차, 최근 5 근무일 같은 slot 평균
- F1-4 profile 기반 편차: "현재값 − 지난주 같은 시각", "오늘 지금까지 누적 / 지난주 같은 시간까지 누적" 비율, 오늘 profile과 최근 k주 profile의 상관
- F1-5 다중 스케일 rolling: 1h·2h·4h·8h·24h·168h의 mean/max/min/std/quantile(0.9), EWMA(span 4·16·96)
- F1-6 추세·변화: 최근 1h·4h 기울기, 1차·2차 차분, 마지막 큰 변화 이후 경과시간
- F1-7 피크 이력: 지난 24h·7일 τ 초과 횟수, 마지막 τ 초과 이후 경과시간, 오늘 최대값 대비 현재 수준
- F1-8 달력 정교화: 휴일 전후일·징검다리, 근무일 순번(주 내), slot × day-type 상호작용, 월 진행도
- F1-9 **구 파이프라인 feature 이식**: `src/features.py`의 `lag_0..7, lag_96, lag_672, target_slot_{k}d_ago, recent_4_slope, cbl_mid_6_10(_adjusted)`
- F1-10 생산량 1회 확인(Phase B에서 약속한 비선형 확인): 완료된 과거 생산량(직전 완료 시간, 같은 slot 7일 평균)을 최선 LightGBM에 추가하는 1회 실험만 한다. 튜닝하지 않는다.
- F1-11 feature 선택: permutation/SHAP 기반으로 상위 k개만 남기는 축약판. 성능 손실과 단순성의 교환을 기록한다.

### F2. 목표 변환

같은 feature와 모델로 목표만 바꾼다.

- F2-1 직접 y (기준)
- F2-2 Δ 목표: y(t+h) − y(t)
- F2-3 weekly residual: y(t+h) − y(t+h−7d)
- F2-4 robust profile residual: y(t+h) − (최근 k주 같은 slot 중앙값)
- F2-5 B5 residual (**하이브리드**): y(t+h) − B5 예측. 최종 = B5 + 보정
- F2-6 log1p(y) 학습 후 역변환(bias 보정 포함)
- F2-7 day-type별 분리 모델 vs 단일 모델

### F3. GBDT 계열

- F3-1 LightGBM Optuna 튜닝: horizon anchor(4·8·12·16) 및 13개 horizon 전체 × fold, stop MAE 기준, trial 수 제한 없음(개선이 수렴할 때까지, 최소 500회), 탐색공간(num_leaves 15~255, min_child_samples 10~200, learning_rate 0.01~0.1, feature_fraction 0.5~1, bagging, lambda_l1/l2, max_bin)
- F3-2 손실함수: L2, L1, Huber(δ 3종), fair, quantile(0.5/0.55/0.6), tweedie
- F3-3 피크 가중: w = 1, 1.5, 2, 3, 4와 연속 가중 w = 1 + α·max(0, y−q80)/(τ−q80) (α = 0.5, 1, 2)
- F3-4 XGBoost, CatBoost (같은 feature, F3-1과 같은 수준의 무제한 튜닝): 앙상블 다양성 확보 목적
- F3-5 **Global-H**: horizon을 feature로 넣은 단일 모델(13 horizon 공동 학습)
- F3-6 **MIMO / multi-output**: 13 horizon 동시 예측(sklearn MultiOutput 또는 chain)
- F3-7 seed 앙상블(5·10·20 seed 평균)
- F3-8 feature 묶음 × 손실 × 가중의 전체 교차 조합 탐색

### F4. 통계·상태공간 계열

- F4-1 B5 확장: (i) weekly 대신 최근 k주 profile 기준 편차, (ii) hour-of-day별 잡음 분산, (iii) day-type별 φ, (iv) level + 편차 2-state, (v) 관측잡음 robust(Student-t 근사)
- F4-2 DSHW/TBATS/ETS 다중계절, 고친 MSTL(F0-2), AutoARIMA with Fourier terms
- F4-3 **Kalman + GBDT 하이브리드**: B5 예측과 B5 상태(편차·분산)를 LightGBM feature로 투입
- F4-4 Theta/AutoTheta (statsforecast)

### F5. 딥러닝 계열 (4(a) 승인 시)

공통: fit 구간만 학습, stop으로 early stopping, 정규화는 fit 통계, seed 5개 평균. context 길이는 96 / 192 / 336 / 672 / 1344 / 2016 / 2688을 비교한다. 각 모델은 Optuna 등으로 구조·학습률·dropout·batch·context를 무제한 탐색한다.

- F5-1 TCN context 확장(현재 96 → 위 grid), 피크 가중 손실, quantile 출력 head
- F5-2 **DLinear / NLinear** (가볍고 강한 선형 기준, 반드시 수행)
- F5-3 N-HiTS, N-BEATS (neuralforecast)
- F5-4 PatchTST, TiDE (외생 변수로 달력 포함)
- F5-5 LSTM/GRU (단일 설정, 참고용)
- F5-6 손실: MAE, Huber, peak-weighted MAE, quantile multi-head
- F5-7 TimesNet, iTransformer, TSMixer, xLSTM 등 설치 가능한 최신 구조 추가 비교
- F5-8 확장 feature(F1 최선 묶음)를 외생 변수로 넣은 DL 변형

### F6. Foundation model 계열 (4(b) 승인 시)

- F6-1 Chronos-2 zero-shot context sweep: 512 / 1024 / 2048 / 4096 / 8192(가능 시), median vs mean 출력
- F6-2 **Chronos-2 + 공변량**: 미래 달력(시간·요일·휴일)을 known-future covariate로 제공
- F6-3 Chronos-2 quantile 출력 활용: q0.5 대신 q0.55~0.65를 점예측으로 쓰는 변형(피크 과소추정 보정)
- F6-4 **Chronos-2 fine-tuning**: fold별 fit 구간만 사용, stop으로 early stopping. full / LoRA(가능 시), learning rate·step 수·context 길이를 무제한 탐색
- F6-6 fine-tuning한 foundation model + 공변량 + quantile 점예측 조합
- F6-5 다른 foundation model: TimesFM, Moirai, TiRex, Chronos-Bolt 등 **실제 설치 가능한 것만** zero-shot 비교. 설치 실패는 레지스트리에 `failed`로 남긴다.
- 주의: 모든 foundation model 입력은 origin까지의 시계열로만 자른다. 5.4 미래 교란 검사를 반드시 통과해야 한다.

### F7. 앙상블·결합

base 후보는 각 계열의 EXPLORE 상위 3개다. 조합은 전수 탐색해도 된다(선택 편향은 CONFIRM으로 통제).

- F7-1 단순 평균, 중앙값
- F7-2 inverse-MAE 가중(cal OOF 기준)
- F7-3 **stacking**: 비음수 Ridge / constrained least squares, cal 구간 OOF로 학습, horizon별 가중치
- F7-4 **regime-gated 결합**: hour 구간(00–08/08–16/16–24) × day-type별 가중치
- F7-5 **피크 인지 결합**: 예측 위험 확률(또는 Chronos q0.9 > τ)이 높은 구간에서는 피크 성능이 좋은 모델의 가중치를 키움
- F7-6 핵심 조합 명시 실험: {B5, best LGBM}, {B5, Chronos-2}, {Chronos-2, best LGBM}, {B5, Chronos-2, best LGBM, best DL}

### F8. 피크 특화

- F8-1 2단계: 피크 확률 분류기(LightGBM) + 조건부 회귀(피크/비피크 별도)로 결합
- F8-2 bias 보정: hour × day-type별 cal 구간 잔차 중앙값을 더하는 후처리
- F8-3 quantile-to-point: 위험이 높을 때 상위 분위수 쪽으로 이동하는 규칙
- F8-4 피크 구간 과소추정 원인 분석: 피크 행에서 각 모델의 부호 있는 오차 분포를 비교하고 그림으로 남긴다

### F9. 학습 데이터 전략

- F9-1 sliding window(최근 4/6/8주) vs expanding
- F9-2 최근 가중(지수 가중, half-life 1·2·4주)
- F9-3 **중복 profile 처리**: fit에서 중복 profile 일을 제거(D2식) vs 포함 vs 가중 축소
- F9-4 주간 재학습 시뮬레이션(walk-forward, score 구간 안에서 7일마다 재적합 — 재적합에는 해당 시점 이전 관측만 사용)

### F10. 오류 진단 (모든 Stage 후 갱신)

- 최선 모델의 오차를 hour, 요일, 휴일 전후, 생산량 구간(사후 분석용), 피크 여부, fold, D1/D2별로 분해한다.
- **7월 말 2주 붕괴 구간**을 별도로 분석한다: 레벨 shift인지, 패턴 변화인지, 결측·증강 영향인지.
- fold 1(중복 지배) 결과를 따로 해석한다.

### F11. 하류 영향 (최종 후보 확정 후)

- (5.6에 따라 상위 후보는 탐색 중에 이미 자동 평가된다.) 최종 대표 후보 1~2개는 **CONFIRM 포함 전체 score로 Phase E 파이프라인을 재실행**한다(probability·Platt·conformal·C/L·episode alert, 같은 고정 grid). 점예측 개선이 경보 recall·peak-only coverage를 얼마나 바꾸는지 B5 기준 Phase E 결과와 나란히 보고한다.

---

## 7. 실행 순서 (예산·일정 제약 없음)

시간과 계산 예산 제한은 없다. 아래는 **우선순위 순서**일 뿐이며 모든 Stage를 끝까지 수행한다. 어떤 실험도 시간 때문에 생략하지 않는다.

| Stage | 내용 |
|---|---|
| Stage 0 | 0.1 보고, 하네스·지표·누수 검사·split_lock, F0 |
| Stage 1 (Tier 1) | **F1 전체(LightGBM·Ridge)**, F6-1·F6-2·F6-3, F5-2 DLinear, F2-3·F2-5, F0-2 |
| Stage 2 (Tier 2) | F3 전체, F4-1·F4-3, F5-1·F5-3, F6-4, F7 전체, F9-3 |
| Stage 3 (Tier 3) | 나머지 F2·F4·F5·F6·F8·F9 전부, 그리고 Stage 1~3에서 유망하게 나온 방향의 자율 확장 실험 |
| Stage 3+ (반복) | 상위 설정을 parent로 다시 ablation·조합·튜닝을 반복한다. 아래 중단 기준을 만족하면 Stage 4로 넘어간다 |
| Stage 4 | 최종 후보 선정(8장) → CONFIRM 1회 → F10·F11 → 보고 |

규칙:

- 각 Stage가 끝날 때마다 `outputs/phase_f/STAGE_<n>_summary.md`에 상위 10개 표(EXPLORE 기준)와 다음 Stage 계획을 쓰고 커밋한다.
- Stage 1 결과로 1순위 가설(정보 예산)이 맞는지 판정한다. 맞으면 확장 feature를 이후 모든 계열의 기본 입력으로 삼는다.
- 자율 확장 실험도 실행 전에 레지스트리에 `planned`로 먼저 등록하고, 어떤 결과를 보고 세운 가설인지 `note`에 적는다.
- 장시간 작업은 체크포인트를 남겨 중단 후 재개할 수 있게 한다.
- **중단 기준(계열별·전체 공통)**: 최근 2회 반복에서 (i) walk-forward EXPLORE AUC-MAE의 최선 대비 개선이 2% 미만이거나, (ii) 개선의 paired 95% CI가 0을 포함하거나, (iii) 개선이 seed 표준편차보다 작으면 그 방향을 중단한다. 단, AUC-PeakMAE·h16 Peak-MAE·Phase E 2/2 recall 중 하나라도 같은 기준으로 유의하게 좋아지는 방향은 계속 탐색한다(피크·경보 개선이 이번 Phase의 2순위 목표다).
- GPU 작업과 LightGBM 작업은 동시에 돌리지 않는다(OpenMP 경합 기록 있음).

---

## 8. 최종 후보 선정 규칙 (Stage 4 시작 전에 이대로 고정)

1. **자격** (모두 walk-forward EXPLORE, seed 평균 기준): (i) 누수 검사 통과, (ii) B5 대비 AUC-MAE 개선의 95% CI 하한 > 0, (iii) AUC-PeakMAE가 B5 대비 CI 상한 기준으로 명확히 악화되지 않음, (iv) D2에서 B5 대비 AUC-MAE 역전 없음, (v) EXPLORE 주의 2/3 이상에서 B5 대비 MAE 개선, (vi) 5.6 Phase E 자동 평가에서 c=.10, 2/2 episode recall이 같은 행의 B5보다 낮지 않음.
2. 자격을 갖춘 설정 중 **EXPLORE AUC-MAE 상위 5개**를 finalist로 고정한다(서로 다른 계열 우선, 같은 계열은 최대 2개). 이와 별도로 **AUC-PeakMAE 최선 1개와 Phase E episode F1 최선 1개**가 상위 5개에 없으면 finalist에 추가한다(최대 7개).
3. CONFIRM(walk-forward CONFIRM 주 + Phase C 3-fold CONFIRM 날짜)을 finalist + 기준 3개(B5, M1, R1)에 대해 **한 번만** 평가한다. Phase E 자동 평가도 CONFIRM에서 같은 1회에 포함한다.
4. 대표 후보 = CONFIRM에서 자격 (ii)~(iv)·(vi)를 유지한 finalist 중 CONFIRM AUC-MAE 최소. 최소 모델과 CI가 겹치는 모델이 있으면 그중 피크·경보 지표가 더 좋은 모델과 더 단순한 모델을 함께 제시하되, 선택은 사람에게 넘긴다.
5. CONFIRM에서 아무도 자격을 유지하지 못하면 "탐색 개선이 확인되지 않음"으로 보고한다. 기준을 낮추지 않는다.

이 단계는 **후보 제안**이다. 제출 모델 교체, 동결, holdout 평가는 사람이 결정한다.

---

## 9. 최종 산출물

1. `outputs/phase_f/PHASE_F_REPORT.md`
   - 총 시도 설정 수, 계열별 시도 수, 실패 수
   - EXPLORE 리더보드 상위 20, finalist의 CONFIRM 결과, B5/M1/R1 대비 paired CI
   - **nMAE·CV(RMSE)로 환산한 절대 성능**과 ASHRAE Guideline 14(시간 단위 CV(RMSE) ≤ 30%) 대비 위치
   - 1순위 가설(정보 예산) 판정: feature·context 확장의 기여를 ablation으로 정량화
   - 피크 지표 2종(실제 피크, 예측 피크)과 daily-max error 비교
   - walk-forward 주별 개선 그림(주마다 B5 대비 MAE 차이), seed 표준편차 표
   - 상위 후보별 Phase E 자동 평가표(점예측 개선이 경보 recall·peak coverage로 이어졌는지)
   - 5.0 점검 결과(CONFIRM 열람 여부, R1 parity, 중복 설정 정리)
   - F10 오류 진단, F11 Phase E 재실행 결과
   - 한계: 탐색 단계라는 점, 선택 편향, fold 1 중복, 단위 미확인, holdout 미사용
2. `outputs/phase_f/registry.csv` (전체 실험)
3. `outputs/phase_f/tables/` (leaderboard, ablation, pairwise_ci, fold, D2, peak 분해)
4. `outputs/phase_f/figures/` (horizon별 MAE·Peak 곡선, context 길이 효과, feature 묶음 ablation, 피크 구간 부호 오차 분포, 대표 주간 예측 겹쳐 그리기)
5. `DECISIONS.md`(4장 승인, split_lock, 선정 규칙 고정 시각)와 `PROGRESS.md`(Stage별) 갱신
6. 재현 명령: `python -m phase_f.run --stage all`과 Stage별 명령

---

## 10. 보고 시 금지 표현

- CONFIRM 이전 EXPLORE 수치를 "검증 성능"이라고 부르지 않는다.
- "최종 성능", "테스트 성능", "실제 피크 저감"이라고 쓰지 않는다(holdout 미사용, 현장 미검증).
- 수백 개 중 최고값을 보고할 때 총 시도 수를 함께 쓰지 않으면 안 된다.
- Chronos 계열이 이겨도 "정보량이 같은 조건"인지 명시한다(context 길이를 표에 함께 표기).
- 단위를 kW/kWh로 단정하지 않는다.