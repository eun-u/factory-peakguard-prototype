# Phase F 실행 전 검토안

작성일: 2026-10-03. 상태: **사용자 승인 대기**. 이 파일은 실행 전 검토 문서이며 사전등록, 실험 결과, 사용자 승인 기록이 아니다. 새 브랜치 생성, 모델 학습, 패키지 설치, 최종 평가를 아직 시작하지 않았다.

## 1. 요청과 확인된 실행 상태

사용자 제공 Phase F 지시서 §0.1은 네 가지를 보고하고 사람의 답을 받은 뒤 시작하도록 명시한다. 아래 수정안을 함께 승인받은 후 §4의 규칙 변경을 DECISIONS.md에 기록한다.

| 항목 | 현재 확인 내용 |
| --- | --- |
| 저장소 | eun-u/factory-peakguard-prototype |
| 기준 브랜치 | Phase3_experiments |
| 기준 커밋 | e76523c84587727ec2137f8d34c75dc371b7be62 |
| 예정 브랜치 | Phase4_performance; 로컬에 아직 없음 |
| 원자료 | data/raw/task05_power/okm_augumented_2021.csv 존재; 이번 사전 점검에서 본문 미열람 |
| 실행 환경 | .venv-phase-c, Python 3.13.14, torch 2.11.0+cu128, CUDA 사용 가능 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB; 점검 당시 2,650 MiB 사용 중. 독점 가용성은 보장되지 않음 |
| 주요 설치 패키지 | numpy 2.5.3, pandas 2.3.3, lightgbm 4.7.0, scikit-learn 1.9.1, chronos-forecasting 2.3.2 |
| 새 최종 평가 상태 | outputs/logs/run_status.json: completed_development, holdout=locked |
| 승인/동결/최종 파일 존재 점검 | human_freeze_approval.json, freeze_record.json, final_test.csv, final_test_predictions.csv 모두 없음; 본문을 열지 않음 |
| 과거 열람 사실 | eval_protocol.md에는 과거 마지막 15% 테스트 1회 열람 기록이 있음. Phase C에도 과거 final artifact 열람 incident가 기록됨. 따라서 '한 번도 보지 않은 holdout'이라고 주장하지 않음 |

DECISIONS.md 및 PROGRESS.md의 최근 관리 항목은 2026-09-29이며, Phase C/E의 별도 봉인 로그와 구 파이프라인의 관리 기록을 구분한다. 10월 1일 시간 조건은 지났지만 사람 승인 조건은 남아 있다. 이 점검은 현재 checkout에 대한 판단이며 다른 checkout이나 원격 실행의 완료 여부를 증명하지 않는다. 예약 작업의 현재 상태는 이번에 조회하거나 변경하지 않았다.

이번 사전 점검: historical_final_artifact_read=false, holdout_read=false, raw_data_body_read=false. 이는 이번 점검에만 적용하며 과거 incident를 취소하지 않는다.

## 2. 성능 개선 방향에 대한 판단

Phase C의 model_auc_summary.csv에 기록된 기존 D1 개발 점추정은 다음과 같다. 아직 Phase F의 동일 키 재검증을 수행한 결과가 아니다.

| 모델 | AUC-MAE | AUC-PeakMAE |
| --- | ---: | ---: |
| B5 Kalman | 13.036670 | 18.481253 |
| M1 LightGBM | 12.757335 | 20.225566 |
| M2 TCN | 10.050901 | 31.128641 |
| R1 Chronos-2 | 6.418943 | 15.296336 |

R1의 기존 MAE 점추정은 B5보다 약 50.8% 낮다. 따라서 R1을 정식 후보로 평가하고, 긴 과거 문맥의 효과를 같은 평가 키에서 분리해 확인하는 것이 우선이다. 문맥 길이만으로 격차가 설명된다는 인과 결론은 아직 없다. TCN의 평균 오차 감소와 큰 피크 오차 증가는 평균 MAE만 최적화하면 안 되는 근거다.

보고서는 (1) 기존 R1을 후보로 인정했을 때의 변화와 (2) 새로운 학습/특징/결합이 고정 R1보다 추가로 개선한 효과를 분리한다. R1보다 좋아지지 않으면 '새 모델 개선'으로 포장하지 않는다.

## 3. 승인에 포함할 설계 수정안

1. **CONFIRM의 의미와 분리 키.** 기존 Phase C/E가 이미 평가한 개발 score를 재사용하므로 '재사용 개발자료 내 잠금 확인'으로 표기한다. 새로운 독립 검증이나 최종 holdout이라고 부르지 않는다. 분리는 origin 날짜가 아닌 target 날짜의 ISO 주차로 고정하며, 13개 horizon에 같은 지도를 적용한다. 같은 target이 서로 다른 arm에 들어가지 않는지 검사한다. 기존 chronological fold/purge는 보존하며, 앞선 score 날짜가 이후 fold의 과거 fit/history에 포함될 수 있다는 한계도 공개한다. 실제 예측 시점에 관측된 과거 전력 사용은 허용하되 CONFIRM 오차·순위로 설정이나 갱신 규칙을 바꾸지 않는다.

2. **F0-4 미래 정답 oracle.** 실데이터의 미래 정답을 예측 입력으로 쓰는 실험은 수행하지 않는다. F0-4 ID는 유지하고 합성 데이터에서 의도적 미래 누수를 검출하는 음성 대조 검사로 대체한다. 모든 실제 후보의 관측 최대 시각은 origin 이하로 제한한다.

3. **F9-4 score 내 재학습.** 원래 고정 fold의 주요 후보 선정에서 제외한다. CONFIRM 종료 후, origin까지 도착한 과거 관측만 쓰는 사전 고정 온라인 갱신 진단으로 별도 수행한다. 기준 모델에도 같은 갱신 기회를 부여하고 원래 frozen-fit 성능과 섞지 않는다. 결과로 주요 후보를 다시 선택하지 않는다.

4. **학습·튜닝·보정 역할.** fit에서 모델, 정규화, 피크 τ를 추정한다. stop은 조기 종료에 사용한다. cal에서만 보정기·앙상블 수치 가중치·stacker를 적합하며, 하위 모델의 cal 예측은 해당 cal 정답을 학습하지 않은 모델이 생성한다. EXPLORE에서는 특징/설정/결합 방법을 선택하되 score 정답으로 가중치나 보정기를 적합하지 않는다. B5 잔차 모델의 fit 목표는 시간순으로 생성한 인과적 B5 예측 잔차를 사용한다.

5. **긴 문맥과 비교 행.** score 평가 키와 누락 처리 정책을 실험 전 고정하고 동일 키의 B5/M1/R1과 paired 비교한다. 결측 때문에 모델에 유리한 행만 버리지 않는다. 지원 가능한 모델은 과거값과 마스크만 사용하는 사전 고정 처리, 그렇지 않은 설정은 미지원으로 기록한다. 8192-slot 문맥은 약 85.3일로 fold 0의 약 72일 fit 이력보다 길다. 완전한 8192-slot fit 창을 요구하는 설정은 그 fold에서 불가능하며, 가변 문맥을 허용하는 경우 실제 길이를 별도 기록한다. 세 fold를 모두 충족하지 못한 설정은 3-fold 공식 후보가 될 수 없다.

6. **F1-10 과거 완료 생산량.** §4(c)의 전력 정보 확장에 대한 한 가지 명시적 예외로 승인 범위에 포함한다. 실제 완료·가용 시각이 origin 이하인 값만 사용하며, 현재 진행 중인 생산량·인원·미래 기상은 계속 금지한다. 가용 시각을 증명하지 못하면 실행 불가로 기록한다.

7. **선정 기준 수식.** 주 지표는 13개 horizon의 pooled MAE 평균이다. B5 대비 개선량 ΔM = MAE(B5) − MAE(후보)의 paired 95% CI 하한 > 0, 피크 악화량 ΔP = PeakMAE(후보) − PeakMAE(B5)의 CI 상한 ≤ 0을 요구한다. 피크 허용 악화폭은 0이다. D2의 ΔM 점추정 > 0 및 ΔP 점추정 ≤ 0, D1의 3개 fold 중 2개 이상 ΔM > 0을 요구한다. CI 계산 불가나 피크 표본 부족은 통과가 아니라 판정불가다. h16의 평균·피크 결과는 별도 공개하고 운영 후보 판단에 반영한다. EXPLORE에서 주요 후보 1개를 고정한 뒤 CONFIRM을 열고, 최대 4개 보조 후보는 보조 결과로만 보고한다. CONFIRM 이후 최저값을 새로 골라 대표 모델로 바꾸지 않는다. 주요 후보가 실패하면 성공 후보를 선정했다고 선언하지 않는다.

8. **반복 탐색과 종료.** F0–F11 계열을 모두 다루며 실패·미지원도 원인과 함께 기록한다. LightGBM의 최소 500회 탐색 요구를 보존한다. 유망 계열의 확장 라운드는 시작 전에 설정 목록과 종료 조건을 잠근다. 모든 지정 계열을 처리한 뒤, 현재 적격 최선 후보 대비 EXPLORE AUC-MAE가 0.5% 이상 개선되고 피크 보호 조건도 통과하는 후보가 두 확장 라운드 연속 없으면 탐색을 끝낸다. 각 라운드는 유한한 후보/Optuna trial 묶음으로 구성하고 다음 라운드와 구분한다. 총 계산 시간 상한은 두지 않는다. 성공하지 않은 설정이나 도구 설치 실패를 완료된 비교 실험으로 세지 않는다.

9. **통계 해석.** 요청한 날짜 블록 bootstrap 1000회·seed 42를 유지하되 여러 날짜에 반복된 증강 profile 및 짧은 평가 기간의 한계를 명시한다. D2와 주 단위 재표집/반복 profile 민감도 결과를 함께 보고한다. 수백 번 탐색한 EXPLORE의 CI는 선택 편향이 보정된 확증 통계가 아니다. CONFIRM도 과거 개발 사용 이력 때문에 완전히 독립적이지 않다.

10. **F11 운영 지표.** 주요 후보 고정 이후 기술적·운영적 진단으로 수행한다. B5의 native Kalman sigma를 다른 모델에 그대로 붙이지 않는다. 후보별 확률/분위수 또는 fit/cal로 구성한 잔차 기반 불확실성 변환을 평가 전에 고정한다. 보정·평가용 데이터를 분리하고 Phase E의 기존 산출물을 보존한다. F11 결과로 CONFIRM 후보나 경보 임계를 다시 선택하지 않는다.

## 4. 승인 후 실행 순서와 범위

- Phase3_experiments의 위 커밋에서 Phase4_performance 생성. §4(a) 신경망 학습, (b) foundation 모델의 정식 후보화 및 fit-only fine-tuning, (c) 입력 확장과 위 F1-10 예외, (d) 사후 탐색 및 기존 C/D 선정 보존을 실제 승인 내용으로 기록한다.
- Phase C/E 코드·설정·산출물 해시를 보존하고 새 phase_f/와 outputs/phase_f/에서 작업한다. 큰 예측·모델·가중치·원자료는 Git에 넣지 않는다.
- 하네스·split/cohort lock·누수 검사·기준선 재현(F0) 후, 긴 과거 전력 특징/target(F1/F2), R1 문맥 비교(F6), 통계 모델 보완(F4)부터 진행한다.
- GBDT(F3), 학습형 시계열 모델(F5), foundation 적합(F6), 결합(F7), 피크 보완(F8), 학습 창/중복 민감도(F9), 오류 분석(F10)을 진행하고 유망 교차 조합을 확장한다.
- 각 단계에서 수치·실패·연산량·선택 이력을 기록하고 로컬 커밋한다. 설치 변경은 기존 환경을 보존할 수 있는 별도 환경에서 수행한다. 기존 GPU 프로세스를 중단하지 않는다.
- 주요 후보 고정 후 CONFIRM 1회, F11 운영 진단 및 최종 보고. 최종 holdout 열람·동결 승인·예약 활성화·제품 배포는 이번 승인에 포함하지 않는다. GitHub push는 별도 게시 요청에 따른다.

## 5. 상용 수준 판정

이 데이터에서 오차를 낮추는 것과 상용 수준을 입증하는 것은 구분한다. 먼저 데이터 단위/집계 의미와 증강 중복을 확인하고, 절대 및 정규화 오차·피크 미탐/오탐·기간별 안정성·추론 지연·메모리를 함께 보고한다. 상용 판정에는 새로운 비증강 현장 데이터에서의 시간순 검증과 현장별 허용 오차/경보 부담/비용 기준이 추가로 필요하다. 해당 기준이나 데이터가 없으면 상용 적합성은 판정불가다.

ASHRAE Guideline 14의 시간별 CV(RMSE) 30% 기준은 에너지 절감 측정·검증의 보정 시뮬레이션 맥락이다. 15분 간격 공장 전력의 1–4시간 미래 예측이 상용 수준이라는 인증 기준으로 쓰지 않는다. 비교값을 표시하더라도 시간 해상도와 정규화 분모 정의의 차이를 명시한다.

근거:

- ASHRAE, Building Energy and Water Monitoring: https://handbook.ashrae.org/handbooks/A19/IP/A19_Ch42/a19_ch42_ip.aspx
- U.S. DOE FEMP, M&V Guidelines 5.0: https://www.energy.gov/sites/default/files/2024-10/mv_guide_5_0.pdf
- 로컬: eval_protocol.md, DECISIONS.md, PROGRESS.md, outputs/logs/run_status.json, phase_c/data.py, phase_c/evaluation.py, phase_c/README.md, configs/phase_c.json, outputs/phase_c/tables/fold_manifest.csv, outputs/phase_c/tables/model_auc_summary.csv.

이 검토에는 독립 코드·설계 검토를 반영했다. 아직 Phase F 성능 향상 수치를 산출하지 않았다.
