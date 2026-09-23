# 설계 요구와 산출물 대응

현재 수치는 개발 rolling-origin 3폴드 결과다. 최종 테스트는 2026-10-01 18:00 KST 이후 별도 일회 평가한다.
실행·검증 상태는 [PROGRESS](../PROGRESS.md), 원문 설계와 조정은 [PROJECT_DESIGN](../PROJECT_DESIGN.md), [DECISIONS](../DECISIONS.md)를 따른다.

| 설계 ID | 생성 파일 (`outputs/` 기준) | 상태 |
| --- | --- | --- |
| F1-1 | figures/F1-1_power_distribution.png | 월·요일·시간 분포 |
| F1-2 | figures/F1-2_quality_timeline.png | 복원·0값 위치 |
| F1-3 | figures/F1-3_production_baseload.png | 생산·전력 및 기저부하 |
| T1-1 | tables/T1-1_variable_dictionary.csv, report/ch1_data.md | 사전·가용 시점 |
| T1-2 | figures/T1-2_split_protocol.png, report/ch1_data.md | 시간 분할·누수 규칙 |
| T2-1 | tables/T2-1_fva.csv | 후보별 기준·채택·비교 CI |
| F2-1 | figures/horizon_curve.png, tables/horizon_advantage.csv | 네 거리의 MAE·에피소드 F1 및 쌍별 CI |
| F2-2 | figures/F2-2_hits_false_alarms.png | 점예측 위치 TP·FP |
| F2-3 | figures/F2-3_coverage.png, tables/coverage_comparison.csv, tables/coverage_by_fold.csv | A/B 보정, 표본 수·날짜 CI |
| F2-4 | figures/F2-4_episode_errors.png, tables/peak_episode_errors.csv | 매칭된 피크 시점·크기 오차 |
| T2-2 | tables/T2-2_pooled_metrics.csv, logs/development_selection.json | 후보 지표와 선정 비교 근거 |
| F3-1 | figures/condition_fn_fp_heatmap.png, tables/error_conditions.csv | 조건별 FN·FP |
| F3-2 | figures/evening_missed_cases.png, tables/evening_episode_details.csv | 저녁 미탐, 경보 겹침·매칭 원인 |
| F3-3 | figures/horizon_importance.png, tables/permutation_importance_top10.csv | 선택 LightGBM 또는 명시된 비교 모델 |
| F3-4 | figures/peak_type_distribution.png, tables/peak_type_summary.csv | 동시간 설명 기준선의 사후 분해·CI |
| F4-1 | figures/F4-1_forecast_alert_action.png, predictions/decision_feed.csv | 확률·여유·조치 적용 여부 |
| F4-2 | figures/alert_rule_tradeoff.png, tables/alert_rules.csv | 확인 규칙·준비 시간·선행시간 |
| F4-3 | figures/relative_economic_value.png, tables/relative_economic_value.csv | C/L 가정별 가치·날짜 CI |
| F4-4 | figures/tariff_counterfactual.png, tables/tariff_counterfactual_summary.csv | 단가 없는 시간대 순서 가중 시나리오 |
| F4-5 | figures/shift_scenario.png, tables/shift_summary.csv | 시간 선후 제약·새 피크 여부 |
| T5-1 | tables/T5-1_adoption.csv | 사전 조건별 채택·기각 |
| T6-1 | tables/T6-1_run_steps.csv, tables/T6-1_cold_run_steps.csv, logs/fresh_reproduction.json | 캐시 사용·캐시 없는 실행 시간 구분 |
| 보조 T2 | tables/t2_development_cv.csv, predictions/t2_development_oof.csv | 익일 최대와 기준선 |
| 확률 품질 | tables/probability_reliability.csv, figures/F2_probability_reliability.png | 초과 확률 신뢰도 |
| 기대 초과량 | tables/expected_exceedance_summary.csv | 순위상관·에피소드 최대 대리 비교 오차 |
| 적용 영역 | tables/applicability_distance.csv | 개발 평가 입력의 학습 분포 거리 |

파일명과 실제 존재 여부는 재현 검증에서 확인한다. 표·그림은 직접 고치지 않고 생성 코드를 통해 갱신한다.

## 제출 준비 상태

- 구현·개발 실험·분석·1~6장 Markdown 및 HTML 로드맵: 생성.
- 발표: 수정 가능한 14장 HTML·발표자 메모·PDF 초안. 지정 제작 런타임 부재로 PPTX 미생성.
- 로컬 재현 ZIP: 원자료와 코드를 포함. 공개 Git에는 원자료·모델·대용량 예측·ZIP을 제외.
- 최종 테스트·제출 예측: 날짜 잠금 유지. 완료 시 `freeze_record.json`과 각 결과 해시로 확인.
- 필수 성공 목표: CBL 대비 피크 MAE 유의 개선 미입증. q95 상위 커버리지 부족 폭을 정량 제시.
- 실제 비용: 계약종별·선택요금·단가 미확인. 상대 가정의 결과를 원화 절감으로 해석하지 않음.
- 외부 증빙: 본인 설문 캡처·문의 발송/답변·보고서 hwpx/PDF·포털 완료 화면 미확보.
- 선택 실험 생략: 파운데이션, 확률 분류B, τ90/97.5 재학습 및 T97.5 민감도, PDP/ICE. 설정에 남은 계획값을 실행 결과로 해석하지 않음.
- 블라인드: 자동 소스 검사와 최종 제출용 편집은 별개. 동결 증거의 과거 경로를 임의 삭제하지 않음.
