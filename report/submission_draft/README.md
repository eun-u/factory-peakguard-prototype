# 제출용 보고서 초안

| 파일 | 내용 |
| :-- | :-- |
| `report_draft_v2.md` | **현재 초안.** 논문형 보고서 원문 (요약·출제 요구 대응표·연구 기여, 대회 양식 1~6장, 결론·참고문헌·설문 자리). 4장에 일·월 최대 사전 포착, 비용-손실(REV), 피크전력 저감 방안 3종 포함 |
| `../../submission/report/report_draft_v2.docx` | v2의 Word 변환본 (휴먼명조 14/10, 줄간격 160, 문서 속성 작성자·회사 공란) |
| `report_draft_v1.md`, `../../submission/report/report_draft_v1.docx` | 이전 초안 (평가 이력·한계 서술을 포함한 판본, 비교용 보존) |

## 다시 만들기

    python scripts/submission_report/analysis.py                 # 2~4장 표·그림 (원자료 필요, CPU 약 20초, reduction.py 포함)
    node scripts/submission_report/build_docx.js                  # v2 Word 변환 (npm docx 필요)

## 수치 출처

- 2장 모델 비교·경보·불확실성: `outputs/phase_f/final_fg_r11/FINAL_TEST_RESULT.json`
- 테스트 평가 이력과 라운드별 개발 수치: `outputs/phase_f/final_fg_r8/FINAL_TEST_SUMMARY.md`, `outputs/phase_f/goal_fm_ensemble_v1/{RESULTS,MOS_RESULTS,TH_FT_RESULTS,SWEEP_RESULTS}.md`
- 1장 데이터 진단: `report/REPORT_DRAFT.md`의 FG-R11 블록과 데이터 진단 JSON
- 3·4장 조건 분석·REV·이동 시나리오: `outputs/tables/submission/*.csv`, `summary.json`
- 4장 일·월 최대 사전 포착과 목표 최대수요 운영: `scripts/submission_report/reduction.py` → `daily_max_capture.csv`, `daily_max_lead_curve.csv`, `demand_cap_scenarios.csv`, `demand_cap_events.csv`, `reduction_summary.json`
- 사전 검증 비교 수치: `verification/results/05b_summary.json`

수치를 바꿀 때는 원 결과 파일을 고친 뒤 이 원문을 갱신하고 Word를 다시 만든다. hwpx 이관 전에 팀명, 설문 캡처, 한전 약관·Chronos-2 서지 원문 대조가 필요하다.
