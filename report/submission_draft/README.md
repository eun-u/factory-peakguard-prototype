# 제출용 보고서 초안

| 파일 | 내용 |
| :-- | :-- |
| `report_draft_v1.md` | 논문형 보고서 초안 원문 (대회 양식 1~6장 + 결론·참고문헌·설문 자리) |
| `../../submission/report/report_draft_v1.docx` | 위 원문의 Word 변환본 (휴먼명조 14/10, 줄간격 160) |

## 다시 만들기

    python scripts/submission_report/analysis.py                 # 3·4장 표·그림 (원자료 필요, CPU 약 20초)
    node scripts/submission_report/build_docx.js                  # Word 변환 (npm docx 필요)

## 수치 출처

- 2장 모델 비교·경보·불확실성: `outputs/phase_f/final_fg_r11/FINAL_TEST_RESULT.json`
- 테스트 평가 이력과 라운드별 개발 수치: `outputs/phase_f/final_fg_r8/FINAL_TEST_SUMMARY.md`, `outputs/phase_f/goal_fm_ensemble_v1/{RESULTS,MOS_RESULTS,TH_FT_RESULTS,SWEEP_RESULTS}.md`
- 1장 데이터 진단: `report/REPORT_DRAFT.md`의 FG-R11 블록과 데이터 진단 JSON
- 3·4장 조건 분석·REV·이동 시나리오: `outputs/tables/submission/*.csv`, `summary.json`
- 사전 검증 비교 수치: `verification/results/05b_summary.json`

수치를 바꿀 때는 원 결과 파일을 고친 뒤 이 원문을 갱신하고 Word를 다시 만든다. hwpx 이관 전에 팀명, 설문 캡처, 한전 약관·Chronos-2 서지 원문 대조가 필요하다.
