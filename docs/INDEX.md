# 문서 안내

| 목적 | 문서 | 상태와 사용 규칙 |
| --- | --- | --- |
| 실행 시작 | [README](../README.md) | 설치·한 줄 실행·동결 및 제출 경계 |
| 작업 규칙·가정 | [CLAUDE](../CLAUDE.md) | A1~A8 유지. 구조·일정은 설계서가 대체 |
| 무엇을 만드는가 | [PROJECT_DESIGN](../PROJECT_DESIGN.md) | 사용자 설계와 무개입 실행 해석 |
| 판단 근거 | [DECISIONS](../DECISIONS.md) | 설계 모순 해소·분석 한계·채택 정책 |
| 사전 평가 계획 | [eval_protocol](../eval_protocol.md), [채택 기준](../outputs/logs/adoption_criteria.md) | 개발 결과 전에 고정한 기준 |
| 현재 완료 증거 | [PROGRESS](../PROGRESS.md), [실행 기록](../outputs/logs/run_status.json) | 계획일과 실제 수행을 구분 |
| 시각적 로드맵 | [roadmap.html](roadmap.html) | 보고서 단계에서 자동 생성 |
| 본문 초안 | [REPORT_DRAFT](../report/REPORT_DRAFT.md) | 1~6장 통합; 장별 파일도 제공 |
| 발표 초안 | [14장 HTML](../slides/development_deck.html), [PDF](../slides/development_deck.pdf), [발표자 메모](../slides/speaker_notes.md) | 개발 결과. PPTX 미생성 |
| 요구 대응 | [DELIVERABLES](DELIVERABLES.md) | 설계서의 표·그림 및 미완료 항목 대응 |
| 공개 근거 | [SOURCES](SOURCES.md), [요금 검증](tariff_sources.md) | 공식 출처·검증 범위·미확인 항목 |
| 주최 문의 | [organizer_inquiry](organizer_inquiry.md) | 발송하지 않은 문안 |
| 사람 인계 | [HANDOFF](../submission/HANDOFF.md) | 10/5~6 hwpx·PDF 작업 및 외부 증빙 |
| 과거 증거 | [verification](../verification/) | 동결. 새 모델 결과와 혼용 금지 |
| 이전 시제품 | [prototype](../prototype/) | 기존 Streamlit 앱, 이번 대회 평가 파이프라인과 별개 |

`outputs/smoke*`, `_validation/`, 로그의 `.log` 파일은 로컬 검증용이다. 공개 Git에는 원자료·모델 바이너리·대용량 예측·ZIP을 올리지 않는다. 로컬 재현 ZIP에는 필요한 원자료가 들어 있으므로 원자료 사용 조건을 따른다.

보고서 표·그림은 직접 수정하지 않고 `python run_all.py --from report`로 갱신한다. 동결 후에는 모델·기준·테스트를 변경하지 않는다.
