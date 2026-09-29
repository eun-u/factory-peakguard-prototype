# 문서 안내

| 목적 | 기준 문서 |
|---|---|
| 프로젝트 진입점 | [README](../README.md) |
| 에이전트 작업 규칙 | [AGENTS](../AGENTS.md) |
| 연구 기본 정의·현재 상태 | [연구 현황](../research/generated/status.md), [정의 원본](../research/charter.json) |
| 주장·근거 | [claims.json](../research/claims.json), [현재 스냅샷](../research/generated/current_snapshot.json) |
| 실험 계획·이력 | [실험 관리](../experiments/README.md), [등록부](../experiments/registry.json) |
| 논문 집필 | [paper](../paper/README.md), [본문](../paper/manuscript.md) |
| 과학적 평가 | [평가 프로토콜](../eval_protocol.md), [기존 채택 기준](../outputs/logs/adoption_criteria.md) |
| 진행·결정 | [PROGRESS](../PROGRESS.md), [DECISIONS](../DECISIONS.md) |
| 기존 서비스 리뷰 | [9월 28일 기록](reviews/2026-09-28/service_review.md) — 당시 모델 기준 |
| 기존 제출 산출 | [보고서](../report/REPORT_DRAFT.md), [전달 목록](DELIVERABLES.md), [인계](../submission/HANDOFF.md) |
| 도메인 출처 | [SOURCES](SOURCES.md), [요금 출처](tariff_sources.md) |
| 구조 개편 기록 | [리팩토링 검토](REFACTOR_20260929.md) |
| 이전 설계·지침 | [보존본](archive/pre_paper_20260929/) — 역사적 자료 |

집필 갱신은 `python -m research build`, 정합성 확인은 `python -m research check`. 이 경로는 전체 데이터 로더나 최종 평가를 호출하지 않는다. 과거 `run_all.py --from report` 안내는 일반 집필 경로로 사용하지 않는다.
