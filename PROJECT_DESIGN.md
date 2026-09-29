# 프로젝트 설계 — 논문형 연구 관리

확정일: 2026-09-29. 제조 전력피크 예측과 불확실성 기반 선행 경보를 다루는 기술 중심의 응용연구다.

기본 정의 12항목과 연구 질문은 [research/charter.json](research/charter.json), 읽을 현황은 [연구 현황](research/generated/status.md)에 있다. 이 문서에 선정 수치·현재 실험 상태를 중복 기재하지 않는다.

| 영역 | 책임 |
|---|---|
| research/ | 연구 정의·주장·근거·생성기와 검증 |
| experiments/ | 가설·계획·상태·기존 구현/결과 연결 |
| paper/ | 선행연구·방법·설계·결과·논의·위협·결론 집필 |
| src/, research_p4/, scripts/ | 기존 과학적 계산과 실험 실행 |
| outputs/, verification/ | 기존 결과와 보존 증거 |
| prototype/ | 과거 흐름 시연; 연구 결과와의 연결은 후속 실험 |
| report/, slides/, submission/ | 기존 산출 및 대회 양식 전달 |

새 연구 방향은 이전 구현을 재사용한다. 실제 예측기와 경보 정책은 실험으로 평가한다. 파일 이동이나 시스템 기능 수를 연구 기여로 삼지 않는다. 독창성과 현장 효과는 그에 맞는 증거가 필요하다.

기존 설계와 A1~A8의 구체 가정은 [보존본](docs/archive/pre_paper_20260929/PROJECT_DESIGN.md)에 있다. 과학적 프로토콜과 최종 평가 경계는 그대로 유효하다. 관련 결정·검증 범위는 DECISIONS.md와 PROGRESS.md를 따른다.
