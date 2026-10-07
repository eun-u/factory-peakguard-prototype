# PeakGuard 에이전트 진입점

프로젝트는 사용자 결정에 따라 논문형 연구 관리로 전환했다. 모든 에이전트는 [AGENTS.md](AGENTS.md)를 공통 작업 규칙으로 읽는다.

- 연구 정의: [research/charter.json](research/charter.json)
- 현재 질문·주장·실험: [research/generated/status.md](research/generated/status.md)
- 과학적 평가 규칙: [eval_protocol.md](eval_protocol.md), [configs/default.yaml](configs/default.yaml), 각 역사적 사전 계획
- 실행: `python -m research status`, `python -m research check`

이전 단계별 마스터 문서는 [보존본](docs/archive/pre_paper_20260929/CLAUDE.md)에 있다. 오래된 일정·선정 수치를 현재 기준으로 사용하지 않는다. 기존 데이터 가정, 테스트 열람 공개, 날짜·사람 승인·해시·일회 평가 조건은 AGENTS.md와 과학적 프로토콜에서 유지한다.
