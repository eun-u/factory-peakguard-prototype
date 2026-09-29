# PeakGuard — 제조 전력피크 예측 연구

**제조 전력의 피크 예측 정확도와 선행 경보의 유효성을 실험으로 검증하는 응용연구 프로젝트.**

주 과제는 한 시간 뒤에 끝나는 15분 구간의 전력값 예측이다. 피크 오차 개선을 중심으로 불확실성·오경보·준비시간·실패 조건을 분석한다. 현재 결과는 개발 평가이며 실제 현장 절감 효과를 입증한 시스템이 아니다.

## 먼저 읽을 것

| 목적 | 문서 |
|---|---|
| Background~Contribution 12항목과 RQ | [연구 현황](research/generated/status.md) / [원본 정의](research/charter.json) |
| 가설·실험·다음 행동 | [실험 관리](experiments/README.md) / [등록부](experiments/registry.json) |
| 현재 주장과 근거 | [주장 등록부](research/claims.json) / [현재 스냅샷](research/generated/current_snapshot.json) |
| 논문 초안 | [본문](paper/manuscript.md) / [집필과 대회 목차 매핑](paper/README.md) |
| 현재 수치 | [자동 결과표](paper/generated/results.md) |
| 작업 규칙·변경 이력 | [AGENTS.md](AGENTS.md), [PROGRESS.md](PROGRESS.md), [DECISIONS.md](DECISIONS.md) |

## 연구 관리와 집필 — Python 3.10+, 추가 패키지 불필요

```bash
python -m research status
python -m research show R2-EX02
python -m research validate
python -m research build
python -m research check
python -m unittest discover -s research/tests -v
```

이 명령은 Git에 있는 작은 결과 파일만 읽는다. 원자료·GPU·모델 학습 없이 연구 현황과 본문을 생성한다. 계획의 존재와 검증된 성과를 구분하며, 근거 누락·비교 행 중복·모델 선정 불일치·생성물 갱신 누락은 실패로 처리한다.

## 구조

| 위치 | 역할 |
|---|---|
| research/ | 정의·주장·근거·검증·문서 생성 |
| experiments/ | 연구 질문별 계획과 기존 실험 이력 |
| paper/ | 수동 집필 절 + 자동 결과 + 통합 본문 |
| src/, research_p4/, scripts/ | 예측·평가·기존 실험 구현 |
| outputs/, verification/ | 결과·사전 계획·동결 증거 |
| prototype/ | 기존 Streamlit 흐름 시연 |
| report/, slides/, submission/ | 이전 산출물과 대회 제출 전달 |

## 모델 개발 실행 — 별도 데이터·환경 필요

원자료는 Git에 포함하지 않는다. 사용 권한이 있는 원자료를 `data/raw/task05_power/okm_augumented_2021.csv`에 두고 기존 환경 잠금 파일을 사용한다. 학습 환경 기준은 Python 3.13 및 `requirements-pipeline.lock.txt`이며 연구 관리의 표준 라이브러리 환경과 구별한다.

```bash
python -m pip install -r requirements.txt
python run_all.py --development-only
```

개발 명령은 기존 봉인 개발 로더와 검증을 사용한다. 모델/설정 변경은 캐시를 무효화하며 새 개발 실행이 필요하다. `run_all.py --from report`는 기존 전체 데이터 단계의 게이트와 연결되어 있으므로 일반 집필 명령으로 쓰지 않는다. 논문 갱신은 `python -m research build`를 사용한다.

최종 평가는 기존 날짜·사람 승인·해시·일회 실행 조건을 따른다. 문서 생성·구조 개편·main 반영은 동결 승인이나 예약 활성화를 뜻하지 않는다. 과거 마지막 15% 열람 이력, 단위·시간 경계 미확인, 통계적 피크와 실제 계약 한도의 차이를 유지한다.

## 과거 결과와 제출

현재 선정은 [스냅샷](research/generated/current_snapshot.json)으로 확인한다. P4/P5/P6는 기존 기록으로 보존하고 새 실험은 R2-EX01~06의 planned 상태에서 시작한다. [기존 보고서](report/REPORT_DRAFT.md)·발표·프로토타입은 생성 시점과 모델 범위가 다를 수 있다.

논문형 본문을 대회 6장과 배점(15/40/15/10/10/10)에 맞춰 옮긴다. PDF·PPT·코드/데이터·예측결과·설문 및 블라인드 조건은 [인계](submission/HANDOFF.md)와 공식 양식을 따른다. Git push를 대회 제출 완료로 표현하지 않는다.
