# 결과 보고서 초안 (v0.1, 2026-10-07)

대회 양식(`제6회 K-인공지능 제조데이터 분석 경진대회 보고서` hwpx)의 장 구성을 그대로 따르고,
서술은 논문 형식(주장 → 방법 → 결과 → 대안 설명 점검 → 판정)으로 쓴다. 작성 규칙은
[benchmark_papers.md](benchmark_papers.md) 3절(R1–R10)에 있다. 분량 조정은 hwpx로 옮길 때 한다.

## 파일과 양식 대응

| 파일 | 양식 항목 | 배점 |
| :-- | :-- | --: |
| [00_summary.md](00_summary.md) | 표지: 프로젝트명·팀명·내용요약 | — |
| [ch1_data.md](ch1_data.md) | 제1장 데이터 이해 및 진단 | 15 |
| [ch2_model.md](ch2_model.md) | 제2장 AI 예측모델 개발 및 성능평가 | 40 |
| [ch3_errors.md](ch3_errors.md) | 제3장 영향요인 및 오류분석 | 15 |
| [ch4_field.md](ch4_field.md) | 제4장 현장 활용방안 | 10 |
| [ch5_novelty.md](ch5_novelty.md) | 제5장 창의성 및 차별성 | 10 |
| [ch6_repro.md](ch6_repro.md) | 제6장 코드 구성 및 재현성 | 10 |
| [ch7_conclusion.md](ch7_conclusion.md) | 추가 기술(자유): 결론·한계·향후 과제, 참고문헌, 만족도 조사 캡처 | — |

## 수치 출처 표기 (hwpx 변환 시 삭제)

| 표기 | 뜻 |
| :-- | :-- |
| 〔V〕 | 사전 검증 실행 결과(`verification/results/05_summary.json`, `05b_summary.json` 등). 제출 파이프라인(`src/`)이 같은 계산을 한다(합성 자료 126개 항목 동일 확인). 원자료로 `python run_all.py`를 실행해 `outputs/logs/run_summary.json`의 `reproduction_check`가 통과하면 이 표기를 지운다 |
| 〔D〕 | 사전 검증의 동결 예측 파일(`05b_predictions.csv`, `05_test_predictions.csv`)을 모델·임계값 변경 없이 다시 집계한 기술 통계(2026-10-07). 같은 계산이 `run_all.py`에 들어 있다 |
| 〔R: 파일〕 | 원자료로 `run_all.py`를 실행한 뒤 해당 출력 파일에서 채울 값. 아직 수치 없음 |

수치는 모두 `점추정 [날짜 블록 부트스트랩 95% 신뢰구간]`이다. 전력 단위는 미확인이므로 "원자료 단위"로 쓴다.

## hwpx 변환 체크리스트

- [ ] 원자료로 `python run_all.py` 실행 → `reproduction_check.passed = true` 확인 → 〔V〕 표기 삭제
- [ ] 〔R〕 자리를 `outputs/tables/`의 값으로 채우고, 결과가 초안의 서술과 다르면 서술을 결과에 맞춰 고친다(반대로 하지 않는다)
- [ ] 그림은 `outputs/figures/`에서 가져온다(손으로 만든 그림 금지)
- [ ] 본문 휴먼명조 14 / 줄간격 160, 세부 휴먼명조 10, "작성 요령" 상자 삭제
- [ ] 블라인드: 학교·기관명, 로고, 개인 경로·계정, 문서 속성의 작성자·회사 필드 제거
- [ ] 만족도 조사 완료 화면 캡처 첨부(필수)
