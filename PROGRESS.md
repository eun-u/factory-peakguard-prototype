# 진행 기록

## [Phase 0·5 / 2026-10-07]

오늘 작업은 로드맵상 Phase 0(기반 구축: 사전 검증 로직 이식)과 Phase 5(재현성·보고서 완성)에 해당한다. 일정상으로는 내부 목표 제출일(10-07)이며 Phase 1~4의 신규 실험은 대부분 수행되지 않은 상태에서 진행했다.

**완료**
- 제출 소스 트리 구성: `run_all.py`, `configs/default.yaml`, `src/`(data, features, split, evaluate, daily_max, viz, models/, analysis/), `tests/`, `tools/package_submission.py`, 평가자용 `README.md`, 고정 버전 `requirements.txt`.
- 사전 검증(`t5_power.py`, `t5b_persistence.py`)과의 동일성: 같은 합성 입력에서 126개 출력 항목 일치.
- 추가 분석 코드: 변수 사전·0값 구간·달력 분포, 피크 정의 민감도, 개발 폴드 예측거리 곡선·conformal, 시험 이전 피크 조건, permutation importance·PDP, k/n 경보 규칙, 생산량 이동 시뮬레이션(4방식).
- 단위 점검 8개 통과, ZIP 생성 → 새 폴더 재실행 결과 일치(합성 자료), 블라인드 점검 음성 시험 통과.
- 보고서 초안 v0.1: `report/00_summary.md` ~ `ch7_conclusion.md`, 벤치마킹 자료 `report/benchmark_papers.md`.
- `DECISIONS.md`, 사후 문서화 `eval_protocol.md` 작성.

**핵심 수치** (사전 검증 〔V〕, 동결 예측 재집계 〔D〕)
- 피크 위치 MAE: persistence 27.73 [25.45, 30.53] → LightGBM 17.77 [13.88, 22.28], 개선률 0.359 [0.196, 0.501] 〔V〕
- 에피소드 F1 0.451 → 0.636, 위치 오경보 774 → 305 〔V〕
- 최종 조치 경보 기준 16–24시 재현율 0.388 [0.250, 0.561], 08–16시 0.854 [0.773, 0.920] 〔D〕 (분류기 경보 기준 사전 검증값 0.143은 다른 경보 방식)
- 08–16시가 오경보의 83%, 생산량 > 636 구간이 90% 〔D〕
- 월요일 FN 30개 중 15개가 2021-08-09 하루(장기 휴무 후 첫 가동일 추정). 이 날 제외 시 월요일 재현율 0.762 〔D〕
- REV: C/L 0.25~0.40에서 0.16~0.20, 구간 하한 > 0 〔V〕
- 부정적 결과: 0.95 분위수 커버리지 0.911, 익일 최대 MAPE 28.3% vs 나이브 18.6% 〔V〕

**완료 기준 충족 여부**
- Phase 0 기준("run_all.py가 27.73 / 17.77 재현"): **미확인**. 원자료(`data/raw/`)가 Git 추적 제외라 이 작업 환경에 없었다. 합성 자료 동일성은 확인. 원자료로 한 번 실행하면 `outputs/logs/run_summary.json`의 `reproduction_check.passed`로 판정된다.
- `run_all.py` 오류 없이 실행: 합성 자료에서 확인(약 65초).

**막힌 점 / 사람 확인 필요**
1. 원자료로 `python run_all.py` 실행 → 재현 확인, 보고서 〔R〕 채우기.
2. DECISIONS 2026-10-07 #1(최종 모델 = 미튜닝 고정 설정) 승인.
3. 만족도 조사 캡처, 팀명, hwpx 서식 변환, 발표자료(Phase 6) 미착수.
4. 주최측 문의 답변 여부 미기록.
5. 4.4절 한전 요금적용전력 설명은 약관 원문 대조 필요. 참고문헌 6~10번 서지 확인 필요.

**다음 할 일**
1. 원자료 배치 후 `python run_all.py` → `reproduction_check` 확인.
2. 〔R〕 자리 채우기(예측거리 곡선, 민감도, conformal, 경보 규칙, 피크 조건, 영향변수, 이동 시뮬레이션). 결과가 초안 서술과 다르면 서술을 고친다.
3. `python tools/package_submission.py` → ZIP을 새 폴더에 풀어 재실행.
4. hwpx 변환·PDF, 발표자료 PDF·PPT, 설문 캡처, KAMP 제출.
