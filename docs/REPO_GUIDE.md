# 저장소 안내 (내부용, 제출 ZIP에 포함하지 않음)

루트 `README.md`는 제출 ZIP에 들어가는 평가자용 설명이다. 이 문서는 저장소 전체의 폴더 역할을 정리한다.

| 경로 | 역할 | 제출 ZIP |
| :-- | :-- | :-: |
| `run_all.py`, `configs/`, `src/`, `tests/`, `requirements.txt`, `README.md` | 제출 파이프라인 | 포함 |
| `data/raw/task05_power/okm_augumented_2021.csv` | 학습용 원자료. Git 추적 제외, 팀원에게 별도 전달 | 포함 |
| `outputs/` | `run_all.py` 산출물 | 포함 |
| `report/` | 보고서 장별 초안(md)과 벤치마킹 자료 | 제외 |
| `submission/` | 제출물 조립 공간. `tools/package_submission.py`가 `submission/source/`에 ZIP 생성 | — |
| `tools/` | 제출 ZIP 생성·블라인드 점검 | 제외 |
| `verification/` | 과제 선택 전 사전 검증 코드와 결과 스냅샷. **동결, 수정 금지** | 제외 |
| `prototype/` | 과거 시점 전력예측 화면 시제품(Streamlit). 의존성은 `prototype/requirements.txt` | 제외 |
| `docs/official/` | 대회 공문 원본 | 제외 |
| `docs/decisions/` | 과제 선택 의사결정 기록 DOCX (내부) | 제외 |
| `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md`, `eval_protocol.md` | 작업 규칙·진행·결정 기록 | 제외 |

## 사전 검증 결과의 재현 관계

`src/`는 `verification/t5_power.py`, `t5b_persistence.py`의 계산을 동작 변경 없이 옮긴 것이다.
같은 합성 입력에서 두 코드의 예측값·지표·신뢰구간·rolling-origin·조건별 집계·익일 최대 결과 126개 항목이
수치적으로 같음을 확인했다(2026-10-07, `PROGRESS.md`). 원자료에서는 `outputs/logs/run_summary.json`의
`reproduction_check`가 persistence 27.73, LightGBM 17.77(피크 위치 MAE)과의 일치 여부를 기록한다.

## 화면 시제품 실행

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r prototype\requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run prototype/app.py --server.address 127.0.0.1

## 사전 검증 스크립트

`verification/README.md` 참고. `verification/run_all.py`와 `t2_weld.py`는 기존 결과 스냅샷을 덮어쓰므로
이 저장소에서 직접 실행하지 않는다.
