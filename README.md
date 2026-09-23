# 제6회 K-인공지능 제조데이터 분석 경진대회 · 과제 ⑤

선택 과제는 **⑤ 제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석**이다. 이 저장소는 원본 자료, 사전 타당성 검증, 화면 시제품, 제출 준비물을 서로 구분해 관리한다. 현재 시제품과 사전 검증 결과는 최종 제출 모델이나 실제 전력요금 절감의 증거가 아니다.

## 폴더 안내

| 경로 | 역할 |
| :-- | :-- |
| docs/official/ | 대회 공문 원본. 과제 문구, 평가표, 제출 형식의 기준 |
| docs/decisions/ | ⑤ 선택에 관한 내부 의사결정 기록. 제출물 아님 |
| data/raw/task05_power/ | 선택 과제의 원본 CSV와 원본 ZIP 사본 |
| data/raw/task02_welding/, data/raw/task03_press/ | 과제 비교에 사용한 원본 자료 |
| data/README.md | 원본 파일의 이전·현재 경로와 SHA-256 대응표 |
| verification/ | ②·③·⑤ 사전 타당성 검증 코드와 당시 결과 스냅샷 |
| prototype/ | ⑤의 과거 시점 전력예측 화면 시제품과 테스트 |
| solution/ | 향후 ⑤ 제출용 학습·추론 구현 공간. 아직 최종 모델 없음 |
| submission/ | 공문에 맞춘 제출물 준비 안내. 완성된 제출물은 아직 없음 |

원본 자료는 변경하지 않는다. data/raw/는 Git에서 제외하며, 필요한 팀원에게 원본을 별도로 전달한다. 기존 verification/results/의 보고서에는 재편 이전 원본 경로가 남아 있다. 이는 당시 실행 기록으로 보존했고, 새 경로는 data/README.md에서 확인한다.

## 현재 상태

- ⑤의 1시간 후 15분 전력값 예측과 피크 사건을 사전 검증했다. 계절 나이브, persistence, LightGBM 비교는 [05b 보고서](verification/results/05b_report.md)에 있다.
- ②·③·⑤의 데이터가 뒷받침하는 주장 범위는 [비교 보고서](verification/results/comparison_2_3_5.md)에 남겼다. 과제 선택은 ⑤로 확정했지만 이 비교 결과를 수상 우열로 해석하지 않는다.
- [화면 시제품](prototype/README.md)은 과거 시점의 예측·불확실성·운영자 검토 상태를 확인하는 용도다. 제출용 학습·추론 파이프라인은 별도로 확정해야 한다.

## 실행

저장소 루트에서 Python 3.13과 프로젝트 가상환경을 사용한다.

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run prototype/app.py --server.address 127.0.0.1
    .\.venv\Scripts\python.exe -m unittest discover -s prototype/tests -v

검증 스크립트의 별도 의존성은 verification/requirements.txt에 고정했다. 이전 결과 파일을 유지할 때는 verification/run_all.py를 이 작업 폴더에서 재실행하지 않는다. 이 스크립트는 기존 결과 스냅샷을 덮어쓴다.

## 공문과 제출

[공문 원본](<docs/official/제6회 K-인공지능 제조데이터 분석 경진대회 과제공개(일반국민,대학(원)생) (1).hwpx>)에 적힌 제출 기한은 **2026년 10월 8일 23:59**다. 공문은 보고서 PDF, 소스코드 ZIP, 발표자료 PDF·PPT, 설문 완료 화면을 요구하며 모든 제출물에서 소속·로고 등 참가자 식별정보를 금지한다. 상세 파일 구성은 [제출 안내](submission/README.md)에 정리했다. 이후 공지 변경 여부는 제출 전에 공식 포털에서 다시 확인해야 한다.
