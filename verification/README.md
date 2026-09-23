# 사전 타당성 검증 기록

과제 ②·③·⑤를 비교한 뒤 ⑤를 선택했다. 이 폴더는 선택 전 검증과 ⑤ persistence 추가 비교의 코드를 보존한다. 기존 results/의 파일은 **당시 실행 스냅샷**이며, 그 안의 원본 상대경로는 폴더 재편 전 경로다. 현재 파일 위치와 SHA-256은 ../data/README.md에 기록했다.

원본 자료는 ../data/raw/task02_welding/, task03_press/, task05_power/에 있다. ② 스크립트는 task02_welding 폴더에서 워크북을 발견하고 시트 구조를 검사한다. ③·⑤ 스크립트는 원자료 스키마로 파일을 탐색한다.

## 실행

저장소 루트에서 Python 3.13과 verification/requirements.txt의 패키지를 사용한다.

    .\.venv\Scripts\python.exe -m pip install -r verification\requirements.txt
    .\.venv\Scripts\python.exe -X utf8 verification\t5b_persistence.py

과거 결과를 별도 작업 공간에서 재현할 때 사용할 명령:

    .\.venv\Scripts\python.exe -X utf8 verification\t2_weld.py
    .\.venv\Scripts\python.exe -X utf8 verification\run_all.py

t2_weld.py는 02_ 보고서와 비교문서를, run_all.py는 인벤토리·③·⑤의 기존 보고서를 **덮어쓴다**. 현재 저장소의 과거 스냅샷을 유지할 때는 별도 복사본이나 작업 트리에서 실행한다. t5b_persistence.py는 기존 05_ 테스트 예측과 새 계산을 대조하며 05b_ 결과만 생성한다.

사전 기준은 00_decision_criteria.md, 02_decision_criteria.md, 05b_decision_criteria.md에 고정했다. 개별 불량 라벨이 없는 ②에는 지도학습 지표를 만들지 않는다.
