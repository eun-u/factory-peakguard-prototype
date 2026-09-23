# ③·⑤ 사전 타당성 검증 재실행

저장소 루트에서 Python 3.13을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r verification\requirements.txt
.\.venv\Scripts\python.exe -X utf8 verification\run_all.py
```

`run_all.py` 한 번으로 인벤토리, ③ P0–P5, ⑤ Q0–Q5, 통합 보고서를 순서대로 다시 만든다. 결과는 `verification/results/`에 저장한다. 원본 CSV는 수정하지 않는다. 데이터 파일명은 고정하지 않고 스키마로 찾는다. ③ 정상·이상 데이터가 없으면 해당 실험은 `판정불가`로 출력한다.

사전 고정 기준은 별도 커밋 `d4be250`의 `results/00_decision_criteria.md`이며, 분석 뒤 변경하지 않았다. 원자료 목록과 해시는 `results/01_inventory.md`, 비교 결과와 가정은 `results/report.md`에 있다. 보고서의 신뢰구간은 1000회 재표집 결과이다.
