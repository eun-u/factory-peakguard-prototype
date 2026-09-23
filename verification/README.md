# ②·③·⑤ 사전 타당성 검증 재실행

저장소 루트에서 Python 3.13으로 ②와 비교문서만 재실행한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r verification\requirements.txt
.\.venv\Scripts\python.exe -X utf8 verification\t2_weld.py
```

`t2_weld.py` 한 번으로 ②의 구조 진단·보고서와 ②·③·⑤ 비교문서를 만든다. 비교문서에서 ③·⑤ 수치는 기존 보고서에서 인용하며 재계산하지 않는다. 기존 ③ P0–P5와 ⑤ Q0–Q5를 별도로 재계산할 때만 `verification\run_all.py`를 실행한다. 결과는 `verification/results/`에 저장한다. 원본은 수정하지 않는다. ② 스크립트는 사용자 지정 ② 폴더의 워크북을 발견한 뒤 시트 구조를 검사한다. 개별 불량 라벨이 확인되지 않으면 감독학습 지표를 `판정불가`로 출력한다.

③·⑤ 사전 기준은 별도 커밋 `d4be250`의 `results/00_decision_criteria.md`, ② 사전 기준은 `c2849a8`의 `results/02_decision_criteria.md`이며 분석 뒤 변경하지 않았다. ② 구조와 가정은 `results/02_report.md`, 세 과제 비교는 `results/comparison_2_3_5.md`에 있다. 유효한 지표의 신뢰구간은 1000회 재표집 결과이다.
