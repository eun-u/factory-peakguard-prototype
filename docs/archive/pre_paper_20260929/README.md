# 과제 ⑤ 전력사용량 예측 및 최대피크 위험조건 분석

15분 전력의 피크·초과 크기·불확실성을 개발 교차검증으로 비교하고 CBL 기준선과 운영 시나리오를 분석한다.
**2026-09-24 체크포인트는 개발 결과이며 최종 테스트는 2026-10-01 18:00 KST까지 잠겨 있다.** 최신 상태는 `outputs/logs/run_status.json`을 따른다.

[로드맵 HTML](docs/roadmap.html) · [설계](PROJECT_DESIGN.md) · [결정](DECISIONS.md) · [진행](PROGRESS.md) · [보고서](report/REPORT_DRAFT.md) · [인계](submission/HANDOFF.md)

전체 문서의 역할과 생성 규칙은 [문서 안내](docs/INDEX.md)에 정리했다.

개발1시간 피크 MAE는 선정14.44·CBL18.26·persistence34.26이다. CBL 대비 개선95% CI가0을 포함하여 필수 성공 목표는 미입증이다.
상위 q95 커버리지는92.9%이며 부족 폭과 폴드별 실패를 보고한다. 자동 테스트45개, 캐시 없는 독립 실행26분52초와 수치 일치를 검증했다([재현 기록](outputs/logs/fresh_reproduction.json)).

## 설치와 실행

Python3.14 CPU 환경. 대회 CSV를 `data/raw/task05_power/okm_augumented_2021.csv`에 둔다.
SHA-256: `8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830`.
원자료는 공개 Git에서 제외하며 로컬 개발 재현ZIP에는 학습 CSV가 들어 있다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pipeline.lock.txt
.\.venv\Scripts\python.exe run_all.py
```

가상환경 활성화 후에는 `python run_all.py` 한 줄이다.
검증 환경은 Windows·Python3.14이며 lock 파일은 실제 재현 실행에 사용한 버전을 고정한다.
`requirements.txt`는 직접 의존성, `requirements-pipeline.lock.txt`는 전이 의존성까지 포함한 이번 결과의 설치 기준이다.
data→development→final 날짜검사→analysis→report→package 순서로 실행한다.
특징·기준선·점예측·분위수·위험·선정은 development 엔진에서 처리한다.

```powershell
python -m pytest tests -q
python run_all.py --only data
python run_all.py --from analysis
python run_all.py --only final
python run_all.py --rebuild-dev
```

`--only`/`--from`은 `data`, `development`, `final`, `analysis`, `report`, `package`를 받는다.
코드·설정·원본·기준 및 저장된 예측·표·모델 해시가 같은 개발 캐시만 재사용한다. 날짜 잠금 전 final은 테스트를 열지 않는다.
CPU30분은 목표이며 실제 시간과 상태는 `outputs/logs/run_status.json`을 따른다.

현재 작업 PC에는 `FactoryPeakguard-Task05-Freeze-20261001` 예약 작업을 등록했다.
10월1일18시 KST 이후 `.venv`로 `run_all.py --from final`을 실행해 최종 평가·분석·보고서·ZIP을 갱신한다.
PC가 사용 가능하고 해당 Windows 사용자가 로그인한 상태여야 하며, 놓친 시각은 다음 사용 가능 시점에 실행한다.
등록 상태는 `outputs/logs/scheduled_freeze.json`, 실행 로그는 `outputs/logs/scheduled_freeze.log`다.
다른 PC에서 재등록하려면 `powershell -NoProfile -File scripts/register_freeze_task.ps1`을 사용한다.
완료된 테스트는 해시가 같은 결과만 재사용하고, 중단된 예약은 자동 재평가하지 않는다.
동결 완료 뒤 개발 캐시 재생성과 모델 재선정을 거부한다.

## 구조와 산출물

| 위치 | 내용 |
| --- | --- |
| configs/ | 기본설정·요금시간대·검증해시 |
| src/, tests/ | 데이터·모델·분석·실행·검증 |
| outputs/predictions/development_oof.csv | 개발 평가 예측, 테스트 아님 |
| outputs/tables/development_cv.csv | 모델·폴드·거리 성능 |
| outputs/logs/development_selection.json | 채택·선정 증거 |
| outputs/tables/t2_development_cv.csv | 익일 최대 보조과제 |
| outputs/figures/ | 한국어 그림 |
| report/, slides/ | 보고서1~6장·14장 발표 HTML/PDF·발표자 메모 |
| docs/roadmap.html | 제공 로드맵을 갱신한 문서 |
| submission/source/ | 개발 재현ZIP·해시 |
| verification/ | 동결된 과거 사전검증 |
| prototype/ | 기존 Streamlit 시제품 |

HTML: `python -m http.server 8765 --bind 127.0.0.1` 후 `/docs/roadmap.html`.
선택 화면검증: `pip install playwright`, `python -m playwright install chromium`, `python scripts/qa_roadmap.py`.

보고서 단계는 표·그림에서 `slides/development_deck.html`과 `slides/speaker_notes.md`도 생성한다.
발표 PDF는 선택 도구로 `pip install -r requirements-artifacts.txt`, `python -m playwright install chromium`,
`python scripts/export_slides.py`를 실행한다. PDF 생성은 모델 재현에 필요한 의존성이 아니다.
14장·16:9·텍스트 보존·페이지 밖 넘침을 검사하며 렌더링 검토 기록은 `outputs/logs/slides_validation.json`이다.
PPTX는 지정 제작 런타임 부재로 미생성이다. 발표 HTML 원본과 PDF 초안은 최종 테스트 수치를 포함하지 않는다.

## 해석과 제출 경계

원점은 구간 종료 직후이며 관측시각≤원점이다. 미래 생산·실측기상·인원·같은 시간 평균전력은 예측에서 제외한다.
시간복원48행과 의존 특징·목표를 제외하고0값은 유지한다. CBL 참고값도 원점에 가용해야 한다.
전력 단위·구간경계는 미확인 가정이다. 원화절감·인과효과·계절일반화를 주장하지 않는다.
CBL은15분 준용이며 공식 DR정산의 완전 복제가 아니다. 요금단가 공란, 순서가중은 가정이다.
저녁 경보 뒤 지난 낮으로 이동하는 계산을 실행 가능한 절감으로 표시하지 않는다.
사전검증 테스트1회 열람을 공개하며 새 모델 선택은 개발구간만 사용한다.

개발ZIP은 최종 블라인드 제출물과 다르다. 동결 증거의 과거 경로를 보존하고 검사 결과를 별도로 제공한다.
설문은 본인 응답이 필요하며 완료 캡처를 대신 만들지 않는다. 10/5~6에 hwpx서식·PDF를 인계한다.
발표 PDF 초안과 보고서 제출 PDF를 구분한다. 발표PPTX·최종예측·보고서hwpx/PDF·포털완료증거는 아직 미완료다.
