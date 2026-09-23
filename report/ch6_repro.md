# 6 코드 및 재현성

이 초안의 새 모델 수치는 **개발 교차검증** 결과다. 2026-10-01 18:00 KST 전에는 최종 테스트를 실행하지 않는다. 사전 검증에서 고정 설정으로 테스트를 한 번 본 이력이 있으며 verification/의 성능은 신규 모델의 성능으로 재사용하지 않는다.

README의 설치 후 `python run_all.py`로 데이터, 개발모델, 분석, 그림·표, 보고서, 제출 준비ZIP을 생성한다. `--only data|development|final|analysis|report|package`, `--from STEP`, `--rebuild-dev`를 지원한다. 원본과 verification 해시를 실행 전후 확인하고 코드·설정·기준·원본 해시가 다르면 개발 캐시를 거부한다. 최종 테스트 전에는 날짜 잠금을 확인한다.

실제 실행시간·오류·캐시 상태: [run_status.json](../outputs/logs/run_status.json). 자동 테스트: `python -m pytest tests -q`. 신규 가상환경 설치·전체 실행 결과와 독립 검토는 PROGRESS.md에 기록한다.

## 제출 경계

현 단계 ZIP은 개발 재현용 준비물이다. 보고서 hwpx·PDF, 설문 캡처, 발표 PDF·PPT, 최종 테스트 파일과 포털 제출 완료 화면이 모두 확보되기 전에는 최종 제출 완료라고 부르지 않는다.


## 캐시 없는 독립 재현 실행

재현 ZIP을 별도 폴더에 풀고 모델·개발 캐시 없이 실행했다. 예측·지표·선정의 원본 실행과의 일치 여부는 [재현 검증](../outputs/logs/fresh_reproduction.json)에 기록한다.

| step | seconds | cache |
| --- | --- | --- |
| data | 0.1680 | cold_run |
| development | 1430.2660 | cold_run |
| final | 0.0000 | cold_run |
| analysis | 146.4050 | cold_run |
| report | 32.7780 | cold_run |
| package | 1.9830 | cold_run |

이 시간표는 독립 전체 실행 기록이다. 이후 캐시 사용 실행과 예약 최종 평가의 최신 상태는 run_status.json을 따른다.
