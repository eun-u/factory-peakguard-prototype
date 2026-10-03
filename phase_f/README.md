# Phase F: development performance exploration

RQ: 과거 전력의 정보량, 목표 변환, 모델 구조와 결합이 1–4시간 예측 오차를 줄이면서 피크 오차를 보호하는가?

`outputs/phase_f/logs/preflight_review.md`의 수정안과 실제 사용자 승인은 `user_approval.json`에 보존한다. 기준 브랜치는 Phase3_experiments, 실험 브랜치는 Phase4_performance다. Phase C/E를 수정하거나 최종 holdout을 여는 실행 경로는 포함하지 않는다.

## 데이터와 선택

- 유일한 원자료 로더는 `phase_c.data.load_history`다. 2021-08-09 09:45 이후 행은 파싱하지 않는다.
- 기존 MAIN10 유한 공통 cohort, 100,010 score 키와 39 horizon/fold 셀을 고정했다. target 날짜 ISO 짝수 주는 EXPLORE, 홀수 주는 CONFIRM이다.
- CONFIRM은 과거에 개발 평가에 쓰였던 날짜의 잠금 재확인이다. 새 독립 검증이나 최종 테스트가 아니다.
- fit: 모델·정규화·tau; stop: 조기 종료; cal: 수치 가중치·보정; EXPLORE: 설정 선택. CONFIRM 결과로 재선정하지 않는다.
- 주요 후보 1개와 최대 4개 보조 후보를 EXPLORE에서 먼저 고정한다. B5 대비 MAE 개선 CI 하한 >0, 피크 악화 CI 상한 <=0, D2 보호와 2/3 fold 개선을 요구한다.
- 긴 문맥의 결측 처리와 weekly fallback을 설정에 미리 명시한다. 유리한 score 행만 제거하지 않는다.
- 합성 미래 누수 음성 대조는 진단 전용이다. 주간 재적합은 CONFIRM 후 별도 진단이다.

## 실행 상태

Stage 0의 기준선 평가·MSTL 수정·CBL/seasonal 기준선·합성 누수 검사를 실행했다. 저장된 부모 모델 234개 셀의 예측은 최대 차이 0으로 재현했다. B5/M1/M1-W/M2는 대표 h4/fold0를 새로 적합해 차이 0을 확인했다. 전체 39개 셀을 새로 적합했다는 뜻은 아니다.

후속 모델 어댑터와 합성 검사는 구현 중이며, 구현과 실제 실험 완료는 레지스트리에서 구분한다. 전체 Phase F 완료 여부는 `logs/driver_status.json`과 최종 보고서로 판단한다. 현재 `full_phase_complete=false`다. 구현되지 않은 Stage를 완료로 간주하지 않는다.

## 재현

원래 `.venv-phase-c`는 보존한다. 새 의존성은 `outputs/phase_f/env`, 호환성 격리가 필요한 모델은 `optional_envs`에 둔다. 예측·체크포인트·환경은 Git 제외다.

```powershell
outputs/phase_f/env/Scripts/python.exe -m phase_f.run --stage 0
outputs/phase_f/env/Scripts/python.exe -m phase_f.run --stage 1 --exp F6-1-c2048-median
outputs/phase_f/env/Scripts/python.exe -m pytest phase_f/tests -q -p no:cacheprovider
```

GPU 학습·추론과 LightGBM 적합은 동시에 실행하지 않는다. 각 설정은 실행 전에 레지스트리에 등록한다. 캐시의 설정·소스·split 식별자가 달라지면 기존 결과를 덮어쓰지 않고 새 실험 ID를 사용한다.

## 해석과 논문 위치

정보량 가설은 F1 특징 제거 비교 및 F5/F6 문맥 길이 곡선으로 판단한다. F0/F3–F9는 방법·실험 절, F10/F11은 오류·운영 진단 절에 대응한다. EXPLORE 최고값은 전체 시도 수와 함께 공개한다. 상용 적합성은 이 증강 개발자료만으로 증명할 수 없으며 단위, 현장 허용 오차, 비증강 미래 관측 검증이 필요하다.
