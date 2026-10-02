# Phase E — Peak Risk, Uncertainty & Alert

이 폴더는 `Phase3_experiments` 브랜치에서 수행하는 **development-only** B5 h16 평가를 담는다. 고정 조건은 B5 Weekly-deviation Kalman, 16개 15분 간격(240분), fold별 fit power Q0.95 통계적 고부하 사건 경계다. 현재 checkout에 별도 Phase D 디렉터리는 없었고, h16은 이번 사용자 지시로 고정됐다. 이 실험에서 Phase C/D 모델·horizon을 다시 고르지 않는다.

연구 질문은 독립적이다. E1은 native Gaussian exceedance probability와 cal-only Platt 보정의 Brier·reliability, E2는 raw q95와 standardized one-sided split conformal U95의 경험적 coverage와 폭, E3은 여섯 개 고정 C/L 비율의 정규화 결정가치, E4는 1/1 Watch와 2/2 Confirmed Alert의 episode 재현율·오경보 부담·직접 lead time을 평가한다. 하나의 합성 PASS 점수나 최적 임계값은 만들지 않는다.

`logs/phase_e_protocol_lock.md`와 `logs/phase_e_config_lock.json`은 결과를 보기 전에 봉인했다. `logs/source_parity_audit.json`에는 Phase C B5 h16 원점예측과 재생성값의 최대 절대차이, 전체 유효 B5 score 행과 Phase C MAIN10 공통 코호트의 지표를 구분해 기록한다. 전체 B5 행의 수치가 공통 코호트 published 값과 다르더라도 같은 모집단처럼 취급하지 않는다.

`code/run_phase_e.py`는 단계별 실행기로, 각 단계가 이전 단계의 증거와 봉인을 검사한다. 이 실행의 결과는 이미 보존되어 있으므로 명령을 다시 실행하지 말고 산출물과 해시를 확인한다. 독립 검토 후 `logs/analysis_code_seal.json`에 고정한 구현·의존 소스 67개와 기존 C/D 파일은 평가 직전 다시 검증했다. 원래 parity seal 이후 추가된 입력 QA와 실행 gate 수정은 `logs/implementation_amendment_before_evaluation.md`에 공개했다.

일반 재실행은 기존 주요 파일을 덮어쓰지 않는다. 예외적으로 독립 검토가 확인한 F1의 미정의값 계산 오류를 단 한 번 수정했다. `code/correct_f1.py`는 네 원본 CSV를 `logs/metric_correction_v1/`에 바이트 그대로 보존한 뒤, 저장된 TP/FP/FN에서 `2TP/(2TP+FP+FN)`이 정의되는 빈 F1만 0으로 바꿨다. 나머지 모든 셀·순서·스키마와 예측 해시는 불변이다. 실제 보고서는 수정된 CSV를 읽는다. 적합이나 평가·정책을 다시 선택하지 않았다.

실행 이력의 명령은 아래와 같다. 새 과학적 실행에는 별도의 출력 복사본과 사전 잠금이 필요하다.

```powershell
& .\.venv-phase-c\Scripts\python.exe -B outputs/phase_e/code/run_phase_e.py --stage parity
& .\.venv-phase-c\Scripts\python.exe -B outputs/phase_e/code/run_phase_e.py --stage evaluate
& .\.venv-phase-c\Scripts\python.exe -B outputs/phase_e/code/correct_f1.py
& .\.venv-phase-c\Scripts\python.exe -B outputs/phase_e/code/verify_artifacts.py
& .\.venv-phase-c\Scripts\python.exe -B outputs/phase_e/code/run_phase_e.py --stage report
& .\.venv-phase-c\Scripts\python.exe -B outputs/phase_e/code/run_phase_e.py --stage finalize
```

`predictions/phase_e_cal_inputs.parquet`와 `phase_e_score_inputs.parquet`는 parity를 통과한 재생성 입력이다. `predictions/phase_e_h16_oof.parquet`는 cal 전용 Platt/conformal 적합을 score에 적용한 예측이다. `tables/risk_metrics.csv`, `uncertainty_metrics.csv`, `decision_value_curve.csv`, `alert_episode_metrics.csv` 등이 raw/cal/climatology Brier, BSS, PR-AUC, coverage/width, 여섯 C/L 비율, 두 alert 정책을 D1과 D2로 나누어 보존한다. `tables/phase_e_summary.csv`는 E1–E4 각 layer와 D1/D2별 빠른 색인이다. 원 수치와 모든 fold·threshold·policy 행은 해당 원본 표에 있다.

`phase_e_result.md`와 `figures/`의 여덟 이미지는 표에서 기계적으로 생성한다. 대표 TP/FN/FP 사례는 고정된 c=0.10, 2/2 정책에서 각 상태의 시간순 첫 사례를 표시한다. 결측 사례는 그림에 표시하며, 사례를 보고 임계값이나 정책을 고르지 않는다. 오경보 episodes/day의 분모는 **평가 관측일**(표현된 target-calendar day)이다. 실제 공장 운영일로 검증되지 않았다.

실행 접근 기록은 `logs/access_*.json`에 단계별로 남고, `logs/leakage_audit.json` 및 `logs/preservation_audit.json`은 finalize 단계에서 생성된다. 원본 raw file의 전체 SHA는 파일 식별에만 사용한다. 숫자 파싱·적합·평가에는 개발경계 `2021-08-09 09:45` 이후 관측값이나 historical final/test 결과를 사용하지 않는다. Phase C에서 있었던 historical artifact read incident를 이번 Phase E 실행의 `false` 기록이 소급해 지우지 않는다.

τ는 계약전력·설비 정격·안전 한계가 아니다. C/L은 실제 공장 조치비용이나 화폐 절감액이 아니다. U95는 확률 임계값 경보에 결합하지 않는 설명용 upper uncertainty다. Phase E 결과가 좋아도 현장 피크 예방·요금 절감·생산손실 감소를 주장하지 않는다. Final holdout 평가는 전체 pipeline의 code·parameter·policy·보고 기준 및 보존 감사가 동결된 후 별도의 승인 단계가 필요하다.

이 작업은 Git commit, push, merge를 수행하지 않는다.
