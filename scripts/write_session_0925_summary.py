"""Write the final handoff only after sealed P3 reproduction has passed."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/logs"


def load(name):
    return json.loads((LOGS / name).read_text(encoding="utf-8-sig"))


def read_process_log(name):
    raw = (LOGS / name).read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def main():
    repro = load("session_0925_repro.json")
    dry = load("freeze_dry_run.json")
    run = load("full_run_status.json")
    schedule = load("scheduled_freeze.json")
    integration = load("P2_research_integration.json")
    operations = load("P3_operations_repro.json")
    if (not repro["partial_refit_executed"] or repro["no_final"]["test_opened"]
            or dry["holdout_read"] or schedule["state"] != "Disabled"
            or run["status"] != "completed_development"
            or run["steps"]["development"]["detail"]["cache"] != "rebuilt"
            or dry["date_gate_open"] or dry["human_approval"] != "pending_or_invalid"
            or operations["status"] != "pass" or operations["holdout_read"]
            or not operations["source_unchanged"]):
        raise AssertionError("Final handoff requires fresh CV, partial reproduction and a sealed holdout")
    for name in ("P3_full_development_exit.txt", "P3_pytest_exit.txt", "P3_repro_exit.txt", "P3_operations_exit.txt"):
        if int(read_process_log(name).strip()) != 0:
            raise AssertionError(f"Direct process failed: {name}")
    test_log = read_process_log("P3_pytest.log")
    test_result = re.search(r"(\d+) passed in ([\d.]+)s", test_log)
    if test_result is None:
        raise AssertionError("Full pytest pass count is absent")
    tests, test_seconds = test_result.groups()
    metrics = pd.read_csv(ROOT / "outputs/analysis_p2/P2_auxiliary_metrics.csv")
    h16 = metrics.loc[metrics.horizon.eq(16)].set_index("model")
    before, after = h16.loc["c3_holiday_hybrid"], h16.loc["lgbm_residual_cbl"]
    selection_table = pd.read_csv(ROOT / "outputs/analysis_p2/P2_selection_before_after.csv")
    rows = ["| 예측거리 | 기존 점모델 | 최종 개발 점모델 | 위험 출력 | 사유 |",
            "| --- | --- | --- | --- | --- |"]
    reasons = {1: "등록 신규 후보 없음", 4: "M1 참고 전용", 16: "피크/F1 CI 겹침 후 위치 FP CI 감소",
               96: "잔차 마지막 폴드 MAE1.504배로 배제"}
    for row in selection_table.itertuples(index=False):
        rows.append(f"| h{row.horizon} | {row.point_before} | {row.point_after} | {row.risk_after} | {reasons[row.horizon]} |")
    ledger = load("forking_paths_0925.json")
    ledger["P2"]["status"] = "completed"
    ledger["P3"] = {
        "status": "completed_development_only", "preregistration_commit": "0cae04d",
        "full_cv_runs": 1, "original_horizon_fold_runs": 12, "original_t2_fold_runs": 3,
        "original_grid_evaluations": 24, "original_model_horizon_pairs": 84,
        "new_candidate_configurations": 0, "new_hypotheses": 0,
        "research_residual_model_fits": integration["m1_model_fit_count"],
        "research_quantile_model_fits": integration["m4_model_fit_count"],
        "partial_residual_model_fits": 1, "partial_quantile_model_fits": 5,
        "retrospective_ridge_fits": 6, "operational_scenarios": 16,
        "test_opened": False,
        "full_cv_seconds": run["steps"]["development"]["seconds"],
        "runtime_target_seconds": 1800,
        "runtime_target_met": run["steps"]["development"]["seconds"] <= 1800,
    }
    ledger["totals"] = {"registered_confirmatory_hypotheses": 50,
                         "new_model_horizon_candidates": 11,
                         "new_forecast_feature_sets": 1,
                         "research_related_lightgbm_fits_including_reproduction": 114,
                         "note": "P1 진단10특징집합은 별도 등록, 로지스틱 적격 적합0; 기존 모델 전체CV 재학습은 별도 P3 단계 수로 표시"}
    (LOGS / "forking_paths_0925.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    parity = repro["oof_parity"]
    original_diff = parity["original"]["max_abs_difference"]
    new_diff = max(row["max_abs_difference"] for row in parity["new_candidates"].values())
    development_seconds = run["steps"]["development"]["seconds"]
    runtime_note = ("30분 목표 초과이며 실행시간 요건은 미충족이다. 전체 재학습은 반복하지 않았다."
                    if development_seconds > 1800 else "30분 실행시간 목표를 충족했다.")
    body = f"""# 0925 피크 예측가능성 → 모델 개선 → 동결 준비 인계

작업 브랜치: `05_experiments`. 2026-09-24 실행. 최신 마스터 지시 범위의 P1~P3 완료이며, **최종 테스트·사람 동결 승인은 미완료**다. 목표값은 2021-08-09 09:45 이전만 사용했다. 보고서/슬라이드/제출물/로드맵은 이번 세션에서 갱신하지 않았다.

## Phase와 사전 판정

| 단계 | 상태 | 사전 판정 결과 |
| --- | --- | --- |
| P1 A1 | 완료·미입증 | 재가동 근접 차이+0.685%p CI[-1.236,2.978], Holm p=1 |
| P1 A2/A3 | 완료·평가 불가 | 고정 슬롯 코호트 피크0개, h16/h96 AUC 및 크기 회귀 산출 불가 |
| P1 A4 | 완료·미입증 | 오류 집중 효과 h16 -11.301[-32.663,8.245], h96 -16.267[-37.163,2.319] |
| P1 A5 | 완료·사후 가정 | h16 준비30/60분 D20% 조치172/183, 완벽정보 참고178/183 |
| P2 M1 | 완료·h16만 채택 | h16 피크 MAE 개선 미입증, 후순위 FP CI로 선정; h4 참고, h96 안정성 탈락 |
| P2 M2/M3 | 게이트로 미실행 | 허용 신규 재가동 특징0개; 절제/위약도 해당 없음 |
| P2 M4 | 완료·8후보 기각 | 모두 마지막 폴드 커버리지 조건 미달, B 유지 |
| P3 | 검증 완료·시간 목표 미충족 | 전체 개발 CV, 원/신 후보 재현, 캐시 재봉인, 데이터 로딩 없는 dry-run 통과; 전체 실행31분21초 |

## 핵심 발견 (5줄)

1. 사전 정의한 재가동 가설은 채택하지 않았다. 실제 재가동 전체가 없다는 뜻은 아니며 휴식시간표가 미확인이다.
2. 4시간 전 AUC는 계산 불가다. 선택된 슬롯의 피크 표본0개가 원인이며 일반적 예측 불가능성을 입증한 결과가 아니다.
3. h16 조치 가능 피크 출발창은172/183=93.99%, 완벽정보 참고는178/183=97.27%다. 고정 발령 규칙의 참고이며 전역 최적 조치 상한은 아니다.
4. h16 잔차는 위치 FP {int(before.position_fp)}→{int(after.position_fp)}로 채택됐지만 피크 MAE {before.peak_mae:.3f}→{after.peak_mae:.3f}, 개선량 CI[-12.073,0.407]로 개선 미입증이다.
5. q95는 기존 B를 유지하므로 P2 이후 h16 운영 조치율은 그대로다. h96 rolling의 전체 폴드를 합친 상위 구간 커버리지 개선도 Holm 후 미입증이고 마지막 폴드가 악화됐다.

## 선정 전후

{chr(10).join(rows)}

h16은 첫 폴드 피크 MAE23.760→68.570으로 크게 악화했다. 마지막 두 폴드 개선과 마지막 폴드10% 악화 방지 조건 통과 뒤, 기존 순서에서 피크/F1 CI가 겹쳐 FP CI로 선정됐다. 평균 피크 오차 개선이나 모든 폴드 안정성을 주장하지 않는다. h16 에피소드 F1은{before.episode_f1:.3f}→{after.episode_f1:.3f}로 CI가 겹친다.

후보별 원인·CI·폴드 방향은 `outputs/analysis_p2/P2_candidate_decisions.csv`, `M1_fold_comparison.csv`, `M4_hypotheses.csv`, `P2_hypotheses_holm.csv`에 있다. 전체27검정의 Holm 통과8개는 pinball5% 허용폭이며 커버리지 개선8개가 아니다. 대칭 MAE·편향·과대예측률은 `P2_auxiliary_metrics.csv`의 보조 지표로만 사용했다.

## 재현·동결 준비 증거

- 전체 개발 단계 {development_seconds:.3f}초, 직접 exit0. 원후보 {parity['original']['rows']:,}행과 신규80,816행, 통합 {parity['integrated_rows']:,}행을 확인했다.
- {runtime_note}
- 원후보 최대 절대차 {original_diff:.3g}, 신규 후보 최대 절대차 {new_diff:.3g}; 허용오차1e-10. h1/h4/h96 선정과 원 선정 기록 보존, h16 변경 및 채택 사유도 P2/P3 일치.
- h16 fold2 잔차1개·h96 fold2 분위수5개를 캐시 없이 다시 적합하고4개 온라인 보정을 재생했다. 세부 결과는 `session_0925_repro.json`.
- P1 핵심 수치·표 재계산 최대차0, 원본 입력/P1 산출물40파일 불변. 보호75파일 및3개 사전등록 문서는 보존됐다.
- 통합 OOF로 운영16시나리오·보조지표·Holm 표를 별도 폴더에 재계산하여 기존 P2의 {len(operations['tables'])}개 표와1e-10 이내 일치했다. `P3_operations_repro.json` 참조.
- 전체 pytest {tests}개 통과({test_seconds}초), 누수 테스트 포함. 실제 final 경로를 실행하지 않았고 final_test.csv·동결 예약 기록·사람 승인 파일은 없다.
- dry-run `{dry['status']}`, holdout_read=false, 날짜 게이트 닫힘·사람 승인 미완료. Windows 예약 `{schedule['state']}` 유지. 캐시는 현재 코드/기본설정/원자료 불투명 해시/선정/사전등록/특징 게이트/모델 번들과 연결됐다.

## forking paths

P1 확증23가설(추정 가능3, 평가 불가20), 진단 특징 집합10개, 등록 로지스틱16구성 중 적격 학습0개, 재가동 정의5개(민감도는 기술 분석). P2 확증27가설과 고유 후보11구성(h4 1, h16 5, h96 5), 신규 예측 특징 집합1개다. 재현은 후보 추가로 세지 않았다. M1 최초9fits·M4 동일 설정2회60fits, P3 연구 후보39fits·부분 재현6fits로 연구 관련 LightGBM114fits이며 기존 전체CV12폴드/T2 3폴드는 별도 기록했다. `forking_paths_0925.json` 참조.

## 사람이 할 일 — 모두 미완료

- 실제 휴식/점심/재가동 시간표 확인, 한국 공휴일 공식 확인.
- 10-01 동결 승인 및 승인 해시에 맞춘 예약 수동 재활성화. 자동으로 켜지 않는다.
- 산업용(을)의 해당 선택요금·공식 단가 입력. 단위·15분 경계 및 주최측 문의 답변 확인.
- 설문 본인 응답·캡처, hwpx/PDF 서식, PPTX와 최종 제출 작업은 별도 인계 사항이다.

## 한계와 확인하지 못한 것

재가동 정의의 희소성으로 AUC/크기 요인 규명이 막혔다. 사후 코호트 확대나 임계값 변경으로 이를 구제하지 않았다. 개발 CV를 여러 연구에 사용했으며 신규 독립 일반화 검증은 아니다. h16 선택에는 초기 폴드 악화와 주 지표 개선 미입증이 남고 q95 운영 개선도 없다. 부하 이동은 사후 관측 생산량·Ridge 계수·순서형 요금 가중치를 사용한 부분 월 시나리오여서 실제 금액·설비 실행가능성·인과효과를 확인하지 못했다. 단가·단위·계절 일반화·휴식시간표는 미확인이다. 기존 README/보고서/로드맵에는 이전 체크포인트 설명이 남아 있으므로 현재 실행·동결 상태는 이 인계와 검증 JSON을 따른다.

## 재실행 명령

기존 가상환경에서 `python run_all.py`는 안전한 개발 단계만 실행/캐시 확인한다. 캐시 없는 전체 재현은 `python run_all.py --development-only --rebuild-dev`, 동결 준비 확인은 `python run_all.py --dry-run-freeze`다. 전체 모델 재학습을 불필요하게 반복하지 않는다.

검증은 `python scripts/verify_predictability_p3.py`, 이어 `python scripts/verify_session_0925.py --refit-partial`이다. 검증용 과거 스냅샷은 `_validation/session_0925_original/`, `_validation/session_0925_P2/`를 보존해야 한다. 이는 로컬 보존 증거와 원자료가 필요한 재현이며 Git만으로 데이터가 제공되는 것은 아니다. 대용량 원자료·OOF·모델은 기존 ignore 규칙을 따랐다.
"""
    (LOGS / "session_0925_summary.md").write_text(body, encoding="utf-8")
    print("Wrote session_0925_summary.md and forking_paths_0925.json from verified artifacts")


if __name__ == "__main__":
    main()
