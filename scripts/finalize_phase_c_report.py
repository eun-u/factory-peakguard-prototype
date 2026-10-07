"""Finalize the Phase C report only after strict result validation succeeds."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase_c"
sys.path.insert(0, str(ROOT))


def read(name: str) -> dict:
    return json.loads((OUT / "logs" / name).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    import pandas as pd
    from phase_c.report import write_report
    from phase_c.run import assert_seal, check_protected

    completion_path = OUT / "logs/phase_c_completion_manifest.json"
    if completion_path.exists():
        raise RuntimeError("Completed evidence already exists; preserve it rather than silently rewrite it")
    assert_seal()
    protected = check_protected()
    audit = read("artifact_integrity_audit.json")
    metric_audit = read("metric_recomputation_audit.json")
    cbl_audit = read("cbl_structural_missingness_audit.json")
    leakage = read("leakage_audit.json")
    runtime = read("runtime.json")
    selection = read("model_selection.json")
    if audit["status"] != "passed" or metric_audit["status"] != "passed" or not leakage["all_applicable_checks_pass"]:
        raise RuntimeError("Strict validation must pass before reporting completion")
    if runtime["stages"][-1]["status"] != "completed":
        raise RuntimeError("The experiment is still running or failed")
    tests = (OUT / "logs/phase_c_pytest.txt").read_text(encoding="utf-8")
    verifier_tests = (OUT / "logs/artifact_verifier_pytest.txt").read_text(encoding="utf-8")
    verified_test_counts = re.findall(r"(\d+) passed", verifier_tests)
    if "27 passed" not in tests or not verified_test_counts:
        raise RuntimeError("Expected synthetic test evidence is missing")
    verifier_count = int(verified_test_counts[-1])
    dedicated_count = 20 + verifier_count
    structural_missing = int(cbl_audit["nan_prediction_rows"])
    original_keys = metric_audit["raw_main_rows"] // 10
    if original_keys - metric_audit["paired_keys_per_model"] != structural_missing:
        raise RuntimeError("CBL diagnosis and paired cohort counts differ")
    verified_at = datetime.now(timezone.utc)
    elapsed_since_first_job = (verified_at - datetime.fromisoformat(runtime["first_job_started_at"])).total_seconds()
    runtime["completion_validation"] = {
        "verified_at": verified_at.isoformat(),
        "elapsed_wall_seconds_since_first_job": elapsed_since_first_job,
        "definition": "Elapsed time from first model download job through final validation, including between-stage preparation, audit diagnosis and documentation; separate from actual_job_seconds",
        "artifact_integrity": audit["status"], "metric_recomputation": metric_audit["status"],
    }
    (OUT / "logs/runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
    path = write_report(OUT)
    original_report = path.read_text(encoding="utf-8")
    path.write_text(original_report.replace("실행 중 예외 로그: 없음.",
        "모델 학습·추론 runner의 예외 로그: 없음. 추가 독립 검증기의 두 차례 실패와 수정은 아래 최종 검증 항목 및 independent_review.md에 별도로 기록했다.",1),encoding="utf-8")
    auc = pd.read_csv(OUT / "tables/model_auc_summary.csv").set_index(["dataset","model"])
    ci = pd.read_csv(OUT / "tables/model_pairwise_ci.csv")
    stability = pd.read_csv(OUT / "tables/model_stability.csv").set_index("model")
    decision = pd.read_csv(OUT / "tables/model_selection_decision.csv").set_index("model")
    def difference(dataset, metric, a, b):
        forward = ci.loc[ci.dataset.eq(dataset)&ci.metric.eq(metric)&ci.model_a.eq(a)&ci.model_b.eq(b)]
        if not forward.empty:
            row = forward.iloc[0]
            return row.difference_a_minus_b, row.ci_low, row.ci_high
        reverse = ci.loc[ci.dataset.eq(dataset)&ci.metric.eq(metric)&ci.model_a.eq(b)&ci.model_b.eq(a)].iloc[0]
        return -reverse.difference_a_minus_b, -reverse.ci_high, -reverse.ci_low
    chosen = selection["selected_M_star"]
    base = selection["strongest_baseline"]
    primary = selection["primary_min_auc_model"]
    if chosen and primary and chosen != primary:
        diff, low, high = difference("D1","AUC_MAE",chosen,primary)
        gain = 100 * (1 - auc.loc[("D1",chosen),"AUC_MAE"] / auc.loc[("D1",base),"AUC_MAE"])
        peak_diff, peak_low, peak_high = difference("D1","AUC_PeakMAE","M2",base)
        explanation = f"""## 핵심 판정 해설

{chosen} 선택은 독보적인 정확도 우위가 아니라 사전등록된 complexity tie-break의 결과다. 적격 후보 중 최소 AUC-MAE는 {primary}이고, {chosen}−{primary} 차이는 {diff:.4f}, 95% CI [{low:.4f}, {high:.4f}]로 0을 포함한다. 동률 집합은 {selection['statistical_tie_models']}이다. {chosen}의 AUC-MAE는 strongest baseline {base}보다 {gain:.2f}% 낮지만, 모든 fold에서 개선된 것은 아니다({int(stability.loc[chosen,'positive_fold_count'])}/3 fold 개선).

TCN(M2)은 MAIN10의 전체 MAE가 가장 낮지만 peak safeguard에서 탈락했다. M2−{base}의 AUC-PeakMAE 차이는 {peak_diff:.4f}, 95% CI [{peak_low:.4f}, {peak_high:.4f}]다. Residual(C1)의 기각 사유는 `{decision.loc['C1','C1_rejection_reasons']}`이며, 기대와 달라도 구제 튜닝을 하지 않았다. R1은 전체 MAE가 낮아도 reference-only이므로 선택 대상이 아니다.

D2에서 {chosen}의 AUC-MAE는 {auc.loc[('D2',chosen),'AUC_MAE']:.4f}, {primary}는 {auc.loc[('D2',primary),'AUC_MAE']:.4f}, {base}는 {auc.loc[('D2',base),'AUC_MAE']:.4f}다. {chosen}의 D1/D2 순위는 각각 {int(stability.loc[chosen,'d1_rank'])}/{int(stability.loc[chosen,'d2_rank'])}위다. baseline 대비 개선 방향은 유지되지만 유일한 MAE 1위라는 뜻은 아니다.

"""
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("## 실험 목적과 고정 설계",explanation+"## 실험 목적과 고정 설계",1),encoding="utf-8")
    reference = audit["R1_reference_coverage"]
    appendix = f"""

## 최종 독립 검증과 보존

첫 실행 job부터 최종 검증까지의 전체 경과시간은 {elapsed_since_first_job:.1f}초다. 이 시간에는 단계 사이 준비·독립 감사의 진단 및 문서 정리가 포함된다. 앞의 실제 model job 합계 {runtime['actual_job_seconds']:.1f}초와 구분하며, 둘을 더하지 않는다.

Phase C 전용 검사는 고유 테스트 기준 {dedicated_count}개가 통과했다. 기본 20개와 최초 검증기 7개를 묶은 27개 실행, 이후 보강된 검증기 {verifier_count}개 실행을 구분해 로그에 보존했다. 같은 테스트를 중복 합산하지 않았다. 별도 기존 synthetic 회귀 검사에서는 15개가 통과했고 1개는 legacy seasonal 모듈의 미설치 numba 의존성 때문에 실패했다. 해당 테스트는 실제 holdout을 읽지 않았으며, 통과했다고 주장하지 않는다. 실행 중 패키지를 변경하지 않았다.

별도 결과 검증 상태: **{audit['status']}**. MAIN10 원래 예측 {audit['checks']['main_prediction_rows']:,}행을 score context의 origin, target_time, y, tau, D2와 대조했다. 입력행 누락·중복·추가행을 거부하고 39개 score chunk와 최종 parquet의 값이 일치함을 확인했다. 현재 configuration lock이 score 파일보다 먼저 생성된 순서도 확인했다. 저장 모델의 실제 설정과 봉인 코드·입력 해시를 검증했으며, {audit['artifact_files_hashed']}개 cache 파일의 해시를 output_cache_manifest.json에 기록했다.

예외는 원래 CBL 계산 규칙상 불가능한 B2 예측 {structural_missing:,}개(horizon별 행 수 합계)다. 선택된 참조일의 3시간 보정 창이 7월 13·15일의 보정 제외 관측을 포함해 결측이 됐다. 원래 CBL 구현과 봉인된 history로 모든 값과 결측 위치를 재현하고, 그 외 비유한 예측은 거부했다. 사전등록된 공통 유효 시점 정책에 따라 모든 모델에서 동일한 {structural_missing:,}개 key를 제외하여 모델별 {original_keys:,}개 중 {metric_audit['paired_keys_per_model']:,}개로 비교했다. score에서 CBL을 바꾸거나 결측 예측을 임의 보충하지 않았다. 최초 추가 검증기의 절대적 유한성 가정이 사전등록보다 엄격했던 점을 수정했으며, 실패 기록·원인 진단은 별도 보존했다. 학습·예측·평가 코드 및 결과는 이 수정으로 바뀌지 않았다.

추가로 평가 구현을 호출하지 않고 저장된 예측에서 MAE·Peak-MAE·RMSE를 재계산했다. fold별 {metric_audit['model_horizon_fold_metrics_rows_checked']}개 행, pooled {metric_audit['model_horizon_pooled_metrics_rows_checked']}개 행, AUC {metric_audit['auc_rows_checked']}개 행이 결과표와 일치하고 39개 fit-label Q95도 일치한다. 이 재계산은 point estimate 검사이며, CI는 synthetic 검사와 독립 소스 검토로 검증했다.

모델 파일 자체로 복원할 수 없는 M1-W 학습 가중치 및 C1 residual target 생성은 봉인된 코드, synthetic 경로 검사, 예측 설정 metadata를 근거로 한다. 저장 booster만으로 학습 이력을 완전히 증명했다고 주장하지 않는다. 파일 생성 시각 역시 사전학습 부재의 암호학적 증명은 아니다. 모델 파일과 parquet는 후속 검증 시 기존 manifest와 비교하고, 불일치하면 manifest를 덮어쓰지 않고 실패한다.

독립 검증기는 모든 model/horizon/fold의 D1/D2 기대·실제 건수를 0건도 포함해 기록하고 R1 coverage를 별도로 남긴다. R1 상태는 아래와 같다. 자세한 셀별 수치는 artifact_integrity_audit.json을 따른다.

```json
{json.dumps({k:v for k,v in reference.items() if not isinstance(v,list)},ensure_ascii=False,indent=2)}
```

stability 표의 d2_rank_reversal은 strongest baseline 대비 오차 차이의 부호 변화다. 실제 순위 변화는 d1_rank/d2_rank 열로 판단한다. 기존 보호 파일 {protected['protected_baseline_files_including_ignored']}개의 바이트 해시가 유지됐고 기존 final artifact 수정은 없다. 원시 holdout 미파싱과 과거 final 요약 파일의 열람 사고는 앞서 밝힌 대로 별개의 사실이다.

RQ는 잠긴 정보집합에서 1~4시간을 대표할 model family 선택이며, 비교·지표·결과 근거는 이 보고서와 7개 결과표에 한정한다. 논문에서는 development model-selection 결과로만 사용할 수 있다. Phase D에는 저장된 M* 곡선을 넘기며, h*·calibration·현장 운영 효과는 이번 판단에 포함하지 않는다. 기존 프로젝트 문서 대신 이 독립 namespace에 진행 및 결정을 기록했다.

전체 모델을 포함한 원래 4개 그림을 보존했다. 아래는 큰 MSTL 오차에 다른 곡선이 눌리지 않도록 선택 모델·주요 비교군·참조모델만 확대한 보조 그림이다. 모델 선정 계산이나 비교 표본은 변경하지 않았다.

![Phase C selection detail](<{(OUT / 'figures/selection_detail.png').as_posix()}>)
"""
    path.write_text(path.read_text(encoding="utf-8") + appendix, encoding="utf-8")
    required = [
        "logs/preregistration_phase_c.md", "logs/model_config_lock.json",
        "logs/run_environment.json", "logs/leakage_audit.json", "logs/runtime.json",
        *[f"tables/{name}.csv" for name in (
            "model_horizon_fold_metrics", "model_horizon_pooled_metrics", "model_pairwise_ci",
            "d2_novel_profile_metrics", "model_auc_summary", "model_stability", "model_selection_decision")],
        *[f"figures/{name}.png" for name in (
            "mae_horizon_curve", "peak_mae_horizon_curve", "skill_horizon_curve", "d1_d2_comparison")],
        "phase_c_result.md", "logs/model_selection.json", "logs/output_cache_manifest.json",
        "logs/artifact_integrity_audit.json", "logs/phase_c_pytest.txt", "logs/artifact_verifier_pytest.txt",
        "logs/legacy_boundary_pytest.txt", "logs/independent_review.md",
        "logs/metric_recomputation_audit.json",
        "logs/cbl_structural_missingness_audit.json", "logs/artifact_integrity_attempt1_failed.json",
        "logs/artifact_integrity_attempt2_failed.json",
        "figures/selection_detail.png",
    ]
    for name in required:
        if not (OUT / name).is_file():
            raise RuntimeError(f"Missing required evidence: {name}")
    completion = {
        "development_only": True, "completed_at": datetime.now(timezone.utc).isoformat(),
        "branch": "Phase3_experiments", "committed": False, "pushed": False, "merged": False,
        "selected_M_star": selection["selected_M_star"], "h_star_selected": False,
        "selection_status": selection["selection_status"],
        "raw_holdout_observations_parsed": False, "historical_final_artifact_read": True,
        "dedicated_phase_c_tests_passed": dedicated_count,
        "additional_legacy_tests": {"passed": 15, "failed_missing_optional_dependency": 1},
        "protected_files": protected,
        "artifacts_sha256": {name: digest(OUT / name) for name in required},
        "finalizer_sha256": digest(Path(__file__)),
    }
    completion_path.write_text(json.dumps(completion,ensure_ascii=False,indent=2),encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
