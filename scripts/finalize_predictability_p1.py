"""Apply the immutable Phase 1 family correction and export the Phase 2 gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.research_stats import holm_adjust


def finalize(root=ROOT, output=None):
    root = Path(root)
    out = Path(output) if output else root / "outputs/analysis_p1"
    rows = pd.concat([pd.read_csv(out / "A1_A3_hypotheses.csv"),
                      pd.read_csv(out / "A4_hypotheses.csv")], ignore_index=True)
    if len(rows) != 23 or rows.hypothesis_id.duplicated().any():
        raise ValueError("Phase 1 requires exactly 23 unique registered hypotheses")
    rows["p_holm"] = holm_adjust(rows.p_raw, family_size=23)
    eligible = rows.eligible.astype(str).str.lower().eq("true")
    rows["passed"] = (eligible & rows.ci_low.gt(rows.null_value)
                      & rows.p_holm.lt(.05) & rows.n_positive_folds.ge(2))
    # Two-sided explanatory coefficients are reported, never used as forecast features.
    coeff = rows.analysis_id.eq("A3")
    rows.loc[coeff, "passed"] = (eligible[coeff] & rows.loc[coeff, "p_holm"].lt(.05)
        & (rows.loc[coeff, "ci_low"].gt(0) | rows.loc[coeff, "ci_high"].lt(0)))
    rows["verdict"] = np.where(rows.passed, "통과", "미입증")
    contrary = eligible & ~rows.passed & rows.ci_high.lt(rows.null_value) & ~coeff
    rows.loc[contrary, "verdict"] = "기각"
    rows["selection_used"] = False
    rows.to_csv(out / "P1_hypotheses_holm.csv", index=False, encoding="utf-8-sig")

    a1 = rows.loc[rows.analysis_id.eq("A1")]
    multi = rows.loc[rows.analysis_id.eq("A2") & rows.test_kind.eq("auc_multi")]
    if len(a1) != 1 or len(multi) != 2:
        raise ValueError("A1/A2 primary gate rows are incomplete")
    a1_pass = bool(a1.passed.iloc[0])
    a2h16 = multi.loc[multi.horizon.eq(16)].iloc[0]
    negative_override = not a1_pass and not bool(a2h16.passed)
    allowed = {str(h): [] for h in (16, 96)}
    feature_rows = []
    for h in (16, 96):
        if a1_pass and not negative_override:
            for name in ("target_restart_slot", "minutes_to_next_restart"):
                allowed[str(h)].append(name)
                feature_rows.append({"horizon": h, "feature": name,
                    "evidence": a1.iloc[0].hypothesis_id,
                    "estimate": float(a1.iloc[0].estimate),
                    "ci95": [float(a1.iloc[0].ci_low), float(a1.iloc[0].ci_high)],
                    "availability": "past-only training restart schedule; deterministic target calendar",
                    "leakage_test": "fit schedule before origin; future perturbation invariance"})
        for _, row in rows.loc[rows.analysis_id.eq("A2") & rows.horizon.eq(h)
                               & rows.test_kind.eq("auc_single") & rows.passed].iterrows():
            # Existing lag/current features already remain in M1. Add only genuinely new signals.
            name = str(row.feature)
            if name not in ("slot_rate7", "slot_rate28", "previous_restart_rise") or negative_override:
                continue
            allowed[str(h)].append(name)
            feature_rows.append({"horizon": h, "feature": name, "evidence": row.hypothesis_id,
                "estimate": float(row.estimate), "ci95": [float(row.ci_low), float(row.ci_high)],
                "availability": "observations <= min(origin,fold fit_end) for target rates; completed past events",
                "leakage_test": "per-row training cutoff, provenance <= origin, future perturbation"})
    finite = lambda value: float(value) if np.isfinite(value) else None
    spec = {"a1_passed": a1_pass, "a2_h16_passed": bool(a2h16.passed),
            "a2_h16_auc": finite(a2h16.estimate),
            "a2_h16_auc_ci95": [finite(a2h16.ci_low), finite(a2h16.ci_high)],
            "negative_override": negative_override, "allowed_by_horizon": allowed,
            "features": feature_rows, "family_size": 23,
            "m2_allowed": any(allowed.values()), "test_opened": False}
    logs = root / "outputs/logs"
    (logs / "feature_spec_P2.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Phase 2 특징 게이트", "", "근거: preregistration_0925_P1.md 및 analysis_p1/P1_hypotheses_holm.csv.", "",
             f"- A1: {a1.iloc[0].verdict}; h16 다변수 A2: {a2h16.verdict}.",
             (f"- AUC: {a2h16.estimate:.6f}, 95% CI [{a2h16.ci_low:.6f}, {a2h16.ci_high:.6f}], Holm p={a2h16.p_holm:.6f}."
              if np.isfinite(a2h16.estimate) else "- AUC 및 CI 평가 불가: 사전 선정 슬롯의 양성/음성 표본 요건 미충족. 일반적인 피크 예측 불가능성의 증거가 아니다."),
             f"- M2 허용: {spec['m2_allowed']}. 양쪽 실패 강제 제외: {negative_override}.", "",
             "| 거리 | 허용 신규 특징 | 근거 | 효과 및 95% CI | 가용 시각 / 누수 점검 |",
             "| --- | --- | --- | --- | --- |"]
    for f in feature_rows:
        lines.append(f"| h{f['horizon']} | {f['feature']} | {f['evidence']} | {f['estimate']:.6f} {f['ci95']} | {f['availability']}; {f['leakage_test']} |")
    if not feature_rows:
        lines.append("| — | 없음 | P1 사전 게이트 미통과 | 미입증 | M1 기존 특징 및 M4 보정만 진행 |")
    lines += ["", "A3 실측 생산량/기온은 사후 설명 전용이다. A4 오류 집중은 원인 분석이며 단독으로 새 특징을 허가하지 않는다.",
              "기존 부하/시차 특징은 M1에 그대로 남는다. 다변수 유의성만으로 각 개별 특징을 승인하지 않는다. 모든 폴드 방향과 표본 적격 여부는 가설 표를 참조한다."]
    (logs / "feature_spec_P2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return spec


if __name__ == "__main__":
    print(json.dumps(finalize(), ensure_ascii=False))
