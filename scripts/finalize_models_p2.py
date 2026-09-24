"""Combine the fixed P2 hypothesis family and operating evidence, without fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.research_stats import holm_adjust
from src.session_data import load_development_history, read_development_oof
from src.analysis.predictability_operations import run_a5
from src.analysis.symmetric_metrics import symmetric_peak_table


def hypothesis_family(output: Path) -> pd.DataFrame:
    m1 = pd.read_csv(output / "analysis_p2/M1_comparisons.csv")
    m4 = pd.read_csv(output / "analysis_p2/M4_hypotheses.csv")
    if len(m1) != 3 or len(m4) != 24:
        raise AssertionError("P2 must retain all 27 registered hypotheses")
    m1["model"] = m1.candidate_model
    m1["effect"] = "peak_mae_improvement"
    m4["hypothesis_id"] = [f"M4_h{h}_{m}_{e}" for h, m, e in
                            zip(m4.horizon, m4.model, m4.effect)]
    columns = ["hypothesis_id", "horizon", "model", "effect", "estimate",
               "ci_low", "ci_high", "p_raw", "n_positive_folds", "bootstrap_valid_draws"]
    family = pd.concat([m1[columns], m4[columns]], ignore_index=True)
    if family.hypothesis_id.duplicated().any():
        raise AssertionError("Duplicate P2 hypothesis identity")
    family["p_holm"] = holm_adjust(family.p_raw.to_numpy(), family_size=27)
    positive_ci = np.isfinite(family.ci_low) & (family.ci_low > 0)
    family["verdict"] = np.where(positive_ci & family.p_holm.lt(.05), "통과",
                                  np.where(family.ci_high.lt(0), "기각", "미입증"))
    family["selection_p_gate"] = False
    family["note"] = "Holm은 증거 강도 보고용; 기존 선정 및 사전 채택 문턱을 변경하지 않음"
    family.to_csv(output / "analysis_p2/P2_hypotheses_holm.csv", index=False, encoding="utf-8-sig")
    return family


def operating_evaluation(output: Path, predictions: pd.DataFrame, manifest: dict) -> dict:
    history = load_development_history(ROOT)
    tariff = yaml.safe_load((ROOT / "configs/tariff_2026.yaml").read_text(encoding="utf-8"))
    target = output / "analysis_p2/operations"
    target.mkdir(parents=True, exist_ok=True)
    results = run_a5(history, predictions, manifest, tariff, target)
    chosen = manifest["selection"]["by_horizon"]
    for key, frame in results.items():
        frame["evaluation_phase"] = "P2_fixed_selected_combination"
        if "horizon" in frame:
            frame["selected_point_model"] = frame.horizon.map(lambda h: chosen[str(int(h))]["point_model"])
            frame["selected_risk_model"] = frame.horizon.map(lambda h: chosen[str(int(h))].get(
                "risk_model", "lgbm_quantile_" + chosen[str(int(h))]["conformal"]))
        frame.to_csv(target / f"A5_{key}.csv", index=False, encoding="utf-8-sig")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    family = hypothesis_family(args.output_dir)
    if bool(args.manifest) != bool(args.predictions):
        parser.error("Supply both manifest and predictions for operating evaluation")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        predictions = read_development_oof(args.predictions)
        cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
        choice = manifest.get("original_selection", manifest["selection"])["by_horizon"]
        mask = predictions.model.str.startswith(("q95_", "lgbm_residual"))
        for h, selected in choice.items():
            names = [selected["point_model"], selected["cbl"], selected["persistence"],
                     f"lgbm_quantile_{selected['conformal']}"]
            mask |= predictions.horizon.eq(int(h)) & predictions.model.isin(names)
        metrics = symmetric_peak_table(predictions.loc[mask], cfg)
        metrics["reference_only"] |= metrics.model.str.startswith("q95_") | (
            metrics.model.eq("lgbm_residual_cbl") & metrics.horizon.eq(4))
        metrics.to_csv(args.output_dir / "analysis_p2/P2_auxiliary_metrics.csv",
                       index=False, encoding="utf-8-sig")
        operating_evaluation(args.output_dir, predictions, manifest)
    print(json.dumps({"hypotheses": len(family), "holm_supported": int(family.verdict.eq("통과").sum()),
                      "holdout_read": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
