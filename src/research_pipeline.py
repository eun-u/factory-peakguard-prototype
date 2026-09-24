"""Phase 2 development-only candidate generation and selection integration."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate_all
from .research_selection import M1_NAME, M4_NAMES, extend_selection
from .session_data import SEALED_BOUNDARY


def _safe_history(history: pd.DataFrame) -> pd.DataFrame:
    if "ts_end" in history:
        history = history.set_index("ts_end")
    if (not isinstance(history.index, pd.DatetimeIndex) or history.empty
            or history.index.has_duplicates or not history.index.is_monotonic_increasing
            or history.index.max() >= SEALED_BOUNDARY):
        raise ValueError("Research CV requires sealed development-only history")
    return history


def _validate_gate(root: Path, feature_spec: dict | None) -> dict:
    if feature_spec is None:
        path = root / "outputs/logs/feature_spec_P2.json"
        feature_spec = json.loads(path.read_text(encoding="utf-8"))
    if (feature_spec.get("m2_allowed") is not False
            or feature_spec.get("features") != []
            or feature_spec.get("allowed_by_horizon", {}).get("16") != []
            or feature_spec.get("allowed_by_horizon", {}).get("96") != []
            or feature_spec.get("test_opened") is not False):
        raise AssertionError("P1 gate does not permit the fixed M1/M4-only research path")
    return feature_spec


def _assert_candidate_oof(original: pd.DataFrame, candidates: pd.DataFrame,
                          boundary: pd.Timestamp) -> dict:
    required = {"origin", "target_time", "horizon", "fold", "model", "y", "pred", "tau", "alert"}
    if candidates.empty or not required <= set(candidates):
        raise ValueError("Phase 2 candidates lack the original OOF schema")
    if (pd.to_datetime(candidates.origin).ge(boundary).any()
            or pd.to_datetime(candidates.target_time).ge(boundary).any()
            or pd.to_datetime(candidates.target_time).isna().any()
            or pd.to_datetime(candidates.origin).isna().any()):
        raise AssertionError("Candidate OOF crossed the locked holdout boundary")
    valid_h = candidates.horizon.isin((4, 16, 96))
    expected = {(4, M1_NAME)} | {(h, M1_NAME) for h in (16, 96)} | {
        (h, name) for h in (16, 96) for name in M4_NAMES}
    found = {(int(row.horizon), str(row.model)) for row in candidates[["horizon", "model"]].drop_duplicates().itertuples(index=False)}
    if not valid_h.all() or found != expected:
        raise AssertionError(f"Unexpected Phase 2 candidate set: {found ^ expected}")
    if candidates.duplicated(["origin", "target_time", "horizon", "fold", "model"]).any():
        raise AssertionError("Candidate OOF contains duplicate score rows")
    score = original.drop_duplicates(["origin", "target_time", "horizon", "fold"])
    reference = score[["origin", "target_time", "horizon", "fold", "y", "tau"]]
    merged = candidates.merge(reference, on=["origin", "target_time", "horizon", "fold"],
                              suffixes=("", "_original"), indicator=True, validate="many_to_one")
    if merged._merge.ne("both").any():
        raise AssertionError("Candidate score row is outside the original CV grid")
    if (not np.allclose(merged.y, merged.y_original, rtol=0, atol=1e-10, equal_nan=True)
            or not np.allclose(merged.tau, merged.tau_original, rtol=0, atol=1e-10, equal_nan=True)):
        raise AssertionError("Candidate changed original target or fold peak threshold")
    delta = pd.to_datetime(candidates.target_time) - pd.to_datetime(candidates.origin)
    if not (delta == pd.to_timedelta(candidates.horizon.to_numpy(dtype=int) * 15, unit="m")).all():
        raise AssertionError("Candidate horizon/origin/target alignment changed")
    group_counts = candidates.groupby(["horizon", "model", "fold"]).size()
    if any((h, name, fold) not in group_counts.index or group_counts[h, name, fold] == 0
           for h, name in expected for fold in range(3)):
        raise AssertionError("Candidate is missing an original development fold")
    m1_h4 = candidates.loc[candidates.horizon.eq(4) & candidates.model.eq(M1_NAME)]
    if (not m1_h4.reference_only.fillna(False).astype(bool).all()
            or m1_h4.selection_used.fillna(True).astype(bool).any()):
        raise AssertionError("h4 residual model must remain reference-only")
    for horizon in (16, 96):
        baseline = original.loc[original.horizon.eq(horizon) & original.model.eq("lgbm_quantile_b")]
        for name in M4_NAMES:
            candidate = candidates.loc[candidates.horizon.eq(horizon) & candidates.model.eq(name)]
            joined = candidate.merge(baseline[["origin", "target_time", "horizon", "fold", "pred", "q50", "q90_cal"]],
                                     on=["origin", "target_time", "horizon", "fold"],
                                     suffixes=("", "_b"), validate="one_to_one")
            if len(joined) != len(candidate):
                raise AssertionError("Adaptive q95 has score rows outside original B")
            for column in ("pred", "q50", "q90_cal"):
                if not np.allclose(joined[column], joined[column + "_b"], atol=1e-10, rtol=0,
                                   equal_nan=True):
                    raise AssertionError(f"Adaptive q95 changed fixed {column}")
            if ((candidate.q95_cal.to_numpy(dtype=float) < candidate.q90_cal.to_numpy(dtype=float) - 1e-10).any()
                    or (candidate.q975_cal.to_numpy(dtype=float) < candidate.q95_cal.to_numpy(dtype=float) - 1e-10).any()
                    or not candidate.p_exceed.between(0, 1).all()):
                raise AssertionError("Adaptive quantiles or peak probability are invalid")
    return {"candidate_pairs": len(found), "candidate_rows": len(candidates),
            "candidate_counts_by_horizon": {str(h): sum(k[0] == h for k in found) for h in (4, 16, 96)},
            "score_boundary_exclusive": str(boundary)}


def run_research_development(
    history: pd.DataFrame, cfg: dict, original_predictions: pd.DataFrame,
    original_metrics: pd.DataFrame, original_selection: dict, folds: list[dict],
    output_dir: str | Path = "outputs", *, root: str | Path | None = None,
    feature_spec: dict | None = None,
) -> dict:
    """Fit preregistered M1/M4, append OOF, and preserve original selection."""
    history = _safe_history(history)
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    if boundary != SEALED_BOUNDARY:
        raise ValueError("Research CV boundary differs from sealed first test origin")
    if original_predictions.empty or pd.to_datetime(original_predictions.target_time).ge(boundary).any():
        raise ValueError("Original development OOF is absent or crosses the holdout boundary")
    if cfg.get("research_0925", {}).get("enabled") is not True:
        raise ValueError("Research development requires explicit config enablement")
    project_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    _validate_gate(project_root, feature_spec)
    output = Path(output_dir)
    manifest = {"selection": deepcopy(original_selection), "folds": deepcopy(folds)}
    source_snapshot = original_predictions.copy(deep=True)
    selection_snapshot = deepcopy(original_selection)

    from .research_cv import fit_residual_candidates
    from .models.adaptive_q95 import fit_adaptive_candidates

    m1 = fit_residual_candidates(history, cfg, deepcopy(manifest),
                                 original_predictions.copy(deep=True), output_dir=output)
    m4 = fit_adaptive_candidates(history, cfg, deepcopy(manifest),
                                 original_predictions.copy(deep=True), output_dir=output)
    pd.testing.assert_frame_equal(original_predictions, source_snapshot,
                                  check_dtype=True, check_exact=True)
    if (json.dumps(original_selection, sort_keys=True, default=str, allow_nan=True)
            != json.dumps(selection_snapshot, sort_keys=True, default=str, allow_nan=True)):
        raise AssertionError("Candidate fitting mutated original model selection")
    candidates = pd.concat([m1["predictions"], m4["predictions"]], ignore_index=True, sort=False)
    validation = _assert_candidate_oof(original_predictions, candidates, boundary)
    predictions = pd.concat([original_predictions, candidates], ignore_index=True, sort=False)
    metrics = evaluate_all(predictions, cfg=cfg)
    selection, decisions = extend_selection(original_selection, original_predictions,
                                            candidates, cfg)
    if selection["by_horizon"].get("1") != original_selection["by_horizon"].get("1"):
        raise AssertionError("h1 selection changed without a registered candidate")
    before_after = pd.DataFrame([
        {"horizon": h,
         "point_before": original_selection["by_horizon"][str(h)]["point_model"],
         "point_after": selection["by_horizon"][str(h)]["point_model"],
         "conformal_before": original_selection["by_horizon"][str(h)].get("conformal"),
         "conformal_after": selection["by_horizon"][str(h)].get("conformal"),
         "risk_before": f"lgbm_quantile_{original_selection['by_horizon'][str(h)].get('conformal')}",
         "risk_after": selection["by_horizon"][str(h)].get("risk_model", f"lgbm_quantile_{selection['by_horizon'][str(h)].get('conformal')}"),
         "point_changed": original_selection["by_horizon"][str(h)]["point_model"] != selection["by_horizon"][str(h)]["point_model"],
         "risk_changed": (selection["by_horizon"][str(h)].get("risk_model")
                          not in (None, f"lgbm_quantile_{original_selection['by_horizon'][str(h)].get('conformal')}")),
        } for h in (1, 4, 16, 96)])
    analysis_dir = output / "analysis_p2"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    # Machine-readable evidence keeps nested CIs; tabular decisions flatten
    # their most important fields while retaining full JSON in the log.
    table = decisions.drop(columns=["evidence"]).copy()
    table.to_csv(analysis_dir / "P2_candidate_decisions.csv", index=False, encoding="utf-8-sig")
    before_after.to_csv(analysis_dir / "P2_selection_before_after.csv", index=False, encoding="utf-8-sig")
    integration = {"scope": "development_oof_only", "validated": validation,
                   "original_selection_unchanged_during_fits": True,
                   "candidate_decisions": decisions.to_dict("records"),
                   "before_after": before_after.to_dict("records"),
                   "m1_model_fit_count": len(m1.get("fit_metadata", [])),
                   "m4_model_fit_count": int(m4.get("model_fit_count", 0)),
                   "forking_paths": {"new_candidates": 11,
                                     "by_horizon": {"4": 1, "16": 5, "96": 5},
                                     "new_feature_sets": 1,
                                     "m2_m3_executed": False,
                                     "confirmatory_tests": 27}}
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "P2_research_integration.json").write_text(
        json.dumps(integration, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"predictions": predictions, "metrics": metrics, "selection": selection,
            "original_selection": original_selection,
            "candidates": candidates, "decisions": decisions,
            "before_after": before_after,
            "m1": m1, "m4": m4, "integration": integration}
