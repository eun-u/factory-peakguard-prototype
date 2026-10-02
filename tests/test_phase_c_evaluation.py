"""Synthetic checks for the isolated Phase C evaluation contract."""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from phase_c.evaluation import HORIZONS, MAIN, evaluate


def predictions(*, c1_d2_error=3.0):
    errors = {"B0": 15.0, "B1": 10.0, "B2": 11.0, "B3": 12.0,
              "B4": 8.0, "B5": 9.0, "M1": 4.0, "M1-W": 4.0,
              "M2": 5.0, "C1": 3.0, "R1": 1.0}
    rows = []
    for fold in range(3):
        for horizon in HORIZONS:
            for day in range(4):
                target = datetime(2021, 1, 1, 12) + timedelta(days=fold * 15 + day)
                origin = target - timedelta(minutes=15 * horizon)
                y = 100.0 + day
                is_d2 = day == 0
                for model in (*MAIN, "R1"):
                    error = c1_d2_error if model == "C1" and is_d2 else errors[model]
                    rows.append({"model": model, "horizon": horizon, "fold": fold,
                                 "origin": origin, "target_time": target,
                                 "y": y, "pred": y + error, "tau": 95.0,
                                 "d2": is_d2, "train_seconds": float(fold + 1),
                                 "inference_seconds": .5,
                                 "selected_config": json.dumps({"model": model}),
                                 "development_only": True})
    return pd.DataFrame(rows)


def test_complete_paired_evaluation_selects_clear_c1_and_writes_contract(tmp_path):
    frame = predictions()
    # Missing reference rows must not shrink the ten-model comparison cohort.
    frame = frame.loc[~(frame.model.eq("R1") & frame.fold.eq(0) & frame.horizon.eq(4))]
    result = evaluate(frame, tmp_path, bootstrap_n=100, seed=7)
    assert result["selected_M_star"] == "C1"
    assert result["strongest_baseline"] == "B1"
    assert result["D2_available"] is True
    assert len(result["paths"]) == 11
    auc = pd.read_csv(tmp_path / "tables/model_auc_summary.csv")
    c1 = auc[(auc.dataset == "D1") & (auc.model == "C1")].iloc[0]
    assert c1.AUC_MAE == pytest.approx(3.0)
    assert c1.AUC_PeakMAE == pytest.approx(3.0)
    assert c1.n_horizons == 13
    fold = pd.read_csv(tmp_path / "tables/model_horizon_fold_metrics.csv")
    assert fold[(fold.dataset == "D1") & (fold.model == "M1")].n.eq(4).all()
    pairs = pd.read_csv(tmp_path / "tables/model_pairwise_ci.csv")
    assert set(pairs.dataset) == {"D1", "D2"}
    assert set(pairs.metric) == {"MAE", "Peak_MAE", "AUC_MAE", "AUC_PeakMAE"}
    assert len(pairs) == 2 * 45 * 14 * 2
    assert pairs.ci_status.eq("available").all()
    assert pairs.valid_draw_count.eq(100).all()
    decision = pd.read_csv(tmp_path / "tables/model_selection_decision.csv").set_index("model")
    assert decision.loc["B0", "complexity_rank"] == 1
    assert decision.loc["M2", "complexity_rank"] == 10
    assert all((tmp_path / "figures" / name).is_file() for name in (
        "mae_horizon_curve.png", "peak_mae_horizon_curve.png",
        "skill_horizon_curve.png", "d1_d2_comparison.png"))


def test_main_missing_prediction_reduces_every_main_model_equally(tmp_path):
    frame = predictions()
    bad = frame.model.eq("M2") & frame.horizon.eq(4) & frame.fold.eq(0)
    frame.loc[bad.idxmax(), "pred"] = np.nan
    result = evaluate(frame, tmp_path, bootstrap_n=30)
    assert {counts["removed"] for counts in result["removed_counts"].values()} == {1}
    assert result["removed_counts"]["M2"]["nonfinite"] == 1
    assert result["removed_counts"]["M2"]["raw_missing"] == 0
    assert result["removed_counts"]["B0"]["removed_for_pairing"] == 1
    fold = pd.read_csv(tmp_path / "tables/model_horizon_fold_metrics.csv")
    affected = fold[(fold.dataset == "D1") & (fold.horizon == 4) & (fold.fold == 0) & fold.model.isin(MAIN)]
    assert affected.n.eq(3).all()


def test_missing_origin_is_counted_and_paired_not_rejected(tmp_path):
    frame = predictions()
    one = frame.model.eq("M2") & frame.horizon.eq(4) & frame.fold.eq(0)
    frame = frame.drop(index=one.idxmax())
    result = evaluate(frame, tmp_path, bootstrap_n=30)
    assert result["removed_counts"]["M2"]["raw_missing"] == 1
    assert result["removed_counts"]["M2"]["nonfinite"] == 0
    assert result["removed_counts"]["B0"]["removed_for_pairing"] == 1
    fold = pd.read_csv(tmp_path / "tables/model_horizon_fold_metrics.csv")
    affected = fold[(fold.dataset == "D1") & (fold.horizon == 4) & (fold.fold == 0) & fold.model.isin(MAIN)]
    assert affected.n.eq(3).all()


def test_residual_d2_deterioration_rejects_c1(tmp_path):
    result = evaluate(predictions(c1_d2_error=7.0), tmp_path, bootstrap_n=80)
    decision = pd.read_csv(tmp_path / "tables/model_selection_decision.csv").set_index("model")
    assert "D2_worse_than_M1" in decision.loc["C1", "C1_rejection_reasons"]
    assert not bool(decision.loc["C1", "eligible"])
    assert result["selected_M_star"] == "M1"


def test_unavailable_d2_and_peak_are_explicit(tmp_path):
    frame = predictions()
    frame["d2"] = False
    frame["tau"] = 200.0
    result = evaluate(frame, tmp_path, bootstrap_n=30)
    assert result["D2_available"] is False
    assert result["peak_safeguard_available"] is False
    assert result["selected_M_star"] is None
    assert result["selection_status"] == "insufficient_peak_evidence"
    decision = pd.read_csv(tmp_path / "tables/model_selection_decision.csv").set_index("model")
    assert decision.peak_ci_status.eq("unavailable").all()
    assert decision.selection_status.eq("insufficient_peak_evidence").all()
    assert not decision.eligible.any()
    assert "D2_unavailable" in decision.loc["C1", "C1_rejection_reasons"]
    pairs = pd.read_csv(tmp_path / "tables/model_pairwise_ci.csv")
    unavailable = pairs[pairs.metric.isin(("Peak_MAE", "AUC_PeakMAE"))]
    assert unavailable.ci_status.eq("unavailable").all()
    assert unavailable.ci_reason.eq("nonfinite_point_estimate").all()
    assert unavailable.valid_draw_count.eq(0).all()


def test_boundary_duplicate_truth_and_missing_horizon_fail_closed(tmp_path):
    base = predictions()
    cases = []
    boundary = base.copy()
    boundary.loc[0, "target_time"] = pd.Timestamp("2021-08-09 09:45")
    cases.append(boundary)
    cases.append(pd.concat([base, base.iloc[[0]]], ignore_index=True))
    truth = base.copy()
    peer = truth.model.eq("B1") & truth.horizon.eq(4) & truth.fold.eq(0)
    truth.loc[peer.idxmax(), "y"] = 999.0
    cases.append(truth)
    cases.append(base.loc[base.horizon.ne(16)])
    reference = base.copy()
    reference.loc[reference.model.eq("R1").idxmax(), "y"] = 999.0
    cases.append(reference)
    for frame in cases:
        with pytest.raises(ValueError):
            evaluate(frame, tmp_path, bootstrap_n=10)
