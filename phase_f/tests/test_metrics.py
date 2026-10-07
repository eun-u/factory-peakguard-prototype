"""Synthetic contract checks for Phase F development metrics."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from phase_f.metrics import HORIZONS, evaluate, paired_ci


def predictions(model: str = "B5") -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2021-07-01", "2021-07-02", "2021-07-08",
                            "2021-07-09", "2021-07-15", "2021-07-16"])
    for fold in range(3):
        for day_number in (2 * fold, 2 * fold + 1):
            for slot in (0, 1):
                if fold == 0 and slot == 1:
                    continue  # unequal fold row counts test weighted normalizers
                target = dates[day_number] + timedelta(hours=12 + 6 * slot)
                actual = float(10 + fold + slot + 4 * (day_number % 2))
                error = (4 + day_number % 2) if model == "B5" else (1 + day_number % 3)
                for horizon in HORIZONS:
                    rows.append({"model": model, "horizon": horizon, "fold": fold,
                                 "origin": target - timedelta(minutes=15 * horizon),
                                 "target_time": target, "y": actual, "pred": actual + error,
                                 "tau": 12., "d2": bool(day_number % 2),
                                 "role": "score", "arm": "EXPLORE",
                                 "fit_mean": float((10, 20, 40)[fold]),
                                 "mase_scale": float((1, 2, 4)[fold]),
                                 "train_seconds": 30., "inference_seconds": 2.})
    return pd.DataFrame(rows)


def test_evaluate_filters_arm_before_normalization_and_weights_fit_mean():
    b5 = predictions()
    confirm = b5.iloc[[0]].copy()
    confirm["arm"] = "CONFIRM"
    confirm["y"] = 9999.
    confirm["fit_mean"] = 9999.
    confirm["pred"] = 9999.
    out = evaluate(pd.concat([b5, confirm], ignore_index=True))
    pooled = out["pooled"].query("dataset == 'D1' and model == 'B5' and horizon == 4").iloc[0]
    source = b5[b5.horizon.eq(4)]
    expected_fit_mean = (2 * 10 + 4 * 20 + 4 * 40) / 10
    error = (source.pred - source.y).to_numpy(float)
    assert pooled.n == 10
    assert pooled.fit_mean == expected_fit_mean
    assert pooled.nMAE == pytest.approx(np.abs(error).mean() / expected_fit_mean)
    assert pooled.CVRMSE == pytest.approx(np.sqrt(np.mean(error**2)) / expected_fit_mean)
    assert pooled.NMBE == pytest.approx(-error.mean() / expected_fit_mean)
    assert pooled.MASE == pytest.approx(np.mean(np.abs(error) / source.mase_scale))
    assert out["auc"].query("dataset == 'D1' and model == 'B5'").iloc[0].AUC_MAE == pytest.approx(pooled.MAE)
    assert len(out["fold"]) == 2 * 3 * len(HORIZONS)
    assert out["daily_summary"].query("subset == 'full_96'").n_days.eq(0).all()


def test_peak_masks_optional_probabilities_and_zero_actuals():
    b5 = predictions()
    b5["y"] = 0.
    b5["pred"] = 2.
    b5["tau"] = 1.
    b5["q10"] = -1.
    b5["q50"] = 0.
    b5["q90"] = 1.
    b5["q95"] = 2.
    b5["p_peak"] = .8
    pooled = evaluate(b5)["pooled"].query("dataset == 'D1' and horizon == 4").iloc[0]
    assert pooled.peak_n == 0
    assert np.isnan(pooled.Peak_MAE)
    assert pooled.predicted_peak_n == pooled.n
    assert pooled.PredPeak_MAE == 2
    assert pooled.Brier == pytest.approx(.64)
    assert pooled.pinball_q50 == 0
    assert pooled.coverage_10_90 == 1
    assert np.isnan(pooled.MAPE_nonzero)
    assert pooled.mape_nonzero_n == 0


def test_daily_max_reports_partial_and_full_days_separately():
    b5 = predictions()
    mask = b5.fold.eq(0) & b5.horizon.eq(4) & b5.target_time.dt.normalize().eq(pd.Timestamp("2021-07-01"))
    b5 = b5.loc[~mask].copy()
    template = predictions().loc[mask].iloc[0].to_dict()
    rows = []
    for slot in range(96):
        target = pd.Timestamp("2021-07-01") + timedelta(minutes=15 * slot)
        rows.append({**template, "target_time": target,
                     "origin": target - timedelta(minutes=60),
                     "y": float(slot), "pred": float(slot + (1 if slot == 94 else 0))})
    b5 = pd.concat([b5, pd.DataFrame(rows)], ignore_index=True)
    result = evaluate(b5)
    day = result["daily"].loc[lambda t: t.dataset.eq("D1") & t.fold.eq(0) & t.horizon.eq(4) &
                              t.target_day.eq(pd.Timestamp("2021-07-01"))].iloc[0]
    assert day.full_day
    assert day.observed_slots == 96
    assert day.daily_max_error == 0
    assert day.timing_error_minutes == -15
    assert day.timing_abs_error_minutes == 15
    full = result["daily_summary"].query("dataset == 'D1' and scope == 'fold' and fold == 0 and horizon == 4 and subset == 'full_96'").iloc[0]
    assert full.n_days == 1
    partial = result["daily_summary"].query("dataset == 'D1' and scope == 'fold' and fold == 1 and horizon == 4 and subset == 'full_96'").iloc[0]
    assert partial.n_days == 0
    assert np.isnan(partial.daily_max_MAE)


def test_paired_ci_joint_target_day_and_week_blocks():
    b5, candidate = predictions("B5"), predictions("F1")
    result = paired_ci(candidate, b5, n=1000, seed=42)
    h4 = result.query("dataset == 'D1' and horizon == 4 and metric == 'MAE_improvement'").iloc[0]
    auc = result.query("dataset == 'D1' and metric == 'AUC_MAE_improvement'").iloc[0]
    expected = (b5.query("horizon == 4").pred.to_numpy() - candidate.query("horizon == 4").pred.to_numpy()).mean()
    assert h4.estimate == pytest.approx(expected)
    assert auc.estimate == pytest.approx(h4.estimate)
    assert auc.ci_low == pytest.approx(h4.ci_low)
    assert auc.ci_high == pytest.approx(h4.ci_high)
    assert auc.n_blocks == 6
    assert auc.ci_status == "available"
    d2_peak = result.query("dataset == 'D2' and metric == 'AUC_PeakMAE_degradation'").iloc[0]
    assert d2_peak.ci_status == "available"
    weekly = paired_ci(candidate, b5, n=1000, seed=42, block="week")
    assert weekly.query("dataset == 'D1' and metric == 'AUC_MAE_improvement'").iloc[0].n_blocks == 3


def test_paired_ci_rejects_unmatched_keys_or_truth_and_marks_zero_peaks_unknown():
    b5, candidate = predictions("B5"), predictions("F1")
    with pytest.raises(ValueError, match="identical forecast keys"):
        paired_ci(candidate.iloc[:-1], b5, n=10)
    changed = candidate.copy()
    changed.loc[0, "y"] += 1
    with pytest.raises(ValueError, match="Paired y conflicts"):
        paired_ci(changed, b5, n=10)
    b5["tau"] = 1000.
    candidate["tau"] = 1000.
    ci = paired_ci(candidate, b5, n=100)
    peak = ci.query("dataset == 'D1' and metric == 'AUC_PeakMAE_degradation'").iloc[0]
    assert np.isnan(peak.ci_low)
    assert peak.ci_status == "unavailable"


def test_missing_horizon_is_an_error_and_empty_d2_is_visible():
    b5 = predictions()
    missing = b5.loc[~(b5.fold.eq(1) & b5.horizon.eq(16))]
    with pytest.raises(ValueError, match="Missing horizons"):
        evaluate(missing)
    b5["d2"] = False
    result = evaluate(b5)
    d2 = result["auc"].query("dataset == 'D2'").iloc[0]
    assert d2.n_available_horizons == 0
    assert np.isnan(d2.AUC_MAE)
    assert result["fold"].query("dataset == 'D2'").n.eq(0).all()


def test_evaluate_refuses_unpaired_model_cohorts():
    b5, candidate = predictions("B5"), predictions("F1")
    wrong_key = candidate.copy()
    changed = wrong_key.fold.eq(1) & wrong_key.horizon.eq(4) & wrong_key.target_time.eq(pd.Timestamp("2021-07-08 12:00"))
    wrong_key.loc[changed, "target_time"] += timedelta(minutes=15)
    wrong_key.loc[changed, "origin"] += timedelta(minutes=15)
    with pytest.raises(ValueError, match="different forecast-key cohort"):
        evaluate(pd.concat([b5, wrong_key], ignore_index=True))
    wrong_truth = candidate.copy()
    wrong_truth.loc[changed, "y"] += 1
    with pytest.raises(ValueError, match="inconsistent paired y"):
        evaluate(pd.concat([b5, wrong_truth], ignore_index=True))
