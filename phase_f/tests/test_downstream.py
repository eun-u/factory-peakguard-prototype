"""F11 synthetic fixed-cohort and cal-only contracts."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from phase_f.downstream import _cal_split, _fit_one, evaluate_downstream


def _frames():
    cal_rows, score_rows = [], []
    for fold in range(3):
        beginning = pd.Timestamp("2021-06-07") + timedelta(days=fold * 14)
        for role, rows, count, start in (("cal", cal_rows, 160, beginning),
                                         ("score", score_rows, 72, beginning + timedelta(days=4))):
            for i in range(count):
                origin = start + timedelta(minutes=15 * i)
                target = origin + timedelta(hours=4)
                actual = 103 + 7 * np.sin(i / 4) + 5 * np.cos(i / 11 + fold)
                for model, shift in (("B5", 0.), ("F6-best", 1.2)):
                    predicted = 101 + 4 * np.sin(i / 5 + fold) + shift
                    rows.append({"model": model, "horizon": 16, "fold": fold,
                                 "origin": origin, "target_time": target,
                                 "y": actual, "pred": predicted, "tau": 107.,
                                 "d2": bool(i % 5 == 0), "role": role,
                                 "arm": "CAL" if role == "cal" else "CONFIRM",
                                 "cal_provenance": "fit_only", "sigma": 8.})
    return pd.DataFrame(cal_rows), pd.DataFrame(score_rows)


def _lock():
    return {"primary_candidate": "F6-best", "candidates": ["F6-best"],
            "selection_lock_sha256": "synthetic-selection", "confirm_once_sha256": "synthetic-confirm",
            "confirm_complete": True,
            "fit_peak_prevalence": {model: {str(fold): .2 for fold in range(3)}
                                    for model in ("B5", "F6-best")},
            "probability_source": {"B5": "gaussian", "F6-best": "residual"}}


def test_f11_runs_phase_e_fixed_grid_on_same_confirm_keys():
    cal, score = _frames()
    out = evaluate_downstream(cal, score, _lock(), n_boot=20, seed=42)
    derived = out["score_probabilities"]
    assert len(derived) == len(score)
    assert set(derived.model) == {"B5", "F6-best"}
    assert derived.p_cal.between(0, 1).all()
    assert np.isfinite(derived.U95).all()
    assert set(out["calibration_fit"].source) == {"model_native_gaussian_sigma", "early_cal_empirical_residual_CDF"}
    assert (pd.to_datetime(out["calibration_fit"].source_last_target) <
            pd.to_datetime(out["calibration_fit"].calibrator_first_origin)).all()
    episode = out["alert_episode_metrics"]
    assert set(episode.threshold) == {.01, .05, .10, .20, .30, .50}
    assert set(episode.policy) == {"1/1", "2/2"}
    comparison = out["comparison_vs_B5"]
    assert {"episode_recall", "peak_coverage_conformal", "BS_cal"} <= set(comparison.metric)


def test_score_truth_never_changes_calibration_and_explore_rows_are_unread():
    cal, score = _frames()
    baseline = evaluate_downstream(cal, score, _lock(), n_boot=2)["calibration_fit"]
    changed = score.copy()
    changed["y"] += 999
    extra = score.iloc[[0]].copy()
    extra["arm"] = "EXPLORE"
    extra["y"] = np.nan  # filtered before numerical checks
    changed = pd.concat([changed, extra], ignore_index=True)
    result = evaluate_downstream(cal, changed, _lock(), n_boot=2)["calibration_fit"]
    pd.testing.assert_frame_equal(baseline, result)


def test_gate_rejects_unlocked_confirm_cohort_mismatch_and_ensemble_insample():
    cal, score = _frames()
    lock = _lock()
    lock["confirm_complete"] = False
    with pytest.raises(ValueError, match="completed one-time CONFIRM"):
        evaluate_downstream(cal, score, lock, n_boot=2)
    lock = _lock()
    wrong = score.copy()
    mask = wrong.model.eq("F6-best") & wrong.fold.eq(0) & (wrong.index == wrong.index[1])
    wrong.loc[mask, "origin"] += timedelta(minutes=1)
    wrong.loc[mask, "target_time"] += timedelta(minutes=1)
    with pytest.raises(ValueError, match="forecast keys differ"):
        evaluate_downstream(cal, wrong, lock, n_boot=2)
    lock["ensemble_models"] = ["F6-best"]
    with pytest.raises(ValueError, match="cal provenance is unknown"):
        evaluate_downstream(cal, score, lock, n_boot=2)


def test_ensemble_cal_uses_only_chronological_oof_keys_for_all_models():
    cal, score = _frames()
    lock = _lock()
    lock["ensemble_models"] = ["F6-best"]
    for fold in range(3):
        candidate = cal.model.eq("F6-best") & cal.fold.eq(fold)
        indices = cal.index[candidate]
        cal.loc[indices[:50], "cal_provenance"] = "in_sample"
        cal.loc[indices[50:], "cal_provenance"] = "chronological_oof"
    result = evaluate_downstream(cal, score, lock, n_boot=2)
    fits = result["calibration_fit"]
    assert len(fits) == 6
    assert fits.groupby("fold").residual_source_n.nunique().eq(1).all()
    assert fits.groupby("fold").calibrator_n.nunique().eq(1).all()
    assert (fits.residual_source_n + fits.calibrator_n < 110).all()


def test_internal_cal_split_has_horizon_embargo_and_uses_late_cal_for_platt():
    cal, score = _frames()
    group = cal.loc[cal.model.eq("F6-best") & cal.fold.eq(0)]
    early, late = _cal_split(group)
    assert len(early) >= 30 and len(late) >= 30
    assert early.target_time.max() < late.origin.min()
    scored, audit = _fit_one(group, score.loc[score.model.eq("F6-best") & score.fold.eq(0)],
                             "F6-best", 0, "residual", .2)
    assert audit["residual_source_n"] == len(early)
    assert audit["calibrator_n"] == len(late)
    assert audit["score_labels_used_in_fit"] is False
    assert scored.is_peak.eq(scored.y > scored.tau).all()
