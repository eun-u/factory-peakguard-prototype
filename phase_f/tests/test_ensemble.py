"""Synthetic contracts for cal-only Phase F combinations and F10 slices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_f.diagnostics import summarize_errors
from phase_f.ensemble import base_combinations, combine_predictions, configurations


def _frames() -> dict[str, pd.DataFrame]:
    cal = pd.date_range("2021-06-01", periods=96, freq="15min")
    score = pd.date_range("2021-06-14", periods=96, freq="15min")
    origins = cal.append(score)
    target = origins + pd.Timedelta(hours=1)
    y = 12 + 3 * np.sin(2 * np.pi * np.arange(len(origins)) / 96)
    role = np.array(["cal"] * len(cal) + ["score"] * len(score))
    arm = np.where(role == "cal", "CAL", "EXPLORE")
    base = pd.DataFrame({"horizon": 4, "fold": 0, "origin": origins, "target_time": target,
                         "y": y, "tau": 13., "d2": False, "role": role, "arm": arm,
                         "fit_mean": 12., "mase_scale": 2.})
    a = base.assign(model="A", pred=y + 1, p_peak=np.where(np.arange(len(origins)) % 2 == 0, .9, .1), q90=y + 5)
    b = base.assign(model="B", pred=y - 2)
    return {"A": a, "B": b}


@pytest.mark.parametrize("method", ["mean", "median", "inverse_cal_mae", "nnls", "ridge_positive",
                                      "regime_gate", "peak_gate", "bias_hour_daytype", "quantile_risk_shift"])
def test_all_operators_ignore_score_labels(method):
    frames = _frames()
    config = {"risk_source": "A", "base_id": "A", "shift": .5, "min_count": 8,
              "peak_min_count": 5} if method in {"peak_gate", "quantile_risk_shift", "bias_hour_daytype"} else {}
    before, audit = combine_predictions(frames, method, "COMBINED", params=config)
    changed = {name: frame.copy() for name, frame in frames.items()}
    for frame in changed.values():
        frame.loc[frame.role.eq("score"), "y"] += 1000
    after, _ = combine_predictions(changed, method, "COMBINED", params=config)
    np.testing.assert_allclose(after.pred, before.pred)
    assert audit["fit_roles"] == ["cal"]
    assert audit["score_labels_used_for_fit"] is False
    assert before.role.value_counts().to_dict() == {"cal": 96, "score": 96}
    assert np.isfinite(before.pred).all()


def test_inverse_cal_mae_has_expected_per_fold_weights():
    frames = _frames()
    combined, audit = combine_predictions(frames, "inverse_cal_mae", "INV")
    weights = audit["groups"][0]["weights"]
    assert weights["A"] == pytest.approx(2 / 3)
    assert weights["B"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(combined.pred.to_numpy(), frames["A"].y.to_numpy())
    assert audit["groups"][0]["cal_last_target"] < audit["groups"][0]["score_first_origin"]


def test_cal_weights_are_separate_by_horizon_fold():
    first = _frames()
    second = {name: frame.copy() for name, frame in first.items()}
    for frame in second.values():
        frame["fold"] = 1
        frame["origin"] += pd.Timedelta(days=21)
        frame["target_time"] += pd.Timedelta(days=21)
        frame.loc[frame.role.eq("score"), "arm"] = "CONFIRM"
    # B wins the second fold's calibration; A wins the first fold's.
    second["A"].loc[second["A"].role.eq("cal"), "pred"] += 10
    for name in first:
        first[name] = pd.concat([first[name], second[name]], ignore_index=True)
    combined, audit = combine_predictions(first, "inverse_cal_mae", "INV")
    assert len(audit["groups"]) == 2
    assert audit["groups"][0]["weights"]["A"] > audit["groups"][0]["weights"]["B"]
    assert audit["groups"][1]["weights"]["B"] > audit["groups"][1]["weights"]["A"]
    fold0 = combined.loc[combined.fold.eq(0), "pred"].to_numpy()
    changed = {name: frame.copy() for name, frame in first.items()}
    for frame in changed.values():
        frame.loc[frame.fold.eq(1) & frame.role.eq("cal"), "y"] += 100
    after, _ = combine_predictions(changed, "inverse_cal_mae", "INV")
    np.testing.assert_allclose(after.loc[after.fold.eq(0), "pred"].to_numpy(), fold0)


def test_quantile_shift_only_moves_risky_points_toward_q90():
    frames = _frames()
    result, audit = combine_predictions(frames, "quantile_risk_shift", "SHIFT",
                                        params={"base_id": "A", "risk_source": "A", "shift": .5})
    source = frames["A"]
    expected = source.pred.to_numpy() + .5 * (source.p_peak.to_numpy() >= .5) * (source.q90 - source.pred).to_numpy()
    np.testing.assert_allclose(result.pred.to_numpy(), expected)
    assert audit["groups"][0]["fixed_shift"] == .5


def test_strict_cohort_and_metadata_rejection():
    frames = _frames()
    mismatch = {name: frame.copy() for name, frame in frames.items()}
    mismatch["B"].loc[mismatch["B"].role.eq("score"), "origin"] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="cohort"):
        combine_predictions(mismatch, "mean", "OUT")
    missing = {name: frame.copy() for name, frame in frames.items()}
    missing["B"] = missing["B"].iloc[:-1]
    with pytest.raises(ValueError, match="cohort"):
        combine_predictions(missing, "mean", "OUT")
    truth = {name: frame.copy() for name, frame in frames.items()}
    truth["B"].loc[truth["B"].role.eq("score"), "y"] += 1
    with pytest.raises(ValueError, match="mismatched y"):
        combine_predictions(truth, "mean", "OUT")


def test_cal_target_score_origin_embargo_rejected():
    frames = _frames()
    for frame in frames.values():
        score = frame.role.eq("score")
        frame.loc[score, "origin"] -= pd.Timedelta(days=14)
        frame.loc[score, "target_time"] -= pd.Timedelta(days=14)
    with pytest.raises(ValueError, match="embargo"):
        combine_predictions(frames, "mean", "OUT")


def test_diagnostics_keep_confirm_closed_until_selection():
    frames = _frames()
    a = frames["A"]
    with pytest.raises(PermissionError, match="frozen"):
        summarize_errors(a.assign(arm=np.where(a.role.eq("score"), "CONFIRM", "CAL")), arm="CONFIRM")
    result = summarize_errors(a)
    assert result["audit"]["arm"] == "EXPLORE"
    assert result["by_hour"].n.sum() == 96
    assert result["by_domain"].set_index("group").loc["D1_all", "n"] == 96
    assert result["audit"]["late_july_augmented_status"] == "UNKNOWN"
    assert result["late_july"].n.sum() == 0


def test_diagnostics_only_use_selected_arm_and_late_july_window():
    frames = _frames()
    source = frames["A"].copy()
    source.loc[source.role.eq("score"), "origin"] = pd.date_range("2021-07-26", periods=96, freq="15min")
    source.loc[source.role.eq("score"), "target_time"] = source.loc[source.role.eq("score"), "origin"] + pd.Timedelta(hours=1)
    result = summarize_errors(source)
    late = result["late_july"].set_index("group")
    assert late.loc["late_july_two_weeks", "n"] == 96
    assert late.loc["preceding_two_weeks", "n"] == 0
    # A CONFIRM copy with extreme errors must not affect EXPLORE slices.
    confirm = source.loc[source.role.eq("score")].copy()
    confirm["arm"] = "CONFIRM"
    confirm["origin"] += pd.Timedelta(days=7)
    confirm["target_time"] += pd.Timedelta(days=7)
    confirm["y"] = 1e6
    mixed = pd.concat([source, confirm], ignore_index=True)
    again = summarize_errors(mixed)
    pd.testing.assert_frame_equal(again["by_hour"], result["by_hour"])


def test_configurations_and_explicit_base_sets():
    ids = [row["id"] for row in configurations()]
    assert len(ids) == len(set(ids))
    assert "F7-6-b5-r1-lgbm-dl" in base_combinations()
