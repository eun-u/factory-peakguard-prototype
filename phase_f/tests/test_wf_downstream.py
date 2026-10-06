"""Synthetic weekly Phase E downstream cohort, calibration and alert contracts."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f.wf_downstream import _fit_week, evaluate_candidate, paired_episode_ci


def _fixture(*, include_confirm: bool = False, candidate_quantiles: bool = False):
    contexts = {}
    candidate_rows, baseline_rows = [], []
    starts = [pd.Timestamp("2021-04-19"), pd.Timestamp("2021-05-03")]
    if include_confirm:
        starts.append(pd.Timestamp("2021-04-12"))
    for fold, score_start in enumerate(starts):
        cal_origins = pd.date_range(score_start - pd.Timedelta(days=3), periods=220, freq="15min")
        score_origins = pd.date_range(score_start, periods=96, freq="15min")
        fit_origins = pd.date_range(cal_origins.min() - pd.Timedelta(days=6), periods=400, freq="15min")
        all_origins = fit_origins.append(cal_origins).append(score_origins)
        positions = np.arange(len(all_origins))
        y = pd.Series(100 + 18*np.sin(positions/7) + 5*np.cos(positions/19), index=all_origins)
        target_time = pd.Series(all_origins + pd.Timedelta(hours=4), index=all_origins)
        contexts[(16, fold)] = {"fit": fit_origins, "cal": cal_origins, "score": score_origins,
                                "y": y, "target_time": target_time, "tau": 107.}
        for role, origins in (("cal", cal_origins), ("score", score_origins)):
            for i, origin in enumerate(origins):
                actual = float(y.loc[origin])
                base = {"horizon": 16, "fold": fold, "origin": origin,
                        "target_time": target_time.loc[origin], "y": actual, "tau": 107.,
                        "d2": bool(i % 6 == 0), "role": role,
                        "arm": "CAL" if role == "cal" else ("CONFIRM" if fold == 2 else "EXPLORE"),
                        "cal_provenance": "fit_only" if role == "cal" else "not_cal"}
                bpred = 99 + 8*np.sin(i/11 + fold)
                baseline_rows.append({**base, "model": "B5", "pred": bpred,
                                      "sigma": 13., "q10": np.nan, "q50": np.nan,
                                      "q90": np.nan, "q95": np.nan})
                cpred = 100 + 11*np.sin(i/10 + fold)
                candidate = {**base, "model": "F-candidate", "pred": cpred,
                             "sigma": np.nan}
                if candidate_quantiles:
                    candidate.update({"q10": cpred-15, "q50": cpred,
                                      "q90": cpred+15, "q95": cpred+20})
                candidate_rows.append(candidate)
    return SimpleNamespace(contexts=contexts), pd.DataFrame(candidate_rows), pd.DataFrame(baseline_rows)


def test_weekly_explore_uses_protected_phase_e_grid_and_same_cohort():
    prepared, candidate, baseline = _fixture()
    result = evaluate_candidate(prepared, candidate, baseline, n_boot=4)
    tables, fields = result["tables"], result["registry_fields"]
    assert {"E_brier", "E_prauc", "E_peak_coverage", "E_c10_22_recall",
            "E_c10_22_precision", "E_c10_22_FP"} <= set(fields)
    assert set(tables["alert_episode_metrics"].threshold) == {.01, .05, .10, .20, .30, .50}
    assert set(tables["alert_episode_metrics"].policy) == {"1/1", "2/2"}
    assert len(tables["score_probabilities"]) == 2*2*96
    assert set(tables["calibration_fit"].source) == {
        "B5_native_Kalman_Gaussian_sigma",
        "early_CAL_hour_x_day_type_standardized_residual_Gaussian"}
    assert tables["calibration_fit"].score_labels_used_in_fit.eq(False).all()
    assert (pd.to_datetime(tables["calibration_fit"].source_last_target) <
            pd.to_datetime(tables["calibration_fit"].calibrator_first_origin)).all()
    assert set(tables["comparison_vs_B5"].dataset) == {"D1", "D2"}
    assert set(tables["miss_decomposition"].miss_class) == {
        "hit", "late_hit", "decision_miss", "forecast_miss"}
    assert result["paired_episode_ci"]["n_boot"] == 1000
    assert result["paired_episode_ci"]["seed"] == 42
    assert "E_c10_22_recall_delta_ci_low_vs_B5" in fields
    assert "E_c10_22_F1_delta_ci_high_vs_B5" in fields
    assert result["manifest"]["paired_episode_ci"] == result["paired_episode_ci"]


def test_unselected_confirm_numbers_are_not_read():
    prepared, candidate, baseline = _fixture(include_confirm=True)
    candidate.loc[candidate.arm.eq("CONFIRM"), ["y", "pred", "tau"]] = np.nan
    baseline.loc[baseline.arm.eq("CONFIRM"), ["y", "pred", "tau"]] = np.nan
    result = evaluate_candidate(prepared, candidate, baseline, n_boot=2)
    assert result["manifest"]["week_folds"] == ["0", "1"]
    assert result["manifest"]["arm"] == "EXPLORE"


def test_full_horizon_input_reads_only_h16_cal_and_score_values():
    prepared, candidate, baseline = _fixture()
    expected = evaluate_candidate(prepared, candidate, baseline, n_boot=2)
    other_candidate = candidate.copy()
    other_baseline = baseline.copy()
    for frame in (other_candidate, other_baseline):
        frame["horizon"] = 4
        frame["model"] = "poisoned-other-horizon"
        for column in ("y", "pred", "tau", "sigma", "q10", "q50", "q90", "q95"):
            if column in frame:
                frame[column] = np.nan
        frame["origin"] = "not-a-date"
        frame["target_time"] = "not-a-date"
    mixed_candidate = pd.concat([other_candidate, candidate], ignore_index=True)
    mixed_baseline = pd.concat([other_baseline, baseline], ignore_index=True)
    actual = evaluate_candidate(prepared, mixed_candidate, mixed_baseline, n_boot=2)
    pd.testing.assert_frame_equal(actual["tables"]["score_probabilities"],
                                  expected["tables"]["score_probabilities"])
    assert actual["registry_fields"] == expected["registry_fields"]
    assert actual["manifest"]["candidate_input_sha256"] == expected["manifest"]["candidate_input_sha256"]
    with pytest.raises(ValueError, match="requires role, arm, horizon and model"):
        evaluate_candidate(prepared, candidate.drop(columns="horizon"), baseline, n_boot=2)


def test_selected_score_truth_never_changes_calibrator():
    prepared, candidate, _ = _fixture()
    context = prepared.contexts[(16, 0)]
    cal = candidate.loc[candidate.role.eq("cal") & candidate.fold.eq(0)].copy()
    score = candidate.loc[candidate.role.eq("score") & candidate.fold.eq(0)].copy()
    first, audit = _fit_week(cal, score, context, baseline=False)
    changed = score.copy()
    changed["y"] += 1000
    second, changed_audit = _fit_week(cal, changed, context, baseline=False)
    pd.testing.assert_series_equal(first.p_cal, second.p_cal)
    pd.testing.assert_series_equal(first.U95, second.U95)
    assert audit == changed_audit


def test_native_candidate_quantile_interpolation_and_standardized_conformal():
    prepared, candidate, baseline = _fixture(candidate_quantiles=True)
    result = evaluate_candidate(prepared, candidate, baseline, n_boot=2)
    fits = result["tables"]["calibration_fit"]
    assert fits.loc[fits.model.eq("F-candidate"), "source"].eq(
        "model_native_quantile_interpolation").all()
    assert fits.conformal_basis.eq("(CAL y - distribution location) / distribution scale").all()
    scored = result["tables"]["score_probabilities"]
    assert scored.p_raw.between(0, 1).all()
    assert np.isfinite(scored.U95).all()


def test_missing_native_b5_and_bad_cal_provenance_fail_closed():
    prepared, candidate, baseline = _fixture()
    baseline["sigma"] = np.nan
    with pytest.raises(ValueError, match="B5 requires its native Kalman"):
        evaluate_candidate(prepared, candidate, baseline, n_boot=2)
    _, candidate, baseline = _fixture()
    candidate.loc[candidate.role.eq("cal"), "cal_provenance"] = "in_sample"
    with pytest.raises(ValueError, match="No out-of-CAL-fit candidate predictions"):
        evaluate_candidate(prepared, candidate, baseline, n_boot=2)


def test_paired_key_or_truth_mismatch_rejected_before_phase_e_fit():
    prepared, candidate, baseline = _fixture()
    location = baseline.index[baseline.role.eq("score") & baseline.fold.eq(0)][0]
    baseline.loc[location, "y"] += 1
    with pytest.raises(ValueError, match="score y differs"):
        evaluate_candidate(prepared, candidate, baseline, n_boot=2)
    _, candidate, baseline = _fixture()
    location = candidate.index[candidate.role.eq("score") & candidate.fold.eq(0)][0]
    candidate.loc[location, "origin"] += pd.Timedelta(minutes=1)
    candidate.loc[location, "target_time"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="score forecast keys differ"):
        evaluate_candidate(prepared, candidate, baseline, n_boot=2)


def test_evidence_files_are_hashed_and_immutable(tmp_path):
    prepared, candidate, baseline = _fixture()
    destination = tmp_path / "F-candidate" / "EXPLORE"
    result = evaluate_candidate(prepared, candidate, baseline,
                                destination=destination, n_boot=2)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["registry_fields"]["E_brier"] is not None
    assert manifest["source_sha256"]["outputs/phase_e/code/alerts.py"]
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected
    changed = candidate.copy()
    changed.loc[changed.role.eq("score"), "pred"] += 1
    with pytest.raises(RuntimeError, match="Refusing to overwrite changed downstream artifact"):
        evaluate_candidate(prepared, changed, baseline, destination=destination, n_boot=2)


def test_cal_trained_ensemble_uses_only_later_oof_cal_for_candidate_and_b5():
    prepared, candidate, baseline = _fixture()
    candidate["model"] = "F7-ensemble"
    for fold in (0, 1):
        rows = candidate.index[candidate.role.eq("cal") & candidate.fold.eq(fold)]
        candidate.loc[rows[:100], "cal_provenance"] = "in_sample"
        candidate.loc[rows[100:], "cal_provenance"] = "chronological_oof"
    result = evaluate_candidate(prepared, candidate, baseline, n_boot=2)
    assert result["manifest"]["cal_cohort"] == "candidate chronological_oof matched to B5"
    assert result["manifest"]["cal_rows_before_provenance_filter"] == 440
    assert result["manifest"]["cal_rows_after_provenance_filter"] == 240
    fit = result["tables"]["calibration_fit"]
    assert fit.groupby("fold").calibrator_n.nunique().eq(1).all()
    assert fit.groupby("fold").residual_source_n.nunique().eq(1).all()


def _episode_rows():
    days = pd.date_range("2021-04-19", periods=8, freq="D")
    represented = pd.DataFrame({"fold": [0]*4 + [1]*4,
                                "target_time": days + pd.Timedelta(hours=12)})
    rows = []
    for model in ("new", "best"):
        for j in range(4):
            start = days[j] + pd.Timedelta(hours=12)
            hit = j < (4 if model == "new" else 3)
            rows.append({"model": model, "subset": "D1", "fold": "0",
                         "threshold": .10, "policy": "2/2", "status": "TP" if hit else "FN",
                         "actual_start": start, "actual_end": start + pd.Timedelta(minutes=15),
                         "alert_start": start-pd.Timedelta(hours=4) if hit else pd.NaT,
                         "onset_day": days[j]})
        fp_days = (4, 5) if model == "new" else (6,)
        for j in fp_days:
            rows.append({"model": model, "subset": "D1", "fold": "1",
                         "threshold": .10, "policy": "2/2", "status": "FP",
                         "actual_start": pd.NaT, "actual_end": pd.NaT,
                         "alert_start": days[j] + pd.Timedelta(hours=8),
                         "onset_day": days[j]})
    events = pd.DataFrame(rows)
    return events, represented


def test_paired_episode_ci_uses_common_day_weights_on_asymmetric_tp_fp_dates():
    events, represented = _episode_rows()
    pooled = events.copy()
    pooled["fold"] = "pooled"
    result = paired_episode_ci(pd.concat([events, pooled], ignore_index=True),
                               "new", "best", represented_days=represented)
    assert result["TP_FP_FN_candidate"] == [4, 2, 0]
    assert result["TP_FP_FN_baseline"] == [3, 1, 1]
    assert result["recall_candidate"] == 1.
    assert result["recall_baseline"] == .75
    assert result["recall_delta"] == .25
    assert result["recall_ci_status"] == "ok"
    assert result["F1_ci_status"] == "ok"
    assert result["recall_delta_ci_low"] <= .25 <= result["recall_delta_ci_high"]
    assert result["F1_delta_ci_low"] <= result["F1_delta"] <= result["F1_delta_ci_high"]
    assert result["fold_date_counts"] == {"0": 4, "1": 4}
    assert result["date_basis"] == "all_represented_score_target_calendar_days"
    assert "no rematching" in result["event_matching"]


def test_paired_episode_ci_returns_none_for_undefined_recall_or_f1():
    events, represented = _episode_rows()
    empty = events.iloc[0:0].copy()
    result = paired_episode_ci(empty, "new", "best", represented_days=represented)
    assert result["recall_delta"] is None
    assert result["recall_delta_ci_low"] is None
    assert result["F1_delta"] is None
    assert result["F1_delta_ci_high"] is None
    assert result["recall_ci_status"] == "unavailable_degenerate_point_metric"
    broken = events.copy()
    mask = broken.model.eq("best") & broken.status.eq("FN")
    broken.loc[mask, "actual_start"] += pd.Timedelta(days=1)
    broken.loc[mask, "actual_end"] += pd.Timedelta(days=1)
    broken.loc[mask, "onset_day"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="actual episode cohorts differ"):
        paired_episode_ci(broken, "new", "best", represented_days=represented)


def test_single_date_cannot_claim_a_paired_confidence_interval():
    events, represented = _episode_rows()
    one_day = represented.iloc[:1].copy()
    restricted = events.loc[events.onset_day.eq(one_day.target_time.iloc[0].normalize())]
    result = paired_episode_ci(restricted, "new", "best", represented_days=one_day)
    assert result["recall_delta"] == 0.
    assert result["recall_delta_ci_low"] is None
    assert result["recall_ci_status"] == "unavailable_insufficient_date_blocks"
