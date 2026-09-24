"""Safety and preregistered-ranking checks for the optional threshold audit."""

import numpy as np
import pandas as pd

from scripts.peak_sensitivity_0924 import (
    _assert_q95_parity, _day_metric_ci, _fixed_candidates, _reference_ranking,
)


def test_fixed_candidates_use_only_original_selection_and_family_representatives():
    selected = {"by_horizon": {"4": {"point_model": "lgbm_no_holiday_weight_2",
                                     "cbl": "c2a_max_4_5_adjusted",
                                     "persistence": "p1_latest", "seasonal": "s2_week"},
                               "96": {"point_model": "c3_holiday_hybrid",
                                      "cbl": "c3_holiday_hybrid",
                                      "persistence": "p1_latest", "seasonal": "s2_week"}}}
    assert _fixed_candidates(selected, 4) == ["lgbm_no_holiday_weight_2", "c2a_max_4_5_adjusted",
                                               "p1_latest", "s2_week"]
    assert _fixed_candidates(selected, 96) == ["c3_holiday_hybrid", "p1_latest", "s2_week"]


def test_date_block_intervals_respect_peak_counts_and_episode_matches():
    times = pd.to_datetime(["2021-01-01 12:00", "2021-01-01 12:15",
                            "2021-01-02 12:00", "2021-01-02 12:15"])
    frame = pd.DataFrame({"target_time": times, "fold": 0, "horizon": 4,
                          "y": [12., 0., 12., 0.], "pred": [10., 0., 10., 0.],
                          "tau": 10., "alert": [True, False, True, False]})
    ci = _day_metric_ci(frame, n=1000, seed=42)
    assert ci["peak_mae_ci95"] == [2., 2.]
    assert ci["episode_f1_ci95"] == [1., 1.]
    assert ci["false_alarm_ci95"] == [0., 0.]


def test_empty_episode_blocks_use_original_zero_f1_convention():
    frame = pd.DataFrame({"target_time": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                          "fold": [0, 0], "horizon": [4, 4], "y": [0., 0.],
                          "pred": [0., 0.], "tau": [10., 10.], "alert": [False, False]})
    ci = _day_metric_ci(frame, n=1000, seed=42)
    assert ci["episode_f1_ci95"] == [0., 0.]
    assert all(np.isnan(value) for value in ci["peak_mae_ci95"])


def test_q95_parity_requires_exact_alerts_and_numeric_predictions():
    times = pd.date_range("2021-01-01 12:00", periods=2, freq="15min")
    frame = pd.DataFrame({"origin": times - pd.Timedelta(hours=1), "target_time": times,
                          "horizon": 4, "fold": 0, "model": "p1_latest", "y": [12., 0.],
                          "pred": [11., 1.], "tau": 10., "alert_cutoff": 10.5,
                          "alert": [True, False]})
    assert _assert_q95_parity([frame.copy()], frame.copy(), 4, 0)[0]["alert_exact"]
    changed = frame.copy()
    changed.loc[0, "pred"] = 11.1
    import pytest
    with pytest.raises(AssertionError, match="pred changed"):
        _assert_q95_parity([changed], frame, 4, 0)
    changed = frame.copy()
    changed.loc[0, "alert"] = False
    with pytest.raises(AssertionError, match="alert parity"):
        _assert_q95_parity([changed], frame, 4, 0)


def test_reference_ranking_uses_paired_peak_error_on_fixed_rows():
    times = pd.date_range("2021-01-01 12:00", periods=20, freq="1D")
    common = pd.DataFrame({"target_time": times, "origin": times - pd.Timedelta(hours=1),
                           "horizon": 4, "fold": 0, "y": 20., "tau": 10., "alert": True})
    weak, strong = common.copy(), common.copy()
    weak["model"] = "weak"; weak["pred"] = 10.
    strong["model"] = "strong"; strong["pred"] = 20.
    scores = pd.DataFrame({"model": ["weak", "strong"],
                           "episode_f1_ci95_low": [.5, .5], "episode_f1_ci95_high": [.6, .6],
                           "false_alarm_ci95_low": [0., 0.], "false_alarm_ci95_high": [0., 0.]})
    winner, evidence = _reference_ranking(pd.concat([weak, strong]), scores, ["weak", "strong"], 42)
    assert winner == "strong"
    assert evidence[0]["decision"] == "challenger_peak_mae_ci"
    assert np.isclose(evidence[0]["peak_mae_gain"]["estimate"], 10.)
