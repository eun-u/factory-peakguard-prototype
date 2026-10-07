"""Synthetic only: locked alert policy, matching, and normalized decision QA."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).resolve().parents[1] / "code" / "alerts.py"
SPEC = importlib.util.spec_from_file_location("phase_e_alerts", MODULE)
assert SPEC and SPEC.loader
alerts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alerts)


def make_score(*, minutes, p, peaks=(), fold="f1", d2=(), start="2021-01-01 04:00"):
    target = pd.Timestamp(start) + pd.to_timedelta(minutes, unit="m")
    n = len(minutes)
    return pd.DataFrame({"fold": fold, "origin": target - pd.Timedelta(hours=4),
                         "target_time": target, "y": [20.0 if i in peaks else 0.0 for i in range(n)],
                         "tau": 10.0, "p_cal": p,
                         "is_d2_novel_profile": [i in d2 for i in range(n)],
                         "horizon": 16})


def take(table, *, subset="D1", fold="f1", threshold=.5, policy=None):
    sel = table.loc[table.subset.eq(subset) & table.fold.eq(fold) & table.threshold.eq(threshold)]
    if policy is not None:
        sel = sel.loc[sel.policy.eq(policy)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_fixed_thresholds_and_gap_reset():
    score = make_score(minutes=[0, 15, 45, 60], p=[.7, .7, .7, .7], peaks=(1, 3))
    out = alerts.evaluate_alerts(score, n_boot=10)
    assert tuple(sorted(out["decision_value_curve"].threshold.unique())) == alerts.THRESHOLDS
    assert set(out["alert_position_metrics"].policy) == {"1/1", "2/2"}
    row = take(out["alert_position_metrics"], policy="2/2")
    assert (row.TP, row.FP, row.FN, row.TN) == (2, 0, 0, 2)


def test_confirmation_stops_at_fold_boundary():
    first = make_score(minutes=[0], p=[.9], fold="f1")
    second = make_score(minutes=[15], p=[.9], fold="f2")
    out = alerts.evaluate_alerts(pd.concat([first, second], ignore_index=True), n_boot=0)
    assert take(out["alert_position_metrics"], fold="f2", policy="2/2").FP == 0


def test_one_to_one_straddle_and_diagnostic_decision_miss():
    # One continuous alert overlaps two distinct actual episodes.  It cannot
    # be counted twice, and the unmatched actual carried high risk information.
    score = make_score(minutes=[0, 15, 30], p=[.9, .9, .9], peaks=(0, 2))
    out = alerts.evaluate_alerts(score, n_boot=0)
    row = take(out["alert_episode_metrics"], threshold=.1, policy="1/1")
    assert (row.actual_peak_episodes, row.detected_peak_episodes,
            row.missed_peak_episodes, row.false_alert_episodes) == (2, 1, 1, 0)
    miss = out["miss_decomposition"]
    assert take(miss.loc[miss.miss_class.eq("decision_miss")], threshold=.1, policy="1/1").episodes == 1


def test_late_hit_has_negative_direct_lead():
    # A long actual episode extends beyond the 4h issue horizon; the first
    # alert targets its final point and is issued after the actual onset.
    minutes = list(range(0, 21 * 15, 15))
    score = make_score(minutes=minutes, p=[0.0] * 20 + [.9], peaks=range(21))
    out = alerts.evaluate_alerts(score, n_boot=0)
    row = take(out["alert_episode_metrics"], policy="1/1")
    assert row.direct_lead_min_minutes == -60.0
    events = out["alert_episode_events"]
    matched = events.loc[events.threshold.eq(.5) & events.policy.eq("1/1") & events.status.eq("TP")]
    assert len(matched) == 1
    assert matched.iloc[0].miss_class == "late_hit"


def test_d2_uses_confirmation_frozen_on_full_d1():
    score = make_score(minutes=[0, 15, 30], p=[.9, .9, 0.0], peaks=(1,), d2=(1,))
    out = alerts.evaluate_alerts(score, n_boot=0)
    d2 = take(out["alert_position_metrics"], subset="D2", policy="2/2")
    assert (d2.TP, d2.FN) == (1, 0)


def test_uncertainty_fields_do_not_trigger_alert_and_forecast_miss():
    score = make_score(minutes=[0, 15], p=[0.0, 0.0], peaks=(0,))
    score["U95"] = 999.0
    score["q95_raw"] = 999.0
    out = alerts.evaluate_alerts(score, n_boot=0)
    assert take(out["alert_position_metrics"], policy="1/1").TP == 0
    miss = out["miss_decomposition"]
    assert take(miss.loc[miss.miss_class.eq("forecast_miss")], policy="1/1").episodes == 1


def test_normalized_cost_loss_exact_and_ci_fields():
    score = make_score(minutes=[0, 15], p=[.9, 0.0], peaks=(0,))
    out = alerts.evaluate_alerts(score, n_boot=20, seed=42)
    row = take(out["decision_value_curve"])
    assert row.actions == 1
    assert (row.TP, row.FP, row.FN, row.TN) == (1, 0, 0, 1)
    assert row.E_model == .5
    assert row.E_no == row.E_all == 1.0
    assert row.E_perf == .5
    assert row.relative_value == 1.0
    episode = take(out["alert_episode_metrics"], policy="1/1")
    assert episode.represented_target_calendar_days == 1
    assert np.isfinite(episode.episode_recall_ci_low)
    assert np.isfinite(episode.direct_lead_median_minutes_ci_low)


def test_invalid_probability_and_mismatched_label_rejected():
    score = make_score(minutes=[0], p=[1.01])
    with pytest.raises(ValueError, match="p_cal"):
        alerts.evaluate_alerts(score, n_boot=0)
    score = make_score(minutes=[0], p=[.2])
    score["is_peak"] = True
    with pytest.raises(ValueError, match="is_peak"):
        alerts.evaluate_alerts(score, n_boot=0)
