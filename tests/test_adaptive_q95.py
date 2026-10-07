"""Causality and ordering checks for the preregistered adaptive q95 replay."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.adaptive_q95 import replay_adaptive_q95


def _seed(n=30):
    targets = pd.date_range("2021-01-01 00:15", periods=n, freq="15min")
    return pd.DataFrame({"target_time": targets, "y": np.full(n, 101.),
                         "q95": np.full(n, 100.)})


def _issued(horizon=96):
    origins = pd.date_range("2021-01-02 00:00", periods=4, freq="15min")
    return pd.DataFrame({"origin": origins,
                         "target_time": origins + pd.Timedelta(minutes=15*horizon),
                         "y": [300., 99., 101., 102.], "q95": [100.]*4,
                         "q90_cal": [90.]*4, "q95_cal": [110.]*4,
                         "q975_cal": [120.]*4})


def test_h96_future_labels_cannot_change_earlier_issued_quantiles():
    source = _issued()
    first = replay_adaptive_q95(_seed(), source, gamma=.05)
    altered = source.copy()
    altered.loc[0, "y"] = -10000.
    later = replay_adaptive_q95(_seed(), altered, gamma=.05)
    assert np.array_equal(first.q95_cal_online, later.q95_cal_online)
    assert np.array_equal(first.beta_at_issue, later.beta_at_issue)
    assert (pd.to_datetime(first.latest_observation) <= first.origin).all()
    assert first.q95_cal_online.ge(first.q90_cal).all()
    assert first.q975_cal_online.ge(first.q95_cal_online).all()


def test_matured_label_updates_next_origin_using_issued_q95_hit():
    issued = _issued(horizon=1)
    issued.loc[0, "y"] = 300.
    first = replay_adaptive_q95(_seed(), issued, gamma=.05)
    changed = issued.copy()
    changed.loc[0, "y"] = 0.
    second = replay_adaptive_q95(_seed(), changed, gamma=.05)
    assert first.q95_cal_online.iloc[0] == second.q95_cal_online.iloc[0]
    assert first.beta_at_issue.iloc[0] == pytest.approx(.95)
    assert first.beta_at_issue.iloc[1] == pytest.approx(.9975)
    assert second.beta_at_issue.iloc[1] == pytest.approx(.9475)
    assert first.latest_observation.iloc[1] == first.target_time.iloc[0]


def test_initial_calibration_and_score_separation_and_fallback():
    source = _issued(horizon=4)
    fallback = replay_adaptive_q95(_seed(29), source, gamma=None)
    assert fallback.fallback.iloc[0]
    assert fallback.q95_cal_online.iloc[0] == 110.
    ready = replay_adaptive_q95(_seed(30), source, gamma=None)
    assert not ready.fallback.iloc[0]
    changed = source.copy()
    changed.y = 10000.
    unmodified = replay_adaptive_q95(_seed(30), changed, gamma=None)
    assert ready.q95_cal_online.iloc[0] == unmodified.q95_cal_online.iloc[0]


def test_reject_initial_labels_after_first_forecast_origin():
    seed = _seed(30)
    seed.loc[0, "target_time"] = pd.Timestamp("2021-01-03")
    with pytest.raises(ValueError, match="not all known"):
        replay_adaptive_q95(seed, _issued(), gamma=None)


def test_final_replay_scope_requires_explicit_external_gate():
    source = _issued(horizon=1)
    source["origin"] = source.origin + pd.Timedelta(days=221)
    source["target_time"] = source.target_time + pd.Timedelta(days=221)
    with pytest.raises(ValueError, match="sealed test"):
        replay_adaptive_q95(_seed(), source, gamma=None)
    # The pure causal core can later be replayed by the separately gated
    # human-approved final path; this test never invokes that path.
    replayed = replay_adaptive_q95(_seed(), source, gamma=None, sealed_only=False)
    assert (pd.to_datetime(replayed.latest_observation) <= replayed.origin).all()


def test_sealed_scope_checks_initial_calibration_targets_too():
    seed = _seed()
    seed["target_time"] = seed.target_time + pd.Timedelta(days=221)
    issued = _issued()
    issued["origin"] = issued.origin + pd.Timedelta(days=222)
    issued["target_time"] = issued.target_time + pd.Timedelta(days=222)
    with pytest.raises(ValueError, match="sealed test"):
        replay_adaptive_q95(seed, issued, gamma=None)


def test_rejects_non_fifo_target_order_or_mixed_horizons():
    source = _issued(horizon=4)
    reordered = source.copy()
    reordered.loc[1, "target_time"] = reordered.target_time.iloc[0]
    with pytest.raises(ValueError, match="strictly chronological"):
        replay_adaptive_q95(_seed(), reordered, gamma=None)
    mixed = source.copy()
    mixed.loc[1, "target_time"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="one fixed forecast horizon"):
        replay_adaptive_q95(_seed(), mixed, gamma=None)
