"""Synthetic Phase C data checks; no real source or historical result is read."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from phase_c.data import (
    CORE_FEATURES,
    HORIZONS,
    _core_features,
    _d2_score_mask,
    _daily_profiles,
    _prepare_history,
    build_contexts,
)


def _history(periods=3200):
    index = pd.date_range("2021-01-01 00:15", periods=periods, freq="15min", name="ts_end")
    slot = np.arange(periods) % 96
    # Repeated full-day profiles make D2 duplicate detection observable.
    power = 50 + 4 * np.sin(2 * np.pi * slot / 96) + slot / 100
    return pd.DataFrame({
        "power": power.astype(float),
        "headcount": np.full(periods, 8.0),
        "time_repaired": np.zeros(periods, dtype=bool),
    }, index=index)


def test_gap_requires_complete_four_zero_hour_and_missing_headcount():
    source = _history(288)
    source.iloc[8:12, source.columns.get_loc("power")] = 0
    source.iloc[8:12, source.columns.get_loc("headcount")] = np.nan
    source.iloc[20, source.columns.get_loc("power")] = 0
    source.iloc[20, source.columns.get_loc("headcount")] = np.nan
    source.iloc[28:32, source.columns.get_loc("power")] = 0
    source.iloc[40, source.columns.get_loc("time_repaired")] = True

    history, audit = _prepare_history(source)
    assert history["power"].iloc[8:12].isna().all()
    assert history["power_raw"].iloc[8:12].eq(0).all()
    assert history["power"].iloc[20] == 0
    assert history["power"].iloc[28:32].eq(0).all()
    assert np.isnan(history["power"].iloc[40])
    assert audit["gap_hours"] == 1
    assert audit["gap_intervals"] == 4
    assert audit["raw_zero_power_intervals"] == 9


def test_future_power_cannot_change_origin_features_and_boundary_is_rejected():
    history, _ = _prepare_history(_history(1000))
    origin = pd.DatetimeIndex([history.index[800]], name="origin")
    target = origin[0] + pd.Timedelta(hours=1)
    original, _ = _core_features(history, origin, horizon=4)
    changed = history.copy()
    changed.loc[target, "power"] += 9000
    future_changed, _ = _core_features(changed, origin, horizon=4)
    pd.testing.assert_frame_equal(original, future_changed)

    crossed = _history(96)
    crossed.index = pd.date_range("2021-08-09 09:45", periods=96, freq="15min", name="ts_end")
    with pytest.raises(ValueError, match="sealed"):
        _prepare_history(crossed)


def test_exact_15_core_features_full_grid_purge_and_causal_sequence():
    history, _ = _prepare_history(_history())
    contexts = build_contexts(history)
    assert len(contexts) == 13 * 3
    assert set(h for h, _ in contexts) == set(HORIZONS)

    for (horizon, fold), ctx in contexts.items():
        assert fold in (0, 1, 2)
        assert list(ctx["x"].columns) == list(CORE_FEATURES)
        assert ctx["x"].index.equals(ctx["y"].index)
        assert ctx["x"].index.equals(ctx["seq"].index)
        assert ctx["seq"].shape[1] == 96
        assert ctx["d2"].index.equals(ctx["x"].index)
        assert not ctx["d2"].loc[ctx["fit"]].any()
        assert not ctx["d2"].loc[ctx["stop"]].any()
        assert not ctx["d2"].loc[ctx["cal"]].any()
        assert ctx["target_time"].loc[ctx["fit"]].max() < ctx["stop"].min()
        assert ctx["target_time"].loc[ctx["stop"]].max() < ctx["cal"].min()
        assert ctx["target_time"].loc[ctx["cal"]].max() < ctx["score"].min()
        assert ctx["tau"] == pytest.approx(float(np.quantile(ctx["y"].loc[ctx["fit"]], .95)))
        for feature, timestamps in ctx["provenance"].items():
            assert timestamps.index.equals(ctx["x"].index)
            if feature in CORE_FEATURES[:6]:
                assert timestamps.isna().all()
            else:
                assert (timestamps <= timestamps.index).all()

    ctx = contexts[(4, 0)]
    origin = ctx["score"][10]
    target = ctx["target_time"].loc[origin]
    row = ctx["x"].loc[origin]
    sequence = history.loc[origin - pd.Timedelta(minutes=15 * 95):origin, "power"].to_numpy()
    np.testing.assert_array_equal(ctx["seq"].loc[origin].to_numpy(), sequence)
    assert ctx["y"].loc[origin] == history.loc[target, "power"]
    assert row["current"] == history.loc[origin, "power"]
    assert row["lag4"] == history.loc[origin - pd.Timedelta(hours=1), "power"]
    assert row["slot1d"] == history.loc[target - pd.Timedelta(days=1), "power"]
    assert row["slot7d"] == history.loc[target - pd.Timedelta(days=7), "power"]
    assert row["r4_mean"] == pytest.approx(float(np.mean(sequence[-4:])))
    assert row["r4_max"] == pytest.approx(float(np.max(sequence[-4:])))
    assert row["r4_std"] == pytest.approx(float(np.std(sequence[-4:], ddof=1)))
    assert row["r16_max"] == pytest.approx(float(np.max(sequence[-16:])))
    assert row["r96_max"] == pytest.approx(float(np.max(sequence)))
    hour_float = target.hour + target.minute / 60
    sunday_dow = (target.dayofweek + 1) % 7
    assert row["hour_sin"] == pytest.approx(np.sin(2 * np.pi * hour_float / 24))
    assert row["dow_cos"] == pytest.approx(np.cos(2 * np.pi * sunday_dow / 7))


def test_d2_hashes_only_complete_interval_days_and_never_deletes_fit():
    source = _history(96 * 4)
    source.iloc[96 * 2:96 * 3, source.columns.get_loc("power")] += 3
    history, _ = _prepare_history(source)
    profiles, summary = _daily_profiles(history)
    assert summary == {"complete_profile_days": 4, "incomplete_profile_days": 0}
    assert profiles.iloc[0]["sha256"] == profiles.iloc[1]["sha256"]
    assert profiles.iloc[2]["sha256"] != profiles.iloc[0]["sha256"]
    expected_hash = hashlib.sha256(np.asarray(
        history["power"].iloc[:96].to_numpy(), dtype="<f8"
    ).tobytes()).hexdigest()
    assert profiles.iloc[0]["sha256"] == expected_hash

    origins = pd.DatetimeIndex([history.index[95], history.index[191], history.index[250]], name="origin")
    target_time = pd.Series([history.index[95], history.index[191], history.index[250]], index=origins)
    fit = origins[:2]
    score = origins[2:]
    mask, d2_summary = _d2_score_mask(profiles, target_time, fit, score)
    assert mask.index.equals(origins)
    assert not mask.loc[fit].any()
    assert bool(mask.loc[score[0]])
    assert d2_summary["d2_fit_profile_hashes"] == 1
    assert d2_summary["d2_score_novel"] == 1
