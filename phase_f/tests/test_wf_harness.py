"""Revised weekly evaluation geometry on synthetic development-only history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_f.harness import KEY, make_frame
from phase_f.wf_harness import CAL_SLOTS, WeeklyPrepared, _score_keys, build_weekly_contexts


@pytest.fixture(scope="module")
def weekly_case():
    index = pd.date_range("2021-01-01 00:15", "2021-04-25 23:45", freq="15min")
    slot = np.arange(len(index))
    history = pd.DataFrame({
        "power": 100 + 7 * np.sin(2 * np.pi * slot / 96)
                 + 2 * np.cos(2 * np.pi * slot / 672),
        "headcount": np.full(len(index), 3.0),
        "time_repaired": np.zeros(len(index), dtype=bool),
    }, index=index)
    contexts = build_weekly_contexts(
        history, boundary=pd.Timestamp("2021-04-26"), horizons=(4, 16))
    return history, contexts


def test_full_target_weeks_phase_c_ratio_and_earliest_origin_embargo(weekly_case):
    _, contexts = weekly_case
    assert set(contexts) == {(4, 0), (16, 0), (4, 1), (16, 1)}
    for (h, fold), context in contexts.items():
        summary = context["summary"]
        assert summary["week_start"] == str(pd.Timestamp("2021-04-12") + pd.Timedelta(days=7 * fold))
        assert len(context["cal"]) == CAL_SLOTS == 362
        assert len(context["score"]) == 672
        assert summary["fit_count"] > summary["stop_count"] > 0
        assert len(context["fit"]) / (len(context["fit"]) + len(context["stop"])) == pytest.approx(.8, abs=.002)
        assert context["target_time"].loc[context["fit"]].max() < context["stop"].min()
        assert context["target_time"].loc[context["stop"]].max() < context["cal"].min()
        assert context["target_time"].loc[context["cal"]].max() < context["score"].min()
        score_target = pd.DatetimeIndex(context["target_time"].loc[context["score"]])
        assert score_target.min() == pd.Timestamp(summary["week_start"])
        assert score_target.max() == pd.Timestamp(summary["week_end_exclusive"]) - pd.Timedelta(minutes=15)
        assert np.isfinite(context["x"].loc[context["score"]].to_numpy(float)).all()
        assert np.isfinite(context["seq"].loc[context["score"]].to_numpy(float)).all()
        assert context["tau"] == pytest.approx(float(np.quantile(context["y"].loc[context["fit"]], .95)))
        assert context["d2"].loc[context["score"]].dtype == bool
        if h == 16:
            assert context["score"].min() == pd.Timestamp(summary["week_start"]) - pd.Timedelta(hours=4)


def test_even_odd_week_arms_and_prediction_contract(weekly_case):
    _, contexts = weekly_case
    keys = _score_keys(contexts)
    assert keys.groupby("fold").arm.first().to_dict() == {0: "CONFIRM", 1: "EXPLORE"}
    assert len(keys) == 2 * 2 * 672
    prepared = object.__new__(WeeklyPrepared)
    prepared.contexts = contexts
    prepared.keys = keys
    prepared._key_index = pd.MultiIndex.from_frame(keys[KEY])
    frames = []
    for (h, fold), context in contexts.items():
        for role in ("cal", "score"):
            origins = context[role]
            frames.append(make_frame(context, origins, h, fold,
                                     context["y"].loc[origins], "synthetic", role))
    frame = pd.concat(frames, ignore_index=True)
    assert prepared.validate_predictions(frame) is frame
    assert prepared.origins(16, 0, "fit").equals(contexts[(16, 0)]["fit"])
    assert prepared.origins(16, 0, "stop").equals(contexts[(16, 0)]["stop"])
    assert prepared.origins(16, 0, "cal").equals(contexts[(16, 0)]["cal"])
    assert prepared.origins(16, 0, "score").equals(contexts[(16, 0)]["score"])

    changed = frame.copy()
    changed.loc[changed.role.eq("cal").idxmax(), "target_time"] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="target times changed"):
        prepared.validate_predictions(changed)
    changed = frame.copy()
    changed.loc[changed.role.eq("score").idxmax(), "y"] += 1
    with pytest.raises(ValueError, match="truth changed"):
        prepared.validate_predictions(changed)


def test_fit_history_requirement_moves_first_week(weekly_case):
    history, _ = weekly_case
    contexts = build_weekly_contexts(
        history, first_week=pd.Timestamp("2021-02-01"),
        boundary=pd.Timestamp("2021-04-26"), min_fit_weeks=8, horizons=(16,))
    first = contexts[(16, 0)]["summary"]
    assert pd.Timestamp(first["week_start"]) > pd.Timestamp("2021-02-01")
    assert first["skipped_candidate_weeks"]
    assert "fit span" in first["skipped_candidate_weeks"][0]["reason"]
    for context in contexts.values():
        fit = context["fit"]
        targets = context["target_time"].loc[fit]
        assert targets.max() - targets.min() >= pd.Timedelta(weeks=8)


def test_future_boundary_and_incomplete_week_fail_closed(weekly_case):
    history, _ = weekly_case
    with pytest.raises(ValueError, match="sealed development boundary"):
        build_weekly_contexts(history, boundary=pd.Timestamp("2021-08-10"), horizons=(4,))
    with pytest.raises(ValueError, match="No complete target-calendar"):
        build_weekly_contexts(history, first_week=pd.Timestamp("2021-04-26"),
                              boundary=pd.Timestamp("2021-04-26"), horizons=(4,))
