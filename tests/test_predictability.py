"""Synthetic leakage and boundary checks for Phase 1 signal analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.predictability import (
    QUARTER, _fold_auc_summary, build_a2_features, causal_restart_events,
    detect_restart_events, discover_restart_slots, fit_restart_thresholds,
)


def _history(start="2021-03-01 00:00", days=18, *, unit="us"):
    idx = pd.date_range(start, periods=days * 96, freq="15min").as_unit(unit)
    power = 25 + 5 * np.sin(np.arange(len(idx)) * 2 * np.pi / 96)
    power[(idx.hour == 8) & (idx.minute.isin([0, 15]))] = 2
    power[(idx.hour == 8) & (idx.minute == 30)] = 55
    return pd.DataFrame({"power": power, "time_repaired": False}, index=idx)


def test_restart_threshold_and_event_timestamps_work_for_microsecond_index():
    history = _history(unit="us")
    threshold = fit_restart_thresholds(history, history.index[-1])
    assert threshold["rise"] > 0
    events = detect_restart_events(history, threshold)
    assert len(events) > 0
    assert set(events.index.minute) == {30}
    assert (pd.to_datetime(events.threshold_fit_end) >= events.index).all()  # training fit descriptor
    score = detect_restart_events(history, threshold, start=history.index[96 * 10],
                                  end=history.index[96 * 12])
    assert score.index.isin(events.index).all()


def test_restart_run_cannot_cross_missing_repair_or_calendar_date():
    idx = pd.date_range("2021-03-01 23:30", periods=9, freq="15min").as_unit("us")
    history = pd.DataFrame({"power": [1., 1., 1., 1., 1., 1., 11., 2., 12.],
                            "time_repaired": [False] * 9}, index=idx)
    threshold = {"low": 2., "rise": 8., "fit_end": str(idx[0])}
    events = detect_restart_events(history, threshold)
    assert idx[6] in events.index
    assert idx[8] not in events.index
    broken = history.copy()
    broken.loc[idx[5], "time_repaired"] = True
    assert idx[6] not in detect_restart_events(broken, threshold).index


def test_train_slot_discovery_never_uses_score_events():
    history = _history()
    fit_end = history.index[96 * 10 - 1]
    threshold = fit_restart_thresholds(history, fit_end)
    events = detect_restart_events(history, threshold)
    first = discover_restart_slots(events, history, fit_end)
    manipulated = events.copy()
    manipulated.loc[manipulated.index > fit_end, "slot"] = 75
    second = discover_restart_slots(manipulated, history, fit_end)
    pd.testing.assert_frame_equal(first, second)


def test_a2_features_use_no_future_measured_value_even_for_train_rows():
    history = _history(days=40)
    train_origin = history.index[96 * 20 + 40]
    score_origin = history.index[96 * 36 + 40]
    fit_end = history.index[96 * 30 - 1]
    origins = pd.DatetimeIndex([train_origin, score_origin])
    events = causal_restart_events(history, origins, fit_end)
    for horizon in (16, 96):
        features, used = build_a2_features(history, origins, horizon, tau=40.,
                                           fit_end=fit_end, restart_events=events)
        for origin, row in used.iterrows():
            assert all(pd.isna(value) or value <= origin for value in row)
        assert used.loc[train_origin, "slot_rate7"] == train_origin
        assert used.loc[score_origin, "slot_rate7"] == fit_end
        early_perturbation = history.copy()
        early_perturbation.loc[early_perturbation.index > train_origin, "power"] = 9000.
        early_events = causal_restart_events(early_perturbation, origins[:1], fit_end)
        unchanged_train, _ = build_a2_features(early_perturbation, origins[:1], horizon,
                                               tau=40., fit_end=fit_end,
                                               restart_events=early_events)
        pd.testing.assert_series_equal(features.loc[train_origin], unchanged_train.loc[train_origin])
        modified = history.copy()
        modified.loc[modified.index > score_origin, "power"] = 9000.
        changed, _ = build_a2_features(modified, origins, horizon, tau=40.,
                                        fit_end=fit_end, restart_events=events)
        # Changed future power can include the train row's future but not the
        # score row's past. Check the score row against a perturbation after it.
        pd.testing.assert_series_equal(features.loc[score_origin], changed.loc[score_origin])


def test_a2_rejects_restart_threshold_fitted_after_training_origin():
    history = _history(days=10)
    origin = pd.DatetimeIndex([history.index[96 * 5 + 40]])
    later_fit = history.index[96 * 8]
    threshold = fit_restart_thresholds(history, later_fit)
    events = detect_restart_events(history, threshold)
    with pytest.raises(AssertionError, match="after its origin"):
        build_a2_features(history, origin, 16, tau=40., fit_end=later_fit,
                          restart_events=events)


def test_auc_bootstrap_requires_both_classes_in_every_fold():
    rows = []
    for fold in range(3):
        for i in range(12):
            rows.append({"fold": fold, "target_time": pd.Timestamp("2021-04-01")
                         + pd.Timedelta(days=fold * 15 + i),
                         "label": int(i >= (7 if fold == 2 else 6)),
                         "probability": .8 if i >= 6 else .2})
    result, _ = _fold_auc_summary(pd.DataFrame(rows), null_value=.6, n_boot=20)
    assert result["eligible"] is True
    frame = pd.DataFrame(rows)
    frame.loc[frame.fold.eq(2) & frame.label.eq(1), "label"] = 0
    invalid, _ = _fold_auc_summary(frame, null_value=.6, n_boot=20)
    assert invalid["eligible"] is False
