"""F9-4 weekly refit uses arrived labels only, on synthetic sealed history."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from phase_f import walkforward as wf
from phase_f.registry import config_hash


def _fixture():
    index = pd.date_range("2021-01-01", periods=5000, freq="15min")
    slot = np.arange(len(index))
    power = 75 + 5 * np.sin(2 * np.pi * slot / 96) + np.cos(2 * np.pi * slot / 672)
    history = pd.DataFrame({"power": power}, index=index)
    score = pd.DatetimeIndex(index[3000:4800:96])
    h = 4
    context = {"y": history.power.shift(-h),
               "target_time": pd.Series(index + timedelta(minutes=15 * h), index=index),
               "tau": 80.0,
               "d2": pd.Series(False, index=score)}
    cfg = {"horizon": h, "target": "direct", "groups": (),
           "model_params": {"alpha": 30.}}
    lock = {"primary_candidate": "F2-ridge", "confirm_complete": True,
            "selection_lock_sha256": "synthetic-selection", "confirm_once_sha256": "synthetic-confirm",
            "frozen_config_hash": config_hash(cfg),
            "score_origin_sha256": config_hash([str(item) for item in score])}
    return history, context, score, cfg, lock


def test_weekly_refits_primary_and_b5_on_arrived_labels_only(monkeypatch):
    history, context, score, cfg, lock = _fixture()
    b5_fit_ends = []

    def fake_fit(power, selected):
        b5_fit_ends.append(selected.max())
        return {"n_fit_updates": len(selected), "fit_end": selected.max()}

    def fake_predict(bundle, power, origins, horizon):
        assert bundle["fit_end"] < origins.min()
        return power.reindex(origins).to_numpy(float)

    monkeypatch.setattr(wf, "fit_kalman", fake_fit)
    monkeypatch.setattr(wf, "predict_kalman", fake_predict)
    result = wf.simulate_weekly_refit(history, context, score, candidate="F2-ridge",
                                      kind="ridge", frozen_config=cfg, lock=lock, horizon=4)
    pred, updates = result["predictions"], result["updates"]
    assert len(pred) == len(score)
    assert pred.origin.tolist() == list(score)
    assert len(updates) == 3
    assert (updates.max_train_target_time <= updates.update_time).all()
    assert (updates.max_train_origin < updates.update_time).all()
    assert updates.train_n.is_monotonic_increasing
    assert updates.arrived_score_labels_used_at_update.iloc[0] == 0
    assert updates.arrived_score_labels_used_at_update.iloc[-1] > 0
    assert not updates.unarrived_score_labels_used_at_update.any()
    assert len(b5_fit_ends) == len(updates)
    assert np.isfinite(pred[["candidate_pred", "b5_pred", "y"]].to_numpy(float)).all()


def test_walkforward_is_postconfirm_and_locked_or_explicitly_unsupported():
    history, context, score, cfg, lock = _fixture()
    lock["confirm_complete"] = False
    with pytest.raises(ValueError, match="completed one-time CONFIRM"):
        wf.simulate_weekly_refit(history, context, score, candidate="F2-ridge",
                                 kind="ridge", frozen_config=cfg, lock=lock, horizon=4)
    lock["confirm_complete"] = True
    with pytest.raises(ValueError, match="config differs"):
        wf.simulate_weekly_refit(history, context, score, candidate="F2-ridge",
                                 kind="ridge", frozen_config={**cfg, "model_params": {"alpha": 1.}},
                                 lock=lock, horizon=4)
    with pytest.raises(NotImplementedError, match="independent tabular"):
        wf.simulate_weekly_refit(history, context, score, candidate="F2-ridge",
                                 kind="two_stage", frozen_config=cfg, lock=lock, horizon=4)


def test_score_origin_lock_and_boundary_fail_closed():
    history, context, score, cfg, lock = _fixture()
    with pytest.raises(ValueError, match="fixed diagnostic lock"):
        wf.simulate_weekly_refit(history, context, score[:-1], candidate="F2-ridge",
                                 kind="ridge", frozen_config=cfg, lock=lock, horizon=4)
    future = pd.concat([history, pd.DataFrame({"power": [1.]},
                                              index=[pd.Timestamp("2021-08-09 09:45")])])
    with pytest.raises(ValueError, match="sealed development prefix"):
        wf.simulate_weekly_refit(future, context, score, candidate="F2-ridge",
                                 kind="ridge", frozen_config=cfg, lock=lock, horizon=4)
