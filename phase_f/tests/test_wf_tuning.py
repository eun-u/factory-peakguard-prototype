"""Stop-only Optuna bookkeeping for the weekly evaluation geometry."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import optuna
import pytest

from phase_f import wf_tuning


class Prepared:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.split_lock = {"lock_sha256": "synthetic-week-lock"}
        self.seal = {"raw_sha256": "synthetic-raw-seal"}
        score = pd.DatetimeIndex([pd.Timestamp("2021-04-19 09:00")])
        stop = pd.DatetimeIndex([pd.Timestamp("2021-04-18 08:00")])
        self.contexts = {(4, 0): {"score": score, "stop": stop,
                                  "target_time": pd.Series(score + pd.Timedelta(hours=1), index=score)}}

    def origins(self, h, fold, role):
        return self.contexts[(h, fold)][role]


PARENT = {"id": "F3-parent", "adapter": "regression", "kind": "lightgbm",
          "target": "direct", "groups": ("slot_1_7d", "rolling")}


@pytest.fixture
def small_minimum(monkeypatch):
    monkeypatch.setattr(wf_tuning, "MIN_COMPLETED", 2)
    monkeypatch.setattr(wf_tuning, "ROUND_COMPLETED", 1)


def _ready(spec, *, stop=8.):
    return {"status": "prediction_ready", "stop_MAE": stop,
            "stop_cells": 5, "n_seeds": 5, "prediction_sha": "a" * 64}


def test_tpe_counts_only_full_stop_cells_and_extends_existing_study(tmp_path, small_minimum):
    prepared = Prepared(Path(__file__).resolve().parents[2], tmp_path)
    calls = []

    def execute(spec, *, score):
        assert score is False
        calls.append(spec)
        assert spec["adapter"] == "regression" and spec["force_seed_repeats"] is True
        if len(calls) == 1:
            return {**_ready(spec), "stop_cells": 4}  # invalid, cannot count
        return _ready(spec, stop=8 - len(calls) / 10)

    state = wf_tuning.run_search(prepared, "lightgbm", execute, parent_spec=PARENT,
                                 target_completed=2)
    assert state["status"] == "complete" and state["completed_trials"] == 2
    assert len(calls) == 3 and state["expected_stop_cells"] == 5
    assert "stop" in state["objective"].lower() and "score" not in state["objective"].lower()
    ledger = pd.read_csv(tmp_path / "logs/tuning/wf_tpe_lightgbm_trials.csv")
    assert ledger.state.tolist() == ["FAIL", "COMPLETE", "COMPLETE"]
    unchanged = wf_tuning.run_search(prepared, "lightgbm", execute, parent_spec=PARENT,
                                     target_completed=2)
    assert unchanged["completed_trials"] == 2 and len(calls) == 3
    extended = wf_tuning.run_search(prepared, "lightgbm", execute, parent_spec=PARENT,
                                    target_completed=3)
    assert extended["completed_trials"] == 3 and len(calls) == 4


def test_running_trial_resumes_same_number_and_params(tmp_path, small_minimum):
    prepared = Prepared(Path(__file__).resolve().parents[2], tmp_path)
    captured = []

    def interrupt(spec, *, score):
        captured.append((spec["id"], dict(spec["model_params"])))
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        wf_tuning.run_search(prepared, "lightgbm", interrupt, parent_spec=PARENT,
                             target_completed=2)

    def execute(spec, *, score):
        if not captured or len(captured) == 1:
            captured.append((spec["id"], dict(spec["model_params"])))
        return _ready(spec)

    state = wf_tuning.run_search(prepared, "lightgbm", execute, parent_spec=PARENT,
                                 target_completed=2)
    assert state["completed_trials"] == 2
    assert captured[0] == captured[1]


def test_public_target_requires_500_and_no_parent_drift(tmp_path):
    prepared = Prepared(Path(__file__).resolve().parents[2], tmp_path)
    with pytest.raises(ValueError, match="500"):
        wf_tuning.run_search(prepared, "lightgbm", lambda *_: None,
                             parent_spec=PARENT, target_completed=499)
    with pytest.raises(ValueError, match="parent"):
        wf_tuning._scope(prepared, "xgboost", PARENT, None)


def test_incomplete_prediction_status_never_counts_as_trial():
    with pytest.raises(ValueError, match="prediction_ready"):
        wf_tuning._objective({"status": "completed", "stop_MAE": 1.,
                              "stop_cells": 5, "n_seeds": 5,
                              "prediction_sha": "a" * 64}, 5)


def test_neural_tpe_uses_distinct_complete_settings_and_best_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(wf_tuning, "NEURAL_MIN_COMPLETED", 2)
    monkeypatch.setattr(wf_tuning, "ROUND_COMPLETED", 1)
    prepared = Prepared(Path(__file__).resolve().parents[2], tmp_path)
    parent = {"id": "F5-parent", "adapter": "neural", "kind": "dlinear",
              "target_baseline": "weekly"}
    seen = []

    def execute(spec, *, score):
        assert score is False and spec["adapter"] == "neural"
        assert spec["kind"] == "dlinear" and spec["n_seeds"] == 5
        seen.append(spec)
        return _ready(spec, stop=3 + len(seen))

    state = wf_tuning.run_neural_search(prepared, "dlinear", execute,
                                        parent_spec=parent, target_completed=2)
    assert state["completed_trials"] == 2 and len(seen) == 2
    assert state["best_spec"]["id"] == seen[0]["id"]
    assert state["best_spec"]["context_length"] in (96, 192, 336, 672, 1344, 2016, 2688)
    ledger = pd.read_csv(tmp_path / "logs/tuning/wf_tpe_dlinear_trials.csv")
    assert ledger.param_hash.nunique() == 2
    with pytest.raises(ValueError, match="Unsupported"):
        wf_tuning.run_neural_search(prepared, "lstm", execute,
                                    parent_spec={**parent, "kind": "lstm"}, target_completed=2)


def test_trial_parameters_override_inherited_nested_params():
    study = optuna.create_study()
    trial = study.ask()
    gbdt = wf_tuning._trial_spec({**PARENT, "params": {"model_params": {"num_leaves": 15}}},
                                 "lightgbm", trial, "nested")
    assert gbdt["params"]["model_params"] == gbdt["model_params"]
    study = optuna.create_study()
    trial = study.ask()
    neural = wf_tuning._trial_spec({"id": "parent", "adapter": "neural", "kind": "dlinear",
                                    "params": {"context_length": 96, "seeds": [42, 43, 44, 45, 46]}},
                                   "dlinear", trial, "nested-neural", group="neural")
    assert neural["params"]["context_length"] == neural["context_length"]
    assert "seeds" not in neural["params"]
