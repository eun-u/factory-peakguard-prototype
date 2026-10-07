"""Focused tests for the isolated transition expert and signed runner cache."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import joblib

from phase_f import goal_r1, goal_r1_paths, goal_transition
from phase_f.models import transition_expert as model
from phase_f.models.regression import _preprocessor
from phase_f.registry import sha256


class _Classifier:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, x):
        return np.tile([.2, .3, .5], (len(x), 1))


class _Magnitude:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


@pytest.mark.parametrize("change", [None, "origin", "horizon", "order", "duplicate", "value"])
def test_signed_rolling_cohort_compares_keys_across_integer_dtypes(change):
    paths = goal_r1_paths._sort_paths(pd.DataFrame({
        "origin": pd.date_range("2021-01-01", periods=2, freq="15min"),
        "horizon": [4, 16], "r1": [100., 101.], "q10": [90., 91.],
        "q50": [100., 101.], "q90": [110., 111.], "q95": [115., 116.]}))
    required = paths.loc[:, ["origin", "horizon"]].copy()
    required["horizon"] = required.horizon.astype("int64")
    assert paths.horizon.dtype == np.dtype("int16")
    assert not paths.loc[:, ["origin", "horizon"]].equals(required)
    digest = goal_r1_paths._hash_frame(paths)
    manifest = {"path_rows": 2, "paths_sha256": digest}
    parent = {"rolling_paths_sha256": digest}
    if change is None:
        goal_transition._verify_path_cohort(paths, required, manifest, parent)
        return
    if change == "origin":
        required.loc[0, "origin"] += pd.Timedelta(minutes=15)
    elif change == "horizon":
        required.loc[0, "horizon"] = 5
    elif change == "order":
        required = required.iloc[::-1].reset_index(drop=True)
    elif change == "duplicate":
        paths = pd.concat([paths.iloc[[0]], paths.iloc[[0]]], ignore_index=True)
        required = paths.loc[:, ["origin", "horizon"]].copy()
        digest = goal_r1_paths._hash_frame(paths)
        manifest["paths_sha256"] = parent["rolling_paths_sha256"] = digest
    elif change == "value":
        paths.loc[0, "r1"] += 1.
    with pytest.raises(ValueError, match="signed required cohort"):
        goal_transition._verify_path_cohort(paths, required, manifest, parent)


def test_class_boundaries_and_supported_mixture():
    assert model._classes(np.array([-30.01, -30, 0, 30, 30.01])).tolist() == [0, 1, 1, 1, 2]
    bundle = {"classifier": _Classifier(),
              "magnitudes": [_Magnitude(0), _Magnitude(100), _Magnitude(-100)]}
    point, probs = model._predict_expert(bundle, np.ones((2, 1)), np.array([100., 200.]))
    # Class supports become [-30, +30, +30]; probability mixture is +18.
    np.testing.assert_array_equal(point, [118., 218.])
    np.testing.assert_array_equal(probs[0], [.2, .3, .5])


def test_stop_objective_uses_observed_peaks_and_fails_without_them():
    truth = np.array([100., 100., 150., 150.])
    baseline = np.array([100., 100., 100., 100.])
    expert = np.array([90., 90., 145., 145.])
    winner, metrics = model._choose_alpha(truth, baseline, expert,
                                           np.array([False, False, True, True]))
    assert winner == 1.0
    assert metrics["0.0"]["PeakMAE"] == 50.
    with pytest.raises(ValueError, match="observed peak"):
        model._choose_alpha(truth, baseline, expert, np.zeros(4, bool))
    with pytest.raises(ValueError, match="finite"):
        model._choose_alpha(truth, baseline, np.array([np.nan, 0, 0, 0]),
                            np.ones(4, bool))


def _seed_frame(seed: int, *, r1: float = 100.) -> pd.DataFrame:
    return pd.DataFrame({
        "horizon": [4], "fold": [0], "role": ["score"],
        "origin": [pd.Timestamp("2021-05-01 00:00")],
        "target_time": [pd.Timestamp("2021-05-01 01:00")],
        "arm": ["EXPLORE"], "model": [f"seed{seed}"],
        "pred": [100. + seed], "y": [120.], "tau": [110.],
        "d2": [False], "fit_mean": [100.], "mase_scale": [1.],
        "r1": [r1], "transition_origin_power": [100.],
        "transition_p_fall": [.1 + seed * .01],
        "transition_p_neutral": [.3],
        "transition_p_rise": [.6 - seed * .01],
        "transition_expert_point": [110. + seed],
    })


def test_five_seed_mean_rebuilds_probabilities_expert_and_r1():
    frames = [_seed_frame(i) for i in range(5)]
    mean = goal_transition._mean_of_seeds(frames, "FG-R3-test")
    assert mean.pred.iloc[0] == 102.
    assert mean.transition_p_fall.iloc[0] == pytest.approx(.12)
    assert mean.transition_expert_point.iloc[0] == 112.
    assert mean.applied_correction.iloc[0] == 2.
    bad = copy.deepcopy(frames)
    bad[-1].loc[0, "r1"] = 101.
    with pytest.raises(ValueError, match="R1 path differs"):
        goal_transition._mean_of_seeds(bad, "FG-R3-test")


def test_score_only_transition_diagnostics_are_five_seed_probabilities(tmp_path):
    frame = pd.concat([_seed_frame(i) for i in range(3)], ignore_index=True)
    frame.loc[:, "transition_p_fall"] = [.1, .2, .3]
    frame.loc[:, "transition_p_neutral"] = [.3, .3, .3]
    frame.loc[:, "transition_p_rise"] = [.6, .5, .4]
    frame.loc[:, "y"] = [140., 69., 100.]
    frame.loc[:, "transition_origin_power"] = 100.
    frame.loc[:, "tau"] = 110.
    digest, quality = goal_transition._transition_diagnostic(frame, tmp_path / "classes.csv")
    assert digest == sha256(tmp_path / "classes.csv")
    assert quality["n_score_rows"] == 3
    assert quality["multiclass_logloss"] > 0
    assert quality["multiclass_brier_sum"] > 0
    table = pd.read_csv(tmp_path / "classes.csv")
    assert table.actual_rows.tolist() == [1, 1, 1]
    assert table.loc[table["class"] == "rise", "peak_rows"].iloc[0] == 1
    assert table.phase_e_peak_event_metric.eq(False).all()


def test_checkpoint_cache_identity_tamper_and_orphan_preservation(tmp_path):
    path = tmp_path / "seed_42.parquet"
    frame = _seed_frame(0)
    goal_r1._publish(path, frame, {"identity": "signed", "audit": {"seed": 42}})
    assert goal_transition._cached(path, "signed")[0].equals(frame)
    with pytest.raises(ValueError, match="identity changed"):
        goal_transition._cached(path, "other")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata["arm"] = "CONFIRM"
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata arm changed"):
        goal_transition._cached(path, "signed")
    path.with_suffix(".json").unlink()
    assert goal_transition._cached(path, "signed") is None
    assert list(tmp_path.glob("seed_42.orphan-*.parquet"))
    assert not path.exists()


def test_seed_cache_requires_exact_fold_model_and_metadata_bytes(monkeypatch, tmp_path):
    child = {"id": "FG-R3-test__seed42", "seed": 42, "adapter": "transition_expert",
             "arm": "EXPLORE"}
    output = tmp_path / "transition"
    path = output / "predictions/seeds/EXPLORE/FG-R3-test/seed_42.parquet"
    checkpoint = output / "models/transition_expert/FG-R3-test__seed42/seed42_f0.joblib"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model-original")
    model_meta = {"identity": "locked-model", "model_sha256": sha256(checkpoint),
                  "seed": 42, "fold": 0, "fit_class_count": [20, 20, 20],
                  "stop_class_count": [20, 20, 20], "alpha": .5}
    checkpoint.with_suffix(".json").write_text(json.dumps(model_meta), encoding="utf-8")
    cell = {**model_meta,
            "checkpoint_metadata_sha256": sha256(checkpoint.with_suffix(".json")),
            "future_perturbation_max_abs_difference": 0.0}
    audit = {"seed": 42, "leakage_test": "passed", "rolling_audit_sha256": "rolling",
             "paths_sha256": "paths", "future_perturbation_max_abs_difference": 0.0,
             "smoke_first_fold": False, "n_fold_models": 1,
             "cells": [cell], "holdout_read": False}
    goal_r1._publish(path, _seed_frame(0), {"identity": "signed-seed", "audit": audit,
                                              "child_spec": child, "seed": 42})
    prepared = SimpleNamespace(out=output, contexts={(4, 0): {}})
    monkeypatch.setattr(goal_transition, "_validate_frame", lambda *args: None)
    monkeypatch.setattr(model, "_common_roles", lambda *args: (None, None, {}))
    monkeypatch.setattr(model, "_identity", lambda *args: "locked-model")
    goal_transition._verify_seed(prepared, child, path, "signed-seed", "rolling", "paths")
    prepared.contexts = {(4, fold): {} for fold in range(8)}
    with pytest.raises(ValueError, match="seed audit is incomplete"):
        goal_transition._verify_seed(prepared, child, path, "signed-seed", "rolling", "paths")
    prepared.contexts = {(4, 0): {}}
    metadata_path = path.with_suffix(".json")
    original_metadata = metadata_path.read_text(encoding="utf-8")
    changed = json.loads(original_metadata)
    changed["audit"]["future_perturbation_max_abs_difference"] = .1
    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="seed audit is incomplete"):
        goal_transition._verify_seed(prepared, child, path, "signed-seed", "rolling", "paths")
    changed["audit"]["future_perturbation_max_abs_difference"] = 0.0
    changed["audit"]["cells"][0]["future_perturbation_max_abs_difference"] = True
    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fold future-perturbation"):
        goal_transition._verify_seed(prepared, child, path, "signed-seed", "rolling", "paths")
    metadata_path.write_text(original_metadata, encoding="utf-8")
    checkpoint.write_bytes(b"model-tampered")
    with pytest.raises(ValueError, match="checkpoint provenance changed"):
        goal_transition._verify_seed(prepared, child, path, "signed-seed", "rolling", "paths")


def test_search_rejects_missing_or_tampered_technical_smoke(monkeypatch, tmp_path):
    origin = pd.Timestamp("2021-05-01 00:00")
    contexts = {(h, 0): {"cal": pd.DatetimeIndex([origin]),
                         "score": pd.DatetimeIndex([origin + pd.Timedelta(days=1)])}
                for h in range(4, 17)}
    prepared = SimpleNamespace(out=tmp_path, contexts=contexts)
    plan = {"plan_sha256": "locked", "rolling_evidence": {
        "rolling_audit_sha256": "rolling", "paths_sha256": "paths"}}
    monkeypatch.setattr(goal_transition, "_verify_frozen", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="technical smoke is missing"):
        goal_transition.run_search(prepared, pd.DataFrame(), plan)
    spec = goal_transition.SPECS[0]
    path = tmp_path / "smoke" / spec["id"] / "seed42_first_fold.parquet"
    path.parent.mkdir(parents=True)
    _seed_frame(0).to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"sha256": "tampered"}), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke provenance changed"):
        goal_transition.run_search(prepared, pd.DataFrame(), plan)


def _causal_fixture():
    index = pd.date_range("2021-01-01 00:00", periods=2500, freq="15min")
    power = 100 + np.sin(np.arange(len(index)) / 20)
    frame = pd.DataFrame({"power": power, "quality_bad": False,
                          "time_repaired": False}, index=index)
    frame["production_completed"] = np.where(index.minute == 0, 8., np.nan)
    frame["production_completed_bad"] = False
    frame["production_known"] = pd.Series(frame.production_completed, index=index).ffill()
    frame["production_known_bad"] = False
    frame["production_target"] = np.nan
    frame["production_hour_date"] = (index - pd.Timedelta(nanoseconds=1)).normalize()
    origin = index[-100]
    previous = origin - pd.Timedelta(hours=4)
    paths = pd.DataFrame({"origin": [previous, origin], "horizon": [16, 16],
                          "r1": [101., 102.], "q10": [90., 91.],
                          "q50": [101., 102.], "q90": [110., 111.],
                          "q95": [115., 116.]}).set_index(["origin", "horizon"])
    return frame, origin, paths


def test_future_power_and_production_perturbation_cannot_change_features_or_prediction():
    history, origin, paths = _causal_fixture()
    chosen = pd.DatetimeIndex([origin])
    groups = ("production",)
    x = model._matrix(history, paths, chosen, 16, 120., groups)
    bundle = {"classifier": _Classifier(),
              "magnitudes": [_Magnitude(-50), _Magnitude(0), _Magnitude(50)],
              "preprocessor": _preprocessor(x, scale=False),
              "alpha": .5, "feature_columns": list(x.columns)}
    view = SimpleNamespace(history=history)
    assert model._perturbation_check(view, paths, 0, {"score": chosen, "tau": 120.},
                                      bundle, groups) == 0.0
    availability = model._production_availability(history, chosen, x)
    assert availability["production_group_used"] is True
    assert availability["zero_posting_latency_assumed"] is True


def test_insufficient_classes_fail_before_training(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    index = pd.date_range("2021-01-01", periods=80, freq="15min")
    fit, stop = index[:40], index[40:60]
    history = pd.DataFrame({"power": 100.}, index=index)
    contexts = {}
    for h in range(4, 17):
        contexts[(h, 0)] = {"fit": fit, "stop": stop, "cal": index[60:65],
                            "score": index[65:70],
                            "y": pd.Series(100., index=index),
                            "target_time": pd.Series(index + pd.Timedelta(minutes=15*h), index=index),
                            "tau": 110.}
    paths = pd.concat([pd.DataFrame({"origin": index[:60], "horizon": h,
                                     "r1": 100.}) for h in range(4, 17)],
                      ignore_index=True).set_index(["origin", "horizon"])
    view = SimpleNamespace(root=root, out=tmp_path, history=history,
                           contexts=contexts, split_lock={"lock_sha256": "synthetic"},
                           seal={"raw_sha256": "synthetic"})
    monkeypatch.setattr(model, "_matrix", lambda hist, paths, origins, h, tau, groups:
                        pd.DataFrame({"dummy": np.ones(len(origins))}, index=origins))
    with pytest.raises(ValueError, match="under minimum"):
        model._fit_or_load(view, {"id": "synthetic", "seed": 42}, paths,
                           "0" * 64, 0, fit, stop, ())


@pytest.mark.parametrize("corruption,reason", [
    ("nul_metadata", "metadata_json_corrupt"),
    ("wrong_model_hash", "model_sha256_changed"),
    ("bad_payload_shape", "payload_shape_changed"),
])
def test_corrupt_checkpoint_pair_is_hashed_preserved_and_rejected(
        monkeypatch, tmp_path, corruption, reason):
    view = SimpleNamespace(out=tmp_path)
    spec = {"id": "synthetic", "seed": 42}
    checkpoint = tmp_path / "models/transition_expert/synthetic/seed42_f0.joblib"
    checkpoint.parent.mkdir(parents=True)
    if corruption == "bad_payload_shape":
        joblib.dump({"wrong": True}, checkpoint)
    else:
        checkpoint.write_bytes(b"model-before-corruption")
    metadata = checkpoint.with_suffix(".json")
    if corruption == "nul_metadata":
        metadata.write_bytes(b"\x00" * 37)
    else:
        recorded = sha256(checkpoint) if corruption == "bad_payload_shape" else "0" * 64
        metadata.write_text(json.dumps({"identity": "expected",
                                        "model_sha256": recorded}), encoding="utf-8")
    model_sha, metadata_sha = sha256(checkpoint), sha256(metadata)
    monkeypatch.setattr(model, "_identity", lambda *args: "expected")
    with pytest.raises(RuntimeError, match=f"rejected \\({reason}\\)"):
        model._fit_or_load(view, spec, pd.DataFrame(), "paths", 0,
                           pd.DatetimeIndex([]), pd.DatetimeIndex([]), ())
    records = list((tmp_path / "logs/checkpoint_corruption").glob("*.json"))
    assert len(records) == 1
    incident = json.loads(records[0].read_text(encoding="utf-8"))
    assert incident["reason"] == reason
    assert incident["expected_identity"] == "expected"
    assert incident["model_sha256"] == model_sha
    assert incident["metadata_sha256"] == metadata_sha
    assert incident["refit_performed_in_this_attempt"] is False
    assert sha256(Path(incident["model_preserved"])) == model_sha
    assert sha256(Path(incident["metadata_preserved"])) == metadata_sha
    assert not checkpoint.exists() and not metadata.exists()
