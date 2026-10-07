"""FG-R5 raw-history analog causality and checkpoint regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from phase_f.models import recent_day_analog as model
from phase_f.registry import config_hash, sha256


ROOT = Path(__file__).resolve().parents[2]


def _view(tmp_path, *, length=16, k=3):
    idx = pd.date_range("2021-04-01 00:15", periods=1400, freq="15min")
    slot = idx.hour.to_numpy() * 4 + idx.minute.to_numpy() // 15
    day = np.arange(len(idx)) // 96
    power = 100 + 20 * np.sin(slot * 2 * np.pi / 96) + (day % 5) * 4
    history = pd.DataFrame({"power": power, "quality_bad": False,
                            "time_repaired": False}, index=idx)
    roles = {"fit": idx[96:768], "stop": idx[800:880],
             "cal": idx[900:910], "score": idx[950:960]}
    contexts = {}
    for h in model.HORIZONS:
        contexts[(h, 0)] = {**roles,
            "y": pd.Series(power, index=idx).shift(-h), "tau": 100.,
            "x": pd.DataFrame({"slot7d": 100.}, index=idx),
            "d2": pd.Series(False, index=idx),
            "target_time": pd.Series(idx + h * pd.Timedelta(minutes=15), index=idx),
            "summary": {"arm": "EXPLORE"}}
    paths = pd.concat([pd.DataFrame({"origin": origins, "horizon": h,
                                      "r1": 100., "q50": 100., "q10": 90.,
                                      "q90": 110., "q95": 115.})
                       for h in model.HORIZONS for origins in roles.values()],
                      ignore_index=True)
    proof = {"leakage_test": "passed", "input_cutoff_rule": "history.index <= origin",
             "future_perturbation_max_abs_difference": 0.0}
    paths.attrs["causal_provenance"] = proof
    spec = {"id": f"FG-R5-recent-w{length}-k{k}",
            "adapter": "recent_day_analog", "arm": "EXPLORE",
            "prefix_length": length, "top_k": k, "bank_days": 56,
            "rolling_audit_sha256": config_hash(proof)}
    view = SimpleNamespace(root=ROOT, out=tmp_path, history=history,
                           contexts=contexts, split_lock={"lock_sha256": "split"},
                           seal={"raw_sha256": "raw"})
    return view, spec, paths


def test_exact_midnight_references_recent_first_and_weighted_median():
    idx = pd.date_range("2021-04-01 00:00", periods=8 * 96 + 1, freq="15min")
    power = pd.Series(100., index=idx)
    origin = pd.DatetimeIndex([pd.Timestamp("2021-04-08 00:00")])
    result = model._retrieve(power, origin, 16, 3,
                             score_start=pd.Timestamp("2021-04-05"))
    assert result["support"].tolist() == [6]
    assert result["selected_refs_ns"][0].tolist() == [
        (origin[0] - pd.Timedelta(days=d)).value for d in (1, 2, 3)]
    assert result["analog"].shape == (1, 13)
    assert np.all(result["analog"] == 100.)
    assert result["latest_source_ns"].tolist() == [origin[0].value]
    assert (result["elapsed_eval_observation_reads"] > 0).all()


def test_past_changes_feature_future_power_and_quality_do_not():
    idx = pd.date_range("2021-04-01", periods=9 * 96 + 1, freq="15min")
    history = pd.DataFrame({"power": 100., "quality_bad": False,
                            "time_repaired": False}, index=idx)
    origin = pd.DatetimeIndex([idx[8 * 96]])
    first = model._retrieve(model._power(history), origin, 16, 3)
    changed_past = history.copy()
    for day in (1, 2, 3):
        changed_past.loc[origin[0] - pd.Timedelta(days=day)
                         + pd.Timedelta(hours=1), "power"] = 150.
    past = model._retrieve(model._power(changed_past), origin, 16, 3)
    assert not np.array_equal(first["analog"], past["analog"], equal_nan=True)
    changed_future = history.copy()
    changed_future.loc[changed_future.index > origin[0], "power"] = 10000.
    changed_future.loc[changed_future.index > origin[0], "quality_bad"] = True
    changed_future.loc[changed_future.index > origin[0], "time_repaired"] = True
    future = model._retrieve(model._power(changed_future), origin, 16, 3)
    np.testing.assert_array_equal(first["analog"], future["analog"])
    np.testing.assert_array_equal(first["selected_refs_ns"], future["selected_refs_ns"])
    np.testing.assert_array_equal(first["distances"], future["distances"])


def test_exact_timestamp_missing_and_bad_suffix_remove_reference():
    idx = pd.date_range("2021-04-01", periods=8 * 96 + 1, freq="15min")
    history = pd.DataFrame({"power": 100., "quality_bad": False,
                            "time_repaired": False}, index=idx)
    origin = pd.DatetimeIndex([idx[7 * 96]])
    base = model._retrieve(model._power(history), origin, 16, 5)
    assert base["support"].item() == 6
    # Every selected case must have all 13 suffixes, even when evaluating h4.
    reference = origin[0] - pd.Timedelta(days=1)
    bad_target = reference + pd.Timedelta(hours=4)
    history.loc[bad_target, "quality_bad"] = True
    excluded = model._retrieve(model._power(history), origin, 16, 5)
    assert excluded["support"].item() == 5
    assert reference.value not in excluded["selected_refs_ns"][0]
    history = history.drop(origin[0] - pd.Timedelta(minutes=15))
    fallback = model._retrieve(model._power(history), origin, 16, 5)
    assert fallback["fallback"].item()
    assert np.isnan(fallback["analog"]).all()
    assert model._blend(np.array([77.]), fallback["analog"][:, 0], 1.).item() == 77.


def test_fewer_than_k_and_negative_analog_clipping():
    idx = pd.date_range("2021-04-01", periods=4 * 96 + 1, freq="15min")
    power = pd.Series(100., index=idx)
    query = pd.DatetimeIndex([idx[4 * 96]])
    result = model._retrieve(power, query, 16, 5)
    assert result["support"].item() == 3
    assert result["fallback"].item()
    with pytest.raises(ValueError, match="fixed alpha"):
        model._blend(np.array([100.]), np.array([100.]), True)
    # Query level 0 with prior reference level 100 shifts all suffixes below 0.
    power.loc[query[0]] = 0.
    clipped = model._retrieve(power, query, 16, 3)
    assert clipped["analog"].min() == 0.


@pytest.mark.parametrize("length,k", [(16, 3), (96, 5)])
def test_full_fold_api_no_truth_in_inference_and_fresh_replay(tmp_path, length, k):
    view, spec, paths = _view(tmp_path / "primary", length=length, k=k)
    frame, audit = model.run(view, spec, paths, smoke_first_fold=True)
    assert len(frame) == 13 * 20
    assert not frame.duplicated(["fold", "horizon", "role", "origin"]).any()
    assert frame.recent_analog_fallback.eq(frame.recent_analog_point.isna()).all()
    assert np.isfinite(frame.pred).all()
    assert audit["historical_final_artifact_read"] is False
    assert audit["cells"][0]["future_perturbation_unblended_all13_equal"] is True
    assert audit["cells"][0]["future_perturbation_probe_origins"] == 13
    assert len(audit["cells"][0]["coverage"]) == 26
    model.verify_cells(view, spec, paths, audit, smoke_first_fold=True)

    replay = copy.copy(view)
    replay.out = tmp_path / "fresh_replay"
    replay_frame, replay_audit = model.run(replay, spec, paths, smoke_first_fold=True)
    pd.testing.assert_frame_equal(frame, replay_frame)
    assert replay_audit["cells"][0]["identity"] == audit["cells"][0]["identity"]
    model.verify_cells(replay, spec, paths, replay_audit, smoke_first_fold=True)

    changed = copy.copy(view)
    changed.contexts = copy.deepcopy(view.contexts)
    for context in changed.contexts.values():
        context["y"].loc[context["cal"]] = 999.
        context["y"].loc[context["score"]] = 999.
    altered, altered_audit = model.run(changed, spec, paths, smoke_first_fold=True)
    np.testing.assert_array_equal(frame.pred, altered.pred)
    assert altered_audit["cells"][0]["identity"] == audit["cells"][0]["identity"]


def test_stop_power_and_truth_changes_checkpoint_identity(tmp_path):
    view, spec, paths = _view(tmp_path)
    fit, stop, _ = model._common_roles(view.contexts, 0)
    _, path_sha = model._paths_index(paths, view.contexts)
    first = model._retrieve(model._power(view.history), stop, 16, 3)
    identity = model._identity(view, spec, path_sha, 0, fit, stop, first["input_sha256"])
    changed = copy.copy(view)
    changed.history = view.history.copy()
    changed.history.loc[stop[0], "power"] += 3.
    second = model._retrieve(model._power(changed.history), stop, 16, 3)
    assert model._identity(changed, spec, path_sha, 0, fit, stop,
                           second["input_sha256"]) != identity
    changed = copy.copy(view)
    changed.contexts = copy.deepcopy(view.contexts)
    changed.contexts[(4, 0)]["y"].loc[stop[0]] += 1.
    assert model._identity(changed, spec, path_sha, 0, fit, stop,
                           first["input_sha256"]) != identity


@pytest.mark.parametrize("tamper", ["metadata_alpha", "payload_alpha", "metadata_json"])
def test_tampered_checkpoint_rejected_and_preserved(tmp_path, tamper):
    view, spec, paths = _view(tmp_path)
    _, audit = model.run(view, spec, paths, smoke_first_fold=True)
    model_path, metadata_path = model._checkpoint_paths(view, spec, 0)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tamper == "metadata_alpha":
        metadata["alpha"] = 1.0 if metadata["alpha"] != 1.0 else 0.0
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif tamper == "payload_alpha":
        bundle = joblib.load(model_path)
        bundle["alpha"] = 1.0 if bundle["alpha"] != 1.0 else 0.0
        joblib.dump(bundle, model_path)
        metadata["model_sha256"] = sha256(model_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        metadata_path.write_bytes(b"{truncated-json")
    model_sha, metadata_sha = sha256(model_path), sha256(metadata_path)
    with pytest.raises(RuntimeError, match="preserved physical evidence"):
        model.verify_cells(view, spec, paths, audit, smoke_first_fold=True)
    assert sha256(next(model_path.parent.glob("f0.orphan-*.joblib"))) == model_sha
    assert sha256(next(model_path.parent.glob("f0.orphan-*.json"))) == metadata_sha
