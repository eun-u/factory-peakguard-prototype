"""Causal runner boundaries for deterministic FIT-bank analog experiments."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f import goal_day_analog as runner
from phase_f.registry import config_hash, sha256, write_json


def _frame(*, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "horizon": [4, 4], "fold": [1, 1], "role": ["cal", "score"],
        "origin": pd.to_datetime(["2021-05-01 00:00", "2021-05-02 00:00"]),
        "target_time": pd.to_datetime(["2021-05-01 01:00", "2021-05-02 01:00"]),
        "model": [runner.SPECS[0]["id"]] * 2,
        "arm": ["CAL", "EXPLORE"], "pred": [100. + shift, 101. + shift],
        "y": [102., 103.], "tau": [110., 110.], "d2": [False, True],
        "fit_mean": [100., 100.], "mase_scale": [1., 1.],
        "r1": [99., 99.], "analog_point": [104., np.nan],
        "analog_support": [8, 0], "analog_fallback": [False, True],
        "analog_alpha": [.5, .5],
    })


def test_four_fixed_specs_are_deterministic_and_have_unique_ids():
    assert [item["id"] for item in runner.SPECS] == [
        "FG-R4-analog-w16-k3", "FG-R4-analog-w16-k5",
        "FG-R4-analog-w96-k3", "FG-R4-analog-w96-k5"]
    assert {(item["prefix_length"], item["top_k"], item["bank_days"])
            for item in runner.SPECS} == {
        (16, 3, 56), (16, 5, 56), (96, 3, 56), (96, 5, 56)}
    assert all(item["arm"] == "EXPLORE" and item["adapter"] == "day_analog"
               and "seed" not in item for item in runner.SPECS)
    assert runner.CONTRACT["n_stochastic_seeds"] == 0
    assert runner.CONTRACT["n_deterministic_runs"] == 1


def test_replay_requires_same_keys_point_and_auxiliary_values():
    primary = _frame()
    assert runner._compare_replay(primary, primary.copy())["point_predictions_equal"]
    changed = primary.copy()
    changed.loc[1, "pred"] += 0.01
    with pytest.raises(ValueError, match="prediction changed"):
        runner._compare_replay(primary, changed)
    changed = primary.copy()
    changed.loc[1, "analog_support"] += 1
    with pytest.raises(ValueError, match="auxiliary prediction changed"):
        runner._compare_replay(primary, changed)
    changed = primary.iloc[::-1].copy()
    assert runner._compare_replay(primary, changed)["ordered_keys_equal"]
    changed.loc[1, "origin"] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="keys or deterministic prediction changed"):
        runner._compare_replay(primary, changed)


def test_forecast_cache_preserves_incomplete_and_tampered_files(tmp_path):
    path = tmp_path / "forecast.parquet"
    metadata = path.with_suffix(".json")
    path.write_bytes(b"orphaned-forecast")
    original_sha = sha256(path)
    with pytest.raises(ValueError, match="Incomplete analog forecast cache"):
        runner._cached(path, "signed")
    assert sha256(path) == original_sha and not metadata.exists()
    path.unlink()
    runner._publish(path, _frame(), {"identity": "signed", "run": "primary"})
    assert runner._cached(path, "signed")[0].equals(_frame())
    forecast_sha = sha256(path)
    metadata_sha = sha256(metadata)
    with pytest.raises(ValueError, match="identity or physical SHA changed"):
        runner._cached(path, "other")
    assert sha256(path) == forecast_sha and sha256(metadata) == metadata_sha
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity or physical SHA changed"):
        runner._cached(path, "signed")
    assert path.read_bytes() == b"tampered"


def test_runner_binds_producer_and_model_path_hash_domains(monkeypatch):
    called = []
    monkeypatch.setattr(runner.goal_transition, "_verify_path_digests",
                        lambda paths, contexts, evidence: called.append(evidence))
    monkeypatch.setattr(runner, "_model_paths_sha", lambda paths, contexts: "model")
    evidence = {"paths_sha256": "producer", "transition_model_paths_sha256": "transition",
                "model_paths_sha256": "model"}
    runner._verify_path_digests(pd.DataFrame(), {}, evidence)
    assert called == [{"paths_sha256": "producer", "model_paths_sha256": "transition"}]
    with pytest.raises(ValueError, match="normalized model path digest changed"):
        runner._verify_path_digests(pd.DataFrame(), {}, {**evidence,
                                     "model_paths_sha256": "wrong"})


def test_first_fresh_replay_rejects_model_cache_reuse_before_prediction_publish(
        monkeypatch, tmp_path):
    spec = runner.SPECS[0]
    plan = {"plan_sha256": "plan", "rolling_evidence": {
        "rolling_audit_sha256": "a" * 64, "model_paths_sha256": "normalized"}}
    view = SimpleNamespace(out=tmp_path)
    monkeypatch.setattr(runner.day_analog, "run", lambda *args, **kwargs: (
        _frame(), {"cells": [{"cache_reused": True}]}))
    with pytest.raises(ValueError, match="Fresh replay reused"):
        runner._run_prediction(view, spec, pd.DataFrame(), plan, "fresh_replay")
    assert not (tmp_path / "predictions/EXPLORE" / f"{spec['id']}.parquet").exists()


def test_fresh_replay_preserves_partial_checkpoint_before_new_fit(monkeypatch, tmp_path):
    spec = runner.SPECS[0]
    checkpoint_dir = tmp_path / "models/day_analog" / spec["id"]
    checkpoint_dir.mkdir(parents=True)
    partial = checkpoint_dir / "f1.joblib"
    partial.write_bytes(b"interrupted")
    attempted = []
    monkeypatch.setattr(runner.day_analog, "run", lambda *args, **kwargs: attempted.append(True))
    plan = {"plan_sha256": "plan", "rolling_evidence": {
        "rolling_audit_sha256": "a" * 64, "model_paths_sha256": "normalized"}}
    with pytest.raises(ValueError, match="partial pre-existing model cache"):
        runner._run_prediction(SimpleNamespace(out=tmp_path), spec, pd.DataFrame(),
                               plan, "fresh_replay")
    assert partial.read_bytes() == b"interrupted" and not attempted


def test_smoke_cohort_requires_all_thirteen_horizons_and_no_duplicates(monkeypatch):
    monkeypatch.setattr(runner, "_validate_frame", lambda *args: None)
    times = pd.to_datetime(["2021-05-01 00:00", "2021-05-02 00:00"])
    contexts = {}
    frames = []
    for h in range(4, 17):
        context = {"cal": pd.DatetimeIndex(times[:1]),
                   "score": pd.DatetimeIndex(times[1:]),
                   "target_time": pd.Series(
                       times + pd.Timedelta(minutes=15 * h), index=times)}
        contexts[(h, 1)] = context
        part = _frame().copy()
        part["horizon"] = h
        part["target_time"] = part.origin + pd.Timedelta(minutes=15 * h)
        frames.append(part)
    prepared = SimpleNamespace(contexts=contexts,
                               origins=lambda h, fold, role: contexts[(h, fold)][role])
    full = pd.concat(frames, ignore_index=True)
    runner._verify_cohort(prepared, full, runner.SPECS[0], smoke_first_fold=True)
    with pytest.raises(ValueError, match="locked weekly cohort"):
        runner._verify_cohort(prepared, full.loc[full.horizon.ne(16)], runner.SPECS[0],
                              smoke_first_fold=True)
    with pytest.raises(ValueError, match="locked weekly cohort"):
        runner._verify_cohort(prepared, pd.concat([full, full.iloc[[0]]]),
                              runner.SPECS[0], smoke_first_fold=True)


def test_prediction_metadata_calls_physical_cell_verifier(monkeypatch, tmp_path):
    spec = runner.SPECS[0]
    view = SimpleNamespace(out=tmp_path)
    plan = {"plan_sha256": "plan", "rolling_evidence": {
        "rolling_audit_sha256": "a" * 64, "model_paths_sha256": "model"}}
    path = tmp_path / "smoke" / spec["id"] / "first_fold.parquet"
    audit = {"leakage_test": "passed", "holdout_read": False,
             "smoke_first_fold": True, "paths_sha256": "model"}
    runner._publish(path, _frame(), {"identity": runner._prediction_identity(plan, spec, "smoke"),
                   "audit": audit, "spec": spec, "model_spec": runner._model_spec(spec, plan),
                   "run": "smoke", "plan_sha256": "plan", "smoke_first_fold": True})
    checks = []
    monkeypatch.setattr(runner, "_verify_cohort", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.day_analog, "verify_cells",
                        lambda prepared, spec, paths, audit, **kwargs: checks.append(spec))
    runner._verify_prediction(view, spec, pd.DataFrame(), plan, "smoke",
                              smoke_first_fold=True)
    assert checks == [runner._model_spec(spec, plan)]
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata["model_spec"]["rolling_audit_sha256"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata or causal audit changed"):
        runner._verify_prediction(view, spec, pd.DataFrame(), plan, "smoke",
                                  smoke_first_fold=True)


@pytest.mark.parametrize("drift", ["design", "source", "viability"])
def test_frozen_plan_fails_before_models_if_design_source_or_viability_changes(
        monkeypatch, tmp_path, drift):
    root = tmp_path
    out = root / "outputs/phase_f" / runner.NAMESPACE
    out.mkdir(parents=True)
    design = out / runner.DESIGN
    design.write_text("frozen design", encoding="utf-8")
    viability = out / "diagnostics/fit_bank_support_preflight.json"
    viability.parent.mkdir(parents=True)
    viability.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "_sources", lambda root: {"fixed": "source" if drift != "source" else "changed"})
    monkeypatch.setattr(runner.goal_transition, "_runtime", lambda: {"runtime": "fixed"})
    prepared = SimpleNamespace(root=root, out=out,
                               split_lock={"lock_sha256": "split"},
                               seal={"raw_sha256": "raw"})
    plan = {"contract": runner.CONTRACT, "specs": list(runner.SPECS),
            "sources": {"fixed": "source"}, "runtime": {"runtime": "fixed"},
            "design_sha256": sha256(design),
            "fit_bank_support_preflight_sha256": sha256(viability),
            "split_sha256": "split", "seal_sha256": config_hash(prepared.seal),
            "raw_sha256": "raw", "arm": "EXPLORE", "holdout_read": False}
    plan["plan_sha256"] = config_hash(plan)
    write_json(out / "logs/execution_plan.json", plan)
    if drift == "design":
        design.write_text("changed", encoding="utf-8")
    elif drift == "viability":
        viability.write_text("{\"changed\":true}", encoding="utf-8")
    message = ("FIT-bank support preflight changed its scope" if drift == "viability"
               else "Frozen day-analog plan/source/runtime/design changed")
    with pytest.raises(ValueError, match=message):
        runner._verify_frozen(prepared, plan, pd.DataFrame())


def test_search_refuses_missing_smoke_before_primary_fit(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(runner, "_verify_frozen", lambda *args, **kwargs: called.append("frozen"))
    view = SimpleNamespace(out=tmp_path)
    with pytest.raises(ValueError, match="technical smoke is missing"):
        runner.run_search(view, pd.DataFrame(), {"plan_sha256": "locked"})
    assert called == ["frozen"]


def test_support_table_reports_role_and_d2_fallback():
    result = runner._support_table(_frame())
    assert set(result.role) == {"cal", "score"}
    d2 = result.loc[result.cohort.eq("D2")].iloc[0]
    assert d2.role == "score" and d2.analog_fallback_mean == 1.0
    assert d2.analog_support_mean == 0.0


def test_retry_after_scored_manifest_advances_to_next_spec_and_rejects_drift(
        monkeypatch, tmp_path):
    specs = runner.SPECS[:2]
    monkeypatch.setattr(runner, "SPECS", specs)
    out = tmp_path / "outputs/phase_f" / runner.NAMESPACE
    out.mkdir(parents=True)
    prepared = SimpleNamespace(root=tmp_path, out=out)
    guard = tmp_path / "outputs/phase_f" / runner.GUARD \
        / "predictions/EXPLORE" / f"{runner.BEST_GUARD}.parquet"
    guard.parent.mkdir(parents=True)
    _frame().to_parquet(guard, index=False)
    plan = {"guard_evidence": {"mean_sha256": sha256(guard)}}
    monkeypatch.setattr(runner, "_verify_frozen", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_smoke", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_replay_view", lambda view: SimpleNamespace(
        out=view.out / "fresh_replay"))
    monkeypatch.setattr(runner, "_compare_replay", lambda *args: {
        "ordered_keys_equal": True, "point_predictions_equal": True})
    monkeypatch.setattr(runner, "load_predictions", lambda *args: _frame())
    monkeypatch.setattr(runner.wf_metrics, "paired_ci", lambda *args, **kwargs:
                        pd.DataFrame({"MAE_gain": [1.]}))
    monkeypatch.setattr(runner.wf_metrics, "evaluate", lambda *args, **kwargs: {
        "auc": pd.DataFrame({"dataset": ["D1"], "AUC_MAE": [5.],
                             "AUC_PeakMAE": [10.]}),
        "pooled": pd.DataFrame({"dataset": ["D1"] * 13,
                                "horizon": list(range(4, 17)),
                                "MAE": [5.] * 13, "Peak_MAE": [10.] * 13,
                                "nMAE": [.06] * 13})})
    for spec in specs:
        for destination in (out, out / "fresh_replay"):
            frame = _frame()
            frame["model"] = spec["id"]
            runner._publish(runner.prediction_path(SimpleNamespace(out=destination),
                                                   spec["id"]), frame,
                            {"identity": "physical", "audit": {"cells": []}})
            (destination / "models/day_analog" / spec["id"]).mkdir(parents=True)

    def loaded(view, spec, paths, plan, run, **kwargs):
        frame = _frame()
        frame["model"] = spec["id"]
        return frame, {"audit": {"cells": [{"cache_reused": False}]}}

    monkeypatch.setattr(runner, "_run_prediction", loaded)
    actual_record = runner._record
    crash_once = {"pending": True}

    def interrupt_after_first_score(path, value):
        if (crash_once["pending"] and value["status"] == "completed"
                and value["spec"]["id"] == specs[0]["id"]):
            crash_once["pending"] = False
            raise RuntimeError("simulated interruption after score manifest")
        actual_record(path, value)

    monkeypatch.setattr(runner, "_record", interrupt_after_first_score)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run_search(prepared, pd.DataFrame(), plan)
    first_manifest = out / "tables" / specs[0]["id"] / "EXPLORE/manifest.json"
    assert first_manifest.exists()
    assert not (out / "tables" / specs[1]["id"] / "EXPLORE/manifest.json").exists()

    runner.run_search(prepared, pd.DataFrame(), plan)
    assert all((out / "logs/experiments" / f"{spec['id']}.json").exists()
               for spec in specs)
    assert all(json.loads((out / "logs/experiments" / f"{spec['id']}.json")
                          .read_text(encoding="utf-8"))["status"] == "completed"
               for spec in specs)
    assert (out / "tables" / specs[1]["id"] / "EXPLORE/manifest.json").exists()

    original = first_manifest.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["fields"]["AUC_MAE"] = 0.
    first_manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to replace locked artifact"):
        runner.run_search(prepared, pd.DataFrame(), plan)
    assert first_manifest.read_text(encoding="utf-8") == json.dumps(tampered)
