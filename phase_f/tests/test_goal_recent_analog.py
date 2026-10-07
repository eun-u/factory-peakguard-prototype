"""Runner boundaries for causal recent observed-history analog experiments."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f import goal_recent_analog as runner
from phase_f.registry import config_hash, sha256, write_json
from phase_f.wf_models import _arm_view, _ArmView


def _frame(*, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "horizon": [4, 4], "fold": [1, 1], "role": ["cal", "score"],
        "origin": pd.to_datetime(["2021-05-01 00:00", "2021-05-02 00:00"]),
        "target_time": pd.to_datetime(["2021-05-01 01:00", "2021-05-02 01:00"]),
        "model": [runner.SPECS[0]["id"]] * 2,
        "arm": ["CAL", "EXPLORE"], "pred": [100. + shift, 101. + shift],
        "y": [102., 103.], "tau": [110., 110.], "d2": [False, True],
        "fit_mean": [100., 100.], "mase_scale": [1., 1.],
        "r1": [99., 99.], "recent_analog_point": [104., np.nan],
        "recent_analog_support": [8, 0], "recent_analog_fallback": [False, True],
        "recent_analog_alpha": [.5, .5],
        "recent_analog_latest_observed": pd.to_datetime(
            ["2021-05-01 00:00", None]),
        "recent_analog_elapsed_eval_reads": [0, 2],
        "recent_analog_ref1": pd.to_datetime(["2021-04-30 00:00", None]),
        "recent_analog_ref2": pd.to_datetime(["2021-04-29 00:00", None]),
        "recent_analog_ref3": pd.to_datetime(["2021-04-28 00:00", None]),
    })


def test_four_fixed_specs_are_deterministic_and_have_unique_ids():
    assert [item["id"] for item in runner.SPECS] == [
        "FG-R5-recent-w16-k3", "FG-R5-recent-w16-k5",
        "FG-R5-recent-w96-k3", "FG-R5-recent-w96-k5"]
    assert {(item["prefix_length"], item["top_k"], item["bank_days"])
            for item in runner.SPECS} == {
        (16, 3, 56), (16, 5, 56), (96, 3, 56), (96, 5, 56)}
    assert all(item["arm"] == "EXPLORE" and item["adapter"] == "recent_day_analog"
               and "seed" not in item for item in runner.SPECS)
    assert runner.CONTRACT["n_stochastic_seeds"] == 0
    assert runner.CONTRACT["n_deterministic_runs"] == 1
    assert runner.CONTRACT["historical_final_artifact_read"] is False


def test_replay_requires_same_keys_point_and_auxiliary_values():
    primary = _frame()
    assert runner._compare_replay(primary, primary.copy())["point_predictions_equal"]
    changed = primary.copy()
    changed.loc[1, "pred"] += 0.01
    with pytest.raises(ValueError, match="prediction changed"):
        runner._compare_replay(primary, changed)
    changed = primary.copy()
    changed.loc[1, "recent_analog_support"] += 1
    with pytest.raises(ValueError, match="auxiliary prediction changed"):
        runner._compare_replay(primary, changed)
    changed = primary.copy()
    changed.loc[0, "recent_analog_latest_observed"] -= pd.Timedelta(minutes=15)
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
    monkeypatch.setattr(runner.recent_day_analog, "run", lambda *args, **kwargs: (
        _frame(), {"cells": [{"cache_reused": True}]}))
    with pytest.raises(ValueError, match="Fresh replay reused"):
        runner._run_prediction(view, spec, pd.DataFrame(), plan, "fresh_replay")
    assert not (tmp_path / "predictions/EXPLORE" / f"{spec['id']}.parquet").exists()


def test_fresh_replay_preserves_partial_checkpoint_before_new_fit(monkeypatch, tmp_path):
    spec = runner.SPECS[0]
    checkpoint_dir = tmp_path / "models/recent_day_analog" / spec["id"]
    checkpoint_dir.mkdir(parents=True)
    partial = checkpoint_dir / "f1.joblib"
    partial.write_bytes(b"interrupted")
    attempted = []
    monkeypatch.setattr(runner.recent_day_analog, "run", lambda *args, **kwargs: attempted.append(True))
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
    coverage = [{"horizon": h, "role": role,
                 "latest_source_exceeds_origin_count": 0}
                for h in range(4, 17) for role in ("cal", "score")]
    audit = {"leakage_test": "passed", "holdout_read": False,
             "historical_final_artifact_read": False,
             "smoke_first_fold": True, "paths_sha256": "model",
             "cells": [{"coverage": coverage}]}
    runner._publish(path, _frame(), {"identity": runner._prediction_identity(plan, spec, "smoke"),
                   "audit": audit, "spec": spec, "model_spec": runner._model_spec(spec, plan),
                   "run": "smoke", "plan_sha256": "plan", "smoke_first_fold": True,
                   "historical_final_artifact_read": False})
    checks = []
    monkeypatch.setattr(runner, "_verify_cohort", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.recent_day_analog, "verify_cells",
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
    metadata["model_spec"]["rolling_audit_sha256"] = "a" * 64
    metadata["historical_final_artifact_read"] = True
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata or causal audit changed"):
        runner._verify_prediction(view, spec, pd.DataFrame(), plan, "smoke",
                                  smoke_first_fold=True)


@pytest.mark.parametrize("drift", ["design", "source", "contract"])
def test_frozen_plan_fails_before_models_if_design_source_or_contract_changes(
        monkeypatch, tmp_path, drift):
    root = tmp_path
    out = root / "outputs/phase_f" / runner.NAMESPACE
    out.mkdir(parents=True)
    design = out / runner.DESIGN
    design.write_text("frozen design", encoding="utf-8")
    contract = out / "CAUSAL_INPUT_CONTRACT.json"
    write_json(contract, runner.INPUT_CONTRACT)
    monkeypatch.setattr(runner, "_sources", lambda root: {"fixed": "source" if drift != "source" else "changed"})
    monkeypatch.setattr(runner.goal_transition, "_runtime", lambda: {"runtime": "fixed"})
    prepared = SimpleNamespace(root=root, out=out,
                               split_lock={"lock_sha256": "split"},
                               seal={"raw_sha256": "raw"})
    plan = {"contract": runner.CONTRACT, "specs": list(runner.SPECS),
            "sources": {"fixed": "source"}, "runtime": {"runtime": "fixed"},
            "design_sha256": sha256(design),
            "causal_input_contract": runner._causal_input_contract(root),
            "split_sha256": "split", "seal_sha256": config_hash(prepared.seal),
            "raw_sha256": "raw", "arm": "EXPLORE", "holdout_read": False}
    plan["plan_sha256"] = config_hash(plan)
    write_json(out / "logs/execution_plan.json", plan)
    if drift == "design":
        design.write_text("changed", encoding="utf-8")
    elif drift == "contract":
        changed = copy.deepcopy(runner.INPUT_CONTRACT)
        changed["score_tuning"] = "permitted"
        contract.write_text(json.dumps(changed), encoding="utf-8")
    message = ("Recent observed-history causal input contract changed" if drift == "contract"
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
    assert d2.role == "score" and d2.recent_analog_fallback_mean == 1.0
    assert d2.recent_analog_support_mean == 0.0
    assert d2.rows_with_elapsed_evaluation_raw_input == 1
    assert d2.recent_analog_elapsed_eval_reads_sum == 2
    assert d2.latest_observed_after_origin_count == 0


def test_recent_input_contract_is_exact_and_physically_hashed(tmp_path):
    folder = tmp_path / "outputs/phase_f" / runner.NAMESPACE
    folder.mkdir(parents=True)
    path = folder / "CAUSAL_INPUT_CONTRACT.json"
    write_json(path, runner.INPUT_CONTRACT)
    evidence = runner._causal_input_contract(tmp_path)
    assert evidence == {"sha256": sha256(path), "record": runner.INPUT_CONTRACT}
    changed = copy.deepcopy(runner.INPUT_CONTRACT)
    changed["reference_suffix"] = "future suffix permitted"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="causal input contract changed"):
        runner._causal_input_contract(tmp_path)


def test_prior_r4_motivation_binds_completed_scores_to_root_proof(tmp_path):
    base = tmp_path / "outputs/phase_f/goal_r1_day_analog_v2"
    previous_plan = {"contract": {"protocol": "goal_r1_day_analog_v2"},
                     "specs": [{"id": name} for name in runner.R4_MOTIVATION_IDS],
                     "arm": "EXPLORE", "holdout_read": False}
    previous_plan["plan_sha256"] = config_hash(previous_plan)
    write_json(base / "logs/execution_plan.json", previous_plan)
    write_json(base / "logs/driver_status.json", {
        "status": "completed_explore_stage", "candidate_count": 4,
        "full_explore_search_complete": True, "holdout_read": False})
    write_json(base / "diagnostics/fit_bank_support_preflight.json", {
        "analysis_only": True, "arm": "EXPLORE",
        "candidate_fit_performed": False, "candidate_score_computed": False,
        "holdout_read": False})
    rows = []
    for candidate in runner.R4_MOTIVATION_IDS:
        manifest = {"candidate": candidate, "arm": "EXPLORE",
                    "holdout_read": False, "targets_met": False,
                    "n_stochastic_seeds": 0}
        manifest_path = base / "tables" / candidate / "EXPLORE/manifest.json"
        record_path = base / "logs/experiments" / f"{candidate}.json"
        write_json(manifest_path, manifest)
        write_json(record_path, {"status": "completed", "spec": {"id": candidate},
                                 "result": manifest, "holdout_read": False})
        rows.append({"id": candidate, "n_primary_models": 8, "n_fresh_models": 8,
                     "targets_met": False,
                     "score_manifest_sha256": sha256(manifest_path),
                     "completed_record_sha256": sha256(record_path)})
    write_json(tmp_path / "outputs/phase_f/logs/revision_20261006"
               / "day_analog_v2_full_verification.json", {
        "plan_sha256": previous_plan["plan_sha256"],
        "source_runtime_raw_split_parent_verified": True,
        "actual_primary_and_fresh_checkpoints_verified": True,
        "completed_candidate_count": 4, "holdout_read": False,
        "candidate_results": rows})
    evidence = runner._motivation_evidence(tmp_path)
    assert evidence["completed_ids"] == list(runner.R4_MOTIVATION_IDS)
    assert evidence["historical_final_artifact_read"] is False
    assert len(evidence["files"]) == 12
    first = runner.R4_MOTIVATION_IDS[0]
    manifest_path = base / "tables" / first / "EXPLORE/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arm"] = "CONFIRM"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="completed score evidence changed"):
        runner._motivation_evidence(tmp_path)


def test_recent_frame_rejects_observation_or_reference_after_origin(monkeypatch):
    monkeypatch.setattr(runner, "_expected_keys", lambda *args, **kwargs:
                        pd.MultiIndex.from_frame(_frame()[list(runner.FRAME_KEY)]).sort_values())
    monkeypatch.setattr(runner, "_validate_frame", lambda *args: None)
    spec = runner.SPECS[0]
    view = SimpleNamespace(contexts={(4, 1): {}}, origins=lambda *args: [])
    future = _frame()
    future.loc[0, "recent_analog_latest_observed"] = future.loc[0, "origin"] + pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="read raw power after query origin"):
        runner._verify_cohort(view, future, spec, smoke_first_fold=False)
    future = _frame()
    future.loc[0, "recent_analog_ref1"] = future.loc[0, "origin"]
    with pytest.raises(ValueError, match="reference is not historical"):
        runner._verify_cohort(view, future, spec, smoke_first_fold=False)


def test_cached_k5_forecast_requires_all_five_reference_disclosures(monkeypatch):
    monkeypatch.setattr(runner, "_expected_keys", lambda *args, **kwargs:
                        pd.MultiIndex.from_frame(_frame()[list(runner.FRAME_KEY)]).sort_values())
    monkeypatch.setattr(runner, "_validate_frame", lambda *args: None)
    frame = _frame()
    frame["model"] = runner.SPECS[1]["id"]
    frame["recent_analog_ref4"] = pd.to_datetime(["2021-04-27", None])
    frame["recent_analog_ref5"] = pd.to_datetime(["2021-04-26", None])
    view = SimpleNamespace(contexts={(4, 1): {}}, origins=lambda *args: [])
    runner._verify_cohort(view, frame, runner.SPECS[1], smoke_first_fold=False)
    with pytest.raises(ValueError, match="reference provenance or k differs"):
        runner._verify_cohort(view, frame.drop(columns="recent_analog_ref5"),
                              runner.SPECS[1], smoke_first_fold=False)
    with pytest.raises(ValueError, match="reference provenance or k differs"):
        runner._verify_cohort(view, frame.assign(recent_analog_ref6=pd.NaT),
                              runner.SPECS[1], smoke_first_fold=False)


def test_fresh_replay_reconstructs_real_arm_view_without_changing_primary(
        monkeypatch, tmp_path):
    class Source:
        def __init__(self, root, out, contexts):
            self.root, self.out, self.contexts = root, out, contexts
            self.history = pd.DataFrame()
            self.split_lock = {"lock_sha256": "split"}
            self.seal = {"raw_sha256": "raw"}

        def origins(self, h, fold, role):
            return pd.DatetimeIndex(self.contexts[(h, fold)][role])

    contexts = {}
    for h in range(4, 17):
        for fold, origin in ((1, pd.Timestamp("2021-05-03 00:00")),
                             (2, pd.Timestamp("2021-05-10 00:00"))):
            contexts[(h, fold)] = {
                "fit": pd.DatetimeIndex([origin - pd.Timedelta(days=10)]),
                "stop": pd.DatetimeIndex([origin - pd.Timedelta(days=2)]),
                "cal": pd.DatetimeIndex([origin - pd.Timedelta(days=1)]),
                "score": pd.DatetimeIndex([origin]),
                "target_time": pd.Series([origin + pd.Timedelta(minutes=15*h)],
                                         index=pd.DatetimeIndex([origin])),
            }
    primary_out = tmp_path / "primary"
    primary_out.mkdir()
    (primary_out / "models").mkdir()
    source = Source(tmp_path, primary_out, contexts)
    primary = _arm_view(source, "EXPLORE")
    assert isinstance(primary, _ArmView)
    assert len(primary.contexts) == 13
    assert {fold for _, fold in primary.contexts} == {1}

    def local_models(link, target):
        link.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runner, "_junction", local_models)
    replay = runner._replay_view(primary)
    assert isinstance(replay, _ArmView)
    assert replay.source is not source
    assert primary.out == primary_out and source.out == primary_out
    assert replay.out == primary_out / "fresh_replay"
    assert list(replay.contexts) == list(primary.contexts)
    assert all(replay.contexts[key] is primary.contexts[key] for key in primary.contexts)
    assert {fold for _, fold in replay.contexts} == {1}
    assert (replay.out / "models").resolve() != (primary.out / "models").resolve()


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
            (destination / "models/recent_day_analog" / spec["id"]).mkdir(parents=True)

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
    assert all(json.loads((out / "logs/experiments" / f"{spec['id']}.json")
                          .read_text(encoding="utf-8"))["result"]
               ["historical_final_artifact_read"] is False for spec in specs)
    assert (out / "tables" / specs[1]["id"] / "EXPLORE/manifest.json").exists()

    original = first_manifest.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["fields"]["AUC_MAE"] = 0.
    first_manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to replace locked artifact"):
        runner.run_search(prepared, pd.DataFrame(), plan)
    assert first_manifest.read_text(encoding="utf-8") == json.dumps(tampered)

