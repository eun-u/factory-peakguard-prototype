"""Frozen parent preflight and nonlinear five-seed guard integration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f import goal_guard, goal_r1
from phase_f.goal_protocol import CONTRACT, SEEDS, initial_specs
from phase_f.harness import make_frame
from phase_f.registry import config_hash, sha256, write_json
from phase_f.wf_evaluation import BASELINES, prediction_path
from phase_f.wf_models import _mean_frames


def _view(tmp_path):
    root = Path(__file__).resolve().parents[2]
    roles = {
        "fit": pd.date_range("2021-03-01", periods=5, freq="15min", name="origin"),
        "stop": pd.date_range("2021-04-01", periods=3, freq="15min", name="origin"),
        "cal": pd.date_range("2021-04-15", periods=2, freq="15min", name="origin"),
        "score": pd.date_range("2021-04-19", periods=2, freq="15min", name="origin"),
    }
    contexts = {}
    for h in range(4, 17):
        origins = pd.DatetimeIndex(sorted(set().union(*(set(v) for v in roles.values()))), name="origin")
        contexts[(h, 0)] = {
            **roles, "y": pd.Series(55 + np.arange(len(origins), dtype=float), index=origins),
            "target_time": pd.Series(origins + pd.Timedelta(minutes=h*15), index=origins),
            "x": pd.DataFrame({"slot7d": np.full(len(origins), 50.)}, index=origins),
            "tau": 60., "d2": pd.Series(False, index=origins),
            "summary": {"arm": "EXPLORE"},
        }
    view = SimpleNamespace(root=root, out=tmp_path / "guard", contexts=contexts,
                           split_lock={"lock_sha256": "synthetic-week-lock"})
    view.origins = lambda h, fold, role: contexts[(h, fold)][role]
    return view


def _frame(view, model, prediction, *, is_r1=False):
    frames = []
    for (h, fold), context in view.contexts.items():
        for role in ("cal", "score"):
            origins = context[role]
            values = np.full(len(origins), float(prediction))
            extra = ({"q50": np.full(len(origins), 50.),
                      "q90": np.full(len(origins), 70.),
                      "q95": np.full(len(origins), 75.)} if is_r1 else
                     {"r1": np.full(len(origins), 50.),
                      "correction": np.full(len(origins), 1000.)})
            frames.append(make_frame(context, origins, h, fold, values, model, role, **extra))
    return pd.concat(frames, ignore_index=True)


def _write_prediction(path, frame, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {**metadata, "sha256": sha256(path), "arm": "EXPLORE"})


def _parent_artifacts(view, monkeypatch):
    parent_dir = view.root / "outputs/phase_f/goal_r1_guard_fixture_parent"
    # The fixture lives in pytest's temp directory while the immutable
    # preflight resolves parent paths from a root.  Give it a local root with
    # copied source hashes injected for the synthetic source check.
    root = view.out.parent
    view.root = root
    parent_dir = root / "outputs/phase_f/goal_r1_guard_fixture_parent"
    parent_dir.mkdir(parents=True)
    monkeypatch.setattr(goal_r1, "_sources", lambda ignored: {"synthetic-source": "fixed"})
    monkeypatch.setattr(goal_guard, "_sources", lambda ignored: {"synthetic-guard-source": "fixed"})
    baseline_path = prediction_path(view, BASELINES["R1"])
    _write_prediction(baseline_path, _frame(view, BASELINES["R1"], 50., is_r1=True),
                      {"identity": "synthetic-r1-baseline"})
    baseline_hashes = {BASELINES["R1"]: sha256(baseline_path)}
    for name in ("B5", "M1"):
        path = prediction_path(view, BASELINES[name])
        _write_prediction(path, _frame(view, BASELINES[name], 50.),
                          {"identity": "synthetic-" + name})
        baseline_hashes[BASELINES[name]] = sha256(path)
    monkeypatch.setattr(goal_r1, "copy_baselines", lambda original, prepared: baseline_hashes)
    specs = initial_specs()
    parent_plan = {"contract": CONTRACT, "specs": specs, "seeds": list(SEEDS),
                   "sources": goal_r1._sources(root),
                   "baseline_prediction_sha256": baseline_hashes,
                   "split_sha256": view.split_lock["lock_sha256"],
                   "rolling_paths_sha256": "synthetic-path-table"}
    write_json(parent_dir / "logs/execution_plan.json", parent_plan)
    write_json(parent_dir / "logs/driver_status.json", {
        "status": "completed_explore_stage", "full_explore_search_complete": True,
        "holdout_read": False, "goal_achieved": False})
    source = {"anchor_original_future_perturbation_max_abs_difference": 0.0,
              "anchor_prediction_sha256": "synthetic-anchor"}
    identity = {"anchor_source": source, "source_sha256": {"synthetic": "fixed"},
                "execution": {"device": "cpu"}, "model": {"revision": "synthetic-pinned"}}
    producer = {"complete": True, "identity": identity,
                "identity_hash": config_hash(identity),
                "paths_sha256": "synthetic-path-table", "parts": [], "holdout_read": False,
                "path_rows": len(pd.read_parquet(baseline_path)),
                "future_perturbation_max_abs_difference": 0.0,
                "anchor_recipe_check": {"status": "passed"}}
    producer["manifest_sha256"] = config_hash(producer)
    write_json(parent_dir / "r1_paths/manifest.json", producer)
    offsets = dict(zip(SEEDS, (-10., -5., 0., 5., 10.)))
    rolling_digest = goal_guard._producer_lock(parent_dir, parent_plan)[2]
    for spec in specs:
        parent_id = spec["id"]
        identity = config_hash({"spec": spec, "rolling": rolling_digest,
                                "plan": config_hash(parent_plan)})
        seed_frames, seed_rows = [], []
        for seed in SEEDS:
            child = copy.deepcopy(spec)
            child.update(seed=seed, rolling_audit_sha256=rolling_digest)
            child["id"] = parent_id + "__seed" + str(seed)
            seed_identity = config_hash({"mean_identity": identity, "child": child})
            frame = _frame(view, child["id"], 50. + offsets[seed])
            path = parent_dir / "predictions/seeds/EXPLORE" / parent_id / f"seed_{seed}.parquet"
            _write_prediction(path, frame, {"identity": seed_identity,
                                            "audit": {"seed": seed, "leakage_test": "passed",
                                                      "rolling_audit_sha256": rolling_digest}})
            seed_frames.append(frame)
            seed_rows.append({"seed": seed, "prediction_sha256": sha256(path)})
        mean = _mean_frames(seed_frames, parent_id)
        mean.drop(columns="correction", inplace=True)
        mean["applied_correction"] = mean.pred - mean.r1
        mean_path = parent_dir / "predictions/EXPLORE" / f"{parent_id}.parquet"
        _write_prediction(mean_path, mean, {"identity": identity,
                "audit": {"n_seeds": 5, "seeds": seed_rows,
                          "rolling_audit_sha256": rolling_digest, "leakage_test": "passed"}})
        table_dir = parent_dir / "tables" / parent_id / "EXPLORE"
        table_dir.mkdir(parents=True)
        table = table_dir / "pooled.csv"
        table.write_text("dataset,MAE\nD1,5.0\n", encoding="utf-8")
        scored = {"n_seeds": 5, "leakage_test": "passed", "goal_achieved": False,
                  "holdout_read": False, "arm": "EXPLORE",
                  "prediction_sha256": sha256(mean_path),
                  "artifacts": {"pooled.csv": sha256(table)}}
        write_json(table_dir / "manifest.json", scored)
        write_json(parent_dir / "logs/experiments" / f"{parent_id}.json", {
            "spec": spec, "status": "completed",
            "result": scored})
    return parent_dir


def _score(prepared, spec, mean, audit):
    assert audit["n_seeds"] == 5
    fields = {"AUC_MAE": float(mean.pred.mean()), "AUC_PeakMAE": 12.,
              "h4_MAE": 4., "h16_MAE": 6., "AUC_nMAE": .06}
    return {"fields": fields, "targets_met": False, "n_seeds": 5,
            "goal_achieved": False, "holdout_read": False}


def test_seed_level_nonlinear_guard_and_truth_perturbation(tmp_path):
    view = _view(tmp_path)
    original = goal_guard._ordered(_frame(view, BASELINES["R1"], 50., is_r1=True), name="R1")
    negative = _frame(view, "seed-a", 40.)
    positive = _frame(view, "seed-b", 60.)
    a = goal_guard.apply_guard(negative, original, goal_guard.POLICIES[0], "guard")
    b = goal_guard.apply_guard(positive, original, goal_guard.POLICIES[0], "guard")
    assert np.allclose(a.pred, 50.) and np.allclose(b.pred, 60.)
    assert np.allclose(_mean_frames([a, b], "guard").pred, 55.)
    # Applying the nonlinear rule after averaging would incorrectly give 50.
    averaged_parent = _mean_frames([negative, positive], "parent")
    assert np.allclose(goal_guard.apply_guard(
        averaged_parent, original, goal_guard.POLICIES[0], "guard").pred, 50.)
    changed_truth = negative.copy()
    changed_truth["y"] += 10000.
    assert np.array_equal(goal_guard.apply_guard(
        negative, original, goal_guard.POLICIES[2], "guard").pred.to_numpy(),
        goal_guard.apply_guard(changed_truth, original, goal_guard.POLICIES[2], "guard").pred.to_numpy())


def test_all_parents_preflight_then_full_guard_grid_and_resume(tmp_path, monkeypatch):
    view = _view(tmp_path)
    parent_dir = _parent_artifacts(view, monkeypatch)
    evidence = goal_guard.preflight(None, view, parent_namespace=parent_dir.name)
    assert evidence["preflight_complete"] is True and len(evidence["parents"]) == 10
    goal_guard.run_search(view, evidence, score_fn=_score)
    plan = json.loads((view.out / "logs/execution_plan.json").read_text(encoding="utf-8"))
    assert len(plan["specs"]) == 30 and len({spec["id"] for spec in plan["specs"]}) == 30
    assert plan["plan_sha256"] == config_hash({key: value for key, value in plan.items()
                                               if key != "plan_sha256"})
    contract = json.loads((view.out / "logs/target_contract.json").read_text(encoding="utf-8"))
    progress = json.loads((view.out / "logs/goal_progress.json").read_text(encoding="utf-8"))
    assert plan["contract"] == contract == progress["contract"]
    assert "no new fitting" in contract["training"]
    assert contract["targets"]["AUC_MAE"] == 4.5
    example = plan["specs"][0]["id"]
    result = pd.read_parquet(prediction_path(view, example))
    assert np.allclose(result.pred, 53.)  # Mean of five individually guarded seeds.
    assert np.allclose(result.applied_correction, 3.)
    assert "correction" not in result and "guard_triggered" not in result
    before = {p.name: sha256(p) for p in (view.out / "predictions/EXPLORE").glob("*.parquet")}
    goal_guard.run_search(view, evidence, score_fn=_score)
    after = {p.name: sha256(p) for p in (view.out / "predictions/EXPLORE").glob("*.parquet")}
    assert before == after
    own_metadata = view.out / "predictions/seeds/EXPLORE" / example / f"seed_{SEEDS[0]}.json"
    mean_path = prediction_path(view, example)
    mean_meta = mean_path.with_suffix(".json")
    saved_mean, saved_mean_meta = mean_path.read_bytes(), mean_meta.read_bytes()
    mean_path.unlink(); mean_meta.unlink()
    valid_seed_meta = own_metadata.read_bytes()
    for mutation in ("policy", "leakage_test"):
        corrupted = json.loads(valid_seed_meta)
        if mutation == "policy":
            corrupted["audit"]["policy"]["name"] = "unfrozen-policy"
        else:
            corrupted["audit"]["leakage_test"] = "unverified"
        write_json(own_metadata, corrupted)
        with pytest.raises(ValueError, match="Cached guard seed .*metadata changed"):
            goal_guard.run_search(view, evidence, score_fn=_score)
    own_metadata.write_bytes(valid_seed_meta)
    mean_path.write_bytes(saved_mean); mean_meta.write_bytes(saved_mean_meta)
    intact = own_metadata.read_bytes()
    own_metadata.unlink()
    with pytest.raises(ValueError, match="Cached guard seed .*metadata is missing"):
        goal_guard.run_search(view, evidence, score_fn=_score)
    own_metadata.write_bytes(intact)
    # A cached guarded mean cannot mask changed parent seed bytes on resume.
    seed_path = next((parent_dir / "predictions/seeds/EXPLORE" / initial_specs()[0]["id"]).glob("*.parquet"))
    with seed_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Frozen parent seed bytes changed"):
        goal_guard.run_search(view, evidence, score_fn=_score)


def test_missing_parent_completion_fails_before_guard_plan_or_score(tmp_path, monkeypatch):
    view = _view(tmp_path)
    parent_dir = _parent_artifacts(view, monkeypatch)
    missing = parent_dir / "logs/experiments" / f"{initial_specs()[-1]['id']}.json"
    record = json.loads(missing.read_text(encoding="utf-8"))
    record["status"] = "running"
    write_json(missing, record)
    with pytest.raises(ValueError, match="incomplete or unaudited"):
        goal_guard.preflight(None, view, parent_namespace=parent_dir.name)
    assert not (view.out / "logs/execution_plan.json").exists()
