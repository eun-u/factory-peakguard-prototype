"""Isolated LoRA runner contracts without loading or training the GPU model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from phase_f import goal_foundation as ft
from phase_f.goal_protocol import SEEDS, TARGETS
from phase_f.registry import sha256, write_json


REVISION = "a" * 40


def _plan():
    return {"plan_sha256": "frozen-plan", "execution": {"pinned_model": {"revision": REVISION}}}


def _prepared(tmp_path):
    contexts = {(h, fold): {} for fold in (1, 3, 5, 7, 9, 11, 13, 15)
                for h in range(4, 17)}
    return SimpleNamespace(out=tmp_path, root=Path(__file__).resolve().parents[2],
                           contexts=contexts, split_lock={"lock_sha256": "synthetic"},
                           seal={"raw_sha256": "synthetic"},
                           origins=lambda h, fold, role: pd.DatetimeIndex([pd.Timestamp("2021-04-16")]))


def _seed_audit(prepared, spec, seed, folds):
    from phase_f.models.foundation import QUANTILES
    from phase_f.wf_models import _seed_spec

    child = _seed_spec(spec, seed, "EXPLORE")
    input_spec = {key: value for key, value in child.items()
                  if key not in ("id", "parent", "family", "tier", "point", "note")}
    input_spec.update(model_revision=REVISION, quantiles=QUANTILES,
                      source_sha256=sha256(prepared.root / "phase_f/models/foundation.py"),
                      split_sha256=prepared.split_lock["lock_sha256"])
    cache_id = ft.config_hash(input_spec)[:20]
    cells = []
    for fold in folds:
        path = (prepared.out / "models" / f"chronos_{cache_id}"
                / f"finetune_f{fold}" / "finetuned-ckpt" / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"checkpoint-{seed}-{fold}".encode())
        cells.append({"fold": fold, "future_perturbation_max_abs_difference": 0.0,
                      "checkpoint_files": {"config.json": sha256(path)},
                      "training_timing_status": "measured", "train_seconds": 1.0})
    return {"model_revision": REVISION, "cache_id": cache_id, "holdout_read": False,
            "input_spec": input_spec,
            "fit_cells": cells}


def _full_audit(prepared, spec):
    from phase_f.wf_models import _prediction_identity, _seed_spec

    entries = []
    for seed in SEEDS:
        path = (prepared.out / "predictions/seeds/EXPLORE" / spec["id"]
                / f"seed_{seed}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        origin = pd.Timestamp("2021-04-16")
        pd.DataFrame({"horizon": [4, 16], "fold": [1, 15],
                      "role": ["cal", "score"], "origin": [origin, origin],
                      "target_time": [origin + pd.Timedelta(hours=1),
                                      origin + pd.Timedelta(hours=4)],
                      "pred": [float(seed), float(seed + 1)],
                      "y": [5., 7.], "tau": [10., 10.], "d2": [0, 1],
                      "arm": ["EXPLORE", "EXPLORE"],
                      "model": [spec["id"]] * 2,
                      "fit_mean": [6., 6.], "mase_scale": [1., 1.],
                      "cal_provenance": ["fit_only", "not_cal"]}).to_parquet(path, index=False)
        audit = _seed_audit(prepared, spec, seed, ft._folds(prepared))
        child = _seed_spec(spec, seed, "EXPLORE")
        write_json(path.with_suffix(".json"), {
            "identity": _prediction_identity(prepared, child, "EXPLORE"),
            "sha256": sha256(path), "seed": seed, "arm": "EXPLORE",
            "audit": audit, "development_only": True, "holdout_read": False})
        entries.append({"seed": seed, "audit": audit,
                        "prediction_sha256": sha256(path)})
    return {"arm": "EXPLORE", "n_seeds": 5, "seeds": entries,
            "leakage_test": "requires_family_audit"}


def _mean_frame(prepared, spec):
    first = (prepared.out / "predictions/seeds/EXPLORE" / spec["id"]
             / f"seed_{SEEDS[0]}.parquet")
    frame = pd.read_parquet(first)
    frame["pred"] = np.mean([[float(seed), float(seed + 1)] for seed in SEEDS], axis=0)
    frame["n_seeds"] = 5
    return frame


def test_declared_specs_match_approved_grid_and_separate_goal_contract():
    specs = ft.specs()
    assert [spec["id"] for spec in specs] == list(ft.CONFIGS)
    assert [(spec["context_length"], spec["fit_batch_size"]) for spec in specs] == [
        (512, 16), (2048, 8)]
    assert all(spec["finetune"] == "lora" and spec["num_steps"] == 100
               and spec["learning_rate"] == 1e-5 and spec["seeds"] == list(SEEDS)
               for spec in specs)
    assert ft.CONTRACT["targets"] == TARGETS
    assert "FIT-only" in ft.CONTRACT["training"] and "residual" not in ft.CONTRACT["training"]
    assert "best-checkpoint selection" in ft.CONTRACT["training"]
    assert "early stopping" not in ft.CONTRACT["training"]
    assert ft.CONTRACT["holdout_read"] is False


def test_full_candidate_requires_all_folds_five_seeds_and_numeric_leakage(tmp_path, monkeypatch):
    from phase_f import wf_models

    prepared = _prepared(tmp_path)
    spec = ft.specs()[0]
    monkeypatch.setattr(wf_models, "_validate_frame", lambda *_: None)
    valid = _full_audit(prepared, spec)
    frame = _mean_frame(prepared, spec)
    passed = ft._verify_full(prepared, spec, frame, valid, _plan())
    assert passed["leakage_test"] == "passed"
    assert passed["original_leakage_field"] == "requires_family_audit"
    assert set(passed["seed_metadata_sha256"]) == set(map(str, SEEDS))
    cached = ft._verify_full(prepared, spec, frame, passed, _plan(), cached_mean=True)
    assert cached["seed_metadata_sha256"] == passed["seed_metadata_sha256"]
    assert valid["leakage_test"] == "requires_family_audit"
    missing = copy.deepcopy(valid)
    missing["seeds"][0]["audit"]["fit_cells"].pop()
    with pytest.raises(ValueError, match="folds are incomplete"):
        ft._verify_full(prepared, spec, frame, missing, _plan())
    future = copy.deepcopy(valid)
    future["seeds"][0]["audit"]["fit_cells"][0]["future_perturbation_max_abs_difference"] = .01
    with pytest.raises(ValueError, match="future perturbation"):
        ft._verify_full(prepared, spec, frame, future, _plan())
    duplicate = copy.deepcopy(valid)
    duplicate["seeds"][1]["seed"] = duplicate["seeds"][0]["seed"]
    with pytest.raises(ValueError, match="five distinct"):
        ft._verify_full(prepared, spec, frame, duplicate, _plan())
    bad_cache_id = copy.deepcopy(valid)
    bad_cache_id["seeds"][0]["audit"]["cache_id"] = "0" * 20
    with pytest.raises(ValueError, match="model cache identity"):
        ft._verify_full(prepared, spec, frame, bad_cache_id, _plan())
    extra_recipe = copy.deepcopy(valid)
    extra_recipe["seeds"][0]["audit"]["input_spec"]["unapproved_parameter"] = True
    with pytest.raises(ValueError, match="fit recipe differs"):
        ft._verify_full(prepared, spec, frame, extra_recipe, _plan())
    altered_mean = frame.copy()
    altered_mean.loc[0, "pred"] += 1.0
    with pytest.raises(ValueError, match="physical five-seed mean"):
        ft._verify_full(prepared, spec, altered_mean, valid, _plan())
    with pytest.raises(ValueError, match="physical five-seed mean"):
        ft._verify_full(prepared, spec, altered_mean, passed, _plan(), cached_mean=True)
    missing_metadata = copy.deepcopy(passed)
    missing_metadata.pop("seed_metadata_sha256")
    with pytest.raises(ValueError, match="metadata checksums"):
        ft._verify_full(prepared, spec, frame, missing_metadata, _plan(), cached_mean=True)
    metadata = (tmp_path / "predictions/seeds/EXPLORE" / spec["id"]
                / f"seed_{SEEDS[0]}.json")
    original_metadata = metadata.read_bytes()
    modified = json.loads(original_metadata)
    modified["identity"] = "changed"
    metadata.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction metadata changed"):
        ft._verify_full(prepared, spec, frame, valid, _plan())
    metadata.write_bytes(original_metadata)
    wrong_recipe = copy.deepcopy(valid)
    wrong_recipe["seeds"][0]["audit"]["input_spec"]["learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="fit recipe differs"):
        ft._verify_full(prepared, spec, frame, wrong_recipe, _plan())
    cache_id = valid["seeds"][0]["audit"]["cache_id"]
    checkpoint = (tmp_path / "models" / f"chronos_{cache_id}"
                  / "finetune_f1" / "finetuned-ckpt" / "config.json")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        ft._verify_full(prepared, spec, frame, valid, _plan())
    checkpoint.write_bytes(f"checkpoint-{SEEDS[0]}-1".encode())
    (checkpoint.parent / "extra.bin").write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="file inventory changed"):
        ft._verify_full(prepared, spec, frame, valid, _plan())


def test_interrupted_fit_preserves_unknown_training_time(tmp_path, monkeypatch):
    from phase_f import wf_models

    prepared = _prepared(tmp_path)
    spec = ft.specs()[0]
    audit = _full_audit(prepared, spec)
    frame = _mean_frame(prepared, spec)
    monkeypatch.setattr(wf_models, "_validate_frame", lambda *_: None)
    seed_audit = audit["seeds"][0]["audit"]
    for cell in seed_audit["fit_cells"]:
        cell["train_seconds"] = float("nan")
        cell["training_timing_status"] = "unavailable_after_interrupted_fit_transaction"
    metadata = (tmp_path / "predictions/seeds/EXPLORE" / spec["id"]
                / f"seed_{SEEDS[0]}.json")
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["audit"] = seed_audit
    metadata.write_text(json.dumps(record), encoding="utf-8")
    result = ft._verify_full(prepared, spec, frame, audit, _plan())
    assert result["training_timing_status"] == "UNKNOWN"
    assert np.isnan(result["seeds"][0]["audit"]["fit_cells"][0]["train_seconds"])


def test_smoke_is_one_seed_first_fold_and_never_published_as_candidate(tmp_path, monkeypatch):
    from phase_f import goal_foundation, wf_models, support
    from phase_f.models import foundation

    prepared = _prepared(tmp_path)
    spec = ft.specs()[0]
    plan = _plan()
    calls = []

    def fake_run(view, child):
        calls.append((child, sorted({fold for _, fold in view.contexts})))
        return pd.DataFrame({"horizon": list(range(4, 17)), "pred": [1.] * 13}), _seed_audit(
            prepared, spec, SEEDS[0], [1])

    monkeypatch.setattr(foundation, "run", fake_run)
    monkeypatch.setattr(wf_models, "_validate_frame", lambda *_: None)
    monkeypatch.setattr(support, "validate_support", lambda *_: None)
    first = goal_foundation._smoke(prepared, plan, spec)
    assert len(calls) == 1 and calls[0][1] == [1]
    assert calls[0][0]["seed"] == SEEDS[0]
    assert calls[0][0]["id"] == spec["id"] + "__wf_explore__seed42"
    assert first["candidate_scored"] is False
    assert first["mean_forecast_published"] is False
    assert first["goal_achieved"] is False
    assert not (tmp_path / "predictions/EXPLORE" / f"{spec['id']}.parquet").exists()
    second = goal_foundation._smoke(prepared, plan, spec)
    assert first == second and len(calls) == 1


def test_immutable_execution_plan_rejects_source_change(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(ft, "_sources", lambda root: {"source.py": "first"})
    monkeypatch.setattr(ft, "_execution", lambda prepared: {"runtime": "synthetic"})
    a = ft._plan(prepared, {"F0-1-R1": "anchor"})
    b = ft._plan(prepared, {"F0-1-R1": "anchor"})
    assert a == b
    assert a["plan_sha256"] == ft.config_hash({k: v for k, v in a.items()
                                               if k != "plan_sha256"})
    monkeypatch.setattr(ft, "_sources", lambda root: {"source.py": "changed"})
    with pytest.raises(RuntimeError, match="Refusing to replace locked artifact"):
        ft._plan(prepared, {"F0-1-R1": "anchor"})


def test_live_plan_rejects_source_or_runtime_change_after_fit(tmp_path, monkeypatch):
    from phase_f.wf_evaluation import BASELINES, prediction_path

    prepared = _prepared(tmp_path)
    monkeypatch.setattr(ft, "_sources", lambda root: {"source.py": "first"})
    monkeypatch.setattr(ft, "_execution", lambda prepared: {"runtime": "first"})
    baselines = {}
    for key in BASELINES.values():
        path = prediction_path(prepared, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"pred": [1.0]}).to_parquet(path, index=False)
        baselines[key] = sha256(path)
        write_json(path.with_suffix(".json"), {"arm": "EXPLORE", "sha256": baselines[key]})
    plan = ft._plan(prepared, baselines)
    ft._verify_live_plan(prepared, plan)
    monkeypatch.setattr(ft, "_sources", lambda root: {"source.py": "changed"})
    with pytest.raises(RuntimeError, match="frozen execution plan changed"):
        ft._verify_live_plan(prepared, plan)
    monkeypatch.setattr(ft, "_sources", lambda root: {"source.py": "first"})
    monkeypatch.setattr(ft, "_execution", lambda prepared: {"runtime": "changed"})
    with pytest.raises(RuntimeError, match="frozen execution plan changed"):
        ft._verify_live_plan(prepared, plan)
    monkeypatch.setattr(ft, "_execution", lambda prepared: {"runtime": "first"})
    path = prediction_path(prepared, BASELINES["R1"])
    original = path.read_bytes()
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="frozen baseline prediction changed"):
        ft._verify_live_plan(prepared, plan)
    path.write_bytes(original)
    metadata = path.with_suffix(".json")
    write_json(metadata, {"arm": "CONFIRM", "sha256": baselines[BASELINES["R1"]]})
    with pytest.raises(ValueError, match="Changed weekly prediction identity"):
        ft._verify_live_plan(prepared, plan)
    write_json(metadata, {"arm": "EXPLORE", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="Changed weekly prediction identity"):
        ft._verify_live_plan(prepared, plan)


def test_search_does_not_publish_mean_when_five_seed_audit_is_incomplete(tmp_path, monkeypatch):
    from phase_f import wf_models, support

    prepared = _prepared(tmp_path)
    spec = ft.specs()[0]
    plan = _plan()
    with pytest.raises(RuntimeError, match="Run --stage smoke"):
        ft._search(prepared, plan, spec)
    smoke_path = tmp_path / "smoke" / spec["id"] / "manifest.json"
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ft, "_smoke", lambda *_: {"candidate_scored": False,
                                                "mean_forecast_published": False})
    monkeypatch.setattr(ft, "_verify_live_plan", lambda *_: None)
    monkeypatch.setattr(support, "validate_support", lambda *_: None)
    monkeypatch.setattr(wf_models, "run", lambda *_args, **_kwargs: (
        pd.DataFrame({"dummy": [1]}), {"arm": "EXPLORE", "n_seeds": 1, "seeds": []}))
    with pytest.raises(ValueError, match="five distinct"):
        ft._search(prepared, plan, spec)
    assert not (tmp_path / "predictions/EXPLORE" / f"{spec['id']}.parquet").exists()


def test_unknown_only_rejected():
    with pytest.raises(ValueError, match="Unknown LoRA configuration"):
        ft._selected({"specs": ft.specs()}, "unknown")
