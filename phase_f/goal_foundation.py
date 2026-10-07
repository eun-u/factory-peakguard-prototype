"""Isolated, five-seed Chronos-2 LoRA candidates for the accepted point goal.

The first-fold smoke proves that a declared recipe can train and predict. It
never scores or publishes a candidate. Search reuses that model checkpoint,
then requires all eight EXPLORE folds and five distinct seed paths before a
mean forecast receives the fixed point-goal evaluation. No CONFIRM/holdout.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from phase_f.goal_protocol import SEEDS, TARGETS, meets_targets
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f import goal_r1


NAMESPACE = "goal_r1_ft_v1"
CONFIGS = {
    "FG-R2-lora-c512-s100": (512, 16),
    "FG-R2-lora-c2048-s100": (2048, 8),
}
CONTRACT = {
    "protocol": NAMESPACE,
    "targets": TARGETS,
    "target_authorization": "User accepted competitive point-forecast targets and requested execution",
    "primary_geometry": "walkforward_v2 EXPLORE; 13 horizons h4..h16; eight even ISO weeks",
    "ranking": "metric of five-seed mean forecast; individual seed mean and SD disclosed",
    "training": "Chronos-2 LoRA; FIT-only training windows, STOP-only validation and best-checkpoint selection",
    "selection_arm": "EXPLORE",
    "confirmation": "Independent locked WF CONFIRM and pc3 auxiliary before achievement",
    "holdout_read": False,
    "commercial_qualification": "Requires independent site/season field validation",
}
CONTRACT["contract_sha256"] = config_hash(CONTRACT)


def specs() -> list[dict]:
    """Rename two previously approved 100-step settings; retain all fit axes."""
    from phase_f.wf_plan import foundation_fine

    approved = {int(row["context_length"]): row for row in foundation_fine(2)
                if row.get("finetune") == "lora" and row.get("learning_rate") == 1e-5
                and row.get("num_steps") == 100 and int(row["context_length"]) in (512, 2048)}
    if set(approved) != {512, 2048}:
        raise RuntimeError("Approved 100-step Chronos LoRA settings changed")
    result = []
    for exp_id, (length, batch) in CONFIGS.items():
        spec = {**approved[length], "id": exp_id, "fit_batch_size": batch,
                "seeds": list(SEEDS)}
        if (spec["adapter"] != "foundation" or spec["kind"] != "chronos2"
                or spec["point"] != "median" or spec["finetune"] != "lora"
                or spec["num_steps"] != 100 or spec["context_length"] != length):
            raise RuntimeError("Chronos LoRA recipe drifted from approved Phase F grid")
        result.append(spec)
    return result


def _execution(prepared) -> dict:
    import torch
    from phase_f.goal_r1_paths import _execution_fingerprint, _snapshot
    from phase_f.models.foundation import require_finetune_mode

    require_finetune_mode("lora")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, pinned = _snapshot(prepared)
    execution = _execution_fingerprint(device)
    execution["peft_version"] = importlib.metadata.version("peft")
    execution["transformers_version"] = importlib.metadata.version("transformers")
    if device == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        execution["gpu"] = {"name": properties.name,
                            "total_memory": properties.total_memory,
                            "capability": list(torch.cuda.get_device_capability())}
    return {"pinned_model": pinned, "runtime": execution}


def _sources(root) -> dict:
    names = ("phase_f/goal_foundation.py", "phase_f/goal_r1.py",
             "phase_f/goal_protocol.py", "phase_f/goal_r1_paths.py",
             "phase_f/wf_models.py", "phase_f/wf_harness.py",
             "phase_f/wf_run.py", "phase_f/wf_metrics.py",
             "phase_f/wf_plan.py", "phase_f/support.py", "phase_f/registry.py",
             "phase_f/wf_evaluation.py", "phase_f/models/foundation.py",
             "phase_f/runtime.py", "phase_f/harness.py",
             "phase_c/data.py", "src/targets.py")
    root = Path(root)
    return {name: sha256(root / name) for name in names}


def _plan(prepared, baselines: dict) -> dict:
    record = {"contract": CONTRACT, "specs": specs(), "seeds": list(SEEDS),
              "split_sha256": prepared.split_lock["lock_sha256"],
              "seal_sha256": config_hash(prepared.seal),
              "baseline_prediction_sha256": baselines,
              "source_sha256": _sources(prepared.root),
              "execution": _execution(prepared),
              "arm": "EXPLORE", "holdout_read": False}
    record["plan_sha256"] = config_hash(record)
    write_json(prepared.out / "logs/execution_plan.json", record, exclusive=True)
    return record


def _verify_live_plan(prepared, plan: dict) -> None:
    """Refuse publication if the frozen training inputs changed during a long fit."""
    if (_sources(prepared.root) != plan["source_sha256"]
            or _execution(prepared) != plan["execution"]
            or prepared.split_lock["lock_sha256"] != plan["split_sha256"]
            or config_hash(prepared.seal) != plan["seal_sha256"]):
        raise RuntimeError("LoRA frozen execution plan changed during training or resume")
    from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path

    expected = plan["baseline_prediction_sha256"]
    if set(expected) != set(BASELINES.values()):
        raise RuntimeError("LoRA frozen baseline set changed")
    for key, digest in expected.items():
        path = prediction_path(prepared, key)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"LoRA frozen baseline prediction changed: {key}")
        load_predictions(prepared, key, "EXPLORE")  # Verify metadata arm and self SHA.


def _read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected(plan, only) -> list[dict]:
    found = {spec["id"]: spec for spec in plan["specs"]}
    if only is not None:
        if only not in found:
            raise ValueError("Unknown LoRA configuration: " + str(only))
        return [found[only]]
    return list(found.values())


def _folds(prepared) -> list[int]:
    horizons = {int(h) for h, _ in prepared.contexts}
    if horizons != set(range(4, 17)):
        raise ValueError("LoRA goal requires all 13 h4..h16 horizons")
    folds = sorted({int(f) for _, f in prepared.contexts})
    if len(folds) != 8 or any({h for h, f in prepared.contexts if f == fold} != horizons
                              for fold in folds):
        raise ValueError("LoRA goal requires eight complete EXPLORE weeks")
    return folds


def _verify_cells(prepared, audit: dict, spec: dict, folds: list[int], seed: int,
                  expected_revision: str) -> None:
    from phase_f.models.foundation import QUANTILES
    from phase_f.wf_models import _seed_spec

    if audit.get("model_revision") != expected_revision or audit.get("holdout_read") is not False:
        raise ValueError("Chronos model provenance or holdout audit missing")
    input_spec = audit.get("input_spec", {})
    expected_spec = _seed_spec(spec, seed, "EXPLORE")
    expected_cache_spec = {key: value for key, value in expected_spec.items()
                           if key not in ("id", "parent", "family", "tier", "point", "note")}
    expected_cache_spec.update(
        model_revision=expected_revision, quantiles=QUANTILES,
        source_sha256=sha256(Path(prepared.root) / "phase_f/models/foundation.py"),
        split_sha256=prepared.split_lock["lock_sha256"])
    if (input_spec.get("finetune") != "lora" or input_spec.get("seed") != seed
            or input_spec.get("context_length") != spec["context_length"]
            or input_spec.get("fit_batch_size") != spec["fit_batch_size"]
            or input_spec.get("learning_rate") != spec["learning_rate"]
            or input_spec.get("num_steps") != spec["num_steps"]
            or input_spec.get("prediction_length", 96) != 96
            or input_spec.get("covariates", False) is not False
            or input_spec.get("model_revision") != expected_revision
            or input_spec.get("quantiles") != QUANTILES
            or input_spec.get("arm") != "EXPLORE"
            or config_hash(input_spec) != config_hash(expected_cache_spec)):
        raise ValueError("Chronos fit recipe differs from frozen LoRA configuration")
    cells = audit.get("fit_cells", [])
    if sorted(cell.get("fold") for cell in cells) != folds:
        raise ValueError("Chronos LoRA fit/stop folds are incomplete")
    cache_id = audit.get("cache_id")
    if (not isinstance(cache_id, str)
            or cache_id != config_hash(input_spec)[:20]):
        raise ValueError("Chronos LoRA model cache identity is invalid")
    for cell in cells:
        difference = cell.get("future_perturbation_max_abs_difference")
        if (isinstance(difference, bool) or not isinstance(difference, (int, float, np.number))
                or not np.isfinite(difference) or difference != 0.0):
            raise ValueError("Chronos LoRA fitted-pipeline future perturbation did not pass")
        files = cell.get("checkpoint_files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Chronos LoRA checkpoint files are unrecorded")
        base = (Path(prepared.out) / "models" / f"chronos_{cache_id}"
                / f"finetune_f{cell['fold']}" / "finetuned-ckpt")
        for name, expected_sha in files.items():
            if not isinstance(name, str):
                raise ValueError("Unsafe Chronos LoRA checkpoint path")
            relative = PurePosixPath(name)
            if ("\\" in name or ":" in name
                    or relative.is_absolute() or ".." in relative.parts
                    or not relative.parts):
                raise ValueError("Unsafe Chronos LoRA checkpoint path")
            path = base.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(base.resolve()):
                raise ValueError("Chronos LoRA checkpoint escaped its model directory")
            if not path.is_file() or sha256(path) != expected_sha:
                raise ValueError("Chronos LoRA checkpoint bytes changed")
        physical = {}
        for path in sorted(base.rglob("*")):
            if path.is_file():
                if not path.resolve().is_relative_to(base.resolve()):
                    raise ValueError("Chronos LoRA checkpoint escaped its model directory")
                physical[path.relative_to(base).as_posix()] = sha256(path)
        if physical != files:
            raise ValueError("Chronos LoRA checkpoint file inventory changed")
        timing = cell.get("training_timing_status")
        seconds = cell.get("train_seconds")
        if timing == "measured":
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float, np.number))
                    or not np.isfinite(seconds) or seconds < 0):
                raise ValueError("Chronos LoRA fit completion timing is unverified")
        elif timing == "unavailable_after_interrupted_fit_transaction":
            if not isinstance(seconds, (float, np.floating)) or not np.isnan(seconds):
                raise ValueError("Interrupted Chronos LoRA timing must remain unknown")
        else:
            raise ValueError("Chronos LoRA fit completion timing is unverified")


def _verify_full(prepared, spec: dict, frame: pd.DataFrame, audit: dict,
                 plan: dict, *, cached_mean: bool = False) -> dict:
    from phase_f.wf_models import FRAME_KEY, _prediction_identity, _seed_spec, _validate_frame
    from phase_f.wf_evaluation import verified_leakage_status

    folds = _folds(prepared)
    seeds = audit.get("seeds", [])
    if (audit.get("arm") != "EXPLORE" or audit.get("n_seeds") != 5
            or [entry.get("seed") for entry in seeds] != list(SEEDS)):
        raise ValueError("LoRA candidate requires five distinct frozen seeds")
    expected_revision = plan["execution"]["pinned_model"]["revision"]
    metadata_sha = {}
    cache_ids = set()
    reference = None
    predictions = []
    cohort_columns = ("y", "tau", "d2", "arm", "fit_mean", "mase_scale",
                      "cal_provenance")
    for entry in seeds:
        seed = int(entry["seed"])
        child = _seed_spec(spec, seed, "EXPLORE")
        _verify_cells(prepared, entry.get("audit", {}), spec, folds, seed,
                      expected_revision)
        cache_id = entry["audit"]["cache_id"]
        if cache_id in cache_ids:
            raise ValueError("LoRA five seeds share a model cache identity")
        cache_ids.add(cache_id)
        path = (prepared.out / "predictions/seeds/EXPLORE" / spec["id"]
                / f"seed_{seed}.parquet")
        if not path.is_file() or sha256(path) != entry.get("prediction_sha256"):
            raise ValueError("LoRA seed prediction artifact changed")
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise ValueError("LoRA seed prediction metadata is missing")
        metadata = _read(metadata_path)
        if (metadata.get("identity") != _prediction_identity(prepared, child, "EXPLORE")
                or metadata.get("sha256") != entry["prediction_sha256"]
                or metadata.get("seed") != seed or metadata.get("arm") != "EXPLORE"
                or config_hash(metadata.get("audit")) != config_hash(entry["audit"])
                or metadata.get("development_only") is not True
                or metadata.get("holdout_read") is not False):
            raise ValueError("LoRA seed prediction metadata changed")
        metadata_sha[str(seed)] = sha256(metadata_path)
        physical = pd.read_parquet(path)
        required = set(FRAME_KEY) | set(cohort_columns) | {"pred"}
        if not required <= set(physical) or physical.duplicated(list(FRAME_KEY)).any():
            raise ValueError("LoRA seed prediction cohort is incomplete or duplicated")
        physical = physical.sort_values(list(FRAME_KEY), kind="stable").reset_index(drop=True)
        if reference is None:
            reference = physical.loc[:, [*FRAME_KEY, *cohort_columns]].copy()
        else:
            for column in (*FRAME_KEY, *cohort_columns):
                if not np.array_equal(physical[column].to_numpy(), reference[column].to_numpy()):
                    raise ValueError(f"LoRA seed prediction cohort differs: {column}")
        values = physical.pred.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("LoRA seed prediction contains nonfinite values")
        predictions.append(values)
    if not required <= set(frame):
        raise ValueError("LoRA mean prediction cohort is incomplete or duplicated")
    ordered_mean = frame.sort_values(list(FRAME_KEY), kind="stable").reset_index(drop=True)
    if ordered_mean.duplicated(list(FRAME_KEY)).any():
        raise ValueError("LoRA mean prediction cohort is incomplete or duplicated")
    for column in (*FRAME_KEY, *cohort_columns):
        if not np.array_equal(ordered_mean[column].to_numpy(), reference[column].to_numpy()):
            raise ValueError(f"LoRA mean prediction cohort differs: {column}")
    if not np.array_equal(ordered_mean.pred.to_numpy(float), np.mean(predictions, axis=0)):
        raise ValueError("LoRA mean prediction differs from physical five-seed mean")
    if cached_mean and audit.get("seed_metadata_sha256") != metadata_sha:
        raise ValueError("Cached LoRA mean lacks original seed metadata checksums")
    if verified_leakage_status(spec, audit) != "passed":
        raise ValueError("LoRA candidate leakage evidence is incomplete")
    _validate_frame(prepared, frame, spec["id"], "EXPLORE")
    result = copy.deepcopy(audit)
    result["leakage_test"] = "passed"
    result["original_leakage_field"] = audit.get("original_leakage_field",
                                                  audit.get("leakage_test"))
    result["seed_metadata_sha256"] = metadata_sha
    result["training_timing_status"] = (
        "UNKNOWN" if any(cell["training_timing_status"] != "measured"
                         for entry in seeds for cell in entry["audit"]["fit_cells"])
        else "measured")
    return result


def _smoke(prepared, plan: dict, spec: dict) -> dict:
    from phase_f.wf_models import _ArmView, _seed_spec, _validate_frame
    from phase_f.models.foundation import run as foundation_run
    from phase_f.support import validate_support

    folds = _folds(prepared)
    first = folds[0]
    view = _ArmView(prepared, {key: context for key, context in prepared.contexts.items()
                               if int(key[1]) == first})
    validate_support(view, spec)
    child = _seed_spec(spec, SEEDS[0], "EXPLORE")
    destination = prepared.out / "smoke" / spec["id"]
    manifest = destination / "manifest.json"
    frame_path = destination / "seed42_first_fold.parquet"
    expected_revision = plan["execution"]["pinned_model"]["revision"]
    if manifest.exists():
        if not frame_path.is_file():
            raise RuntimeError("LoRA smoke manifest lost its physical forecast; preserve namespace")
        record = _read(manifest)
        if (record.get("plan_sha256") != plan["plan_sha256"]
                or record.get("spec") != spec or record.get("fold") != first
                or record.get("prediction_sha256") != sha256(frame_path)):
            raise RuntimeError("LoRA smoke evidence changed; preserve this namespace")
        _verify_cells(prepared, record["audit"], spec, [first], SEEDS[0],
                      expected_revision)
        _validate_frame(view, pd.read_parquet(frame_path), child["id"], "EXPLORE")
        return record
    if frame_path.exists():
        orphan = frame_path.with_name(frame_path.stem + ".orphan-" + sha256(frame_path)[:12]
                                      + frame_path.suffix)
        if orphan.exists():
            raise RuntimeError("Duplicate orphan LoRA smoke forecast")
        frame_path.rename(orphan)
    frame, audit = foundation_run(view, child)
    _verify_cells(prepared, audit, spec, [first], SEEDS[0], expected_revision)
    _validate_frame(view, frame, child["id"], "EXPLORE")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = frame_path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(frame_path)
    record = {"status": "smoke_passed", "spec": spec,
              "plan_sha256": plan["plan_sha256"], "fold": first, "seed": SEEDS[0],
              "horizons": list(range(4, 17)), "rows": len(frame),
              "prediction_sha256": sha256(frame_path), "audit": audit,
              "candidate_scored": False, "mean_forecast_published": False,
              "goal_achieved": False, "holdout_read": False}
    write_json(manifest, record, exclusive=True)
    print("LORA_SMOKE_READY", spec["id"], "fold", first, flush=True)
    return record


def _verify_scored(prepared, spec: dict, forecast_sha: str) -> dict | None:
    destination = prepared.out / "tables" / spec["id"] / "EXPLORE"
    manifest = destination / "manifest.json"
    if not manifest.is_file():
        return None
    record = _read(manifest)
    if (record.get("candidate") != spec["id"]
            or record.get("prediction_sha256") != forecast_sha
            or record.get("arm") != "EXPLORE"
            or record.get("n_seeds") != 5
            or record.get("leakage_test") != "passed"
            or record.get("goal_achieved") is not False
            or record.get("holdout_read") is not False
            or record.get("targets_met") != meets_targets(record.get("fields", {}))["targets_met"]):
        raise RuntimeError("LoRA scored result changed or is incomplete")
    expected = {"auc.csv", "pooled.csv", "fold.csv", "paired_ci.csv", "seed_metrics.csv"}
    if not expected <= set(record.get("artifacts", {})):
        raise RuntimeError("LoRA scored result lacks required comparison tables")
    if any(sha256(destination / name) != digest
           for name, digest in record.get("artifacts", {}).items()):
        raise RuntimeError("LoRA score table bytes changed")
    return record


def _search(prepared, plan: dict, spec: dict) -> dict:
    from phase_f.wf_models import run as run_model
    from phase_f.goal_r1 import _cached, _publish, score
    from phase_f.wf_evaluation import prediction_path

    if not (prepared.out / "smoke" / spec["id"] / "manifest.json").is_file():
        raise RuntimeError("Run --stage smoke for this LoRA configuration before search")
    smoke = _smoke(prepared, plan, spec)
    if smoke["candidate_scored"] or smoke["mean_forecast_published"]:
        raise ValueError("Partial LoRA smoke cannot be a scored candidate")
    identity = config_hash({"plan_sha256": plan["plan_sha256"], "spec": spec,
                            "arm": "EXPLORE", "five_seed_full_weeks": True})
    path = prediction_path(prepared, spec["id"])
    cached = _cached(path, identity)
    if cached is None:
        from phase_f.support import validate_support

        validate_support(prepared, spec)
        frame, raw_audit = run_model(prepared, spec, arm="EXPLORE")
        _verify_live_plan(prepared, plan)
        audit = _verify_full(prepared, spec, frame, raw_audit, plan)
        _verify_live_plan(prepared, plan)
        _publish(path, frame, {"identity": identity, "audit": audit,
                               "plan_sha256": plan["plan_sha256"],
                               "holdout_read": False})
    else:
        frame, raw_audit = cached
        _verify_live_plan(prepared, plan)
        audit = _verify_full(prepared, spec, frame, raw_audit, plan,
                             cached_mean=True)
    _verify_live_plan(prepared, plan)
    forecast_sha = sha256(path)
    result = _verify_scored(prepared, spec, forecast_sha)
    if result is None:
        result = score(prepared, spec, frame, audit)
        _verify_live_plan(prepared, plan)
        _verify_scored(prepared, spec, forecast_sha)
    record = {"status": "completed_explore", "spec": spec,
              "plan_sha256": plan["plan_sha256"], "result": result,
              "goal_achieved": False, "holdout_read": False}
    write_json(prepared.out / "logs/experiments" / f"{spec['id']}.json",
               record, exclusive=True)
    _update_progress(prepared, plan)
    print("LORA_CANDIDATE_RESULT", json.dumps({"id": spec["id"],
          **result["fields"], "explore_targets_met": result["targets_met"]}), flush=True)
    return record


def _update_progress(prepared, plan: dict) -> None:
    rows = [_read(path) for path in sorted((prepared.out / "logs/experiments").glob("*.json"))]
    completed = sorted((row for row in rows if row.get("status") == "completed_explore"),
                       key=lambda row: (row["result"]["fields"]["AUC_MAE"], row["spec"]["id"]))
    record = {"contract": CONTRACT, "plan_sha256": plan["plan_sha256"],
              "completed_candidates": len(completed),
              "best": completed[0] if completed else None,
              "explore_targets_met": any(row["result"]["targets_met"] for row in completed),
              "independent_confirmation": "pending", "goal_achieved": False,
              "holdout_read": False}
    write_json(prepared.out / "logs/goal_progress.json", record)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "search"), required=True)
    parser.add_argument("--only", choices=tuple(CONFIGS))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    from phase_f.runtime import activate_optional_dependencies

    activate_optional_dependencies(root)
    original, prepared = goal_r1.prepare(root, namespace=NAMESPACE)
    status = prepared.out / "logs/driver_status.json"
    write_json(status, {"status": "running", "pid": os.getpid(), "stage": args.stage,
                        "only": args.only, "started_at": now(),
                        "goal_achieved": False, "holdout_read": False})
    try:
        baselines = goal_r1.copy_baselines(original, prepared)
        plan = _plan(prepared, baselines)
        for spec in _selected(plan, args.only):
            if args.stage == "smoke":
                _smoke(prepared, plan, spec)
            else:
                _search(prepared, plan, spec)
        _update_progress(prepared, plan)
        write_json(status, {"status": "completed_explore_stage", "stage": args.stage,
                            "only": args.only, "at": now(),
                            "independent_confirmation": "pending", "goal_achieved": False,
                            "holdout_read": False})
    except BaseException as exc:
        write_json(status, {"status": "failed", "stage": args.stage,
                            "only": args.only, "at": now(),
                            "error_type": type(exc).__name__, "error": str(exc),
                            "goal_achieved": False, "holdout_read": False})
        raise


if __name__ == "__main__":
    main()
