"""Isolated FG-R3 EXPLORE runner; no numerical CONFIRM or holdout access."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phase_f import goal_guard, goal_r1, goal_r1_paths, wf_metrics
from phase_f.goal_protocol import SEEDS
from phase_f.models import transition_expert
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path
from phase_f.wf_models import FRAME_KEY, _mean_frames, _validate_frame, _arm_view


NAMESPACE = "goal_r1_transition_v3"
PARENT = "goal_r1_v2"
GUARD = "goal_r1_guard_v1"
BEST_GUARD = "FG-R1-core-l1__guard-q95-suppress"
CORE_GROUPS = ("lag_1_16", "slot_7_28d", "profile", "rolling", "trend", "calendar", "peak")
SPECS = (
    {"id": "FG-R3-transition-power", "adapter": "transition_expert",
     "family": "R1-transition-expert", "arm": "EXPLORE", "groups": list(CORE_GROUPS)},
    {"id": "FG-R3-transition-production", "adapter": "transition_expert",
     "family": "R1-transition-expert", "arm": "EXPLORE", "groups": [*CORE_GROUPS, "production"]},
)
CONTRACT = {"protocol": NAMESPACE, "arm": "EXPLORE",
            "past_result_used_for_design": "signed post-observation EXPLORE transition diagnosis",
            "threshold_data_units": 30, "classes": ["fall", "neutral", "rise"],
            "training": "FIT only; pooled all13; max-h16 embargo",
            "stopping": "STOP classifier multi_logloss and class-conditional magnitude L1",
            "alpha_selection": "pooled STOP point MAE + 0.25 * pooled STOP PeakMAE",
            "alpha_grid": list(transition_expert.ALPHAS), "min_fit_stop_per_class": 20,
            "seeds": list(SEEDS), "ranking": "metrics of mean of five actual seed forecasts",
            "independent_confirmation": "pending", "goal_achieved": False,
            "holdout_read": False}
CONTRACT["contract_sha256"] = config_hash(CONTRACT)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources(root: Path) -> dict[str, str]:
    names = set(goal_guard._sources(root)) | set(transition_expert._SOURCES) | {
        "phase_f/goal_transition.py", "phase_f/goal_transition_diagnostics.py",
        "phase_f/wf_evaluation.py", "phase_f/wf_models.py", "phase_f/wf_metrics.py",
        "phase_f/goal_protocol.py", "src/session_data.py", "src/features.py",
        "src/targets.py", "src/holidays.py",
    }
    return {name: sha256(root / name) for name in sorted(names)}


def _runtime() -> dict:
    packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "joblib")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def _diagnosis(root: Path) -> dict:
    base = root / "outputs/phase_f" / NAMESPACE
    path = base / "diagnostics/manifest.json"
    manifest = _read(path)
    if (manifest.get("arm") != "EXPLORE" or manifest.get("analysis_only") is not True
            or manifest.get("future_outcomes_used_as_inputs") is not False
            or manifest.get("holdout_read") is not False
            or manifest.get("threshold_data_units") != 30):
        raise ValueError("Signed post-observation diagnosis changed")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != {"error_cohorts.csv", "past_future_ramp.csv",
                          "fit_stop_class_support.csv"}:
        raise ValueError("Transition diagnosis artifacts are incomplete")
    for filename, digest in artifacts.items():
        if sha256(path.parent / filename) != digest:
            raise ValueError(f"Transition diagnosis changed: {filename}")
    return {"plan_sha256": sha256(base / "TRANSITION_PLAN.md"),
            "manifest_sha256": sha256(path), "artifacts": artifacts}


def _guard_evidence(original, prepared, parent_evidence: dict) -> dict:
    root = Path(prepared.root)
    guard_dir = root / "outputs/phase_f" / GUARD
    guard_plan_path = guard_dir / "logs/execution_plan.json"
    plan = _read(guard_plan_path)
    digest = plan.pop("plan_sha256", None)
    expected_specs = goal_guard._guard_specs(goal_guard.initial_specs())
    if (digest != config_hash(plan) or plan.get("parent_evidence") != parent_evidence
            or plan.get("sources") != goal_guard._sources(root)
            or plan.get("specs") != expected_specs
            or plan.get("seeds") != list(SEEDS)
            or plan.get("holdout_read") is not False):
        raise ValueError("Thirty-guard EXPLORE plan or parent binding changed")
    driver_path = guard_dir / "logs/driver_status.json"
    driver = _read(driver_path)
    if (driver.get("status") != "completed_explore_stage"
            or driver.get("candidate_count") != 30
            or driver.get("holdout_read") is not False):
        raise ValueError("Thirty-guard EXPLORE wave is incomplete")
    record_path = guard_dir / "logs/experiments" / f"{BEST_GUARD}.json"
    record = _read(record_path)
    spec = next(row for row in expected_specs if row["id"] == BEST_GUARD)
    if (record.get("status") != "completed" or record.get("spec") != spec
            or record.get("result", {}).get("n_seeds") != len(SEEDS)):
        raise ValueError("Verified guard comparison is incomplete")
    score_manifest_path = guard_dir / "tables" / BEST_GUARD / "EXPLORE/manifest.json"
    score_manifest = _read(score_manifest_path)
    if score_manifest != record["result"] or score_manifest.get("arm") != "EXPLORE":
        raise ValueError("Guard score record and manifest disagree")
    for filename, sha in score_manifest.get("artifacts", {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_]+\.csv", filename):
            raise ValueError("Unsafe guard score filename")
        if sha256(score_manifest_path.parent / filename) != sha:
            raise ValueError(f"Guard score artifact changed: {filename}")
    guard_source = copy.copy(prepared.source)
    guard_source.out = guard_dir
    guard_view = _arm_view(guard_source, "EXPLORE")
    mean_path = prediction_path(guard_view, BEST_GUARD)
    mean_meta_path = mean_path.with_suffix(".json")
    meta = _read(mean_meta_path)
    identity = config_hash({"spec": spec, "plan": config_hash({**plan, "plan_sha256": digest})})
    if (meta.get("identity") != identity or meta.get("sha256") != sha256(mean_path)
            or meta.get("arm") != "EXPLORE"
            or meta.get("audit", {}).get("n_seeds") != len(SEEDS)
            or score_manifest.get("prediction_sha256") != sha256(mean_path)):
        raise ValueError("Guard mean identity or physical forecast changed")
    goal_guard._verify_guard_seeds(guard_view, spec, identity,
                                   parent_evidence, meta["audit"]["seeds"])
    mean = pd.read_parquet(mean_path)
    _validate_frame(prepared, mean, BEST_GUARD, "EXPLORE")
    ordered = mean.sort_values(list(FRAME_KEY)).reset_index(drop=True)
    seed_digests = {}
    forecasts = []
    for seed in SEEDS:
        path = guard_dir / "predictions/seeds/EXPLORE" / BEST_GUARD / f"seed_{seed}.parquet"
        seed_frame = pd.read_parquet(path).sort_values(list(FRAME_KEY)).reset_index(drop=True)
        _validate_frame(prepared, seed_frame, BEST_GUARD, "EXPLORE")
        if not seed_frame[list(FRAME_KEY)].equals(ordered[list(FRAME_KEY)]):
            raise ValueError("Guard seed keys changed")
        forecasts.append(seed_frame.pred.to_numpy(float))
        seed_digests[str(seed)] = {"prediction_sha256": sha256(path),
                                  "metadata_sha256": sha256(path.with_suffix(".json"))}
    if not np.allclose(ordered.pred, np.mean(forecasts, axis=0), rtol=0, atol=1e-12):
        raise ValueError("Guard mean is not its five-seed mean")
    return {"plan_sha256": sha256(guard_plan_path),
            "driver_sha256": sha256(driver_path),
            "record_sha256": sha256(record_path),
            "score_manifest_sha256": sha256(score_manifest_path),
            "mean_sha256": sha256(mean_path),
            "mean_metadata_sha256": sha256(mean_meta_path),
            "seed_files": seed_digests, "guard_id": BEST_GUARD,
            "guard_namespace": GUARD}


def _verify_path_cohort(paths: pd.DataFrame, required: pd.DataFrame,
                        manifest: dict, parent_plan: dict) -> None:
    # Canonical stored paths use int16 horizons; the required role union uses
    # int64. Compare exact ordered keys as the signed parent producer does.
    keys = pd.MultiIndex.from_frame(paths.loc[:, ["origin", "horizon"]])
    required_keys = pd.MultiIndex.from_frame(required)
    if (len(paths) != len(required) or len(paths) != manifest["path_rows"]
            or paths.duplicated(["origin", "horizon"]).any()
            or goal_r1_paths._hash_frame(paths) != manifest["paths_sha256"]
            or manifest["paths_sha256"] != parent_plan["rolling_paths_sha256"]
            or not keys.equals(required_keys)):
        raise ValueError("Reconstructed R1 rolling paths differ from signed required cohort")


def _rolling_paths(original, prepared, parent_evidence: dict) -> tuple[pd.DataFrame, dict]:
    root = Path(prepared.root)
    parent_dir = root / "outputs/phase_f" / PARENT
    parent_plan = _read(parent_dir / "logs/execution_plan.json")
    manifest_path = parent_dir / "r1_paths/manifest.json"
    manifest = _read(manifest_path)
    signature = manifest.pop("manifest_sha256", None)
    if (signature != config_hash(manifest) or manifest.get("complete") is not True
            or sha256(manifest_path) != parent_evidence["parent_rolling_manifest_sha256"]
            or manifest.get("identity_hash") != config_hash(manifest.get("identity"))):
        raise ValueError("R1 producer signed manifest changed")
    required = goal_r1_paths._required(prepared)
    baseline = load_predictions(original, BASELINES["R1"])
    anchors = goal_r1_paths._anchor_paths(prepared, baseline, required)
    snapshot, model = goal_r1_paths._snapshot(prepared)
    source = goal_r1_paths._verify_anchor_source(prepared, anchors, model["revision"])
    execution = goal_r1_paths._execution_fingerprint(manifest["identity"]["execution"]["device"])
    current_identity = goal_r1_paths._identity(prepared, required, anchors, model, source, execution)
    if current_identity != manifest["identity"]:
        raise ValueError("R1 producer model/source/runtime/anchor identity drifted")
    parts = []
    frozen_parts = []
    for entry in manifest["parts"]:
        filename = entry.get("file", "")
        if not re.fullmatch(r"part_\d{6}\.parquet", filename):
            raise ValueError("Unsafe R1 producer chunk")
        path = manifest_path.parent / filename
        if sha256(path) != entry.get("sha256"):
            raise ValueError("R1 producer chunk physical SHA changed")
        part = goal_r1_paths._sort_paths(pd.read_parquet(path))
        if (len(part) != entry.get("rows") or part.duplicated(["origin", "horizon"]).any()
                or goal_r1_paths._hash_frame(part) != entry.get("content_sha256")):
            raise ValueError("R1 producer chunk content changed")
        parts.append(part)
        frozen_parts.append({"file": filename, "sha256": entry["sha256"],
                             "content_sha256": entry["content_sha256"]})
    paths = goal_r1_paths._sort_paths(pd.concat([anchors, *parts], ignore_index=True))
    _verify_path_cohort(paths, required, manifest, parent_plan)
    _, _, digest, anchor_rows = goal_guard._producer_lock(parent_dir, parent_plan)
    if digest != parent_evidence["rolling_audit_sha256"] or anchor_rows != len(anchors):
        raise ValueError("R1 producer causal audit changed")
    inferred_origins = int(pd.concat([p["origin"] for p in parts], ignore_index=True).nunique())
    proof = {"identity_hash": manifest["identity_hash"], "manifest": str(manifest_path),
             "path_rows": manifest["path_rows"], "anchor_rows": len(anchors),
             "inferred_rows": sum(len(p) for p in parts),
             "inferred_origins": inferred_origins,
             "future_perturbation_max_abs_difference": manifest["future_perturbation_max_abs_difference"],
             "anchor_recipe_check": manifest["anchor_recipe_check"],
             "leakage_test": "passed", "input_cutoff_rule": "history.index <= origin",
             "anchor_prediction_sha256": manifest["identity"]["anchor_source"]["anchor_prediction_sha256"],
             "anchor_source": manifest["identity"]["anchor_source"],
             "source_sha256": manifest["identity"]["source_sha256"],
             "execution": manifest["identity"]["execution"],
             "cache_identity_hash": manifest["identity_hash"], "holdout_read": False,
             "historical_final_artifact_read": False,
             "model_revision": manifest["identity"]["model"]["revision"],
             "local_training": False}
    if config_hash(proof) != digest:
        raise ValueError("Reconstructed R1 causal proof differs from parent")
    paths.attrs["causal_provenance"] = proof
    # The signed producer hashes canonical CSV bytes; model checkpoints bind
    # the normalized table used by _paths_index. These are distinct domains.
    _, model_paths_sha = transition_expert._paths_index(paths, prepared.contexts)
    return paths, {"manifest_sha256": sha256(manifest_path),
                   "paths_sha256": manifest["paths_sha256"],
                   "model_paths_sha256": model_paths_sha,
                   "rolling_audit_sha256": digest, "parts": frozen_parts,
                   "anchor_source": source, "model": model, "execution": execution,
                   "snapshot_path": str(snapshot)}


def preflight(original, prepared) -> tuple[pd.DataFrame, dict]:
    if prepared.out.name != NAMESPACE:
        raise ValueError("Transition runner requires its isolated namespace")
    baselines = goal_r1.copy_baselines(original, prepared)
    parent = goal_guard.preflight(original, prepared, parent_namespace=PARENT)
    if baselines != parent["baseline_prediction_sha256"]:
        raise ValueError("Copied baselines differ from complete R1 parent")
    guard = _guard_evidence(original, prepared, parent)
    paths, rolling = _rolling_paths(original, prepared, parent)
    diagnosis = _diagnosis(Path(prepared.root))
    root = Path(prepared.root)
    from phase_c.data import SOURCE_RELATIVE
    from phase_f.harness import seal_parent
    physical_seal = seal_parent(root)
    if physical_seal != prepared.seal or sha256(root / SOURCE_RELATIVE) != prepared.seal["raw_sha256"]:
        raise ValueError("Protected parent evidence or raw source changed before transition plan")
    weekly_locks = {name: sha256(root / name) for name in (
        "outputs/phase_f/logs/walkforward_lock.json",
        "outputs/phase_f/walkforward_v2/walkforward_lock.json")}
    if any(_read(root / name) != prepared.split_lock for name in weekly_locks):
        raise ValueError("Physical weekly split lock changed")
    baseline_metadata = {key: sha256(prediction_path(prepared, key).with_suffix(".json"))
                         for key in baselines}
    plan = {"contract": CONTRACT, "specs": list(SPECS), "seeds": list(SEEDS),
            "sources": _sources(root), "runtime": _runtime(),
            "split_sha256": prepared.split_lock["lock_sha256"],
            "weekly_lock_file_sha256": weekly_locks,
            "seal_sha256": config_hash(prepared.seal),
            "raw_sha256": prepared.seal["raw_sha256"],
            "baseline_prediction_sha256": baselines,
            "baseline_metadata_sha256": baseline_metadata,
            "parent_evidence": parent, "guard_evidence": guard,
            "rolling_evidence": rolling, "diagnosis": diagnosis,
            "arm": "EXPLORE", "holdout_read": False, "goal_achieved": False}
    plan["plan_sha256"] = config_hash(plan)
    write_json(prepared.out / "logs/target_contract.json", CONTRACT, exclusive=True)
    write_json(prepared.out / "logs/execution_plan.json", plan, exclusive=True)
    return paths, plan


def _verify_path_digests(paths: pd.DataFrame, contexts: dict, rolling: dict) -> None:
    if (goal_r1_paths._hash_frame(goal_r1_paths._sort_paths(paths))
            != rolling["paths_sha256"]
            or transition_expert._paths_index(paths, contexts)[1]
            != rolling["model_paths_sha256"]):
        raise ValueError("Frozen R1 producer or normalized model path digest changed")


def _verify_frozen(prepared, plan: dict, paths: pd.DataFrame,
                   *, deep_model_snapshot: bool = False) -> None:
    root = Path(prepared.root)
    from phase_c.data import SOURCE_RELATIVE
    from phase_f.harness import seal_parent
    stored = _read(prepared.out / "logs/execution_plan.json")
    if (stored != plan or plan.get("plan_sha256") != config_hash(
            {k: v for k, v in plan.items() if k != "plan_sha256"})
            or _sources(root) != plan["sources"] or _runtime() != plan["runtime"]
            or _diagnosis(root) != plan["diagnosis"]
            or prepared.split_lock["lock_sha256"] != plan["split_sha256"]
            or config_hash(prepared.seal) != plan["seal_sha256"]
            or prepared.seal["raw_sha256"] != plan["raw_sha256"]):
        raise ValueError("Frozen transition execution plan/source/runtime/diagnosis changed")
    if (seal_parent(root) != prepared.seal
            or sha256(root / SOURCE_RELATIVE) != plan["raw_sha256"]):
        raise ValueError("Protected 803-file seal or raw source changed")
    for name, digest in plan["weekly_lock_file_sha256"].items():
        if sha256(root / name) != digest or _read(root / name) != prepared.split_lock:
            raise ValueError("Physical weekly split lock changed")
    parent = plan["parent_evidence"]
    goal_guard._verify_frozen_evidence(prepared, parent)
    for name, digest in plan["baseline_prediction_sha256"].items():
        path = prediction_path(prepared, name)
        metadata = _read(path.with_suffix(".json"))
        if (sha256(path) != digest
                or sha256(path.with_suffix(".json")) != plan["baseline_metadata_sha256"][name]
                or metadata.get("sha256") != digest or metadata.get("arm") != "EXPLORE"):
            raise ValueError("Copied baseline changed")
    guard = plan["guard_evidence"]
    guard_dir = root / "outputs/phase_f" / GUARD
    fixed = (("logs/execution_plan.json", "plan_sha256"),
             ("logs/driver_status.json", "driver_sha256"),
             (f"logs/experiments/{BEST_GUARD}.json", "record_sha256"),
             (f"tables/{BEST_GUARD}/EXPLORE/manifest.json", "score_manifest_sha256"),
             (f"predictions/EXPLORE/{BEST_GUARD}.parquet", "mean_sha256"),
             (f"predictions/EXPLORE/{BEST_GUARD}.json", "mean_metadata_sha256"))
    for name, key in fixed:
        if sha256(guard_dir / name) != guard[key]:
            raise ValueError(f"Frozen guard artifact changed: {name}")
    for seed, digests in guard["seed_files"].items():
        path = guard_dir / "predictions/seeds/EXPLORE" / BEST_GUARD / f"seed_{seed}.parquet"
        if (sha256(path) != digests["prediction_sha256"]
                or sha256(path.with_suffix(".json")) != digests["metadata_sha256"]):
            raise ValueError("Frozen guard seed changed")
    rolling = plan["rolling_evidence"]
    manifest_path = root / "outputs/phase_f" / PARENT / "r1_paths/manifest.json"
    if sha256(manifest_path) != rolling["manifest_sha256"]:
        raise ValueError("Frozen R1 producer manifest changed")
    for entry in rolling["parts"]:
        path = manifest_path.parent / entry["file"]
        if (sha256(path) != entry["sha256"]
                or goal_r1_paths._hash_frame(goal_r1_paths._sort_paths(pd.read_parquet(path)))
                != entry["content_sha256"]):
            raise ValueError("Frozen R1 producer part changed")
    _verify_path_digests(paths, prepared.contexts, rolling)
    if (config_hash(paths.attrs.get("causal_provenance", {}))
            != rolling["rolling_audit_sha256"]):
        raise ValueError("Frozen R1 paths/provenance changed")
    if deep_model_snapshot:
        _, current_model = goal_r1_paths._snapshot(prepared)
        current_execution = goal_r1_paths._execution_fingerprint(
            rolling["execution"]["device"])
        if current_model != rolling["model"] or current_execution != rolling["execution"]:
            raise ValueError("Pinned R1 snapshot weights or inference runtime changed")


def _cached(path: Path, identity: str):
    metadata = path.with_suffix(".json")
    if path.exists() != metadata.exists():
        transition_expert._quarantine(path)
        transition_expert._quarantine(metadata)
    value = goal_r1._cached(path, identity)
    if value is not None and _read(metadata).get("arm") != "EXPLORE":
        raise ValueError("Transition prediction metadata arm changed")
    return value


def _publish(path: Path, frame: pd.DataFrame, metadata: dict) -> None:
    transition_expert._quarantine(path.with_suffix(".tmp"))
    goal_r1._publish(path, frame, metadata)


def _zero_numeric(value) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(float(value)) and float(value) == 0.0)


def _verify_seed(prepared, child: dict, path: Path, identity: str,
                 rolling_digest: str, paths_sha: str) -> tuple[pd.DataFrame, dict]:
    seed = int(child["seed"])
    metadata = _read(path.with_suffix(".json"))
    if (metadata.get("arm") != "EXPLORE" or metadata.get("seed") != seed
            or metadata.get("child_spec") != child):
        raise ValueError("Transition seed metadata arm/spec/seed changed")
    cached = _cached(path, identity)
    if cached is None:
        raise ValueError("Transition seed is incomplete")
    frame, audit = cached
    _validate_frame(prepared, frame, child["id"], "EXPLORE")
    folds = sorted({fold for _, fold in prepared.contexts})
    if (audit.get("seed") != seed or audit.get("leakage_test") != "passed"
            or audit.get("rolling_audit_sha256") != rolling_digest
            or audit.get("paths_sha256") != paths_sha
            or not _zero_numeric(audit.get("future_perturbation_max_abs_difference"))
            or audit.get("smoke_first_fold") is not False
            or audit.get("n_fold_models") != len(folds)
            or [cell.get("fold") for cell in audit.get("cells", [])] != folds
            or audit.get("holdout_read") is not False):
        raise ValueError("Transition seed audit is incomplete or changed")
    for cell in audit["cells"]:
        if not _zero_numeric(cell.get("future_perturbation_max_abs_difference")):
            raise ValueError("Transition seed fold future-perturbation audit changed")
        fold = cell["fold"]
        fit, stop, _ = transition_expert._common_roles(prepared.contexts, fold)
        expected_identity = transition_expert._identity(prepared, child, paths_sha,
                                                        fold, fit, stop)
        checkpoint = Path(prepared.out) / "models/transition_expert" / child["id"] / \
            f"seed{seed}_f{fold}.joblib"
        physical_meta = _read(checkpoint.with_suffix(".json"))
        if (cell.get("identity") != expected_identity
                or physical_meta.get("identity") != expected_identity
                or physical_meta.get("seed") != seed
                or physical_meta.get("fold") != fold
                or physical_meta.get("model_sha256") != sha256(checkpoint)
                or cell.get("model_sha256") != physical_meta["model_sha256"]
                or cell.get("checkpoint_metadata_sha256")
                != sha256(checkpoint.with_suffix(".json"))
                or cell.get("fit_class_count") != physical_meta.get("fit_class_count")
                or cell.get("stop_class_count") != physical_meta.get("stop_class_count")
                or cell.get("alpha") != physical_meta.get("alpha")):
            raise ValueError(f"Transition seed {seed} fold {fold} checkpoint provenance changed")
    return frame, audit


def _mean_of_seeds(frames: list[pd.DataFrame], model_id: str) -> pd.DataFrame:
    mean = _mean_frames(frames, model_id)
    ordered = [frame.sort_values(list(FRAME_KEY)).reset_index(drop=True) for frame in frames]
    for column in ("transition_p_fall", "transition_p_neutral", "transition_p_rise",
                   "transition_expert_point"):
        values = np.stack([frame[column].to_numpy(float) for frame in ordered])
        finite = np.isfinite(values)
        if not (finite.all(axis=0) | (~finite).all(axis=0)).all():
            raise ValueError("Transition seed probability coverage differs")
        mean[column] = np.where(finite.all(axis=0),
                                np.sum(np.where(finite, values, 0), axis=0) / len(frames), np.nan)
    powers = [frame.transition_origin_power.to_numpy(float) for frame in ordered]
    if not all(np.array_equal(powers[0], value, equal_nan=True) for value in powers[1:]):
        raise ValueError("Origin power differs across transition seeds")
    r1s = [frame.r1.to_numpy(float) for frame in ordered]
    if not all(np.array_equal(r1s[0], value) for value in r1s[1:]):
        raise ValueError("Frozen R1 path differs across transition seeds")
    mean["transition_origin_power"] = powers[0]
    mean["r1"] = r1s[0]
    mean["applied_correction"] = mean.pred - mean.r1
    probabilities = mean[["transition_p_fall", "transition_p_neutral",
                          "transition_p_rise"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Five-seed mean transition probabilities are incomplete")
    return mean


def _transition_diagnostic(frame: pd.DataFrame, path: Path) -> tuple[str, dict]:
    score = frame.loc[frame.role.eq("score") & frame.arm.eq("EXPLORE")].copy()
    if not np.isfinite(score.transition_origin_power).all():
        raise ValueError("SCORE origin power is unavailable for transition evaluation")
    actual = transition_expert._classes(score.y.to_numpy(float)
                                         - score.transition_origin_power.to_numpy(float))
    prob = score[["transition_p_fall", "transition_p_neutral", "transition_p_rise"]].to_numpy(float)
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1, atol=1e-6):
        raise ValueError("SCORE transition class probabilities are incomplete")
    predicted = np.argmax(prob, axis=1)
    probability_quality = {
        "multiclass_logloss": float(-np.log(np.clip(prob[np.arange(len(actual)), actual], 1e-15, 1)).mean()),
        "multiclass_brier_sum": float(np.square(prob - np.eye(3)[actual]).sum(axis=1).mean()),
        "n_score_rows": len(score),
        "mean_of_actual_five_seed_probabilities": True,
        "calibrated_probabilities_claimed": False,
        "operational_peak_event_metric": False,
    }
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (actual, predicted), 1)
    labels = ("fall", "neutral", "rise")
    rows = []
    for i, name in enumerate(labels):
        truth = int(counts[i].sum())
        chosen = int(counts[:, i].sum())
        subset = score.iloc[np.flatnonzero(actual == i)]
        errors = np.abs(subset.pred.to_numpy(float) - subset.y.to_numpy(float))
        peak = subset.y.to_numpy(float) > subset.tau.to_numpy(float)
        rows.append({"class": name, "actual_rows": truth, "predicted_rows": chosen,
                     "true_positive_rows": int(counts[i, i]),
                     "precision": float(counts[i, i] / chosen) if chosen else np.nan,
                     "recall": float(counts[i, i] / truth) if truth else np.nan,
                     "point_MAE": float(errors.mean()) if truth else np.nan,
                     "peak_rows": int(peak.sum()),
                     "peak_MAE": float(errors[peak].mean()) if peak.any() else np.nan,
                     **{f"predicted_{label}": int(counts[i, j]) for j, label in enumerate(labels)},
                     "diagnostic_only": True, "phase_e_peak_event_metric": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return sha256(path), probability_quality


def _score(prepared, plan: dict, spec: dict, frame: pd.DataFrame, audit: dict) -> dict:
    result = goal_r1.score(prepared, spec, frame, audit)
    guard_path = Path(prepared.root) / "outputs/phase_f" / GUARD / "predictions/EXPLORE" / f"{BEST_GUARD}.parquet"
    if sha256(guard_path) != plan["guard_evidence"]["mean_sha256"]:
        raise ValueError("Guard comparison changed before paired scoring")
    dest = prepared.out / "tables" / spec["id"] / "EXPLORE"
    paired_path = dest / "paired_vs_best_guard.csv"
    wf_metrics.paired_ci(frame, pd.read_parquet(guard_path), n=1000, seed=42).to_csv(paired_path, index=False)
    diagnostic_path = dest / "transition_class_diagnostic.csv"
    diagnostic_hash, probability_quality = _transition_diagnostic(frame, diagnostic_path)
    result["artifacts"] = {**result["artifacts"],
                           "paired_vs_best_guard.csv": sha256(paired_path),
                           "transition_class_diagnostic.csv": diagnostic_hash}
    result.update({"best_guard": BEST_GUARD,
                   "best_guard_mean_sha256": plan["guard_evidence"]["mean_sha256"],
                   "transition_probability_quality": probability_quality,
                   "point_forecast_only": True, "operational_phase_e": "not_evaluated",
                   "post_observation_design": True, "goal_achieved": False,
                   "holdout_read": False})
    write_json(dest / "manifest.json", result)
    return result


def run_smoke(prepared, paths: pd.DataFrame, plan: dict) -> None:
    _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
    for spec in SPECS:
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        child = {**copy.deepcopy(spec), "id": spec["id"] + "__smoke_seed42",
                 "seed": 42, "rolling_audit_sha256": plan["rolling_evidence"]["rolling_audit_sha256"]}
        frame, audit = transition_expert.run(prepared, child, paths, smoke_first_fold=True)
        first_fold = min(fold for _, fold in prepared.contexts)
        expected = {(h, first_fold, role, origin)
                    for h in range(4, 17) for role in ("cal", "score")
                    for origin in prepared.contexts[(h, first_fold)][role]}
        actual = set(frame[["horizon", "fold", "role", "origin"]].itertuples(index=False, name=None))
        if actual != expected or audit["leakage_test"] != "passed" or audit["n_fold_models"] != 1:
            raise ValueError("Transition first-fold all-horizon smoke is incomplete")
        path = prepared.out / "smoke" / spec["id"] / "seed42_first_fold.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = path.with_suffix(".json")
        if path.exists() != metadata_path.exists():
            transition_expert._quarantine(path)
            transition_expert._quarantine(metadata_path)
        if path.exists():
            existing = _read(metadata_path)
            if (existing.get("sha256") != sha256(path)
                    or existing.get("plan_sha256") != plan["plan_sha256"]
                    or existing.get("child_spec") != child
                    or not pd.read_parquet(path).equals(frame)):
                raise ValueError("Existing technical smoke changed; retain original evidence")
        else:
            temporary = path.with_suffix(".tmp")
            transition_expert._quarantine(temporary)
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
            write_json(metadata_path, {"at": now(), "sha256": sha256(path),
                                       "audit": audit, "plan_sha256": plan["plan_sha256"],
                                       "child_spec": child,
                                       "smoke_only": True, "five_seed_score": False})
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)


def _verify_smoke(prepared, plan: dict) -> None:
    first_fold = min(fold for _, fold in prepared.contexts)
    expected = {(h, first_fold, role, origin)
                for h in range(4, 17) for role in ("cal", "score")
                for origin in prepared.contexts[(h, first_fold)][role]}
    for spec in SPECS:
        child = {**copy.deepcopy(spec), "id": spec["id"] + "__smoke_seed42",
                 "seed": 42, "rolling_audit_sha256": plan["rolling_evidence"]["rolling_audit_sha256"]}
        path = prepared.out / "smoke" / spec["id"] / "seed42_first_fold.parquet"
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError(f"First-fold technical smoke is missing for {spec['id']}")
        meta = _read(metadata_path)
        audit = meta.get("audit", {})
        if (meta.get("sha256") != sha256(path)
                or meta.get("plan_sha256") != plan["plan_sha256"]
                or meta.get("child_spec") != child
                or meta.get("smoke_only") is not True
                or meta.get("five_seed_score") is not False
                or audit.get("seed") != 42
                or audit.get("rolling_audit_sha256") != child["rolling_audit_sha256"]
                or audit.get("paths_sha256") != plan["rolling_evidence"]["model_paths_sha256"]
                or audit.get("smoke_first_fold") is not True
                or audit.get("n_fold_models") != 1
                or audit.get("leakage_test") != "passed"
                or not _zero_numeric(audit.get("future_perturbation_max_abs_difference"))
                or audit.get("holdout_read") is not False):
            raise ValueError(f"First-fold technical smoke provenance changed: {spec['id']}")
        cell = audit.get("cells", [])
        if len(cell) != 1 or cell[0].get("fold") != first_fold:
            raise ValueError("Technical smoke lacks its first weekly model")
        fit, stop, _ = transition_expert._common_roles(prepared.contexts, first_fold)
        identity = transition_expert._identity(prepared, child,
            plan["rolling_evidence"]["model_paths_sha256"], first_fold, fit, stop)
        checkpoint = Path(prepared.out) / "models/transition_expert" / child["id"] / \
            f"seed42_f{first_fold}.joblib"
        checkpoint_meta = _read(checkpoint.with_suffix(".json"))
        if (cell[0].get("identity") != identity
                or checkpoint_meta.get("identity") != identity
                or checkpoint_meta.get("model_sha256") != sha256(checkpoint)
                or cell[0].get("model_sha256") != checkpoint_meta["model_sha256"]
                or cell[0].get("checkpoint_metadata_sha256")
                != sha256(checkpoint.with_suffix(".json"))
                or not _zero_numeric(cell[0].get("future_perturbation_max_abs_difference"))):
            raise ValueError("Technical smoke weekly model checkpoint changed")
        frame = pd.read_parquet(path)
        actual = set(frame[["horizon", "fold", "role", "origin"]].itertuples(index=False, name=None))
        if (len(frame) != len(expected) or actual != expected
                or not frame.model.eq(child["id"]).all()
                or not frame.loc[frame.role.eq("score"), "arm"].eq("EXPLORE").all()
                or not np.isfinite(frame.pred.to_numpy(float)).all()):
            raise ValueError("Technical smoke forecast cohort changed")


def run_search(prepared, paths: pd.DataFrame, plan: dict) -> None:
    _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
    _verify_smoke(prepared, plan)
    for spec in SPECS:
        if (prepared.out / "logs/stop_requested.json").exists():
            raise RuntimeError("Requested transition checkpoint stop")
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        record_path = prepared.out / "logs/experiments" / f"{spec['id']}.json"
        write_json(record_path, {"status": "running", "spec": spec,
                                 "at": now(), "holdout_read": False, "goal_achieved": False})
        identity = config_hash({"spec": spec, "plan_sha256": plan["plan_sha256"]})
        frames, entries = [], []
        for seed in SEEDS:
            _verify_frozen(prepared, plan, paths)
            child = {**copy.deepcopy(spec), "id": spec["id"] + f"__seed{seed}",
                     "seed": seed,
                     "rolling_audit_sha256": plan["rolling_evidence"]["rolling_audit_sha256"]}
            path = prepared.out / "predictions/seeds/EXPLORE" / spec["id"] / f"seed_{seed}.parquet"
            seed_identity = config_hash({"mean_identity": identity, "child": child,
                                         "arm": "EXPLORE", "seed": seed})
            cached = _cached(path, seed_identity)
            if cached is None:
                frame, seed_audit = transition_expert.run(prepared, child, paths)
                _validate_frame(prepared, frame, child["id"], "EXPLORE")
                _verify_frozen(prepared, plan, paths)
                _publish(path, frame, {"identity": seed_identity, "audit": seed_audit,
                                       "child_spec": child, "seed": seed})
            frame, seed_audit = _verify_seed(prepared, child, path, seed_identity,
                                             plan["rolling_evidence"]["rolling_audit_sha256"],
                                             plan["rolling_evidence"]["model_paths_sha256"])
            _verify_frozen(prepared, plan, paths)
            frames.append(frame)
            entries.append({"seed": seed, "prediction_sha256": sha256(path),
                            "metadata_sha256": sha256(path.with_suffix(".json")),
                            "identity": seed_identity, "audit": seed_audit})
            print("TRANSITION_SEED_READY", spec["id"], seed, flush=True)
        rebuilt = _mean_of_seeds(frames, spec["id"])
        _validate_frame(prepared, rebuilt, spec["id"], "EXPLORE")
        mean_path = prediction_path(prepared, spec["id"])
        audit = {"n_seeds": len(SEEDS), "seeds": entries,
                 "rolling_audit_sha256": plan["rolling_evidence"]["rolling_audit_sha256"],
                 "leakage_test": "passed", "ranking_basis": "metric of five-seed mean",
                 "independent_confirmation": "pending", "holdout_read": False}
        cached = _cached(mean_path, identity)
        if cached is None:
            _verify_frozen(prepared, plan, paths)
            _publish(mean_path, rebuilt, {"identity": identity, "audit": audit,
                                          "spec": spec, "seeds": list(SEEDS)})
        else:
            mean, old_audit = cached
            mean_metadata = _read(mean_path.with_suffix(".json"))
            if (mean_metadata.get("arm") != "EXPLORE"
                    or mean_metadata.get("spec") != spec
                    or mean_metadata.get("seeds") != list(SEEDS)):
                raise ValueError("Transition mean metadata arm/spec/seeds changed")
            _validate_frame(prepared, mean, spec["id"], "EXPLORE")
            if old_audit != audit or not mean.equals(rebuilt):
                raise ValueError("Cached transition mean differs from actual five seed artifacts")
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        result = _score(prepared, plan, spec, rebuilt, audit)
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        write_json(record_path, {"status": "completed", "spec": spec, "result": result,
                                 "holdout_read": False, "goal_achieved": False})
        print("TRANSITION_CANDIDATE_RESULT", json.dumps({"id": spec["id"],
              "fields": result["fields"], "targets_met": result["targets_met"]}), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "smoke", "search"), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    original, prepared = goal_r1.prepare(root, NAMESPACE)
    status_path = prepared.out / "logs/driver_status.json"
    write_json(status_path, {"status": "running", "pid": os.getpid(),
                             "stage": args.stage, "at": now(),
                             "goal_achieved": False, "holdout_read": False})
    try:
        paths, plan = preflight(original, prepared)
        if args.stage == "smoke":
            run_smoke(prepared, paths, plan)
        elif args.stage == "search":
            run_search(prepared, paths, plan)
        write_json(status_path, {"status": "completed_explore_stage" if args.stage == "search"
                                 else "completed_technical_stage", "stage": args.stage,
                                 "at": now(), "candidate_count": len(SPECS) if args.stage == "search" else 0,
                                 "full_explore_search_complete": args.stage == "search",
                                 "independent_confirmation": "pending", "goal_achieved": False,
                                 "holdout_read": False})
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "stage": args.stage,
                                 "at": now(), "error_type": type(exc).__name__,
                                 "error": str(exc), "goal_achieved": False,
                                 "holdout_read": False})
        raise


if __name__ == "__main__":
    main()
