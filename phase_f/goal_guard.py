"""Frozen, EXPLORE-only peak-tail guards for completed R1 residual forecasts.

Parent EXPLORE results informed this post-observation design.  All ten parents
must be complete, and the full 3 x 10 grid is locked before any guard-candidate
score is computed.  Every guard transforms each of the five *parent seed*
forecasts deterministically; no model is fit and labels never enter the
transform.  The mean of the five guarded forecasts is scored.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from phase_f import goal_r1
from phase_f.goal_protocol import CONTRACT, SEEDS, TARGETS, initial_specs
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path
from phase_f.wf_models import FRAME_KEY, _mean_frames, _validate_frame


JOIN_KEY = ("horizon", "fold", "role", "origin", "target_time", "arm")
POLICIES = (
    {"name": "q90-suppress", "quantile": "q90", "negative_factor": 0.0},
    {"name": "q95-suppress", "quantile": "q95", "negative_factor": 0.0},
    {"name": "q90-half", "quantile": "q90", "negative_factor": 0.5},
)
_NAMESPACE = re.compile(r"goal_r1_[a-z0-9_]+\Z")
GUARD_CONTRACT = {**{key: copy.deepcopy(value) for key, value in CONTRACT.items()
                     if key != "contract_sha256"},
                  "protocol": "goal_r1_guard_v1",
                  "training": "no new fitting; fixed point-forecast transforms of each of five frozen parent seeds",
                  "ranking": "metrics of the mean of five individually guarded seed forecasts",
                  "parent_requirements": "all ten initial R1 residual candidates complete before any guard score",
                  "guard_grid": "three fixed tail policies times ten fixed parent candidates",
                  "past_result_used_for_design": "R1 EXPLORE outcomes, FG-R1-core-l1 five-seed fold-15 negative-correction failure, and a posthoc R1 q90 tail-guard counterfactual",
                  "design_status": "post-observation exploratory hypothesis, frozen before guard-candidate scoring",
                  "selection_arm": "EXPLORE",
                  "confirmation": "independent locked WF CONFIRM and auxiliary checks remain pending",
                  "holdout_read": False}
GUARD_CONTRACT["contract_sha256"] = config_hash(GUARD_CONTRACT)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources(root) -> dict[str, str]:
    sources = goal_r1._sources(root)
    for name in ("phase_f/goal_guard.py", "phase_f/wf_evaluation.py",
                 "phase_f/run.py", "phase_f/registry.py"):
        sources[name] = sha256(Path(root) / name)
    return sources


def _ordered(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not set(JOIN_KEY) <= set(frame) or frame.duplicated(list(JOIN_KEY)).any():
        raise ValueError(f"{name} has missing or repeated cohort keys")
    return frame.sort_values(list(JOIN_KEY), kind="stable").reset_index(drop=True)


def _same_keys(a: pd.DataFrame, b: pd.DataFrame, *, description: str) -> None:
    if len(a) != len(b) or not a[list(JOIN_KEY)].equals(b[list(JOIN_KEY)]):
        raise ValueError(f"{description} does not match the original R1 CAL/SCORE cohort")


def _original_r1(prepared) -> pd.DataFrame:
    frame = _ordered(load_predictions(prepared, BASELINES["R1"]), name="original R1")
    if not set(("pred", "q50", "q90", "q95", "y", "tau", "d2", "fit_mean", "mase_scale")) <= set(frame):
        raise ValueError("Original R1 lacks the fixed median/quantile evidence")
    for column in ("pred", "q50", "q90", "q95", "tau"):
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"Original R1 has nonfinite {column}")
    if not np.allclose(frame.pred, frame.q50, rtol=0, atol=1e-5):
        raise ValueError("Original R1 point is not its frozen median path")
    if not frame.loc[frame.role.eq("score"), "arm"].eq("EXPLORE").all():
        raise ValueError("Original R1 includes non-EXPLORE scores")
    return frame


def _parent_frame(prepared, path: Path, meta: dict, expected_id: str,
                  original: pd.DataFrame) -> pd.DataFrame:
    if meta.get("sha256") != sha256(path) or meta.get("arm") != "EXPLORE":
        raise ValueError(f"Parent forecast or metadata bytes changed: {expected_id}")
    frame = _ordered(pd.read_parquet(path), name=expected_id)
    _validate_frame(prepared, frame, expected_id, "EXPLORE")
    _same_keys(frame, original, description=expected_id)
    for name in ("y", "tau", "d2", "fit_mean", "mase_scale"):
        if not np.array_equal(frame[name].to_numpy(), original[name].to_numpy()):
            raise ValueError(f"Parent {expected_id} differs from original R1 in {name}")
    if "r1" not in frame or not np.allclose(frame.r1, original.pred, rtol=0, atol=1e-9):
        raise ValueError(f"Parent {expected_id} R1 anchor changed")
    return frame


def _producer_lock(parent_dir: Path, parent_plan: dict) -> tuple[str, dict, str, int]:
    path = parent_dir / "r1_paths/manifest.json"
    manifest = _read(path)
    checksum = manifest.pop("manifest_sha256", None)
    if (checksum != config_hash(manifest) or manifest.get("complete") is not True
            or manifest.get("paths_sha256") != parent_plan.get("rolling_paths_sha256")
            or manifest.get("holdout_read") is not False
            or manifest.get("identity_hash") != config_hash(manifest.get("identity"))):
        raise ValueError("Parent rolling R1 producer manifest is not complete and locked")
    inferred_rows = 0
    inferred_origins = []
    for part in manifest.get("parts", ()):
        filename = part.get("file", "")
        if not re.fullmatch(r"part_\d{6}\.parquet", filename):
            raise ValueError("Unsafe producer part filename")
        physical = path.parent / filename
        if not physical.is_file() or sha256(physical) != part.get("sha256"):
            raise ValueError("Parent rolling R1 producer chunk changed")
        origins = pd.read_parquet(physical, columns=["origin"])["origin"]
        if len(origins) != part.get("rows"):
            raise ValueError("Parent rolling R1 producer chunk row count changed")
        inferred_rows += len(origins)
        inferred_origins.append(origins)
    anchor_rows = int(manifest.get("path_rows", -1)) - inferred_rows
    if anchor_rows < 1:
        raise ValueError("Parent rolling R1 producer anchor coverage is incomplete")
    identity = manifest["identity"]
    anchor_source = identity["anchor_source"]
    probe = manifest.get("anchor_recipe_check")
    leakage = ("passed" if manifest.get("future_perturbation_max_abs_difference") == 0.0
               and anchor_source.get("anchor_original_future_perturbation_max_abs_difference") == 0.0
               and isinstance(probe, dict) and probe.get("status") == "passed"
               else "unverified")
    if leakage != "passed":
        raise ValueError("Parent rolling R1 causal producer audit did not pass")
    audit = {"identity_hash": manifest["identity_hash"], "manifest": str(path),
             "path_rows": manifest["path_rows"], "anchor_rows": anchor_rows,
             "inferred_rows": inferred_rows,
             "inferred_origins": int(pd.concat(inferred_origins, ignore_index=True).nunique())
                                 if inferred_origins else 0,
             "future_perturbation_max_abs_difference": manifest["future_perturbation_max_abs_difference"],
             "anchor_recipe_check": probe, "leakage_test": leakage,
             "input_cutoff_rule": "history.index <= origin",
             "anchor_prediction_sha256": anchor_source["anchor_prediction_sha256"],
             "anchor_source": anchor_source,
             "source_sha256": identity["source_sha256"],
             "execution": identity["execution"],
             "cache_identity_hash": manifest["identity_hash"],
             "holdout_read": False, "historical_final_artifact_read": False,
             "model_revision": identity["model"]["revision"], "local_training": False}
    return sha256(path), manifest, config_hash(audit), anchor_rows


def _guard_specs(parent_specs: list[dict]) -> list[dict]:
    return [
        {"id": f"{parent['id']}__guard-{policy['name']}",
         "adapter": "frozen_r1_tail_guard", "parent": parent["id"],
         "family": "R1-tail-guard", "policy": copy.deepcopy(policy),
         "arm": "EXPLORE", "n_seeds": len(SEEDS)}
        for parent in parent_specs for policy in POLICIES
    ]


def preflight(original, prepared, *, parent_namespace="goal_r1_v2") -> dict:
    """Inspect all 10 completed parents before creating the guard search plan."""
    root = Path(prepared.root)
    if not _NAMESPACE.fullmatch(parent_namespace) or parent_namespace == prepared.out.name:
        raise ValueError("Unsafe or overlapping parent goal namespace")
    parent_dir = root / "outputs/phase_f" / parent_namespace
    parent_plan_path = parent_dir / "logs/execution_plan.json"
    parent_plan = _read(parent_plan_path)
    expected_specs = initial_specs()
    if (len(expected_specs) != 10 or parent_plan.get("specs") != expected_specs
            or parent_plan.get("seeds") != list(SEEDS)
            or parent_plan.get("contract") != CONTRACT
            or parent_plan.get("sources") != goal_r1._sources(root)
            or parent_plan.get("split_sha256") != prepared.split_lock.get("lock_sha256")):
        raise ValueError("Parent ten-candidate execution plan or sources changed")
    driver = _read(parent_dir / "logs/driver_status.json")
    if (driver.get("status") != "completed_explore_stage"
            or driver.get("full_explore_search_complete") is not True
            or driver.get("holdout_read") is not False):
        raise ValueError("Parent EXPLORE search is not fully completed")
    baseline_hashes = goal_r1.copy_baselines(original, prepared)
    if baseline_hashes != parent_plan.get("baseline_prediction_sha256"):
        raise ValueError("Parent and guard original baselines differ")
    producer_sha, producer, producer_audit_sha, anchor_rows = _producer_lock(parent_dir, parent_plan)
    original_r1 = _original_r1(prepared)
    if anchor_rows != len(original_r1):
        raise ValueError("R1 producer prefills differ from the original baseline cohort")
    frozen = {}
    rolling_digest = None
    for spec in expected_specs:
        parent_id = spec["id"]
        record_path = parent_dir / "logs/experiments" / f"{parent_id}.json"
        record = _read(record_path)
        mean_path = parent_dir / "predictions/EXPLORE" / f"{parent_id}.parquet"
        mean_meta_path = mean_path.with_suffix(".json")
        mean_meta = _read(mean_meta_path)
        parent_audit = mean_meta.get("audit", {})
        candidate_digest = parent_audit.get("rolling_audit_sha256")
        if (record.get("status") != "completed" or record.get("spec") != spec
                or record.get("result", {}).get("n_seeds") != len(SEEDS)
                or record["result"].get("leakage_test") != "passed"
                or record["result"].get("goal_achieved") is not False
                or record["result"].get("holdout_read") is not False
                or parent_audit.get("n_seeds") != len(SEEDS)
                or parent_audit.get("leakage_test") != "passed"
                or not isinstance(candidate_digest, str)
                or len(candidate_digest) != 64):
            raise ValueError(f"Parent {parent_id} is incomplete or unaudited")
        score_manifest_path = parent_dir / "tables" / parent_id / "EXPLORE/manifest.json"
        score_manifest = _read(score_manifest_path)
        if score_manifest != record["result"] or score_manifest.get("arm") != "EXPLORE":
            raise ValueError(f"Parent {parent_id} score manifest differs from its completed record")
        for filename, digest in score_manifest.get("artifacts", {}).items():
            if not re.fullmatch(r"[A-Za-z0-9_]+\.csv", filename):
                raise ValueError(f"Parent {parent_id} has an unsafe score artifact name")
            artifact = score_manifest_path.parent / filename
            if not artifact.is_file() or sha256(artifact) != digest:
                raise ValueError(f"Parent {parent_id} scored artifact changed")
        if rolling_digest is None:
            rolling_digest = candidate_digest
        elif rolling_digest != candidate_digest:
            raise ValueError("Parent candidates used different rolling R1 provenance")
        expected_identity = config_hash({"spec": spec, "rolling": candidate_digest,
                                         "plan": config_hash(parent_plan)})
        if mean_meta.get("identity") != expected_identity:
            raise ValueError(f"Parent {parent_id} mean identity changed")
        mean = _parent_frame(prepared, mean_path, mean_meta, parent_id, original_r1)
        if record["result"].get("prediction_sha256") != sha256(mean_path):
            raise ValueError(f"Parent {parent_id} score record differs from mean bytes")
        entries = parent_audit.get("seeds", ())
        if [entry.get("seed") for entry in entries] != list(SEEDS):
            raise ValueError(f"Parent {parent_id} lacks five actual ordered seeds")
        seed_evidence, values = [], []
        for seed, entry in zip(SEEDS, entries):
            child = copy.deepcopy(spec)
            child.update(seed=seed, rolling_audit_sha256=candidate_digest)
            child["id"] = parent_id + "__seed" + str(seed)
            seed_identity = config_hash({"mean_identity": expected_identity, "child": child})
            seed_path = parent_dir / "predictions/seeds/EXPLORE" / parent_id / f"seed_{seed}.parquet"
            metadata_path = seed_path.with_suffix(".json")
            metadata = _read(metadata_path)
            if (metadata.get("identity") != seed_identity
                    or metadata.get("audit", {}).get("leakage_test") != "passed"
                    or metadata.get("audit", {}).get("seed") != seed
                    or metadata.get("audit", {}).get("rolling_audit_sha256") != candidate_digest
                    or entry.get("prediction_sha256") != sha256(seed_path)):
                raise ValueError(f"Parent {parent_id} seed {seed} is incomplete or changed")
            seed_frame = _parent_frame(prepared, seed_path, metadata, child["id"], original_r1)
            values.append(seed_frame.pred.to_numpy(float))
            seed_evidence.append({"seed": seed, "forecast_sha256": sha256(seed_path),
                                  "metadata_sha256": sha256(metadata_path),
                                  "path": seed_path.relative_to(root).as_posix()})
        if not np.allclose(mean.pred.to_numpy(float), np.mean(values, axis=0), rtol=0, atol=1e-12):
            raise ValueError(f"Parent {parent_id} mean is not the five-seed forecast mean")
        frozen[parent_id] = {
            "spec": spec, "record_sha256": sha256(record_path),
            "score_manifest_sha256": sha256(score_manifest_path),
            "mean_sha256": sha256(mean_path), "mean_metadata_sha256": sha256(mean_meta_path),
            "seeds": seed_evidence,
        }
    if set(frozen) != {spec["id"] for spec in expected_specs}:
        raise AssertionError("Parent preflight did not cover all initial ten candidates")
    if rolling_digest != producer_audit_sha:
        raise ValueError("Parent rolling-audit digest does not match the physical producer manifest")
    return {"preflight_complete": True, "parent_namespace": parent_namespace,
            "parent_plan_sha256": sha256(parent_plan_path),
            "parent_driver_sha256": sha256(parent_dir / "logs/driver_status.json"),
            "parent_rolling_manifest_sha256": producer_sha,
            "parent_rolling_paths_sha256": producer["paths_sha256"],
            "rolling_audit_sha256": rolling_digest,
            "reconstructed_producer_audit_sha256": producer_audit_sha,
            "baseline_prediction_sha256": baseline_hashes,
            "original_r1_sha256": sha256(prediction_path(prepared, BASELINES["R1"])),
            "parents": frozen, "holdout_read": False}


def _verify_parent_file(root: Path, evidence: dict, parent_id: str, seed: int) -> Path:
    record = next((item for item in evidence["parents"][parent_id]["seeds"]
                   if item["seed"] == seed), None)
    if record is None:
        raise ValueError(f"Missing frozen parent seed {parent_id}/{seed}")
    path = root / record["path"]
    if (sha256(path) != record["forecast_sha256"]
            or sha256(path.with_suffix(".json")) != record["metadata_sha256"]):
        raise ValueError(f"Frozen parent seed bytes changed: {parent_id}/{seed}")
    return path


def _verify_frozen_evidence(prepared, evidence: dict) -> None:
    root = Path(prepared.root)
    parent_specs = initial_specs()
    if (evidence.get("preflight_complete") is not True
            or set(evidence.get("parents", {})) != {row["id"] for row in parent_specs}
            or evidence.get("holdout_read") is not False):
        raise ValueError("All ten parent candidates must pass frozen preflight")
    parent_dir = root / "outputs/phase_f" / evidence["parent_namespace"]
    fixed = ((parent_dir / "logs/execution_plan.json", evidence["parent_plan_sha256"]),
             (parent_dir / "logs/driver_status.json", evidence["parent_driver_sha256"]),
             (parent_dir / "r1_paths/manifest.json", evidence["parent_rolling_manifest_sha256"]))
    for path, digest in fixed:
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Frozen parent plan, status, or producer changed: {path.name}")
    parent_plan = _read(parent_dir / "logs/execution_plan.json")
    if parent_plan.get("sources") != goal_r1._sources(root):
        raise ValueError("Frozen parent source bytes changed")
    producer_sha, producer, producer_audit_sha, anchor_rows = _producer_lock(parent_dir, parent_plan)
    if (producer_sha != evidence["parent_rolling_manifest_sha256"]
            or producer.get("paths_sha256") != evidence["parent_rolling_paths_sha256"]
            or producer_audit_sha != evidence["rolling_audit_sha256"]
            or producer_audit_sha != evidence["reconstructed_producer_audit_sha256"]):
        raise ValueError("Frozen parent rolling producer changed")
    for name, key in BASELINES.items():
        path = prediction_path(prepared, key)
        if not path.is_file() or sha256(path) != evidence["baseline_prediction_sha256"][key]:
            raise ValueError(f"Frozen {name} baseline bytes changed")
    if sha256(prediction_path(prepared, BASELINES["R1"])) != evidence["original_r1_sha256"]:
        raise ValueError("Frozen original R1 baseline bytes changed")
    if anchor_rows != len(_original_r1(prepared)):
        raise ValueError("Frozen R1 producer anchor row count changed")
    for spec in parent_specs:
        parent_id = spec["id"]
        record = evidence["parents"][parent_id]
        base = parent_dir / "predictions/EXPLORE" / f"{parent_id}.parquet"
        files = ((parent_dir / "logs/experiments" / f"{parent_id}.json", record["record_sha256"]),
                 (parent_dir / "tables" / parent_id / "EXPLORE/manifest.json",
                  record["score_manifest_sha256"]),
                 (base, record["mean_sha256"]),
                 (base.with_suffix(".json"), record["mean_metadata_sha256"]))
        for path, digest in files:
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Frozen parent candidate bytes changed: {parent_id}")
        if [row["seed"] for row in record["seeds"]] != list(SEEDS):
            raise ValueError(f"Frozen parent seeds changed: {parent_id}")
        for seed in SEEDS:
            _verify_parent_file(root, evidence, parent_id, seed)


def apply_guard(parent: pd.DataFrame, original: pd.DataFrame, policy: dict, model_id: str) -> pd.DataFrame:
    """Predict from frozen forecast columns only; labels are never read."""
    if policy not in POLICIES:
        raise ValueError("Guard policy is outside the frozen three-policy grid")
    source = _ordered(parent, name="parent seed")
    baseline = _ordered(original, name="original R1")
    _same_keys(source, baseline, description="Guard input")
    if "r1" not in source or not np.allclose(source.r1, baseline.pred, rtol=0, atol=1e-9):
        raise ValueError("Parent seed median anchor differs from original R1")
    if not np.array_equal(source.tau.to_numpy(float), baseline.tau.to_numpy(float)):
        raise ValueError("Guard FIT-only threshold differs from original R1")
    if not np.isfinite(source.pred.to_numpy(float)).all():
        raise ValueError("Guard parent predictions must be finite")
    quantile = baseline[policy["quantile"]].to_numpy(float)
    tau = baseline.tau.to_numpy(float)
    anchor = baseline.pred.to_numpy(float)
    correction = source.pred.to_numpy(float) - anchor
    trigger = (quantile >= tau) & (correction < 0)
    factor = np.where(trigger, float(policy["negative_factor"]), 1.0)
    out = source.copy()
    out["pred"] = anchor + factor * correction
    out["model"] = model_id
    for name in ("correction", "applied_correction"):
        if name in out:
            out.drop(columns=name, inplace=True)
    for name in ("q10", "q50", "q90", "q95"):
        if name in out:
            out.drop(columns=name, inplace=True)
    out["r1"] = anchor
    out["r1_q90"] = baseline.q90.to_numpy(float)
    out["r1_q95"] = baseline.q95.to_numpy(float)
    out["applied_correction"] = out.pred.to_numpy(float) - anchor
    out["guard_triggered"] = trigger
    return out


def _verify_one_guard_seed(prepared, spec: dict, identity: str,
                           parent_path: Path, seed: int) -> dict:
    seed_path = prepared.out / "predictions/seeds/EXPLORE" / spec["id"] / f"seed_{seed}.parquet"
    metadata_path = seed_path.with_suffix(".json")
    if not seed_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Cached guard seed {seed} file/metadata is missing")
    expected_identity = config_hash({"mean_identity": identity, "seed": seed,
                                     "parent_forecast_sha256": sha256(parent_path)})
    metadata = _read(metadata_path)
    seed_audit = metadata.get("audit", {})
    if (metadata.get("identity") != expected_identity
            or metadata.get("sha256") != sha256(seed_path)
            or metadata.get("arm") != "EXPLORE"
            or seed_audit.get("seed") != seed
            or seed_audit.get("parent") != spec["parent"]
            or seed_audit.get("parent_forecast_sha256") != sha256(parent_path)
            or seed_audit.get("policy") != spec["policy"]
            or seed_audit.get("leakage_test") != "passed"
            or seed_audit.get("holdout_read") is not False):
        raise ValueError(f"Cached guard seed {seed} metadata changed")
    return {"seed": seed, "prediction_sha256": metadata["sha256"],
            "metadata_sha256": sha256(metadata_path),
            "identity": expected_identity, "audit": seed_audit}


def _verify_guard_seeds(prepared, spec: dict, identity: str, evidence: dict,
                        entries: list[dict]) -> None:
    if [entry.get("seed") for entry in entries] != list(SEEDS):
        raise ValueError("Cached guard mean lacks five ordered seed artifacts")
    for seed, entry in zip(SEEDS, entries):
        parent_path = _verify_parent_file(Path(prepared.root), evidence, spec["parent"], seed)
        if entry != _verify_one_guard_seed(prepared, spec, identity, parent_path, seed):
            raise ValueError(f"Cached guard mean seed {seed} lineage changed")


def update_progress(prepared) -> None:
    """Guard-specific progress with the same absolute performance targets."""
    directory = prepared.out / "logs/experiments"
    rows = [_read(path) for path in directory.glob("*.json")]
    completed = sorted((row for row in rows if row.get("status") == "completed"),
                       key=lambda row: (row["result"]["fields"]["AUC_MAE"], row["spec"]["id"]))
    write_json(prepared.out / "logs/goal_progress.json", {
        "at": now(), "contract": GUARD_CONTRACT, "completed_candidates": len(completed),
        "best": completed[0] if completed else None,
        "explore_targets_met": any(row["result"].get("targets_met") is True for row in completed),
        "local_training": False, "independent_confirmation": "pending",
        "goal_achieved": False, "holdout_read": False})
    lines = ["# Frozen R1 tail-guard EXPLORE results", "",
             "Thirty fixed guard candidates from ten completed five-seed parents. "
             "No new model fitting; independent confirmation remains pending.", "",
             f"Absolute targets: MAE <= {TARGETS['AUC_MAE']}, peak MAE <= {TARGETS['AUC_PeakMAE']}, "
             f"h4 MAE <= {TARGETS['h4_MAE']}, h16 MAE <= {TARGETS['h16_MAE']}, "
             f"nMAE <= {TARGETS['AUC_nMAE']*100:.1f}%.", "",
             "| Parent | Guard | MAE | Peak MAE | h4 MAE | h16 MAE | nMAE % | Targets met |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in completed:
        fields = row["result"]["fields"]
        lines.append(f"| {row['spec']['parent']} | {row['spec']['policy']['name']} | "
                     f"{fields['AUC_MAE']:.4f} | {fields['AUC_PeakMAE']:.4f} | "
                     f"{fields['h4_MAE']:.4f} | {fields['h16_MAE']:.4f} | "
                     f"{fields['AUC_nMAE']*100:.2f} | {row['result'].get('targets_met')} |")
    (prepared.out / "GOAL_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_search(prepared, evidence: dict, *, score_fn=None) -> None:
    """Freeze all 30 policies, then score each five-seed mean in EXPLORE."""
    _verify_frozen_evidence(prepared, evidence)
    original = _original_r1(prepared)
    parent_specs = initial_specs()
    specs = _guard_specs(parent_specs)
    if len(specs) != 30 or len({row["id"] for row in specs}) != 30:
        raise AssertionError("Guard design must freeze thirty unique candidates")
    plan = {"protocol": "goal_r1_guard_v1", "contract": GUARD_CONTRACT,
            "specs": specs, "seeds": list(SEEDS),
            "sources": _sources(prepared.root), "parent_evidence": evidence,
            "split_sha256": prepared.split_lock["lock_sha256"],
            "policies": [copy.deepcopy(policy) for policy in POLICIES],
            "ranking": "metrics of five individually guarded seed forecasts",
            "selection_arm": "EXPLORE", "independent_confirmation": "pending",
            "goal_achieved": False, "holdout_read": False}
    plan["plan_sha256"] = config_hash(plan)
    plan_path = prepared.out / "logs/execution_plan.json"
    write_json(prepared.out / "logs/target_contract.json", GUARD_CONTRACT, exclusive=True)
    write_json(plan_path, plan, exclusive=True)
    # Caller may inject a scorer only for synthetic tests; production uses the
    # original paired metrics and seed uncertainty implementation.
    scoring = score_fn or goal_r1.score
    root = Path(prepared.root)
    for spec in specs:
        if (prepared.out / "logs/stop_requested.json").exists():
            raise RuntimeError("Requested guard checkpoint stop")
        parent_id = spec["parent"]
        record_path = prepared.out / "logs/experiments" / f"{spec['id']}.json"
        write_json(record_path, {"spec": spec, "status": "running", "at": now(),
                                 "goal_achieved": False, "holdout_read": False})
        identity = config_hash({"spec": spec, "plan": config_hash(plan)})
        mean_path = prediction_path(prepared, spec["id"])
        cached = goal_r1._cached(mean_path, identity)
        if cached is None:
            guarded_frames, seed_audits = [], []
            for seed in SEEDS:
                parent_path = _verify_parent_file(root, evidence, parent_id, seed)
                seed_path = prepared.out / "predictions/seeds/EXPLORE" / spec["id"] / f"seed_{seed}.parquet"
                seed_identity = config_hash({"mean_identity": identity, "seed": seed,
                                             "parent_forecast_sha256": sha256(parent_path)})
                existing = goal_r1._cached(seed_path, seed_identity)
                if existing is None:
                    parent = pd.read_parquet(parent_path)
                    guarded = apply_guard(parent, original, spec["policy"], spec["id"])
                    _validate_frame(prepared, guarded, spec["id"], "EXPLORE")
                    seed_audit = {"seed": seed, "parent": parent_id, "policy": spec["policy"],
                                  "parent_forecast_sha256": sha256(parent_path),
                                  "triggered_rows": int(guarded.guard_triggered.sum()),
                                  "leakage_test": "passed", "holdout_read": False}
                    goal_r1._publish(seed_path, guarded, {"identity": seed_identity, "audit": seed_audit})
                else:
                    guarded, seed_audit = existing
                    _validate_frame(prepared, guarded, spec["id"], "EXPLORE")
                verified_seed = _verify_one_guard_seed(prepared, spec, identity, parent_path, seed)
                if seed_audit != verified_seed["audit"]:
                    raise ValueError("Guard seed cache audit changed after load")
                guarded_frames.append(guarded)
                seed_audits.append(verified_seed)
            mean = _mean_frames(guarded_frames, spec["id"])
            # _mean_frames copies arbitrary extra columns from seed one.  Only
            # the mean prediction may define the final applied correction.
            for name in ("correction", "applied_correction", "guard_triggered"):
                if name in mean:
                    mean.drop(columns=name, inplace=True)
            mean["r1"] = original.pred.to_numpy(float)
            mean["applied_correction"] = mean.pred.to_numpy(float) - mean.r1.to_numpy(float)
            _validate_frame(prepared, mean, spec["id"], "EXPLORE")
            audit = {"n_seeds": len(SEEDS), "seeds": seed_audits,
                     "ranking_basis": "metric of mean of individually guarded forecasts",
                     "parent": parent_id, "policy": spec["policy"],
                     "leakage_test": "passed", "local_training": False,
                     "independent_confirmation": "pending", "holdout_read": False}
            goal_r1._publish(mean_path, mean, {"identity": identity, "audit": audit})
        else:
            mean, audit = cached
            _validate_frame(prepared, mean, spec["id"], "EXPLORE")
            if (audit.get("n_seeds") != len(SEEDS) or audit.get("leakage_test") != "passed"
                    or audit.get("parent") != parent_id or audit.get("policy") != spec["policy"]):
                raise ValueError("Cached guard mean lacks five-seed provenance")
            _verify_guard_seeds(prepared, spec, identity, evidence, audit.get("seeds", []))
        result = scoring(prepared, spec, mean, audit)
        result = {**result, "parent": parent_id, "policy": spec["policy"],
                  "parent_plan_sha256": evidence["parent_plan_sha256"],
                  "guard_triggered_rows_by_seed": {
                      str(entry["seed"]): entry["audit"].get("triggered_rows") for entry in audit["seeds"]},
                  "local_training": False, "independent_confirmation": "pending",
                  "goal_achieved": False, "holdout_read": False}
        if score_fn is None:
            from phase_f import wf_metrics
            parent_path = root / "outputs/phase_f" / evidence["parent_namespace"] / \
                "predictions/EXPLORE" / f"{parent_id}.parquet"
            if sha256(parent_path) != evidence["parents"][parent_id]["mean_sha256"]:
                raise ValueError("Parent mean bytes changed before paired guard comparison")
            paired = wf_metrics.paired_ci(mean, pd.read_parquet(parent_path), n=1000, seed=42)
            dest = prepared.out / "tables" / spec["id"] / "EXPLORE"
            paired_path = dest / "paired_vs_parent.csv"
            paired.to_csv(paired_path, index=False)
            result["paired_vs_parent_sha256"] = sha256(paired_path)
            result["artifacts"] = {**result.get("artifacts", {}),
                                   "paired_vs_parent.csv": result["paired_vs_parent_sha256"]}
            write_json(dest / "manifest.json", result)
        write_json(record_path, {"spec": spec, "status": "completed", "result": result,
                                 "goal_achieved": False, "holdout_read": False})
        print("GUARD_RESULT", json.dumps({"id": spec["id"], "parent": parent_id,
              "policy": spec["policy"], "fields": result.get("fields", {}),
              "targets_met": result.get("targets_met"), "paired_vs_parent_sha256":
              result.get("paired_vs_parent_sha256")}, ensure_ascii=False), flush=True)
        update_progress(prepared)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-namespace", default="goal_r1_v2")
    parser.add_argument("--namespace", default="goal_r1_guard_v1")
    args = parser.parse_args(argv)
    if not _NAMESPACE.fullmatch(args.namespace) or args.namespace == args.parent_namespace:
        raise ValueError("Unsafe or overlapping guard namespace")
    root = Path(__file__).resolve().parents[1]
    original, prepared = goal_r1.prepare(root, args.namespace)
    status_path = prepared.out / "logs/driver_status.json"
    write_json(status_path, {"status": "preflight", "pid": os.getpid(), "at": now(),
                             "goal_achieved": False, "holdout_read": False})
    try:
        evidence = preflight(original, prepared, parent_namespace=args.parent_namespace)
        write_json(status_path, {"status": "running_explore", "pid": os.getpid(), "at": now(),
                                 "parent_namespace": args.parent_namespace,
                                 "goal_achieved": False, "holdout_read": False})
        run_search(prepared, evidence)
        write_json(status_path, {"status": "completed_explore_stage", "at": now(),
                                 "candidate_count": 30, "independent_confirmation": "pending",
                                 "goal_achieved": False, "holdout_read": False})
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "at": now(),
                                 "error_type": type(exc).__name__, "error": str(exc),
                                 "goal_achieved": False, "holdout_read": False})
        raise


if __name__ == "__main__":
    main()
