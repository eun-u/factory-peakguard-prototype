"""Isolated deterministic recent observed-history analog EXPLORE runner.

The fixed recipe retrieves only quality-masked raw power observed by each
origin. STOP chooses alpha; no evaluation-frame labels update the recipe.
The separate physical replay checks reproducibility, not another seed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from phase_f import goal_guard, goal_r1, goal_r1_paths, goal_transition, wf_metrics
from phase_f.goal_protocol import meets_targets
from phase_f.models import recent_day_analog
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path
from phase_f.wf_final import _junction
from phase_f.wf_models import FRAME_KEY, _arm_view, _validate_frame


NAMESPACE = "goal_r1_recent_analog_v1"
PARENT = goal_transition.PARENT
GUARD = goal_transition.GUARD
BEST_GUARD = goal_transition.BEST_GUARD
DESIGN = "RECENT_ANALOG_PLAN.md"
SPECS = tuple(
    {"id": f"FG-R5-recent-w{window}-k{k}", "adapter": "recent_day_analog",
     "family": "R1-causal-recent-day-analog", "arm": "EXPLORE",
     "prefix_length": window, "top_k": k, "bank_days": 56}
    for window in (16, 96) for k in (3, 5)
)
CONTRACT = {
    "protocol": NAMESPACE, "arm": "EXPLORE", "specs": list(SPECS),
    "prediction_input": "raw observed power through each origin, including elapsed evaluation-period history",
    "training": "stateless recent-history retrieval; no evaluation-frame label access or fitted bank",
    "selection": "purged STOP-only weekly alpha; no SCORE tuning or model-state update",
    "n_stochastic_seeds": 0, "n_deterministic_runs": 1,
    "fresh_replay": "second physical run in separate model namespace; reproducibility only",
    "ranking": "metrics of one deterministic primary forecast",
    "independent_confirmation": "pending", "goal_achieved": False,
    "holdout_read": False, "historical_final_artifact_read": False,
}
CONTRACT["contract_sha256"] = config_hash(CONTRACT)

INPUT_CONTRACT_BASE = {
    "protocol": NAMESPACE, "arm": "EXPLORE", "n_stochastic_seeds": 0,
    "n_deterministic_runs": 1,
    "fixed_specs": [{key: spec[key] for key in (
        "id", "adapter", "prefix_length", "top_k", "bank_days")}
        for spec in SPECS],
    "historical_inputs": "quality-masked raw observed power at timestamps no later than each query origin; may include past elapsed evaluation-period observations",
    "reference_origins": "r=t-d calendar days, integer d=1..56, same quarter slot",
    "reference_suffix": "all h4..h16 source times r+15min*h must be <=t; require all finite and quality-clean",
    "prediction_state_updates": "none; stateless observed-history retrieval, no evaluation-frame labels passed to inference",
    "weight_selection": "pooled STOP-only alpha in [0,.25,.5,.75,1], objective MAE+.25 PeakMAE using FIT tau, ties lower alpha",
    "score_tuning": "forbidden", "current_or_future_target_access": "forbidden",
    "scope": "post-observation exploratory causal input expansion; preserves original search budgets and one frozen final union",
    "independent_confirmation": "pending", "goal_achieved": False,
    "holdout_read": False, "historical_final_artifact_read": False,
}
INPUT_CONTRACT = {**INPUT_CONTRACT_BASE,
                  "contract_sha256": config_hash(INPUT_CONTRACT_BASE)}
R4_MOTIVATION_IDS = tuple(f"FG-R4-analog-w{window}-k{k}"
                          for window in (16, 96) for k in (3, 5))


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources(root: Path) -> dict[str, str]:
    names = set(goal_transition._sources(root)) | set(recent_day_analog._SOURCES) | {
        "phase_f/goal_recent_analog.py", "phase_f/tests/test_goal_recent_analog.py",
        "phase_f/tests/test_recent_day_analog.py",
        "phase_f/models/recent_day_analog.py", "phase_f/wf_final.py",
    }
    return {name: sha256(root / name) for name in sorted(names)}


def _design_sha(root: Path) -> str:
    return sha256(root / "outputs/phase_f" / NAMESPACE / DESIGN)


def _causal_input_contract(root: Path) -> dict:
    path = root / "outputs/phase_f" / NAMESPACE / "CAUSAL_INPUT_CONTRACT.json"
    record = _read(path)
    if record != INPUT_CONTRACT:
        raise ValueError("Recent observed-history causal input contract changed")
    return {"sha256": sha256(path), "record": record}


def _motivation_evidence(root: Path) -> dict:
    """Bind the completed prior EXPLORE wave used to motivate recency."""
    base = root / "outputs/phase_f/goal_r1_day_analog_v2"
    plan_path = base / "logs/execution_plan.json"
    driver_path = base / "logs/driver_status.json"
    support_path = base / "diagnostics/fit_bank_support_preflight.json"
    proof_path = root / "outputs/phase_f/logs/revision_20261006/day_analog_v2_full_verification.json"
    previous_plan = _read(plan_path)
    plan_digest = previous_plan.get("plan_sha256")
    if (previous_plan.get("contract", {}).get("protocol") != "goal_r1_day_analog_v2"
            or plan_digest != config_hash({k: v for k, v in previous_plan.items()
                                          if k != "plan_sha256"})
            or [item.get("id") for item in previous_plan.get("specs", [])]
            != list(R4_MOTIVATION_IDS)
            or previous_plan.get("arm") != "EXPLORE"
            or previous_plan.get("holdout_read") is not False):
        raise ValueError("Completed FG-R4 execution plan changed")
    driver = _read(driver_path)
    if (driver.get("status") != "completed_explore_stage"
            or driver.get("candidate_count") != 4
            or driver.get("full_explore_search_complete") is not True
            or driver.get("holdout_read") is not False):
        raise ValueError("FG-R4 prior EXPLORE wave is incomplete")
    support = _read(support_path)
    if (support.get("analysis_only") is not True
            or support.get("arm") != "EXPLORE"
            or support.get("candidate_fit_performed") is not False
            or support.get("candidate_score_computed") is not False
            or support.get("holdout_read") is not False):
        raise ValueError("FG-R4 FIT-bank diagnostic changed")
    proof = _read(proof_path)
    rows = proof.get("candidate_results", [])
    if (proof.get("plan_sha256") != plan_digest
            or proof.get("source_runtime_raw_split_parent_verified") is not True
            or proof.get("actual_primary_and_fresh_checkpoints_verified") is not True
            or proof.get("completed_candidate_count") != 4
            or proof.get("holdout_read") is not False
            or [row.get("id") for row in rows] != list(R4_MOTIVATION_IDS)
            or any(row.get("n_primary_models") != 8
                   or row.get("n_fresh_models") != 8
                   or row.get("targets_met") is not False for row in rows)):
        raise ValueError("FG-R4 independent root verification changed")
    files = {"execution_plan.json": sha256(plan_path),
             "driver_status.json": sha256(driver_path),
             "fit_bank_support_preflight.json": sha256(support_path),
             "day_analog_v2_full_verification.json": sha256(proof_path)}
    for item in rows:
        candidate = item["id"]
        manifest_path = base / "tables" / candidate / "EXPLORE/manifest.json"
        record_path = base / "logs/experiments" / f"{candidate}.json"
        manifest = _read(manifest_path)
        record = _read(record_path)
        if (manifest.get("candidate") != candidate or manifest.get("arm") != "EXPLORE"
                or manifest.get("holdout_read") is not False
                or manifest.get("targets_met") is not False
                or manifest.get("n_stochastic_seeds") != 0
                or record.get("status") != "completed"
                or record.get("spec", {}).get("id") != candidate
                or record.get("result") != manifest
                or record.get("holdout_read") is not False
                or item.get("score_manifest_sha256") != sha256(manifest_path)
                or item.get("completed_record_sha256") != sha256(record_path)):
            raise ValueError(f"FG-R4 completed score evidence changed: {candidate}")
        files[f"{candidate}/score_manifest.json"] = sha256(manifest_path)
        files[f"{candidate}/completed_record.json"] = sha256(record_path)
    return {"motivation_only": True, "arm": "EXPLORE",
            "completed_ids": list(R4_MOTIVATION_IDS), "root_verified": True,
            "files": files, "historical_final_artifact_read": False}


def _model_paths_sha(paths: pd.DataFrame, contexts: dict) -> str:
    return recent_day_analog._paths_index(paths, contexts)[1]


def _model_spec(spec: dict, plan: dict) -> dict:
    return {**spec, "rolling_audit_sha256":
            plan["rolling_evidence"]["rolling_audit_sha256"]}


def preflight(original, prepared) -> tuple[pd.DataFrame, dict]:
    if prepared.out.name != NAMESPACE:
        raise ValueError("Day-analog runner requires its isolated namespace")
    root = Path(prepared.root)
    baselines = goal_r1.copy_baselines(original, prepared)
    parent = goal_guard.preflight(original, prepared, parent_namespace=PARENT)
    if baselines != parent["baseline_prediction_sha256"]:
        raise ValueError("Copied baselines differ from complete R1 parent")
    guard = goal_transition._guard_evidence(original, prepared, parent)
    paths, rolling = goal_transition._rolling_paths(original, prepared, parent)
    rolling["transition_model_paths_sha256"] = rolling.pop("model_paths_sha256")
    rolling["model_paths_sha256"] = _model_paths_sha(paths, prepared.contexts)
    from phase_c.data import SOURCE_RELATIVE
    from phase_f.harness import seal_parent
    if (seal_parent(root) != prepared.seal
            or sha256(root / SOURCE_RELATIVE) != prepared.seal["raw_sha256"]):
        raise ValueError("Protected parent evidence or raw source changed")
    weekly_locks = {name: sha256(root / name) for name in (
        "outputs/phase_f/logs/walkforward_lock.json",
        "outputs/phase_f/walkforward_v2/walkforward_lock.json")}
    if any(_read(root / name) != prepared.split_lock for name in weekly_locks):
        raise ValueError("Physical weekly split lock changed")
    plan = {
        "contract": CONTRACT, "specs": list(SPECS), "sources": _sources(root),
        "runtime": goal_transition._runtime(), "design_sha256": _design_sha(root),
        "causal_input_contract": _causal_input_contract(root),
        "split_sha256": prepared.split_lock["lock_sha256"],
        "weekly_lock_file_sha256": weekly_locks,
        "seal_sha256": config_hash(prepared.seal),
        "raw_sha256": prepared.seal["raw_sha256"],
        "baseline_prediction_sha256": baselines,
        "baseline_metadata_sha256": {
            name: sha256(prediction_path(prepared, name).with_suffix(".json"))
            for name in baselines},
        "parent_evidence": parent, "guard_evidence": guard,
        "prior_r4_motivation_evidence": _motivation_evidence(root),
        "rolling_evidence": rolling, "arm": "EXPLORE",
        "holdout_read": False, "historical_final_artifact_read": False,
        "goal_achieved": False,
    }
    plan["plan_sha256"] = config_hash(plan)
    write_json(prepared.out / "logs/target_contract.json", CONTRACT, exclusive=True)
    write_json(prepared.out / "logs/execution_plan.json", plan, exclusive=True)
    _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
    return paths, plan


def _verify_path_digests(paths: pd.DataFrame, contexts: dict, rolling: dict) -> None:
    goal_transition._verify_path_digests(paths, contexts, {
        "paths_sha256": rolling["paths_sha256"],
        "model_paths_sha256": rolling["transition_model_paths_sha256"]})
    if _model_paths_sha(paths, contexts) != rolling["model_paths_sha256"]:
        raise ValueError("Frozen analog normalized model path digest changed")


def _verify_frozen(prepared, plan: dict, paths: pd.DataFrame,
                   *, deep_model_snapshot: bool = False) -> None:
    root = Path(prepared.root)
    from phase_c.data import SOURCE_RELATIVE
    from phase_f.harness import seal_parent
    stored = _read(prepared.out / "logs/execution_plan.json")
    if (stored != plan or plan.get("plan_sha256") != config_hash(
            {k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("contract") != CONTRACT or plan.get("specs") != list(SPECS)
            or plan.get("sources") != _sources(root)
            or plan.get("runtime") != goal_transition._runtime()
            or plan.get("design_sha256") != _design_sha(root)
            or plan.get("causal_input_contract") != _causal_input_contract(root)
            or plan.get("prior_r4_motivation_evidence") != _motivation_evidence(root)
            or plan.get("split_sha256") != prepared.split_lock["lock_sha256"]
            or plan.get("seal_sha256") != config_hash(prepared.seal)
            or plan.get("raw_sha256") != prepared.seal["raw_sha256"]
            or plan.get("arm") != "EXPLORE" or plan.get("holdout_read") is not False
            or plan.get("historical_final_artifact_read") is not False):
        raise ValueError("Frozen day-analog plan/source/runtime/design changed")
    if (seal_parent(root) != prepared.seal
            or sha256(root / SOURCE_RELATIVE) != plan["raw_sha256"]):
        raise ValueError("Protected 803-file seal or raw source changed")
    for name, digest in plan["weekly_lock_file_sha256"].items():
        if sha256(root / name) != digest or _read(root / name) != prepared.split_lock:
            raise ValueError("Physical weekly split lock changed")
    goal_guard._verify_frozen_evidence(prepared, plan["parent_evidence"])
    for name, digest in plan["baseline_prediction_sha256"].items():
        path = prediction_path(prepared, name)
        meta = _read(path.with_suffix(".json"))
        if (sha256(path) != digest
                or sha256(path.with_suffix(".json"))
                != plan["baseline_metadata_sha256"][name]
                or meta.get("sha256") != digest or meta.get("arm") != "EXPLORE"):
            raise ValueError("Frozen copied baseline changed")
    guard = plan["guard_evidence"]
    guard_dir = root / "outputs/phase_f" / GUARD
    for name, key in (
            ("logs/execution_plan.json", "plan_sha256"),
            ("logs/driver_status.json", "driver_sha256"),
            (f"logs/experiments/{BEST_GUARD}.json", "record_sha256"),
            (f"tables/{BEST_GUARD}/EXPLORE/manifest.json", "score_manifest_sha256"),
            (f"predictions/EXPLORE/{BEST_GUARD}.parquet", "mean_sha256"),
            (f"predictions/EXPLORE/{BEST_GUARD}.json", "mean_metadata_sha256")):
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
    for part in rolling["parts"]:
        path = manifest_path.parent / part["file"]
        if (sha256(path) != part["sha256"]
                or goal_r1_paths._hash_frame(goal_r1_paths._sort_paths(pd.read_parquet(path)))
                != part["content_sha256"]):
            raise ValueError("Frozen R1 producer part changed")
    _verify_path_digests(paths, prepared.contexts, rolling)
    if config_hash(paths.attrs.get("causal_provenance", {})) != rolling["rolling_audit_sha256"]:
        raise ValueError("Frozen R1 path causal provenance changed")
    if deep_model_snapshot:
        _, model = goal_r1_paths._snapshot(prepared)
        execution = goal_r1_paths._execution_fingerprint(rolling["execution"]["device"])
        if model != rolling["model"] or execution != rolling["execution"]:
            raise ValueError("Pinned R1 model weights or runtime changed")


def _expected_keys(prepared, *, smoke_first_fold: bool) -> pd.MultiIndex:
    first = min(fold for _, fold in prepared.contexts)
    entries = []
    for (h, fold), context in prepared.contexts.items():
        if smoke_first_fold and fold != first:
            continue
        for role in ("cal", "score"):
            origins = pd.DatetimeIndex(prepared.origins(h, fold, role))
            target = pd.DatetimeIndex(context["target_time"].loc[origins])
            entries.extend((h, fold, role, origin, end)
                           for origin, end in zip(origins, target))
    return pd.MultiIndex.from_tuples(sorted(entries), names=list(FRAME_KEY))


def _verify_cohort(prepared, frame: pd.DataFrame, spec: dict,
                   *, smoke_first_fold: bool = False) -> None:
    if (frame.empty or not frame.model.eq(spec["id"]).all()
            or not np.isfinite(frame.pred.to_numpy(float)).all()):
        raise ValueError("Analog forecast model or values changed")
    keys = pd.MultiIndex.from_frame(frame.loc[:, list(FRAME_KEY)])
    expected = _expected_keys(prepared, smoke_first_fold=smoke_first_fold)
    if (len(keys) != len(expected) or keys.has_duplicates
            or not keys.sort_values().equals(expected)):
        raise ValueError("Analog forecast does not match locked weekly cohort")
    required_recent = {"recent_analog_point", "recent_analog_support",
                       "recent_analog_fallback", "recent_analog_alpha",
                       "recent_analog_latest_observed",
                       "recent_analog_elapsed_eval_reads"}
    if not required_recent <= set(frame):
        raise ValueError("Recent analog forecast lacks causal input diagnostics")
    fallback = frame.recent_analog_fallback
    analog_point = pd.to_numeric(frame.recent_analog_point, errors="raise")
    support = pd.to_numeric(frame.recent_analog_support, errors="raise")
    if fallback.isna().any() or not fallback.isin((True, False)).all():
        raise ValueError("Recent analog fallback marker changed")
    fallback = fallback.astype(bool)
    if (not np.isfinite(analog_point.loc[~fallback].to_numpy(float)).all()
            or analog_point.loc[fallback].notna().any()
            or support.isna().any() or support.lt(0).any()
            or not np.equal(support, support.astype(int)).all()):
        raise ValueError("Recent analog support, fallback, or unblended point changed")
    latest = pd.to_datetime(frame.recent_analog_latest_observed, errors="raise")
    if (latest.notna() & latest.gt(pd.DatetimeIndex(frame.origin))).any():
        raise ValueError("Recent analog read raw power after query origin")
    elapsed = pd.to_numeric(frame.recent_analog_elapsed_eval_reads, errors="raise")
    if (elapsed.isna().any() or not np.isfinite(elapsed.to_numpy(float)).all()
            or elapsed.lt(0).any() or not np.equal(elapsed, elapsed.astype(int)).all()):
        raise ValueError("Recent analog elapsed evaluation input count changed")
    reference_cols = sorted(column for column in frame if column.startswith("recent_analog_ref"))
    if reference_cols != sorted(f"recent_analog_ref{i}"
                                for i in range(1, spec["top_k"] + 1)):
        raise ValueError("Recent analog reference provenance or k differs")
    for column in reference_cols:
        references = pd.to_datetime(frame[column], errors="raise")
        if (references.notna() & references.ge(pd.DatetimeIndex(frame.origin))).any():
            raise ValueError("Recent analog reference is not historical")
    if smoke_first_fold:
        if (not frame.loc[frame.role.eq("score"), "arm"].eq("EXPLORE").all()
                or not frame.loc[frame.role.eq("cal"), "arm"].eq("CAL").all()
                or set(frame.horizon) != set(range(4, 17))):
            raise ValueError("Analog smoke lacks its full first-fold horizons")
        smoke_view = copy.copy(prepared)
        first = min(fold for _, fold in prepared.contexts)
        smoke_view.contexts = {key: value for key, value in prepared.contexts.items()
                               if key[1] == first}
        _validate_frame(smoke_view, frame, spec["id"], "EXPLORE")
    else:
        _validate_frame(prepared, frame, spec["id"], "EXPLORE")


def _cached(path: Path, identity: str):
    meta_path = path.with_suffix(".json")
    if path.exists() != meta_path.exists():
        raise ValueError(f"Incomplete analog forecast cache; preserve evidence: {path}")
    if not path.exists():
        return None
    meta = _read(meta_path)
    if (meta.get("identity") != identity or meta.get("sha256") != sha256(path)
            or meta.get("arm") != "EXPLORE"):
        raise ValueError("Analog forecast cache identity or physical SHA changed")
    return pd.read_parquet(path), meta


def _publish(path: Path, frame: pd.DataFrame, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    if path.exists() or path.with_suffix(".json").exists() or temporary.exists():
        raise ValueError("Analog forecast output already exists or is partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), {
        **metadata, "sha256": sha256(path), "arm": "EXPLORE"}, exclusive=True)


def _prediction_identity(plan: dict, spec: dict, run: str) -> str:
    return config_hash({"plan_sha256": plan["plan_sha256"],
                        "spec": spec, "run": run, "arm": "EXPLORE"})


def _verify_prediction(prepared, spec: dict, paths: pd.DataFrame,
                       plan: dict, run: str, *, smoke_first_fold: bool = False):
    path = (prepared.out / "smoke" / spec["id"] / "first_fold.parquet"
            if smoke_first_fold else prediction_path(prepared, spec["id"]))
    cached = _cached(path, _prediction_identity(plan, spec, run))
    if cached is None:
        raise ValueError("Analog forecast cache is missing")
    frame, meta = cached
    audit = meta.get("audit", {})
    if (meta.get("spec") != spec or meta.get("model_spec") != _model_spec(spec, plan)
            or meta.get("run") != run
            or meta.get("plan_sha256") != plan["plan_sha256"]
            or meta.get("smoke_first_fold") is not smoke_first_fold
            or audit.get("leakage_test") != "passed"
            or audit.get("holdout_read") is not False
            or audit.get("historical_final_artifact_read") is not False
            or meta.get("historical_final_artifact_read") is not False
            or audit.get("smoke_first_fold") is not smoke_first_fold
            or audit.get("paths_sha256") != plan["rolling_evidence"]["model_paths_sha256"]):
        raise ValueError("Analog prediction metadata or causal audit changed")
    for cell in audit.get("cells", []):
        coverage = cell.get("coverage", [])
        if (len(coverage) != 26
                or {(row.get("horizon"), row.get("role")) for row in coverage}
                != {(h, role) for h in range(4, 17) for role in ("cal", "score")}
                or any(row.get("latest_source_exceeds_origin_count") != 0
                       for row in coverage)):
            raise ValueError("Recent analog checkpoint lacks causal input coverage")
    _verify_cohort(prepared, frame, spec, smoke_first_fold=smoke_first_fold)
    recent_day_analog.verify_cells(prepared, _model_spec(spec, plan), paths, audit,
                            smoke_first_fold=smoke_first_fold)
    return frame, meta


def _run_prediction(prepared, spec: dict, paths: pd.DataFrame, plan: dict,
                    run: str, *, smoke_first_fold: bool = False,
                    frozen_view=None):
    path = (prepared.out / "smoke" / spec["id"] / "first_fold.parquet"
            if smoke_first_fold else prediction_path(prepared, spec["id"]))
    identity = _prediction_identity(plan, spec, run)
    cached = _cached(path, identity)
    if cached is None:
        if run == "fresh_replay":
            checkpoint_dir = prepared.out / "models/recent_day_analog" / spec["id"]
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise ValueError("Fresh replay has a partial pre-existing model cache; preserve evidence")
        frame, audit = recent_day_analog.run(prepared, _model_spec(spec, plan), paths,
                                      smoke_first_fold=smoke_first_fold)
        if run == "fresh_replay" and not all(
                cell.get("cache_reused") is False for cell in audit.get("cells", [])):
            raise ValueError("Fresh replay reused an existing model on first creation")
        _verify_cohort(prepared, frame, spec, smoke_first_fold=smoke_first_fold)
        _verify_frozen(frozen_view or prepared, plan, paths)
        recent_day_analog.verify_cells(prepared, _model_spec(spec, plan), paths, audit,
                                smoke_first_fold=smoke_first_fold)
        _publish(path, frame, {"identity": identity, "audit": audit,
                               "spec": spec, "model_spec": _model_spec(spec, plan),
                               "run": run,
                               "historical_final_artifact_read": False,
                               "plan_sha256": plan["plan_sha256"],
                               "smoke_first_fold": smoke_first_fold})
    return _verify_prediction(prepared, spec, paths, plan, run,
                              smoke_first_fold=smoke_first_fold)


def run_smoke(prepared, paths: pd.DataFrame, plan: dict) -> None:
    _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
    for spec in SPECS:
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        _run_prediction(prepared, spec, paths, plan, "smoke", smoke_first_fold=True)
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        print("ANALOG_SMOKE_READY", spec["id"], flush=True)


def _verify_smoke(prepared, paths: pd.DataFrame, plan: dict) -> None:
    for spec in SPECS:
        path = prepared.out / "smoke" / spec["id"] / "first_fold.parquet"
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise ValueError(f"Analog technical smoke is missing: {spec['id']}")
        _verify_prediction(prepared, spec, paths, plan, "smoke",
                           smoke_first_fold=True)


def _replay_view(prepared):
    source = copy.copy(prepared.source)
    source.out = prepared.out / "fresh_replay"
    replay = _arm_view(source, "EXPLORE")
    if (replay.source is prepared.source
            or list(replay.contexts) != list(prepared.contexts)
            or any(replay.contexts[key] is not prepared.contexts[key]
                   for key in prepared.contexts)):
        raise ValueError("Fresh replay changed the selected EXPLORE weekly contexts")
    replay.out.mkdir(parents=True, exist_ok=True)
    _junction(replay.out / "models", Path(r"D:\PeakGuard_PhaseF_20261003\models")
              / NAMESPACE / "fresh_replay")
    primary_models = (prepared.out / "models").resolve()
    replay_models = (replay.out / "models").resolve()
    if (replay.out.resolve() == prepared.out.resolve()
            or primary_models == replay_models
            or os.path.samefile(primary_models, replay_models)):
        raise ValueError("Fresh replay physical model namespace aliases primary")
    return replay


def _compare_replay(primary: pd.DataFrame, replay: pd.DataFrame) -> dict:
    left = primary.sort_values(list(FRAME_KEY)).reset_index(drop=True)
    right = replay.sort_values(list(FRAME_KEY)).reset_index(drop=True)
    if (not left[list(FRAME_KEY)].equals(right[list(FRAME_KEY)])
            or not left["pred"].equals(right["pred"])):
        raise ValueError("Fresh replay keys or deterministic prediction changed")
    auxiliary = sorted((set(left) | set(right)) & {
        "r1", "applied_correction"})
    auxiliary += sorted(column for column in set(left) | set(right)
                        if column.startswith("recent_analog_") and column not in auxiliary)
    if set(auxiliary) - set(left) or set(auxiliary) - set(right):
        raise ValueError("Fresh replay auxiliary schema changed")
    for column in auxiliary:
        if not left[column].equals(right[column]):
            raise ValueError(f"Fresh replay auxiliary prediction changed: {column}")
    for column in ("y", "tau", "d2", "fit_mean", "mase_scale"):
        if not left[column].equals(right[column]):
            raise ValueError(f"Fresh replay locked context changed: {column}")
    return {"ordered_keys_equal": True, "point_predictions_equal": True,
            "auxiliary_columns_equal": auxiliary, "locked_context_equal": True}


def _write_csv_locked(path: Path, frame: pd.DataFrame) -> str:
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    payload = buffer.getvalue().encode("utf-8")
    import hashlib
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != digest:
            raise ValueError(f"Existing analog score artifact changed: {path.name}")
    else:
        path.write_bytes(payload)
    return digest


def _support_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ("cal", "score"):
        selected = frame.loc[frame.role.eq(role)]
        cohorts = [("all", selected)]
        if role == "score":
            cohorts.append(("D2", selected.loc[selected.d2]))
        for cohort, sample in cohorts:
            for (fold, horizon), group in sample.groupby(["fold", "horizon"], sort=True):
                row = {"role": role, "cohort": cohort, "fold": int(fold),
                       "horizon": int(horizon), "rows": len(group)}
                for name in ("recent_analog_support", "recent_analog_fallback",
                             "recent_analog_elapsed_eval_reads"):
                    values = group[name]
                    row[f"{name}_mean"] = float(values.astype(float).mean())
                    row[f"{name}_sum"] = float(values.astype(float).sum())
                    row[f"{name}_missing"] = int(values.isna().sum())
                latest = pd.to_datetime(group.recent_analog_latest_observed)
                origin = pd.DatetimeIndex(group.origin)
                row["latest_observed_after_origin_count"] = int(
                    (latest.notna() & latest.gt(origin)).sum())
                row["rows_with_elapsed_evaluation_raw_input"] = int(
                    group.recent_analog_elapsed_eval_reads.gt(0).sum())
                if "recent_analog_ref1" in group:
                    first = pd.to_datetime(group.recent_analog_ref1)
                    age = (origin - first).dt.total_seconds() / 86400.
                    row["nearest_reference_age_days_mean"] = float(age.mean())
                    row["nearest_reference_age_days_min"] = float(age.min())
                rows.append(row)
    return pd.DataFrame(rows)


def _score(prepared, plan: dict, spec: dict, frame: pd.DataFrame,
           primary_meta: dict, replay_meta: dict, replay_proof: dict) -> dict:
    tables = wf_metrics.evaluate(frame, "EXPLORE")
    fields = goal_r1._target_fields(tables)
    dest = prepared.out / "tables" / spec["id"] / "EXPLORE"
    artifacts = {f"{name}.csv": _write_csv_locked(dest / f"{name}.csv", table)
                 for name, table in tables.items()}
    pairs = []
    for name in ("B5", "R1"):
        paired = wf_metrics.paired_ci(frame, load_predictions(prepared, BASELINES[name]),
                                      n=1000, seed=42)
        paired["baseline"] = name
        pairs.append(paired)
    guard_path = Path(prepared.root) / "outputs/phase_f" / GUARD / "predictions/EXPLORE" / f"{BEST_GUARD}.parquet"
    if sha256(guard_path) != plan["guard_evidence"]["mean_sha256"]:
        raise ValueError("Frozen guard changed before paired scoring")
    paired = wf_metrics.paired_ci(frame, pd.read_parquet(guard_path), n=1000, seed=42)
    paired["baseline"] = BEST_GUARD
    pairs.append(paired)
    artifacts["paired_ci.csv"] = _write_csv_locked(dest / "paired_ci.csv",
                                                   pd.concat(pairs, ignore_index=True))
    artifacts["recent_analog_support.csv"] = _write_csv_locked(
        dest / "recent_analog_support.csv", _support_table(frame))
    primary_path = prediction_path(prepared, spec["id"])
    replay_path = prepared.out / "fresh_replay" / "predictions/EXPLORE" / f"{spec['id']}.parquet"
    primary_model_dir = (prepared.out / "models/recent_day_analog" / spec["id"]).resolve()
    replay_model_dir = (prepared.out / "fresh_replay/models/recent_day_analog" / spec["id"]).resolve()
    if primary_model_dir == replay_model_dir or os.path.samefile(primary_model_dir, replay_model_dir):
        raise ValueError("Primary and fresh replay physical checkpoint directories alias")
    if not all(cell.get("cache_reused") is False
               for cell in replay_meta["audit"]["cells"]):
        raise ValueError("Fresh replay contains a reused fold model")
    record = {"candidate": spec["id"], "arm": "EXPLORE", "fields": fields,
              **meets_targets(fields), "n_seeds": 1, "n_stochastic_seeds": 0,
              "n_distinct_deterministic_forecasts": 1,
              "ranking_basis": "one deterministic primary forecast",
              "individual_seed_sd": "not_applicable",
              "fresh_replay": {**replay_proof,
                  "primary_prediction_sha256": sha256(primary_path),
                  "primary_metadata_sha256": sha256(primary_path.with_suffix(".json")),
                  "replay_prediction_sha256": sha256(replay_path),
                  "replay_metadata_sha256": sha256(replay_path.with_suffix(".json")),
                  "primary_model_directory": str(primary_model_dir),
                  "replay_model_directory": str(replay_model_dir),
                  "replay_initial_models_were_fresh": True,
                  "primary_cells": primary_meta["audit"]["cells"],
                  "replay_cells": replay_meta["audit"]["cells"]},
              "prediction_sha256": sha256(primary_path),
              "best_guard": BEST_GUARD,
              "best_guard_mean_sha256": plan["guard_evidence"]["mean_sha256"],
              "recent_raw_input_support_diagnostic_only": True,
              "operational_phase_e": "not_evaluated",
              "post_observation_design": True,
              "independent_confirmation": "pending",
              "goal_achieved": False, "holdout_read": False,
              "historical_final_artifact_read": False,
              "artifacts": artifacts}
    write_json(dest / "manifest.json", record, exclusive=True)
    return record


def _record(path: Path, value: dict) -> None:
    if path.exists():
        existing = _read(path)
        if existing.get("status") == "completed" and existing != value:
            raise ValueError("Completed analog candidate record changed")
        if existing.get("status") not in {"running", "completed"}:
            raise ValueError("Unexpected analog candidate record status")
    write_json(path, value)


def run_search(prepared, paths: pd.DataFrame, plan: dict) -> None:
    _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
    _verify_smoke(prepared, paths, plan)
    replay = _replay_view(prepared)
    for spec in SPECS:
        if (prepared.out / "logs/stop_requested.json").exists():
            raise RuntimeError("Requested analog checkpoint stop")
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        record_path = prepared.out / "logs/experiments" / f"{spec['id']}.json"
        if not record_path.exists():
            _record(record_path, {"status": "running", "spec": spec,
                                  "holdout_read": False, "goal_achieved": False,
                                  "historical_final_artifact_read": False})
        primary_frame, primary_meta = _run_prediction(prepared, spec, paths, plan, "primary")
        _verify_frozen(prepared, plan, paths)
        replay_frame, replay_meta = _run_prediction(replay, spec, paths, plan,
                                                      "fresh_replay", frozen_view=prepared)
        replay_proof = _compare_replay(primary_frame, replay_frame)
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        result = _score(prepared, plan, spec, primary_frame,
                        primary_meta, replay_meta, replay_proof)
        _verify_frozen(prepared, plan, paths, deep_model_snapshot=True)
        _record(record_path, {"status": "completed", "spec": spec,
                              "result": result, "holdout_read": False,
                              "goal_achieved": False,
                              "historical_final_artifact_read": False})
        update_progress(prepared)
        print("ANALOG_CANDIDATE_RESULT", json.dumps({"id": spec["id"],
              "fields": result["fields"], "targets_met": result["targets_met"]}),
              flush=True)


def update_progress(prepared) -> None:
    base = prepared.out / "logs/experiments"
    rows = [_read(path) for path in base.glob("*.json")] if base.exists() else []
    completed = sorted((row for row in rows if row.get("status") == "completed"),
                       key=lambda row: (row["result"]["fields"]["AUC_MAE"], row["spec"]["id"]))
    write_json(prepared.out / "logs/goal_progress.json", {
        "at": now(), "contract": CONTRACT, "completed_candidates": len(completed),
        "best": completed[0] if completed else None,
        "explore_targets_met": any(row["result"]["targets_met"] for row in completed),
        "independent_confirmation": "pending", "goal_achieved": False,
        "holdout_read": False, "historical_final_artifact_read": False})
    lines = ["# Causal recent observed-history analog EXPLORE", "",
             "Post-observation exploratory results; independent confirmation pending. No holdout read.", "",
             "| Candidate | MAE | Peak MAE | h4 MAE | h16 MAE | nMAE % | Targets met |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in completed:
        f = row["result"]["fields"]
        lines.append(f"| {row['spec']['id']} | {f['AUC_MAE']:.4f} | {f['AUC_PeakMAE']:.4f} | "
                     f"{f['h4_MAE']:.4f} | {f['h16_MAE']:.4f} | {f['AUC_nMAE']*100:.2f} | "
                     f"{row['result']['targets_met']} |")
    (prepared.out / "GOAL_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "smoke", "search"), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    original, prepared = goal_r1.prepare(root, NAMESPACE)
    status_path = prepared.out / "logs/driver_status.json"
    write_json(status_path, {"status": "running", "pid": os.getpid(),
                             "stage": args.stage, "at": now(),
                             "goal_achieved": False, "holdout_read": False,
                             "historical_final_artifact_read": False})
    try:
        paths, plan = preflight(original, prepared)
        if args.stage == "smoke":
            run_smoke(prepared, paths, plan)
        elif args.stage == "search":
            run_search(prepared, paths, plan)
        write_json(status_path, {
            "status": "completed_explore_stage" if args.stage == "search"
                      else "completed_technical_stage",
            "stage": args.stage, "at": now(),
            "candidate_count": len(SPECS) if args.stage == "search" else 0,
            "full_explore_search_complete": args.stage == "search",
            "independent_confirmation": "pending", "goal_achieved": False,
            "holdout_read": False, "historical_final_artifact_read": False})
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "stage": args.stage,
                                 "at": now(), "error_type": type(exc).__name__,
                                 "error": str(exc), "goal_achieved": False,
                                 "holdout_read": False,
                                 "historical_final_artifact_read": False})
        raise


if __name__ == "__main__":
    main()

