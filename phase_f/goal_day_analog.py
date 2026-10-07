"""Isolated deterministic historical-day analog EXPLORE runner.

The four fixed configurations use one frozen FIT library per weekly fold.  A
fresh run in a separate model directory checks deterministic reproducibility;
it is never counted as another statistical seed.  No numerical CONFIRM or
holdout artifact is opened here.
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
from phase_f.models import day_analog
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path
from phase_f.wf_final import _junction
from phase_f.wf_models import FRAME_KEY, _arm_view, _validate_frame


NAMESPACE = "goal_r1_day_analog_v2"
PARENT = goal_transition.PARENT
GUARD = goal_transition.GUARD
BEST_GUARD = goal_transition.BEST_GUARD
DESIGN = "DAY_ANALOG_PLAN.md"
SPECS = tuple(
    {"id": f"FG-R4-analog-w{window}-k{k}", "adapter": "day_analog",
     "family": "R1-frozen-fit-day-analog", "arm": "EXPLORE",
     "prefix_length": window, "top_k": k, "bank_days": 56}
    for window in (16, 96) for k in (3, 5)
)
CONTRACT = {
    "protocol": NAMESPACE, "arm": "EXPLORE", "specs": list(SPECS),
    "training": "frozen FIT-only day library per weekly fold; max-h16 embargo",
    "selection": "FIT/STOP-only deterministic recipe; no SCORE library update",
    "n_stochastic_seeds": 0, "n_deterministic_runs": 1,
    "fresh_replay": "second physical run in separate model namespace; reproducibility only",
    "ranking": "metrics of one deterministic primary forecast",
    "independent_confirmation": "pending", "goal_achieved": False,
    "holdout_read": False,
}
CONTRACT["contract_sha256"] = config_hash(CONTRACT)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources(root: Path) -> dict[str, str]:
    names = set(goal_transition._sources(root)) | set(day_analog._SOURCES) | {
        "phase_f/goal_day_analog.py", "phase_f/tests/test_goal_day_analog.py",
        "phase_f/tests/test_day_analog.py",
        "phase_f/models/day_analog.py", "phase_f/wf_final.py",
    }
    return {name: sha256(root / name) for name in sorted(names)}


def _design_sha(root: Path) -> str:
    return sha256(root / "outputs/phase_f" / NAMESPACE / DESIGN)


def _viability_sha(root: Path) -> str:
    path = root / "outputs/phase_f" / NAMESPACE \
        / "diagnostics/fit_bank_support_preflight.json"
    record = _read(path)
    if (record.get("analysis_only") is not True
            or record.get("arm") != "EXPLORE"
            or record.get("library_from_FIT_only") is not True
            or record.get("query_metadata_from_STOP_only") is not True
            or record.get("candidate_fit_performed") is not False
            or record.get("candidate_score_computed") is not False
            or record.get("holdout_read") is not False):
        raise ValueError("FIT-bank support preflight changed its scope")
    rows = record.get("rows", [])
    expected = {(fold, window, h) for fold in (1, 3, 5, 7, 9, 11, 13, 15)
                for window in (16, 96) for h in range(4, 17)}
    if (len(rows) != len(expected)
            or {(row["fold"], row["prefix_length"], row["horizon"])
                for row in rows} != expected
            or any(row["minimum_clean_per_slot"] < 5
                   or row["slots_below_top5"] != 0 for row in rows)):
        raise ValueError("FIT-bank support preflight cohort or viability changed")
    return sha256(path)


def _model_paths_sha(paths: pd.DataFrame, contexts: dict) -> str:
    return day_analog._paths_index(paths, contexts)[1]


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
        "fit_bank_support_preflight_sha256": _viability_sha(root),
        "split_sha256": prepared.split_lock["lock_sha256"],
        "weekly_lock_file_sha256": weekly_locks,
        "seal_sha256": config_hash(prepared.seal),
        "raw_sha256": prepared.seal["raw_sha256"],
        "baseline_prediction_sha256": baselines,
        "baseline_metadata_sha256": {
            name: sha256(prediction_path(prepared, name).with_suffix(".json"))
            for name in baselines},
        "parent_evidence": parent, "guard_evidence": guard,
        "rolling_evidence": rolling, "arm": "EXPLORE",
        "holdout_read": False, "goal_achieved": False,
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
            or plan.get("fit_bank_support_preflight_sha256") != _viability_sha(root)
            or plan.get("split_sha256") != prepared.split_lock["lock_sha256"]
            or plan.get("seal_sha256") != config_hash(prepared.seal)
            or plan.get("raw_sha256") != prepared.seal["raw_sha256"]
            or plan.get("arm") != "EXPLORE" or plan.get("holdout_read") is not False):
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
            or audit.get("smoke_first_fold") is not smoke_first_fold
            or audit.get("paths_sha256") != plan["rolling_evidence"]["model_paths_sha256"]):
        raise ValueError("Analog prediction metadata or causal audit changed")
    _verify_cohort(prepared, frame, spec, smoke_first_fold=smoke_first_fold)
    day_analog.verify_cells(prepared, _model_spec(spec, plan), paths, audit,
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
            checkpoint_dir = prepared.out / "models/day_analog" / spec["id"]
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise ValueError("Fresh replay has a partial pre-existing model cache; preserve evidence")
        frame, audit = day_analog.run(prepared, _model_spec(spec, plan), paths,
                                      smoke_first_fold=smoke_first_fold)
        if run == "fresh_replay" and not all(
                cell.get("cache_reused") is False for cell in audit.get("cells", [])):
            raise ValueError("Fresh replay reused an existing model on first creation")
        _verify_cohort(prepared, frame, spec, smoke_first_fold=smoke_first_fold)
        _verify_frozen(frozen_view or prepared, plan, paths)
        day_analog.verify_cells(prepared, _model_spec(spec, plan), paths, audit,
                                smoke_first_fold=smoke_first_fold)
        _publish(path, frame, {"identity": identity, "audit": audit,
                               "spec": spec, "model_spec": _model_spec(spec, plan),
                               "run": run,
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
        "r1", "applied_correction", "analog_point", "analog_support",
        "analog_neighbors", "analog_fallback", "analog_level_offset"})
    auxiliary += sorted(column for column in set(left) | set(right)
                        if column.startswith("analog_") and column not in auxiliary)
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
    columns = [name for name in ("analog_neighbors", "analog_support",
                                  "analog_fallback") if name in frame]
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
                for name in columns:
                    values = group[name]
                    row[f"{name}_mean"] = float(values.astype(float).mean())
                    row[f"{name}_missing"] = int(values.isna().sum())
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
    artifacts["analog_support.csv"] = _write_csv_locked(dest / "analog_support.csv",
                                                        _support_table(frame))
    primary_path = prediction_path(prepared, spec["id"])
    replay_path = prepared.out / "fresh_replay" / "predictions/EXPLORE" / f"{spec['id']}.parquet"
    primary_model_dir = (prepared.out / "models/day_analog" / spec["id"]).resolve()
    replay_model_dir = (prepared.out / "fresh_replay/models/day_analog" / spec["id"]).resolve()
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
              "analog_support_diagnostic_only": True,
              "operational_phase_e": "not_evaluated",
              "post_observation_design": True,
              "independent_confirmation": "pending",
              "goal_achieved": False, "holdout_read": False,
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
                                  "holdout_read": False, "goal_achieved": False})
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
                              "goal_achieved": False})
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
        "holdout_read": False})
    lines = ["# Frozen FIT day analog EXPLORE", "",
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
                             "goal_achieved": False, "holdout_read": False})
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
            "holdout_read": False})
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "stage": args.stage,
                                 "at": now(), "error_type": type(exc).__name__,
                                 "error": str(exc), "goal_achieved": False,
                                 "holdout_read": False})
        raise


if __name__ == "__main__":
    main()
