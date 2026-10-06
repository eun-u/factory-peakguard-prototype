"""Resumable, sequential Phase F search and one-time development confirmation.

Factories consume only completed EXPLORE records. A wave's saved specification
is authoritative on resume, including its chosen parents. This module never
loads the final holdout or historical final artifacts.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from phase_f.registry import Registry, config_hash, now, sha256, write_json

TERMINAL = frozenset({"completed", "failed", "unsupported", "rejected", "diagnostic_passed"})
FAMILIES = tuple(f"F{i}" for i in range(12))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_path(prepared, stage):
    return prepared.out / "logs" / "workflow" / f"stage_{stage}.json"


def _status(prepared, stage, **fields):
    path = _stage_path(prepared, stage)
    record = _read(path) if path.exists() else {"stage": stage, "started_at": now()}
    record.update(fields, updated_at=now(), historical_final_artifact_read=False,
                  holdout_read=False)
    write_json(path, record)
    return record


def _locked_specs(prepared, name, factory, hypothesis):
    """Read a lock before calling an adaptive EXPLORE-dependent factory."""
    from phase_f.experiment_plan import lock_wave

    path = prepared.out / "logs" / "waves" / f"{name}.json"
    if path.exists():
        record = _read(path)
        if (record.get("name") != name or record.get("selection_arm") != "EXPLORE" or
                record.get("config_sha256") != config_hash(record.get("specs")) or
                record.get("holdout_read") is not False or
                record.get("historical_final_artifact_read") is not False):
            raise ValueError(f"Changed or invalid wave lock: {name}")
        return record["specs"]
    specs = factory()
    if not isinstance(specs, list) or not specs:
        raise RuntimeError(f"No configurations produced for required wave {name}")
    if any(not isinstance(spec, dict) or not spec.get("id") for spec in specs):
        raise ValueError(f"Malformed configurations in {name}")
    return lock_wave(prepared.root, name, specs, hypothesis)


def _execute_wave(prepared, name, factory, hypothesis, *, retry=False):
    from phase_f.run import execute

    registry = Registry(prepared.root)
    specs = _locked_specs(prepared, name, factory, hypothesis)
    for spec in specs:
        registry.register(spec)
    checkpoint = prepared.out / "logs" / "workflow" / f"{name}.json"
    state = _read(checkpoint) if checkpoint.exists() else {"name": name, "spec_ids": [s["id"] for s in specs]}
    if state["spec_ids"] != [s["id"] for s in specs]:
        raise ValueError(f"Wave checkpoint differs from lock: {name}")
    failures = 0
    for spec in specs:
        row = execute(prepared, spec, registry, retry=retry)
        if row["status"] not in TERMINAL:
            raise RuntimeError(f"Unresolved experiment {spec['id']}: {row['status']}")
        failures = failures + 1 if row["status"] in ("failed", "rejected") else 0
        state.update(last_id=spec["id"], resolved=sum(registry.read(s["id"])["status"] in TERMINAL for s in specs),
                     total=len(specs), status="running", updated_at=now(),
                     historical_final_artifact_read=False, holdout_read=False)
        write_json(checkpoint, state)
        if failures >= 3:
            raise RuntimeError(f"Three consecutive experiment failures in {name}; repair and resume explicitly")
    state.update(status="completed", completed_at=now())
    write_json(checkpoint, state)
    return state


def _best_diagnostic(prepared):
    from phase_f.experiment_plan import completed

    rows = completed(prepared.root)
    return rows[0]["exp_id"] if rows else None


def _diagnostic(prepared, name, factory):
    """Register a non-candidate diagnostic, preserving explicit failure evidence."""
    registry = Registry(prepared.root)
    spec = {"id": name, "family": name.split("-")[0], "tier": 3,
            "adapter": "diagnostic", "note": "Descriptive evidence only; ineligible for model selection"}
    row = registry.register(spec)
    if row["status"] == "diagnostic_passed":
        return row
    try:
        artifacts = factory()
        registry.update(name, status="diagnostic_passed", artifacts=artifacts)
    except Exception as exc:
        registry.update(name, status="failed", error_type=type(exc).__name__,
                        error=str(exc).replace(str(prepared.root), "<repo>"))
        raise
    return registry.read(name)


def _progress(prepared, stage):
    from phase_f.reporting import generate_progress
    from phase_f.run import stage_summary

    stage_summary(prepared, stage, Registry(prepared.root))
    candidate = _best_diagnostic(prepared)
    generate_progress(prepared.root, selected_id=candidate)
    _diagnostic(prepared, f"F10-stage{stage}", lambda: {"progress_report_sha256":
        sha256(prepared.out / "progress_report.md"), "candidate": candidate})
    if stage == 3:
        _diagnostic(prepared, "F8-4-peak-analysis", lambda: {
            "candidate": candidate,
            "figure_sha256": sha256(prepared.out / "figures" / "peak_signed_errors.png")})


def _git_commit(prepared, message):
    """Commit only explicit Phase F code and small evidence, never model data."""
    root = prepared.root
    allowed = [root/name for name in ('DECISIONS.md','PROGRESS.md') if (root/name).is_file()]
    for folder, dirs, files in os.walk(root / "phase_f", topdown=True):
        dirs[:] = [name for name in dirs if name not in ("__pycache__", ".pytest_cache", "_workflow_tmp")
                 and not name.startswith(("pytest", "test_", "_tmp"))]
        allowed.extend(Path(folder) / name for name in files if Path(name).suffix in (".py", ".md")
                       and not (Path(folder) / name).is_symlink())
    out = prepared.out
    central = root/'outputs/phase_f'
    if out != central:
        allowed.extend(p for p in (central/'registry.csv',central/'logs/active_revision.json',
            central/'logs/walkforward_lock.json',out/'.gitignore') if p.is_file())
        revision = central/'logs/revision_20261006'
        if (revision/'.gitignore').is_file():allowed.append(revision/'.gitignore')
        allowed.extend(p for p in revision.rglob('*') if p.is_file() and p.suffix in ('.md','.json','.csv','.py'))
        allowed.extend(p for length in (512,1024,2048,4096,8192)
                       if (p := central/'logs/experiments'/f'F6-1-c{length}-native_mean.json').is_file())
    blocked = {"optional_envs", "env", "cache", "models", "predictions", "tests_tmp", "test_scratch",
               ".pytest_cache", "__pycache__"}
    for folder, dirs, files in os.walk(out, topdown=True):
        relative_folder = Path(folder).relative_to(out)
        if relative_folder == Path("."):
            dirs[:] = [name for name in dirs if name in ("logs", "tables", "figures", "finalists_v2", "pc3_finalists")]
        else:
            dirs[:] = [name for name in dirs if name not in blocked and
                     not name.startswith(("pytest", "test_", "_tmp"))]
        allowed.extend(Path(folder) / name for name in files
                       if Path(name).suffix in (".md", ".csv", ".json", ".png") and
                       name != 'score_probabilities.csv' and
                       not name.startswith(("pytest", "test_", "_tmp")) and
                       not (Path(folder) / name).is_symlink())
    if not allowed:
        return None
    relative = [str(p.relative_to(root)) for p in sorted(set(allowed))]
    for start in range(0, len(relative), 100):
        subprocess.run(["git", "add", "--", *relative[start:start + 100]], cwd=root, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=root).decode().split("\0")
    staged = [name for name in staged if name]
    admitted = {Path(path).as_posix() for path in relative}
    if not set(staged) <= admitted:
        raise RuntimeError("Other staged files exist; refusing to include them in a Phase F commit")
    if not staged:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _verify_committed_selection(prepared, lock):
    """A resumed CONFIRM reservation can only use its already committed lock."""
    path = "outputs/phase_f/logs/confirm_lock.json"
    committed = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=prepared.root)
    if config_hash(json.loads(committed)) != config_hash(lock):
        raise ValueError("Reserved CONFIRM selection lock differs from HEAD")


def _stage0(prepared, retry):
    from phase_f.run import catalog
    from phase_f.baseline_replay import run as baseline_replay
    from phase_f.leakage_tests import run_oracle_negative_control

    run_oracle_negative_control(prepared.root)
    replay = prepared.out / "logs" / "parent_reproduction.json"
    if not replay.exists() or not all((prepared.out / "cache" / f"parent_{model}_cal_score.parquet").is_file()
                                  for model in ("B1", "B5", "M1", "M1-W", "M2")):
        baseline_replay(prepared)
    _execute_wave(prepared, "stage0_catalog", lambda: catalog(0),
                  "Replay fixed baselines and corrected statistical comparators", retry=retry)


def _stage1(prepared, retry):
    from phase_f.run import catalog
    from phase_f import experiment_plan as plan

    _execute_wave(prepared, "stage1_catalog", lambda: catalog(1),
                  "Complete F1 feature and F5/F6 tier-one grids", retry=retry)
    _execute_wave(prepared, "stage1_feature_union", lambda: plan.feature_union(prepared.root),
                  "Combine only individually helpful EXPLORE feature groups", retry=retry)
    _execute_wave(prepared, "stage1_feature_finish", lambda: plan.feature_finish(prepared.root),
                  "One production check and fit-only feature compression", retry=retry)
    _execute_wave(prepared, "stage1_targets", lambda: [s for s in
                  _locked_specs(prepared, "stage1_target_plan", lambda: plan.targets(prepared.root),
                                "Freeze one feature/model parent for every F2 target") if s["tier"] == 1],
                  "Weekly and B5 residual target hypotheses", retry=retry)


def _stage2(prepared, retry):
    from phase_f.run import catalog
    from phase_f import experiment_plan as plan
    from phase_f.tuning import run_search

    _execute_wave(prepared, "stage2_tabular", lambda: plan.tabular_variants(prepared.root),
                  "GBDT losses, peak weights, multihorizon variants and Kalman features", retry=retry)
    for kind in ("lightgbm", "xgboost", "catboost"):
        run_search(prepared, kind, group="gbdt", minimum=500, batch_size=100)
    _execute_wave(prepared, "stage2_cross", lambda: plan.tabular_cross(prepared.root),
                  "Full declared feature by loss by weight cross", retry=retry)
    _execute_wave(prepared, "stage2_statistics", lambda: catalog(2),
                  "Kalman and seasonal state-space alternatives", retry=retry)
    _execute_wave(prepared, "stage2_neural", lambda: plan.neural_grid(prepared.root, 2),
                  "All tier-two neural architectures and context lengths", retry=retry)
    _execute_wave(prepared, "stage2_foundation", lambda: plan.foundation_fine(2),
                  "Fit-only Chronos-2 fine-tuning grid", retry=retry)
    _execute_wave(prepared, "stage2_data", lambda: [s for s in
                  _locked_specs(prepared, "stage2_data_plan", lambda: plan.data_strategies(prepared.root),
                                "Freeze one tabular parent for F8/F9 data strategies") if s["tier"] == 2],
                  "Complete-profile duplicate sensitivity", retry=retry)
    _execute_wave(prepared, "stage2_ensemble", lambda: plan.ensemble_specs(prepared.root, "stage2"),
                  "All declared pair combinations and fixed F7-6 core combinations", retry=retry)


def _stage3(prepared, retry):
    from phase_f.run import catalog
    from phase_f import experiment_plan as plan
    from phase_f.models.other_foundation import configurations as other_foundation
    from phase_f.tuning import run_search

    _execute_wave(prepared, "stage3_targets", lambda: [s for s in
                  _locked_specs(prepared, "stage1_target_plan", lambda: plan.targets(prepared.root),
                                "Freeze one feature/model parent for every F2 target") if s["tier"] == 3],
                  "Remaining target transforms and daytype separation", retry=retry)
    _execute_wave(prepared, "stage3_statistics", lambda: catalog(3),
                  "ETS, TBATS, ARIMA and Theta remaining comparators", retry=retry)
    _execute_wave(prepared, "stage3_neural", lambda: plan.neural_grid(prepared.root, 3),
                  "Remaining neural architectures and contexts including single LSTM/GRU", retry=retry)
    _execute_wave(prepared, "stage3_neural_extensions", lambda: plan.neural_extensions(prepared.root),
                  "Loss and causal feature ablations per completed architecture", retry=retry)
    from phase_f.experiment_plan import completed, spec
    kinds = sorted({spec(r)["kind"] for r in completed(prepared.root, adapter="neural")
                    if spec(r)["kind"] not in ("lstm", "gru")})
    for kind in kinds:
        run_search(prepared, kind, group="neural", minimum=20, batch_size=10)
    _execute_wave(prepared, "stage3_finetune_extensions", lambda: plan.foundation_fine(3),
                  "Fit-only fine-tuned foundation calendar and quantile ablations", retry=retry)
    _execute_wave(prepared, "stage3_other_foundation", other_foundation,
                  "Other foundation models on two declared history contexts", retry=retry)
    _execute_wave(prepared, "stage3_data", lambda: [s for s in
                  _locked_specs(prepared, "stage2_data_plan", lambda: plan.data_strategies(prepared.root),
                                "Freeze one tabular parent for F8/F9 data strategies") if s["tier"] == 3],
                  "Sliding, recent weighted and two-stage peak models", retry=retry)
    _execute_wave(prepared, "stage3_peak_postprocessing", lambda: plan.peak_postprocessors(prepared.root),
                  "Cal-only bias and quantile risk shift", retry=retry)
    _execute_wave(prepared, "stage3_ensemble_refresh", lambda: plan.ensemble_specs(prepared.root, "stage3"),
                  "Refresh cross-family combinations after all model families", retry=retry)
    _expansion(prepared, retry)


def _expansion_specs(prepared, number):
    from phase_f.experiment_plan import child, completed, foundation_training_expansion

    result = []
    parents = completed(prepared.root, adapter="neural")
    if parents:
        parent = json.loads(parents[0]["config_json"])
        result.append(child(parent, f"F5-expansion-{number}-{parent['kind']}", tier=3,
                            loss=("huber" if number % 2 == 0 else "peak_weighted_mae")))
    result.extend(foundation_training_expansion(prepared.root, number))
    if not result:
        raise RuntimeError("No completed neural or foundation parent for expansion ablation")
    return result


def _expansion(prepared, retry):
    from phase_f.tuning import eligible_best, meaningful, run_search

    path = prepared.out / "logs" / "workflow" / "expansion_rounds.json"
    state = _read(path) if path.exists() else {"rounds": [], "stopping_rule":
        "After all families, two consecutive finite waves without >=0.5% eligible EXPLORE MAE improvement"}
    while len(state["rounds"]) < 2 or any(r["meaningful_improvement"] for r in state["rounds"][-2:]):
        number = len(state["rounds"])
        round_path = prepared.out / "logs" / "workflow" / f"expansion_{number}.json"
        if round_path.exists():
            locked = _read(round_path)
        else:
            locked = {"round": number, "before": eligible_best(prepared.root),
                      "hypothesis": "Finite 100-trial GBDT plus neural and foundation training-budget ablations after all families",
                      "selection_arm": "EXPLORE", "holdout_read": False,
                      "historical_final_artifact_read": False}
            write_json(round_path, locked, exclusive=True)
        _execute_wave(prepared, f"expansion_{number}_ablations",
                      lambda n=number: _expansion_specs(prepared, n),
                      locked["hypothesis"], retry=retry)
        for kind in ("lightgbm", "xgboost", "catboost"):
            run_search(prepared, kind, group="gbdt", minimum=100, batch_size=100,
                       max_batches=1, label=f"expansion_{number}_{kind}")
        after = eligible_best(prepared.root)
        round_record = {**locked, "after": after,
                        "meaningful_improvement": bool(meaningful(locked["before"], after))}
        if round_path.exists():
            write_json(round_path, round_record)
        state["rounds"].append(round_record)
        write_json(path, state)
    return state["rounds"]


def _search_manifest(prepared):
    from phase_f.selection import complete_metric_manifests
    from phase_f.selection import foundation_training_coverage
    from phase_f.selection import verify_search_complete

    path = prepared.out / "logs" / "search_complete.json"
    if path.exists():
        verify_search_complete(prepared)
        return _read(path)

    complete_metric_manifests(prepared)
    registry = Registry(prepared.root)
    rows = [_read(p) for p in sorted(registry.records.glob("*.json"))]
    by_id = {row["exp_id"]: row for row in rows}
    unresolved = [r["exp_id"] for r in rows if r["status"] not in TERMINAL]
    if unresolved:
        raise RuntimeError(f"Unresolved experiments before selection: {unresolved[:5]}")
    waves = {p.relative_to(prepared.out).as_posix(): sha256(p)
             for p in sorted((prepared.out / "logs" / "waves").glob("*.json"))}
    tuning = {p.relative_to(prepared.out).as_posix(): sha256(p)
              for p in sorted((prepared.out / "logs" / "tuning").glob("*.json"))}
    if not waves or not tuning:
        raise RuntimeError("Required wave and TPE state evidence is absent")
    locked_ids = set()
    for relative in waves:
        wave = _read(prepared.out / relative)
        if wave.get("config_sha256") != config_hash(wave.get("specs")):
            raise ValueError(f"Changed wave specifications: {relative}")
        for item in wave["specs"]:
            locked_ids.add(item["id"])
            row = by_id.get(item["id"])
            if row is None or row["config_hash"] != config_hash(item) or row["status"] not in TERMINAL:
                raise RuntimeError(f"Unresolved locked wave experiment: {item['id']}")
    for relative in tuning:
        state = _read(prepared.out / relative)
        if {"kind", "group", "minimum"} <= set(state) and state.get("complete") is not True:
            raise RuntimeError(f"Incomplete TPE search: {relative}")
    expansion = _read(prepared.out / "logs" / "workflow" / "expansion_rounds.json")["rounds"]
    fine_coverage = foundation_training_coverage(prepared)
    if any(item["exp_id"] not in locked_ids for evidence in fine_coverage.values()
           for item in evidence["experiments"]):
        raise ValueError("F6 training evidence is outside the locked waves")
    coverage = {}
    for family in FAMILIES[:-1]:
        matched = [r for r in rows if r["exp_id"].split("-")[0] == family or r["family"] == family]
        done = [r for r in matched if r["status"] in ("completed", "diagnostic_passed")]
        if not matched:
            raise RuntimeError(f"No {family} experiment was attempted")
        coverage[family] = {"status": "completed" if done else "failed_with_evidence",
                            "resolved": True, "attempted": len(matched), "completed": len(done),
                            "failed": sum(r["status"] != "completed" and r["status"] != "diagnostic_passed" for r in matched)}
    record = {"version": 1, "created_at": now(), "waves": waves, "tuningfiles": tuning,
              "experiment_ids": [r["exp_id"] for r in rows], "required_family_coverage": coverage,
              "foundation_training_coverage": fine_coverage,
              "expansion_rounds": expansion, "all_waves_complete": True,
              "historical_final_artifact_read": False, "holdout_read": False}
    write_json(path, record, exclusive=True)
    return record


def _downstream(prepared, selection_lock, confirm):
    from phase_f.downstream import evaluate_downstream
    from phase_f.run import base_frame

    path = prepared.out / "logs" / "downstream_status.json"
    if path.exists():
        previous = _read(path)
        if previous.get("selection_lock_sha256") != selection_lock["lock_sha256"] or previous.get("confirm_once_sha256") != confirm["confirm_once_sha256"]:
            raise ValueError("F11 status differs from fixed selection/CONFIRM")
        artifacts = previous.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("F11 artifact inventory is missing")
        if previous.get("status") == "unavailable":
            if selection_lock["candidates"] or artifacts or not previous.get("reason"):
                raise ValueError("F11 unavailable status conflicts with selected candidates")
        elif previous.get("status") == "completed":
            required = {f"tables/downstream/{name}.csv" for name in
                        ("risk_metrics", "uncertainty_metrics", "alert_episode_metrics",
                         "comparison_vs_B5", "decision_value_curve")}
            required.add("predictions/F11_score_probabilities.parquet")
            if not required <= set(artifacts):
                raise ValueError("F11 required artifact inventory is incomplete")
        else:
            raise ValueError("F11 status is unresolved")
        for name, digest in artifacts.items():
            allowed = (name.startswith("tables/downstream/") and name.endswith(".csv") and
                       not any(c in name[len("tables/downstream/"):] for c in ("/", "\\"))) or name == "predictions/F11_score_probabilities.parquet"
            if not allowed:
                raise ValueError("F11 artifact path outside expected output tables")
            target = (prepared.out / name).resolve()
            if not target.is_relative_to(prepared.out.resolve()) or sha256(target) != digest:
                raise ValueError(f"Changed F11 artifact: {name}")
        return previous
    selected = selection_lock["candidates"][:2]
    if not selected:
        record = {"status": "unavailable", "reason": "No eligible EXPLORE primary was selected",
                  "artifacts": {}}
    else:
        b5_path = prepared.out / "cache" / "parent_B5_cal_score.parquet"
        if not b5_path.is_file():
            raise FileNotFoundError("Phase F B5 cal replay is required for F11")
        b5 = pd.read_parquet(b5_path)
        b5["cal_provenance"] = np.where(b5.role.eq("cal"), "fit_only", "not_cal")
        # Reconstruct Phase E's genuine fit-only Kalman variance on the Phase F
        # h16 cohort. The point mean must reproduce this exact B5 replay.
        import joblib
        from outputs.phase_e.code.distribution import predict_kalman_distribution
        for fold in (0, 1, 2):
            artifact = joblib.load(prepared.root / "outputs" / "phase_c" / "models" /
                                   f"B5_h16_f{fold}.joblib")
            bundle = artifact["bundle"]
            if bundle.get("fit_parameters_only") is not True or bundle.get("development_only") is not True:
                raise ValueError("B5 h16 distribution lacks fit-only development provenance")
            for role in ("cal", "score"):
                part = b5.loc[b5.fold.eq(fold) & b5.role.eq(role)].sort_values("origin")
                mean, sigma = predict_kalman_distribution(bundle, prepared.history.power,
                                                           pd.DatetimeIndex(part.origin), 16)
                if not np.allclose(mean, part.pred.to_numpy(float), atol=1e-10, rtol=0):
                    raise ValueError("B5 Phase E distribution mean differs from Phase F replay")
                b5.loc[part.index, "sigma"] = sigma
        frames = [b5]
        for candidate in selected:
            frames.append(base_frame(prepared, candidate))
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.loc[frame.horizon.eq(16)].copy()
        prevalence = {}
        for model in ["B5", *selected]:
            prevalence[model] = {}
            for fold in (0, 1, 2):
                c = prepared.contexts[(16, fold)]
                prevalence[model][str(fold)] = float((c["y"].loc[c["fit"]] > c["tau"]).mean())
        source = {"B5": "gaussian"}
        for model in selected:
            part = frame.loc[frame.model.eq(model)]
            source[model] = "native" if {"p_peak", "q95"} <= set(part) and part[["p_peak", "q95"]].notna().all().all() else "residual"
        f11_lock = {"confirm_complete": True, "selection_lock_sha256": selection_lock["lock_sha256"],
                    "confirm_once_sha256": confirm["confirm_once_sha256"],
                    "primary_candidate": selection_lock["primary_candidate"], "candidates": selected,
                    "fit_peak_prevalence": prevalence, "probability_source": source,
                    "ensemble_models": [name for name in selected if json.loads(Registry(prepared.root).read(name)["config_json"]).get("adapter") in ("ensemble", "seed_ensemble")]}
        results = evaluate_downstream(frame.loc[frame.role.eq("cal")],
                                      frame.loc[frame.role.eq("score")], f11_lock)
        artifacts = {}
        for name, table in results.items():
            dest = (prepared.out / "predictions" / "F11_score_probabilities.parquet" if name == "score_probabilities"
                    else prepared.out / "tables" / "downstream" / f"{name}.csv")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == "score_probabilities":
                table.to_parquet(dest, index=False)
            else:
                table.to_csv(dest, index=False)
            artifacts[dest.relative_to(prepared.out).as_posix()] = sha256(dest)
        record = {"status": "completed", "reason": "", "artifacts": artifacts,
                  "models": ["B5", *selected], "probability_source": source}
    record.update(selection_lock_sha256=selection_lock["lock_sha256"],
                  confirm_once_sha256=confirm["confirm_once_sha256"],
                  historical_final_artifact_read=False, holdout_read=False)
    write_json(path, record, exclusive=True)
    return record


def _weekly_fixed(prepared, selection_lock, confirm, candidate):
    row = Registry(prepared.root).read(candidate)
    if row is None or row["status"] != "completed" or row["config_hash"] != selection_lock.get("weekly_diagnostic_config_hash"):
        raise ValueError("Weekly diagnostic model differs from its frozen registry configuration")
    spec = json.loads(row["config_json"])
    frozen = {**spec, "horizon": 16}
    origins = {fold: pd.DatetimeIndex(prepared.origins(16, fold, "score")).sort_values() for fold in (0, 1, 2)}
    fixed = {"candidate": candidate, "frozen_config_hash": config_hash(frozen),
             "score_origin_hashes": {str(f): config_hash([str(t) for t in origins[f]]) for f in origins},
             "selection_lock_sha256": selection_lock["lock_sha256"],
             "confirm_once_sha256": confirm["confirm_once_sha256"], "horizon": 16,
             "historical_final_artifact_read": False, "holdout_read": False}
    return spec, frozen, origins, fixed


def _weekly(prepared, selection_lock, confirm):
    from phase_f.walkforward import simulate_weekly_refit

    path = prepared.out / "logs" / "weekly_diagnostic_status.json"
    candidate = selection_lock.get("weekly_diagnostic_candidate")
    lock_path = prepared.out / "logs" / "weekly_diagnostic_lock.json"
    if path.exists():
        record = _read(path)
        if (record.get("selection_lock_sha256") != selection_lock["lock_sha256"] or
                record.get("confirm_once_sha256") != confirm["confirm_once_sha256"]):
            raise ValueError("Changed weekly diagnostic selection or CONFIRM lock")
        if record.get("status") == "unavailable":
            if candidate is not None or not record.get("reason"):
                raise ValueError("Weekly diagnostic unavailable status conflicts with fixed candidate")
            return record
        if record.get("status") != "completed" or record.get("candidate") != candidate:
            raise ValueError("Weekly diagnostic status conflicts with fixed candidate")
        _, _, _, fixed = _weekly_fixed(prepared, selection_lock, confirm, candidate)
        if not lock_path.is_file() or _read(lock_path) != fixed:
            raise ValueError("Weekly diagnostic lock changed")
        expected = {f"predictions/F9-4-fold{f}-predictions.parquet" for f in (0, 1, 2)} | {
            f"tables/F9-4-fold{f}-updates.csv" for f in (0, 1, 2)}
        if set(record.get("artifacts", {})) != expected:
            raise ValueError("Weekly diagnostic artifact inventory changed")
        for name, digest in record["artifacts"].items():
            if sha256(prepared.out / name) != digest:
                raise ValueError(f"Weekly diagnostic artifact changed: {name}")
        return record
    if candidate is None:
        status = {"status": "unavailable", "reason": "No supported frozen tabular candidate"}
    else:
        spec, frozen, origins, fixed = _weekly_fixed(prepared, selection_lock, confirm, candidate)
        write_json(lock_path, fixed, exclusive=True)
        artifacts = {}
        for fold in (0, 1, 2):
            lock = {"confirm_complete": True, "confirm_once_sha256": confirm["confirm_once_sha256"],
                    "selection_lock_sha256": selection_lock["lock_sha256"],
                    "weekly_diagnostic_candidate": candidate, "frozen_config_hash": fixed["frozen_config_hash"],
                    "score_origin_sha256": fixed["score_origin_hashes"][str(fold)]}
            result = simulate_weekly_refit(prepared.history, prepared.contexts[(16, fold)], origins[fold],
                                           candidate=candidate, kind=spec["kind"], frozen_config=frozen, lock=lock)
            for name, table in result.items():
                dest = prepared.out / ("predictions" if name == "predictions" else "tables") / f"F9-4-fold{fold}-{name}.{ 'parquet' if name == 'predictions' else 'csv'}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if name == "predictions": table.to_parquet(dest, index=False)
                else: table.to_csv(dest, index=False)
                artifacts[dest.relative_to(prepared.out).as_posix()] = sha256(dest)
        status = {"status": "completed", "candidate": candidate, "artifacts": artifacts,
                  "reason": "Post-confirm diagnostic only; does not reselect primary"}
    status.update(selection_lock_sha256=selection_lock["lock_sha256"],
                  confirm_once_sha256=confirm["confirm_once_sha256"],
                  historical_final_artifact_read=False, holdout_read=False)
    write_json(path, status, exclusive=True)
    return status


def _stage4(prepared):
    from phase_f.selection import lock_selection, confirm_once
    from phase_f.reporting import generate_final, generate_progress

    _search_manifest(prepared)
    lock = lock_selection(prepared, all_waves_complete=True, expansion_converged=True)
    # The immutable selection and search evidence must be in HEAD before the
    # one-time reused-development CONFIRM is opened.
    reservation = prepared.out / "logs" / "confirm_reserved.json"
    complete = prepared.out / "logs" / "confirm_complete.json"
    if reservation.exists() or complete.exists():
        _verify_committed_selection(prepared, lock)
    else:
        _git_commit(prepared, "Phase F: freeze completed EXPLORE search and candidate selection")
        _verify_committed_selection(prepared, lock)
    done = confirm_once(prepared)
    downstream = _downstream(prepared, lock, done)
    weekly = _weekly(prepared, lock, done)
    generate_progress(prepared.root, selected_id=lock["primary_candidate"])
    coverage = _read(prepared.out / "logs" / "search_complete.json")["required_family_coverage"]
    coverage = {family: ({**state, "status": "failed_explicit"} if state["status"] == "failed_with_evidence" else state)
                for family, state in coverage.items()}
    coverage["F11"] = {"status": "completed" if downstream["status"] == "completed" else "unsupported_explicit",
                       "resolved": True, "reason": downstream.get("reason", "")}
    gate = {"ready_for_report": True, "full_phase_complete": False,
            "required_family_coverage": coverage,
            "downstream_resolved": downstream["status"] in ("completed", "unavailable"),
            "expansion_converged": True, "confirm_once_sha256": done["confirm_once_sha256"],
            "weekly_diagnostic": weekly["status"],
            "weekly_diagnostic_status_sha256": sha256(prepared.out / "logs" / "weekly_diagnostic_status.json"),
            "weekly_diagnostic_lock_sha256": (sha256(prepared.out / "logs" / "weekly_diagnostic_lock.json")
                 if (prepared.out / "logs" / "weekly_diagnostic_lock.json").is_file() else None),
            "historical_final_artifact_read": False, "holdout_read": False}
    gate_path = prepared.out / "logs" / "completion_gate.json"
    if gate_path.exists():
        previous = _read(gate_path)
        stable = lambda value: {key: item for key, item in value.items()
                                if key not in ("ready_for_report", "full_phase_complete")}
        if stable(previous) != stable(gate):
            raise ValueError("Resumed completion gate differs from the same frozen evidence")
    write_json(gate_path, gate)
    result = generate_final(prepared.root, provisional=True)
    if result.get("report_ready") is not True or result.get("full_phase_complete") is not False:
        raise RuntimeError(f"Provisional report gate unresolved: {result.get('reason')}")
    return result


def run_stage(prepared, stage: int, *, retry: bool = False):
    """Execute one dependency-ordered stage; no concurrent GPU/GBDT jobs."""
    from phase_f.harness import seal_parent
    from phase_f.selection import complete_metric_manifests

    if stage not in (0, 1, 2, 3, 4):
        raise ValueError("Phase F stage must be 0..4")
    for prior in range(stage):
        path = _stage_path(prepared, prior)
        if not path.exists() or _read(path).get("status") != "completed":
            raise RuntimeError(f"Stage {prior} must finish before stage {stage}")
    existing = _stage_path(prepared, stage)
    if existing.exists() and _read(existing).get("status") == "completed":
        seal_parent(prepared.root)
        if stage == 4 and not (prepared.out / "PHASE_F_REPORT.md").is_file():
            raise RuntimeError("Stage 4 checkpoint exists without its final report")
        return _read(existing)
    _status(prepared, stage, status="running", full_phase_complete=False)
    try:
        if stage == 0: _stage0(prepared, retry)
        elif stage == 1: _stage1(prepared, retry)
        elif stage == 2: _stage2(prepared, retry)
        elif stage == 3: _stage3(prepared, retry)
        else: result = _stage4(prepared)
        complete_metric_manifests(prepared)
        if stage < 4:
            _progress(prepared, stage)
            _git_commit(prepared, f"Phase F: complete Stage {stage} EXPLORE evidence")
        seal_parent(prepared.root)
        if stage == 4:
            from phase_f.reporting import generate_final
            _git_commit(prepared, "Phase F: commit verified provisional report and CONFIRM evidence")
            gate_path = prepared.out / "logs" / "completion_gate.json"
            gate = _read(gate_path)
            if gate.get("ready_for_report") is not True or gate.get("full_phase_complete") is not False:
                raise ValueError("Provisional completion gate changed before final report")
            write_json(gate_path, {**gate, "full_phase_complete": True})
            result = generate_final(prepared.root)
            if result.get("full_phase_complete") is not True:
                raise RuntimeError(f"Final report gate unresolved: {result.get('reason')}")
            _git_commit(prepared, "Phase F: commit final development report and completion gate")
        record = _status(prepared, stage, status="completed", full_phase_complete=stage == 4,
                         result=result if stage == 4 else None)
        if stage == 4:
            write_json(prepared.out / "logs" / "driver_status.json",
                       {"status": "completed", "stage": "all", "at": now(), "full_phase_complete": True,
                        "historical_final_artifact_read": False, "holdout_read": False})
            _git_commit(prepared, "Phase F: record durable completed workflow status")
        return record
    except Exception as exc:
        if stage == 4:
            gate_path = prepared.out / "logs" / "completion_gate.json"
            if gate_path.exists():
                gate = _read(gate_path)
                write_json(gate_path, {**gate, "full_phase_complete": False})
        _status(prepared, stage, status="failed", full_phase_complete=False,
                error_type=type(exc).__name__, error=str(exc).replace(str(prepared.root), "<repo>"))
        write_json(prepared.out / "logs" / "driver_status.json",
                   {"status": "failed", "stage": stage, "at": now(), "full_phase_complete": False,
                    "error_type": type(exc).__name__, "error": str(exc).replace(str(prepared.root), "<repo>"),
                    "historical_final_artifact_read": False, "holdout_read": False})
        raise


def run_all(prepared, *, retry: bool = False):
    """Run every approved wave through reporting; resume locked checkpoints."""
    if not (prepared.out / "logs" / "user_approval.json").is_file():
        raise RuntimeError("Approved Phase F contract is required")
    for stage in range(5):
        run_stage(prepared, stage, retry=retry)
    return _read(_stage_path(prepared, 4))
