"""Durable walk-forward TPE searches using only fitted stop-label error.

The orchestrator supplies a locked parent and an execution callback.  This
module never reads EXPLORE scores, CONFIRM rows, or Phase C's old registry.
Every GBDT family must finish at least 500 full weekly five-seed trials.
Additional calls can extend that same study in exact 100-completion rounds.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import joblib
import optuna
import pandas as pd

from phase_f.models.regression import suggest_parameters
from phase_f.registry import config_hash, sha256, write_json
from phase_f.wf_models import _arm_view

KINDS = ("lightgbm", "xgboost", "catboost")
NEURAL_KINDS = ("dlinear", "nlinear", "tcn", "nhits", "nbeats", "patchtst",
                "tide", "tsmixer", "timesnet", "itransformer", "xlstm")
MIN_COMPLETED = 500
ROUND_COMPLETED = 100
NEURAL_MIN_COMPLETED = 20
_SAFE = re.compile(r"[A-Za-z0-9_.+-]+\Z")


def _scope(prepared, kind, parent_spec, label, group="gbdt"):
    allowed = KINDS if group == "gbdt" else NEURAL_KINDS
    if kind not in allowed:
        raise ValueError(f"Unsupported WF TPE kind: {kind}")
    adapter = "regression" if group == "gbdt" else "neural"
    if not isinstance(parent_spec, dict) or parent_spec.get("adapter") != adapter:
        raise ValueError(f"TPE needs a locked {adapter} parent specification")
    if parent_spec.get("kind") not in (None, kind):
        raise ValueError("TPE parent kind differs from requested family")
    if not _SAFE.fullmatch(str(parent_spec.get("id", ""))):
        raise ValueError("TPE parent needs a safe ID")
    name = label or f"wf_tpe_{kind}"
    if not _SAFE.fullmatch(name):
        raise ValueError("Unsafe TPE study label")
    view = _arm_view(prepared, "EXPLORE")
    if any(not len(view.origins(h, f, "stop")) for h, f in view.contexts):
        raise ValueError("Every EXPLORE weekly cell needs a stop window")
    source_paths = ["phase_f/wf_tuning.py", "phase_f/wf_models.py",
                    f"phase_f/models/{adapter}.py"]
    if group == "gbdt":
        source_paths.append("phase_f/features_ext.py")
    source = {path: sha256(Path(prepared.root) / path) for path in source_paths}
    identity = config_hash({"name": name, "kind": kind, "parent_spec": parent_spec,
                            "group": group,
                            "source": source, "split_lock": view.split_lock["lock_sha256"],
                            "raw_sha256": view.seal["raw_sha256"],
                            "active_cells": sorted((h, str(f)) for h, f in view.contexts),
                            "n_seeds": 5})
    return name, identity, len(view.contexts) * 5, source


def _study(prepared, name, identity, sampler_path):
    folder = Path(prepared.out) / "logs" / "tuning"
    folder.mkdir(parents=True, exist_ok=True)
    database = folder / f"{name}.sqlite"
    if sampler_path.exists():
        sampler = joblib.load(sampler_path)
        if not isinstance(sampler, optuna.samplers.TPESampler):
            raise RuntimeError("Saved sampler is not a TPESampler")
    else:
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=25)
    storage = "sqlite:///" + database.resolve().as_posix()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=name, storage=storage, sampler=sampler,
                                direction="minimize", load_if_exists=True)
    prior = study.user_attrs.get("wf_identity")
    if prior is not None and prior != identity:
        raise RuntimeError("TPE study source/parent/split identity changed; use a new label")
    if prior is None:
        if study.trials:
            raise RuntimeError("Unidentified nonempty TPE study cannot be adopted")
        study.set_user_attr("wf_identity", identity)
        study.set_user_attr("objective", "mean original-unit stop MAE across EXPLORE weekly cells and five seeds")
    return study, database


def _valid_completed(study, expected_cells, *, distinct=False):
    result = []
    seen = set()
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        attrs = trial.user_attrs
        if (attrs.get("status") != "prediction_ready" or attrs.get("stop_cells") != expected_cells
                or attrs.get("n_seeds") != 5 or not math.isfinite(float(trial.value))
                or not _digest(attrs.get("prediction_sha"))
                or attrs.get("trial_role") != "stop_only_completed"):
            raise RuntimeError(f"Completed TPE trial {trial.number} lacks full stop/seed evidence")
        if distinct:
            param_hash = attrs.get("param_hash")
            if not _digest(param_hash) or param_hash in seen:
                raise RuntimeError("Neural TPE completed settings are missing or duplicated")
            seen.add(param_hash)
        result.append(trial)
    return result


def _digest(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _trial_spec(parent_spec, kind, trial, name, *, group="gbdt"):
    if group == "gbdt":
        params = suggest_parameters(trial, kind)
    else:
        from phase_f.models.neural import suggest_parameters as neural_parameters
        params = neural_parameters(trial, kind)
    spec = {**parent_spec,
            "id": f"WF-{name}-t{trial.number:05d}",
            "adapter": "regression" if group == "gbdt" else "neural", "kind": kind,
            "n_seeds": 5,
            "arm": "EXPLORE", "tpe_study": name, "tpe_trial": trial.number,
            "tpe_objective": "stop_MAE_only"}
    if group == "gbdt":
        spec.update(model_params=params, force_seed_repeats=True)
        if "params" in spec:
            spec["params"] = {**spec["params"], "model_params": params,
                              "force_seed_repeats": True}
    else:
        spec.update(params)
        if "params" in spec:
            spec["params"] = {**spec["params"], **params}
            spec["params"].pop("seeds", None)
    return spec


def _trial_rows(study):
    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        rows.append({"trial_number": trial.number, "exp_id": attrs.get("exp_id"),
                     "state": trial.state.name, "params_json": json.dumps(trial.params, sort_keys=True),
                     "stop_MAE": float(trial.value) if trial.value is not None else None,
                     "stop_cells": attrs.get("stop_cells"), "n_seeds": attrs.get("n_seeds"),
                     "prediction_sha": attrs.get("prediction_sha"),
                     "param_hash": attrs.get("param_hash"),
                     "trial_role": attrs.get("trial_role"),
                     "error_type": attrs.get("error_type"),
                     "objective_source": "fit/stop only"})
    return pd.DataFrame(rows)


def _persist(study, folder, name, state, sampler_path):
    # SQLite is the authoritative trial ledger. This CSV is a derived,
    # replaceable index for inspection and orchestration.
    rows = _trial_rows(study)
    registry = folder / f"{name}_trials.csv"
    temp = registry.with_suffix(".tmp")
    rows.to_csv(temp, index=False)
    temp.replace(registry)
    state["trial_registry_sha256"] = sha256(registry)
    sampler_tmp = sampler_path.with_suffix(".tmp")
    joblib.dump(study.sampler, sampler_tmp)
    sampler_tmp.replace(sampler_path)
    write_json(folder / f"{name}.json", state)


def _objective(row, expected_cells):
    if not isinstance(row, dict):
        raise ValueError("TPE callback must return one registry row dictionary")
    value = row.get("stop_MAE")
    digest = row.get("prediction_sha") or row.get("prediction_sha256")
    if (row.get("status") != "prediction_ready" or row.get("stop_cells") != expected_cells
            or row.get("n_seeds") != 5 or value is None or not math.isfinite(float(value))
            or not _digest(digest)):
        raise ValueError("TPE trial lacks prediction_ready status, 5 seeds, all stop cells, finite stop MAE, or prediction hash")
    return float(value), digest


def _annotate_registry(root, spec, row, name, number, value, digest):
    """Mark a fitted trial's role without promoting it to scored completion."""
    if not row.get("exp_id"):
        return  # Minimal synthetic callback used by module tests.
    if row["exp_id"] != spec["id"]:
        raise RuntimeError("TPE callback returned a different experiment ID")
    from phase_f.wf_registry import WFRegistry

    registry = WFRegistry(root)
    stored = registry.read(spec["id"])
    if stored is None or stored.get("status") != "prediction_ready":
        raise RuntimeError("TPE registry has no prediction_ready trial to annotate")
    current_hash = stored.get("prediction_sha") or stored.get("prediction_sha256")
    if current_hash != digest:
        raise RuntimeError("TPE registry prediction hash changed before annotation")
    registry.update(spec["id"], trial_role="stop_only_completed",
                    tpe_stop_completed=True, tuning_study=name,
                    tuning_trial=number, tuning_stop_MAE=value,
                    tuning_stop_cells=stored["stop_cells"])


def _run_search(prepared, kind, execute_trial, *, parent_spec,
                target_completed, label=None, group="gbdt"):
    """Shared implementation for stop-only GBDT and neural study ledgers.

    `execute_trial(spec, score=False)` must return a registry row after fitting
    and predicting, before any score summary. A crash with an Optuna RUNNING
    trial resumes that same numbered trial. Failed trials do not count toward
    500, and three consecutive failures pause for investigation.
    """
    target_completed = int(target_completed)
    minimum = MIN_COMPLETED if group == "gbdt" else NEURAL_MIN_COMPLETED
    if target_completed < minimum or (target_completed - minimum) % ROUND_COMPLETED:
        raise ValueError(f"WF TPE target must be {minimum} or {minimum} plus whole 100-trial rounds")
    name, identity, expected_cells, source = _scope(prepared, kind, parent_spec, label, group=group)
    folder = Path(prepared.out) / "logs" / "tuning"
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / f"{name}.json"
    sampler_path = folder / f"{name}_sampler.joblib"
    study, database = _study(prepared, name, identity, sampler_path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity or state.get("kind") != kind or state.get("group") != group:
            raise RuntimeError("Saved WF TPE state conflicts with parent/source/split lock")
        if int(state["target_completed"]) > target_completed:
            raise ValueError("TPE completion target cannot decrease")
    else:
        state = {"name": name, "kind": kind, "group": group, "identity": identity,
                 "parent_spec": parent_spec, "source_sha256": source,
                 "objective": "mean original-unit stop MAE across all EXPLORE weekly horizon/fold/seed cells",
                 "expected_stop_cells": expected_cells, "n_seeds": 5,
                 "completed_trials": 0, "target_completed": target_completed,
                 "consecutive_failures": 0, "status": "running",
                 "historical_final_artifact_read": False, "holdout_read": False}
    state["target_completed"] = target_completed
    state["status"] = "running"
    write_json(state_path, state)
    while True:
        completed = _valid_completed(study, expected_cells, distinct=group == "neural")
        state["completed_trials"] = len(completed)
        if len(completed) >= target_completed:
            best = min(completed, key=lambda trial: (float(trial.value), trial.number))
            state.update(status="complete", best_trial_number=best.number,
                         best_stop_MAE=float(best.value), best_params=best.params,
                         best_exp_id=best.user_attrs["exp_id"],
                         best_spec=best.user_attrs["spec"],
                         sqlite_path=str(database),
                         trial_registry_path=str(folder / f"{name}_trials.csv"))
            _persist(study, folder, name, state, sampler_path)
            return state
        running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
        if len(running) > 1:
            raise RuntimeError("Multiple RUNNING TPE trials; sequential ownership was lost")
        trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
        spec = _trial_spec(parent_spec, kind, trial, name, group=group)
        trial.set_user_attr("exp_id", spec["id"])
        trial.set_user_attr("spec", spec)
        param_hash = config_hash(trial.params)
        if group == "neural":
            prior = {t.user_attrs.get("param_hash") for t in study.trials
                     if t.number != trial.number and t.state == optuna.trial.TrialState.COMPLETE}
            if param_hash in prior:
                trial.set_user_attr("error_type", "DuplicateConfiguration")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                state["completed_trials"] = len(_valid_completed(study, expected_cells, distinct=True))
                _persist(study, folder, name, state, sampler_path)
                continue
            trial.set_user_attr("param_hash", param_hash)
        try:
            row = execute_trial(spec, score=False)
            value, digest = _objective(row, expected_cells)
            trial.set_user_attr("status", "prediction_ready")
            trial.set_user_attr("trial_role", "stop_only_completed")
            trial.set_user_attr("stop_cells", expected_cells)
            trial.set_user_attr("n_seeds", 5)
            trial.set_user_attr("prediction_sha", digest)
            _annotate_registry(prepared.root, spec, row, name, trial.number, value, digest)
            study.tell(trial, value)
            state["consecutive_failures"] = 0
        except Exception as exc:
            trial.set_user_attr("error_type", type(exc).__name__)
            trial.set_user_attr("error", str(exc)[:500])
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        state["completed_trials"] = len(_valid_completed(study, expected_cells,
                                                         distinct=group == "neural"))
        state["status"] = "running"
        # A prior extension may have published a 500-trial best. Keep its
        # identity as historical evidence; it is never used by the sampler.
        _persist(study, folder, name, state, sampler_path)
        if state["consecutive_failures"] >= 3:
            raise RuntimeError(f"Three consecutive {kind} WF TPE failures; inspect {name}_trials.csv before resuming")


def run_search(prepared, kind, execute_trial, *, parent_spec,
               target_completed=MIN_COMPLETED, label=None):
    """Finish 500 genuine GBDT trials, then extend by 100 completed trials."""
    return _run_search(prepared, kind, execute_trial, parent_spec=parent_spec,
                       target_completed=target_completed, label=label, group="gbdt")


def run_neural_search(prepared, kind, execute_trial, *, parent_spec,
                      target_completed=NEURAL_MIN_COMPLETED, label=None):
    """Finish 20 distinct neural settings, then extend in 100-trial rounds.

    LSTM/GRU are fixed reference architectures under the revised protocol.
    Their recorded base grid remains available through the normal workflow.
    """
    return _run_search(prepared, kind, execute_trial, parent_spec=parent_spec,
                       target_completed=target_completed, label=label, group="neural")
