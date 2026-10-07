"""EXPLORE-only, causal three-regime change expert for frozen R1 paths.

Each weekly model uses FIT for preprocessing and training, STOP for early
stopping and a fixed mixture weight, and CAL/SCORE only for prediction.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from phase_f.features_ext import _power, build_features
from phase_f.harness import make_frame
from phase_f.models.regression import _features, _preprocessor, _transform
from phase_f.models.r1_residual import (
    HORIZONS, ROLES, _common_roles, _digest_frame, _known_error,
    _path_values, _paths_index,
)
from phase_f.registry import config_hash, sha256, write_json


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
THRESHOLD = 30.0
MIN_CLASS = 20
_ID = re.compile(r"[A-Za-z0-9_.+-]+\Z")
_SOURCES = (
    "phase_f/models/transition_expert.py", "phase_f/models/r1_residual.py",
    "phase_f/models/regression.py", "phase_f/features_ext.py",
    "phase_f/harness.py", "phase_c/data.py", "src/session_data.py",
    "src/features.py", "src/targets.py", "src/holidays.py",
)
PARAMS = {"n_estimators": 900, "learning_rate": 0.035,
          "num_leaves": 31, "min_child_samples": 100,
          "colsample_bytree": 0.85, "subsample": 0.85,
          "subsample_freq": 1, "reg_lambda": 5.0, "n_jobs": 4,
          "verbosity": -1}


def _origin_power(history: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
    return _power(history).reindex(origins).to_numpy(float)


def _production_availability(history: pd.DataFrame, origins: pd.DatetimeIndex,
                             features: pd.DataFrame) -> dict:
    if "production_last_completed_hour" not in features:
        return {"production_group_used": False}
    completed = pd.to_numeric(history["production_completed"], errors="coerce")
    if "production_completed_bad" in history:
        completed = completed.mask(history.production_completed_bad.fillna(True).astype(bool))
    last = features.production_last_completed_hour.to_numpy(float)
    same = features.production_same_slot_7d_mean.to_numpy(float)
    direct = np.column_stack([
        completed.reindex((origins - i * pd.Timedelta(days=1)).floor("h")).to_numpy(float)
        for i in range(1, 8)
    ])
    stale = np.isfinite(same) & ~np.isfinite(direct).all(axis=1)
    return {"production_group_used": True, "rows": len(origins),
            "last_completed_hour_missing_or_bad": int((~np.isfinite(last)).sum()),
            "same_slot_7d_missing": int((~np.isfinite(same)).sum()),
            "same_slot_7d_older_hour_substitution": int(stale.sum()),
            "same_slot_7d_expected_prior_hour_missing_or_bad": int((~np.isfinite(direct)).any(axis=1).sum()),
            "zero_posting_latency_assumed": True,
            "field_posting_latency": "unknown"}


def _matrix(history: pd.DataFrame, paths: pd.DataFrame,
            origins: pd.DatetimeIndex, h: int, tau: float,
            groups: tuple[str, ...]) -> pd.DataFrame:
    x = _features(history, origins, h, tau, groups)
    p = _path_values(paths, origins, h)
    power = _power(history)
    now = power.reindex(origins).to_numpy(float)
    ramp4 = now - power.reindex(origins - pd.Timedelta(hours=1)).to_numpy(float)
    ramp16 = now - power.reindex(origins - pd.Timedelta(hours=4)).to_numpy(float)
    extra = pd.DataFrame({
        "r1_point": p.r1.to_numpy(float),
        "r1_q10": p.q10.to_numpy(float),
        "r1_q90": p.q90.to_numpy(float),
        "r1_q95": p.q95.to_numpy(float),
        "r1_width_80": (p.q90 - p.q10).to_numpy(float),
        "horizon_quarters": float(h),
        "power_origin": now,
        "ramp_1h_signed": ramp4,
        "ramp_4h_signed": ramp16,
        "ramp_1h_rise": np.maximum(0, ramp4),
        "ramp_1h_fall": np.maximum(0, -ramp4),
        "ramp_4h_rise": np.maximum(0, ramp16),
        "ramp_4h_fall": np.maximum(0, -ramp16),
        "last_matured_r1_error": _known_error(history, paths, origins, h),
    }, index=origins)
    extra["last_matured_r1_abs_error"] = np.abs(extra.last_matured_r1_error)
    out = pd.concat([x, extra], axis=1)
    if out.columns.has_duplicates:
        raise ValueError("Duplicate transition expert features")
    return out


def _classes(delta: np.ndarray) -> np.ndarray:
    if not np.isfinite(delta).all():
        raise ValueError("Transition target contains nonfinite values")
    return np.where(delta < -THRESHOLD, 0,
                    np.where(delta > THRESHOLD, 2, 1)).astype(np.int8)


def _supports(delta: np.ndarray) -> np.ndarray:
    value = np.asarray(delta, float).copy()
    value[:, 0] = np.minimum(value[:, 0], -THRESHOLD)
    value[:, 1] = np.clip(value[:, 1], -THRESHOLD, THRESHOLD)
    value[:, 2] = np.maximum(value[:, 2], THRESHOLD)
    return value


def _predict_expert(bundle: dict, transformed: np.ndarray,
                    origin_power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(bundle["classifier"].predict_proba(transformed), float)
    if probs.shape != (len(origin_power), 3) or not np.isfinite(probs).all():
        raise ValueError("Three-class probabilities are incomplete")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("Three-class probabilities lie outside [0, 1]")
    if not np.array_equal(bundle["classifier"].classes_, [0, 1, 2]):
        raise ValueError("Three-class label order changed")
    if not np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Three-class probabilities do not sum to one")
    delta = _supports(np.column_stack([model.predict(transformed)
                                       for model in bundle["magnitudes"]]))
    if not np.isfinite(delta).all():
        raise ValueError("Conditional magnitude prediction is nonfinite")
    point = np.maximum(0, origin_power + np.sum(probs * delta, axis=1))
    return point, probs


def _choose_alpha(truth: np.ndarray, r1: np.ndarray, expert: np.ndarray,
                  peak: np.ndarray) -> tuple[float, dict[str, dict[str, float]]]:
    if (not peak.any() or not np.isfinite(truth).all()
            or not np.isfinite(r1).all() or not np.isfinite(expert).all()):
        raise ValueError("STOP needs finite forecasts and observed peak cases")
    metrics = {}
    for alpha in ALPHAS:
        pred = np.maximum(0, (1 - alpha) * r1 + alpha * expert)
        error = np.abs(truth - pred)
        mae = float(error.mean())
        peak_mae = float(error[peak].mean())
        metrics[str(alpha)] = {"MAE": mae, "PeakMAE": peak_mae,
                               "objective": mae + 0.25 * peak_mae}
    winner = min(ALPHAS, key=lambda a: (metrics[str(a)]["objective"], a))
    return winner, metrics


def _identity(view, spec: dict, paths_sha: str, fold: int,
              fit: pd.DatetimeIndex, stop: pd.DatetimeIndex) -> str:
    root = Path(view.root)
    roles = {f"h{h}_{role}": _digest_frame(pd.DataFrame({"origin": view.contexts[(h, fold)][role]}))
             for h in HORIZONS for role in ROLES}
    targets = {f"h{h}_{role}": _digest_frame(pd.DataFrame({
        "y": view.contexts[(h, fold)]["y"].loc[origins].to_numpy(float),
        "target_time": pd.DatetimeIndex(view.contexts[(h, fold)]["target_time"].loc[origins]),
    })) for h in HORIZONS for role, origins in (("fit", fit), ("stop", stop))}
    return config_hash({"spec": spec, "paths_sha256": paths_sha, "fold": fold,
                        "roles": roles, "fit": fit.asi8.tolist(),
                        "stop": stop.asi8.tolist(), "targets": targets,
                        "taus": {str(h): float(view.contexts[(h, fold)]["tau"])
                                 for h in HORIZONS},
                        "split_sha256": view.split_lock["lock_sha256"],
                        "raw_sha256": view.seal["raw_sha256"],
                        "sources": {name: sha256(root / name) for name in _SOURCES}})


def _quarantine(path: Path) -> Path | None:
    if not path.exists():
        return None
    digest = sha256(path)[:12]
    target = path.with_name(f"{path.stem}.orphan-{digest}{path.suffix}")
    suffix = 0
    while target.exists():
        suffix += 1
        target = path.with_name(f"{path.stem}.orphan-{digest}-{suffix}{path.suffix}")
    path.rename(target)
    return target


def _reject_checkpoint(view, model_path: Path, meta_path: Path,
                       expected_identity: str, reason: str) -> None:
    """Preserve both physical files and an immutable byte-hash incident record."""
    model_sha = sha256(model_path) if model_path.is_file() else None
    metadata_sha = sha256(meta_path) if meta_path.is_file() else None
    model_saved = _quarantine(model_path)
    metadata_saved = _quarantine(meta_path)
    record = {
        "reason": reason, "expected_identity": expected_identity,
        "model_original": str(model_path), "model_sha256": model_sha,
        "model_preserved": str(model_saved) if model_saved else None,
        "metadata_original": str(meta_path), "metadata_sha256": metadata_sha,
        "metadata_preserved": str(metadata_saved) if metadata_saved else None,
        "quarantined_without_deserializing_unverified_model": True,
        "refit_performed_in_this_attempt": False,
    }
    if (model_saved and sha256(model_saved) != model_sha) or (
            metadata_saved and sha256(metadata_saved) != metadata_sha):
        raise RuntimeError("Transition checkpoint changed while preserving corrupt bytes")
    incident = Path(view.out) / "logs/checkpoint_corruption" / \
        f"{model_path.stem}-{config_hash(record)[:20]}.json"
    write_json(incident, record, exclusive=True)
    raise RuntimeError(f"Transition checkpoint rejected ({reason}); preserved physical files and {incident}")


def _fit_or_load(view, spec: dict, paths: pd.DataFrame, paths_sha: str,
                 fold: int, fit: pd.DatetimeIndex, stop: pd.DatetimeIndex,
                 groups: tuple[str, ...]) -> tuple[dict, dict]:
    import lightgbm as lgb

    model_dir = Path(view.out) / "models/transition_expert" / spec["id"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"seed{int(spec['seed'])}_f{fold}.joblib"
    meta_path = model_path.with_suffix(".json")
    identity = _identity(view, spec, paths_sha, fold, fit, stop)
    if model_path.exists() != meta_path.exists():
        _quarantine(model_path)
        _quarantine(meta_path)
    if model_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            _reject_checkpoint(view, model_path, meta_path, identity, "metadata_json_corrupt")
        if not isinstance(meta, dict):
            _reject_checkpoint(view, model_path, meta_path, identity, "metadata_shape_changed")
        if meta.get("identity") != identity:
            _reject_checkpoint(view, model_path, meta_path, identity, "identity_changed")
        if meta.get("model_sha256") != sha256(model_path):
            _reject_checkpoint(view, model_path, meta_path, identity, "model_sha256_changed")
        try:
            bundle = joblib.load(model_path)
        except Exception:
            _reject_checkpoint(view, model_path, meta_path, identity, "payload_unreadable")
        if (not isinstance(bundle, dict) or
                set(bundle) != {"classifier", "magnitudes", "preprocessor", "alpha", "feature_columns"}):
            _reject_checkpoint(view, model_path, meta_path, identity, "payload_shape_changed")
        return bundle, {**meta, "cache_reused": True}

    start = perf_counter()
    x_fit, x_stop, y_fit, y_stop, origin_fit, origin_stop, stop_r1, stop_peak = [], [], [], [], [], [], [], []
    for h in HORIZONS:
        c = view.contexts[(h, fold)]
        for origins, xs, ys, powers in ((fit, x_fit, y_fit, origin_fit),
                                        (stop, x_stop, y_stop, origin_stop)):
            xs.append(_matrix(view.history, paths, origins, h, c["tau"], groups))
            ys.append(c["y"].loc[origins].to_numpy(float))
            powers.append(_origin_power(view.history, origins))
        stop_r1.append(_path_values(paths, stop, h).r1.to_numpy(float))
        stop_peak.append(y_stop[-1] > float(c["tau"]))
    x_fit, x_stop = pd.concat(x_fit, ignore_index=True), pd.concat(x_stop, ignore_index=True)
    y_fit, y_stop = np.concatenate(y_fit), np.concatenate(y_stop)
    origin_fit, origin_stop = np.concatenate(origin_fit), np.concatenate(origin_stop)
    stop_r1, stop_peak = np.concatenate(stop_r1), np.concatenate(stop_peak)
    valid_fit, valid_stop = np.isfinite(origin_fit), np.isfinite(origin_stop)
    if not (valid_fit.all() and valid_stop.all()
            and np.isfinite(y_fit).all() and np.isfinite(y_stop).all()
            and np.isfinite(stop_r1).all()):
        raise ValueError("FIT/STOP observed origin power, target or frozen R1 path is nonfinite")
    fit_delta, stop_delta = y_fit[valid_fit] - origin_fit[valid_fit], y_stop[valid_stop] - origin_stop[valid_stop]
    fit_class, stop_class = _classes(fit_delta), _classes(stop_delta)
    fit_count = np.bincount(fit_class, minlength=3)
    stop_count = np.bincount(stop_class, minlength=3)
    if np.any(fit_count < MIN_CLASS) or np.any(stop_count < MIN_CLASS):
        raise ValueError(f"Transition classes under minimum {MIN_CLASS}: FIT {fit_count.tolist()}, STOP {stop_count.tolist()}")
    if not stop_peak[valid_stop].any():
        raise ValueError("STOP has no observed peak among valid origin-power rows")
    pre = _preprocessor(x_fit.loc[valid_fit], scale=False)
    transformed_fit = _transform(x_fit.loc[valid_fit], pre)
    transformed_stop = _transform(x_stop.loc[valid_stop], pre)
    seed = int(spec["seed"])
    classifier = lgb.LGBMClassifier(**PARAMS, objective="multiclass", metric="multi_logloss",
                                    num_class=3, random_state=seed)
    classifier.fit(transformed_fit, fit_class,
                   eval_set=[(transformed_stop, stop_class)], eval_metric="multi_logloss",
                   callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
    magnitudes = []
    iterations = {"classifier": int(classifier.best_iteration_)}
    for cls in range(3):
        fit_mask, stop_mask = fit_class == cls, stop_class == cls
        model = lgb.LGBMRegressor(**PARAMS, objective="regression_l1", metric="l1",
                                  random_state=seed)
        model.fit(transformed_fit[fit_mask], fit_delta[fit_mask],
                  eval_set=[(transformed_stop[stop_mask], stop_delta[stop_mask])],
                  eval_metric="l1",
                  callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
        magnitudes.append(model)
        iterations[str(cls)] = int(model.best_iteration_)
    bundle = {"classifier": classifier, "magnitudes": magnitudes,
              "preprocessor": pre, "alpha": None,
              "feature_columns": list(x_fit.columns)}
    expert, _ = _predict_expert(bundle, transformed_stop, origin_stop[valid_stop])
    alpha, stop_objectives = _choose_alpha(y_stop[valid_stop], stop_r1[valid_stop],
                                           expert, stop_peak[valid_stop])
    bundle["alpha"] = alpha
    temp = model_path.with_suffix(".joblib.tmp")
    _quarantine(temp)
    joblib.dump(bundle, temp)
    os.replace(temp, model_path)
    meta = {"identity": identity, "model_sha256": sha256(model_path),
            "seed": seed, "fold": fold, "alpha": alpha,
            "stop_objectives": stop_objectives, "fit_class_count": fit_count.tolist(),
            "stop_class_count": stop_count.tolist(),
            "fit_invalid_origin_count": int((~valid_fit).sum()),
            "stop_invalid_origin_count": int((~valid_stop).sum()),
            "stop_peak_count": int(stop_peak[valid_stop].sum()),
            "production_availability": {
                "fit": _production_availability(view.history,
                    pd.DatetimeIndex(np.tile(fit, len(HORIZONS))), x_fit),
                "stop": _production_availability(view.history,
                    pd.DatetimeIndex(np.tile(stop, len(HORIZONS))), x_stop),
            },
            "n_fit": int(valid_fit.sum()), "n_stop": int(valid_stop.sum()),
            "best_iterations": iterations, "train_seconds": perf_counter() - start,
            "fit_only_preprocessing": True, "fit_only_peak_tau": True,
            "cache_reused": False}
    write_json(meta_path, meta)
    return bundle, meta


def _predict(bundle: dict, x: pd.DataFrame, power: np.ndarray,
             r1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if list(x.columns) != bundle["feature_columns"]:
        raise ValueError("Transition feature columns changed")
    pred = np.asarray(r1, float).copy()
    if not np.isfinite(power).all() or not np.isfinite(pred).all():
        raise ValueError("Transition origin power or frozen R1 path is unavailable")
    probabilities = np.full((len(pred), 3), np.nan)
    expert_points = np.full(len(pred), np.nan)
    valid = np.isfinite(power)
    if valid.any():
        expert, probs = _predict_expert(bundle, _transform(x.loc[valid], bundle["preprocessor"]), power[valid])
        alpha = float(bundle["alpha"])
        pred[valid] = np.maximum(0, (1 - alpha) * pred[valid] + alpha * expert)
        probabilities[valid] = probs
        expert_points[valid] = expert
    if not np.isfinite(pred).all():
        raise ValueError("Transition forecast is nonfinite")
    return pred, probabilities, expert_points


def _perturbation_check(view, paths, fold, context, bundle, groups) -> float:
    h = 16
    origin = pd.Timestamp(context["score"][0])
    chosen = pd.DatetimeIndex([origin])
    altered = view.history.copy()
    future = altered.index > origin
    rng = np.random.default_rng(42)
    touched = []
    for name in altered:
        if name == "power" or name.startswith("production_"):
            if pd.api.types.is_bool_dtype(altered[name]):
                altered.loc[future, name] = ~altered.loc[future, name].astype(bool)
            elif pd.api.types.is_datetime64_any_dtype(altered[name]):
                altered.loc[future, name] = altered.loc[future, name] + pd.Timedelta(days=90)
            elif pd.api.types.is_numeric_dtype(altered[name]):
                altered.loc[future, name] = rng.normal(10000, 1000, int(future.sum()))
            else:
                raise ValueError(f"Unrecognized production observation type: {name}")
            touched.append(name)
    if "power" not in touched or not {"production_completed", "production_known"} <= set(touched):
        raise ValueError("Future probe did not perturb required observed input columns")
    a, provenance = build_features(view.history.loc[:origin], chosen, h, context["tau"], groups)
    b, altered_provenance = build_features(altered.loc[:origin], chosen, h, context["tau"], groups)
    if any((used > chosen).fillna(False).any()
           for used in (*provenance.values(), *altered_provenance.values())):
        raise AssertionError("Transition features read a future observed value")
    if not a.equals(b):
        raise AssertionError("Future observations changed transition input features")
    xa = _matrix(view.history, paths, chosen, h, context["tau"], groups)
    xb = _matrix(altered, paths, chosen, h, context["tau"], groups)
    if not xa.equals(xb):
        raise AssertionError("Future observations changed transition matrix")
    if not np.array_equal(_transform(xa, bundle["preprocessor"]),
                          _transform(xb, bundle["preprocessor"])):
        raise AssertionError("Future observations changed transition transform")
    power = _origin_power(view.history, chosen)
    r1 = _path_values(paths, chosen, h).r1.to_numpy(float)
    original = _predict(bundle, xa, power, r1)
    changed = _predict(bundle, xb, power, r1)
    if not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(original, changed)):
        raise AssertionError("Future observations changed transition prediction")
    return 0.0


def run(prepared, spec: dict, rolling_paths: pd.DataFrame,
        *, smoke_first_fold: bool = False) -> tuple[pd.DataFrame, dict]:
    if (not _ID.fullmatch(str(spec.get("id", ""))) or "seed" not in spec
            or spec.get("adapter") != "transition_expert"
            or spec.get("arm") != "EXPLORE"):
        raise ValueError("Transition expert requires explicit EXPLORE spec, seed and safe ID")
    if not prepared.split_lock.get("lock_sha256") or not prepared.seal.get("raw_sha256"):
        raise ValueError("Transition expert requires sealed split and raw source")
    if {h for h, _ in prepared.contexts} != set(HORIZONS):
        raise ValueError("Transition expert requires thirteen horizons")
    folds = sorted({fold for _, fold in prepared.contexts})
    if not folds or any({h for h, f in prepared.contexts if f == fold} != set(HORIZONS)
                        for fold in folds):
        raise ValueError("Transition expert needs a complete weekly horizon grid")
    if any(prepared.contexts[(4, fold)].get("summary", {}).get("arm") not in (None, "EXPLORE")
           for fold in folds):
        raise ValueError("Transition expert received a non-EXPLORE weekly context")
    groups = tuple(spec.get("groups", ()))
    if groups not in (
        ("lag_1_16", "slot_7_28d", "profile", "rolling", "trend", "calendar", "peak"),
        ("lag_1_16", "slot_7_28d", "profile", "rolling", "trend", "calendar", "peak", "production"),
    ):
        raise ValueError("Transition expert groups differ from fixed two-configuration design")
    paths, paths_sha = _paths_index(rolling_paths, prepared.contexts)
    proof = rolling_paths.attrs.get("causal_provenance", {})
    upstream = spec.get("rolling_audit_sha256")
    if (not isinstance(upstream, str) or not re.fullmatch(r"[0-9a-f]{64}", upstream)
            or not isinstance(proof, dict) or config_hash(proof) != upstream
            or proof.get("leakage_test") != "passed"
            or proof.get("input_cutoff_rule") != "history.index <= origin"
            or proof.get("future_perturbation_max_abs_difference") != 0.0):
        raise ValueError("Transition expert requires signed causal rolling R1 provenance")
    frames, audits = [], []
    for fold in folds[:1] if smoke_first_fold else folds:
        fit, stop, roles = _common_roles(prepared.contexts, fold)
        bundle, cell = _fit_or_load(prepared, spec, paths, paths_sha, fold, fit, stop, groups)
        checkpoint = Path(prepared.out) / "models/transition_expert" / spec["id"] / \
            f"seed{int(spec['seed'])}_f{fold}.joblib"
        cell["checkpoint_metadata_sha256"] = sha256(checkpoint.with_suffix(".json"))
        perturbation = _perturbation_check(prepared, paths, fold,
                                           prepared.contexts[(16, fold)], bundle, groups)
        prediction_production_audit = []
        for h in HORIZONS:
            c = prepared.contexts[(h, fold)]
            for role in ("cal", "score"):
                origins = pd.DatetimeIndex(c[role])
                x = _matrix(prepared.history, paths, origins, h, c["tau"], groups)
                origin_power = _origin_power(prepared.history, origins)
                r1 = _path_values(paths, origins, h).r1.to_numpy(float)
                pred, prob, expert_point = _predict(bundle, x, origin_power, r1)
                part = make_frame(c, origins, h, fold, pred, spec["id"], role,
                                  r1=r1, transition_p_fall=prob[:, 0],
                                  transition_p_neutral=prob[:, 1],
                                  transition_p_rise=prob[:, 2],
                                  transition_expert_point=expert_point)
                part["transition_origin_power"] = origin_power
                frames.append(part)
                if "production" in groups:
                    prediction_production_audit.append({"horizon": h, "role": role,
                        **_production_availability(prepared.history, origins, x)})
        audits.append({**roles, **cell, "fold": fold,
                       "prediction_production_availability": prediction_production_audit,
                       "future_perturbation_max_abs_difference": perturbation})
        if (Path(prepared.out) / "logs/stop_requested.json").exists():
            raise RuntimeError(f"Transition stop requested after fold {fold} checkpoint")
    frame = pd.concat(frames, ignore_index=True)
    expected = sum(len(prepared.contexts[(h, fold)][role])
                   for fold in (folds[:1] if smoke_first_fold else folds)
                   for h in HORIZONS for role in ("cal", "score"))
    if len(frame) != expected or frame.duplicated(["horizon", "fold", "role", "origin"]).any():
        raise ValueError("Transition expert changed the selected weekly cohort")
    return frame, {"adapter": "transition_expert", "paths_sha256": paths_sha,
                   "rolling_audit_sha256": upstream,
                   "rolling_upstream_causal_manifest_verified": True,
                   "leakage_test": "passed", "cells": audits,
                   "n_fold_models": len(audits), "seed": int(spec["seed"]),
                   "future_perturbation_max_abs_difference": 0.0,
                   "smoke_first_fold": smoke_first_fold,
                   "development_only": True, "holdout_read": False}
