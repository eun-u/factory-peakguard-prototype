"""Causal, weekly LightGBM correction of frozen Chronos-2 R1 forecasts.

``rolling_paths`` is produced by the separate pinned, zero-shot Chronos
inference job.  This module never fits Chronos, never selects on CAL/SCORE,
and keeps all thirteen horizons of a week in one residual estimator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from phase_f.features_ext import build_features
from phase_f.harness import BOUNDARY, make_frame
from phase_f.models.regression import _features, _preprocessor, _transform
from phase_f.registry import config_hash, sha256, write_json


HORIZONS = tuple(range(4, 17))
ROLES = ("fit", "stop", "cal", "score")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_ID = re.compile(r"[A-Za-z0-9_.+-]+\Z")
_SLOT = pd.Timedelta(minutes=15)
_CACHE_SOURCES = (
    "phase_f/models/r1_residual.py", "phase_f/models/regression.py",
    "phase_f/features_ext.py", "phase_f/harness.py", "phase_c/data.py",
)


def _digest_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def _paths_index(rolling_paths: pd.DataFrame, contexts: dict) -> tuple[pd.DataFrame, str]:
    required = {"origin", "horizon", "r1", "q10", "q50", "q90", "q95"}
    if not required <= set(rolling_paths):
        raise ValueError(f"R1 rolling paths missing columns: {sorted(required - set(rolling_paths))}")
    data = rolling_paths[list(sorted(required))].copy()
    data["origin"] = pd.to_datetime(data["origin"])
    if data.origin.isna().any() or data.origin.dt.tz is not None:
        raise ValueError("R1 rolling origins must be naive, finite timestamps")
    raw_horizon = pd.to_numeric(data.horizon, errors="raise")
    if not np.isfinite(raw_horizon.to_numpy(float)).all() or not raw_horizon.eq(raw_horizon.astype(int)).all():
        raise ValueError("R1 rolling horizon must be an integer")
    data["horizon"] = raw_horizon.astype(int)
    if data.duplicated(["origin", "horizon"]).any() or data.empty:
        raise ValueError("R1 rolling paths require unique origin/horizon rows")
    if not data.horizon.isin(HORIZONS).all():
        raise ValueError("R1 rolling paths contain an unsupported horizon")
    if (data.origin + pd.to_timedelta(data.horizon * 15, unit="m") >= BOUNDARY).any():
        raise ValueError("R1 rolling paths cross the sealed target boundary")
    for name in ("r1", "q10", "q50", "q90", "q95"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name].to_numpy(float)).all():
            raise ValueError(f"R1 rolling paths contain nonfinite {name}")
    # Quantile crossings can occur in an upstream forecast; preserve the path.
    # The median point recipe itself must agree with its q50 column.
    if not np.allclose(data.r1, data.q50, rtol=0, atol=1e-5):
        raise ValueError("The pinned R1 median and q50 path disagree")
    ordered = data.sort_values(["origin", "horizon"], kind="stable").reset_index(drop=True)
    indexed = ordered.set_index(["origin", "horizon"], verify_integrity=True)
    for (h, fold), context in contexts.items():
        for role in ROLES:
            origins = pd.DatetimeIndex(context[role])
            keys = pd.MultiIndex.from_arrays([origins, np.repeat(h, len(origins))], names=indexed.index.names)
            if not keys.isin(indexed.index).all():
                raise ValueError(f"R1 path misses {role} origins for h={h}, fold={fold}")
    return indexed, _digest_frame(ordered)


def _common_roles(contexts: dict, fold: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, dict]:
    selected = {h: contexts[(h, fold)] for h in HORIZONS}
    fit = pd.DatetimeIndex(sorted(set.intersection(*(set(c["fit"]) for c in selected.values()))))
    stop = pd.DatetimeIndex(sorted(set.intersection(*(set(c["stop"]) for c in selected.values()))))
    if fit.empty or stop.empty:
        raise ValueError(f"No common FIT/STOP origins across all horizons in fold {fold}")
    cal_start = min(pd.DatetimeIndex(c["cal"]).min() for c in selected.values())
    fit_before, stop_before = len(fit), len(stop)
    # The maximum horizon purges every pooled target before the next role's
    # earliest origin.  This is stricter than a horizon-specific purge.
    fit = fit[fit + max(HORIZONS) * _SLOT < stop.min()]
    stop = stop[stop + max(HORIZONS) * _SLOT < cal_start]
    if len(fit) < 200 or len(stop) < 20:
        raise ValueError(f"Insufficient common embargoed FIT/STOP rows in fold {fold}")
    for h, c in selected.items():
        if not (c["target_time"].loc[fit].max() < stop.min()
                and c["target_time"].loc[stop].max() < cal_start):
            raise AssertionError(f"Pooled target embargo failed for h={h}, fold={fold}")
    return fit, stop, {"common_fit_before_purge": fit_before,
                        "common_stop_before_purge": stop_before,
                        "common_fit_after_purge": len(fit),
                        "common_stop_after_purge": len(stop),
                        "fit_last_target": str(fit.max() + max(HORIZONS) * _SLOT),
                        "stop_last_target": str(stop.max() + max(HORIZONS) * _SLOT),
                        "stop_first_origin": str(stop.min()), "cal_first_origin": str(cal_start)}


def _path_values(paths: pd.DataFrame, origins: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    keys = pd.MultiIndex.from_arrays([origins, np.repeat(h, len(origins))], names=paths.index.names)
    values = paths.loc[keys].reset_index(drop=True)
    values.index = origins
    return values


def _known_error(history: pd.DataFrame, paths: pd.DataFrame,
                 origins: pd.DatetimeIndex, h: int) -> np.ndarray:
    previous = origins - h * _SLOT
    keys = pd.MultiIndex.from_arrays([previous, np.repeat(h, len(previous))], names=paths.index.names)
    prior = paths.r1.reindex(keys).to_numpy(float)
    observed = pd.to_numeric(history.power.reindex(origins), errors="coerce").to_numpy(float)
    for flag in ("time_repaired", "quality_bad"):
        if flag in history:
            bad = history[flag].reindex(origins).fillna(True).to_numpy(bool)
            observed[bad] = np.nan
    if (previous > origins).any() or (previous + h * _SLOT > origins).any():
        raise AssertionError("Unmatured prior R1 forecast used as a feature")
    return observed - prior


def _matrix(history: pd.DataFrame, paths: pd.DataFrame, origins: pd.DatetimeIndex,
            h: int, tau: float, groups: tuple[str, ...], known_error: bool) -> pd.DataFrame:
    x = _features(history, origins, h, tau, groups)
    p = _path_values(paths, origins, h)
    extra = pd.DataFrame({
        "r1_point": p.r1.to_numpy(float),
        "r1_q10": p.q10.to_numpy(float),
        "r1_q90": p.q90.to_numpy(float),
        "r1_q95": p.q95.to_numpy(float),
        "r1_width_80": (p.q90 - p.q10).to_numpy(float),
        "horizon_quarters": np.full(len(origins), float(h)),
    }, index=origins)
    if known_error:
        error = _known_error(history, paths, origins, h)
        extra["last_matured_r1_error"] = error
        extra["last_matured_r1_abs_error"] = np.abs(error)
    result = pd.concat([x, extra], axis=1)
    if result.columns.has_duplicates:
        raise ValueError("Duplicate residual feature names")
    return result


def _weights(view, fit: pd.DatetimeIndex, h: int, fold: int, spec: dict) -> np.ndarray:
    c = view.contexts[(h, fold)]
    y = c["y"].loc[fit].to_numpy(float)
    peak_weight = float(spec.get("peak_weight", 1.0))
    if not np.isfinite(peak_weight) or peak_weight <= 0:
        raise ValueError("peak_weight must be positive and finite")
    weights = np.where(y > float(c["tau"]), peak_weight, 1.0)
    half_life = spec.get("recency_halflife_days")
    if half_life is not None:
        half_life = float(half_life)
        if not np.isfinite(half_life) or half_life <= 0:
            raise ValueError("recency_halflife_days must be positive and finite")
        age_days = np.asarray((fit.max() - fit) / pd.Timedelta(days=1), float)
        weights *= np.exp2(-age_days / half_life)
    if not np.isfinite(weights).all() or not (weights > 0).any():
        raise ValueError("No positive finite FIT sample weights")
    return weights


def _identity(view, spec: dict, paths_sha: str, fold: int, fit, stop, groups) -> str:
    root = Path(view.root)
    sources = {name: sha256(root / name) for name in _CACHE_SOURCES}
    roles = {f"h{h}_{role}": _digest_frame(pd.DataFrame({"origin": view.contexts[(h, fold)][role]}))
             for h in HORIZONS for role in ROLES}
    targets = {f"h{h}_{role}": _digest_frame(pd.DataFrame({
        "y": view.contexts[(h, fold)]["y"].loc[origins].to_numpy(float),
        "target_time": pd.DatetimeIndex(view.contexts[(h, fold)]["target_time"].loc[origins]),
    })) for h in HORIZONS for role, origins in (("fit", fit), ("stop", stop))}
    taus = {str(h): float(view.contexts[(h, fold)]["tau"]) for h in HORIZONS}
    return config_hash({"spec": spec, "paths_sha256": paths_sha, "fold": fold,
                        "roles": roles, "pooled_fit": fit.asi8.tolist(),
                        "pooled_stop": stop.asi8.tolist(), "groups": groups,
                        "fit_stop_targets": targets, "fit_only_taus": taus,
                        "split_sha256": view.split_lock.get("lock_sha256"),
                        "raw_sha256": view.seal.get("raw_sha256"), "sources": sources})


def _quarantine(path: Path) -> None:
    if not path.exists():
        return
    digest = sha256(path)[:12]
    target = path.with_name(f"{path.stem}.orphan-{digest}{path.suffix}")
    n = 0
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}.orphan-{digest}-{n}{path.suffix}")
    path.rename(target)


def _fit_or_load(view, spec, paths, paths_sha, fold, fit, stop, groups):
    import lightgbm as lgb

    out = Path(view.out) / "models" / "r1_residual" / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"seed{int(spec['seed'])}_f{fold}.joblib"
    meta_path = model_path.with_suffix(".json")
    identity = _identity(view, spec, paths_sha, fold, fit, stop, groups)
    if model_path.exists() != meta_path.exists():
        _quarantine(model_path)
        _quarantine(meta_path)
    if model_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("identity") != identity or meta.get("model_sha256") != sha256(model_path):
            raise RuntimeError("R1 residual checkpoint identity/hash mismatch")
        bundle = joblib.load(model_path)
        if set(bundle) != {"model", "preprocessor", "alpha", "feature_columns"}:
            raise RuntimeError("R1 residual checkpoint payload changed")
        return bundle, {**meta, "cache_reused": True}

    tick = perf_counter()
    fit_x, stop_x, fit_y, stop_y, fit_w = [], [], [], [], []
    for h in HORIZONS:
        c = view.contexts[(h, fold)]
        fit_x.append(_matrix(view.history, paths, fit, h, c["tau"], groups,
                             bool(spec.get("known_error_features", True))))
        stop_x.append(_matrix(view.history, paths, stop, h, c["tau"], groups,
                              bool(spec.get("known_error_features", True))))
        fit_y.append(c["y"].loc[fit].to_numpy(float) - _path_values(paths, fit, h).r1.to_numpy(float))
        stop_y.append(c["y"].loc[stop].to_numpy(float) - _path_values(paths, stop, h).r1.to_numpy(float))
        fit_w.append(_weights(view, fit, h, fold, spec))
    fit_x = pd.concat(fit_x, ignore_index=True)
    stop_x = pd.concat(stop_x, ignore_index=True)
    fit_y = np.concatenate(fit_y)
    stop_y = np.concatenate(stop_y)
    fit_w = np.concatenate(fit_w)
    if not np.isfinite(fit_y).all() or not np.isfinite(stop_y).all():
        raise ValueError("Nonfinite residual FIT/STOP target")
    pre = _preprocessor(fit_x, scale=False)
    x_fit, x_stop = _transform(fit_x, pre), _transform(stop_x, pre)
    params = {"n_estimators": 900, "learning_rate": .035, "num_leaves": 31,
              "min_child_samples": 50, "colsample_bytree": .85,
              "subsample": .85, "subsample_freq": 1, "n_jobs": 4,
              "verbosity": -1, "objective": "regression", **dict(spec.get("model_params", {}))}
    if params["objective"] in {"gamma", "poisson", "tweedie"}:
        raise ValueError("Signed residuals require a signed LightGBM objective")
    params["random_state"] = int(spec["seed"])
    params["metric"] = "l1"  # Stop iteration is chosen by absolute residual error.
    model = lgb.LGBMRegressor(**params)
    patience = int(spec.get("early_stopping_rounds", 75))
    if patience < 1:
        raise ValueError("early_stopping_rounds must be positive")
    model.fit(x_fit, fit_y, sample_weight=fit_w,
              eval_set=[(x_stop, stop_y)], eval_metric="l1",
              callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
    stop_correction = model.predict(x_stop)
    stop_r1 = np.concatenate([_path_values(paths, stop, h).r1.to_numpy(float) for h in HORIZONS])
    stop_truth = stop_y + stop_r1
    forced = spec.get("shrinkage")
    if forced is None:
        candidates = ALPHAS
    else:
        forced = float(forced)
        if not np.isfinite(forced) or not 0 <= forced <= 1:
            raise ValueError("Forced shrinkage must lie in [0,1]")
        candidates = (forced,)
    stop_maes = {str(alpha): float(np.mean(np.abs(stop_truth - np.maximum(0, stop_r1 + alpha * stop_correction))))
                 for alpha in candidates}
    alpha = min(candidates, key=lambda value: (stop_maes[str(value)], value))
    bundle = {"model": model, "preprocessor": pre, "alpha": float(alpha),
              "feature_columns": list(fit_x.columns)}
    # A process interruption can leave a .tmp; retain its original bytes.
    temp = model_path.with_suffix(".joblib.tmp")
    _quarantine(temp)
    joblib.dump(bundle, temp)
    os.replace(temp, model_path)
    meta = {"identity": identity, "model_sha256": sha256(model_path),
            "seed": int(spec["seed"]), "fold": fold, "alpha": float(alpha),
            "stop_MAE": stop_maes[str(alpha)], "stop_MAE_by_alpha": stop_maes,
            "n_fit": len(fit_y), "n_stop": len(stop_y),
            "best_iteration": int(model.best_iteration_),
            "train_seconds": perf_counter() - tick,
            "fit_only_preprocessing": True, "fit_only_peak_tau": True,
            "cache_reused": False}
    write_json(meta_path, meta)
    return bundle, meta


def _perturbation_check(view, paths, fold, h, origin, context, bundle, groups, known_error) -> float:
    origin = pd.Timestamp(origin)
    chosen = pd.DatetimeIndex([origin])
    original = view.history
    altered = original.copy()
    future = altered.index > origin
    altered.loc[future, "power"] = np.random.default_rng(42).normal(10000, 1000, int(future.sum()))
    a, aprovenance = build_features(original, chosen, h, context["tau"], groups)
    b, bprovenance = build_features(altered, chosen, h, context["tau"], groups)
    if any((times > chosen).fillna(False).any() for times in (*aprovenance.values(), *bprovenance.values())):
        raise AssertionError("A selected power feature reads a future observation")
    if not a.equals(b):
        raise AssertionError("Future power perturbation changed a causal feature")
    xa = _matrix(original, paths, chosen, h, context["tau"], groups, known_error)
    xb = pd.concat([b, xa.loc[:, xa.columns.difference(b.columns)]], axis=1).loc[:, xa.columns]
    if not np.array_equal(_transform(xa, bundle["preprocessor"]),
                          _transform(xb, bundle["preprocessor"])):
        raise AssertionError("Future perturbation changed transformed correction features")
    first = bundle["model"].predict(_transform(xa, bundle["preprocessor"]))
    second = bundle["model"].predict(_transform(xb, bundle["preprocessor"]))
    difference = float(np.max(np.abs(first - second)))
    if difference != 0:
        raise AssertionError("Future perturbation changed R1 residual prediction")
    return difference


def run(prepared, spec: dict, rolling_paths: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit/reuse one LightGBM per week; return every CAL and SCORE key.

    The caller must pass one EXPLORE arm view.  A rolling producer manifest
    digest is required before this adapter reports the entire path causal.
    """
    if not _ID.fullmatch(str(spec.get("id", ""))) or "seed" not in spec:
        raise ValueError("R1 residual needs a safe ID and explicit actual seed")
    if spec.get("adapter") != "r1_residual" or spec.get("arm", "EXPLORE") != "EXPLORE":
        raise ValueError("R1 residual adapter accepts EXPLORE arm only")
    if not prepared.split_lock.get("lock_sha256") or not prepared.seal.get("raw_sha256"):
        raise ValueError("R1 residual requires the weekly split and source seals")
    groups = tuple(spec.get("groups", ()))
    if "production" in groups:
        raise ValueError("Future production is unavailable to the R1 residual adapter")
    if not prepared.contexts or {h for h, _ in prepared.contexts} != set(HORIZONS):
        raise ValueError("R1 residual requires all thirteen horizons")
    folds = sorted({fold for _, fold in prepared.contexts})
    if any({h for h, f in prepared.contexts if f == fold} != set(HORIZONS) for fold in folds):
        raise ValueError("R1 residual requires a complete horizon set per week")
    for fold in folds:
        arm = prepared.contexts[(4, fold)].get("summary", {}).get("arm")
        if arm not in (None, "EXPLORE"):
            raise ValueError("R1 residual received a non-EXPLORE weekly context")
    paths, paths_sha = _paths_index(rolling_paths, prepared.contexts)
    upstream = spec.get("rolling_audit_sha256")
    proof = rolling_paths.attrs.get("causal_provenance", {})
    difference = proof.get("future_perturbation_max_abs_difference") if isinstance(proof, dict) else None
    upstream_verified = (
        isinstance(upstream, str) and bool(re.fullmatch(r"[0-9a-f]{64}", upstream))
        and isinstance(proof, dict)
        and proof.get("leakage_test") == "passed"
        and proof.get("input_cutoff_rule") == "history.index <= origin"
        and isinstance(proof.get("model_revision"), str) and bool(proof["model_revision"])
        and isinstance(difference, (int, float, np.integer, np.floating))
        and not isinstance(difference, (bool, np.bool_))
        and np.isfinite(float(difference)) and float(difference) == 0.0
    )
    frames, audits = [], []
    for fold in folds:
        fit, stop, role_audit = _common_roles(prepared.contexts, fold)
        bundle, cell = _fit_or_load(prepared, spec, paths, paths_sha, fold, fit, stop, groups)
        if not bundle["feature_columns"]:
            raise RuntimeError("Checkpoint has no features")
        check_h = 16
        check_c = prepared.contexts[(check_h, fold)]
        perturbation = _perturbation_check(prepared, paths, fold, check_h, check_c["score"][0],
                                           check_c, bundle, groups,
                                           bool(spec.get("known_error_features", True)))
        for h in HORIZONS:
            c = prepared.contexts[(h, fold)]
            for role in ("cal", "score"):
                origins = pd.DatetimeIndex(c[role])
                x = _matrix(prepared.history, paths, origins, h, c["tau"], groups,
                            bool(spec.get("known_error_features", True)))
                if list(x.columns) != bundle["feature_columns"]:
                    raise RuntimeError("R1 residual feature columns changed after fitting")
                correction = bundle["model"].predict(_transform(x, bundle["preprocessor"]))
                r1 = _path_values(paths, origins, h).r1.to_numpy(float)
                prediction = np.maximum(0, r1 + bundle["alpha"] * correction)
                frames.append(make_frame(c, origins, h, fold, prediction, spec["id"], role,
                                         r1=r1, correction=correction))
        audits.append({**role_audit, **cell, "fold": fold,
                       "future_perturbation_max_abs_difference": perturbation})
        stop_request = Path(prepared.out) / "logs" / "stop_requested.json"
        if stop_request.exists():
            raise RuntimeError(f"R1 residual stop requested after durable fold {fold} checkpoint")
    frame = pd.concat(frames, ignore_index=True)
    expected = sum(len(c[role]) for c in prepared.contexts.values() for role in ("cal", "score"))
    if len(frame) != expected or frame.duplicated(["horizon", "fold", "role", "origin"]).any():
        raise AssertionError("R1 residual changed the locked CAL/SCORE cohort")
    if not np.isfinite(frame.pred.to_numpy(float)).all():
        raise AssertionError("R1 residual returned nonfinite predictions")
    return frame, {"adapter": "r1_residual", "paths_sha256": paths_sha,
                   "rolling_audit_sha256": upstream,
                   "rolling_upstream_causal_manifest_verified": upstream_verified,
                   "rolling_upstream_causal_provenance": proof,
                   "leakage_test": "passed" if upstream_verified else "requires_upstream_audit",
                   "future_perturbation_max_abs_difference": max(
                       c["future_perturbation_max_abs_difference"] for c in audits),
                   "cells": audits, "n_fold_models": len(audits),
                   "n_fit_total": sum(c["n_fit"] for c in audits),
                   "n_stop_total": sum(c["n_stop"] for c in audits),
                   "seed": int(spec["seed"]), "development_only": True,
                   "holdout_read": False}
