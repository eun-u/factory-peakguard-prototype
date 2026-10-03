"""F3-5 shared-horizon and F3-6 direct multi-output LightGBM experiments.

One fitted bundle belongs to one sealed fold.  The shared model stacks fit
labels across horizons and gets an explicit horizon feature.  The multi-output
variant fits 13 independent LightGBM heads to the same origin-level matrix,
which includes causal past and known target-calendar features for all horizons.
Neither variant uses cal or score labels while fitting or selecting iterations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from phase_f.features_ext import build_features
from phase_f.harness import make_frame
from phase_f.models.regression import _history, _preprocessor, _transform
from phase_f.registry import config_hash, sha256, write_json


HORIZONS = tuple(range(4, 17))


def configurations():
    """Named F3 shared and direct multi-output starting configurations."""
    groups = ("slot_1_7d", "rolling", "trend")
    return [
        {"id": f"F3-5-global-h-{target}", "family": "F3", "tier": 3,
         "kind": "global_h", "target": target, "groups": groups}
        for target in ("direct", "weekly")
    ] + [
        {"id": f"F3-6-multioutput-{target}", "family": "F3", "tier": 3,
         "kind": "multioutput", "target": target, "groups": groups}
        for target in ("direct", "weekly")
    ]


def _origins(prepared, horizon, fold, role):
    return pd.DatetimeIndex(prepared.origins(horizon, fold, role))


def _as_config(spec):
    cfg = {**spec, **spec.get("params", {})}
    if cfg.get("kind") not in {"global_h", "multioutput"}:
        raise ValueError("kind must be global_h or multioutput")
    if cfg.get("target", "direct") not in {"direct", "weekly"}:
        raise ValueError("Only direct and weekly targets are supported")
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", str(cfg.get("id", ""))):
        raise ValueError("Unsafe experiment ID")
    groups = tuple(cfg.get("groups", ()))
    if not all(isinstance(group, str) for group in groups):
        raise ValueError("groups must be feature-group names")
    cfg["groups"] = groups
    cfg["early_stopping_rounds"] = int(cfg.get("early_stopping_rounds", 50))
    if cfg["early_stopping_rounds"] < 1:
        raise ValueError("early_stopping_rounds must be positive")
    return cfg


def _folds(prepared):
    found = {fold for _, fold in prepared.contexts}
    for fold in found:
        missing = [h for h in HORIZONS if (h, fold) not in prepared.contexts]
        if missing:
            raise ValueError(f"Fold {fold} misses horizons {missing}")
    if not found:
        raise ValueError("No sealed fold contexts")
    return sorted(found)


def _feature_map(prepared, fold, cfg):
    """Build each horizon once for every origin needed by the fitted fold."""
    roles = ("fit", "stop", "cal", "score")
    all_origins = pd.DatetimeIndex(sorted(set().union(*(
        set(_origins(prepared, h, fold, role)) for h in HORIZONS for role in roles
    ))))
    if all_origins.empty:
        raise ValueError("Fold has no origins")
    history = prepared.history.loc[:all_origins.max()]
    result = {}
    for h in HORIZONS:
        context = prepared.contexts[(h, fold)]
        tau = float(context["tau"])
        if not np.isfinite(tau):
            raise ValueError("Fold fit-only tau is missing")
        x, provenance = build_features(history, all_origins, h, tau, cfg["groups"])
        if x.columns.duplicated().any():
            raise ValueError("Duplicate feature columns")
        if any((times > all_origins).fillna(False).any() for times in provenance.values()):
            raise AssertionError("Noncausal feature provenance")
        result[h] = x
    return result


def _joint_features(feature_map, origins):
    return pd.concat([feature_map[h].loc[origins].add_prefix(f"h{h:02d}_")
                      for h in HORIZONS], axis=1)


def _target(context, origins):
    values = context["y"].reindex(origins).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selected fit/stop labels are not finite")
    return values


def _anchor(features, target):
    if target == "direct":
        return np.zeros(len(features), dtype=float)
    return features["slot7d"].to_numpy(dtype=float)


def _prediction_anchor(features, target, fit_median):
    if target == "direct":
        return np.zeros(len(features), dtype=float), 0
    weekly = features["slot7d"].to_numpy(dtype=float)
    current = features["current"].to_numpy(dtype=float)
    missing = ~np.isfinite(weekly)
    anchor = np.where(np.isfinite(weekly), weekly,
                      np.where(np.isfinite(current), current, fit_median))
    return anchor, int(missing.sum())


def _matrix_and_labels(prepared, fold, cfg, features, role):
    """Global-h rows; train/stop target times obey the earliest next partition."""
    barrier = min(prepared.contexts[(h, fold)]["stop"].min() for h in HORIZONS) if role == "fit" else min(
        prepared.contexts[(h, fold)]["cal"].min() for h in HORIZONS)
    x_parts, y_parts, counts = [], [], {}
    for h in HORIZONS:
        context = prepared.contexts[(h, fold)]
        origins = _origins(prepared, h, fold, role)
        target_times = pd.DatetimeIndex(context["target_time"].reindex(origins))
        keep = target_times < barrier
        x = features[h].loc[origins]
        anchor = _anchor(x, cfg["target"])
        keep &= np.isfinite(anchor)
        origins, x, anchor = origins[keep], x.iloc[keep].copy(), anchor[keep]
        if len(origins):
            x["forecast_horizon_quarters"] = float(h)
            x_parts.append(x)
            y_parts.append(_target(context, origins) - anchor)
        counts[h] = int(len(origins))
    if not x_parts:
        raise ValueError(f"No {role} labels after common embargo and anchor filter")
    x_all = pd.concat(x_parts, axis=0, ignore_index=True)
    y_all = np.concatenate(y_parts)
    if role == "fit" and len(y_all) < 30:
        raise ValueError("Fewer than 30 shared fit labels")
    if role == "stop" and len(y_all) < 2:
        raise ValueError("Fewer than two shared stop labels")
    return x_all, y_all, counts


def _intersect_origins(prepared, fold, role, barrier):
    indices = [_origins(prepared, h, fold, role) for h in HORIZONS]
    common = indices[0]
    for item in indices[1:]:
        common = common.intersection(item)
    common = common.sort_values()
    valid = np.ones(len(common), dtype=bool)
    for h in HORIZONS:
        times = pd.DatetimeIndex(prepared.contexts[(h, fold)]["target_time"].reindex(common))
        valid &= times < barrier
    return common[valid]


def _multi_matrix_and_labels(prepared, fold, cfg, features, role):
    barrier = min(prepared.contexts[(h, fold)]["stop"].min() for h in HORIZONS) if role == "fit" else min(
        prepared.contexts[(h, fold)]["cal"].min() for h in HORIZONS)
    origins = _intersect_origins(prepared, fold, role, barrier)
    if origins.empty:
        raise ValueError(f"No common {role} origins for all horizons")
    x = _joint_features(features, origins)
    y = np.column_stack([_target(prepared.contexts[(h, fold)], origins) -
                         _anchor(features[h].loc[origins], cfg["target"])
                         for h in HORIZONS])
    keep = np.isfinite(y).all(axis=1)
    x, y, origins = x.iloc[keep], y[keep], origins[keep]
    if role == "fit" and len(origins) < 30:
        raise ValueError("Fewer than 30 common fit origins")
    if role == "stop" and len(origins) < 2:
        raise ValueError("Fewer than two common stop origins")
    return x, y, origins


def _fit_lightgbm(x, y, x_stop, y_stop, cfg, seed):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is unavailable") from exc
    params = {"n_estimators": 700, "learning_rate": .03, "num_leaves": 31,
              "verbosity": -1, **dict(cfg.get("model_params", {}))}
    requested_threads = int(params.pop("n_jobs", 4))
    params["n_jobs"] = min(4, max(1, requested_threads))
    params["random_state"] = int(seed)
    model = lgb.LGBMRegressor(**params)
    fit_options = {}
    if x_stop is not None and len(y_stop) >= 2:
        fit_options.update(eval_set=[(x_stop, y_stop)], eval_metric="l1",
                           callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    model.fit(x, y, **fit_options)
    return model


def _fit_fold(prepared, fold, cfg, features):
    tick = perf_counter()
    if cfg["kind"] == "global_h":
        x, y, fit_counts = _matrix_and_labels(prepared, fold, cfg, features, "fit")
        x_stop, y_stop, stop_counts = _matrix_and_labels(prepared, fold, cfg, features, "stop")
        pre = _preprocessor(x, scale=False)
        model = _fit_lightgbm(_transform(x, pre), y, _transform(x_stop, pre), y_stop,
                              cfg, int(cfg.get("seed", 42)))
        models = {"global": model}
        n_fit, n_stop = len(y), len(y_stop)
        detail = {"fit_by_horizon": fit_counts, "stop_by_horizon": stop_counts,
                  "best_iteration": int(model.best_iteration_ or model.n_estimators_)}
    else:
        x, y, fit_origins = _multi_matrix_and_labels(prepared, fold, cfg, features, "fit")
        x_stop, y_stop, stop_origins = _multi_matrix_and_labels(prepared, fold, cfg, features, "stop")
        pre = _preprocessor(x, scale=False)
        fit_matrix, stop_matrix = _transform(x, pre), _transform(x_stop, pre)
        models = {}
        for col, h in enumerate(HORIZONS):
            models[h] = _fit_lightgbm(fit_matrix, y[:, col], stop_matrix, y_stop[:, col],
                                      cfg, int(cfg.get("seed", 42)) + h)
        n_fit, n_stop = len(fit_origins), len(stop_origins)
        detail = {"fit_intersection_start": str(fit_origins.min()),
                  "fit_intersection_end": str(fit_origins.max()),
                  "stop_intersection_start": str(stop_origins.min()),
                  "best_iteration_by_horizon": {
                      str(h): int(models[h].best_iteration_ or models[h].n_estimators_) for h in HORIZONS}}
    earliest_stop = min(prepared.contexts[(h, fold)]["stop"].min() for h in HORIZONS)
    fit_values = []
    for h in HORIZONS:
        context = prepared.contexts[(h, fold)]
        fit = _origins(prepared, h, fold, "fit")
        fit = fit[pd.DatetimeIndex(context["target_time"].reindex(fit)) < earliest_stop]
        fit_values.append(_target(context, fit))
    fit_values = np.concatenate(fit_values)
    return {"kind": cfg["kind"], "target": cfg["target"], "preprocessor": pre,
            "models": models, "fit_median": float(np.median(fit_values)),
            "n_fit": n_fit, "n_stop": n_stop, "fit_seconds": perf_counter() - tick,
            "detail": detail, "fit_parameters_only": True, "development_only": True}


def _predict(bundle, features, origins, horizon):
    if bundle["kind"] == "global_h":
        x = features[horizon].loc[origins].copy()
        x["forecast_horizon_quarters"] = float(horizon)
        model = bundle["models"]["global"]
    else:
        x = _joint_features(features, origins)
        model = bundle["models"][horizon]
    residual = np.asarray(model.predict(_transform(x, bundle["preprocessor"])), dtype=float)
    anchor, missing = _prediction_anchor(features[horizon].loc[origins],
                                         bundle["target"], bundle["fit_median"])
    return residual + anchor, missing


def _identity(prepared, fold, spec):
    context_roles = {str(h): {role: [str(t) for t in _origins(prepared, h, fold, role)]
                              for role in ("fit", "stop", "cal", "score")}
                     for h in HORIZONS}
    return config_hash({"spec": spec, "fold": fold,
                        "source": sha256(Path(__file__)),
                        "features": sha256(Path(build_features.__code__.co_filename)),
                        "contexts": config_hash(context_roles),
                        "split": prepared.split_lock["lock_sha256"],
                        "raw": getattr(prepared, "seal", {}).get("raw_sha256", "synthetic")})


def _checkpoint(prepared, fold, spec, features, cfg):
    folder = Path(prepared.out) / "models" / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    path, meta = folder / f"fold_{fold}.joblib", folder / f"fold_{fold}.json"
    identity = _identity(prepared, fold, spec)
    if path.exists() or meta.exists():
        if not (path.is_file() and meta.is_file()):
            raise RuntimeError("Incomplete fold checkpoint; preserve it for investigation")
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("identity") != identity or record.get("sha256") != sha256(path):
            raise RuntimeError("Fold checkpoint identity or hash changed; use a new experiment ID")
        return joblib.load(path), True
    bundle = _fit_fold(prepared, fold, cfg, features)
    temp = path.with_suffix(".tmp")
    joblib.dump(bundle, temp)
    digest = sha256(temp)
    temp.replace(path)
    write_json(meta, {"identity": identity, "sha256": digest,
                      "fit_seconds": bundle["fit_seconds"], "n_fit": bundle["n_fit"],
                      "n_stop": bundle["n_stop"]})
    return bundle, False


def _future_probe(prepared, fold, cfg, bundle, features):
    """Compare actual fitted predictions after perturbing all later power."""
    score = _origins(prepared, HORIZONS[0], fold, "score")
    if score.empty:
        raise ValueError("Score origins are empty")
    origin = score[0]
    altered = prepared.history.copy()
    future = altered.index > origin
    altered.loc[future, "power"] = np.random.default_rng(42).normal(10000, 500, int(future.sum()))
    modified = {}
    probe = pd.DatetimeIndex([origin])
    for h in HORIZONS:
        tau = float(prepared.contexts[(h, fold)]["tau"])
        modified[h], _ = build_features(altered, probe, h, tau, cfg["groups"])
    differences = []
    for h in HORIZONS:
        before, _ = _predict(bundle, features, probe, h)
        after, _ = _predict(bundle, modified, probe, h)
        if not np.array_equal(before, after):
            raise AssertionError(f"Future perturbation changed fold {fold}, h{h} prediction")
        differences.append(float(np.max(np.abs(before - after))))
    return max(differences)


def run(prepared, spec):
    """Fit/cache per fold and return all fixed calibration and score keys."""
    cfg = _as_config(spec)
    _history(prepared.history)
    frames, fold_audits = [], []
    for fold in _folds(prepared):
        features = _feature_map(prepared, fold, cfg)
        bundle, reused = _checkpoint(prepared, fold, spec, features, cfg)
        difference = _future_probe(prepared, fold, cfg, bundle, features)
        inference_seconds, anchor_fallback = 0., 0
        for h in HORIZONS:
            context = prepared.contexts[(h, fold)]
            for role in ("cal", "score"):
                origins = _origins(prepared, h, fold, role)
                tick = perf_counter()
                pred, fallback = _predict(bundle, features, origins, h)
                elapsed = perf_counter() - tick
                inference_seconds += elapsed
                anchor_fallback += fallback
                frames.append(make_frame(context, origins, h, fold, pred, spec["id"], role,
                                         train_seconds=bundle["fit_seconds"],
                                         inference_seconds=elapsed))
        fold_audits.append({"fold": fold, "cache_reused": reused,
                            "n_fit": bundle["n_fit"], "n_stop": bundle["n_stop"],
                            "train_seconds": bundle["fit_seconds"],
                            "inference_seconds": inference_seconds,
                            "weekly_anchor_fallback_rows": anchor_fallback,
                            "future_perturbation_max_abs_difference": difference,
                            **bundle["detail"]})
    frame = pd.concat(frames, ignore_index=True)
    audit = {"kind": cfg["kind"], "target": cfg["target"], "folds": fold_audits,
             "leakage_test": "passed", "fit_parameters_only": True,
             "historical_final_artifact_read": False, "holdout_read": False}
    return frame, audit
