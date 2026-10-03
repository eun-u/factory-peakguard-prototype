"""Fit-only Phase F tabular regressors with causal prediction features.

This module never reads score labels, selects on EXPLORE, or writes artifacts.
The caller owns the sealed fold, fixed score cohort and configuration search.
For one horizon/fold, `fit_model` uses only context.fit labels, and optionally
context.stop for early stopping. Calibration and score labels are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
from datetime import timedelta
import hashlib
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from phase_c.statistical import fit_kalman, predict_kalman
from phase_f.features_ext import available_group_configs, build_features


KINDS = frozenset({"ridge", "lightgbm", "xgboost", "catboost", "two_stage"})
TARGETS = frozenset({"direct", "delta", "weekly", "profile", "b5_residual", "log1p"})
BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
_FEATURE_CACHE = OrderedDict()
_FEATURE_CACHE_BYTES = 0
_B5_CACHE = OrderedDict()


def configurations() -> list[dict]:
    """Finite named F1/F2/F3/F8/F9 starting configurations, not results."""
    rows = [{**entry, "kind": "lightgbm", "target": "direct"}
            for entry in available_group_configs() if entry["id"] != "F1-11"]
    rows.append({"id": "F1-11", "kind": "ridge", "target": "direct", "groups": (), "feature_top_k": 10})
    rows += [{"id": f"F2-{target}", "kind": "lightgbm", "target": target,
              "groups": ("slot_1_7d", "rolling")}
             for target in ("direct", "delta", "weekly", "profile", "b5_residual", "log1p")]
    rows += [{"id": f"F3-{kind}", "kind": kind, "target": "direct",
              "groups": ("slot_1_7d", "rolling", "trend")}
             for kind in ("ridge", "lightgbm", "xgboost", "catboost")]
    rows.append({"id": "F4-3-kalman-features", "kind": "lightgbm", "target": "weekly",
                 "groups": ("slot_1_7d", "rolling"), "kalman_features": True})
    rows += [{"id": f"F3-lightgbm-{loss}", "kind": "lightgbm", "target": "direct",
              "groups": ("slot_1_7d", "rolling", "trend"),
              "model_params": {"objective": loss, **({"alpha": .5} if loss == "quantile" else {})}}
             for loss in ("regression_l1", "huber", "quantile")]
    rows.append({"id": "F2-daytype", "kind": "lightgbm", "target": "direct",
                 "groups": ("calendar", "slot_1_7d"), "daytype": True})
    rows += [{"id": "F8-two-stage", "kind": "two_stage", "base_kind": "lightgbm",
              "target": "direct", "groups": ("peak", "rolling")},
             {"id": "F8-peak-weight", "kind": "lightgbm", "target": "direct",
              "groups": ("peak", "rolling"), "peak_weight": 3.0},
             {"id": "F8-continuous-peak", "kind": "lightgbm", "target": "direct",
              "groups": ("peak", "rolling"), "continuous_peak_alpha": 2.0}]
    rows += [{"id": f"F9-{weeks}w", "kind": "lightgbm", "target": "direct",
              "groups": ("slot_1_7d",), "window_weeks": weeks}
             for weeks in (4, 8, 12)]
    rows += [{"id": "F9-recent", "kind": "lightgbm", "target": "direct",
              "groups": ("slot_1_7d",), "half_life_weeks": 4.0},
             {"id": "F9-dedup-drop", "kind": "lightgbm", "target": "direct",
              "groups": ("slot_1_7d",), "dedup": "drop"},
             {"id": "F9-dedup-weight", "kind": "lightgbm", "target": "direct",
              "groups": ("slot_1_7d",), "dedup": "weight"}]
    return rows


def suggest_parameters(trial, kind: str) -> dict:
    """A broad, bounded Optuna space; the runner controls trial count."""
    if kind == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 300, 2400),
                "learning_rate": trial.suggest_float("learning_rate", .01, .1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 12]),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "max_bin": trial.suggest_int("max_bin", 63, 511),
                "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1.),
                "subsample": trial.suggest_float("subsample", .5, 1.),
                "subsample_freq": 1,
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100., log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100., log=True),
                "objective": trial.suggest_categorical("objective", ["regression", "regression_l1", "huber"])}
    if kind == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 300, 2000),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", .01, .1, log=True),
                "subsample": trial.suggest_float("subsample", .5, 1.),
                "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1.),
                "min_child_weight": trial.suggest_float("min_child_weight", 1., 100., log=True),
                "max_bin": trial.suggest_int("max_bin", 64, 512),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100., log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100., log=True)}
    if kind == "catboost":
        return {"iterations": trial.suggest_int("iterations", 300, 2000),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", .01, .1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100., log=True)}
    if kind == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e4, log=True)}
    raise ValueError(f"Unsupported parameter-search kind: {kind}")


def _horizon(context: Mapping, config: Mapping) -> int:
    value = config.get("horizon", context.get("horizon", context.get("summary", {}).get("horizon_quarters")))
    if value is None or int(value) != value or not 4 <= int(value) <= 16:
        raise ValueError("horizon 4..16 must be provided by context or config")
    return int(value)


def _history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or "power" not in history:
        raise ValueError("history must be a power DataFrame")
    if not isinstance(history.index, pd.DatetimeIndex) or history.empty or not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("history needs sorted unique timestamps")
    if history.index.max() >= BOUNDARY:
        raise ValueError("history reaches sealed boundary")
    if not history.index.equals(pd.date_range(history.index.min(), history.index.max(), freq="15min", name=history.index.name)):
        raise ValueError("history needs a complete quarter-hour grid")
    return history


def _role(context: Mapping, role: str, history: pd.DataFrame) -> pd.DatetimeIndex:
    value = context.get(role)
    if value is None:
        if role == "stop":
            return pd.DatetimeIndex([])
        raise ValueError("context.fit is required")
    origins = pd.DatetimeIndex(value)
    if origins.has_duplicates or not origins.is_monotonic_increasing or not origins.isin(history.index).all():
        raise ValueError(f"{role} origins must be sorted, unique, and in history")
    if role == "fit" and origins.empty:
        raise ValueError("fit is empty")
    return origins


def _truth(context: Mapping, origins: pd.DatetimeIndex) -> np.ndarray:
    series = context.get("y")
    if not isinstance(series, pd.Series):
        raise ValueError("context.y must be an origin-indexed Series")
    y = pd.to_numeric(series.reindex(origins), errors="coerce").to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Selected fit/stop targets must be finite")
    return y


def _validate_config(kind: str, cfg: Mapping) -> tuple[str, str, tuple[str, ...]]:
    if kind not in KINDS:
        raise ValueError(f"Unknown regression kind: {kind}")
    target = str(cfg.get("target", "direct"))
    if target not in TARGETS:
        raise ValueError(f"Unknown regression target: {target}")
    if kind == "two_stage" and target != "direct":
        raise ValueError("two_stage uses a direct conditional demand target")
    if kind == "two_stage" and cfg.get("daytype"):
        raise ValueError("daytype and two_stage are separate experiment variants")
    groups = tuple(cfg.get("groups", ()))
    if target == 'profile':
        weeks = int(cfg.get('profile_weeks',4))
        if weeks not in (2,3,4):
            raise ValueError('profile_weeks must be 2, 3 or 4')
        group = f'same_slot_{weeks}w'
        groups = tuple(dict.fromkeys((*groups,group)))
    if any(not isinstance(group, str) for group in groups):
        raise ValueError("groups must contain feature-group names")
    dedup = str(cfg.get("dedup", "keep"))
    if dedup not in {"keep", "drop", "weight"}:
        raise ValueError("dedup must be keep, drop, or weight")
    for name in ("peak_weight", "continuous_peak_alpha"):
        if float(cfg.get(name, 0 if name == "continuous_peak_alpha" else 1)) < 0:
            raise ValueError(f"{name} cannot be negative")
    return target, dedup, groups


def _select_window(fit: pd.DatetimeIndex, weeks: object) -> pd.DatetimeIndex:
    if weeks is None:
        return fit
    value = float(weeks)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("window_weeks must be positive")
    result = fit[fit >= fit.max() - timedelta(days=7 * value)]
    if len(result) < 30:
        raise ValueError("sliding fit window has fewer than 30 target rows")
    return result


def _duplicate_weights(context: Mapping, origins: pd.DatetimeIndex, y: np.ndarray,
                       horizon: int, policy: str) -> np.ndarray:
    """Exact completed target-profile duplicates, identified from FIT only."""
    weights = np.ones(len(origins), dtype=float)
    if policy == "keep":
        return weights
    target_times = context.get("target_time")
    if not isinstance(target_times, pd.Series):
        raise ValueError("dedup needs context.target_time")
    target = pd.DatetimeIndex(target_times.reindex(origins))
    if target.isna().any() or not (target == origins + pd.to_timedelta(horizon * 15, unit="m")).all():
        raise ValueError("target_time inconsistent for fit duplicates")
    # Incomplete days are never asserted identical to another day.
    day = (target - pd.Timedelta(nanoseconds=1)).normalize()
    slot = ((target - day) / pd.Timedelta(minutes=15)).astype(int)
    cells = pd.DataFrame({"day": day, "slot": slot, "y": y, "position": np.arange(len(y))})
    by_day = cells.groupby("day", sort=True)
    seen: dict[tuple[tuple[int, float], ...], list[pd.Timestamp]] = {}
    for label, part in by_day:
        if len(part) != 96 or part.slot.nunique() != 96:
            continue
        signature = tuple(zip(part.sort_values("slot").slot.astype(int),
                              part.sort_values("slot").y.astype(float)))
        seen.setdefault(signature, []).append(label)
    for days in seen.values():
        if len(days) <= 1:
            continue
        for order, label in enumerate(days):
            positions = cells.loc[cells.day.eq(label), "position"].to_numpy(int)
            if policy == "drop" and order > 0:
                weights[positions] = 0
            elif policy == "weight":
                weights[positions] /= len(days)
    return weights


def _weights(context: Mapping, origins: pd.DatetimeIndex, y: np.ndarray,
             tau: float, horizon: int, cfg: Mapping, dedup: str) -> np.ndarray:
    weights = _duplicate_weights(context, origins, y, horizon, dedup)
    weeks = cfg.get("half_life_weeks")
    if weeks is not None:
        half_life = float(weeks)
        if not np.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life_weeks must be positive")
        age_weeks = (origins.max() - origins) / pd.Timedelta(days=7)
        weights *= np.exp2(-np.asarray(age_weeks, dtype=float) / half_life)
    weights *= np.where(y > tau, float(cfg.get("peak_weight", 1.)), 1.)
    alpha = float(cfg.get("continuous_peak_alpha", 0.))
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("continuous_peak_alpha must be finite and nonnegative")
    if alpha > 0:
        # y contains only selected fit labels, on the original demand scale.
        q80 = float(np.quantile(y, .8))
        scale = float(tau) - q80
        if not np.isfinite(scale) or scale <= 0:
            from phase_f.support import UnsupportedConfiguration
            raise UnsupportedConfiguration("Continuous peak weights require fit tau > fit q80",
                                           {"fit_q80": q80, "fit_tau": float(tau),
                                            "continuous_peak_alpha": alpha})
        weights *= 1 + alpha * np.maximum(y - q80, 0) / scale
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("No positive finite training weights")
    return weights


def _features(history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
              tau: float, groups: tuple[str, ...]) -> pd.DataFrame:
    global _FEATURE_CACHE_BYTES
    if origins.empty:
        return pd.DataFrame(index=origins)
    past = history.loc[:origins.max()]
    # Hash observed rows only. A changed or mutated history cannot reuse stale features.
    key=(hashlib.sha256(pd.util.hash_pandas_object(past,index=True).to_numpy().tobytes()).digest(),
         hashlib.sha256(origins.asi8.tobytes()).digest(),horizon,float(tau),tuple(groups))
    if key in _FEATURE_CACHE:
        _FEATURE_CACHE.move_to_end(key)
        return _FEATURE_CACHE[key].copy()
    x, provenance = build_features(past, origins, horizon, tau, groups)
    if any((used > origins).fillna(False).any() for used in provenance.values()):
        raise AssertionError("Noncausal feature provenance")
    if x.columns.duplicated().any():
        raise ValueError("Duplicate feature columns")
    size=int(x.memory_usage(deep=True).sum())
    while _FEATURE_CACHE and _FEATURE_CACHE_BYTES+size>1_500_000_000:
        _,removed=_FEATURE_CACHE.popitem(last=False)
        _FEATURE_CACHE_BYTES-=int(removed.memory_usage(deep=True).sum())
    _FEATURE_CACHE[key]=x.copy()
    _FEATURE_CACHE_BYTES+=size
    return x


def _preprocessor(x: pd.DataFrame, *, scale: bool) -> dict:
    values = x.to_numpy(float)
    finite = np.isfinite(values)
    medians = np.array([float(np.median(values[finite[:, j], j])) if finite[:, j].any() else 0.
                        for j in range(values.shape[1])])
    imputed = np.where(finite, values, medians)
    augmented = np.column_stack([imputed, (~finite).astype(float)])
    mean = augmented.mean(axis=0) if scale else np.zeros(augmented.shape[1])
    std = augmented.std(axis=0) if scale else np.ones(augmented.shape[1])
    std[std <= 1e-12] = 1.
    return {"columns": list(x.columns), "medians": medians,
            "mean": mean, "std": std, "scaled": scale,
            "always_missing": (~finite.any(axis=0)),
            "all_missing": [str(x.columns[j]) for j in range(values.shape[1]) if not finite[:, j].any()]}


def _transform(x: pd.DataFrame, pre: Mapping) -> np.ndarray:
    if list(x.columns) != list(pre["columns"]):
        x = x.loc[:, pre["columns"]]
    values = x.to_numpy(float)
    finite = np.isfinite(values)
    finite[:, np.asarray(pre["always_missing"], dtype=bool)] = False
    augmented = np.column_stack([np.where(finite, values, pre["medians"]), (~finite).astype(float)])
    return (augmented - pre["mean"]) / pre["std"]


def _fit_estimator(kind: str, x: np.ndarray, y: np.ndarray, weight: np.ndarray,
                   params: Mapping, seed: int, stop: tuple[np.ndarray, np.ndarray] | None,
                   patience: int):
    if kind == "ridge":
        model = Ridge(alpha=float(params.get("alpha", 100.)), random_state=seed)
        model.fit(x, y, sample_weight=weight)
        return model
    if kind == "lightgbm":
        if params.get('objective') in ('tweedie','gamma','poisson'):
            fit_negative=int(np.count_nonzero(y<0))
            stop_negative=int(np.count_nonzero(stop[1]<0)) if stop is not None else 0
            if fit_negative or stop_negative:
                from phase_f.support import UnsupportedConfiguration
                raise UnsupportedConfiguration('Nonnegative objective is incompatible with negative transformed labels',
                    {'objective':params['objective'],'fit_negative_labels':fit_negative,
                     'stop_negative_labels':stop_negative,'fit_target_min':float(np.min(y)),
                     'policy':'Do not clip or silently change the locked target transform'})
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is unavailable in this environment") from exc
        opts = {"n_estimators": 700, "learning_rate": .03, "num_leaves": 31,
                "n_jobs": 4, "random_state": seed, "verbosity": -1, **params}
        model = lgb.LGBMRegressor(**opts)
        fit_opts = {"sample_weight": weight}
        if stop is not None and len(stop[0]) >= 2:
            fit_opts.update(eval_set=[stop], eval_metric="l1",
                            callbacks=[lgb.early_stopping(patience, verbose=False)])
        model.fit(x, y, **fit_opts)
        return model
    if kind == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is unavailable in this environment") from exc
        opts = {"n_estimators": 700, "learning_rate": .03, "max_depth": 6,
                "n_jobs": 4, "random_state": seed, "tree_method": "hist", **params}
        if stop is not None and len(stop[0]) >= 2:
            opts["early_stopping_rounds"] = patience
            opts.setdefault("eval_metric", "mae")
        model = xgb.XGBRegressor(**opts)
        fit_opts = {"sample_weight": weight}
        if stop is not None and len(stop[0]) >= 2:
            fit_opts["eval_set"] = [stop]
            fit_opts["verbose"] = False
        model.fit(x, y, **fit_opts)
        return model
    if kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("catboost is unavailable in this environment") from exc
        opts = {"iterations": 700, "learning_rate": .03, "depth": 6,
                "thread_count": 4, "random_seed": seed, "verbose": False, **params}
        opts['allow_writing_files'] = False
        model = CatBoostRegressor(**opts)
        fit_opts = {"sample_weight": weight}
        if stop is not None and len(stop[0]) >= 2:
            fit_opts.update(eval_set=stop, early_stopping_rounds=patience, use_best_model=True)
        model.fit(x, y, **fit_opts)
        return model
    raise ValueError(f"Unknown estimator: {kind}")


def _anchor(target: str, x: pd.DataFrame, b5: np.ndarray | None = None, profile_weeks: int = 4) -> np.ndarray:
    if target in {"direct", "log1p"}:
        return np.zeros(len(x))
    if target == "delta":
        return x["current"].to_numpy(float)
    if target == "weekly":
        return x["slot7d"].to_numpy(float)
    if target == "profile":
        median = x[f'same_slot_{profile_weeks}w_median'].to_numpy(float)
        weekly = x["slot7d"].to_numpy(float)
        return np.where(np.isfinite(median), median, weekly)
    if target == "b5_residual":
        if b5 is None:
            raise ValueError("B5 residual target needs causal B5 predictions")
        return b5
    raise ValueError(f"Unknown target: {target}")


def _kalman_state(bundle: Mapping, power: pd.Series,
                  origins: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Causal B5 posterior deviation mean/variance through each origin."""
    phi, q, r = (float(bundle[name]) for name in ("phi", "q", "r"))
    if not np.isfinite([phi, q, r]).all() or not -0.99 <= phi <= 0.99 or q <= 0 or r <= 0:
        raise ValueError("Invalid fit-only B5 state parameters")
    if origins.empty:
        return np.empty(0), np.empty(0)
    observed = pd.to_numeric(power.loc[:origins.max()], errors="coerce").to_numpy(float)
    positions = power.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("B5 state origin missing from power history")
    state = 0.
    variance = q / (1 - phi * phi)
    means = np.empty(len(origins), dtype=float)
    variances = np.empty(len(origins), dtype=float)
    lookup = {int(position): row for row, position in enumerate(positions)}
    for position in range(int(positions.max()) + 1):
        if position:
            state *= phi
            variance = phi * phi * variance + q
        deviation = observed[position] - observed[position - 672] if position >= 672 else np.nan
        if np.isfinite(deviation):
            gain = variance / (variance + r)
            state += gain * (deviation - state)
            variance *= 1 - gain
        row = lookup.get(position)
        if row is not None:
            means[row] = state
            variances[row] = variance
    return means, variances


def _prequential_b5(history: pd.DataFrame, fit: pd.DatetimeIndex, horizon: int,
                    blocks: int, include_state: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Forward-block out-of-fit B5 labels and optional states, cached per fit."""
    warmup = max(672 + 32, int(np.ceil(len(fit) * .4)))
    if warmup >= len(fit) - 30:
        raise ValueError("B5 residual target needs >734 eligible chronological fit rows")
    blocks = max(2, int(blocks))
    observed_fit_prefix = history.power.loc[:fit.max()]
    key = (hashlib.sha256(pd.util.hash_pandas_object(observed_fit_prefix, index=True).to_numpy().tobytes()).digest(),
           hashlib.sha256(fit.asi8.tobytes()).digest(), horizon, blocks, include_state,
           id(fit_kalman), id(predict_kalman))
    if key in _B5_CACHE:
        _B5_CACHE.move_to_end(key)
        oof, valid, content = _B5_CACHE[key]
        return oof.copy(), valid.copy(), copy.deepcopy(content)
    edges = np.linspace(warmup, len(fit), blocks + 1, dtype=int)
    oof = np.full(len(fit), np.nan)
    state_mean = np.full(len(fit), np.nan) if include_state else None
    state_variance = np.full(len(fit), np.nan) if include_state else None
    power = pd.to_numeric(history.power, errors="coerce")
    cutoffs = []
    for start, end in zip(edges[:-1], edges[1:]):
        if start == end:
            continue
        fitted = fit_kalman(power, fit[:start])
        oof[start:end] = predict_kalman(fitted, power, fit[start:end], horizon)
        if include_state:
            state_mean[start:end], state_variance[start:end] = _kalman_state(fitted, power, fit[start:end])
        cutoffs.append({"fit_end": str(fit[start - 1]), "oof_first": str(fit[start]),
                        "oof_last": str(fit[end - 1])})
    valid = np.isfinite(oof)
    if valid.sum() < 30:
        raise ValueError("Too few finite forward B5 residual labels")
    if include_state:
        valid &= np.isfinite(state_mean) & np.isfinite(state_variance) & (state_variance > 0)
        if valid.sum() < 30:
            raise ValueError("Too few finite forward B5 state features")
    full = fit_kalman(power, fit)
    content = {"final": full, "cutoffs": cutoffs, "warmup_excluded": int((~valid).sum())}
    if include_state:
        content["oof_state_deviation"] = state_mean
        content["oof_state_variance"] = state_variance
    _B5_CACHE[key] = (oof.copy(), valid.copy(), copy.deepcopy(content))
    while len(_B5_CACHE) > 16:
        _B5_CACHE.popitem(last=False)
    return oof, valid, content


def _add_kalman_features(x: pd.DataFrame, prediction: np.ndarray,
                         deviation: np.ndarray, variance: np.ndarray) -> pd.DataFrame:
    result = x.copy()
    if len(prediction) != len(result) or len(deviation) != len(result) or len(variance) != len(result):
        raise ValueError("Kalman feature arrays do not match origins")
    result["b5_forecast"] = prediction
    result["state_deviation"] = deviation
    result["state_variance"] = variance
    return result


def _daytypes(x: pd.DataFrame) -> np.ndarray:
    # Phase B core holiday/weekend are target-calendar features, known ahead.
    holiday = x["holiday"].to_numpy(float) > .5
    weekend = x["weekend"].to_numpy(float) > .5
    return np.where(holiday, 2, np.where(weekend, 1, 0)).astype(int)


def _inner_indices(index, fraction, horizon):
    split = int(len(index)*fraction)
    valid = np.arange(split,len(index))
    train = np.flatnonzero(index+timedelta(minutes=15*horizon)<index[split])
    if len(train)<30 or len(valid)<10:
        raise ValueError('Purged inner split needs >=30 train and >=10 validation rows')
    return train,valid


def _rank_features(x: pd.DataFrame, y: np.ndarray, top_k: int, seed: int, horizon: int = 16) -> tuple[list[str], dict]:
    """One deterministic chronological inner fit/validation permutation rank."""
    if top_k < 1:
        raise ValueError("feature_top_k must be positive")
    if top_k >= len(x.columns):
        return list(x.columns), {"inner_validation_n": 0, "method": "all_columns"}
    train_ix,valid_ix = _inner_indices(x.index,.75,horizon)
    inner = _preprocessor(x.iloc[train_ix], scale=True)
    train, valid = _transform(x.iloc[train_ix], inner), _transform(x.iloc[valid_ix], inner)
    model = Ridge(alpha=100.)
    model.fit(train, y[train_ix])
    base = np.mean(np.abs(model.predict(valid) - y[valid_ix]))
    rng = np.random.default_rng(seed)
    gains = []
    width = len(x.columns)
    for j, name in enumerate(x.columns):
        loss = []
        for _ in range(2):
            order = rng.permutation(len(valid))
            changed = valid.copy()
            changed[:, j] = valid[order, j]
            changed[:, width + j] = valid[order, width + j]
            loss.append(float(np.mean(np.abs(model.predict(changed) - y[valid_ix])) - base))
        gains.append((float(np.mean(loss)), j, str(name)))
    ranking = sorted(gains, key=lambda item: (-item[0], item[1]))
    chosen_set = {name for _, _, name in ranking[:top_k]}
    chosen = [name for name in x.columns if name in chosen_set]
    return chosen, {"inner_validation_n": len(valid), "inner_train_n":len(train_ix),
                    "inner_train_target_max":str(x.index[train_ix[-1]]+timedelta(minutes=15*horizon)),
                    "inner_valid_origin_min":str(x.index[valid_ix[0]]),
                    "method": "2x paired value-and-mask permutation MAE",
                    "ranked": [(name, importance) for importance, _, name in ranking]}


def fit_model(kind: str, history: pd.DataFrame, context: Mapping,
              config: Mapping | None = None) -> dict:
    """Fit one h/f candidate; return a serializable-in-memory model bundle."""
    history = _history(history)
    cfg = dict(config or {})
    target, dedup, groups = _validate_config(kind, cfg)
    horizon = _horizon(context, cfg)
    fit_all = _role(context, "fit", history)
    stop = _role(context, "stop", history)
    if len(stop) and not (fit_all.max() + timedelta(minutes=horizon * 15) < stop.min()):
        raise ValueError("fit-to-stop target embargo is missing")
    fit = _select_window(fit_all, cfg.get("window_weeks"))
    tau = float(context.get("tau", np.nan))
    if not np.isfinite(tau):
        raise ValueError("fit-only tau is required")
    y = _truth(context, fit)
    x = _features(history, fit, horizon, tau, groups)
    x_stop = _features(history, stop, horizon, tau, groups) if len(stop) else None
    y_stop = _truth(context, stop) if len(stop) else None
    b5 = None
    fit_keep = np.ones(len(fit), dtype=bool)
    kalman_features = bool(cfg.get("kalman_features", False))
    if target == "b5_residual" or kalman_features:
        b5_oof, b5_valid, b5 = _prequential_b5(
            history, fit, horizon, int(cfg.get("b5_oof_blocks", 4)), include_state=kalman_features)
        fit_keep &= b5_valid
        if x_stop is not None:
            stop_b5 = predict_kalman(b5["final"], history.power, stop, horizon)
        else:
            stop_b5 = None
        if kalman_features:
            x = _add_kalman_features(x, b5_oof, b5["oof_state_deviation"], b5["oof_state_variance"])
            if x_stop is not None:
                stop_state, stop_variance = _kalman_state(b5["final"], history.power, stop)
                x_stop = _add_kalman_features(x_stop, stop_b5, stop_state, stop_variance)
    if target == "b5_residual":
        anchor_fit = _anchor(target, x, b5_oof)
        anchor_stop = _anchor(target, x_stop, stop_b5) if x_stop is not None else None
    else:
        anchor_fit = _anchor(target, x, profile_weeks=int(cfg.get('profile_weeks',4)))
        anchor_stop = _anchor(target, x_stop, profile_weeks=int(cfg.get('profile_weeks',4))) if x_stop is not None else None
    fit_keep &= np.isfinite(anchor_fit)
    if not fit_keep.any():
        raise ValueError("No fit rows with a causal target anchor")
    if anchor_stop is not None and not np.isfinite(anchor_stop).all():
        raise ValueError("Stop target anchors are incomplete")
    fit_used = fit[fit_keep]
    x = x.iloc[fit_keep].copy()
    y = y[fit_keep]
    anchor_fit = anchor_fit[fit_keep]
    weights = _weights(context, fit_used, y, tau, horizon, cfg, dedup)
    if dedup == "drop":
        keep = weights > 0
        fit_used, x, y, anchor_fit, weights = fit_used[keep], x.iloc[keep], y[keep], anchor_fit[keep], weights[keep]
    if len(y) < 30:
        raise ValueError("Fewer than 30 effective fit labels")
    if target == "log1p":
        if np.any(y < 0) or (y_stop is not None and np.any(y_stop < 0)):
            raise ValueError("log1p demand target requires nonnegative values")
        z = np.log1p(y)
        stop_z = np.log1p(y_stop) if y_stop is not None else None
    else:
        z = y - anchor_fit
        stop_z = y_stop - anchor_stop if y_stop is not None else None
    daytype_fit = _daytypes(x) if cfg.get("daytype") else None
    daytype_stop = _daytypes(x_stop) if cfg.get("daytype") and x_stop is not None else None
    top_k = cfg.get("feature_top_k")
    rank = None
    if top_k is not None:
        columns, rank = _rank_features(x, z, int(top_k), int(cfg.get("seed", 42)),horizon)
        x = x.loc[:, columns]
        if x_stop is not None:
            x_stop = x_stop.loc[:, columns]
    base_kind = str(cfg.get("base_kind", "lightgbm")) if kind == "two_stage" else kind
    if base_kind not in KINDS - {"two_stage"}:
        raise ValueError(f"Unknown base_kind: {base_kind}")
    pre = _preprocessor(x, scale=(base_kind == "ridge"))
    matrix = _transform(x, pre)
    stop_matrix = _transform(x_stop, pre) if x_stop is not None else None
    stop_data = (stop_matrix, stop_z) if stop_matrix is not None else None
    params = dict(cfg.get("model_params", {}))
    seed = int(cfg.get("seed", 42))
    patience = int(cfg.get("early_stopping_rounds", 50))
    if patience < 1:
        raise ValueError("early_stopping_rounds must be positive")
    if kind == "two_stage":
        labels = y > tau
        if labels.sum() < 2 or (~labels).sum() < 2:
            raise ValueError("two_stage needs both fit classes with at least two observations")
        if labels.sum() < 20 or (~labels).sum() < 20:
            raise ValueError("two_stage needs >=20 peak and >=20 nonpeak fit rows")
        from lightgbm import LGBMClassifier, early_stopping
        classifier = LGBMClassifier(n_estimators=700, learning_rate=.03, num_leaves=15,
                                    n_jobs=4, random_state=seed, verbosity=-1)
        classifier_options = {'sample_weight':weights}
        if stop_matrix is not None:
            classifier_options.update(eval_set=[(stop_matrix,(y_stop>tau).astype(int))],
                                      callbacks=[early_stopping(patience,verbose=False)])
        classifier.fit(matrix, labels.astype(int), **classifier_options)
        regressors = {}
        for state in (False, True):
            selected = labels == state
            regressors[state] = _fit_estimator(base_kind, matrix[selected], y[selected], weights[selected],
                                                params, seed, None, patience)
        model = {"classifier": classifier, "regressors": regressors}
        bias = 0.
    else:
        global_model = _fit_estimator(base_kind, matrix, z, weights, params, seed, stop_data, patience)
        if cfg.get("daytype"):
            by_daytype = {}
            classes = daytype_fit
            stop_classes = daytype_stop
            for state in (0, 1, 2):
                selected = classes == state
                if selected.sum() < int(cfg.get("daytype_min_fit", 30)):
                    continue
                state_stop = (stop_matrix[stop_classes == state], stop_z[stop_classes == state]) if stop_classes is not None and np.any(stop_classes == state) else None
                by_daytype[state] = _fit_estimator(base_kind, matrix[selected], z[selected], weights[selected],
                                                   params, seed, state_stop, patience)
            model = {"global": global_model, "by_daytype": by_daytype}
        else:
            model = global_model
        bias = 0.
        if target == "log1p":
            # Fit-only inner chronology estimates retransformation bias; the
            # final model is still trained on all selected fit rows.
            train_ix,valid_ix = _inner_indices(x.index,.8,horizon)
            inner_pre = _preprocessor(x.iloc[train_ix], scale=(base_kind == "ridge"))
            inner_train = _transform(x.iloc[train_ix], inner_pre)
            inner_valid = _transform(x.iloc[valid_ix], inner_pre)
            inner = _fit_estimator(base_kind, inner_train, z[train_ix], weights[train_ix],
                                   params, seed, None, patience)
            bias = float(np.mean(y[valid_ix] - np.expm1(inner.predict(inner_valid))))
    bundle = {"kind": kind, "base_kind": base_kind, "target": target, "horizon": horizon,
            "tau": tau, "groups": groups, "config": cfg, "preprocessor": pre,
            "model": model, "bias": bias, "b5": b5, "feature_rank": rank,
            "kalman_features": kalman_features,
            "b5_oof_cutoffs": b5["cutoffs"] if b5 is not None else [],
            "fit_first": fit_used.min(), "fit_end": fit_used.max(),
            "n_fit_labels": len(y), "n_original_fit": len(fit_all),
            "stop_used": bool(stop_data is not None and (base_kind != "ridge" or kind == "two_stage")),
            "daytype": bool(cfg.get("daytype", False)),
            "daytype_trained": sorted(model["by_daytype"]) if cfg.get("daytype") else [],
            "development_only": True, "fit_parameters_only": True}
    if len(stop):
        stop_prediction = predict_model(bundle, history, stop, horizon)
        bundle["stop_mae"] = float(np.mean(np.abs(stop_prediction - y_stop)))
        bundle["stop_n"] = len(stop)
    else:
        bundle["stop_mae"] = np.nan
        bundle["stop_n"] = 0
    return bundle


def predict_model(bundle: Mapping, history: pd.DataFrame, origins: pd.DatetimeIndex,
                  horizon: int, *, return_details: bool = False):
    """Predict one target per origin using only power observed through origin."""
    history = _history(history)
    origins = pd.DatetimeIndex(origins)
    if horizon != bundle["horizon"] or not 4 <= horizon <= 16:
        raise ValueError("Prediction horizon differs from fitted bundle")
    if origins.has_duplicates or not origins.is_monotonic_increasing or not origins.isin(history.index).all():
        raise ValueError("origins must be unique, chronological, and in history")
    if len(origins) == 0:
        empty = np.empty(0, dtype=float)
        return {"pred": empty, "p_peak": empty} if return_details else empty
    x = _features(history, origins, horizon, float(bundle["tau"]), tuple(bundle["groups"]))
    b5_prediction = None
    if bundle.get("kalman_features"):
        b5_prediction = predict_kalman(bundle["b5"]["final"], history.power, origins, horizon)
        state, variance = _kalman_state(bundle["b5"]["final"], history.power, origins)
        x = _add_kalman_features(x, b5_prediction, state, variance)
    classes = _daytypes(x) if bundle.get("daytype") else None
    anchor_features = x
    x = x.loc[:, bundle["preprocessor"]["columns"]]
    matrix = _transform(x, bundle["preprocessor"])
    if bundle["kind"] == "two_stage":
        model = bundle["model"]
        p = model["classifier"].predict_proba(matrix)[:, 1]
        ordinary = model["regressors"][False].predict(matrix)
        peaks = model["regressors"][True].predict(matrix)
        prediction = (1 - p) * ordinary + p * peaks
    else:
        if bundle.get("daytype"):
            fitted = bundle["model"]
            estimate = fitted["global"].predict(matrix)
            for state, local in fitted["by_daytype"].items():
                selected = classes == state
                if np.any(selected):
                    estimate[selected] = local.predict(matrix[selected])
        else:
            estimate = bundle["model"].predict(matrix)
        target = bundle["target"]
        if target == "log1p":
            prediction = np.expm1(estimate) + float(bundle["bias"])
        else:
            b5 = (b5_prediction if b5_prediction is not None else
                  predict_kalman(bundle["b5"]["final"], history.power, origins, horizon)) if target == "b5_residual" else None
            prediction = _anchor(target, anchor_features, b5, int(bundle['config'].get('profile_weeks',4))) + estimate
        p = np.full(len(origins), np.nan)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Candidate returned nonfinite predictions on fixed cohort")
    return {"pred": prediction, "p_peak": np.asarray(p, float)} if return_details else prediction
