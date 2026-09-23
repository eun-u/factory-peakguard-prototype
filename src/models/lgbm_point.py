"""Direct LightGBM point forecasts with an isolated early-stop window."""
from __future__ import annotations

import numpy as np


def _lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to train the LightGBM models") from exc
    return lgb


def fit_point(x_train, y_train, x_stop, y_stop, cfg, *, peak_threshold=None,
              peak_weight=1.0, params=None):
    """Fit on train only; early-stop labels never enter calibration or scoring."""
    lgb = _lightgbm()
    settings = cfg.get("lgbm", {})
    options = dict(params or {})
    model = lgb.LGBMRegressor(
        objective="regression", num_leaves=int(options.get("num_leaves", settings.get("num_leaves", 31))),
        min_child_samples=int(options.get("min_child_samples", options.get("min_data_in_leaf", settings.get("min_child_samples", 40)))),
        learning_rate=float(options.get("learning_rate", settings.get("learning_rate", .05))),
        n_estimators=int(options.get("n_estimators", settings.get("n_estimators", 400))),
        random_state=int(cfg.get("seed", 42)), n_jobs=int(settings.get("n_jobs", 2)),
        verbosity=-1,
    )
    weights = None
    if peak_threshold is not None and peak_weight != 1:
        weights = np.where(np.asarray(y_train, dtype=float) > peak_threshold, peak_weight, 1.0)
    callbacks = [lgb.early_stopping(int(settings.get("early_stopping", 50)), verbose=False)] if len(x_stop) else []
    kwargs = {"sample_weight": weights} if weights is not None else {}
    if len(x_stop):
        kwargs.update(eval_set=[(x_stop, y_stop)], eval_metric="l1", callbacks=callbacks)
    model.fit(x_train, y_train, **kwargs)
    return model


def point_grid(cfg):
    """A small, deterministic search; values can be overridden in YAML."""
    raw = cfg.get("lgbm", {}).get("grid")
    if raw:
        return list(raw)
    return [{"num_leaves": 15, "min_data_in_leaf": 40},
            {"num_leaves": 31, "min_data_in_leaf": 40}]
