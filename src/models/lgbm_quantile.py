"""Direct conditional quantiles; crossing is removed at inference."""
from __future__ import annotations

import numpy as np
from .lgbm_point import _lightgbm


def fit_quantiles(x_train, y_train, x_stop, y_stop, cfg, alphas=None, params=None):
    lgb = _lightgbm()
    settings = cfg.get("lgbm", {})
    params = params or {}
    alphas = list(alphas or cfg.get("quantiles", [.1, .5, .9, .95, .975]))
    out = {}
    for alpha in alphas:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=float(alpha),
            num_leaves=int(params.get("num_leaves", settings.get("num_leaves", 31))),
            min_child_samples=int(params.get("min_child_samples", params.get("min_data_in_leaf", settings.get("min_child_samples", 40)))),
            learning_rate=float(params.get("learning_rate", settings.get("learning_rate", .05))),
            n_estimators=int(params.get("n_estimators", settings.get("n_estimators", 400))),
            random_state=int(cfg.get("seed", 42)), n_jobs=int(settings.get("n_jobs", 2)),
            verbosity=-1,
        )
        kwargs = {}
        if len(x_stop):
            kwargs = {"eval_set": [(x_stop, y_stop)], "eval_metric": "quantile",
                      "callbacks": [lgb.early_stopping(int(settings.get("early_stopping", 50)), verbose=False)]}
        model.fit(x_train, y_train, **kwargs)
        out[float(alpha)] = model
    return out


def predict_quantiles(models, x):
    alphas = sorted(models)
    arr = np.column_stack([models[a].predict(x) for a in alphas])
    arr.sort(axis=1)
    return {a: arr[:, i] for i, a in enumerate(alphas)}
