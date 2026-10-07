"""Conditional exceedance size, estimated only from past peak cases."""
from __future__ import annotations

import numpy as np
from .lgbm_point import _lightgbm


def fit_exceedance(x_train, y_train, tau, cfg):
    y = np.asarray(y_train, dtype=float)
    peak = y > tau
    if peak.sum() < 100:
        return {"method": "mean", "value": float(np.mean(y[peak] - tau)) if peak.any() else 0.0,
                "cases": int(peak.sum())}
    settings = cfg.get("lgbm", {})
    model = _lightgbm().LGBMRegressor(
        num_leaves=15, min_child_samples=20,
        n_estimators=min(200, int(settings.get("n_estimators", 400))),
        learning_rate=float(settings.get("learning_rate", .05)),
        random_state=int(cfg.get("seed", 42)), n_jobs=int(settings.get("n_jobs", 2)), verbosity=-1,
    ).fit(x_train.iloc[np.flatnonzero(peak)], y[peak] - tau)
    return {"method": "lgbm", "model": model, "cases": int(peak.sum())}


def predict_exceedance(fitted, x, p_exceed):
    m = fitted["model"].predict(x) if fitted["method"] == "lgbm" else fitted["value"]
    return np.maximum(np.asarray(m, dtype=float), 0.0) * np.asarray(p_exceed, dtype=float)
