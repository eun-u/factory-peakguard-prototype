"""Peak exceedance probabilities inferred from calibrated quantile knots."""
from __future__ import annotations

import numpy as np


def exceedance_from_quantiles(quantiles: dict[float, np.ndarray], threshold: float):
    """Piecewise-linear conditional CDF; tails use terminal knot mass."""
    levels = np.array(sorted(quantiles), dtype=float)
    knots = np.column_stack([np.asarray(quantiles[level], dtype=float) for level in levels])
    knots.sort(axis=1)
    result = np.empty(len(knots), dtype=float)
    for i, row in enumerate(knots):
        unique, inverse = np.unique(row, return_inverse=True)
        # Ties share the highest observed CDF level.
        cdf = np.array([levels[inverse == j].max() for j in range(len(unique))])
        # Outside fitted knots, do not assert an impossible event: the upper
        # terminal probability remains at least 1-max(alpha).
        result[i] = 1.0 - np.interp(threshold, unique, cdf, left=0.0, right=float(levels[-1]))
    return np.clip(result, 0.0, 1.0)


def fit_classifier(x_train, y_train, tau, x_cal, y_cal, cfg):
    """Optional direct classifier, calibrated on a disjoint later block."""
    from sklearn.isotonic import IsotonicRegression
    from .lgbm_point import _lightgbm
    labels = np.asarray(y_train) > tau
    if labels.sum() < 30 or (~labels).sum() < 30:
        return None
    settings = cfg.get("lgbm", {})
    model = _lightgbm().LGBMClassifier(
        num_leaves=int(settings.get("num_leaves", 31)),
        n_estimators=int(settings.get("n_estimators", 400)),
        learning_rate=float(settings.get("learning_rate", .05)),
        min_child_samples=int(settings.get("min_child_samples", 40)),
        random_state=int(cfg.get("seed", 42)), n_jobs=int(settings.get("n_jobs", 2)), verbosity=-1,
    ).fit(x_train, labels)
    raw = model.predict_proba(x_cal)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(raw, np.asarray(y_cal) > tau)
    return model, cal


def predict_classifier(fitted, x):
    model, calibration = fitted
    return calibration.predict(model.predict_proba(x)[:, 1])
