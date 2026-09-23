"""One-sided split conformal corrections for upper conditional quantiles."""
from __future__ import annotations

import math
import numpy as np


def _finite_correction(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = len(scores)
    if n == 0:
        raise ValueError("Conformal calibration requires nonempty finite scores")
    rank = min(n, math.ceil((n + 1) * float(alpha)))
    return float(scores[rank - 1])


def fit_conformal(y_cal, q_cal, alpha, bins=None):
    """Fit y-q correction. bins is an integer category per calibration row."""
    y, q = np.asarray(y_cal, dtype=float), np.asarray(q_cal, dtype=float)
    if len(y) != len(q):
        raise ValueError("Calibration arrays must have equal lengths")
    global_delta = _finite_correction(y - q, alpha)
    result = {"alpha": float(alpha), "global": global_delta, "by_bin": {}}
    if bins is not None:
        categories = np.asarray(bins)
        if len(categories) != len(y):
            raise ValueError("Calibration bin count must equal label count")
        for cat in np.unique(categories):
            mask = categories == cat
            # Sparse bins use the global correction and remain explicit in metadata.
            if mask.sum() >= 30:
                result["by_bin"][str(cat)] = _finite_correction((y - q)[mask], alpha)
    return result


def apply_conformal(q_pred, params, bins=None):
    values = np.asarray(q_pred, dtype=float)
    if bins is None:
        return values + float(params["global"])
    categories = np.asarray(bins)
    if len(values) != len(categories):
        raise ValueError("Prediction bin count must equal prediction count")
    corrections = np.array([params["by_bin"].get(str(cat), params["global"]) for cat in categories], dtype=float)
    return values + corrections


def fit_mondrian_edges(q50_cal, quantiles=(.5, .9)):
    return np.quantile(np.asarray(q50_cal, dtype=float), quantiles).tolist()


def assign_mondrian_bins(q50, edges):
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(q50, dtype=float), side="right")
