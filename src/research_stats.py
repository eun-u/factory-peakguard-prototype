"""Fixed-family multiplicity control for the preregistered development studies."""
from __future__ import annotations

import numpy as np


def holm_adjust(p_values, family_size=None):
    """Holm step-down adjusted p values; unavailable tests conservatively equal 1.

    family_size may exceed the supplied count for explicitly unexecuted tests,
    but may never be smaller. This prevents silently dropping failed analyses.
    """
    raw = np.asarray(p_values, dtype=float)
    if raw.ndim != 1:
        raise ValueError("p values must be one dimensional")
    size = len(raw) if family_size is None else int(family_size)
    if size < len(raw):
        raise ValueError("Family size cannot omit supplied hypotheses")
    if ((raw[np.isfinite(raw)] < 0) | (raw[np.isfinite(raw)] > 1)).any():
        raise ValueError("p values must be in [0,1]")
    p = np.where(np.isfinite(raw), raw, 1.0)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * (size - np.arange(len(p)))))
    return adjusted
