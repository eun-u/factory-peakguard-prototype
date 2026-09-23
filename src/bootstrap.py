"""Paired target-date bootstrap intervals for model comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_bootstrap(frame: pd.DataFrame, statistic, n=1000, seed=42):
    """Recompute a scalar statistic after sampling whole target dates."""
    dates = pd.to_datetime(frame["target_time"]).dt.normalize().to_numpy()
    unique = np.unique(dates)
    if len(unique) == 0:
        return (float("nan"), float("nan"))
    positions = [np.flatnonzero(dates == date) for date in unique]
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=float)
    for i in range(n):
        chosen = rng.integers(len(unique), size=len(unique))
        sample = frame.iloc[np.concatenate([positions[k] for k in chosen])]
        values[i] = statistic(sample)
    finite = values[np.isfinite(values)]
    return tuple(float(v) for v in np.quantile(finite, [.025, .975])) if len(finite) else (float("nan"), float("nan"))


def day_mean_ci(frame: pd.DataFrame, values, n=1000, seed=42):
    """Exact day-block resampling of a mean using daily sums and counts."""
    dates = pd.to_datetime(frame["target_time"]).dt.normalize().to_numpy()
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) != len(dates):
        raise ValueError("One value is required per prediction row")
    daily = pd.DataFrame({"date": dates, "value": series})
    grouped = daily.groupby("date", sort=True).value.agg(["sum", "count"])
    if grouped.empty:
        return (float("nan"), float("nan"))
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(len(sums), size=(n, len(sums)))
    denominator = counts[chosen].sum(axis=1)
    estimates = np.divide(sums[chosen].sum(axis=1), denominator,
                          out=np.full(n, np.nan), where=denominator > 0)
    finite = estimates[np.isfinite(estimates)]
    return tuple(float(v) for v in np.quantile(finite, [.025, .975])) if len(finite) else (float("nan"), float("nan"))


def paired_mae_improvement(frame_a, frame_b, *, peak_only=False, n=1000, seed=42):
    keys = ["target_time", "horizon", "fold"]
    a = frame_a[keys + ["y", "pred", "tau"]].rename(columns={"pred": "pred_a"})
    b = frame_b[keys + ["pred"]].rename(columns={"pred": "pred_b"})
    merged = a.merge(b, on=keys, validate="one_to_one")
    merged = merged.loc[np.isfinite(merged.y) & np.isfinite(merged.pred_a) & np.isfinite(merged.pred_b)]
    if peak_only:
        merged = merged.loc[merged.y > merged.tau]
    if merged.empty:
        return {"estimate": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    def improvement(part):
        return float(np.mean(np.abs(part.y - part.pred_a) - np.abs(part.y - part.pred_b)))
    differences = np.abs(merged.y.to_numpy() - merged.pred_a.to_numpy()) - np.abs(merged.y.to_numpy() - merged.pred_b.to_numpy())
    return {"estimate": improvement(merged), "ci95": list(day_mean_ci(merged, differences, n=n, seed=seed)), "n": len(merged)}
