"""Persistence and seasonal naive predictions available at each origin."""

from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_predictions(
    df: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int, cfg: dict | None = None,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    origins = pd.DatetimeIndex(origins)
    power = df["power"].where(~df.get("time_repaired", pd.Series(False, index=df.index)).astype(bool))
    target = origins + pd.Timedelta(minutes=15 * horizon)
    value = lambda times: power.reindex(times).to_numpy(dtype=float)
    recent = np.column_stack([value(origins - pd.Timedelta(minutes=15 * i)) for i in range(4)])
    # If any constituent is unavailable, a mean baseline is unavailable.
    mean = np.where(np.isfinite(recent).all(axis=1), recent.mean(axis=1), np.nan)
    yesterday = value(target - pd.Timedelta(days=1)) if horizon <= 96 else np.full(len(origins), np.nan)
    last_week = value(target - pd.Timedelta(days=7)) if horizon <= 672 else np.full(len(origins), np.nan)
    return pd.DataFrame({
        "p1_latest": recent[:, 0],
        "p2_hour_slot": value(origins - pd.Timedelta(hours=1)),
        "p3_recent_mean": mean,
        "s1_day": yesterday,
        "s2_week": last_week,
        "s3_day_week_mean": np.where(np.isfinite(yesterday) & np.isfinite(last_week), (yesterday + last_week) / 2, np.nan),
    }, index=origins)
