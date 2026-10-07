"""Forecast targets and train-only peak definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_targets(df: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    origins = pd.DatetimeIndex(origins, name="origin")
    target_times = origins + pd.Timedelta(minutes=15 * horizon)
    data = df.reindex(target_times)
    result = pd.DataFrame({
        "target_time": target_times,
        "y": pd.to_numeric(data["power"], errors="coerce").to_numpy(),
        "target_repaired": data.get("time_repaired", pd.Series(False, index=data.index)).fillna(True).astype(bool).to_numpy(),
    }, index=origins)
    result["valid_target"] = result["y"].notna() & ~result["target_repaired"]
    return result


def next_day_max_targets(df: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """At 23:45 predict the 96 intervals ending during the following production day."""
    origins = pd.DatetimeIndex(origins, name="origin")
    if not ((origins.hour == 23) & (origins.minute == 45)).all():
        raise ValueError("T2 origins must be 23:45")
    rows = []
    for origin in origins:
        # Interval ending at midnight belongs to the preceding production day.
        next_date = origin.normalize() + pd.Timedelta(days=1)
        ends = pd.date_range(next_date + pd.Timedelta(minutes=15), periods=96, freq="15min")
        sample = df.reindex(ends)
        repaired = sample.get("time_repaired", pd.Series(False, index=sample.index)).fillna(True).astype(bool)
        valid = len(sample) == 96 and sample["power"].notna().all() and not repaired.any()
        rows.append({"target_date": next_date, "target_time": ends[-1], "y": float(sample["power"].max()) if valid else np.nan, "valid_target": valid})
    return pd.DataFrame(rows, index=origins)


def training_peak_threshold(df: pd.DataFrame, train_origins: pd.DatetimeIndex, quantile: float = .95) -> float:
    """Estimate τ from observations in the fold's training period only."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly within (0, 1)")
    train_origins = pd.DatetimeIndex(train_origins)
    if len(train_origins) == 0:
        raise ValueError("empty training origins")
    rows = df.loc[df.index <= train_origins.max()]
    if "time_repaired" in rows:
        rows = rows.loc[~rows["time_repaired"].fillna(True).astype(bool)]
    power = pd.to_numeric(rows["power"], errors="coerce").dropna()
    if power.empty:
        raise ValueError("no clean training power")
    return float(power.quantile(quantile))
