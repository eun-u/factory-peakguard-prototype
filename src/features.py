"""Origin-available forecast features, with an explicit latest-used timestamp."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.holidays import calendar_flags
from src.models.cbl import cbl_predict


def build_features(
    df: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int, cfg: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and per-origin latest observed timestamp.

    The origin is *immediately after* the 15-minute interval ending at ``t``;
    thus a reading stamped ``t`` is available. All measured dependencies have
    stamps <= t. Target-time calendar fields are known in advance. The validity
    mask in ``X.attrs['valid_mask']`` excludes repaired or missing dependencies.
    """
    if horizon < 1 or horizon > 96:
        raise ValueError("Supported horizons are 1..96 quarter-hours")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("df must have a unique, sorted ts_end index")
    origins = pd.DatetimeIndex(origins, name="origin")
    if origins.has_duplicates:
        raise ValueError("Duplicate origins")
    if not origins.isin(df.index).all():
        raise ValueError("All origins must be on the observed 15-minute grid")
    target = origins + pd.Timedelta(minutes=15 * horizon)
    power = pd.to_numeric(df["power"], errors="coerce")
    repaired = df.get("time_repaired", pd.Series(False, index=df.index)).fillna(True).astype(bool)
    safe_power = power.mask(repaired)
    x = pd.DataFrame(index=origins)
    latest: dict[str, pd.Series] = {}

    def at(times: pd.DatetimeIndex) -> np.ndarray:
        return safe_power.reindex(times).to_numpy(dtype=float)

    # Power lags are known when their own intervals close.
    for lag in (0, 1, 2, 3, 4, 5, 6, 7, 96, 672):
        name = "current" if lag == 0 else f"lag_{lag}"
        x[name] = at(origins - pd.Timedelta(minutes=15 * lag))
        latest[name] = pd.Series(origins - pd.Timedelta(minutes=15 * lag), index=origins)
    for days in (1, 7):
        # Target's same 15-minute slot on a previous date, used only if it is
        # before the origin; h<=96 makes the day lag nonnegative.
        times = target - pd.Timedelta(days=days)
        name = f"target_slot_{days}d_ago"
        x[name] = at(times)
        latest[name] = pd.Series(times, index=origins)

    for window in (1, 4, 16, 96):
        lookback = np.column_stack([at(origins - pd.Timedelta(minutes=15 * i)) for i in range(window)])
        complete = np.isfinite(lookback).all(axis=1)
        for stat, value in (("mean", np.mean(lookback, axis=1)), ("max", np.max(lookback, axis=1)), ("std", np.std(lookback, axis=1))):
            x[f"recent_{window}_{stat}"] = np.where(complete, value, np.nan)
            latest[f"recent_{window}_{stat}"] = pd.Series(origins, index=origins)
    x["recent_4_slope"] = (x["current"] - x["lag_3"]) / 3
    latest["recent_4_slope"] = pd.Series(origins, index=origins)

    # Peak history uses a fold-specific τ supplied by the caller. A missing τ
    # disables these features; it never estimates τ from future data.
    tau = cfg.get("_tau")
    if tau is not None and np.isfinite(tau):
        bad = safe_power.isna()
        peak = safe_power.gt(float(tau)).astype(float)
        peak[bad] = np.nan
        peak_runs = peak.rolling(672, min_periods=672).sum()
        x["peak_count_7d"] = peak_runs.reindex(origins).to_numpy(dtype=float)
        last_peak = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        last_peak.loc[peak.eq(1)] = df.index[peak.eq(1)]
        last_peak = last_peak.ffill().reindex(origins)
        x["since_last_peak_intervals"] = ((origins - pd.DatetimeIndex(last_peak)) / pd.Timedelta(minutes=15)).to_numpy(dtype=float)
        x.loc[peak_runs.reindex(origins).isna().to_numpy(), "since_last_peak_intervals"] = np.nan
        x["since_last_peak_intervals"] = x["since_last_peak_intervals"].fillna(672).clip(upper=672)
        latest["peak_count_7d"] = latest["since_last_peak_intervals"] = pd.Series(origins, index=origins)

    # Target calendar is fixed before the origin. It uses no future reading.
    calendar = calendar_flags(target)
    for name, values in {
        "target_hour": target.hour, "target_quarter": target.minute // 15,
        "target_weekday": target.dayofweek, "target_month": target.month,
    }.items():
        x[name] = values
        latest[name] = pd.Series(pd.NaT, index=origins, dtype="datetime64[ns]")
    for name in ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day"):
        x[name] = calendar[name].to_numpy(dtype=int)
        latest[name] = pd.Series(pd.NaT, index=origins, dtype="datetime64[ns]")

    # Production is known only at each completed hour. The same-hour raw
    # production_target is deliberately absent from forecast features.
    completed = pd.to_numeric(df.get("production_completed", pd.Series(np.nan, index=df.index)), errors="coerce")
    completed_bad = df.get("production_completed_bad", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    completed = completed.mask(completed_bad)
    for back in (0, 1, 2):
        end = origins.floor("h") - pd.Timedelta(hours=back)
        name = f"production_completed_{back + 1}h"
        x[name] = completed.reindex(end).to_numpy(dtype=float)
        latest[name] = pd.Series(end, index=origins)
    # The preceding calendar day's total can be used after its final hour
    # has completed at midnight. This never uses the unfinished origin day.
    complete_hour_dates = (completed.index - pd.Timedelta(nanoseconds=1)).normalize()
    daily = completed.groupby(complete_hour_dates).agg(["sum", "count"])
    daily.loc[daily["count"] != 24, "sum"] = np.nan
    previous_date = origins.normalize() - pd.Timedelta(days=1)
    x["production_previous_day_total"] = daily["sum"].reindex(previous_date).to_numpy(dtype=float)
    latest["production_previous_day_total"] = pd.Series(origins.normalize(), index=origins)
    prod_known = pd.to_numeric(df.get("production_known", pd.Series(np.nan, index=df.index)), errors="coerce")
    prod_known = prod_known.mask(df.get("production_known_bad", pd.Series(False, index=df.index)).fillna(True).astype(bool))
    historical_prod = np.column_stack([prod_known.reindex(origins - pd.Timedelta(days=i)).to_numpy(dtype=float) for i in range(1, 8)])
    x["production_same_slot_7d_mean"] = np.where(np.isfinite(historical_prod).all(axis=1), historical_prod.mean(axis=1), np.nan)
    latest["production_same_slot_7d_mean"] = pd.Series(origins - pd.Timedelta(days=1), index=origins)

    # CBL-style forecasts use only completed prior reference days and an
    # adjustment window ending at the origin. Missing history stays missing.
    if cfg.get("_include_cbl", True):
        x["cbl_mid_6_10"] = cbl_predict(df, target, "mid_6_10", False, origins).to_numpy(dtype=float)
        x["cbl_mid_6_10_adjusted"] = cbl_predict(df, target, "mid_6_10", True, origins).to_numpy(dtype=float)
        latest["cbl_mid_6_10"] = latest["cbl_mid_6_10_adjusted"] = pd.Series(origins, index=origins)

    # A source tariff field is never read at target time. Only a known schedule
    # supplied in cfg can add this feature.
    tariff_schedule = cfg.get("tariff_2021_hour_bands")
    if tariff_schedule is not None:
        x["tariff_2021_band"] = [tariff_schedule[int(hour)] for hour in target.hour]
        latest["tariff_2021_band"] = pd.Series(pd.NaT, index=origins, dtype="datetime64[ns]")

    max_used = pd.Series(origins, index=origins, name="max_used_time")
    if any((used > origins).fillna(False).any() for used in latest.values()):
        raise AssertionError("A forecast feature uses an observation after its origin")
    valid = x.notna().all(axis=1)
    x.attrs["valid_mask"] = valid
    x.attrs["latest_observation_by_feature"] = latest
    x.attrs["calendar_features_preknown"] = True
    return x, max_used
