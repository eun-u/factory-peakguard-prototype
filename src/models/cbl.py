"""Auditable CBL-style reference load estimates (not official DR settlement)."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pandas as pd

from src.holidays import HOLIDAYS_2021

METHODS = ("mid_6_10", "max_4_5", "holiday_mid_4_6")


def _interval_day(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(nanoseconds=1)).normalize()


def _eligible_ref_day(day: pd.Timestamp, method: str) -> bool:
    holiday = day.strftime("%Y-%m-%d") in HOLIDAYS_2021
    if method == "holiday_mid_4_6":
        # Saturdays and legal holidays; Sundays are included as legal holidays.
        return holiday or day.dayofweek >= 5
    return day.dayofweek < 5 and not holiday


def _reference_values(
    values_by_time: dict, eligible_days: list[pd.Timestamp], target: pd.Timestamp,
    origin: pd.Timestamp, method: str,
) -> tuple[list[pd.Timestamp], np.ndarray] | None:
    day = _interval_day(target)
    offset = target - day
    n = 6 if method == "holiday_mid_4_6" else 5 if method == "max_4_5" else 10
    values: list[float] = []
    days: list[pd.Timestamp] = []
    latest_complete = origin.normalize() - pd.Timedelta(days=1)
    latest_ref = min(day - pd.Timedelta(days=1), latest_complete)
    i = bisect_right(eligible_days, latest_ref) - 1
    while i >= 0 and len(values) < n:
        candidate = eligible_days[i]
        val = values_by_time.get(candidate + offset, float("nan"))
        if np.isfinite(val):
            values.append(val)
            days.append(candidate)
        i -= 1
    if len(values) < n:
        return None
    order = np.argsort(values)
    if method == "mid_6_10":
        selected = order[2:-2]
    elif method == "max_4_5":
        selected = order[1:]
    else:
        selected = order[1:-1]
    return [days[i] for i in selected], np.asarray(values, dtype=float)[selected]


def _context(df: pd.DataFrame, method: str) -> tuple[dict, dict, list[pd.Timestamp]]:
    safe = df["power"].mask(df.get("time_repaired", pd.Series(False, index=df.index)).fillna(True).astype(bool))
    values_by_time = safe.to_dict()
    rolling_window = safe.rolling(12, min_periods=12).mean().to_dict()
    days = pd.date_range(_interval_day(df.index.min()), _interval_day(df.index.max()), freq="D")
    eligible = [day for day in days if _eligible_ref_day(day, method)]
    return values_by_time, rolling_window, eligible


def cbl_predict(
    df: pd.DataFrame,
    target_times: pd.DatetimeIndex,
    method: str,
    same_day_adjust: bool = False,
    origin_times: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Estimate CBL using reference days fully available before each origin.

    Mid 6/10 trims two highs and lows; Max 4/5 removes the minimum;
    holiday Mid 4/6 trims one high and low. DR participation exclusions are
    unavailable, so these are labeled CBL-style approximations.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown CBL method: {method}")
    target_times = pd.DatetimeIndex(target_times)
    if origin_times is None:
        # This default is only a safe 1-hour horizon. Longer horizons must
        # supply their actual origins; otherwise the reference can leak.
        origin_times = target_times - pd.Timedelta(hours=1)
    origin_times = pd.DatetimeIndex(origin_times)
    if len(target_times) != len(origin_times):
        raise ValueError("target_times and origin_times must align")
    if np.any(target_times <= origin_times):
        raise ValueError("Each target must follow its origin")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("CBL requires a sorted unique ts_end index")
    values_by_time, windows, eligible_days = _context(df, method)
    results: list[float] = []
    for target, origin in zip(target_times, origin_times):
        if method == "holiday_mid_4_6" and not _eligible_ref_day(_interval_day(target), method):
            results.append(float("nan"))
            continue
        selected = _reference_values(values_by_time, eligible_days, target, origin, method)
        if selected is None:
            results.append(float("nan"))
            continue
        reference_days, values = selected
        estimate = float(values.mean())
        if same_day_adjust:
            current = windows.get(origin, float("nan"))
            reference_windows = [windows.get(day + (origin - origin.normalize()), float("nan")) for day in reference_days]
            if not np.isfinite(current) or not np.isfinite(reference_windows).all():
                estimate = float("nan")
            else:
                estimate += current - float(np.mean(reference_windows))
        results.append(estimate)
    result = pd.Series(results, index=origin_times, name=f"cbl_{method}{'_adjusted' if same_day_adjust else ''}")
    result.attrs["cbl_approximation"] = "Reference-day rules approximated; DR participation flags unavailable"
    return result


def cbl_all_predictions(df: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Return five all-row candidates; C3 is a holiday-aware hybrid.

    On nonholiday weekdays C3 falls back to C1, enabling paired all-row
    comparisons. The standalone holiday method returns NaN on those days.
    """
    origins = pd.DatetimeIndex(origins)
    targets = origins + pd.Timedelta(minutes=15 * horizon)
    names = {
        "c1_mid_6_10": ("mid_6_10", False),
        "c2_max_4_5": ("max_4_5", False),
        "c3_holiday_mid_4_6": ("holiday_mid_4_6", False),
        "c1a_mid_6_10_adjusted": ("mid_6_10", True),
        "c2a_max_4_5_adjusted": ("max_4_5", True),
    }
    frame = pd.DataFrame({name: cbl_predict(df, targets, method, adjust, origins).to_numpy()
                          for name, (method, adjust) in names.items()}, index=origins)
    frame["c3_holiday_mid_4_6"] = frame["c3_holiday_mid_4_6"].fillna(frame["c1_mid_6_10"])
    frame.attrs["c3_hybrid"] = "Holiday/Saturday/Sunday Mid4/6; weekday Mid6/10 fallback"
    return frame
