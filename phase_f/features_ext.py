"""Phase F causal power features and provenance.

Input timestamps are interval ends. An origin's own reading is available;
later readings are never used. Missing observations remain NaN so the shared
evaluation keys cannot silently change with a feature group.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from phase_c.data import _core_features
from src.holidays import HOLIDAYS_2021, calendar_flags
from src.models.cbl import cbl_predict


_SLOT = pd.Timedelta(minutes=15)
_DAY = pd.Timedelta(days=1)
_WEEK = pd.Timedelta(days=7)
_WINDOWS = (4, 8, 16, 32, 96, 672)
_STATS = ("mean", "max", "min", "std", "q90")

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "lag_1_4": tuple(f"lag_{i}" for i in range(1, 5)),
    "lag_1_8": tuple(f"lag_{i}" for i in range(1, 9)),
    "lag_1_16": tuple(f"lag_{i}" for i in range(1, 17)),
    "slot_1_7d": tuple(f"slot_{i}d" for i in range(1, 8)),
    "slot_7_28d": tuple(f"slot_{i}d" for i in (7, 14, 21, 28)),
    **{
        f"same_slot_{weeks}w": tuple(
            f"same_slot_{weeks}w_{stat}" for stat in ("mean", "median", "max", "std")
        )
        for weeks in (2, 3, 4)
    },
    "same_slot_5work": ("same_slot_5work_mean",),
    "profile": (
        "current_minus_lastweek", "today_to_lastweek_ratio",
        *(f"profile_corr_{weeks}w" for weeks in (1, 2, 3, 4)),
    ),
    "rolling": (
        *(f"rolling_{window}_{stat}" for window in _WINDOWS for stat in _STATS),
        *(f"ewm_{span}" for span in (4, 16, 96)),
    ),
    "trend": (
        "slope_1h", "slope_4h", "diff_1", "diff_2",
        "since_large_change_slots",
    ),
    "peak": (
        "peak_count_24h", "peak_count_7d", "since_peak_slots",
        "current_over_today_max",
    ),
    "calendar": (
        "pre_holiday", "post_holiday", "bridge_day", "workday_ordinal",
        "slot_x_day_type", "month_progress",
    ),
    "legacy": (
        *(f"legacy_lag_{i}" for i in range(8)),
        "legacy_lag_96", "legacy_lag_672",
        "legacy_target_slot_1d_ago", "legacy_target_slot_7d_ago",
        "legacy_recent_4_slope", "cbl_mid_6_10", "cbl_mid_6_10_adjusted",
    ),
    "production": ("production_last_completed_hour", "production_same_slot_7d_mean"),
}


def available_group_configs() -> list[dict[str, object]]:
    """The F1 ablations; F1-11 is a downstream selection method, not a feature.

    Each entry has a stable experiment ID and selected feature groups. Callers
    may combine groups while retaining the original F1 ID as parent_exp.
    """
    return [
        {"id": "F1-1-lag1-4", "groups": ("lag_1_4",)},
        {"id": "F1-1-lag1-8", "groups": ("lag_1_8",)},
        {"id": "F1-1-lag1-16", "groups": ("lag_1_16",)},
        {"id": "F1-2-daily", "groups": ("slot_1_7d",)},
        {"id": "F1-2-weekly", "groups": ("slot_7_28d",)},
        *({"id": f"F1-3-{weeks}w", "groups": (f"same_slot_{weeks}w",)} for weeks in (2, 3, 4)),
        {"id": "F1-3-five-workdays", "groups": ("same_slot_5work",)},
        {"id": "F1-4", "groups": ("profile",)},
        {"id": "F1-5", "groups": ("rolling",)},
        {"id": "F1-6", "groups": ("trend",)},
        {"id": "F1-7", "groups": ("peak",)},
        {"id": "F1-8", "groups": ("calendar",)},
        {"id": "F1-9", "groups": ("legacy",)},
        {"id": "F1-10", "groups": ("production",)},
        {"id": "F1-11", "groups": (), "method": "fit-only feature ranking"},
    ]


def _validate(history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("History requires unique, sorted interval-end timestamps")
    if "power" not in history or history.empty:
        raise ValueError("History requires a nonempty clean power column")
    if not history.index.equals(pd.date_range(history.index.min(), history.index.max(), freq="15min", name=history.index.name)):
        raise ValueError("History must use a complete 15-minute timestamp grid")
    if not 4 <= horizon <= 16:
        raise ValueError("Phase F requires horizons 4..16 quarter-hours")
    origins = pd.DatetimeIndex(origins, name="origin")
    if origins.has_duplicates or not origins.is_monotonic_increasing or not origins.isin(history.index).all():
        raise ValueError("Origins must be unique, chronological timestamps in history")
    return origins


def _power(history: pd.DataFrame) -> pd.Series:
    power = pd.to_numeric(history["power"], errors="coerce")
    for flag in ("time_repaired", "quality_bad"):
        if flag in history:
            power = power.mask(history[flag].fillna(True).astype(bool))
    return power


def _timestamps(origins: pd.DatetimeIndex, times: object) -> pd.Series:
    if times is pd.NaT:
        return pd.Series(pd.NaT, index=origins, dtype="datetime64[ns]")
    return pd.Series(pd.DatetimeIndex(times), index=origins)


def _complete_window(values: pd.Series, length: int, method: str) -> pd.Series:
    roll = values.rolling(length, min_periods=length)
    if method == "q90":
        return roll.quantile(0.9)
    if method == "std":
        return roll.std(ddof=1)
    return getattr(roll, method)()


def _profile_correlations(power: pd.Series, weeks: int) -> pd.Series:
    other = power.shift(672 * weeks)
    day = (power.index - pd.Timedelta(nanoseconds=1)).normalize()
    group = pd.Series(day, index=power.index)
    valid = power.notna() & other.notna()

    def running(value: pd.Series) -> pd.Series:
        return value.fillna(0).groupby(group).cumsum()

    count = valid.astype(int).groupby(group).cumsum()
    slots = pd.Series(1, index=power.index).groupby(group).cumsum()
    a, b = power.where(valid), other.where(valid)
    sa, sb = running(a), running(b)
    va = running(a * a) - sa * sa / count.replace(0, np.nan)
    vb = running(b * b) - sb * sb / count.replace(0, np.nan)
    covariance = running(a * b) - sa * sb / count.replace(0, np.nan)
    result = covariance / np.sqrt(va * vb)
    return result.where((count == slots) & (count >= 2) & (va > 0) & (vb > 0)).clip(-1, 1)


def _five_complete_workdays(
    power: pd.Series, origins: pd.DatetimeIndex, target: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.Series]:
    # Interval days run 00:15 through the following 00:00. Checking a whole
    # reference day is permitted only after its 00:00 completion timestamp.
    interval_day = (power.index - pd.Timedelta(nanoseconds=1)).normalize()
    complete = power.notna().groupby(interval_day).agg(["sum", "count"])
    complete_days = set(complete.index[(complete["sum"] == 96) & (complete["count"] == 96)])
    target_day = (target - pd.Timedelta(nanoseconds=1)).normalize()
    slot_offset = target - target_day
    available = power.to_dict()
    out = np.full(len(origins), np.nan)
    used = pd.Series(pd.NaT, index=origins, dtype="datetime64[ns]")
    for row, (origin, day, offset) in enumerate(zip(origins, target_day, slot_offset)):
        values = []
        last_completion = pd.NaT
        for back in range(1, 36):
            reference_day = day - pd.Timedelta(days=back)
            completion = reference_day + _DAY
            if completion > origin or reference_day not in complete_days:
                continue
            if reference_day.dayofweek >= 5 or reference_day.strftime("%Y-%m-%d") in HOLIDAYS_2021:
                continue
            value = available.get(reference_day + offset, np.nan)
            if np.isfinite(value):
                values.append(float(value))
                if pd.isna(last_completion):
                    last_completion = completion
            if len(values) == 5:
                out[row] = float(np.mean(values))
                used.iloc[row] = last_completion
                break
    return out, used


def build_features(
    history: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizon: int,
    tau: float,
    groups: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Add selected Phase F feature groups to the 15 sealed Phase B features.

    ``tau`` must be fitted on the fold's fit partition before calling. Calendar
    features carry NaT provenance because they are known in advance; all
    observed features carry the latest measurement or complete-day timestamp.
    No row filtering, imputation, scaling, or target access occurs here.
    """
    origins = _validate(history, origins, horizon)
    chosen = tuple(FEATURE_GROUPS) if groups is None else tuple(groups)
    unknown = set(chosen) - FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    if "peak" in chosen and not np.isfinite(tau):
        raise ValueError("Peak features require a finite fit-only tau")

    x, provenance = _core_features(history, origins, horizon)
    x = x.copy()
    provenance = {name: used.copy() for name, used in provenance.items()}
    added: dict[str, np.ndarray] = {}
    power = _power(history)
    target = origins + horizon * _SLOT
    requested = set(name for group in chosen for name in FEATURE_GROUPS[group])

    def add(name: str, values: object, times: object = None) -> None:
        if name not in requested:
            return
        added[name] = np.asarray(values, dtype=float)
        provenance[name] = _timestamps(origins, origins if times is None else times)

    def observed(times: pd.DatetimeIndex) -> np.ndarray:
        if (times > origins).any():
            raise AssertionError("Feature requests power after origin")
        return power.reindex(times).to_numpy(dtype=float)

    for lag in range(1, 17):
        name = f"lag_{lag}"
        if name in requested:
            times = origins - lag * _SLOT
            add(name, observed(times), times)
    for days in sorted({*range(1, 8), 14, 21, 28}):
        name = f"slot_{days}d"
        if name in requested:
            times = target - days * _DAY
            add(name, observed(times), times)

    for weeks in (2, 3, 4):
        if f"same_slot_{weeks}w_mean" not in requested:
            continue
        times = [target - i * _WEEK for i in range(1, weeks + 1)]
        samples = np.column_stack([observed(t) for t in times])
        valid = np.isfinite(samples).all(axis=1)
        stats = {
            "mean": samples.mean(axis=1),
            "median": np.median(samples, axis=1),
            "max": samples.max(axis=1),
            "std": samples.std(axis=1, ddof=1),
        }
        for stat, values in stats.items():
            add(f"same_slot_{weeks}w_{stat}", np.where(valid, values, np.nan), times[0])
    if "same_slot_5work_mean" in requested:
        values, used = _five_complete_workdays(power, origins, target)
        added["same_slot_5work_mean"] = values
        provenance["same_slot_5work_mean"] = used

    if any(name in requested for name in FEATURE_GROUPS["profile"]):
        previous = power.shift(672)
        add("current_minus_lastweek", (power - previous).reindex(origins).to_numpy())
        day = (power.index - pd.Timedelta(nanoseconds=1)).normalize()
        key = pd.Series(day, index=power.index)
        today_sum = power.fillna(0).groupby(key).cumsum()
        lastweek_sum = previous.fillna(0).groupby(key).cumsum()
        present = power.notna().astype(int).groupby(key).cumsum()
        prior_present = previous.notna().astype(int).groupby(key).cumsum()
        count = pd.Series(1, index=power.index).groupby(key).cumsum()
        ratio = (today_sum / lastweek_sum.replace(0, np.nan)).where((present == count) & (prior_present == count))
        add("today_to_lastweek_ratio", ratio.reindex(origins).to_numpy())
        for weeks in (1, 2, 3, 4):
            name = f"profile_corr_{weeks}w"
            if name in requested:
                add(name, _profile_correlations(power, weeks).reindex(origins).to_numpy())

    for window in _WINDOWS:
        if not any(f"rolling_{window}_{stat}" in requested for stat in _STATS):
            continue
        for stat in _STATS:
            name = f"rolling_{window}_{stat}"
            if name in requested:
                add(name, _complete_window(power, window, stat).reindex(origins).to_numpy())
    for span in (4, 16, 96):
        name = f"ewm_{span}"
        if name in requested:
            add(name, power.ewm(span=span, adjust=False, min_periods=span).mean().reindex(origins).to_numpy())

    if any(name in requested for name in FEATURE_GROUPS["trend"]):
        delta = power.diff()
        add("slope_1h", ((power - power.shift(3)) / 3).reindex(origins).to_numpy())
        add("slope_4h", ((power - power.shift(15)) / 15).reindex(origins).to_numpy())
        add("diff_1", delta.reindex(origins).to_numpy())
        add("diff_2", delta.diff().reindex(origins).to_numpy())
        if "since_large_change_slots" in requested:
            changed = delta.abs().gt(0.1 * float(tau)) & delta.notna()
            markers = pd.Series(np.where(changed, np.arange(len(power), dtype=float), np.nan), index=power.index).ffill()
            elapsed = pd.Series(np.arange(len(power), dtype=float), index=power.index) - markers
            add("since_large_change_slots", elapsed.reindex(origins).to_numpy())

    if "peak" in chosen:
        is_peak = power.gt(tau).astype(float).mask(power.isna())
        for slots, suffix in ((96, "24h"), (672, "7d")):
            add(f"peak_count_{suffix}", is_peak.rolling(slots, min_periods=slots).sum().reindex(origins).to_numpy())
        markers = pd.Series(np.where(is_peak.eq(1), np.arange(len(power), dtype=float), np.nan), index=power.index).ffill()
        elapsed = pd.Series(np.arange(len(power), dtype=float), index=power.index) - markers
        add("since_peak_slots", elapsed.reindex(origins).to_numpy())
        day = (power.index - pd.Timedelta(nanoseconds=1)).normalize()
        daily_max = power.groupby(day).cummax()
        level = (power / daily_max.replace(0, np.nan)).where(power.notna())
        add("current_over_today_max", level.reindex(origins).to_numpy())

    if "calendar" in chosen:
        flags = calendar_flags(target)
        for name in ("pre_holiday", "post_holiday", "bridge_day"):
            add(name, flags[name].to_numpy(), pd.NaT)
        # Keep the same target timestamp calendar convention as Phase C core.
        target_day = target.normalize()
        work_ord = []
        for day in target_day:
            monday = day - pd.Timedelta(days=day.dayofweek)
            weekdays = pd.date_range(monday, day, freq="D")
            work_ord.append(sum(t.dayofweek < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS_2021 for t in weekdays))
        add("workday_ordinal", work_ord, pd.NaT)
        slot = (target.hour.to_numpy() * 4 + target.minute.to_numpy() / 15).astype(float)
        day_type = np.where(flags["is_offday"].to_numpy(dtype=bool), 1, 0)
        add("slot_x_day_type", slot * day_type, pd.NaT)
        add("month_progress", (target_day.day.to_numpy() - 1) / target_day.days_in_month.to_numpy(), pd.NaT)

    if "legacy" in chosen:
        for lag in (*range(8), 96, 672):
            times = origins - lag * _SLOT
            add(f"legacy_lag_{lag}", observed(times), times)
        for days in (1, 7):
            times = target - days * _DAY
            add(f"legacy_target_slot_{days}d_ago", observed(times), times)
        add("legacy_recent_4_slope", ((power - power.shift(3)) / 3).reindex(origins).to_numpy())
        if "cbl_mid_6_10" in requested:
            add("cbl_mid_6_10", cbl_predict(history, target, "mid_6_10", False, origins).to_numpy())
            add("cbl_mid_6_10_adjusted", cbl_predict(history, target, "mid_6_10", True, origins).to_numpy())

    if "production" in chosen:
        # The loader timestamps each hourly production value when the hour
        # finishes. A raw same-hour target value is never a substitute.
        completed = pd.to_numeric(history.get("production_completed", pd.Series(np.nan, index=history.index)), errors="coerce")
        if completed.notna().any() and not (completed.index[completed.notna()].minute == 0).all():
            raise ValueError("Production completion timestamps must be exact hour ends")
        if "production_completed_bad" in history:
            completed = completed.mask(history["production_completed_bad"].fillna(True).astype(bool))
        latest_hour = origins.floor("h")
        add("production_last_completed_hour", completed.reindex(latest_hour).to_numpy(), latest_hour)
        known = completed.ffill()
        times = [origins - i * _DAY for i in range(1, 8)]
        prior = np.column_stack([known.reindex(t).to_numpy(dtype=float) for t in times])
        mean = np.where(np.isfinite(prior).all(axis=1), prior.mean(axis=1), np.nan)
        add("production_same_slot_7d_mean", mean, times[0])

    if added:
        x = pd.concat([x, pd.DataFrame(added, index=origins)], axis=1)
    if set(x.columns) != set(provenance):
        raise AssertionError("Every feature must have a provenance series")
    if any((used > origins).fillna(False).any() for used in provenance.values()):
        raise AssertionError("Feature reads an observation after origin")
    return x, provenance


def causal_sequences(
    history: pd.DataFrame, origins: pd.DatetimeIndex, context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return right-aligned past power and finite-observation mask.

    Values before available history are NaN/False. This function makes no
    forecast-specific imputation choice and does not remove any origin.
    """
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    origins = _validate(history, origins, 4)
    power = _power(history).to_numpy(dtype=float)
    positions = history.index.get_indexer(origins)
    values = np.full((len(origins), context_length), np.nan, dtype=float)
    mask = np.zeros((len(origins), context_length), dtype=bool)
    for row, position in enumerate(positions):
        count = min(context_length, position + 1)
        selected = power[position + 1 - count:position + 1]
        values[row, -count:] = selected
        mask[row, -count:] = np.isfinite(selected)
    return values, mask
