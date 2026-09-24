"""Development-only peak predictability analysis.

Every measured predictor carries the timestamp of its latest observation.  The
fold-specific thresholds below are fitted on the original fit prefix, while
per-row histories are cut at that row's forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.holidays import calendar_flags
from src.session_data import SEALED_BOUNDARY


QUARTER = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)
SINGLE_FEATURES = (
    "previous_day_max", "previous_week_max", "slot_rate7", "slot_rate28",
    "origin_level", "origin_slope", "previous_restart_rise",
)
EXTRA_FEATURES = (
    "previous_day_exceed", "previous_week_exceed", "target_weekday", "holiday_near",
)


def _ns(stamps: pd.DatetimeIndex) -> np.ndarray:
    """Normalize pandas 2.x/3.x microsecond or nanosecond indices to ns."""
    return pd.DatetimeIndex(stamps).to_numpy(dtype="datetime64[ns]").view("int64")


def _safe_history(history: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(history.index, pd.DatetimeIndex) or history.index.empty
            or not history.index.is_unique or not history.index.is_monotonic_increasing
            or history.index.max() >= SEALED_BOUNDARY or "power" not in history):
        raise ValueError("Expected sorted, unique, development-only 15-minute history")
    if history.index.unit != "ns":
        history = history.copy(deep=False)
        history.index = history.index.as_unit("ns")
    return history


def _clean_power(history: pd.DataFrame) -> pd.Series:
    power = pd.to_numeric(history.power, errors="coerce").astype(float)
    repaired = history.get("time_repaired", pd.Series(False, index=history.index))
    return power.mask(repaired.fillna(True).astype(bool))


def fit_restart_thresholds(
    history: pd.DataFrame, fit_end: pd.Timestamp, *, low_q: float = .25,
    rise_q: float = .90,
) -> dict:
    """Fit low-load and positive single-step rise thresholds on workdays only."""
    history = _safe_history(history)
    fit_end = pd.Timestamp(fit_end)
    if not 0 < low_q < 1 or not 0 < rise_q < 1 or fit_end >= SEALED_BOUNDARY:
        raise ValueError("Invalid fold threshold specification")
    train = history.loc[history.index <= fit_end]
    if train.empty:
        raise ValueError("No fold history before fit_end")
    power = _clean_power(train)
    flags = calendar_flags(train.index)
    workday = ~flags.is_offday.astype(bool).to_numpy()
    valid = power.notna().to_numpy() & workday
    low = power.iloc[np.flatnonzero(valid)]
    previous = power.shift(1)
    contiguous = np.r_[False, np.diff(_ns(train.index)) == QUARTER.value]
    same_day = np.r_[False, train.index.normalize()[1:] == train.index.normalize()[:-1]]
    delta = (power - previous).where(contiguous & same_day & workday)
    positive = delta.loc[delta.gt(0) & delta.notna()]
    if low.empty or positive.empty:
        raise ValueError("Fold has no clean workday power or positive rises")
    return {"low": float(low.quantile(low_q)), "rise": float(positive.quantile(rise_q)),
            "low_q": float(low_q), "rise_q": float(rise_q), "fit_end": str(fit_end),
            "latest_observation": str(train.index.max()),
            "n_power": int(len(low)), "n_positive_rises": int(len(positive))}


def detect_restart_events(
    history: pd.DataFrame, thresholds: dict, *, min_low_slots: int = 2,
    start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Find first qualifying jump after each contiguous low run.

    ``start`` and ``end`` filter returned events after detection; earlier
    history remains available to establish the low run.
    """
    history = _safe_history(history)
    if min_low_slots < 1 or not np.isfinite(float(thresholds["low"])) or not np.isfinite(float(thresholds["rise"])):
        raise ValueError("Invalid restart thresholds")
    idx = history.index
    value = _clean_power(history).to_numpy(dtype=float)
    low = np.isfinite(value) & (value <= float(thresholds["low"]))
    step = np.r_[False, np.diff(_ns(idx)) == QUARTER.value]
    same_day = np.r_[False, idx.normalize()[1:] == idx.normalize()[:-1]]
    rows: list[dict] = []
    run_start = -1
    for i in range(len(idx) + 1):
        continuing = (i < len(idx) and low[i] and (run_start < 0 or (step[i] and same_day[i])))
        if continuing:
            if run_start < 0:
                run_start = i
            continue
        if run_start >= 0:
            run_end = i - 1
            count = run_end - run_start + 1
            if count >= min_low_slots:
                for j in (run_end + 1, run_end + 2):
                    if (j >= len(idx) or not step[j] or not same_day[j]
                            or not np.isfinite(value[j]) or not np.isfinite(value[j - 1])):
                        break
                    rise = float(value[j] - value[j - 1])
                    if rise >= float(thresholds["rise"]) and value[j] > float(thresholds["low"]):
                        rows.append({"ts_end": idx[j], "restart_rise": rise,
                                     "low_run_slots": count, "low_end_time": idx[run_end],
                                     "slot": idx[j].hour * 4 + idx[j].minute // 15,
                                     "is_offday": bool(calendar_flags(pd.DatetimeIndex([idx[j]])).is_offday.iloc[0]),
                                     "threshold_low": float(thresholds["low"]),
                                     "threshold_rise": float(thresholds["rise"]),
                                     "threshold_fit_end": pd.Timestamp(thresholds["fit_end"])})
                        break
            run_start = -1
        if i < len(idx) and low[i]:
            run_start = i
    columns = ("restart_rise", "low_run_slots", "low_end_time", "slot", "is_offday",
               "threshold_low", "threshold_rise", "threshold_fit_end")
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="ts_end"))
    out = pd.DataFrame(rows).drop_duplicates("ts_end", keep="first").set_index("ts_end").sort_index()
    out.index.name = "ts_end"
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def discover_restart_slots(
    events: pd.DataFrame, history: pd.DataFrame, fit_end: pd.Timestamp,
    *, min_events: int = 5, min_rate: float = .10, cap: int = 8,
) -> pd.DataFrame:
    """Rank restart slots from fold training observations alone."""
    history = _safe_history(history)
    fit_end = pd.Timestamp(fit_end)
    if min_events < 1 or not 0 <= min_rate <= 1 or cap < 1:
        raise ValueError("Invalid slot discovery criteria")
    train = history.loc[history.index <= fit_end]
    valid = _clean_power(train).notna().to_numpy()
    workday = ~calendar_flags(train.index).is_offday.astype(bool).to_numpy()
    observed = train.index[valid & workday]
    denominator = pd.DataFrame({"date": observed.normalize(),
                                "slot": observed.hour * 4 + observed.minute // 15})
    denom = denominator.drop_duplicates().groupby("slot").size()
    e = events.loc[events.index <= fit_end]
    if not e.empty:
        e = e.loc[~e.is_offday.astype(bool)]
    counts = e.groupby("slot").size() if not e.empty else pd.Series(dtype=int)
    result = pd.DataFrame({"eligible_dates": denom.astype(int),
                           "events": counts.reindex(denom.index, fill_value=0).astype(int)})
    result["rate"] = result.events / result.eligible_dates
    result["selected"] = False
    # A rate tie is resolved by earlier clock slot, not event count.
    ranked = result.loc[result.events.ge(min_events) & result.rate.ge(min_rate)].reset_index()
    ranked = ranked.sort_values(["rate", "slot"], ascending=[False, True], kind="stable")
    chosen = ranked.head(cap).slot.to_numpy(dtype=int)
    result.loc[chosen, "selected"] = True
    return result.reset_index().sort_values("slot").reset_index(drop=True)


def _history_values_at(history: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    return _clean_power(history).reindex(times).to_numpy(dtype=float, copy=True)


def causal_restart_events(
    history: pd.DataFrame, origins: pd.DatetimeIndex, fit_end: pd.Timestamp,
    *, low_q: float = .25, rise_q: float = .90, min_low_slots: int = 2,
) -> pd.DataFrame:
    """Detect each day's events with thresholds available before that day.

    For a score date the original fold fit threshold is frozen.  For an earlier
    training date it is recomputed from the prefix ending the preceding day.
    No event from an earlier date is recomputed using a later threshold.
    """
    history = _safe_history(history)
    origin_index = pd.DatetimeIndex(origins)
    if origin_index.empty:
        return pd.DataFrame(columns=["restart_rise", "threshold_fit_end"],
                            index=pd.DatetimeIndex([], name="ts_end"))
    dates = pd.date_range(history.index.min().normalize(), origin_index.max().normalize(), freq="D")
    fold_end = pd.Timestamp(fit_end)
    rows = []
    frozen = fit_restart_thresholds(history, fold_end, low_q=low_q, rise_q=rise_q)
    for day in dates:
        prefix_end = min(fold_end, day - pd.Timedelta(nanoseconds=1))
        if prefix_end < history.index.min():
            continue
        try:
            thresholds = frozen if prefix_end == fold_end else fit_restart_thresholds(
                history, prefix_end, low_q=low_q, rise_q=rise_q)
        except ValueError:
            continue
        day_rows = history.loc[(history.index >= day) & (history.index < day + DAY)]
        if day_rows.empty:
            continue
        events = detect_restart_events(day_rows, thresholds, min_low_slots=min_low_slots)
        if not events.empty:
            rows.append(events)
    if not rows:
        return detect_restart_events(history.iloc[:1], frozen, min_low_slots=min_low_slots)
    out = pd.concat(rows).sort_index()
    if out.index.has_duplicates or (pd.to_datetime(out.threshold_fit_end) >= out.index).any():
        raise AssertionError("Causal event detection used an unavailable threshold")
    return out


def build_a2_features(
    history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
    *, tau: float, fit_end: pd.Timestamp, restart_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fixed A2 predictors and each predictor's latest measured time."""
    history = _safe_history(history)
    origins = pd.DatetimeIndex(origins, name="origin").as_unit("ns")
    if horizon not in (16, 96) or origins.has_duplicates or not origins.isin(history.index).all():
        raise ValueError("A2 requires unique observed h16/h96 origins")
    if (origins >= SEALED_BOUNDARY).any() or not np.isfinite(tau):
        raise ValueError("Origin crosses sealed boundary or peak threshold is invalid")
    target = origins + horizon * QUARTER
    if (target >= SEALED_BOUNDARY).any():
        raise ValueError("A2 target crosses sealed boundary")
    fit_end = pd.Timestamp(fit_end)
    x = pd.DataFrame(index=origins)
    used = pd.DataFrame(index=origins)

    def assign(name: str, values, stamps) -> None:
        x[name] = values
        used[name] = pd.DatetimeIndex(stamps) if not np.isscalar(stamps) else stamps

    for days, name in ((1, "previous_day"), (7, "previous_week")):
        stamp = target - days * DAY
        valid = stamp <= origins
        values = _history_values_at(history, stamp)
        values[~valid] = np.nan
        assign(f"{name}_max", values, stamp)
        assign(f"{name}_exceed", np.where(np.isfinite(values), (values > tau).astype(float), np.nan), stamp)

    safe = _clean_power(history)
    idx = history.index
    usable = safe.notna().to_numpy()
    peaks = (safe.to_numpy(dtype=float) > tau) & usable
    cumulative_valid = np.r_[0, np.cumsum(usable)]
    cumulative_peak = np.r_[0, np.cumsum(peaks)]
    last_global = np.minimum(_ns(origins), fit_end.value)
    global_end = _ns(idx).searchsorted(last_global, side="right")
    prior_n = cumulative_valid[global_end]
    prior_p = cumulative_peak[global_end]
    global_rate = np.divide(prior_p, prior_n, out=np.full(len(origins), np.nan), where=prior_n > 0)
    for days in (7, 28):
        stamps = np.column_stack([_ns(target - d * DAY) for d in range(1, days + 1)])
        caps = np.minimum(_ns(origins), fit_end.value)[:, None]
        allowed = stamps <= caps
        vals = np.column_stack([safe.reindex(pd.DatetimeIndex(stamps[:, k])).to_numpy(dtype=float)
                                for k in range(days)])
        valid = np.isfinite(vals) & allowed
        count = valid.sum(axis=1)
        peak_count = ((vals > tau) & valid).sum(axis=1)
        shrink = 5.0
        rate = np.where((count > 0) & np.isfinite(global_rate),
                        (peak_count + shrink * global_rate) / (count + shrink), np.nan)
        # The shrinkage prior sees the whole clean prefix through this bound,
        # which can be newer than the last same-slot sample.
        assign(f"slot_rate{days}", rate, pd.DatetimeIndex(last_global))
    level = safe.reindex(origins).to_numpy(dtype=float)
    assign("origin_level", level, origins)
    lag3 = origins - 3 * QUARTER
    lag3_value = safe.reindex(lag3).to_numpy(dtype=float)
    between = np.column_stack([safe.reindex(origins - k * QUARTER).to_numpy(dtype=float)
                               for k in range(4)])
    slope = (level - lag3_value) / 3
    slope[~np.isfinite(between).all(axis=1)] = np.nan
    assign("origin_slope", slope, origins)
    event_idx = restart_events.index
    if not isinstance(event_idx, pd.DatetimeIndex) or (event_idx >= SEALED_BOUNDARY).any():
        raise ValueError("Restart event timestamps must remain in development history")
    if len(event_idx):
        order = np.argsort(_ns(event_idx))
        event_ns = _ns(event_idx)[order]
        event_rise = restart_events.restart_rise.to_numpy(dtype=float)[order]
        event_pos = np.searchsorted(event_ns, _ns(origins), side="right") - 1
        event_value = np.full(len(origins), np.nan)
        event_time = np.full(len(origins), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        threshold_time = np.full(len(origins), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        valid = event_pos >= 0
        event_value[valid] = event_rise[event_pos[valid]]
        event_time[valid] = event_ns[event_pos[valid]].view("datetime64[ns]")
        if "threshold_fit_end" not in restart_events:
            raise ValueError("Restart events must include threshold provenance")
        threshold_ns = pd.to_datetime(restart_events.threshold_fit_end).to_numpy(dtype="datetime64[ns]").view("int64")[order]
        threshold_time[valid] = threshold_ns[event_pos[valid]].view("datetime64[ns]")
    else:
        event_value = np.full(len(origins), np.nan)
        event_time = np.full(len(origins), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        threshold_time = np.full(len(origins), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    latest_event_dependency = np.maximum(event_time, threshold_time)
    assign("previous_restart_rise", event_value, pd.DatetimeIndex(latest_event_dependency))
    cal = calendar_flags(target)
    x["target_weekday"] = target.dayofweek.astype(float)
    x["holiday_near"] = cal[["is_offday", "pre_holiday", "post_holiday", "bridge_day"]].any(axis=1).to_numpy(dtype=float)
    used["target_weekday"] = pd.NaT
    used["holiday_near"] = pd.NaT
    for name in x.columns:
        measured = pd.to_datetime(used[name], errors="coerce")
        if (measured.notna() & (measured > origins)).any():
            raise AssertionError(f"A2 feature {name} uses an observation after its origin")
    x.attrs["latest_observation_by_feature"] = used.copy()
    return x, used


@dataclass
class PredictabilityResult:
    tables: dict[str, pd.DataFrame]
    tests: pd.DataFrame
    summary: dict


def episode_starts(score: pd.DataFrame) -> pd.DataFrame:
    """Start each actual peak episode inside one fold and one calendar day."""
    frame = score.sort_values("target_time").copy()
    times = pd.DatetimeIndex(frame.target_time)
    actual = pd.to_numeric(frame.y, errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(frame.tau, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(actual) & np.isfinite(tau) & (actual > tau)
    rows = []
    for i in np.flatnonzero(mask):
        previous = i - 1
        if (previous >= 0 and mask[previous] and times[i] - times[previous] == QUARTER
                and times[i].normalize() == times[previous].normalize()):
            continue
        j = i
        while (j + 1 < len(times) and mask[j + 1] and times[j + 1] - times[j] == QUARTER
               and times[j + 1].normalize() == times[j].normalize()):
            j += 1
        rows.append({"fold": int(frame.fold.iloc[i]), "start": times[i], "end": times[j],
                     "max_power": float(np.max(actual[i:j + 1])), "tau": float(tau[i]),
                     "slot": int(times[i].hour * 4 + times[i].minute // 15),
                     "is_offday": bool(calendar_flags(pd.DatetimeIndex([times[i]])).is_offday.iloc[0])})
    return pd.DataFrame(rows, columns=["fold", "start", "end", "max_power", "tau", "slot", "is_offday"])


def _bootstrap_percentile(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [.025, .975])))


def _date_bootstrap_mean(frame: pd.DataFrame, column: str, *, n_boot: int = 1000,
                         seed: int = 42) -> tuple[float, float, float]:
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    dates = pd.to_datetime(frame.date).dt.normalize()
    groups = [frame.loc[dates.eq(d), column].to_numpy(dtype=float) for d in dates.drop_duplicates()]
    total = sum(len(group) for group in groups)
    estimate = float(np.nansum([np.nansum(group) for group in groups]) / total)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        denominator = sum(len(groups[i]) for i in chosen)
        reps[b] = sum(np.nansum(groups[i]) for i in chosen) / denominator
    return estimate, *_bootstrap_percentile(reps)


def _clock_minutes(stamps: pd.DatetimeIndex) -> np.ndarray:
    return stamps.hour.to_numpy() * 60 + stamps.minute.to_numpy()


def _near_schedule(minute: int, schedule: np.ndarray) -> bool:
    return bool(len(schedule) and np.min(np.abs(schedule - minute)) <= 30)


def _a1_proximity(
    peaks: pd.DataFrame, score_events: pd.DataFrame, score_dates: pd.DatetimeIndex,
    *, n_perm: int = 1000, seed: int = 42,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Permute complete daily restart schedules within fold and day type."""
    if peaks.empty or len(score_dates) == 0:
        return peaks.assign(near_restart=False, null_probability=np.nan), {
            "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_raw": 1.0,
            "n_positive_folds": 0, "eligible": False, "null_value": 0.0,
            "observed_fraction": np.nan, "null_fraction": np.nan}, pd.DataFrame()
    date_type = pd.DataFrame({"date": score_dates.normalize().unique()})
    date_type["is_offday"] = calendar_flags(pd.DatetimeIndex(date_type.date)).is_offday.to_numpy(dtype=bool)
    schedules: dict[pd.Timestamp, np.ndarray] = {}
    for day in date_type.date:
        stamp = pd.Timestamp(day)
        on_day = score_events.loc[score_events.index.normalize() == stamp]
        schedules[stamp] = _clock_minutes(on_day.index)
    rows = peaks.copy()
    rows["date"] = rows.start.dt.normalize()
    rows["minute"] = rows.start.dt.hour * 60 + rows.start.dt.minute
    rows["near_restart"] = [
        _near_schedule(int(row.minute), schedules.get(row.date, np.array([], dtype=int)))
        for row in rows.itertuples(index=False)]
    groups = {bool(off): dates.date.to_list() for off, dates in date_type.groupby("is_offday")}
    expected = []
    for row in rows.itertuples(index=False):
        candidate_dates = groups[bool(row.is_offday)]
        expected.append(np.mean([_near_schedule(int(row.minute), schedules[d]) for d in candidate_dates]))
    rows["null_probability"] = expected
    rows["effect"] = rows.near_restart.astype(float) - rows.null_probability
    observed = float(rows.near_restart.mean())
    null_fraction = float(rows.null_probability.mean())
    effect, ci_low, ci_high = _date_bootstrap_mean(rows, "effect", n_boot=n_perm, seed=seed)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    date_to_type = {row.date: bool(row.is_offday) for row in date_type.itertuples(index=False)}
    for iteration in range(n_perm):
        reassigned = {}
        for off, dates in groups.items():
            if not dates:
                continue
            shuffling = rng.permutation(len(dates))
            reassigned.update({d: schedules[dates[i]] for d, i in zip(dates, shuffling)})
        null[iteration] = np.mean([
            _near_schedule(int(row.minute), reassigned.get(row.date, np.array([], dtype=int)))
            for row in rows.itertuples(index=False)])
    p = (1 + int(np.count_nonzero(null >= observed))) / (n_perm + 1)
    fold_rows = []
    for fold, group in rows.groupby("fold"):
        fold_rows.append({"fold": int(fold), "peak_starts": len(group),
                          "observed_near": float(group.near_restart.mean()),
                          "null_expected": float(group.null_probability.mean()),
                          "effect": float(group.effect.mean())})
    result = {"estimate": effect, "ci_low": ci_low, "ci_high": ci_high,
              "p_raw": float(p), "n_positive_folds": sum(row["effect"] > 0 for row in fold_rows),
              "eligible": len(rows) > 0 and len(fold_rows) == 3,
              "null_value": 0.0, "observed_fraction": observed,
              "null_fraction": null_fraction,
              "n_peak_starts": len(rows), "n_score_days": len(date_type),
              "permutation_n": n_perm, "bootstrap_n": n_perm}
    return rows, result, pd.DataFrame(fold_rows)


def analyze_a1(history: pd.DataFrame, h4_score: pd.DataFrame,
               folds: dict[int, dict], *, n_boot: int = 1000,
               seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict]:
    history = _safe_history(history)
    events_by_fold = []
    slots_by_fold = []
    sensitivity = []
    start_tables = []
    schedule_rows = []
    peak_day_rows = []
    fold_dates = []
    for fold, score in h4_score.groupby("fold"):
        fid = int(fold)
        meta = folds[fid]
        fit_end = pd.Timestamp(meta["fit_end"])
        threshold = fit_restart_thresholds(history, fit_end)
        train_events = detect_restart_events(history.loc[history.index <= fit_end], threshold)
        score_dates = pd.DatetimeIndex(pd.to_datetime(score.target_time).dt.normalize().unique()).sort_values()
        score_events = detect_restart_events(history, threshold,
                                             start=score_dates.min(),
                                             end=score_dates.max() + DAY - QUARTER)
        score_events = score_events.copy()
        score_events["fold"] = fid
        events_by_fold.append(score_events.reset_index())
        slots = discover_restart_slots(train_events, history, fit_end)
        slots["fold"] = fid
        slots_by_fold.append(slots)
        fold_dates.append(pd.DataFrame({"fold": fid, "date": score_dates}))
        starts = episode_starts(score)
        starts["selected_restart_slot"] = starts.slot.isin(slots.loc[slots.selected, "slot"])
        start_tables.append(starts)
        for day in score_dates:
            day_events = score_events.loc[score_events.index.normalize() == day]
            for slot in slots.loc[slots.selected, "slot"]:
                target = day + int(slot) * QUARTER
                record = score.loc[pd.to_datetime(score.target_time).eq(target)]
                if record.empty:
                    continue
                row = record.iloc[0]
                peak = bool(row.y > row.tau)
                peak_day_rows.append({"fold": fid, "date": day, "slot": int(slot),
                                      "peak": peak, "power": float(row.y),
                                      "excess": float(max(0, row.y - row.tau)),
                                      "restart_observed": target in day_events.index})
        # One-at-a-time sensitivity checks remain descriptive and cannot
        # replace the frozen primary hypothesis.
        variants = (("low_q20", .20, 2), ("low_q30", .30, 2),
                    ("duration_15m", .25, 1), ("duration_45m", .25, 3))
        for label, low_q, duration in variants:
            adjusted = fit_restart_thresholds(history, fit_end, low_q=low_q)
            variant_events = detect_restart_events(history, adjusted, min_low_slots=duration,
                                                   start=score_dates.min(),
                                                   end=score_dates.max() + DAY - QUARTER)
            rate = np.mean([_near_schedule(int(t.hour * 60 + t.minute),
                           _clock_minutes(variant_events.loc[variant_events.index.normalize() == t.normalize()].index))
                            for t in starts.start]) if not starts.empty else np.nan
            sensitivity.append({"fold": fid, "variant": label, "low_q": low_q,
                                "min_low_slots": duration, "restart_events": len(variant_events),
                                "peak_starts": len(starts), "observed_proximity": rate,
                                "analysis_role": "descriptive_sensitivity_only"})
        weeks = score_events.index.isocalendar().week.to_numpy(dtype=int)
        for event, week in zip(score_events.itertuples(), weeks):
            schedule_rows.append({"fold": fid, "date": event.Index.normalize(),
                                  "week": int(week), "slot": int(event.slot),
                                  "selected_slot": bool(event.slot in set(slots.loc[slots.selected, "slot"])),
                                  "restart_rise": float(event.restart_rise)})
    all_starts = pd.concat(start_tables, ignore_index=True) if start_tables else pd.DataFrame()
    all_events = pd.concat(events_by_fold, ignore_index=True) if events_by_fold else pd.DataFrame()
    all_slots = pd.concat(slots_by_fold, ignore_index=True) if slots_by_fold else pd.DataFrame()
    all_dates = pd.concat(fold_dates, ignore_index=True) if fold_dates else pd.DataFrame()
    # Zero-event slot/week cells are retained so stability is not inflated by
    # conditioning on dates when a restart happened.
    weekly_rows = []
    drift_rows = []
    clean = _clean_power(history)
    for fid, dates in all_dates.groupby("fold"):
        fid = int(fid)
        score_days = pd.DatetimeIndex(dates.date)
        fold_events = all_events.loc[all_events.fold.eq(fid)]
        fold_slots = all_slots.loc[all_slots.fold.eq(fid)].set_index("slot")
        for slot in range(96):
            candidate = score_days + slot * QUARTER
            workday = ~calendar_flags(candidate).is_offday.astype(bool).to_numpy()
            valid = clean.reindex(candidate).notna().to_numpy() & workday
            event_stamps = pd.DatetimeIndex(fold_events.ts_end)
            active = candidate.isin(event_stamps) & valid
            score_rate = float(active.sum() / valid.sum()) if valid.sum() else np.nan
            train_rate = float(fold_slots.loc[slot, "rate"]) if slot in fold_slots.index else np.nan
            drift_rows.append({"fold": fid, "slot": slot, "train_rate": train_rate,
                               "score_rate": score_rate, "train_events": int(fold_slots.loc[slot, "events"])
                               if slot in fold_slots.index else 0,
                               "score_events": int(active.sum()), "score_eligible_dates": int(valid.sum()),
                               "selected_slot": bool(fold_slots.loc[slot, "selected"]) if slot in fold_slots.index else False})
            week_starts = score_days - pd.to_timedelta(score_days.dayofweek, unit="D")
            for week in pd.DatetimeIndex(week_starts).unique():
                mask = week_starts == week
                denominator = int(valid[mask].sum())
                weekly_rows.append({"fold": fid, "week_start": week, "slot": slot,
                                    "eligible_dates": denominator,
                                    "events": int(active[mask].sum()),
                                    "rate": float(active[mask].sum() / denominator) if denominator else np.nan,
                                    "selected_fold_slot": bool(fold_slots.loc[slot, "selected"])
                                    if slot in fold_slots.index else False})
    near_rows = []
    for fid, starts in all_starts.groupby("fold"):
        events = all_events.loc[all_events.fold.eq(fid)].copy().set_index("ts_end")
        dates = pd.DatetimeIndex(all_dates.loc[all_dates.fold.eq(fid), "date"])
        schedule = {d: _clock_minutes(events.loc[events.index.normalize() == d].index) for d in dates}
        for row in starts.itertuples(index=False):
            minute = row.start.hour * 60 + row.start.minute
            near_rows.append({"fold": fid, "start": row.start, "date": row.start.normalize(),
                              "minute": minute, "is_offday": row.is_offday,
                              "near_restart": _near_schedule(minute, schedule[row.start.normalize()])})
    near = pd.DataFrame(near_rows)
    # Pooled permutation retains fold and day type; no episode joins or
    # cross-fold schedule borrowing.
    details = []
    expected_rows = []
    pool = {}
    for fid, dates in all_dates.groupby("fold"):
        fid = int(fid)
        ev = all_events.loc[all_events.fold.eq(fid)].set_index("ts_end")
        for day in pd.DatetimeIndex(dates.date):
            off = bool(calendar_flags(pd.DatetimeIndex([day])).is_offday.iloc[0])
            pool[(fid, day, off)] = _clock_minutes(ev.loc[ev.index.normalize() == day].index)
    grouped_pool = {}
    for (fid, day, off), schedule in pool.items():
        grouped_pool.setdefault((fid, off), []).append((day, schedule))
    for row in near.itertuples(index=False):
        candidates = grouped_pool[(int(row.fold), bool(row.is_offday))]
        probability = float(np.mean([_near_schedule(int(row.minute), sched) for _, sched in candidates]))
        expected_rows.append(probability)
    if not near.empty:
        near["null_probability"] = expected_rows
        near["effect"] = near.near_restart.astype(float) - near.null_probability
        effect, ci_low, ci_high = _date_bootstrap_mean(near, "effect", n_boot=n_boot, seed=seed)
        observed = float(near.near_restart.mean())
        null_fraction = float(near.null_probability.mean())
        rng = np.random.default_rng(seed)
        perm = np.empty(n_boot)
        for b in range(n_boot):
            mapping = {}
            for key, dates_schedules in grouped_pool.items():
                order = rng.permutation(len(dates_schedules))
                for (day, _), source in zip(dates_schedules, order):
                    mapping[(key[0], day, key[1])] = dates_schedules[source][1]
            perm[b] = np.mean([_near_schedule(int(row.minute), mapping[(int(row.fold), row.date, bool(row.is_offday))])
                               for row in near.itertuples(index=False)])
        p_raw = (1 + int((perm >= observed).sum())) / (n_boot + 1)
        fold_effects = near.groupby("fold").effect.mean()
        positive_folds = int(fold_effects.gt(0).sum())
    else:
        effect = ci_low = ci_high = observed = null_fraction = np.nan
        p_raw = 1.0
        positive_folds = 0
    hypothesis = {"analysis_id": "A1", "hypothesis_id": "A1_proximity",
                  "horizon": 4, "feature": "restart_slot", "test_kind": "proximity",
                  "estimate": effect, "ci_low": ci_low, "ci_high": ci_high,
                  "p_raw": p_raw, "n_positive_folds": positive_folds,
                  "eligible": len(near) > 0 and near.fold.nunique() == 3,
                  "null_value": 0.0}
    summary = {"hypothesis": hypothesis, "n_peak_starts": len(all_starts),
               "n_selected_slot_starts": int(all_starts.selected_restart_slot.sum()) if len(all_starts) else 0,
               "observed_proximity": observed, "null_mean_proximity": null_fraction,
               "fold_effects": near.groupby("fold").effect.mean().to_dict() if len(near) else {},
               "analysis_role": "retrospective_restart_proximity"}
    tables = {"A1_peak_starts": all_starts, "A1_restart_events": all_events,
              "A1_restart_slots": all_slots, "A1_proximity": near,
              "A1_sensitivity": pd.DataFrame(sensitivity),
              "A1_slot_peak_days": pd.DataFrame(peak_day_rows),
              "A1_weekly_schedules": pd.DataFrame(schedule_rows),
              "A1_week_slot_rates": pd.DataFrame(weekly_rows),
              "A1_fold_slot_drift": pd.DataFrame(drift_rows)}
    return tables, summary


def _score_cohort(score: pd.DataFrame, selected_slots: set[int]) -> pd.DataFrame:
    frame = score.sort_values("target_time").copy()
    stamp = pd.DatetimeIndex(frame.target_time)
    slot = stamp.hour * 4 + stamp.minute // 15
    workday = ~calendar_flags(stamp).is_offday.astype(bool).to_numpy()
    frame = frame.loc[np.asarray(pd.Index(slot).isin(selected_slots)) & workday].copy()
    frame = frame.loc[pd.to_numeric(frame.y, errors="coerce").notna()]
    if frame.duplicated("target_time").any():
        raise AssertionError("A2 score cohort has duplicate target times")
    return frame


def _training_cohort(history: pd.DataFrame, selected_slots: set[int], horizon: int,
                     fit_end: pd.Timestamp, excluded_dates: pd.DatetimeIndex) -> pd.DataFrame:
    end = pd.Timestamp(fit_end)
    idx = history.index[history.index <= end]
    slot = idx.hour * 4 + idx.minute // 15
    workday = ~calendar_flags(idx).is_offday.astype(bool).to_numpy()
    eligible = pd.Index(slot).isin(selected_slots) & workday & ~idx.normalize().isin(excluded_dates)
    targets = idx[eligible]
    origins = targets - horizon * QUARTER
    power = _clean_power(history).reindex(targets).to_numpy(dtype=float)
    valid = origins.isin(history.index) & np.isfinite(power) & (targets <= end)
    return pd.DataFrame({"origin": origins[valid], "target_time": targets[valid],
                         "y": power[valid]}).sort_values("target_time").reset_index(drop=True)


def _fold_auc_summary(predictions: pd.DataFrame, *, null_value: float,
                      n_boot: int = 1000, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Fold-size weighted AUC, resampling whole dates inside each score fold."""
    fold_rows = []
    blocks: dict[int, list[pd.DataFrame]] = {}
    for fid, fold in predictions.groupby("fold"):
        n_pos = int(fold.label.sum())
        n_neg = int(len(fold) - n_pos)
        estimate = float(roc_auc_score(fold.label, fold.probability)) if n_pos and n_neg else np.nan
        fold_rows.append({"fold": int(fid), "n": len(fold), "n_positive": n_pos,
                          "n_negative": n_neg, "auc": estimate,
                          "effect": estimate - null_value if np.isfinite(estimate) else np.nan})
        dates = pd.to_datetime(fold.target_time).dt.normalize()
        blocks[int(fid)] = [fold.loc[dates.eq(day)] for day in dates.drop_duplicates()]
    fold_result = pd.DataFrame(fold_rows)
    adequately_powered = (len(fold_result) == 3
                           and fold_result.n_positive.ge(5).all()
                           and fold_result.n_negative.ge(5).all())
    valid_fold = fold_result.loc[fold_result.auc.notna()]
    if valid_fold.empty:
        summary = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                   "p_raw": 1.0, "n_positive_folds": 0, "eligible": False,
                   "null_value": null_value, "n_boot_valid": 0}
        return summary, fold_result
    observed = float(np.average(valid_fold.auc, weights=valid_fold.n))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        aucs = []
        weights = []
        for fid in sorted(blocks):
            day_blocks = blocks[fid]
            picked = rng.integers(0, len(day_blocks), len(day_blocks))
            sample = pd.concat([day_blocks[k] for k in picked], ignore_index=True)
            if sample.label.nunique() < 2:
                break
            aucs.append(float(roc_auc_score(sample.label, sample.probability)))
            weights.append(len(sample))
        if len(aucs) == len(blocks):
            boot.append(float(np.average(aucs, weights=weights)))
    low, high = _bootstrap_percentile(np.asarray(boot))
    # Center the bootstrap effect around its observed value for a one-sided
    # null test.  The original bootstrap draws also form the percentile CI.
    centered = np.asarray(boot) - observed
    effect = observed - null_value
    p_raw = (1 + int(np.count_nonzero(centered >= effect))) / (len(centered) + 1) if len(centered) else 1.0
    summary = {"estimate": observed, "ci_low": low, "ci_high": high,
               "p_raw": float(p_raw),
               "n_positive_folds": int(fold_result.effect.gt(0).sum()),
               "eligible": bool(adequately_powered and len(boot) >= .8 * n_boot),
               "null_value": null_value, "n_boot_valid": len(boot)}
    return summary, fold_result


def analyze_a2(history: pd.DataFrame, oof_by_horizon: dict[int, pd.DataFrame],
               folds: dict[tuple[int, int], dict], *, n_boot: int = 1000,
               seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict, dict[int, dict]]:
    """Fit fixed logistic signals in each existing fold with causal features."""
    history = _safe_history(history)
    all_predictions = []
    all_cohorts = []
    all_slots = []
    all_audit = []
    hypothesis_rows = []
    fold_rows = []
    per_fold_auc = []
    for horizon in (16, 96):
        score_all = oof_by_horizon[horizon]
        for fid, score in score_all.groupby("fold"):
            fid = int(fid)
            meta = folds[(horizon, fid)]
            fit_end = pd.Timestamp(meta["fit_end"])
            threshold = fit_restart_thresholds(history, fit_end)
            train_events = detect_restart_events(history.loc[history.index <= fit_end], threshold)
            slots = discover_restart_slots(train_events, history, fit_end)
            slots["fold"], slots["horizon"] = fid, horizon
            all_slots.append(slots)
            selected = set(slots.loc[slots.selected, "slot"].astype(int))
            cohort_score = _score_cohort(score, selected)
            train = _training_cohort(history, selected, horizon, fit_end,
                                     pd.DatetimeIndex(pd.to_datetime(cohort_score.target_time).dt.normalize().unique()))
            fold_rows.append({"horizon": horizon, "fold": fid,
                              "selected_slots": len(selected), "train_n": len(train),
                              "score_n": len(cohort_score),
                              "all_score_peaks": int((score.y > score.tau).sum()),
                              "selected_score_peaks": int((cohort_score.y > cohort_score.tau).sum()),
                              "fit_end": str(fit_end),
                              "score_start": str(pd.to_datetime(score.origin).min()),
                              "score_end": str(pd.to_datetime(score.origin).max())})
            if train.empty or cohort_score.empty:
                continue
            origins = pd.DatetimeIndex(pd.concat([train.origin, cohort_score.origin], ignore_index=True))
            causal_events = causal_restart_events(history, origins, fit_end)
            x_train, latest_train = build_a2_features(
                history, pd.DatetimeIndex(train.origin), horizon, tau=float(meta["tau"]),
                fit_end=fit_end, restart_events=causal_events)
            x_score, latest_score = build_a2_features(
                history, pd.DatetimeIndex(cohort_score.origin), horizon, tau=float(meta["tau"]),
                fit_end=fit_end, restart_events=causal_events)
            if not pd.DatetimeIndex(train.target_time).max() <= fit_end:
                raise AssertionError("A2 train labels extend beyond fold fit_end")
            if pd.DatetimeIndex(train.target_time).normalize().isin(
                    pd.DatetimeIndex(cohort_score.target_time).normalize()).any():
                raise AssertionError("A2 score dates overlap train dates")
            for x, latest, frame, partition in ((x_train, latest_train, train, "fit"),
                                                 (x_score, latest_score, cohort_score, "score")):
                for origin, target in zip(frame.origin, frame.target_time):
                    if pd.Timestamp(target) != pd.Timestamp(origin) + horizon * QUARTER:
                        raise AssertionError("A2 origin/target alignment changed")
                audit = latest.copy()
                audit["origin"] = frame.origin.to_numpy()
                audit["target_time"] = frame.target_time.to_numpy()
                audit["partition"] = partition
                audit["fold"] = fid
                audit["horizon"] = horizon
                all_audit.append(audit.reset_index(drop=True))
            train_label = (train.y.to_numpy(dtype=float) > float(meta["tau"])).astype(int)
            score_label = (cohort_score.y.to_numpy(dtype=float) > float(meta["tau"])).astype(int)
            cohort_score = cohort_score.copy()
            cohort_score["label"] = score_label
            cohort_score["horizon"] = horizon
            cohort_score["fold"] = fid
            cohort_score["selected_slot"] = (
                pd.DatetimeIndex(cohort_score.target_time).hour * 4
                + pd.DatetimeIndex(cohort_score.target_time).minute // 15)
            all_cohorts.append(cohort_score)
            if len(np.unique(train_label)) < 2 or min(np.bincount(train_label)) < 5:
                continue
            for feature in (*SINGLE_FEATURES, "multivariate"):
                columns = list(SINGLE_FEATURES + EXTRA_FEATURES) if feature == "multivariate" else [feature]
                if not np.isfinite(x_train[columns].to_numpy(dtype=float)).any():
                    continue
                pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                         StandardScaler(),
                                         LogisticRegression(C=1, max_iter=1000, random_state=seed))
                pipeline.fit(x_train[columns], train_label)
                probability = pipeline.predict_proba(x_score[columns])[:, 1]
                scored = cohort_score[["origin", "target_time", "horizon", "fold", "label"]].copy()
                scored["feature"] = feature
                scored["probability"] = probability
                scored["reference_only"] = True
                all_predictions.append(scored)
    prediction = (pd.concat(all_predictions, ignore_index=True) if all_predictions else
                  pd.DataFrame(columns=["origin", "target_time", "horizon", "fold", "label",
                                        "feature", "probability", "reference_only"]))
    fold_cohorts = pd.DataFrame(fold_rows)
    for horizon in (16, 96):
        for feature in (*SINGLE_FEATURES, "multivariate"):
            family = prediction.loc[prediction.horizon.eq(horizon) & prediction.feature.eq(feature)] if len(prediction) else pd.DataFrame()
            null = .60 if feature == "multivariate" else .50
            if family.empty:
                result = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                          "p_raw": 1., "n_positive_folds": 0, "eligible": False,
                          "null_value": null, "n_boot_valid": 0}
            else:
                result, per_fold = _fold_auc_summary(family, null_value=null,
                                                      n_boot=n_boot, seed=seed)
                per_fold["horizon"] = horizon
                per_fold["feature"] = feature
                per_fold_auc.append(per_fold)
            hypothesis_rows.append({"analysis_id": "A2", "hypothesis_id": f"A2_h{horizon}_{feature}",
                                    "horizon": horizon, "feature": feature,
                                    "test_kind": "auc_multi" if feature == "multivariate" else "auc_single",
                                    **result})
    tables = {"A2_predictions": prediction,
              "A2_score_cohort": pd.concat(all_cohorts, ignore_index=True) if all_cohorts else pd.DataFrame(),
              "A2_fold_cohorts": fold_cohorts,
              "A2_restart_slots": pd.concat(all_slots, ignore_index=True) if all_slots else pd.DataFrame(),
              "A2_feature_provenance": pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame(),
              "A2_fold_auc": pd.concat(per_fold_auc, ignore_index=True)
              if per_fold_auc else pd.DataFrame(columns=["fold", "n", "n_positive", "n_negative",
                                                   "auc", "effect", "horizon", "feature"])}
    summary = {"hypotheses": hypothesis_rows,
               "cohort_fold_counts": fold_cohorts.to_dict("records"),
               "h16_multivariate": next((r for r in hypothesis_rows if r["hypothesis_id"] == "A2_h16_multivariate"), {}),
               "h96_multivariate": next((r for r in hypothesis_rows if r["hypothesis_id"] == "A2_h96_multivariate"), {}),
               "analysis_role": "causal_origin_available_predictability"}
    a3 = {16: {"score_cohort": tables["A2_score_cohort"].loc[
        tables["A2_score_cohort"].horizon.eq(16)].copy() if not tables["A2_score_cohort"].empty else pd.DataFrame(),
        "folds": folds}}
    return tables, summary, a3


def _a3_values(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["origin_level", "target_weekday", "observed_production", "temperature"])
    origin = pd.DatetimeIndex(frame.origin)
    target = pd.DatetimeIndex(frame.target_time)
    if (target >= SEALED_BOUNDARY).any() or (origin >= target).any():
        raise ValueError("A3 targets must precede sealed boundary and follow their origins")
    safe = _clean_power(history)
    return pd.DataFrame({
        "origin_level": safe.reindex(origin).to_numpy(dtype=float),
        "target_weekday": target.dayofweek.astype(float),
        "observed_production": pd.to_numeric(history.get("production_target", pd.Series(dtype=float)),
                                               errors="coerce").reindex(target).to_numpy(dtype=float),
        "temperature": pd.to_numeric(history.get("temperature", pd.Series(dtype=float)),
                                      errors="coerce").reindex(target).to_numpy(dtype=float),
    })


def _ols_coefficient(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if len(y) < max(30, x.shape[1] + 2):
        return None
    design = np.column_stack([np.ones(len(y)), x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    return np.linalg.lstsq(design, y, rcond=None)[0][1:]


def analyze_a3(history: pd.DataFrame, a2_tables: dict[str, pd.DataFrame],
               folds: dict[tuple[int, int], dict], *, n_boot: int = 1000,
               seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict]:
    """Separate origin-available and retrospective peak-size associations."""
    history = _safe_history(history)
    scores = a2_tables["A2_score_cohort"]
    slot_frame = a2_tables["A2_restart_slots"]
    score_parts = []
    train_parts = []
    strata_rows = []
    for fid in range(3):
        meta = folds[(16, fid)]
        fit_end = pd.Timestamp(meta["fit_end"])
        selected = set(slot_frame.loc[slot_frame.horizon.eq(16) & slot_frame.fold.eq(fid)
                                      & slot_frame.selected, "slot"].astype(int))
        fold_score = scores.loc[scores.horizon.eq(16) & scores.fold.eq(fid)].copy()
        train = _training_cohort(history, selected, 16, fit_end,
                                 pd.DatetimeIndex(pd.to_datetime(fold_score.target_time).dt.normalize().unique()))
        train = train.loc[train.y.gt(float(meta["tau"]))].copy().reset_index(drop=True)
        fold_score = fold_score.loc[fold_score.y.gt(fold_score.tau)].copy().reset_index(drop=True)
        train_x = _a3_values(history, train)
        score_x = _a3_values(history, fold_score)
        for feature in ("origin_level", "target_weekday", "observed_production", "temperature"):
            values = pd.to_numeric(train_x[feature], errors="coerce")
            finite = values.loc[np.isfinite(values)]
            edges = np.unique(finite.quantile([0, .2, .4, .6, .8, 1]).to_numpy(dtype=float)) if len(finite) else np.array([])
            if len(edges) < 2:
                continue
            selected_score = score_x[feature].to_numpy(dtype=float)
            excess = (fold_score.y - fold_score.tau).to_numpy(dtype=float)
            for group_index in range(len(edges) - 1):
                low, high = edges[group_index:group_index + 2]
                mask = (selected_score >= low) & (selected_score <= high if group_index == len(edges) - 2 else selected_score < high)
                if mask.any():
                    strata_rows.append({"fold": fid, "feature": feature,
                                        "analysis_role": "descriptive_only",
                                        "bin_low_fit": low, "bin_high_fit": high,
                                        "n_score_peaks": int(mask.sum()),
                                        "median_excess": float(np.median(excess[mask]))})
        train_block = train[["origin", "target_time"]].copy()
        train_block["excess"] = (train.y - float(meta["tau"])).to_numpy(dtype=float)
        train_block["fold"] = fid
        train_parts.append(pd.concat([train_block, train_x], axis=1))
        score_block = fold_score[["origin", "target_time"]].copy()
        score_block["excess"] = (fold_score.y - fold_score.tau).to_numpy(dtype=float)
        score_block["fold"] = fid
        score_parts.append(pd.concat([score_block, score_x], axis=1))
    training = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    scored = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    hypotheses = []
    coefficient_rows = []
    for role, columns in (("pre_origin", ("origin_level", "target_weekday")),
                          ("retrospective_only", ("observed_production", "temperature"))):
        standardized_parts = []
        for fid in range(3):
            tr = training.loc[training.fold.eq(fid)]
            sc = scored.loc[scored.fold.eq(fid)].copy()
            if tr.empty or sc.empty:
                continue
            for column in columns:
                values = pd.to_numeric(tr[column], errors="coerce")
                mean = float(values.mean())
                scale = float(values.std(ddof=0))
                if not np.isfinite(scale) or scale <= 0:
                    sc[column + "_std"] = np.nan
                else:
                    sc[column + "_std"] = (pd.to_numeric(sc[column], errors="coerce") - mean) / scale
            sc = sc.loc[np.isfinite(sc[[c + "_std" for c in columns]].to_numpy(dtype=float)).all(axis=1)]
            if not sc.empty:
                standardized_parts.append(sc)
        cohort = pd.concat(standardized_parts, ignore_index=True) if standardized_parts else pd.DataFrame()
        if cohort.empty:
            observed = None
        else:
            observed = _ols_coefficient(cohort[[c + "_std" for c in columns]].to_numpy(dtype=float),
                                        cohort.excess.to_numpy(dtype=float))
        reps = []
        if observed is not None:
            groups = []
            for fid, group in cohort.groupby("fold"):
                dates = pd.to_datetime(group.target_time).dt.normalize()
                groups.extend(group.loc[dates.eq(day)] for day in dates.drop_duplicates())
            rng = np.random.default_rng(seed)
            for _ in range(n_boot):
                sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))],
                                   ignore_index=True)
                coeff = _ols_coefficient(sample[[c + "_std" for c in columns]].to_numpy(dtype=float),
                                         sample.excess.to_numpy(dtype=float))
                if coeff is not None:
                    reps.append(coeff)
        array = np.asarray(reps, dtype=float) if reps else np.empty((0, len(columns)))
        for j, column in enumerate(columns):
            estimate = float(observed[j]) if observed is not None else np.nan
            low, high = _bootstrap_percentile(array[:, j]) if len(array) else (np.nan, np.nan)
            centered = array[:, j] - estimate if len(array) else np.array([])
            p_raw = ((1 + np.count_nonzero(np.abs(centered) >= abs(estimate))) / (len(centered) + 1)
                     if len(centered) else 1.)
            directions = 0
            for fid, group in cohort.groupby("fold") if len(cohort) else []:
                fold_coef = _ols_coefficient(group[[c + "_std" for c in columns]].to_numpy(dtype=float),
                                             group.excess.to_numpy(dtype=float))
                if fold_coef is not None and np.sign(fold_coef[j]) == np.sign(estimate):
                    directions += 1
            row = {"analysis_id": "A3", "hypothesis_id": f"A3_{column}",
                   "horizon": 16, "feature": column, "test_kind": "coefficient",
                   "estimate": estimate, "ci_low": low, "ci_high": high,
                   "p_raw": float(p_raw), "n_positive_folds": directions,
                   "eligible": bool(observed is not None and len(array) >= .8 * n_boot),
                   "null_value": 0.0, "analysis_role": role,
                   "fold_direction_definition": "sign_matches_pooled_estimate",
                   "n_score_peaks": len(cohort), "n_boot_valid": len(array)}
            hypotheses.append(row)
            coefficient_rows.append(row)
    tables = {"A3_score_peak_cohort": scored, "A3_train_peak_cohort": training,
              "A3_coefficients": pd.DataFrame(coefficient_rows),
              "A3_strata": pd.DataFrame(strata_rows, columns=["fold", "feature", "analysis_role",
                                                                "bin_low_fit", "bin_high_fit",
                                                                "n_score_peaks", "median_excess"])}
    summary = {"hypotheses": hypotheses, "n_score_peaks": len(scored),
               "analysis_role": "pre_origin_and_retrospective_separated",
               "retrospective_features_not_allowed_for_forecast": ["observed_production", "temperature"]}
    return tables, summary
