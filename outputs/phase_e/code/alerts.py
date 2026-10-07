"""Phase E fixed-grid alert and normalized decision diagnostics.

The episode assignment is performed once on the observed series for each
fold/subset/threshold/policy.  Bootstrap samples *complete onset-day event
clusters* within each fold: an actual TP/FN belongs to its actual onset day,
and an unmatched alert FP belongs to its alert onset day.  It never stitches
episodes across sampled days or rematches a resampled time series.  This
conditional event-cluster CI therefore omits matching uncertainty; it is not
an operating-day or causal benefit estimate.  Exposure is the number of
represented target-calendar days within each fold (summed for pooled rows).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.analysis.errors import episodes, match_episode_table


THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
POLICIES = ("1/1", "2/2")
STEP = pd.Timedelta(minutes=15)
REQUIRED = ("fold", "origin", "target_time", "y", "tau", "p_cal", "is_d2_novel_profile")
EVENT_COLUMNS = ("subset", "fold", "threshold", "policy", "status", "miss_class",
                 "actual_start", "actual_end", "alert_start", "alert_end",
                 "first_direct_issue_time", "direct_lead_minutes", "actual_duration_intervals",
                 "alert_duration_intervals", "actual_max", "max_p_cal_actual_range",
                 "overlapping_alarm_count", "matching_reason", "onset_day", "diagnostic_basis")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    return _ratio(2 * precision * recall, precision + recall)


def _prepare(score: pd.DataFrame) -> pd.DataFrame:
    absent = [c for c in REQUIRED if c not in score]
    if absent:
        raise ValueError(f"score lacks required fields: {absent}")
    data = score.copy()
    if data["fold"].isna().any():
        raise ValueError("fold contains null")
    for col in ("origin", "target_time"):
        data[col] = pd.to_datetime(data[col], errors="raise")
        if data[col].isna().any():
            raise ValueError(f"{col} contains null")
    for col in ("y", "tau", "p_cal"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must be finite")
    if data["p_cal"].lt(0).any() or data["p_cal"].gt(1).any():
        raise ValueError("p_cal must be in [0, 1]")
    if data["is_d2_novel_profile"].isna().any():
        raise ValueError("is_d2_novel_profile contains null")
    if not data["is_d2_novel_profile"].isin((True, False)).all():
        raise ValueError("is_d2_novel_profile must be boolean")
    data["is_d2_novel_profile"] = data["is_d2_novel_profile"].astype(bool)
    if "is_peak" in data and not data["is_peak"].astype(bool).eq(data.y.gt(data.tau)).all():
        raise ValueError("is_peak disagrees with y > tau")
    if "horizon" in data:
        horizon = data["horizon"].astype(str).str.lower().str.removeprefix("h")
        if not pd.to_numeric(horizon, errors="coerce").eq(16).all():
            raise ValueError("alert evaluation is locked to horizon 16")
    if not data.target_time.sub(data.origin).eq(pd.Timedelta(hours=4)).all():
        raise ValueError("target_time must be exactly 240 minutes after origin")
    if data.duplicated(["fold", "origin"]).any() or data.duplicated(["fold", "target_time"]).any():
        raise ValueError("fold/origin and fold/target_time must be unique")
    data = data.sort_values(["fold", "origin"], kind="stable").reset_index(drop=True)
    data["actual_flag"] = data.y.gt(data.tau)
    return data


def _counts(frame: pd.DataFrame, flag: pd.Series) -> dict[str, int]:
    actual = frame["actual_flag"].to_numpy(dtype=bool)
    action = np.asarray(flag, dtype=bool)
    return {"TP": int((actual & action).sum()), "FP": int((~actual & action).sum()),
            "FN": int((actual & ~action).sum()), "TN": int((~actual & ~action).sum())}


def _position_row(frame: pd.DataFrame, flag: pd.Series, **keys) -> dict:
    counts = _counts(frame, flag)
    precision = _ratio(counts["TP"], counts["TP"] + counts["FP"])
    recall = _ratio(counts["TP"], counts["TP"] + counts["FN"])
    return {**keys, "n_positions": len(frame), **counts, "precision": precision,
            "recall": recall, "F1": _f1(precision, recall)}


def _decision_row(frame: pd.DataFrame, flag: pd.Series, **keys) -> dict:
    row = _position_row(frame, flag, **keys)
    c = keys["threshold"]
    n = len(frame)
    events = row["TP"] + row["FN"]
    actions = row["TP"] + row["FP"]
    e_model = c * actions + row["FN"]
    e_no = float(events)
    e_all = c * n
    e_clim = min(e_no, e_all)
    e_perf = c * events
    return {**row, "threshold": c, "actions": actions, "E_model": e_model,
            "E_no": e_no, "E_all": e_all, "E_clim": e_clim, "E_perf": e_perf,
            "relative_value": _ratio(e_clim - e_model, e_clim - e_perf),
            "value_definition": "normalized cost-loss; L=1, C=threshold; 1/1 only"}


def _fold_events(frame: pd.DataFrame, flag: pd.Series, **keys) -> tuple[pd.DataFrame, list[int]]:
    pred = frame[["origin", "target_time", "y", "tau", "p_cal"]].copy()
    pred["alert_flag"] = np.asarray(flag, dtype=bool)
    event_table = match_episode_table(pred)
    alert_ranges = episodes(pred.target_time, pred.alert_flag)
    alert_by_start = {start: (end, int((end - start) / STEP) + 1) for start, end in alert_ranges}
    records = []
    for item in event_table.itertuples(index=False):
        is_actual = item.status in ("TP", "FN")
        a_start = item.start if is_actual else pd.NaT
        a_end = item.end if is_actual else pd.NaT
        alarm_start = item.alarm_start if item.status in ("TP", "FP") else pd.NaT
        alarm_end, alarm_duration = alert_by_start[alarm_start] if pd.notna(alarm_start) else (pd.NaT, np.nan)
        first_issue = pd.NaT
        lead = float("nan")
        if item.status == "TP":
            overlap_start = max(a_start, alarm_start)
            overlap_end = min(a_end, alarm_end)
            direct = pred.loc[pred.alert_flag & pred.target_time.between(overlap_start, overlap_end)]
            if direct.empty:
                raise AssertionError("matched episode has no positive row in its intersection")
            first_issue = direct.origin.min()
            lead = float((a_start - first_issue) / pd.Timedelta(minutes=1))
        peak_segment = pred.loc[pred.target_time.between(a_start, a_end)] if is_actual else pred.iloc[0:0]
        max_p = float(peak_segment.p_cal.max()) if is_actual else float("nan")
        if item.status == "FP":
            miss_class = "false_alert"
        elif item.status == "TP":
            miss_class = "hit" if lead > 0 else "late_hit"
        else:
            miss_class = "forecast_miss" if max_p < THRESHOLDS[0] else "decision_miss"
        onset = a_start if is_actual else alarm_start
        records.append({**keys, "status": item.status, "miss_class": miss_class,
                        "actual_start": a_start, "actual_end": a_end,
                        "alert_start": alarm_start, "alert_end": alarm_end,
                        "first_direct_issue_time": first_issue, "direct_lead_minutes": lead,
                        "actual_duration_intervals": int(item.duration_intervals) if is_actual else np.nan,
                        "alert_duration_intervals": alarm_duration,
                        "actual_max": float(item.actual_max), "max_p_cal_actual_range": max_p,
                        "overlapping_alarm_count": item.overlapping_alarm_count,
                        "matching_reason": item.missed_reason,
                        "onset_day": onset.normalize(),
                        "diagnostic_basis": "one-to-one target overlap; FN probability < fixed 0.01 grid minimum separates forecast/decision misses"})
    return pd.DataFrame.from_records(records, columns=EVENT_COLUMNS), [duration for _, duration in alert_by_start.values()]


def _episode_point(events: pd.DataFrame, exposure_days: int, alert_durations: list[int], **keys) -> dict:
    status = events.status if not events.empty else pd.Series(dtype=str)
    tp = int(status.eq("TP").sum())
    fp = int(status.eq("FP").sum())
    fn = int(status.eq("FN").sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    leads = events.loc[status.eq("TP"), "direct_lead_minutes"].to_numpy(dtype=float) if not events.empty else np.empty(0)
    leads = leads[np.isfinite(leads)]
    durations = np.asarray(alert_durations, dtype=float)
    quantile = lambda q: float(np.quantile(leads, q)) if len(leads) else float("nan")
    return {**keys, "actual_peak_episodes": tp + fn, "detected_peak_episodes": tp,
            "missed_peak_episodes": fn, "false_alert_episodes": fp,
            "episode_precision": precision, "episode_recall": recall,
            "episode_F1": _f1(precision, recall), "alert_episode_count": tp + fp,
            "alert_episode_duration_mean": float(durations.mean()) if len(durations) else float("nan"),
            "alert_episode_duration_median": float(np.median(durations)) if len(durations) else float("nan"),
            "represented_target_calendar_days": int(exposure_days),
            "false_alert_episodes_per_operating_day": _ratio(fp, exposure_days),
            "day_basis": "distinct represented target-calendar days per fold; not verified factory operating days",
            "direct_lead_count": len(leads),
            "direct_lead_mean_minutes": float(leads.mean()) if len(leads) else float("nan"),
            "direct_lead_median_minutes": quantile(.5),
            "direct_lead_p10_minutes": quantile(.1), "direct_lead_p25_minutes": quantile(.25),
            "direct_lead_p75_minutes": quantile(.75), "direct_lead_p90_minutes": quantile(.9),
            "direct_lead_min_minutes": float(leads.min()) if len(leads) else float("nan"),
            "direct_lead_max_minutes": float(leads.max()) if len(leads) else float("nan")}


def _bootstrap(events_by_fold: dict, days_by_fold: dict, n_boot: int, seed: int) -> dict:
    """Stratified onset-day bootstrap on fixed one-to-one event assignments."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    fold_clusters = []
    for fold, days in days_by_fold.items():
        byday = defaultdict(list)
        events = events_by_fold[fold]
        for row in events.itertuples(index=False):
            byday[row.onset_day].append(row)
        clusters = []
        for day in days:
            rows = byday[day]
            counts = np.array([sum(r.status == state for r in rows) for state in ("TP", "FP", "FN")], dtype=int)
            leads = np.array([r.direct_lead_minutes for r in rows if r.status == "TP"], dtype=float)
            clusters.append((counts, leads))
        fold_clusters.append(clusters)
    values = {k: np.full(n_boot, np.nan) for k in
              ("episode_precision", "episode_recall", "false_alert_episodes_per_operating_day",
               "direct_lead_mean_minutes", "direct_lead_median_minutes")}
    exposure = sum(len(clusters) for clusters in fold_clusters)
    for b in range(n_boot):
        counts = np.zeros(3, dtype=int)
        leads = []
        for clusters in fold_clusters:
            if not clusters:
                continue
            for i in rng.integers(0, len(clusters), size=len(clusters)):
                day_counts, day_leads = clusters[i]
                counts += day_counts
                if len(day_leads):
                    leads.extend(day_leads)
        tp, fp, fn = counts
        values["episode_precision"][b] = _ratio(tp, tp + fp)
        values["episode_recall"][b] = _ratio(tp, tp + fn)
        values["false_alert_episodes_per_operating_day"][b] = _ratio(fp, exposure)
        if leads:
            values["direct_lead_mean_minutes"][b] = float(np.mean(leads))
            values["direct_lead_median_minutes"][b] = float(np.median(leads))
    out = {"bootstrap_n": n_boot, "bootstrap_seed": seed,
           "bootstrap_unit": "stratified fold x target-calendar-onset-day; fixed complete episodes"}
    for metric, samples in values.items():
        valid = samples[np.isfinite(samples)]
        out[f"{metric}_ci_low"] = float(np.quantile(valid, .025)) if len(valid) else float("nan")
        out[f"{metric}_ci_high"] = float(np.quantile(valid, .975)) if len(valid) else float("nan")
        out[f"{metric}_bootstrap_valid"] = len(valid)
    return out


def evaluate_alerts(score: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Evaluate locked E3/E4 grid on D1 and frozen-flag D2 without fitting.

    Returns five named tables.  Undefined precision/recall/value/lead are NaN.
    `fold='pooled'` aggregates independently matched fold episodes.  Every
    D2 flag is first computed from its complete D1 fold origin sequence.
    """
    if not isinstance(n_boot, int) or n_boot < 0:
        raise ValueError("n_boot must be a nonnegative integer")
    data = _prepare(score)
    decisions, positions, episodes_metrics, events_all, misses = [], [], [], [], []
    folds = list(data.fold.unique())
    for threshold in THRESHOLDS:
        # Build flags on the complete D1 origin sequence, before either subset.
        flags = {}
        for fold in folds:
            part = data.loc[data.fold.eq(fold)]
            watch = part.p_cal.ge(threshold)
            contiguous = part.origin.diff().eq(STEP)
            confirmed = watch & watch.shift(1, fill_value=False) & contiguous
            flags[fold] = pd.DataFrame({"watch": watch.to_numpy(dtype=bool),
                                        "confirmed": confirmed.to_numpy(dtype=bool)}, index=part.index)
        all_flags = pd.concat(flags.values()).sort_index() if flags else pd.DataFrame(index=data.index)
        for subset in ("D1", "D2"):
            selected = data if subset == "D1" else data.loc[data.is_d2_novel_profile]
            if selected.empty:
                continue
            # Decision value is fixed to the 1/1 action candidate.
            for fold_key, frame in [(str(f), selected.loc[selected.fold.eq(f)]) for f in folds] + [("pooled", selected)]:
                if frame.empty:
                    continue
                keys = {"subset": subset, "fold": fold_key, "threshold": threshold}
                decisions.append(_decision_row(frame, all_flags.loc[frame.index, "watch"], **keys))
            for policy, flag_col in (("1/1", "watch"), ("2/2", "confirmed")):
                frame_events, durations_by_fold, days_by_fold = {}, {}, {}
                for f in folds:
                    frame = selected.loc[selected.fold.eq(f)]
                    if frame.empty:
                        continue
                    keys = {"subset": subset, "fold": str(f), "threshold": threshold, "policy": policy}
                    flag = all_flags.loc[frame.index, flag_col]
                    positions.append(_position_row(frame, flag, **keys))
                    event_rows, durations = _fold_events(frame, flag, **keys)
                    frame_events[str(f)] = event_rows
                    durations_by_fold[str(f)] = durations
                    days = pd.DatetimeIndex(frame.target_time.dt.normalize().unique()).sort_values()
                    days_by_fold[str(f)] = list(days)
                    events_all.append(event_rows)
                    metric = _episode_point(event_rows, len(days), durations, **keys)
                    metric.update(_bootstrap({str(f): event_rows}, {str(f): list(days)}, n_boot, seed))
                    episodes_metrics.append(metric)
                pooled_keys = {"subset": subset, "fold": "pooled", "threshold": threshold, "policy": policy}
                pooled_flags = all_flags.loc[selected.index, flag_col]
                positions.append(_position_row(selected, pooled_flags, **pooled_keys))
                pooled_events = pd.concat(frame_events.values(), ignore_index=True) if frame_events else pd.DataFrame()
                pooled_duration = [d for values in durations_by_fold.values() for d in values]
                pooled_metric = _episode_point(pooled_events, sum(map(len, days_by_fold.values())), pooled_duration,
                                               **pooled_keys)
                pooled_metric.update(_bootstrap(frame_events, days_by_fold, n_boot, seed))
                episodes_metrics.append(pooled_metric)
                for fold_key, ev in [*frame_events.items(), ("pooled", pooled_events)]:
                    actual = ev.loc[ev.status.ne("FP")] if not ev.empty else ev
                    for category in ("hit", "late_hit", "decision_miss", "forecast_miss"):
                        misses.append({"subset": subset, "fold": fold_key, "threshold": threshold,
                                       "policy": policy, "miss_class": category,
                                       "episodes": int(actual.miss_class.eq(category).sum()) if not actual.empty else 0,
                                       "actual_peak_episodes": len(actual),
                                       "classification_basis": "TP direct lead >0 hit; TP <=0 late; FN max p_cal <0.01 forecast; other FN decision"})
    return {"decision_value_curve": pd.DataFrame(decisions),
            "alert_episode_metrics": pd.DataFrame(episodes_metrics),
            "alert_position_metrics": pd.DataFrame(positions),
            "alert_episode_events": pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame(columns=EVENT_COLUMNS),
            "miss_decomposition": pd.DataFrame(misses)}
