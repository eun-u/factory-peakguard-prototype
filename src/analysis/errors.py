"""Condition-level prediction errors and missed evening peak episodes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.viz import configure, save
from src.holidays import calendar_flags
from ._common import attach_history, write_table


def episodes(times: pd.Series | pd.DatetimeIndex, flags) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """A missing prediction breaks an episode, even if adjacent row positions touch."""
    stamps = pd.DatetimeIndex(pd.to_datetime(times))
    positive = sorted(stamps[np.asarray(flags, dtype=bool)])
    if not positive:
        return []
    out = []
    start = end = positive[0]
    for stamp in positive[1:]:
        if stamp - end == pd.Timedelta(minutes=15):
            end = stamp
        else:
            out.append((start, end))
            start = end = stamp
    out.append((start, end))
    return out


def match_episode_table(pred: pd.DataFrame, alert_col: str = "alert_flag") -> pd.DataFrame:
    actual = episodes(pred.target_time, pred.y.to_numpy() > pred.tau.to_numpy())
    alarms = episodes(pred.target_time, pred[alert_col].to_numpy())
    # Maximise distinct detected episodes first, then total overlap. A greedy
    # longest-overlap match can undercount detections when alarms straddle two
    # true episodes.
    matched: dict[int, int] = {}
    overlapping_alarms = [
        [j for j, (b_start, b_end) in enumerate(alarms)
         if min(a_end, b_end) >= max(a_start, b_start)]
        for a_start, a_end in actual
    ]
    if actual and alarms:
        weights = np.zeros((len(actual), len(alarms)+len(actual)), dtype=float)
        for i, (a_start, a_end) in enumerate(actual):
            for j, (b_start, b_end) in enumerate(alarms):
                overlap = (min(a_end, b_end)-max(a_start, b_start))/pd.Timedelta(minutes=15)+1
                if overlap > 0:
                    weights[i, j] = 1_000_000+overlap
        rows_i, cols_j = linear_sum_assignment(-weights)
        matched = {int(i): int(j) for i, j in zip(rows_i, cols_j) if j < len(alarms) and weights[i, j] > 0}
    matched_alarms = set(matched.values())
    rows = []
    for i, (a_start, a_end) in enumerate(actual):
        if i in matched:
            j = matched[i]
            status = "TP"
            alarm_start = alarms[j][0]
        else:
            status, alarm_start = "FN", pd.NaT
        missed_reason = ("no_overlap" if not overlapping_alarms[i] else "one_to_one_assignment") if status == "FN" else ""
        segment = pred.loc[pred.target_time.between(a_start, a_end)]
        peak_row = segment.loc[segment.y.idxmax()]
        rows.append({"status": status, "start": a_start, "end": a_end,
                     "actual_max": float(segment.y.max()), "peak_time": peak_row.target_time,
                     "alarm_start": alarm_start, "duration_intervals": len(segment),
                     "overlapping_alarm_count": len(overlapping_alarms[i]),
                     "missed_reason": missed_reason})
    for j, (b_start, b_end) in enumerate(alarms):
        if j not in matched_alarms:
            rows.append({"status": "FP", "start": b_start, "end": b_end,
                         "actual_max": float("nan"), "peak_time": pd.NaT,
                         "alarm_start": b_start, "duration_intervals": int((b_end-b_start)/pd.Timedelta(minutes=15))+1,
                         "overlapping_alarm_count": np.nan, "missed_reason": ""})
    return pd.DataFrame(rows, columns=["status", "start", "end", "actual_max", "peak_time",
                                       "alarm_start", "duration_intervals", "overlapping_alarm_count",
                                       "missed_reason"])


def condition_errors(pred: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    p = attach_history(pred, history)
    p["actual_peak"] = p.y > p.tau
    p["alert_flag"] = pd.to_numeric(p.get("alert", p.pred > p.tau), errors="coerce").fillna(0).astype(bool)
    p["state"] = np.select(
        [p.actual_peak & p.alert_flag, p.actual_peak & ~p.alert_flag, ~p.actual_peak & p.alert_flag],
        ["TP", "FN", "FP"], default="TN")
    p["abs_error"] = (p.y - p.pred).abs()
    t = p.target_time.dt
    p["time_block"] = pd.cut(t.hour, bins=[-1, 7, 15, 23], labels=["00-08", "08-16", "16-24"])
    p["weekday"] = t.day_name()
    p["month"] = t.month
    offday = p["is_offday"].astype(bool) if "is_offday" in p else calendar_flags(pd.DatetimeIndex(p.target_time))["is_offday"].to_numpy(dtype=bool)
    p["offday"] = np.where(offday, "offday", "workday")
    if "production_target" in p:
        numeric = pd.to_numeric(p.production_target, errors="coerce")
        # Keep equal production values together; ties can legitimately leave
        # fewer than five distinct quantile bins.
        p["production_quintile"] = pd.qcut(numeric, 5, duplicates="drop").astype(str)
    if "tariff_2021_band" in p:
        p["old_tariff_band"] = p.tariff_2021_band
    rows = []
    for factor in ("time_block", "weekday", "month", "offday", "production_quintile", "old_tariff_band"):
        if factor not in p:
            continue
        for key, group in p.groupby(factor, observed=True, dropna=False):
            peaks = int(group.actual_peak.sum())
            nonpeaks = len(group) - peaks
            rows.append({"factor": factor, "value": str(key), "n": len(group),
                         "mae": float(group.abs_error.mean()), "peak_n": peaks,
                         "tp": int(group.state.eq("TP").sum()), "fn": int(group.state.eq("FN").sum()),
                         "fp": int(group.state.eq("FP").sum()),
                         "peak_recall": float(group.state.eq("TP").sum()/peaks) if peaks else np.nan,
                         "false_alert_rate": float(group.state.eq("FP").sum()/nonpeaks) if nonpeaks else np.nan})
    return pd.DataFrame(rows)


def evening_misses(pred: pd.DataFrame, history: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the four hours before actual episode starts; case selection uses OOF FN only."""
    history = history.sort_index()
    evening = events.loc[events.status.isin(["TP", "FN"]) & events.start.dt.hour.ge(16)]
    rows = []
    for event in evening.itertuples(index=False):
        before = history.loc[(history.index >= event.start-pd.Timedelta(hours=4)) & (history.index < event.start)]
        if before.empty:
            continue
        power = pd.to_numeric(before.power, errors="coerce")
        prod = pd.to_numeric(before.get("production_target", pd.Series(dtype=float)), errors="coerce")
        rows.append({"status": event.status, "start": event.start, "actual_max": event.actual_max,
                     "overlapping_alarm_count": event.overlapping_alarm_count,
                     "missed_reason": event.missed_reason,
                     "weekday": event.start.day_name(), "lookback_n": len(before),
                     "power_trend_4h": float(power.iloc[-1]-power.iloc[0]),
                     "power_mean_4h": float(power.mean()),
                     "production_change_4h": float(prod.iloc[-1]-prod.iloc[0]) if len(prod) else np.nan})
    details = pd.DataFrame(rows)
    if details.empty:
        return details, pd.DataFrame()
    compare = details.groupby("status", as_index=False).agg(
        episodes=("start", "size"), power_trend_4h=("power_trend_4h", "median"),
        production_change_4h=("production_change_4h", "median"),
        power_mean_4h=("power_mean_4h", "median"))
    return details, compare


def plot_evening_cases(pred: pd.DataFrame, history: pd.DataFrame, details: pd.DataFrame, path: Path) -> Path | None:
    if details.empty:
        return None
    cases = details.loc[details.status.eq("FN")].sort_values("actual_max", ascending=False).head(3)
    if cases.empty:
        return None
    configure()
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.7*len(cases)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases.itertuples(index=False)):
        window = pred.loc[pred.target_time.between(case.start-pd.Timedelta(hours=4), case.start+pd.Timedelta(hours=2))]
        ax.plot(window.target_time, window.y, label="실제", color="#1C2A39")
        ax.plot(window.target_time, window.pred, label="예측", color="#3E6283")
        for j, (alarm_start, alarm_end) in enumerate(episodes(window.target_time, window.alert_flag)):
            half = pd.Timedelta(minutes=7.5)
            ax.axvspan(alarm_start-half, alarm_end+half, color="#D8EDE7", alpha=.65,
                       label="선택 위험 경보" if j == 0 else None, zorder=0)
        ax.axhline(float(window.tau.iloc[0]), color="#B8741A", linestyle="--", label="피크 임계")
        ax.axvline(case.start, color="#B8741A", alpha=0.6)
        reason = ("경보 에피소드 1:1 매칭" if case.missed_reason == "one_to_one_assignment"
                  else "경보 겹침 없음")
        ax.set_title(f"미탐 사례 {case.start:%Y-%m-%d %H:%M} · {reason}  |  실제 최대 {case.actual_max:.1f}")
        ax.set_ylabel("전력 원자료값")
        ax.grid(True)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.text(.5, .012, "1:1 매칭: 겹친 경보가 다른 실제 피크에 배정되면 미탐으로 집계합니다.",
             ha="center", fontsize=9, color="#4F5F70")
    return save(fig, path)


def run_errors(pred: pd.DataFrame, history: pd.DataFrame, outdir: Path) -> dict:
    tables, figures = outdir/"tables", outdir/"figures"
    if "alert" not in pred:
        pred = pred.assign(alert=(pred.pred > pred.tau).astype(int))
    pred = pred.assign(alert_flag=pd.to_numeric(pred.alert, errors="coerce").fillna(0).astype(bool))
    conditions = condition_errors(pred, history)
    event_table = match_episode_table(pred)
    details, compare = evening_misses(pred, history, event_table)
    paths = {
        "conditions": str(write_table(conditions, tables/"error_conditions.csv")),
        "episodes": str(write_table(event_table, tables/"error_episodes.csv")),
        "evening_details": str(write_table(details, tables/"evening_episode_details.csv")),
        "evening_comparison": str(write_table(compare, tables/"evening_episode_comparison.csv")),
    }
    fig = plot_evening_cases(pred, history, details, figures/"evening_missed_cases.png")
    if fig:
        paths["evening_cases_figure"] = str(fig)
    return {"paths": paths, "evening_fn": int((details.status == "FN").sum()) if not details.empty else 0,
            "evening_tp": int((details.status == "TP").sum()) if not details.empty else 0,
            "warning": "조건별 비교는 탐색적 사후 분석이며 교대 및 인과 효과를 뜻하지 않습니다."}
