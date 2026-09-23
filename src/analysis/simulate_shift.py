"""Retrospective same-day load-shift sensitivity, with operational feasibility gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import write_table
from .energy_baseline import EnergyBaseline
from .tariff import classify_tariff, discount_factor, tariff_weight_basis


def _hour(stamp: pd.Timestamp) -> pd.Timestamp:
    return (stamp-pd.Timedelta(nanoseconds=1)).floor("h")


def simulate_shift(pred: pd.DataFrame, history: pd.DataFrame, train: pd.DataFrame,
                   baseline: EnergyBaseline, tariff: dict,
                   fractions=(0.1, 0.2, 0.3), prep_minutes: int = 30,
                   destination_start: int = 11, destination_end: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline.production_coefficient_per_hour <= 0:
        raise ValueError("Non-positive explanatory production coefficient cannot support a load reduction scenario")
    if "production_target" not in history:
        raise ValueError("Hourly production is unavailable")
    p = pred.loc[pred.horizon.eq(4)].copy()
    if p.empty:
        raise ValueError("No 1-hour OOF forecasts for action stage")
    prob = pd.to_numeric(p.get("p_exceed", pd.Series(np.nan, index=p.index)), errors="coerce")
    q95 = pd.to_numeric(p.get("q95_cal", pd.Series(np.nan, index=p.index)), errors="coerce")
    p["action"] = prob.gt(0.1) | q95.gt(p.tau)
    p["source_hour"] = p.target_time.map(_hour)
    p = p.loc[p.action].sort_values("origin").drop_duplicates("source_hour")
    train_slots = train.copy()
    train_slots["slot_hour"] = train_slots.index.hour
    slot_medians = train_slots.groupby("slot_hour").power.median()
    bands = classify_tariff(history.index, tariff)
    weight_map, basis = tariff_weight_basis(tariff)
    if weight_map is None:
        raise ValueError("No rate or declared scenario weight is available to rank cheaper target hours")
    candidate_hours = list(range(destination_start, destination_end))
    records = []
    for row in p.itertuples(index=False):
        source_hour = row.source_hour
        source_times = pd.date_range(source_hour+pd.Timedelta(minutes=15), periods=4, freq="15min")
        if not source_times.isin(history.index).all():
            continue
        if "time_repaired" in history and history.loc[source_times, "time_repaired"].astype(bool).any():
            continue
        if not np.isfinite(pd.to_numeric(history.loc[source_times, "power"], errors="coerce")).all():
            continue
        prod = pd.to_numeric(history.loc[source_times, "production_target"], errors="coerce")
        if prod.isna().any() or prod.iloc[0] <= 0:
            continue
        source_band = bands.reindex(source_times).iloc[0]
        source_effective_weight = weight_map[source_band]*float(np.mean(discount_factor(source_times, tariff)))
        available_after = row.origin + pd.Timedelta(minutes=prep_minutes)
        dests = []
        for hour in candidate_hours:
            start = source_hour.normalize()+pd.Timedelta(hours=hour)
            times = pd.date_range(start+pd.Timedelta(minutes=15), periods=4, freq="15min")
            if start <= available_after or start <= source_hour or not times.isin(history.index).all():
                continue
            if "time_repaired" in history and history.loc[times, "time_repaired"].astype(bool).any():
                continue
            if not np.isfinite(pd.to_numeric(history.loc[times, "power"], errors="coerce")).all():
                continue
            if len(set(bands.reindex(times).dropna())) != 1:
                continue
            dest_band = bands.reindex(times).iloc[0]
            dest_effective_weight = weight_map[dest_band]*float(np.mean(discount_factor(times, tariff)))
            if dest_effective_weight >= source_effective_weight:
                continue
            dests.append((float(slot_medians.get(hour, np.inf)), start, times))
        if not dests:
            records.append({"origin": row.origin, "source_hour": source_hour,
                            "destination_hour": pd.NaT, "status": "no_future_low_band_window",
                            "production": prod.iloc[0]})
            continue
        _, dest_hour, dest_times = min(dests)
        records.append({"origin": row.origin, "source_hour": source_hour,
                        "destination_hour": dest_hour, "status": "retrospective_feasible_time",
                        "production": float(prod.iloc[0])})
    actions = pd.DataFrame(records)
    if actions.empty:
        return actions, pd.DataFrame()
    summaries = []
    original = history.power.astype(float)
    for fraction in fractions:
        moved = original.copy()
        applied = 0
        for act in actions.loc[actions.status.eq("retrospective_feasible_time")].itertuples(index=False):
            src = pd.date_range(act.source_hour+pd.Timedelta(minutes=15), periods=4, freq="15min")
            dst = pd.date_range(act.destination_hour+pd.Timedelta(minutes=15), periods=4, freq="15min")
            delta = baseline.production_coefficient_per_hour*act.production*float(fraction)/4
            if (moved.loc[src] < delta).any():
                continue
            moved.loc[src] -= delta
            moved.loc[dst] += delta
            applied += 1
        before_month = original.groupby(original.index.to_period("M")).max()
        after_month = moved.groupby(moved.index.to_period("M")).max()
        for month in before_month.index:
            month_mask = moved.index.to_period("M") == month
            month_tau = float(pred.loc[pred.target_time.dt.to_period("M").eq(month), "tau"].median())
            entry = {"fraction": float(fraction), "month": str(month),
                     "applied_source_hours": applied,
                     "max_before": float(before_month[month]), "max_after": float(after_month[month]),
                     "max_change": float(after_month[month]-before_month[month]),
                     "new_peak_positions": int(((moved > month_tau) & (original <= month_tau) & month_mask).sum()) if np.isfinite(month_tau) else np.nan,
                     "peak_threshold_proxy": month_tau,
                     "weight_basis": basis}
            if weight_map is not None:
                weights = bands.map(weight_map).to_numpy(float)*discount_factor(history.index, tariff)
                if np.isfinite(weights).all():
                    entry["weighted_load_change"] = float(np.sum((moved-original).to_numpy()[month_mask]*weights[month_mask]))
            summaries.append(entry)
    return actions, pd.DataFrame(summaries)


def run_simulate_shift(pred: pd.DataFrame, history: pd.DataFrame, train: pd.DataFrame,
                       baseline: EnergyBaseline | None, tariff: dict | None,
                       outdir: Path, cfg: dict) -> dict:
    if baseline is None or tariff is None:
        return {"status": "unsupported", "reason": "Energy baseline and 2026 band schedule are required"}
    try:
        actions, summary = simulate_shift(pred, history, train, baseline, tariff,
            fractions=cfg.get("shift", {}).get("fractions", [0.1, 0.2, 0.3]),
            prep_minutes=int(cfg.get("alert", {}).get("default_prep_minutes", 30)))
    except ValueError as exc:
        return {"status": "unsupported", "reason": str(exc)}
    paths = {"shift_actions": str(write_table(actions, outdir/"tables"/"shift_actions.csv")),
             "shift_summary": str(write_table(summary, outdir/"tables"/"shift_summary.csv"))}
    return {"status": "ok" if not summary.empty else "unsupported", "paths": paths,
            "time_feasible_actions": int(actions.status.eq("retrospective_feasible_time").sum()) if not actions.empty else 0,
            "no_future_low_band_window": int(actions.status.eq("no_future_low_band_window").sum()) if not actions.empty else 0,
            "warning": "사후 관측 생산량으로 계산한 가상 시나리오이며 설비 제약, 실제 이동 가능성 및 인과 효과는 검증되지 않았습니다. 오후 경보는 이미 지난 11~15시로 이동할 수 없습니다."}
