"""Retrospective h4 load shift D and curtailment upper bound C (v2).

Intervals are (start, end] in the source data. A one-hour window beginning
at 18:00 therefore has four end stamps 18:15, 18:30, 18:45 and 19:00.
Neither scenario identifies a causal production-control effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import calendar

import numpy as np
import pandas as pd

from ._common import write_table
from .energy_baseline_v2 import EnergyV2Result, TEST_ORIGIN, _fold_metadata, _safe_history
from .tariff import classify_tariff, discount_factor, tariff_weight_basis


SCENARIO_LABEL = "사후 관측 생산량 기반 사후 가정 시나리오"
DEMAND_SCOPE = "partial_month_development_oof"


@dataclass
class ShiftV2Result:
    alerts: pd.DataFrame
    actions: pd.DataFrame
    monthly: pd.DataFrame
    summary: pd.DataFrame
    paths: dict[str, str] = field(default_factory=dict)


def prepare_h4_operational_oof(oof: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Attach selected h4 risk estimates; retain original F1 alert for audit.

    The operating trigger is the existing decision-feed rule 1/1:
    P(exceed)>0.1 OR calibrated q95>fold tau. It is intentionally distinct
    from the quantile model's F1-optimised `alert` cutoff.
    """
    choice = selection.get("by_horizon", {}).get("4", {})
    point_name = choice.get("point_model")
    risk_name = f"lgbm_quantile_{choice.get('conformal', 'a')}"
    if not point_name or oof.empty:
        raise ValueError("Frozen development h4 point choice and OOF rows required")
    x = oof.loc[oof.horizon.eq(4)].copy()
    keys = ["origin", "target_time", "horizon", "fold"]
    point = x.loc[x.model.eq(point_name), keys+["y", "pred", "tau"]]
    risk = x.loc[x.model.eq(risk_name), keys+["p_exceed", "q95_cal", "alert"]]
    if point.empty or risk.empty or point.duplicated(keys).any() or risk.duplicated(keys).any():
        raise ValueError("Unique selected point and risk OOF rows are required")
    merged = point.merge(risk, on=keys, how="left", validate="one_to_one")
    if merged[["p_exceed", "q95_cal"]].isna().all(axis=1).any():
        raise ValueError("Selected h4 risk OOF rows do not align with point rows")
    merged["origin"] = pd.to_datetime(merged.origin)
    merged["target_time"] = pd.to_datetime(merged.target_time)
    if merged.origin.ge(TEST_ORIGIN).any() or merged.target_time.ge(TEST_ORIGIN).any():
        raise ValueError("Frozen test boundary reached by v2 OOF")
    merged["risk_alert_original"] = merged.alert
    merged["alert"] = (pd.to_numeric(merged.p_exceed, errors="coerce").gt(.1) |
                       pd.to_numeric(merged.q95_cal, errors="coerce").gt(merged.tau))
    merged["point_model"] = point_name
    merged["risk_model"] = risk_name
    return merged.sort_values(["fold", "origin"]).reset_index(drop=True)


def _validate_operational(history: pd.DataFrame, operational: pd.DataFrame,
                          metadata: dict[int, dict], energy: EnergyV2Result) -> pd.DataFrame:
    required = {"fold", "origin", "target_time", "horizon", "y", "tau", "p_exceed", "q95_cal", "alert"}
    if required-set(operational):
        raise ValueError(f"Operational OOF lacks {sorted(required-set(operational))}")
    p = operational.copy()
    p["origin"] = pd.to_datetime(p.origin)
    p["target_time"] = pd.to_datetime(p.target_time)
    if p.empty or not p.horizon.eq(4).all() or p.origin.ge(TEST_ORIGIN).any():
        raise ValueError("Only development h4 OOF actions allowed")
    if p.duplicated(["fold", "target_time"]).any() or p.target_time.duplicated().any():
        raise ValueError("Duplicate fold OOF target")
    if not p.target_time.isin(history.index).all() or p.target_time.ge(TEST_ORIGIN).any():
        raise ValueError("OOF target outside safe history")
    if "time_repaired" in history and history.loc[pd.DatetimeIndex(p.target_time),
                                                  "time_repaired"].fillna(True).astype(bool).any():
        raise ValueError("Repaired target cannot enter v2 OOF action analysis")
    expected = pd.to_numeric(p.p_exceed, errors="coerce").gt(.1) | pd.to_numeric(p.q95_cal, errors="coerce").gt(p.tau)
    if not np.array_equal(p.alert.fillna(False).astype(bool), expected):
        raise ValueError("h4 action trigger differs from frozen decision-feed rule 1/1")
    if set(p.fold.astype(int)) != set(energy.models):
        raise ValueError("Energy models and operational folds differ")
    for fold, group in p.groupby("fold"):
        row = metadata.get(int(fold))
        if row is None:
            raise ValueError("Missing fold metadata")
        start, end = pd.Timestamp(row["score_start"]), pd.Timestamp(row["score_end"])
        if group.origin.lt(start).any() or group.origin.gt(end).any() or group.tau.nunique() != 1:
            raise ValueError("OOF row or peak threshold conflicts with its fold metadata")
        if not np.allclose(history.loc[pd.DatetimeIndex(group.target_time), "power"], group.y,
                           atol=1e-10, rtol=1e-10):
            raise ValueError("OOF target and safe history differ")
    return p.sort_values(["origin", "fold"])


def _source_start(target_end: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(target_end)-pd.Timedelta(nanoseconds=1)).floor("h")


def _full_windows(index: pd.DatetimeIndex, fold_by_time: dict[pd.Timestamp, int],
                  valid: np.ndarray, weight: np.ndarray) -> dict[tuple[int, pd.Timestamp], tuple[np.ndarray, float]]:
    position = {stamp: i for i, stamp in enumerate(index)}
    windows = {}
    for first_end in index:
        start = first_end-pd.Timedelta(minutes=15)
        stamps = pd.date_range(start+pd.Timedelta(minutes=15), periods=4, freq="15min")
        places = [position.get(stamp) for stamp in stamps]
        if any(i is None for i in places):
            continue
        places = np.asarray(places, dtype=int)
        fold = fold_by_time[first_end]
        if (not all(fold_by_time.get(stamp) == fold for stamp in stamps) or
            not valid[places].all() or not np.isfinite(weight[places]).all()):
            continue
        windows[(fold, start)] = places, float(weight[places].mean())
    return windows


def _alert_sources(operational: pd.DataFrame, windows: dict,
                   prep: int) -> tuple[pd.DataFrame, list[dict]]:
    records, selected = [], []
    alerted = operational.loc[operational.alert.astype(bool)].sort_values(["origin", "fold"])
    order_by_source: dict[tuple[int, pd.Timestamp], int] = {}
    for row in alerted.itertuples(index=False):
        fold = int(row.fold)
        source = _source_start(row.target_time)
        key = (fold, source)
        if source < row.origin+pd.Timedelta(minutes=prep):
            reason = "source_before_preparation"
        elif key not in windows:
            reason = "source_outside_or_invalid_oof_window"
        else:
            reason = "source_eligible"
            order_by_source[key] = order_by_source.get(key, 0)+1
            selected.append({"fold": fold, "origin": row.origin, "target_time": row.target_time,
                             "source_start": source, "tau": float(row.tau),
                             "eligible_alert_order_for_source": order_by_source[key]})
        records.append({"fold": fold, "origin": row.origin, "target_time": row.target_time,
                        "source_start": source, "prep_minutes": prep, "source_status": reason,
                        "eligible_alert_order_for_source": order_by_source.get(key, np.nan)
                        if reason == "source_eligible" else np.nan})
    return pd.DataFrame(records), selected


def _candidate_starts(origin: pd.Timestamp, source_start: pd.Timestamp, prep: int):
    # Delayed movement starts after the complete source hour, never during it.
    first = max(source_start+pd.Timedelta(hours=1), origin+pd.Timedelta(minutes=prep))
    first = first.ceil("15min")
    last = origin+pd.Timedelta(hours=23)
    if first > last:
        return []
    return pd.date_range(first, last, freq="15min")


def _simulate_one(scenario: str, fraction: float, prep: int, sources: list[dict],
                  history: pd.DataFrame, index: pd.DatetimeIndex,
                  windows: dict, original: np.ndarray, weight: np.ndarray,
                  energy: EnergyV2Result) -> tuple[list[dict], np.ndarray]:
    moved = original.copy()
    rows = []
    applied_sources: set[tuple[int, pd.Timestamp]] = set()
    for source in sources:
        fold, start = source["fold"], source["source_start"]
        src, source_weight = windows[(fold, start)]
        result = {"scenario": scenario, "fraction": fraction, "prep_minutes": prep,
                  **source, "destination_start": pd.NaT,
                  "delta_per_interval": np.nan, "status": "unsupported"}
        key = (fold, start)
        if key in applied_sources:
            result["status"] = "duplicate_after_applied"
            rows.append(result)
            continue
        if scenario == "C":
            delta = moved[src]*fraction
            moved[src] -= delta
            result.update(status="applied", delta_per_interval=float(delta.mean()))
            applied_sources.add(key)
            rows.append(result)
            continue
        coefficient = energy.models[fold].production_coefficient_per_hour
        if not np.isfinite(coefficient) or coefficient <= 0:
            result["status"] = "nonpositive_production_coefficient"
            rows.append(result)
            continue
        source_times = index[src]
        prod = pd.to_numeric(history.loc[source_times, "production_target"], errors="coerce")
        if (prod.isna().any() or not np.isfinite(prod).all() or prod.nunique() != 1 or prod.iloc[0] <= 0):
            result["status"] = "invalid_observed_source_production"
            rows.append(result)
            continue
        delta = float(coefficient*prod.iloc[0]*fraction/4)
        if (moved[src] < delta).any():
            result["status"] = "source_load_insufficient"
            rows.append(result)
            continue
        saw_full = saw_cheaper = False
        for candidate in _candidate_starts(source["origin"], start, prep):
            found = windows.get((fold, candidate))
            if found is None:
                continue
            saw_full = True
            dst, destination_weight = found
            if destination_weight >= source_weight:
                continue
            saw_cheaper = True
            if (moved[dst]+delta > source["tau"]).any():
                continue
            moved[src] -= delta
            moved[dst] += delta
            applied_sources.add(key)
            result.update(status="applied", destination_start=candidate,
                          delta_per_interval=delta, observed_production=float(prod.iloc[0]),
                          source_weight=source_weight, destination_weight=destination_weight)
            break
        if result["status"] != "applied":
            result["status"] = ("no_future_valid_oof_window" if not saw_full else
                                "no_cheaper_window" if not saw_cheaper else "destination_capacity_exceeded")
        rows.append(result)
    return rows, moved


def run_shift_v2(history: pd.DataFrame, operational: pd.DataFrame,
                 fold_metadata: list[dict] | pd.DataFrame, energy: EnergyV2Result,
                 tariff_2026: dict, outdir: str | Path | None = None, *,
                 fractions=(.1, .2, .3), prep_minutes=(15, 30, 60)) -> ShiftV2Result:
    """Run disjoint retrospective D/C scenarios on same-fold OOF target windows."""
    history = _safe_history(history)
    metadata = _fold_metadata(fold_metadata)
    p = _validate_operational(history, operational, metadata, energy)
    fractions, prep_minutes = tuple(fractions), tuple(prep_minutes)
    if any(not 0 < float(f) < 1 for f in fractions) or any(int(t) not in (15, 30, 60) for t in prep_minutes):
        raise ValueError("Only preregistered fractions and preparation times are allowed")
    if set(np.round(fractions, 8))-set([.1, .2, .3]):
        raise ValueError("Only 10/20/30 percent scenarios are preregistered")
    weight_map, basis = tariff_weight_basis(tariff_2026)
    if weight_map is None:
        raise ValueError("Verified rates or declared ordinal scenario weights required")
    index = pd.DatetimeIndex(p.target_time).sort_values().unique()
    observed = history.loc[index]
    original = pd.to_numeric(observed.power, errors="coerce").to_numpy(float)
    fold_by_time = dict(zip(p.target_time, p.fold.astype(int)))
    band = classify_tariff(index, tariff_2026)
    weight = band.map(weight_map).to_numpy(float)*discount_factor(index, tariff_2026)
    if not np.isfinite(weight).all():
        raise ValueError("Tariff scenario has missing or nonfinite interval weights")
    valid = np.isfinite(original)
    valid &= np.isfinite(pd.to_numeric(observed.production_target, errors="coerce").to_numpy(float))
    if "time_repaired" in observed:
        valid &= ~observed.time_repaired.fillna(False).astype(bool).to_numpy()
    windows = _full_windows(index, fold_by_time, valid, weight)
    alert_rows, all_actions, month_rows, summary_rows = [], [], [], []
    tau_by_time = p.set_index("target_time").loc[index, "tau"].to_numpy(float)
    month = (index-pd.Timedelta(minutes=15)).to_period("M")
    for prep in prep_minutes:
        audit, sources = _alert_sources(p, windows, int(prep))
        alert_rows.append(audit)
        total_alerts = len(audit)
        for scenario in ("D", "C"):
            for fraction in fractions:
                actions, moved = _simulate_one(scenario, float(fraction), int(prep), sources,
                                                history, index, windows, original, weight, energy)
                all_actions.extend(actions)
                applied = sum(row["status"] == "applied" for row in actions)
                energy_change = float(np.sum(moved-original))
                if scenario == "D" and not np.isclose(energy_change, 0, atol=1e-9, rtol=1e-10):
                    raise AssertionError("Delayed movement did not preserve the total load proxy")
                key = {"scenario": scenario, "fraction": float(fraction),
                       "prep_minutes": int(prep), "weight_basis": basis,
                       "relative_cost_only": True,
                       "alert_positions": total_alerts,
                       "alert_positions_including_duplicate_sources": total_alerts,
                       "eligible_unique_source_windows": len({(s["fold"], s["source_start"]) for s in sources}),
                       "applied_unique_source_windows": applied,
                       "actionable_alert_fraction": applied/total_alerts if total_alerts else np.nan,
                       "actionable_fraction_denominator": "all_h4_alert_positions_including_duplicate_sources",
                       "load_proxy_total_change": energy_change,
                       "energy_conserved": bool(np.isclose(energy_change, 0, atol=1e-9)) if scenario == "D" else False}
                summary_rows.append(key)
                for period in sorted(month.unique()):
                    mask = month == period
                    before_max = float(np.max(original[mask]))
                    after_max = float(np.max(moved[mask]))
                    full_calendar_intervals = calendar.monthrange(period.year, period.month)[1]*96
                    month_rows.append({**key, "month": str(period),
                                       "monthly_max_before": before_max,
                                       "monthly_max_after": after_max,
                                       "monthly_max_change": after_max-before_max,
                                       "weighted_load_change": float(np.sum((moved[mask]-original[mask])*weight[mask])),
                                       "new_peak_positions": int(np.sum(
                                           (moved[mask] > tau_by_time[mask]) & (original[mask] <= tau_by_time[mask]))),
                                       "evaluated_intervals": int(mask.sum()),
                                       "full_calendar_intervals": full_calendar_intervals,
                                       "observed_fraction_of_full_month": float(mask.sum()/full_calendar_intervals)})
    result = ShiftV2Result(pd.concat(alert_rows, ignore_index=True) if alert_rows else pd.DataFrame(),
                           pd.DataFrame(all_actions), pd.DataFrame(month_rows), pd.DataFrame(summary_rows))
    for frame in (result.alerts, result.actions, result.monthly, result.summary):
        frame["scenario_label"] = SCENARIO_LABEL
        frame["demand_scope"] = DEMAND_SCOPE
    if outdir is not None:
        tables = Path(outdir)/"tables"
        for name, data in (("shift_alert_audit_v2", result.alerts),
                           ("shift_actions_v2", result.actions),
                           ("shift_monthly_v2", result.monthly),
                           ("shift_summary_v2", result.summary)):
            result.paths[name] = str(write_table(data, tables/f"{name}.csv"))
    return result
