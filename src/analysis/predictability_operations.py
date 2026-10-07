"""Preregistered P1 A4/A5 development-only operating diagnostics.

A4's restart indicator is a retrospective *outcome condition*, never a
forecast feature. A5 reuses the exact v2 source/window/action primitives;
the only generalisation is selecting the h16 OOF and its fold-local Ridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import calendar

import numpy as np
import pandas as pd

from src.holidays import calendar_flags
from src.session_data import SEALED_BOUNDARY
from ._common import write_table
from .energy_baseline import fit_energy_baseline
from .energy_baseline_v2 import FoldEnergyModelV2, _safe_history
from .errors import match_episode_table
from .shift_v2 import (
    _alert_sources, _candidate_starts, _full_windows, _simulate_one, _source_start,
)
from .tariff import classify_tariff, discount_factor, tariff_weight_basis


N_BOOT = 1000
SEED = 42
SCENARIO_LABEL = "사후 관측 생산량 기반 가정 시나리오"


def _fold_meta(manifest: dict, horizon: int) -> dict[int, dict]:
    rows = manifest.get("folds", [])
    selected = [r for r in rows if int(r["horizon"]) == horizon]
    result = {int(r["fold"]): r for r in selected}
    if len(result) != 3 or len(selected) != 3:
        raise ValueError(f"Three unique original h{horizon} folds required")
    for r in result.values():
        if pd.Timestamp(r["score_end"]) >= SEALED_BOUNDARY:
            raise ValueError("Fold scoring range reaches sealed test boundary")
    return result


def _verify_oof(frame: pd.DataFrame, history: pd.DataFrame, horizon: int,
                metadata: dict[int, dict]) -> pd.DataFrame:
    required = {"origin", "target_time", "horizon", "fold", "y", "pred", "tau"}
    if required - set(frame):
        raise ValueError(f"OOF lacks {sorted(required - set(frame))}")
    x = frame.copy()
    x["origin"] = pd.to_datetime(x.origin)
    x["target_time"] = pd.to_datetime(x.target_time)
    if x.empty or not x.horizon.eq(horizon).all():
        raise ValueError("OOF horizon mismatch or empty frame")
    if x.origin.ge(SEALED_BOUNDARY).any() or x.target_time.ge(SEALED_BOUNDARY).any():
        raise ValueError("Test boundary reached by development OOF")
    if x.duplicated(["fold", "target_time"]).any() or x.target_time.duplicated().any():
        raise ValueError("Duplicate OOF target")
    if not (x.target_time - x.origin).eq(pd.Timedelta(minutes=15*horizon)).all():
        raise ValueError("OOF forecast horizon does not match timestamps")
    if not x.target_time.isin(history.index).all():
        raise ValueError("OOF target is missing from safe history")
    source = history.loc[pd.DatetimeIndex(x.target_time)]
    if "time_repaired" in source and source.time_repaired.fillna(True).astype(bool).any():
        raise ValueError("Repaired target in OOF")
    if not np.allclose(pd.to_numeric(source.power, errors="coerce"), x.y,
                       atol=1e-10, rtol=1e-10):
        raise ValueError("OOF y differs from safe development history")
    for fold, group in x.groupby("fold"):
        meta = metadata.get(int(fold))
        if meta is None:
            raise ValueError("OOF fold missing from original manifest")
        if (group.origin.lt(pd.Timestamp(meta["score_start"])).any() or
                group.origin.gt(pd.Timestamp(meta["score_end"])).any() or
                group.tau.nunique() != 1):
            raise ValueError("OOF row or tau conflicts with original fold")
    return x.sort_values(["fold", "target_time"]).reset_index(drop=True)


def _daily_bootstrap(frame: pd.DataFrame, statistic, *, n=N_BOOT, seed=SEED):
    if n != N_BOOT or seed != SEED:
        raise ValueError("P1 requires 1000 day blocks and seed 42")
    estimate = float(statistic(frame))
    if not np.isfinite(estimate):
        return estimate, np.nan, np.nan, 1.0, 0, n
    days = pd.DatetimeIndex(frame.target_time).normalize()
    blocks = [frame.loc[days == day] for day in days.unique()]
    if len(blocks) < 2:
        return estimate, np.nan, np.nan, 1.0, 0, n
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)
    for i in range(n):
        sampled = pd.concat([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))],
                            ignore_index=True)
        samples[i] = float(statistic(sampled))
    good = samples[np.isfinite(samples)]
    missing = n-len(good)
    if len(good) < 2:
        return estimate, np.nan, np.nan, 1.0, len(good), missing
    low, high = np.quantile(good, [.025, .975])
    # One-sided positive concentration, from the preregistered centered null.
    # Undefined draws (no restart observation after day resampling) count as
    # non-rejections. The percentile CI is conditional on definable draws.
    p = (1 + np.count_nonzero(good - estimate >= estimate) + missing) / (n + 1)
    return estimate, float(low), float(high), float(p), len(good), missing


def _paired_original_models(oof: pd.DataFrame, manifest: dict, horizon: int,
                            history: pd.DataFrame) -> pd.DataFrame:
    history = _safe_history(history)
    metadata = _fold_meta(manifest, horizon)
    cbl_name = manifest["selection"]["by_horizon"][str(horizon)]["cbl"]
    keys = ["origin", "target_time", "horizon", "fold"]
    x = oof.loc[oof.horizon.eq(horizon)]
    cbl = x.loc[x.model.eq(cbl_name), keys + ["y", "pred", "tau"]]
    lgbm = x.loc[x.model.eq("lgbm"), keys + ["y", "pred", "tau"]]
    if cbl.empty or lgbm.empty or cbl.duplicated(keys).any() or lgbm.duplicated(keys).any():
        raise ValueError("Original CBL and full LightGBM rows must be unique")
    joined = cbl.merge(lgbm, on=keys, how="inner", suffixes=("_cbl", "_lgbm"),
                       validate="one_to_one")
    if joined.empty or not np.allclose(joined.y_cbl, joined.y_lgbm, equal_nan=True) or not np.allclose(
            joined.tau_cbl, joined.tau_lgbm, equal_nan=True):
        raise ValueError("Paired original models disagree on actual or tau")
    joined = joined.rename(columns={"y_cbl": "y", "pred_cbl": "cbl_pred",
                                    "pred_lgbm": "lgbm_pred", "tau_cbl": "tau"})
    joined = joined.drop(columns=["y_lgbm", "tau_lgbm"])
    joined = joined.loc[joined[["y", "cbl_pred", "lgbm_pred", "tau"]].notna().all(axis=1)]
    return _verify_oof(joined.assign(pred=joined.lgbm_pred), history, horizon, metadata)


def run_a4(history: pd.DataFrame, oof: pd.DataFrame, manifest: dict,
           restart_api, outdir: str | Path | None = None) -> dict:
    """A4 full-LightGBM minus original CBL error, stratified by restart."""
    history = _safe_history(history)
    fit_thresholds, detect_events = restart_api
    details, fold_rows, contrasts, breakdown = [], [], [], []
    for horizon in (16, 96):
        x = _paired_original_models(oof, manifest, horizon, history)
        metadata = _fold_meta(manifest, horizon)
        ends = pd.DatetimeIndex(x.target_time)
        x["day_type"] = np.where(calendar_flags(ends).is_offday.to_numpy(bool),
                                 "offday", "workday")
        x["slot"] = ends.hour*4 + ends.minute//15
        x["cbl_abs_error"] = (x.y-x.cbl_pred).abs()
        x["lgbm_abs_error"] = (x.y-x.lgbm_pred).abs()
        x["error_diff_lgbm_minus_cbl"] = x.lgbm_abs_error-x.cbl_abs_error
        x["restart"] = False
        for fold, group in x.groupby("fold"):
            meta = metadata[int(fold)]
            thresholds = fit_thresholds(history, pd.Timestamp(meta["fit_end"]))
            events = detect_events(history, thresholds, start=group.target_time.min(),
                                   end=group.target_time.max())
            x.loc[group.index, "restart"] = group.target_time.isin(events.index).to_numpy(bool)

        def contrast(part):
            yes = part.loc[part.restart, "error_diff_lgbm_minus_cbl"]
            no = part.loc[~part.restart, "error_diff_lgbm_minus_cbl"]
            return float(yes.mean()-no.mean()) if len(yes) and len(no) else np.nan

        effect, low, high, p, n_valid, n_missing = _daily_bootstrap(x, contrast)
        fold_values = []
        for fold, part in x.groupby("fold"):
            value = contrast(part)
            fold_values.append(value)
            fold_rows.append({"analysis": "A4", "horizon": horizon, "fold": int(fold),
                              "effect_restart_concentration": value,
                              "direction_positive": bool(np.isfinite(value) and value > 0),
                              "restart_rows": int(part.restart.sum()),
                              "nonrestart_rows": int((~part.restart).sum())})
        n_positive = int(sum(v > 0 for v in fold_values if np.isfinite(v)))
        contrasts.append({"analysis_id": "A4", "hypothesis_id": f"A4_h{horizon}",
                          "horizon": horizon, "estimate": effect, "ci_low": low, "ci_high": high,
                          "p_raw": p, "n_positive_folds": n_positive,
                          "eligible": bool(np.isfinite(effect) and np.isfinite(low) and
                                           sum(np.isfinite(v) for v in fold_values) >= 2),
                          "null_value": 0.0,
                          "bootstrap_draws": N_BOOT, "bootstrap_valid_draws": n_valid,
                          "bootstrap_undefined_draws": n_missing,
                          "ci_condition": "defined_resamples_only; undefined draws conservatively nonrejecting",
                          "restart_rows": int(x.restart.sum()),
                          "nonrestart_rows": int((~x.restart).sum()),
                          "direction_preregistered": "positive",
                          "interpretation": "retrospective error concentration, not feature approval"})
        for (fold, day_type, slot, restart), part in x.groupby(
                ["fold", "day_type", "slot", "restart"], dropna=False):
            breakdown.append({"analysis": "A4", "horizon": horizon, "fold": int(fold),
                              "day_type": day_type, "slot": int(slot), "restart": bool(restart),
                              "n": len(part), "peak_n": int(part.y.gt(part.tau).sum()),
                              "cbl_mae": float(part.cbl_abs_error.mean()),
                              "lgbm_mae": float(part.lgbm_abs_error.mean()),
                              "error_diff_lgbm_minus_cbl": float(part.error_diff_lgbm_minus_cbl.mean())})
        details.append(x.assign(horizon=horizon))
    result = {"paired": pd.concat(details, ignore_index=True),
              "breakdown": pd.DataFrame(breakdown),
              "folds": pd.DataFrame(fold_rows), "hypotheses": pd.DataFrame(contrasts)}
    if outdir is not None:
        target = Path(outdir)
        for key, frame in result.items():
            write_table(frame, target / f"A4_{key}.csv")
    return result


def _prepare_operational(oof: pd.DataFrame, manifest: dict, horizon: int,
                         history: pd.DataFrame) -> pd.DataFrame:
    choice = manifest["selection"]["by_horizon"][str(horizon)]
    point_name = choice["point_model"]
    risk_name = choice.get("risk_model", f"lgbm_quantile_{choice['conformal']}")
    keys = ["origin", "target_time", "horizon", "fold"]
    x = oof.loc[oof.horizon.eq(horizon)]
    point = x.loc[x.model.eq(point_name), keys+["y", "pred", "tau"]]
    risk = x.loc[x.model.eq(risk_name), keys+["p_exceed", "q95_cal", "alert"]]
    if point.empty or risk.empty or point.duplicated(keys).any() or risk.duplicated(keys).any():
        raise ValueError("Selected point and risk OOF must align uniquely")
    result = point.merge(risk, on=keys, how="left", validate="one_to_one")
    if result[["p_exceed", "q95_cal"]].isna().all(axis=1).any():
        raise ValueError("Selected risk OOF row missing")
    result["original_f1_alert"] = result.alert
    result["alert"] = (pd.to_numeric(result.p_exceed, errors="coerce").gt(.1) |
                       pd.to_numeric(result.q95_cal, errors="coerce").gt(result.tau))
    result["point_model"] = point_name
    result["risk_model"] = risk_name
    return _verify_oof(result, history, horizon, _fold_meta(manifest, horizon))


def _energy_for_horizon(history: pd.DataFrame, metadata: dict[int, dict]):
    models = {}
    for fold, meta in metadata.items():
        fit_end = pd.Timestamp(meta["fit_end"])
        trained = fit_energy_baseline(history.loc[history.index <= fit_end])
        if trained.train_end > fit_end:
            raise ValueError("Retrospective Ridge training crossed original fit_end")
        models[fold] = FoldEnergyModelV2(fold, trained.model, trained.temperature_median,
                                          trained.train_end, trained.production_coefficient_per_hour)
    return _EnergyModels(models)


@dataclass
class _EnergyModels:
    models: dict[int, FoldEnergyModelV2]


def _bootstrap_ratio(alerts: pd.DataFrame, actions: pd.DataFrame) -> tuple[float, float, float]:
    # Operational actions are already simulated in chronology. Resample their
    # resulting origin-day counts, never stitch sampled days into new episodes.
    if alerts.empty:
        return np.nan, np.nan, np.nan
    dates = pd.DatetimeIndex(alerts.origin).normalize()
    daily = alerts.assign(day=dates).groupby("day", as_index=False).size().rename(columns={"size": "alarms"})
    successful = actions.loc[actions.status.eq("applied")].copy() if "status" in actions else pd.DataFrame()
    if not successful.empty:
        successful["day"] = pd.DatetimeIndex(successful.origin).normalize()
        counts = successful.groupby("day").size().rename("applied")
        daily = daily.join(counts, on="day")
    else:
        daily["applied"] = 0
    daily["applied"] = daily.applied.fillna(0).astype(int)
    observed = float(daily.applied.sum()/daily.alarms.sum())
    rng = np.random.default_rng(SEED)
    values = np.empty(N_BOOT, dtype=float)
    for k in range(N_BOOT):
        chosen = daily.iloc[rng.integers(0, len(daily), len(daily))]
        values[k] = chosen.applied.sum()/chosen.alarms.sum()
    low, high = np.quantile(values, [.025, .975])
    return observed, float(low), float(high)


def _episode_coverage(operational: pd.DataFrame, alert_audit: pd.DataFrame) -> dict:
    eligible = (alert_audit.loc[alert_audit.source_status.eq("source_eligible"),
                                ["fold", "target_time"]].drop_duplicates()
                if "source_status" in alert_audit else
                pd.DataFrame(columns=["fold", "target_time"]))
    matched = operational[["fold", "target_time", "y", "tau"]].merge(
        eligible.assign(eligible_alert=True), on=["fold", "target_time"], how="left")
    matched["eligible_alert"] = matched.eligible_alert.fillna(False).astype(bool)
    tp = total = 0
    for (_, day), part in matched.groupby(["fold", matched.target_time.dt.normalize()]):
        if not part.y.gt(part.tau).any():
            continue
        events = match_episode_table(part.assign(alert_flag=part.eligible_alert))
        tp += int(events.status.eq("TP").sum())
        total += int(events.status.isin(["TP", "FN"]).sum())
    return {"actual_peak_episodes": total, "eligible_alarm_hit_episodes": tp,
            "eligible_alarm_episode_hit_fraction": tp/total if total else np.nan}


def _simulate_operational(history: pd.DataFrame, operational: pd.DataFrame,
                          metadata: dict[int, dict], energy: _EnergyModels,
                          tariff: dict, *, prep: int, fraction: float = .2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = _safe_history(history)
    horizon = int(operational.horizon.iloc[0])
    p = _verify_oof(operational, history, horizon, metadata)
    expected = p.p_exceed.gt(.1) | p.q95_cal.gt(p.tau)
    if "alarm_source" not in p or p.alarm_source.iloc[0] == "original_operational":
        if not np.array_equal(p.alert.to_numpy(bool), expected.to_numpy(bool)):
            raise ValueError("Operational alarm is not the frozen 1/1 trigger")
    elif p.alarm_source.iloc[0] == "perfect_information_reference":
        if not np.array_equal(p.alert.to_numpy(bool), p.y.gt(p.tau).to_numpy(bool)):
            raise ValueError("Perfect-information reference differs from actual peak labels")
    else:
        raise ValueError("Unexpected alarm source")
    weight_map, weight_basis = tariff_weight_basis(tariff)
    if weight_map is None:
        raise ValueError("Explicit ordinal scenario tariff weights required")
    index = pd.DatetimeIndex(p.target_time).sort_values().unique()
    observed = history.loc[index]
    original = pd.to_numeric(observed.power, errors="coerce").to_numpy(float)
    folds = dict(zip(p.target_time, p.fold.astype(int)))
    band = classify_tariff(index, tariff)
    weights = band.map(weight_map).to_numpy(float)*discount_factor(index, tariff)
    valid = np.isfinite(original)
    valid &= np.isfinite(pd.to_numeric(observed.production_target, errors="coerce").to_numpy(float))
    if "time_repaired" in observed:
        valid &= ~observed.time_repaired.fillna(False).astype(bool).to_numpy()
    windows = _full_windows(index, folds, valid, weights)
    audit, sources = _alert_sources(p, windows, prep)
    if audit.empty:
        audit = pd.DataFrame(columns=["fold", "origin", "target_time", "source_start",
                                      "prep_minutes", "source_status", "eligible_alert_order_for_source"])
    scenarios, monthly = [], []
    month = (index-pd.Timedelta(minutes=15)).to_period("M")
    tau = p.set_index("target_time").loc[index, "tau"].to_numpy(float)
    peak_sources = {pd.Timestamp(_source_start(t)) for t in p.loc[p.y.gt(p.tau), "target_time"]}
    for scenario in ("D", "C"):
        action_rows, moved = _simulate_one(scenario, fraction, prep, sources, history,
                                            index, windows, original, weights, energy)
        actions = pd.DataFrame(action_rows)
        if actions.empty:
            actions = pd.DataFrame(columns=["fold", "origin", "target_time", "source_start",
                                            "status", "scenario", "fraction", "prep_minutes",
                                            "destination_start", "delta_per_interval"])
        actions["alarm_source"] = p.alarm_source.iloc[0]
        actions["horizon"] = horizon
        actions["scenario_label"] = SCENARIO_LABEL
        scenarios.append(actions)
        if scenario == "D" and not np.isclose((moved-original).sum(), 0, atol=1e-9):
            raise AssertionError("D shift failed to conserve the load proxy")
        applied = actions.loc[actions.status.eq("applied")]
        applied_peak = set(pd.DatetimeIndex(applied.source_start)) & peak_sources
        ratio, ci_low, ci_high = _bootstrap_ratio(audit, actions)
        for period in sorted(month.unique()):
            m = month == period
            monthly.append({"analysis": "A5", "horizon": horizon,
                            "alarm_source": p.alarm_source.iloc[0], "prep_minutes": prep,
                            "scenario": scenario, "fraction": fraction,
                            "month": str(period), "monthly_max_before": float(original[m].max()),
                            "monthly_max_after": float(moved[m].max()),
                            "monthly_max_change": float(moved[m].max()-original[m].max()),
                            "weighted_load_change": float(np.sum((moved[m]-original[m])*weights[m])),
                            "new_peak_positions": int(np.sum((moved[m] > tau[m]) & (original[m] <= tau[m]))),
                            "evaluated_intervals": int(m.sum()),
                            "full_calendar_intervals": calendar.monthrange(period.year, period.month)[1]*96,
                            "observed_fraction_of_full_month": float(m.sum()/(calendar.monthrange(period.year, period.month)[1]*96)),
                            "weight_basis": weight_basis, "scenario_label": SCENARIO_LABEL})
        coverage = _episode_coverage(p, audit)
        summary = {"analysis": "A5", "horizon": horizon,
                   "alarm_source": p.alarm_source.iloc[0], "prep_minutes": prep,
                   "scenario": scenario, "fraction": fraction,
                   "alert_positions": len(audit),
                   "time_eligible_alert_positions": int(audit.source_status.eq("source_eligible").sum()),
                   "time_eligible_alert_fraction": float(audit.source_status.eq("source_eligible").mean()) if len(audit) else np.nan,
                   "eligible_unique_source_windows": len({(s["fold"], s["source_start"]) for s in sources}),
                   "applied_unique_source_windows": len(applied),
                   "applied_per_alert_position": ratio,
                   "applied_per_alert_ci_low": ci_low, "applied_per_alert_ci_high": ci_high,
                   "applied_per_eligible_unique_source": len(applied)/len({(s["fold"], s["source_start"]) for s in sources}) if sources else np.nan,
                   "actual_peak_unique_source_hours": len(peak_sources),
                   "applied_actual_peak_source_hours": len(applied_peak),
                   "applied_actual_peak_source_fraction": len(applied_peak)/len(peak_sources) if peak_sources else np.nan,
                   **coverage, "load_proxy_total_change": float((moved-original).sum()),
                   "new_peak_positions": int(np.sum((moved > tau) & (original <= tau))),
                   "weight_basis": weight_basis, "scenario_label": SCENARIO_LABEL,
                   "scope": "partial_month_development_oof",
                   "oracle_scope": "perfect_information_reference_not_global_optimization_bound"}
        yield audit.assign(scenario=scenario), actions, pd.DataFrame(monthly[-len(month.unique()):]), summary


def run_a5(history: pd.DataFrame, oof: pd.DataFrame, manifest: dict,
           tariff_2026: dict, outdir: str | Path | None = None) -> dict:
    """h16 and h4 original alarm versus perfect-information upper reference."""
    history = _safe_history(history)
    records, audits, actions, monthly, matched = [], [], [], [], []
    operational = {}
    for horizon in (16, 4):
        metadata = _fold_meta(manifest, horizon)
        p = _prepare_operational(oof, manifest, horizon, history)
        operational[horizon] = p
        energy = _energy_for_horizon(history, metadata)
        for alarm_source in ("original_operational", "perfect_information_reference"):
            candidate = p.copy()
            candidate["alarm_source"] = alarm_source
            if alarm_source == "perfect_information_reference":
                candidate["alert"] = candidate.y.gt(candidate.tau)
            for prep in (30, 60):
                for audit, action, month, summary in _simulate_operational(
                        history, candidate, metadata, energy, tariff_2026, prep=prep):
                    audit = audit.assign(horizon=horizon, alarm_source=alarm_source,
                                         scenario_label=SCENARIO_LABEL)
                    audits.append(audit)
                    actions.append(action)
                    monthly.append(month)
                    records.append(summary)
    # A direct h4/h16 positional reference on common targets is informative;
    # simulation windows retain each original full OOF cohort above.
    common = set(operational[16].target_time) & set(operational[4].target_time)
    audit_table = pd.concat(audits, ignore_index=True)
    action_table = pd.concat(actions, ignore_index=True)
    for horizon, p in operational.items():
        part = p.loc[p.target_time.isin(common)]
        peak_source_hours = {_source_start(t) for t in part.loc[part.y.gt(part.tau), "target_time"]}
        for source in ("original_operational", "perfect_information_reference"):
            for prep in (30, 60):
                for scenario in ("D", "C"):
                    alarm = audit_table.loc[audit_table.horizon.eq(horizon) &
                                            audit_table.alarm_source.eq(source) &
                                            audit_table.prep_minutes.eq(prep) &
                                            audit_table.scenario.eq(scenario) &
                                            audit_table.target_time.isin(common)]
                    action = action_table.loc[action_table.horizon.eq(horizon) &
                                              action_table.alarm_source.eq(source) &
                                              action_table.prep_minutes.eq(prep) &
                                              action_table.scenario.eq(scenario) &
                                              action_table.target_time.isin(common)]
                    applied = action.loc[action.status.eq("applied")]
                    matched.append({"analysis": "A5", "horizon": horizon,
                                    "alarm_source": source, "prep_minutes": prep, "scenario": scenario,
                                    "cohort_scope": "same_target_time_h4_h16_intersection; full_oof_simulation_action_subset",
                                    "n_target_rows": len(part),
                                    "n_actual_peak_positions": int(part.y.gt(part.tau).sum()),
                                    "n_actual_peak_unique_source_hours": len(peak_source_hours),
                                    "n_alert_positions": len(alarm),
                                    "n_time_eligible_alert_positions": int(alarm.source_status.eq("source_eligible").sum()),
                                    "n_applied_unique_source_hours": len(applied),
                                    "n_applied_actual_peak_source_hours": len(
                                        set(pd.DatetimeIndex(applied.source_start)) & peak_source_hours),
                                    "applied_per_alert_position": len(applied)/len(alarm) if len(alarm) else np.nan,
                                    "tau_min": float(part.tau.min()), "tau_max": float(part.tau.max())})
    result = {"summary": pd.DataFrame(records), "alert_audit": audit_table,
              "actions": action_table,
              "monthly": pd.concat(monthly, ignore_index=True),
              "matched_h4_h16": pd.DataFrame(matched)}
    if outdir is not None:
        target = Path(outdir)
        for key, frame in result.items():
            write_table(frame, target / f"A5_{key}.csv")
    return result
