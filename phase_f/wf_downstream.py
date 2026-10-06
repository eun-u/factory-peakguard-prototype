"""Fixed Phase E risk and alert evaluation on Phase F weekly folds.

Only selected score-arm rows are converted to numeric values. Each week has
its own CAL-only probability shape, Platt fit and standardized upper bound;
the protected Phase E scoring and episode code supplies the fixed policy grid.
This module does not choose a model, threshold or alert policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from outputs.phase_e.code.alerts import THRESHOLDS, evaluate_alerts
from outputs.phase_e.code.probability import CLIP, NOMINAL, Z95, _fit_platt
from phase_f.downstream import _cal_split, _risk_and_coverage
from phase_f.harness import BOUNDARY, arm_for_targets
from phase_f.registry import config_hash
from src.holidays import CALENDAR_REVIEWED, calendar_flags


KEY = ["fold", "origin", "target_time"]
REQUIRED = {"horizon", *KEY, "y", "pred", "tau", "d2", "role", "arm", "cal_provenance"}
QUANTILES = (("q10", .10), ("q50", .50), ("q90", .90), ("q95", .95))
_METHODS = {"native_gaussian", "native_quantiles", "cal_residual_gaussian"}


def _contexts(prepared: Any) -> dict:
    contexts = getattr(prepared, "contexts", None)
    if contexts is None and isinstance(prepared, dict):
        contexts = prepared.get("contexts")
    if not isinstance(contexts, dict) or not contexts:
        raise ValueError("WeeklyPrepared contexts are required")
    return contexts


def _select(frame: pd.DataFrame, arm: str, name: str) -> pd.DataFrame:
    """Filter before reading any prediction, label, quantile or sigma values."""
    if not isinstance(frame, pd.DataFrame) or not {"role", "arm", "fold"} <= set(frame):
        raise ValueError(f"{name} requires role, arm and fold")
    score = frame.loc[frame.role.eq("score") & frame.arm.eq(arm)]
    if score.empty:
        raise ValueError(f"{name} has no {arm} score rows")
    folds = set(score.fold)
    selected = frame.loc[(frame.role.eq("score") & frame.arm.eq(arm)) |
                         (frame.role.eq("cal") & frame.fold.isin(folds))].copy()
    missing = REQUIRED - set(selected)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if "model" not in selected or not selected.model.eq(name).all():
        raise ValueError(f"{name} CAL/score model identity differs")
    selected["model"] = name
    for col in ("origin", "target_time"):
        selected[col] = pd.to_datetime(selected[col], errors="raise")
        if selected[col].isna().any() or selected[col].ge(BOUNDARY).any():
            raise ValueError(f"{name} {col} crossed development boundary")
    for col in ("horizon", "y", "pred", "tau"):
        selected[col] = pd.to_numeric(selected[col], errors="raise")
        if not np.isfinite(selected[col].to_numpy(float)).all():
            raise ValueError(f"{name} nonfinite {col}")
    if not selected.horizon.eq(16).all() or not selected.target_time.eq(
            selected.origin + pd.Timedelta(hours=4)).all():
        raise ValueError("Phase E evaluation requires direct h16 forecasts")
    if selected.d2.isna().any() or not selected.d2.isin((True, False)).all():
        raise ValueError(f"{name} invalid D2 flag")
    selected["d2"] = selected.d2.astype(bool)
    if selected.duplicated(["role", *KEY]).any():
        raise ValueError(f"{name} duplicate CAL or score key")
    score = selected.loc[selected.role.eq("score")]
    if not np.array_equal(arm_for_targets(score.target_time), np.repeat(arm, len(score))):
        raise ValueError(f"{name} score arm disagrees with target ISO week")
    cal = selected.loc[selected.role.eq("cal")]
    if cal.empty or not cal.arm.eq("CAL").all():
        raise ValueError(f"{name} missing CAL rows")
    allowed = {"fit_only", "chronological_oof", "in_sample", "excluded"}
    if cal.cal_provenance.isna().any() or not set(cal.cal_provenance) <= allowed:
        raise ValueError(f"{name} CAL predictions need recognized provenance")
    for col in ("sigma", *(q for q, _ in QUANTILES), "p_peak"):
        if col in selected:
            selected[col] = pd.to_numeric(selected[col], errors="raise")
            valid = selected[col].dropna().to_numpy(float)
            if not np.isfinite(valid).all():
                raise ValueError(f"{name} nonfinite {col}")
    return selected


def _paired(candidate: pd.DataFrame, baseline: pd.DataFrame, contexts: dict,
            arm: str, *, reduced_cal: bool = False) -> list:
    candidate_folds = set(candidate.loc[candidate.role.eq("score"), "fold"])
    baseline_folds = set(baseline.loc[baseline.role.eq("score"), "fold"])
    if candidate_folds != baseline_folds:
        raise ValueError("Candidate and B5 score week folds differ")
    expected_folds = set()
    for (horizon, fold), context in contexts.items():
        if int(horizon) != 16:
            continue
        origins = pd.DatetimeIndex(context["score"])
        targets = pd.DatetimeIndex(context["target_time"].loc[origins])
        if len(origins) and np.all(arm_for_targets(targets) == arm):
            expected_folds.add(fold)
    if candidate_folds != expected_folds or not candidate_folds:
        raise ValueError("Selected week folds differ from prepared score lock")
    for role in ("cal", "score"):
        a = candidate.loc[candidate.role.eq(role)].sort_values(KEY).reset_index(drop=True)
        b = baseline.loc[baseline.role.eq(role)].sort_values(KEY).reset_index(drop=True)
        if not a[KEY].equals(b[KEY]):
            raise ValueError(f"Candidate and B5 {role} forecast keys differ")
        for col in ("y", "tau", "d2"):
            if not a[col].equals(b[col]):
                raise ValueError(f"Candidate and B5 {role} {col} differs")
        for fold in candidate_folds:
            context = contexts[(16, fold)]
            part = a.loc[a.fold.eq(fold)]
            origins = pd.DatetimeIndex(context[role])
            if role == "score":
                targets = pd.DatetimeIndex(context["target_time"].loc[origins])
                origins = origins[arm_for_targets(targets) == arm]
            if role == "score" or not reduced_cal:
                invalid = set(part.origin) != set(origins) or len(part) != len(origins)
            else:
                invalid = not set(part.origin) <= set(origins) or part.empty
            if invalid:
                raise ValueError(f"{role} fold {fold} does not cover prepared origins")
            if not part.target_time.eq(context["target_time"].loc[part.origin].to_numpy()).all():
                raise ValueError(f"{role} fold {fold} target times differ from prepared context")
            truth = context["y"].loc[part.origin].to_numpy(float)
            if not np.array_equal(part.y.to_numpy(float), truth):
                raise ValueError(f"{role} fold {fold} truth differs from prepared context")
            if not part.tau.eq(float(context["tau"])).all():
                raise ValueError(f"{role} fold {fold} tau differs from fit context")
    return sorted(candidate_folds, key=str)


def _source_mode(frame: pd.DataFrame, *, baseline: bool) -> str:
    sigma = frame["sigma"] if "sigma" in frame else pd.Series(np.nan, index=frame.index)
    if baseline and sigma.notna().all() and sigma.gt(0).all():
        return "native_gaussian"
    quantile_columns = [q for q, _ in QUANTILES]
    present = [q for q in quantile_columns if q in frame and frame[q].notna().any()]
    if present:
        if len(present) != len(quantile_columns) or frame[quantile_columns].isna().any().any():
            raise ValueError("Partial native quantile distribution")
        return "native_quantiles"
    if baseline:
        raise ValueError("B5 requires its native Kalman sigma or complete native quantiles")
    if sigma.notna().any():
        raise ValueError("Candidate sigma needs explicit native distribution provenance")
    return "cal_residual_gaussian"


def _calendar_groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    targets = pd.DatetimeIndex(frame.target_time)
    offday = calendar_flags(targets).is_offday.to_numpy(dtype=int)
    return targets.hour.to_numpy(dtype=int), offday


def _point_distribution(source: pd.DataFrame, later: pd.DataFrame,
                        score: pd.DataFrame) -> tuple[tuple[np.ndarray, ...], dict]:
    residual = source.y.to_numpy(float) - source.pred.to_numpy(float)
    sh, sd = _calendar_groups(source)
    global_mu = float(np.mean(residual))
    global_sigma = max(float(np.std(residual, ddof=1)), 1e-6)
    groups = {}
    for hour in range(24):
        for day_type in (0, 1):
            exact = (sh == hour) & (sd == day_type)
            same_hour = sh == hour
            same_type = sd == day_type
            if exact.sum() >= 8:
                mask, source_name = exact, "hour_day_type"
            elif same_hour.sum() >= 8:
                mask, source_name = same_hour, "hour"
            elif same_type.sum() >= 8:
                mask, source_name = same_type, "day_type"
            else:
                mask, source_name = np.ones(len(source), dtype=bool), "global"
            values = residual[mask]
            mu = float(values.mean()) if len(values) else global_mu
            sigma = float(np.std(values, ddof=1)) if len(values) > 1 else global_sigma
            if not np.isfinite(sigma) or sigma < 1e-6:
                sigma = global_sigma
            groups[(hour, day_type)] = (mu, sigma, source_name, int(mask.sum()))

    def project(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hours, offdays = _calendar_groups(frame)
        cells = [groups[(int(h), int(d))] for h, d in zip(hours, offdays)]
        mu = frame.pred.to_numpy(float) + np.asarray([x[0] for x in cells])
        sigma = np.asarray([x[1] for x in cells])
        p = norm.sf((frame.tau.to_numpy(float) - mu) / sigma)
        return mu, sigma, p, mu + Z95 * sigma

    return (*project(later), *project(score)), {
        "source": "early_CAL_hour_x_day_type_standardized_residual_Gaussian",
        "fallback_counts": {name: sum(x[2] == name for x in groups.values())
                            for name in ("hour_day_type", "hour", "day_type", "global")},
        "min_group_n": 8, "calendar_reviewed": CALENDAR_REVIEWED,
    }


def _quantile_distribution(later: pd.DataFrame, score: pd.DataFrame,
                           *, baseline: bool) -> tuple[tuple[np.ndarray, ...], dict]:
    levels = np.array([level for _, level in QUANTILES])
    qcols = [q for q, _ in QUANTILES]

    def project(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = frame[qcols].to_numpy(float)
        if np.any(np.diff(q, axis=1) < 0):
            raise ValueError("Native quantiles cross")
        scale = (q[:, 2] - q[:, 0]) / (norm.ppf(.9) - norm.ppf(.1))
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Native quantile spread cannot standardize conformal scores")
        p = np.array([1.0 - np.interp(tau, row, levels, left=0.0, right=1.0)
                      for tau, row in zip(frame.tau.to_numpy(float), q)])
        return frame.pred.to_numpy(float), scale, p, q[:, 3]

    return (*project(later), *project(score)), {
        "source": "Kalman_native_quantile_interpolation" if baseline else "model_native_quantile_interpolation",
        "quantile_levels": levels.tolist(),
        "tail_rule": "linear interpolation inside supplied quantile range, CDF clipped to 0/1 outside",
        "standardization": "(q90-q10)/(z90-z10)",
    }


def _fit_week(cal: pd.DataFrame, score: pd.DataFrame, context: dict,
              *, baseline: bool) -> tuple[pd.DataFrame, dict]:
    source, later = _cal_split(cal)
    if later.target_time.max() >= score.origin.min():
        raise ValueError("CAL labels reach score origin")
    mode = _source_mode(pd.concat([cal, score]), baseline=baseline)
    if mode == "native_gaussian":
        def project(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mu = frame.pred.to_numpy(float)
            sigma = frame.sigma.to_numpy(float)
            if not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError("B5 native Kalman sigma must be positive")
            return mu, sigma, norm.sf((frame.tau.to_numpy(float)-mu)/sigma), mu+Z95*sigma
        projected = (*project(later), *project(score))
        source_audit = {"source": "B5_native_Kalman_Gaussian_sigma"}
    elif mode == "native_quantiles":
        projected, source_audit = _quantile_distribution(later, score, baseline=baseline)
    else:
        projected, source_audit = _point_distribution(source, later, score)
    mu_cal, sigma_cal, p_cal_raw, q_cal, mu_score, sigma_score, p_score_raw, q_score = projected
    events_cal = later.y.to_numpy(float) > later.tau.to_numpy(float)
    a, b, platt = _fit_platt(p_cal_raw, events_cal)
    standardized = (later.y.to_numpy(float) - mu_cal) / sigma_cal
    rank = min(len(standardized), math.ceil((len(standardized) + 1) * NOMINAL))
    q_star = float(np.partition(standardized, rank - 1)[rank - 1])
    fit = context["y"].loc[context["fit"]].to_numpy(float)
    tau = float(context["tau"])
    climatology = float(np.mean(fit > tau))
    scored = score.copy()
    scored["mu"] = mu_score
    scored["sigma"] = sigma_score
    scored["p_raw"] = p_score_raw
    scored["p_cal"] = expit(a + b*logit(np.clip(p_score_raw, CLIP, 1-CLIP)))
    scored["q95_raw"] = q_score
    scored["U95"] = mu_score + q_star*sigma_score
    scored["uncertainty_flag"] = scored.U95.gt(scored.tau)
    scored["is_peak"] = scored.y.gt(scored.tau)
    scored["is_d2_novel_profile"] = scored.d2
    scored["p_climatology"] = climatology
    for col in ("p_raw", "p_cal", "q95_raw", "U95"):
        if not np.isfinite(scored[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite downstream {col}")
    audit = {**source_audit, "model": str(score.model.iloc[0]), "fold": str(score.fold.iloc[0]),
             "residual_source_n": len(source), "calibrator_n": len(later),
             "calibrator_event_n": int(events_cal.sum()),
             "source_last_target": str(source.target_time.max()),
             "calibrator_first_origin": str(later.origin.min()),
             "calibrator_last_target": str(later.target_time.max()),
             "score_first_origin": str(score.origin.min()),
             "platt_a": a, "platt_b": b, "platt_optimizer": platt["optimizer"],
             "conformal_rank": rank, "conformal_q_star": q_star,
             "conformal_basis": "(CAL y - distribution location) / distribution scale",
             "fit_climatology": climatology, "score_labels_used_in_fit": False,
             "cal_source": "chronological horizon-embargoed CAL subdivision"}
    return scored, audit


def _pooled(frame: pd.DataFrame, **filters: Any) -> pd.Series | None:
    selected = frame
    for name, value in filters.items():
        selected = selected.loc[selected[name].eq(value)]
    if len(selected) > 1:
        raise ValueError(f"Nonunique pooled downstream metric: {filters}")
    return selected.iloc[0] if len(selected) else None


def _registry(tables: dict[str, pd.DataFrame], candidate: str) -> tuple[dict, pd.DataFrame, dict]:
    risk = tables["risk_metrics"]
    coverage = tables["uncertainty_metrics"]
    calibration = tables["calibration_summary"]
    episodes = tables["alert_episode_metrics"]
    misses = tables["miss_decomposition"]
    fields = {}
    comparisons = []
    missing = {}
    names = ("BS_cal", "BSS", "PR_AUC_cal")
    for dataset in ("D1", "D2"):
        for model in (candidate, "B5"):
            r = _pooled(risk, model=model, subset=dataset, fold="pooled")
            diagnostic = _pooled(calibration, model=model, subset=dataset,
                                 fold="pooled", probability="p_cal")
            if dataset == "D1":
                prefix = "E_" if model == candidate else "E_B5_"
                fields[prefix + "calibration_intercept"] = (
                    float(diagnostic.intercept) if diagnostic is not None else float("nan"))
                fields[prefix + "calibration_slope"] = (
                    float(diagnostic.slope) if diagnostic is not None else float("nan"))
            if diagnostic is None or diagnostic.status != "ok":
                missing[f"{model}/{dataset}/calibration"] = (
                    "no rows" if diagnostic is None else str(diagnostic.status))
            u_all = _pooled(coverage, model=model, subset=dataset, fold="pooled",
                            population="all", method="conformal")
            u_peak = _pooled(coverage, model=model, subset=dataset, fold="pooled",
                             population="peak", method="conformal")
            for metric in names:
                value = float(r[metric]) if r is not None else float("nan")
                if not np.isfinite(value):
                    missing[f"{model}/{dataset}/{metric}"] = "no rows or undefined event class/denominator"
                if dataset == "D1":
                    prefix = "E_" if model == candidate else "E_B5_"
                    fields[prefix + {"BS_cal": "brier", "BSS": "bss", "PR_AUC_cal": "prauc"}[metric]] = value
            for suffix, row in (("coverage", u_all), ("peak_coverage", u_peak)):
                value = float(row.coverage) if row is not None else float("nan")
                if not np.isfinite(value):
                    missing[f"{model}/{dataset}/{suffix}"] = "no represented rows or peak events"
                if dataset == "D1":
                    fields[("E_" if model == candidate else "E_B5_") + suffix] = value
            for policy, short in (("1/1", "11"), ("2/2", "22")):
                alert = _pooled(episodes, model=model, subset=dataset, fold="pooled",
                                threshold=.10, policy=policy)
                values = {
                    "recall": float(alert.episode_recall) if alert is not None else float("nan"),
                    "precision": float(alert.episode_precision) if alert is not None else float("nan"),
                    "F1": float(alert.episode_F1) if alert is not None else float("nan"),
                    "FP": int(alert.false_alert_episodes) if alert is not None else None,
                    "lead_median_minutes": float(alert.direct_lead_median_minutes) if alert is not None else float("nan"),
                }
                if dataset == "D1":
                    for suffix, value in values.items():
                        fields[f"{'E_' if model == candidate else 'E_B5_'}c10_{short}_{suffix}"] = value
                for suffix, value in values.items():
                    if value is None or (isinstance(value, float) and not np.isfinite(value)):
                        missing[f"{model}/{dataset}/c10_{short}_{suffix}"] = "no events, no alerts, or no matched lead"
                for miss_type in ("decision_miss", "forecast_miss"):
                    miss = _pooled(misses, model=model, subset=dataset, fold="pooled",
                                   threshold=.10, policy=policy, miss_class=miss_type)
                    if dataset == "D1":
                        fields[f"{'E_' if model == candidate else 'E_B5_'}c10_{short}_{miss_type}"] = (
                            int(miss.episodes) if miss is not None else None)
        for metric in ("BS_cal", "BSS", "PR_AUC_cal"):
            c = _pooled(risk, model=candidate, subset=dataset, fold="pooled")
            b = _pooled(risk, model="B5", subset=dataset, fold="pooled")
            cv = float(c[metric]) if c is not None else float("nan")
            bv = float(b[metric]) if b is not None else float("nan")
            comparisons.append({"dataset": dataset, "metric": metric, "candidate": cv,
                                "B5": bv, "candidate_minus_B5": cv-bv})
        for metric, table, filt, column in (
            ("peak_coverage", coverage, {"population": "peak", "method": "conformal"}, "coverage"),
            ("c10_22_recall", episodes, {"threshold": .10, "policy": "2/2"}, "episode_recall"),
            ("c10_22_F1", episodes, {"threshold": .10, "policy": "2/2"}, "episode_F1"),
        ):
            c = _pooled(table, model=candidate, subset=dataset, fold="pooled", **filt)
            b = _pooled(table, model="B5", subset=dataset, fold="pooled", **filt)
            cv = float(c[column]) if c is not None else float("nan")
            bv = float(b[column]) if b is not None else float("nan")
            comparisons.append({"dataset": dataset, "metric": metric, "candidate": cv,
                                "B5": bv, "candidate_minus_B5": cv-bv})
    return fields, pd.DataFrame(comparisons), missing


def paired_episode_ci(events: pd.DataFrame, candidate: str, baseline: str,
                      n: int = 1000, *, seed: int = 42,
                      represented_days: pd.DataFrame | None = None) -> dict:
    """Paired date bootstrap for fixed D1 h16 c=.10, 2/2 episode assignments.

    The event matcher has already run once for each model. TP/FN are assigned
    to their actual episode onset date; FP to their alert episode onset date.
    The same per-fold date multiplicities weight both models. A resampled date
    never creates a new episode or triggers a second matching pass. Supply
    ``represented_days`` with fold/target_time to include score days without
    an episode; otherwise the date universe is the union of event onset days.
    """
    if not isinstance(events, pd.DataFrame) or not isinstance(n, int) or n < 1:
        raise ValueError("Event table and positive bootstrap count are required")
    if not candidate or not baseline or candidate == baseline or not isinstance(seed, int):
        raise ValueError("Two distinct model IDs and integer seed are required")
    required = {"model", "subset", "fold", "threshold", "policy", "status",
                "actual_start", "actual_end", "alert_start", "onset_day"}
    if not required <= set(events):
        raise ValueError(f"Episode events missing columns: {sorted(required-set(events))}")
    selected = events.loc[events.model.isin((candidate, baseline)) &
                          events.subset.eq("D1") & events.threshold.eq(.10) &
                          events.policy.eq("2/2") & ~events.fold.astype(str).eq("pooled")].copy()
    if selected.empty and represented_days is None:
        return {"candidate": candidate, "baseline": baseline, "subset": "D1",
                "horizon": 16, "threshold": .10, "policy": "2/2", "n_boot": n,
                "seed": seed, "date_basis": "unavailable_no_event_or_score_dates",
                "event_matching": "fixed original one-to-one assignments; no rematching after date resampling",
                **{f"{metric}_{suffix}": None for metric in ("recall", "F1")
                   for suffix in ("candidate", "baseline", "delta", "delta_ci_low", "delta_ci_high")},
                "recall_valid_draws": 0, "F1_valid_draws": 0,
                "recall_ci_status": "unavailable_no_event_or_score_dates",
                "F1_ci_status": "unavailable_no_event_or_score_dates"}
    if not selected.status.isin(("TP", "FP", "FN")).all():
        raise ValueError("Unexpected fixed episode status")
    selected["fold"] = selected.fold.astype(str)
    for col in ("actual_start", "actual_end", "alert_start", "onset_day"):
        selected[col] = pd.to_datetime(selected[col], errors="coerce")
    actual = selected.status.isin(("TP", "FN"))
    if selected.onset_day.isna().any() or selected.loc[actual, ["actual_start", "actual_end"]].isna().any().any() or \
            selected.loc[~actual, "alert_start"].isna().any():
        raise ValueError("Incomplete episode onset or actual-event identity")
    expected_onset = pd.concat((selected.loc[actual, "actual_start"],
                                selected.loc[~actual, "alert_start"])).sort_index().dt.normalize()
    if not selected.onset_day.sort_index().equals(expected_onset):
        raise ValueError("Episode onset day disagrees with Phase E matching output")
    identities = {}
    for model in (candidate, baseline):
        part = selected.loc[selected.model.eq(model) & actual, ["fold", "actual_start", "actual_end"]]
        if part.duplicated().any():
            raise ValueError("An actual episode was counted twice")
        identities[model] = set(part.itertuples(index=False, name=None))
    if identities[candidate] != identities[baseline]:
        raise ValueError("Candidate and baseline actual episode cohorts differ")

    if represented_days is None:
        calendar = selected[["fold", "onset_day"]].drop_duplicates()
        date_basis = "union_of_fixed_episode_onset_days_only"
    else:
        if not isinstance(represented_days, pd.DataFrame) or not {"fold", "target_time"} <= set(represented_days):
            raise ValueError("represented_days needs fold and target_time")
        calendar = represented_days[["fold", "target_time"]].copy()
        calendar["fold"] = calendar.fold.astype(str)
        calendar["onset_day"] = pd.to_datetime(calendar.target_time, errors="raise").dt.normalize()
        if calendar.onset_day.isna().any():
            raise ValueError("represented_days has a missing target date")
        calendar = calendar[["fold", "onset_day"]].drop_duplicates()
        date_basis = "all_represented_score_target_calendar_days"
    if calendar.empty:
        raise ValueError("No date blocks for paired episode bootstrap")
    if not set(selected[["fold", "onset_day"]].itertuples(index=False, name=None)) <= set(
            calendar[["fold", "onset_day"]].itertuples(index=False, name=None)):
        raise ValueError("Episode onset lies outside represented score dates")
    folds = sorted(set(calendar.fold), key=str)
    actual_folds = set(selected.loc[actual, "fold"])
    if not actual_folds <= set(folds):
        raise ValueError("Actual episode fold is absent from date blocks")
    status_idx = {"TP": 0, "FP": 1, "FN": 2}
    arrays: dict[str, dict[str, np.ndarray]] = {candidate: {}, baseline: {}}
    date_counts = {}
    for fold in folds:
        dates = pd.DatetimeIndex(calendar.loc[calendar.fold.eq(fold), "onset_day"].sort_values())
        date_counts[fold] = len(dates)
        positions = {day: j for j, day in enumerate(dates)}
        for model in (candidate, baseline):
            matrix = np.zeros((len(dates), 3), dtype=int)
            rows = selected.loc[selected.fold.eq(fold) & selected.model.eq(model)]
            for row in rows.itertuples(index=False):
                matrix[positions[row.onset_day], status_idx[row.status]] += 1
            arrays[model][fold] = matrix
    point_counts = {model: sum((matrix.sum(axis=0) for matrix in arrays[model].values()),
                               np.zeros(3, dtype=int)) for model in (candidate, baseline)}
    rng = np.random.default_rng(seed)
    draws = {model: np.zeros((n, 3), dtype=int) for model in (candidate, baseline)}
    for fold in folds:
        count = date_counts[fold]
        weights = rng.multinomial(count, np.full(count, 1/count), size=n)
        for model in (candidate, baseline):
            draws[model] += weights @ arrays[model][fold]

    def scores(counts: np.ndarray, metric: str) -> np.ndarray:
        values = np.asarray(counts, float)
        tp, fp, fn = values[..., 0], values[..., 1], values[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = tp/(tp+fn)
            if metric == "recall":
                return recall
            precision = tp/(tp+fp)
            return 2*precision*recall/(precision+recall)

    result = {"candidate": candidate, "baseline": baseline, "subset": "D1",
              "horizon": 16, "threshold": .10, "policy": "2/2",
              "n_boot": n, "seed": seed, "date_basis": date_basis,
              "fold_date_counts": date_counts,
              "represented_date_blocks": int(sum(date_counts.values())),
              "bootstrap_unit": "paired stratified fold x target-calendar day",
              "event_matching": "fixed original one-to-one assignments; no rematching after date resampling",
              "TP_FP_FN_candidate": point_counts[candidate].tolist(),
              "TP_FP_FN_baseline": point_counts[baseline].tolist()}
    for metric in ("recall", "F1"):
        candidate_point = float(scores(point_counts[candidate], metric))
        baseline_point = float(scores(point_counts[baseline], metric))
        delta = scores(draws[candidate], metric) - scores(draws[baseline], metric)
        finite = delta[np.isfinite(delta)]
        valid = len(finite)
        point_valid = np.isfinite(candidate_point) and np.isfinite(baseline_point)
        status = ("unavailable_degenerate_point_metric" if not point_valid else
                  "unavailable_insufficient_date_blocks" if sum(date_counts.values()) < 2 else
                  "unavailable_sparse_bootstrap_draws" if valid < math.ceil(.95*n) else "ok")
        result.update({f"{metric}_candidate": candidate_point if np.isfinite(candidate_point) else None,
                       f"{metric}_baseline": baseline_point if np.isfinite(baseline_point) else None,
                       f"{metric}_delta": candidate_point-baseline_point if point_valid else None,
                       f"{metric}_delta_ci_low": float(np.quantile(finite, .025)) if status == "ok" else None,
                       f"{metric}_delta_ci_high": float(np.quantile(finite, .975)) if status == "ok" else None,
                       f"{metric}_valid_draws": valid,
                       f"{metric}_ci_status": status})
    return result


def _frame_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["role", *KEY]).reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False, float_format="%.17g").encode("utf-8")).hexdigest()


def _finite_json(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_artifacts(destination: Path, tables: dict[str, pd.DataFrame], manifest: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, frame in sorted(tables.items()):
        payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = destination / f"{name}.csv"
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Refusing to overwrite changed downstream artifact: {path}")
        else:
            temp = path.with_suffix(".csv.tmp")
            temp.write_bytes(payload)
            os.replace(temp, path)
        artifacts[path.name] = digest
    record = {**manifest, "artifacts": artifacts}
    record["manifest_sha256"] = config_hash(record)
    path = destination / "manifest.json"
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2,
                         default=str, allow_nan=False).encode("utf-8")
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != record:
            raise RuntimeError("Refusing to replace changed downstream manifest")
    else:
        temp = path.with_suffix(".json.tmp")
        temp.write_bytes(payload)
        os.replace(temp, path)
    return record


def evaluate_candidate(prepared: Any, candidateframe: pd.DataFrame,
                       baselineframe: pd.DataFrame, *, arm: str = "EXPLORE",
                       destination: str | Path | None = None,
                       n_boot: int = 1000, seed: int = 42) -> dict:
    """Return same-cohort Phase E metrics for one h16 candidate and B5.

    CONFIRM can only be called by the root's one-time locked evaluator. A
    caller-supplied destination is an immutable evidence directory; otherwise
    this function performs no filesystem writes.
    """
    if arm not in {"EXPLORE", "CONFIRM"} or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("Expected EXPLORE/CONFIRM arm and positive bootstrap count")
    contexts = _contexts(prepared)
    candidate_ids = set(candidateframe.loc[candidateframe.role.eq("score") &
                                     candidateframe.arm.eq(arm), "model"])
    if len(candidate_ids) != 1 or "B5" in candidate_ids:
        raise ValueError("Exactly one non-B5 candidate is required")
    candidate_id = str(next(iter(candidate_ids)))
    if not baselineframe.loc[baselineframe.role.eq("score") &
                             baselineframe.arm.eq(arm), "model"].eq("B5").all():
        raise ValueError("Baseline must be B5")
    candidate = _select(candidateframe, arm, candidate_id)
    baseline = _select(baselineframe, arm, "B5")
    bcal = baseline.loc[baseline.role.eq("cal")]
    if not set(bcal.cal_provenance) <= {"fit_only", "chronological_oof"}:
        raise ValueError("B5 CAL predictions are in-sample or excluded")
    candidate_cal = candidate.loc[candidate.role.eq("cal")]
    reduced_cal = not set(candidate_cal.cal_provenance) <= {"fit_only", "chronological_oof"}
    original_cal_n = int(len(candidate_cal))
    if reduced_cal:
        if not set(candidate_cal.cal_provenance) <= {"chronological_oof", "in_sample", "excluded"}:
            raise ValueError("Mixed fit_only and CAL-trained candidate provenance")
        honest = candidate_cal.loc[candidate_cal.cal_provenance.eq("chronological_oof"), KEY]
        if honest.empty:
            raise ValueError("No out-of-CAL-fit candidate predictions")
        keys = pd.MultiIndex.from_frame(honest)
        candidate = candidate.loc[candidate.role.eq("score") |
                                  pd.MultiIndex.from_frame(candidate[KEY]).isin(keys)].copy()
        baseline = baseline.loc[baseline.role.eq("score") |
                                pd.MultiIndex.from_frame(baseline[KEY]).isin(keys)].copy()
    folds = _paired(candidate, baseline, contexts, arm, reduced_cal=reduced_cal)
    for model, part in ((candidate_id, candidate), ("B5", baseline)):
        cal = part.loc[part.role.eq("cal")]
        if not set(cal.cal_provenance) <= {"fit_only", "chronological_oof"}:
            raise ValueError(f"{model} CAL calibration cohort contains in-sample predictions")
    outputs, fits = [], []
    for model, frame, is_baseline in (("B5", baseline, True), (candidate_id, candidate, False)):
        for fold in folds:
            cal = frame.loc[frame.role.eq("cal") & frame.fold.eq(fold)].copy()
            score = frame.loc[frame.role.eq("score") & frame.fold.eq(fold)].copy()
            if cal.empty or score.empty:
                raise ValueError(f"Missing CAL or selected score for {model}/{fold}")
            derived, audit = _fit_week(cal, score, contexts[(16, fold)], baseline=is_baseline)
            outputs.append(derived)
            fits.append(audit)
    scored = pd.concat(outputs, ignore_index=True)
    tables = _risk_and_coverage(scored, n_boot, seed)
    alert_tables = []
    for model, part in scored.groupby("model", sort=True):
        alert_tables.append({name: value.assign(model=model)
                             for name, value in evaluate_alerts(part, n_boot=n_boot, seed=seed).items()})
    for name in alert_tables[0]:
        tables[name] = pd.concat([group[name] for group in alert_tables], ignore_index=True)
    tables["calibration_fit"] = pd.DataFrame(fits)
    tables["score_probabilities"] = scored
    fields, comparison, missing = _registry(tables, candidate_id)
    tables["comparison_vs_B5"] = comparison
    paired_ci = paired_episode_ci(
        tables["alert_episode_events"], candidate_id, "B5", n=1000, seed=42,
        represented_days=scored.loc[scored.model.eq(candidate_id), ["fold", "target_time"]])
    for metric in ("recall", "F1"):
        prefix = f"E_c10_22_{metric}_delta"
        fields[f"{prefix}_vs_B5"] = paired_ci[f"{metric}_delta"]
        fields[f"{prefix}_ci_low_vs_B5"] = paired_ci[f"{metric}_delta_ci_low"]
        fields[f"{prefix}_ci_high_vs_B5"] = paired_ci[f"{metric}_delta_ci_high"]
        fields[f"{prefix}_ci_status_vs_B5"] = paired_ci[f"{metric}_ci_status"]
        if paired_ci[f"{metric}_ci_status"] != "ok":
            missing[f"paired_c10_22_{metric}_delta_ci"] = paired_ci[f"{metric}_ci_status"]
    tables["paired_episode_ci"] = pd.DataFrame([{**paired_ci,
        "fold_date_counts": json.dumps(paired_ci.get("fold_date_counts", {}), sort_keys=True)}])
    source_files = (Path(__file__), Path(__file__).resolve().parents[1]/"outputs/phase_e/code/probability.py",
                    Path(__file__).resolve().parents[1]/"outputs/phase_e/code/alerts.py",
                    Path(__file__).resolve().parents[1]/"phase_f/downstream.py",
                    Path(__file__).resolve().parents[1]/"src/analysis/errors.py")
    source_hashes = {path.relative_to(Path(__file__).resolve().parents[1]).as_posix():
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    manifest = {"version": 1, "arm": arm, "candidate": candidate_id,
                "baseline": "B5", "horizon": 16, "week_folds": [str(f) for f in folds],
                "candidate_input_sha256": _frame_digest(candidate),
                "baseline_input_sha256": _frame_digest(baseline),
                "selected_score_rows_per_model": len(scored)//2,
                "cal_cohort": "candidate chronological_oof matched to B5" if reduced_cal else "full weekly CAL",
                "cal_rows_before_provenance_filter": original_cal_n,
                "cal_rows_after_provenance_filter": int(candidate.role.eq("cal").sum()),
                "fixed_thresholds": list(THRESHOLDS), "fixed_policies": ["1/1", "2/2"],
                "cal_only_fits": True, "score_labels_used_in_fit": False,
                "B5_same_score_keys": True, "missing_metrics": missing,
                "paired_episode_ci": paired_ci,
                "registry_fields": {key: _finite_json(value) for key, value in fields.items()},
                "source_sha256": source_hashes,
                "reference_phase_e_values_are_not_used_as_same_cohort_baseline": True}
    digest = config_hash(manifest)
    if destination is not None:
        manifest = _write_artifacts(Path(destination), tables, manifest)
        digest = manifest["manifest_sha256"]
    return {"registry_fields": fields, "tables": tables, "manifest": manifest,
            "manifest_sha256": digest, "missing_metrics": missing,
            "paired_episode_ci": paired_ci}
