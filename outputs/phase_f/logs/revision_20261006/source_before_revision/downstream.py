"""F11 Phase E-style downstream diagnostics for locked Phase F h16 forecasts.

This module has no raw loader or file writes. It reuses the frozen, neutral
Phase E Platt optimizer, bootstrap risk helpers and fixed alert grid. Unlike
Phase E's B5-only driver, it accepts a locked candidate and Phase F B5 on the
same arm and forecast keys. Cal is chronologically subdivided: earlier rows
fit an empirical residual distribution; later, horizon-embargoed rows fit
Platt and one-sided conformal corrections. Score labels are evaluation only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from outputs.phase_e.code.alerts import THRESHOLDS, evaluate_alerts
from outputs.phase_e.code.probability import (
    CLIP, NOMINAL, Z95, _bootstrap_metrics, _draws, _evaluation_logistic,
    _fit_platt, _reliability, _risk_row, _uncertainty_rows,
)


BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
KEY = ["fold", "origin", "target_time"]
REQUIRED = {"model", "horizon", *KEY, "y", "pred", "tau", "d2", "role", "arm"}


def _locked_models(lock: Mapping) -> tuple[str, tuple[str, ...]]:
    if lock.get("confirm_complete") is not True or not lock.get("selection_lock_sha256") or not lock.get("confirm_once_sha256"):
        raise ValueError("F11 requires a primary selection lock and completed one-time CONFIRM state")
    candidates = tuple(lock.get("candidates", ()))
    primary = lock.get("primary_candidate")
    if not 1 <= len(candidates) <= 2 or len(set(candidates)) != len(candidates) or primary not in candidates or "B5" in candidates:
        raise ValueError("F11 requires one or two locked candidates including the primary, plus separate B5")
    return str(primary), candidates


def _part(frame: pd.DataFrame, role: str, arm: str | None) -> pd.DataFrame:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"{role} missing columns: {sorted(missing)}")
    mask = frame.role.eq(role) & frame.arm.eq("CAL" if role == "cal" else arm)
    data = frame.loc[mask].copy()
    if data.empty:
        raise ValueError(f"No {role} rows for {arm}")
    if not data.horizon.eq(16).all():
        raise ValueError("F11 is locked to h16")
    for name in ("origin", "target_time"):
        data[name] = pd.to_datetime(data[name], errors="raise")
        if data[name].isna().any() or data[name].ge(BOUNDARY).any():
            raise ValueError(f"{role} {name} crosses development boundary")
    if role == "score":
        week = data.target_time.dt.isocalendar().week.to_numpy(dtype=int)
        expected_arm = np.where(week % 2 == 0, "EXPLORE", "CONFIRM")
        if not np.array_equal(data.arm.to_numpy(str), expected_arm):
            raise ValueError("Score arm differs from locked target ISO-week rule")
    if not data.target_time.sub(data.origin).eq(pd.Timedelta(hours=4)).all():
        raise ValueError("F11 target must be 240 minutes after origin")
    for name in ("y", "pred", "tau"):
        data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite {role} {name}")
    if data.d2.isna().any() or not data.d2.isin((True, False)).all():
        raise ValueError("d2 must be a nonmissing boolean")
    data["d2"] = data.d2.astype(bool)
    if data.duplicated(["model", *KEY]).any():
        raise ValueError(f"Duplicate {role} model/forecast key")
    if role == "cal":
        if "cal_provenance" not in data or data.cal_provenance.isna().any():
            raise ValueError("Every cal prediction needs out-of-cal-fit provenance")
    return data


def _same_cohort(data: pd.DataFrame, models: tuple[str, ...], role: str) -> None:
    reference = data.loc[data.model.eq("B5")].sort_values(KEY).reset_index(drop=True)
    if reference.empty:
        raise ValueError(f"B5 {role} rows are required on the Phase F cohort")
    for model in models:
        part = data.loc[data.model.eq(model)].sort_values(KEY).reset_index(drop=True)
        if not part[KEY].equals(reference[KEY]):
            raise ValueError(f"{model} {role} forecast keys differ from B5")
        for name in ("y", "tau", "d2"):
            if not part[name].equals(reference[name]):
                raise ValueError(f"{model} {role} {name} differs from B5")


def _cal_split(cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = cal.sort_values("origin").reset_index(drop=True)
    if len(ordered) < 76:
        raise ValueError("Cal needs at least 76 rows for two >=30-row parts and h16 embargo")
    # Choose the most balanced valid split; rows in the horizon gap are unused.
    options = []
    for cut in range(30, len(ordered) - 29):
        later_start = int(ordered.origin.searchsorted(ordered.target_time.iloc[cut - 1], side="right"))
        if later_start <= len(ordered) - 30:
            options.append((min(cut, len(ordered) - later_start), cut, later_start))
    if not options:
        raise ValueError("No chronological cal subdivision with target-time embargo")
    _, cut, later_start = max(options, key=lambda item: (item[0], -abs(item[1] - len(ordered) / 2)))
    source, calibrate = ordered.iloc[:cut].copy(), ordered.iloc[later_start:].copy()
    if not source.target_time.max() < calibrate.origin.min():
        raise AssertionError("Internal cal target-time embargo failed")
    return source, calibrate


def _climatology(lock: Mapping, model: str, fold: int) -> float:
    lookup = lock.get("fit_peak_prevalence", {})
    try:
        value = float(lookup[model][str(fold)])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Fit-only peak climatology missing for {model}/fold{fold}") from exc
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Fit-only peak climatology must be in [0, 1]")
    return value


def _raw_probability(source: pd.DataFrame, later: pd.DataFrame, score: pd.DataFrame,
                     mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if mode == "residual":
        residual = source.y.to_numpy(float) - source.pred.to_numpy(float)
        residual = np.sort(residual)
        q95 = float(np.quantile(residual, NOMINAL))

        def project(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            threshold = frame.tau.to_numpy(float) - frame.pred.to_numpy(float)
            greater = len(residual) - np.searchsorted(residual, threshold, side="right")
            p = (greater + .5) / (len(residual) + 1)
            return p, frame.pred.to_numpy(float) + q95

        p_later, q_later = project(later)
        p_score, q_score = project(score)
        return p_later, q_later, p_score, q_score, "early_cal_empirical_residual_CDF"
    if mode == "gaussian":
        for frame in (source, later, score):
            if "sigma" not in frame or not np.isfinite(pd.to_numeric(frame.sigma, errors="coerce")).all() or frame.sigma.le(0).any():
                raise ValueError("Gaussian F11 requires positive model-native sigma on cal and score")
        def project(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            sigma = frame.sigma.to_numpy(float)
            mu = frame.pred.to_numpy(float)
            return norm.sf((frame.tau.to_numpy(float) - mu) / sigma), mu + Z95 * sigma
        p_later, q_later = project(later)
        p_score, q_score = project(score)
        return p_later, q_later, p_score, q_score, "model_native_gaussian_sigma"
    if mode == "native":
        for frame in (source, later, score):
            if not {"p_peak", "q95"} <= set(frame) or not np.isfinite(frame[["p_peak", "q95"]].to_numpy(float)).all():
                raise ValueError("Native F11 requires model-native p_peak and q95 on cal and score")
            if not frame.p_peak.between(0, 1).all():
                raise ValueError("Native p_peak is outside [0, 1]")
        return (later.p_peak.to_numpy(float), later.q95.to_numpy(float),
                score.p_peak.to_numpy(float), score.q95.to_numpy(float), "model_native_probability_and_quantile")
    raise ValueError(f"Unknown F11 probability source: {mode}")


def _fit_one(cal: pd.DataFrame, score: pd.DataFrame, model: str, fold: int,
             mode: str, climatology: float) -> tuple[pd.DataFrame, dict]:
    source, later = _cal_split(cal)
    labels = later.y.to_numpy(float) > later.tau.to_numpy(float)
    p_later, q_later, p_score, q_score, source_method = _raw_probability(source, later, score, mode)
    a, b, platt = _fit_platt(p_later, labels)
    if mode == "gaussian":
        residual = (later.y.to_numpy(float) - later.pred.to_numpy(float)) / later.sigma.to_numpy(float)
    else:
        residual = later.y.to_numpy(float) - q_later
    rank = min(len(residual), math.ceil((len(residual) + 1) * NOMINAL))
    q_star = float(np.partition(residual, rank - 1)[rank - 1])
    bound = (score.pred.to_numpy(float) + q_star * score.sigma.to_numpy(float)
             if mode == "gaussian" else q_score + q_star)
    calibrated = expit(a + b * logit(np.clip(p_score, CLIP, 1 - CLIP)))
    out = score.copy()
    out["mu"] = out.pred.to_numpy(float)
    out["p_raw"] = p_score
    out["p_cal"] = calibrated
    out["q95_raw"] = q_score
    out["U95"] = bound
    out["uncertainty_flag"] = out.U95.gt(out.tau)
    out["is_peak"] = out.y.gt(out.tau)
    out["is_d2_novel_profile"] = out.d2
    out["p_climatology"] = climatology
    audit = {"model": model, "fold": fold, "source": source_method,
             "residual_source_n": len(source), "calibrator_n": len(later),
             "source_last_target": str(source.target_time.max()),
             "calibrator_first_origin": str(later.origin.min()),
             "calibrator_last_target": str(later.target_time.max()),
             "score_first_origin": str(score.origin.min()),
             "platt_a": a, "platt_b": b, "platt_optimizer": platt["optimizer"],
             "conformal_rank": rank, "conformal_q_star": q_star,
             "conformal_basis": "standardized residual" if mode == "gaussian" else "raw upper-bound residual",
             "fit_climatology": climatology,
             "score_labels_used_in_fit": False, "fit_partition": "horizon-embargoed cal subdivision"}
    return out, audit


def _risk_and_coverage(scored: pd.DataFrame, n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
    risk, uncertainty, raw_bins, cal_bins, summaries = [], [], [], [], []
    for model, by_model in scored.groupby("model", sort=True):
        for subset, population in (("D1", by_model), ("D2", by_model.loc[by_model.d2])):
            if population.empty:
                continue
            for fold, part in [("pooled", population), *list(population.groupby("fold", sort=True))]:
                draws, block = _draws(part, n_boot, seed)
                boot = _bootstrap_metrics(part, draws, block)
                risk.append({"model": model, **_risk_row(part, subset, fold, boot)})
                uncertainty.extend({"model": model, **row} for row in _uncertainty_rows(part, subset, fold, draws, block))
                raw_bins.append(_reliability(part, subset, fold, "p_raw", draws, block).assign(model=model))
                cal_bins.append(_reliability(part, subset, fold, "p_cal", draws, block).assign(model=model))
                for probability in ("p_raw", "p_cal"):
                    fit = _evaluation_logistic(part[probability].to_numpy(float), part.is_peak.to_numpy(bool))
                    summaries.append({"model": model, "subset": subset, "fold": fold,
                                      "probability": probability, "n": len(part),
                                      "event_n": int(part.is_peak.sum()), **fit})
    return {"risk_metrics": pd.DataFrame(risk),
            "uncertainty_metrics": pd.DataFrame(uncertainty),
            "reliability_bins_raw": pd.concat(raw_bins, ignore_index=True) if raw_bins else pd.DataFrame(),
            "reliability_bins_calibrated": pd.concat(cal_bins, ignore_index=True) if cal_bins else pd.DataFrame(),
            "calibration_summary": pd.DataFrame(summaries)}


def _comparison(tables: dict[str, pd.DataFrame], candidates: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    risk = tables["risk_metrics"]
    uncertainty = tables["uncertainty_metrics"]
    episodes = tables["alert_episode_metrics"]
    for candidate in candidates:
        for subset in ("D1", "D2"):
            for metric in ("BS_cal", "BSS", "PR_AUC_cal"):
                for source in (candidate, "B5"):
                    selected = risk.loc[risk.model.eq(source) & risk.subset.eq(subset) & risk.fold.eq("pooled")]
                    if len(selected) != 1:
                        raise ValueError(f"Missing pooled {source}/{subset} risk comparison")
                a = float(risk.loc[risk.model.eq(candidate) & risk.subset.eq(subset) & risk.fold.eq("pooled"), metric].iloc[0])
                b = float(risk.loc[risk.model.eq("B5") & risk.subset.eq(subset) & risk.fold.eq("pooled"), metric].iloc[0])
                rows.append({"candidate": candidate, "baseline": "B5", "subset": subset,
                             "metric": metric, "threshold": np.nan, "policy": "", "candidate_value": a,
                             "baseline_value": b, "difference": a - b})
            for method in ("raw", "conformal"):
                a_row = uncertainty.loc[uncertainty.model.eq(candidate) & uncertainty.subset.eq(subset) &
                                        uncertainty.fold.eq("pooled") & uncertainty.population.eq("peak") &
                                        uncertainty.method.eq(method)]
                b_row = uncertainty.loc[uncertainty.model.eq("B5") & uncertainty.subset.eq(subset) &
                                        uncertainty.fold.eq("pooled") & uncertainty.population.eq("peak") &
                                        uncertainty.method.eq(method)]
                if len(a_row) != 1 or len(b_row) != 1:
                    continue
                a, b = float(a_row.coverage.iloc[0]), float(b_row.coverage.iloc[0])
                rows.append({"candidate": candidate, "baseline": "B5", "subset": subset,
                             "metric": f"peak_coverage_{method}", "threshold": np.nan, "policy": "",
                             "candidate_value": a, "baseline_value": b, "difference": a - b})
            for threshold in THRESHOLDS:
                for policy in ("1/1", "2/2"):
                    a_row = episodes.loc[episodes.model.eq(candidate) & episodes.subset.eq(subset) &
                                         episodes.fold.eq("pooled") & episodes.threshold.eq(threshold) &
                                         episodes.policy.eq(policy)]
                    b_row = episodes.loc[episodes.model.eq("B5") & episodes.subset.eq(subset) &
                                         episodes.fold.eq("pooled") & episodes.threshold.eq(threshold) &
                                         episodes.policy.eq(policy)]
                    if len(a_row) != 1 or len(b_row) != 1:
                        raise ValueError("Fixed alert grid comparison is incomplete")
                    a, b = float(a_row.episode_recall.iloc[0]), float(b_row.episode_recall.iloc[0])
                    rows.append({"candidate": candidate, "baseline": "B5", "subset": subset,
                                 "metric": "episode_recall", "threshold": threshold, "policy": policy,
                                 "candidate_value": a, "baseline_value": b, "difference": a - b})
    return pd.DataFrame(rows)


def evaluate_downstream(cal_frame: pd.DataFrame, score_frame: pd.DataFrame,
                        lock: Mapping, arm: str = "CONFIRM", n_boot: int = 1000,
                        seed: int = 42) -> dict[str, pd.DataFrame]:
    """Evaluate locked h16 candidates and B5 on the same Phase F score arm.

    The return includes probability/coverage/alert tables and paired cohort
    point differences. `arm='CONFIRM'` is allowed only after a completed one-time
    CONFIRM lock. This function performs no candidate selection or file I/O.
    """
    if arm not in {"CONFIRM", "EXPLORE"} or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("Expected arm CONFIRM/EXPLORE and positive bootstrap count")
    _, candidates = _locked_models(lock)
    models = ("B5", *candidates)
    cal = _part(cal_frame, "cal", None)
    score = _part(score_frame, "score", arm)
    if set(cal.model) != set(models) or set(score.model) != set(models):
        raise ValueError("F11 frames must contain exactly B5 and the one/two locked candidates")
    ensemble_models = set(lock.get("ensemble_models", ())) | {model for model in candidates if model.startswith("F7")}
    if not ensemble_models <= set(candidates):
        raise ValueError("Unknown ensemble model in F11 lock")
    admissible = None
    for model in sorted(ensemble_models):
        part = cal.loc[cal.model.eq(model)]
        if not set(part.cal_provenance) <= {"in_sample", "excluded", "chronological_oof"}:
            raise ValueError(f"{model} cal provenance is unknown")
        selected = pd.MultiIndex.from_frame(part.loc[part.cal_provenance.eq("chronological_oof"), KEY])
        admissible = set(selected) if admissible is None else admissible.intersection(selected)
    if admissible is not None:
        if not admissible:
            raise ValueError("No common out-of-cal-fit ensemble calibration cohort")
        cal = cal.loc[pd.MultiIndex.from_frame(cal[KEY]).isin(admissible)].copy()
    _same_cohort(cal, candidates, "cal")
    _same_cohort(score, candidates, "score")
    if set(cal.fold) != set(score.fold) or set(score.fold) != {0, 1, 2}:
        raise ValueError("F11 requires three matching development folds")
    for model, part in cal.groupby("model", sort=True):
        allowed = {"chronological_oof"} if model in ensemble_models else {"fit_only", "chronological_oof"}
        if not set(part.cal_provenance) <= allowed:
            raise ValueError(f"{model} cal predictions are in-sample or have unknown provenance")
    outputs, fits = [], []
    source_map = lock.get("probability_source", {})
    for model in models:
        for fold in (0, 1, 2):
            cal_part = cal.loc[cal.model.eq(model) & cal.fold.eq(fold)].copy()
            score_part = score.loc[score.model.eq(model) & score.fold.eq(fold)].copy()
            if cal_part.empty or score_part.empty or cal_part.target_time.max() >= score_part.origin.min():
                raise ValueError(f"{model}/fold{fold} cal must precede score with horizon embargo")
            mode = str(source_map.get(model, "residual"))
            if mode == "gaussian" and model != "B5":
                raise ValueError("Only B5 can use its native Kalman Gaussian sigma")
            derived, audit = _fit_one(cal_part, score_part, model, fold, mode,
                                      _climatology(lock, model, fold))
            outputs.append(derived)
            fits.append(audit)
    scored = pd.concat(outputs, ignore_index=True)
    tables = _risk_and_coverage(scored, n_boot, seed)
    alert_tables = []
    for model, by_model in scored.groupby("model", sort=True):
        evaluated = evaluate_alerts(by_model, n_boot=n_boot, seed=seed)
        alert_tables.append({name: frame.assign(model=model) for name, frame in evaluated.items()})
    for name in alert_tables[0]:
        tables[name] = pd.concat([item[name] for item in alert_tables], ignore_index=True)
    tables["comparison_vs_B5"] = _comparison(tables, candidates)
    tables["calibration_fit"] = pd.DataFrame(fits)
    tables["score_probabilities"] = scored
    return tables
