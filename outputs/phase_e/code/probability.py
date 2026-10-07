"""Frozen Phase E probability and upper-bound layer for B5 h16 development folds.

Only ``cal`` labels enter Platt and split-conformal fitting. Score labels are
used later for evaluation; D2 is always a score subset of D1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.metrics import average_precision_score


BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
HORIZON = pd.Timedelta("4h")
CLIP = 1e-6
NOMINAL = 0.95
Z95 = float(norm.ppf(NOMINAL))
REQUIRED = {"fold", "origin", "target_time", "y", "tau", "mu", "sigma",
            "is_peak", "is_d2_novel_profile", "p_climatology", "model", "horizon"}


def _prepare(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{partition} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{partition} is empty")
    data = frame.copy()
    if "partition" in data and not data.partition.eq(partition).all():
        raise ValueError(f"Expected only {partition} partition rows")
    if not data.model.eq("B5").all() or not data.horizon.eq(16).all():
        raise ValueError("Phase E probability requires only B5 h16")
    for column in ("origin", "target_time"):
        data[column] = pd.to_datetime(data[column], errors="raise")
        if data[column].isna().any() or data[column].ge(BOUNDARY).any():
            raise ValueError(f"{partition} {column} crosses the sealed boundary")
    if not data.target_time.eq(data.origin + HORIZON).all():
        raise ValueError("Expected a direct 240-minute target")
    if data.duplicated(["fold", "origin", "target_time"]).any():
        raise ValueError(f"Duplicate {partition} forecast key")
    for column in ("y", "tau", "mu", "sigma", "p_climatology"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {partition} {column}")
    if data.sigma.le(0).any():
        raise ValueError(f"Nonpositive {partition} sigma")
    if data.p_climatology.lt(0).any() or data.p_climatology.gt(1).any():
        raise ValueError("Fit climatology must be a probability")
    for column in ("is_peak", "is_d2_novel_profile"):
        if data[column].isna().any() or not data[column].isin((True, False)).all():
            raise ValueError(f"Invalid {partition} {column}")
    if not np.array_equal(data.is_peak.to_numpy(bool), data.y.gt(data.tau).to_numpy(bool)):
        raise ValueError(f"{partition} event labels conflict with fit tau")
    if data.groupby("fold").tau.nunique().gt(1).any():
        raise ValueError(f"{partition} tau varies within a fold")
    if data.groupby("fold").p_climatology.nunique().gt(1).any():
        raise ValueError(f"{partition} fit climatology varies within a fold")
    return data


def _fit_platt(p_raw: np.ndarray, labels: np.ndarray) -> tuple[float, float, dict]:
    if np.unique(labels).size != 2:
        raise ValueError("Platt calibration requires both classes in cal")
    x = logit(np.clip(p_raw, CLIP, 1.0 - CLIP))
    z = labels.astype(float)
    if np.ptp(x) < 1e-12 or np.min(x[z == 1]) > np.max(x[z == 0]):
        raise RuntimeError("Platt calibration has no identifiable finite monotone MLE")
    initial = np.array([float(logit(np.clip(z.mean(), CLIP, 1.0 - CLIP))), 1.0])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = beta[0] + beta[1] * x
        value = float(np.logaddexp(0.0, eta).sum() - np.dot(z, eta))
        residual = expit(eta) - z
        gradient = np.array([residual.sum(), np.dot(residual, x)])
        return value, gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (0.0, None)],
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Platt calibration numerical failure: {result.message}")
    a, b = map(float, result.x)
    return a, b, {"optimizer": "L-BFGS-B", "optimizer_success": True,
                  "optimizer_message": str(result.message), "objective_nll": float(result.fun),
                  "a": a, "b": b, "b_nonnegative": b >= 0.0,
                  "clip_lower": CLIP, "clip_upper": 1.0 - CLIP}


def calibrate(cal: pd.DataFrame, score: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Fit each fold on chronological cal only and apply frozen fits to score."""
    calibration = _prepare(cal, "cal")
    scored = _prepare(score, "score")
    for name, frame in (("cal", calibration), ("score", scored)):
        if "p_raw" in frame:
            supplied = pd.to_numeric(frame.p_raw, errors="coerce").to_numpy(float)
            expected = norm.sf((frame.tau.to_numpy(float) - frame.mu.to_numpy(float)) /
                               frame.sigma.to_numpy(float))
            if not np.isfinite(supplied).all() or not np.allclose(supplied, expected, atol=1e-12, rtol=0):
                raise ValueError(f"{name} supplied p_raw conflicts with the Gaussian formula")
    if set(calibration.fold) != set(scored.fold):
        raise ValueError("Cal and score fold sets differ")
    fits, conformal = [], []
    scored["p_raw"] = norm.sf((scored.tau - scored.mu) / scored.sigma)
    scored["q95_raw"] = scored.mu + Z95 * scored.sigma
    scored["p_cal"] = np.nan
    scored["U95"] = np.nan
    scored["uncertainty_flag"] = False
    for fold, cal_fold in calibration.groupby("fold", sort=True):
        score_fold = scored.loc[scored.fold.eq(fold)]
        if cal_fold.target_time.max() >= score_fold.target_time.min():
            raise ValueError(f"Fold {fold} cal must end before score")
        if cal_fold.tau.iloc[0] != score_fold.tau.iloc[0] or \
                cal_fold.p_climatology.iloc[0] != score_fold.p_climatology.iloc[0]:
            raise ValueError(f"Fold {fold} score does not retain fit tau/climatology")
        p_cal_raw = norm.sf((cal_fold.tau.to_numpy(float) - cal_fold.mu.to_numpy(float)) /
                            cal_fold.sigma.to_numpy(float))
        a, b, fit = _fit_platt(p_cal_raw, cal_fold.is_peak.to_numpy(bool))
        fit.update({"fold": int(fold), "n_cal": int(len(cal_fold)),
                    "event_n_cal": int(cal_fold.is_peak.sum()),
                    "cal_target_start": str(cal_fold.target_time.min()),
                    "cal_target_end": str(cal_fold.target_time.max()),
                    "score_target_start": str(score_fold.target_time.min()),
                    "score_target_end": str(score_fold.target_time.max())})
        fits.append(fit)
        standardized = (cal_fold.y.to_numpy(float) - cal_fold.mu.to_numpy(float)) / \
                       cal_fold.sigma.to_numpy(float)
        if not np.isfinite(standardized).all():
            raise ValueError(f"Fold {fold} has nonfinite standardized cal scores")
        n = len(standardized)
        rank = min(n, math.ceil((n + 1) * NOMINAL))
        q_star = float(np.partition(standardized, rank - 1)[rank - 1])
        if not np.isfinite(q_star):
            raise ValueError(f"Fold {fold} conformal quantile is nonfinite")
        conformal.append({"fold": int(fold), "method": "standardized_one_sided_split_conformal",
                          "nominal": NOMINAL, "n_cal": n, "rank_1_based": rank,
                          "q_star": q_star,
                          "cal_target_start": str(cal_fold.target_time.min()),
                          "cal_target_end": str(cal_fold.target_time.max())})
        location = scored.fold.eq(fold)
        p = scored.loc[location, "p_raw"].to_numpy(float)
        scored.loc[location, "p_cal"] = expit(a + b * logit(np.clip(p, CLIP, 1.0 - CLIP)))
        scored.loc[location, "U95"] = scored.loc[location, "mu"] + q_star * scored.loc[location, "sigma"]
        scored.loc[location, "uncertainty_flag"] = scored.loc[location, "U95"] > scored.loc[location, "tau"]
    for column in ("p_raw", "p_cal", "q95_raw", "U95"):
        if not np.isfinite(scored[column].to_numpy(float)).all():
            raise ValueError(f"Nonfinite score {column}")
    if ((scored[["p_raw", "p_cal"]] < 0) | (scored[["p_raw", "p_cal"]] > 1)).any().any():
        raise ValueError("Probability outside [0,1]")
    fit_log = {"method": "two_parameter_monotone_platt", "fit_partition": "cal",
               "score_labels_used_in_fit": False, "fits": fits}
    conformal_log = {"method": "standardized_one_sided_split_conformal",
                     "fit_partition": "cal", "score_labels_used_in_fit": False,
                     "finite_sample_rule": "min(n, ceil((n + 1) * 0.95))", "fits": conformal}
    return scored, fit_log, conformal_log


def _evaluation_logistic(p: np.ndarray, labels: np.ndarray) -> dict:
    if np.unique(labels).size != 2:
        return {"intercept": np.nan, "slope": np.nan, "status": "single_class"}
    x = logit(np.clip(p, CLIP, 1.0 - CLIP))
    if float(np.ptp(x)) < 1e-12:
        return {"intercept": np.nan, "slope": np.nan, "status": "constant_predictor"}
    z = labels.astype(float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = beta[0] + beta[1] * x
        residual = expit(eta) - z
        return (float(np.logaddexp(0.0, eta).sum() - np.dot(z, eta)),
                np.array([residual.sum(), np.dot(residual, x)]))

    initial = np.array([float(logit(np.clip(z.mean(), CLIP, 1 - CLIP))), 1.0])
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all() or max(abs(result.x)) > 1e4:
        return {"intercept": np.nan, "slope": np.nan, "status": "nonidentifiable_or_failed"}
    return {"intercept": float(result.x[0]), "slope": float(result.x[1]), "status": "ok"}


def _draws(part: pd.DataFrame, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample target-calendar-day blocks within each fold, retaining all rows."""
    keys = pd.MultiIndex.from_arrays([part.fold, part.target_time.dt.normalize()])
    block, uniques = pd.factorize(keys, sort=True)
    draws = np.zeros((n_boot, len(uniques)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    folds = np.asarray(uniques.get_level_values(0))
    for fold in pd.unique(folds):
        indices = np.flatnonzero(folds == fold)
        n = len(indices)
        draws[:, indices] = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return draws, block


def _ci(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "ci_valid_draws": 0,
                "ci_status": "unavailable_sparse"}
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"ci_lower": float(lo), "ci_upper": float(hi),
            "ci_valid_draws": int(len(finite)), "ci_status": "ok"}


def _weighted_ap(labels: np.ndarray, probabilities: np.ndarray,
                 block: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """AP with whole-day multiplicity and sklearn's tied-score threshold rule."""
    order = np.argsort(-probabilities, kind="mergesort")
    p_sorted = probabilities[order]
    z_sorted = labels[order]
    block_sorted = block[order]
    starts = np.r_[0, np.flatnonzero(p_sorted[1:] != p_sorted[:-1]) + 1]
    result = np.full(len(draws), np.nan)
    for i, draw in enumerate(draws):
        weights = draw[block_sorted]
        positives = weights * z_sorted
        total_positive = positives.sum()
        total_negative = weights.sum() - total_positive
        if total_positive <= 0 or total_negative <= 0:
            continue
        group_weight = np.add.reduceat(weights, starts)
        group_positive = np.add.reduceat(positives, starts)
        cum_total = np.cumsum(group_weight)
        precision = np.divide(np.cumsum(group_positive), cum_total,
                              out=np.zeros(len(starts), dtype=float), where=cum_total > 0)
        result[i] = float(np.dot(precision, group_positive) / total_positive)
    return result


def _bootstrap_metrics(part: pd.DataFrame, draws: np.ndarray, block: np.ndarray) -> dict:
    n_blocks = draws.shape[1]
    n = len(part)
    y = part.is_peak.to_numpy(float)
    raw = part.p_raw.to_numpy(float)
    cal = part.p_cal.to_numpy(float)
    clim = part.p_climatology.to_numpy(float)
    residuals = [(raw-y)**2, (cal-y)**2, (clim-y)**2,
                 (part.y.to_numpy(float) <= part.q95_raw.to_numpy(float)).astype(float),
                 (part.y.to_numpy(float) <= part.U95.to_numpy(float)).astype(float),
                 (part.q95_raw - part.mu).to_numpy(float),
                 (part.U95 - part.mu).to_numpy(float)]
    sums = np.column_stack([np.bincount(block, weights=v, minlength=n_blocks) for v in residuals])
    counts = np.bincount(block, minlength=n_blocks)
    aggregate = draws @ sums
    denominator = draws @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        means = aggregate / denominator[:, None]
        bss = 1.0 - means[:, 1] / means[:, 2]
    return {"BS_raw": means[:, 0], "BS_cal": means[:, 1],
            "BS_climatology": means[:, 2], "BS_raw_minus_cal": means[:, 0] - means[:, 1],
            "BS_cal_minus_climatology": means[:, 1] - means[:, 2],
            "BSS": bss, "coverage_raw": means[:, 3], "coverage_u95": means[:, 4],
            "mean_width_raw": means[:, 5], "mean_width_u95": means[:, 6],
            "PR_AUC_raw": _weighted_ap(y, raw, block, draws),
            "PR_AUC_cal": _weighted_ap(y, cal, block, draws)}


def _risk_row(part: pd.DataFrame, subset: str, fold: str | int,
              bootstrap: dict) -> dict:
    z = part.is_peak.to_numpy(float)
    raw = part.p_raw.to_numpy(float)
    cal = part.p_cal.to_numpy(float)
    climatology = part.p_climatology.to_numpy(float)
    bs_raw = float(np.mean((raw-z)**2))
    bs_cal = float(np.mean((cal-z)**2))
    bs_clim = float(np.mean((climatology-z)**2))
    row = {"subset": subset, "fold": fold, "n": len(part), "event_n": int(z.sum()),
           "prevalence": float(z.mean()), "BS_raw": bs_raw, "BS_cal": bs_cal,
           "BS_climatology": bs_clim, "BSS": float(1-bs_cal/bs_clim) if bs_clim > 0 else np.nan,
           "BS_raw_minus_cal": bs_raw-bs_cal,
           "BS_cal_minus_climatology": bs_cal-bs_clim,
           "PR_AUC_raw": float(average_precision_score(z, raw)) if len(np.unique(z)) == 2 else np.nan,
           "PR_AUC_cal": float(average_precision_score(z, cal)) if len(np.unique(z)) == 2 else np.nan}
    for metric in ("BS_raw_minus_cal", "BS_cal_minus_climatology", "BSS", "PR_AUC_raw", "PR_AUC_cal"):
        ci = _ci(bootstrap[metric])
        row.update({f"{metric}_{k}": v for k, v in ci.items()})
    return row


def _reliability(part: pd.DataFrame, subset: str, fold: str | int,
                 probability: str, draws: np.ndarray, block: np.ndarray) -> pd.DataFrame:
    p = part[probability].to_numpy(float)
    z = part.is_peak.to_numpy(int)
    indices = np.minimum((p * 10).astype(int), 9)
    n_blocks = draws.shape[1]
    counts = np.zeros((n_blocks, 10), dtype=float)
    events = np.zeros_like(counts)
    np.add.at(counts, (block, indices), 1.0)
    np.add.at(events, (block, indices), z)
    boot_count, boot_events = draws @ counts, draws @ events
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = boot_events / boot_count
    dates = part.target_time.dt.normalize()
    rows = []
    for j in range(10):
        selected = indices == j
        ci = _ci(rates[:, j])
        rows.append({"subset": subset, "fold": fold, "bin": j,
                     "bin_lower": j / 10, "bin_upper": (j+1) / 10,
                     "bin_upper_inclusive": j == 9, "n": int(selected.sum()),
                     "distinct_target_dates": int(dates[selected].nunique()),
                     "mean_predicted_probability": float(p[selected].mean()) if selected.any() else np.nan,
                     "event_n": int(z[selected].sum()),
                     "observed_event_rate": float(z[selected].mean()) if selected.any() else np.nan,
                     "observed_rate_ci_lower": ci["ci_lower"],
                     "observed_rate_ci_upper": ci["ci_upper"],
                     "observed_rate_ci_valid_draws": ci["ci_valid_draws"],
                     "observed_rate_ci_status": ci["ci_status"]})
    return pd.DataFrame(rows)


def _uncertainty_rows(part: pd.DataFrame, subset: str, fold: str | int,
                      draws: np.ndarray, block: np.ndarray) -> list[dict]:
    rows = []
    for population, mask in (("all", np.ones(len(part), dtype=bool)),
                             ("peak", part.is_peak.to_numpy(bool)),
                             ("nonpeak", ~part.is_peak.to_numpy(bool))):
        if not mask.any():
            continue
        target = part.y.to_numpy(float)[mask]
        mu = part.mu.to_numpy(float)[mask]
        subblock = block[mask]
        count = np.bincount(subblock, minlength=draws.shape[1])
        denom = draws @ count
        for method, bound in (("raw", part.q95_raw.to_numpy(float)[mask]),
                              ("conformal", part.U95.to_numpy(float)[mask])):
            covered = (target <= bound).astype(float)
            widths = bound - mu
            cover_sum = draws @ np.bincount(subblock, weights=covered, minlength=draws.shape[1])
            width_sum = draws @ np.bincount(subblock, weights=widths, minlength=draws.shape[1])
            with np.errstate(divide="ignore", invalid="ignore"):
                cover_draws, width_draws = cover_sum / denom, width_sum / denom
            cover_ci, width_ci = _ci(cover_draws), _ci(width_draws)
            rows.append({"subset": subset, "fold": fold, "population": population,
                         "method": method, "n": int(mask.sum()), "nominal_coverage": NOMINAL,
                         "coverage": float(covered.mean()), "coverage_ci_lower": cover_ci["ci_lower"],
                         "coverage_ci_upper": cover_ci["ci_upper"],
                         "coverage_ci_valid_draws": cover_ci["ci_valid_draws"],
                         "coverage_ci_status": cover_ci["ci_status"],
                         "mean_width": float(widths.mean()), "mean_width_ci_lower": width_ci["ci_lower"],
                         "mean_width_ci_upper": width_ci["ci_upper"],
                         "mean_width_ci_valid_draws": width_ci["ci_valid_draws"],
                         "mean_width_ci_status": width_ci["ci_status"],
                         "median_width": float(np.median(widths)),
                         "width_p10": float(np.quantile(widths, .10)),
                         "width_p25": float(np.quantile(widths, .25)),
                         "width_p75": float(np.quantile(widths, .75)),
                         "width_p90": float(np.quantile(widths, .90))})
    return rows


def evaluate_probability(score: pd.DataFrame, n_boot: int = 1000,
                         seed: int = 42) -> dict[str, pd.DataFrame]:
    """D1/D2 pooled and by-fold evaluation with fold-stratified day-block CIs."""
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    data = _prepare(score, "score")
    for column in ("p_raw", "p_cal", "q95_raw", "U95"):
        if column not in data or not np.isfinite(pd.to_numeric(data[column], errors="coerce")).all():
            raise ValueError(f"Missing or nonfinite score {column}")
    if ((data[["p_raw", "p_cal"]] < 0) | (data[["p_raw", "p_cal"]] > 1)).any().any():
        raise ValueError("Probability outside [0,1]")
    risk, by_fold, raw_bins, cal_bins, summary, uncertainty = [], [], [], [], [], []
    for subset, part in (("D1", data), ("D2", data.loc[data.is_d2_novel_profile].copy())):
        if part.empty:
            continue
        for fold, group in [("pooled", part), *list(part.groupby("fold", sort=True))]:
            draws, block = _draws(group, n_boot, seed)
            boot = _bootstrap_metrics(group, draws, block)
            row = _risk_row(group, subset, fold, boot)
            (risk if fold == "pooled" else by_fold).append(row)
            raw_bins.append(_reliability(group, subset, fold, "p_raw", draws, block))
            cal_bins.append(_reliability(group, subset, fold, "p_cal", draws, block))
            uncertainty.extend(_uncertainty_rows(group, subset, fold, draws, block))
            for probability in ("p_raw", "p_cal"):
                fit = _evaluation_logistic(group[probability].to_numpy(float),
                                           group.is_peak.to_numpy(bool))
                summary.append({"subset": subset, "fold": fold, "probability": probability,
                                "n": len(group), "event_n": int(group.is_peak.sum()), **fit})
    return {"risk_metrics": pd.DataFrame(risk),
            "risk_metrics_by_fold": pd.DataFrame(by_fold),
            "reliability_bins_raw": pd.concat(raw_bins, ignore_index=True) if raw_bins else pd.DataFrame(),
            "reliability_bins_calibrated": pd.concat(cal_bins, ignore_index=True) if cal_bins else pd.DataFrame(),
            "calibration_summary": pd.DataFrame(summary),
            "uncertainty_metrics": pd.DataFrame(uncertainty)}
