"""Pure, development-only Phase F forecast metrics and paired uncertainty.

All score calculations first select one pre-locked arm. AUC is the equally
weighted mean of the 13 horizon metrics, while pooled normalization uses the
number of evaluation rows in each fold times that fold's *fit* denominator.
Normalized metrics are fractions (multiply by 100 to display percentages).
NMBE is positive when predictions understate actual demand.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


HORIZONS = tuple(range(4, 17))
KEY = ["horizon", "fold", "origin", "target_time"]
REQUIRED = {
    "model", *KEY, "y", "pred", "tau", "d2", "role", "arm",
    "fit_mean", "mase_scale",
}
BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
QUANTILES = {"q10": .10, "q50": .50, "q90": .90, "q95": .95}


def _score_arm(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Select before inspecting any numeric values or deriving denominators."""
    if arm not in {"EXPLORE", "CONFIRM"}:
        raise ValueError("arm must be EXPLORE or CONFIRM")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    data = frame.loc[frame.role.eq("score") & frame.arm.eq(arm)].copy()
    if data.empty:
        raise ValueError(f"No score predictions in {arm}")
    if data.model.isna().any() or data.model.astype(str).str.strip().eq("").any():
        raise ValueError("Every prediction needs a nonempty model ID")
    for col in ("origin", "target_time"):
        data[col] = pd.to_datetime(data[col], errors="raise")
        if data[col].isna().any() or data[col].ge(BOUNDARY).any():
            raise ValueError(f"{col} reaches the sealed holdout boundary")
    for col in ("horizon", "fold", "y", "pred", "tau", "fit_mean", "mase_scale"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must be finite on every selected row")
    for col in ("y", "pred", "tau", "fit_mean", "mase_scale"):
        data[col] = data[col].astype(float)
    if not data.horizon.isin(HORIZONS).all() or not np.equal(data.horizon, data.horizon.astype(int)).all():
        raise ValueError("Expected integer horizons 4..16")
    if not np.equal(data.fold, data.fold.astype(int)).all() or data.fold.lt(0).any():
        raise ValueError("fold must be a nonnegative integer")
    data["horizon"] = data.horizon.astype(int)
    data["fold"] = data.fold.astype(int)
    if not data.d2.isin((True, False)).all() or data.d2.isna().any():
        raise ValueError("d2 must be a nonmissing boolean")
    data["d2"] = data.d2.astype(bool)
    if data.fit_mean.lt(0).any() or data.mase_scale.lt(0).any():
        raise ValueError("Fit normalizers cannot be negative")
    if not data.target_time.gt(data.origin).all() or not (
        data.target_time.eq(data.origin + pd.to_timedelta(data.horizon * 15, unit="m"))
    ).all():
        raise ValueError("Target time must be origin + horizon * 15 minutes")
    if data.duplicated(["model", *KEY]).any():
        raise ValueError("Duplicate model/forecast key")
    # A partial configuration cannot look good by silently omitting hard horizons.
    for model, part in data.groupby("model", sort=False):
        if set(part.fold) != {0, 1, 2}:
            raise ValueError(f"Missing expected folds for model={model}")
    for (model, fold), part in data.groupby(["model", "fold"], sort=False):
        if set(part.horizon) != set(HORIZONS):
            raise ValueError(f"Missing horizons for model={model}, fold={fold}")
    for col in ("tau", "fit_mean", "mase_scale"):
        if data.groupby(["model", "horizon", "fold"], sort=False)[col].nunique().gt(1).any():
            raise ValueError(f"{col} changes inside one fitted model/horizon/fold")
    # Even a stand-alone metrics table must use one fixed, comparable cohort.
    reference = None
    for model, part in data.groupby("model", sort=True):
        aligned = part.sort_values(KEY).reset_index(drop=True)
        if reference is None:
            reference = aligned
        else:
            if not aligned[KEY].equals(reference[KEY]):
                raise ValueError(f"Model {model} has a different forecast-key cohort")
            for col in ("y", "tau", "d2"):
                if not aligned[col].equals(reference[col]):
                    raise ValueError(f"Model {model} has inconsistent paired {col}")
    for col in (*QUANTILES, "p_peak", "train_seconds", "inference_seconds"):
        if col not in data:
            data[col] = np.nan
        else:
            data[col] = pd.to_numeric(data[col], errors="raise")
            nonmissing = data[col].dropna().to_numpy(dtype=float)
            if not np.isfinite(nonmissing).all():
                raise ValueError(f"{col} contains nonfinite values")
    if data.p_peak.dropna().lt(0).any() or data.p_peak.dropna().gt(1).any():
        raise ValueError("p_peak must lie in [0, 1]")
    for col in ("train_seconds", "inference_seconds"):
        if data[col].dropna().lt(0).any():
            raise ValueError(f"{col} cannot be negative")
    return data


def _cost(part: pd.DataFrame, column: str) -> float:
    """Sum fitted-cell costs once; return NA when timing is incomplete."""
    if part[column].isna().any():
        return np.nan
    run_id = column + '_run_id'
    if run_id in part and part[run_id].notna().all():
        costs = part.groupby(run_id, sort=False)[column]
        if costs.nunique().gt(1).any():
            raise ValueError('Shared computation has inconsistent cost')
        return float(costs.first().sum())
    cell = part.groupby(["model", "horizon", "fold"], sort=False)[column]
    if cell.nunique().gt(1).any():
        # Supports an explicitly row-measured inference cost as well.
        if column == "inference_seconds":
            return float(part[column].sum())
        raise ValueError("train_seconds varies within a fitted cell")
    return float(cell.first().sum())


def _metrics(part: pd.DataFrame) -> dict:
    n = len(part)
    y = part.y.to_numpy(float)
    pred = part.pred.to_numpy(float)
    tau = part.tau.to_numpy(float)
    err = pred - y
    abs_err = np.abs(err)
    actual_peak = y > tau
    predicted_peak = pred > tau
    fit_mean = float(part.fit_mean.mean())  # row-count-weighted fold fit mean
    scale = part.mase_scale.to_numpy(float)
    result = {
        "n": n,
        "peak_n": int(actual_peak.sum()),
        "predicted_peak_n": int(predicted_peak.sum()),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Peak_MAE": float(abs_err[actual_peak].mean()) if actual_peak.any() else np.nan,
        "PredPeak_MAE": float(abs_err[predicted_peak].mean()) if predicted_peak.any() else np.nan,
        "fit_mean": fit_mean,
        "nMAE": float(abs_err.mean() / fit_mean) if fit_mean > 0 else np.nan,
        "CVRMSE": float(np.sqrt(np.mean(err**2)) / fit_mean) if fit_mean > 0 else np.nan,
        "NMBE": float((-err).mean() / fit_mean) if fit_mean > 0 else np.nan,
        "MASE": float(np.mean(abs_err / scale)) if np.all(scale > 0) else np.nan,
        "mase_eligible_n": int(np.count_nonzero(scale > 0)),
        "MAPE_nonzero": float(np.mean(abs_err[y != 0] / np.abs(y[y != 0]))) if np.any(y != 0) else np.nan,
        "mape_nonzero_n": int(np.count_nonzero(y)),
        "train_seconds": _cost(part, "train_seconds"),
        "inference_seconds": _cost(part, "inference_seconds"),
    }
    for col, q in QUANTILES.items():
        values = part[col].to_numpy(float)
        valid = np.isfinite(values)
        result[f"{col}_n"] = int(valid.sum())
        result[f"pinball_{col}"] = (
            float(np.maximum(q * (y[valid] - values[valid]),
                             (q - 1) * (y[valid] - values[valid])).mean())
            if valid.any() else np.nan
        )
    for low, high in (("q10", "q90"), ("q10", "q95")):
        lower = part[low].to_numpy(float)
        upper = part[high].to_numpy(float)
        valid = np.isfinite(lower) & np.isfinite(upper)
        label = f"coverage_{low[1:]}_{high[1:]}"
        result[f"{label}_n"] = int(valid.sum())
        result[label] = float(np.mean((y[valid] >= lower[valid]) & (y[valid] <= upper[valid]))) if valid.any() else np.nan
        result[f"crossing_{low[1:]}_{high[1:]}_n"] = int(np.count_nonzero(lower[valid] > upper[valid]))
    p = part.p_peak.to_numpy(float)
    valid_p = np.isfinite(p)
    result["p_peak_n"] = int(valid_p.sum())
    result["Brier"] = float(np.mean((p[valid_p] - actual_peak[valid_p])**2)) if valid_p.any() else np.nan
    return result


def _group_metrics(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(by, observed=True, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(by, key)), **_metrics(part)})
    return pd.DataFrame(rows)


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    sample = frame.copy()
    sample["target_day"] = sample.target_time.dt.normalize()
    rows = []
    for key, part in sample.groupby(["dataset", "model", "fold", "horizon", "target_day"], sort=True, observed=True):
        times = part.target_time.drop_duplicates()
        if len(times) != len(part):
            raise ValueError("Daily max requires one forecast per target slot and horizon")
        ordered = part.sort_values("target_time")
        i_actual = ordered.y.idxmax()
        i_pred = ordered.pred.idxmax()
        actual_max = float(ordered.loc[i_actual, "y"])
        predicted_max = float(ordered.loc[i_pred, "pred"])
        actual_time = ordered.loc[i_actual, "target_time"]
        predicted_time = ordered.loc[i_pred, "target_time"]
        signed_minutes = float((predicted_time - actual_time).total_seconds() / 60)
        day = key[-1]
        full_day = (len(times) == 96 and times.min() == day and
                    times.max() == day + timedelta(hours=23, minutes=45) and
                    times.sort_values().diff().dropna().eq(timedelta(minutes=15)).all())
        rows.append({**dict(zip(["dataset", "model", "fold", "horizon", "target_day"], key)),
                     "observed_slots": len(times), "slot_coverage": len(times) / 96,
                     "full_day": bool(full_day), "actual_daily_max": actual_max,
                     "predicted_daily_max": predicted_max,
                     "daily_max_error": predicted_max - actual_max,
                     "daily_max_abs_error": abs(predicted_max - actual_max),
                     "actual_peak_time": actual_time, "predicted_peak_time": predicted_time,
                     "timing_error_minutes": signed_minutes,
                     "timing_abs_error_minutes": abs(signed_minutes)})
    return pd.DataFrame(rows)


def _daily_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, by in (("fold", ["dataset", "model", "fold", "horizon"]),
                      ("pooled", ["dataset", "model", "horizon"])):
        for key, all_days in daily.groupby(by, observed=True, sort=True):
            for subset in ("all_observed", "full_96"):
                part = all_days if subset == "all_observed" else all_days[all_days.full_day]
                rows.append({**dict(zip(by, key)), "scope": scope, "subset": subset,
                             "n_days": len(part), "n_full_days": int(all_days.full_day.sum()),
                             "mean_observed_slots": float(part.observed_slots.mean()) if len(part) else np.nan,
                             "daily_max_MAE": float(part.daily_max_abs_error.mean()) if len(part) else np.nan,
                             "daily_max_bias": float(part.daily_max_error.mean()) if len(part) else np.nan,
                             "timing_MAE_minutes": float(part.timing_abs_error_minutes.mean()) if len(part) else np.nan,
                             "timing_bias_minutes": float(part.timing_error_minutes.mean()) if len(part) else np.nan})
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, arm: str = "EXPLORE") -> dict[str, pd.DataFrame]:
    """Return D1/D2 forecast tables without file I/O or candidate selection.

    `fold` has one row per model/fold/horizon; `pooled` combines folds at each
    horizon; `auc` averages all 13 pooled horizons equally. `daily` reports the
    max and timing of observed target-day slots, including partial days;
    `daily_summary` provides both all-observed and full-96-slot summaries.
    """
    data = _score_arm(frame, arm)
    samples = [data.assign(dataset="D1"), data.loc[data.d2].assign(dataset="D2")]
    scored = pd.concat(samples, ignore_index=True)
    fold = _group_metrics(scored, ["dataset", "model", "fold", "horizon"])
    pooled = _group_metrics(scored, ["dataset", "model", "horizon"])
    models = sorted(data.model.unique())
    fold_index = pd.MultiIndex.from_product(
        [("D1", "D2"), models, (0, 1, 2), HORIZONS],
        names=["dataset", "model", "fold", "horizon"],
    )
    pooled_index = pd.MultiIndex.from_product(
        [("D1", "D2"), models, HORIZONS],
        names=["dataset", "model", "horizon"],
    )
    fold = fold.set_index(["dataset", "model", "fold", "horizon"]).reindex(fold_index).reset_index()
    pooled = pooled.set_index(["dataset", "model", "horizon"]).reindex(pooled_index).reset_index()
    for table in (fold, pooled):
        for col in ("n", "peak_n", "predicted_peak_n", "mase_eligible_n", "mape_nonzero_n",
                    "q10_n", "q50_n", "q90_n", "q95_n", "coverage_10_90_n",
                    "coverage_10_95_n", "crossing_10_90_n", "crossing_10_95_n", "p_peak_n"):
            table[col] = table[col].fillna(0).astype(int)
    auc_rows = []
    for (dataset, model), part in pooled.groupby(["dataset", "model"], observed=True, sort=True):
        if set(part.horizon) != set(HORIZONS):
            raise ValueError(f"Incomplete AUC for {dataset}/{model}")
        auc_rows.append({"dataset": dataset, "model": model, "n_horizons": len(part),
                         "n_available_horizons": int(part.MAE.notna().sum()),
                         "AUC_MAE": float(part.MAE.mean()) if part.MAE.notna().all() else np.nan,
                         "AUC_PeakMAE": float(part.Peak_MAE.mean()) if part.Peak_MAE.notna().all() else np.nan,
                         "AUC_PredPeakMAE": float(part.PredPeak_MAE.mean()) if part.PredPeak_MAE.notna().all() else np.nan,
                         "auc_definition": "mean of 13 pooled horizon metrics"})
    auc = pd.DataFrame(auc_rows)
    daily = _daily(scored)
    return {"fold": fold, "pooled": pooled, "auc": auc, "daily": daily,
            "daily_summary": _daily_summary(daily)}


def _paired_input(candidate: pd.DataFrame, baseline: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    cand = _score_arm(candidate, arm)
    base = _score_arm(baseline, arm)
    if cand.model.nunique() != 1 or base.model.nunique() != 1:
        raise ValueError("paired_ci takes one candidate model and one baseline model")
    cand_name, base_name = str(cand.model.iloc[0]), str(base.model.iloc[0])
    cand = cand.sort_values(KEY).reset_index(drop=True)
    base = base.sort_values(KEY).reset_index(drop=True)
    if not cand[KEY].equals(base[KEY]):
        raise ValueError("Paired comparison needs exactly identical forecast keys")
    for col in ("y", "tau", "d2"):
        if not cand[col].equals(base[col]):
            raise ValueError(f"Paired {col} conflicts on identical keys")
    return cand, base, cand_name, base_name


def _blocks(dates: pd.Series, block: str) -> np.ndarray:
    if block == "day":
        labels = dates.dt.normalize()
    elif block == "week":
        iso = dates.dt.isocalendar()
        labels = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    else:
        raise ValueError("block must be day or week")
    return pd.factorize(labels, sort=True)[0]


def _ci_row(estimate: float, draws: np.ndarray, *, n_blocks: int, **meta: object) -> dict:
    valid = draws[np.isfinite(draws)]
    reason = ""
    if not np.isfinite(estimate):
        reason = "nonfinite_point_estimate"
    elif n_blocks < 2 or len(valid) == 0 or (len(valid) and np.ptp(valid) == 0):
        reason = "degenerate_draws"
    elif len(valid) != len(draws):
        reason = "nonfinite_bootstrap_draws"
    return {**meta, "estimate": float(estimate),
            "ci_low": float(np.quantile(valid, .025)) if not reason else np.nan,
            "ci_high": float(np.quantile(valid, .975)) if not reason else np.nan,
            "ci_status": "available" if not reason else "unavailable",
            "ci_reason": reason, "valid_draw_count": len(valid), "n_blocks": n_blocks}


def paired_ci(candidate: pd.DataFrame, baseline: pd.DataFrame, arm: str = "EXPLORE",
              n: int = 1000, seed: int = 42, block: str = "day") -> pd.DataFrame:
    """Paired D1/D2 MAE-improvement and peak-degradation 95% bootstrap CIs.

    The same target-day (or ISO-week) blocks are drawn jointly across every
    horizon and fold. Improvement is baseline MAE minus candidate MAE; peak
    degradation is candidate actual-peak MAE minus baseline actual-peak MAE.
    Incomplete/degenerate draws yield NA intervals rather than optimistic CIs.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if block not in {"day", "week"}:
        raise ValueError("block must be day or week")
    cand, base, cand_name, base_name = _paired_input(candidate, baseline, arm)
    rows = []
    for dataset, mask in (("D1", np.ones(len(cand), dtype=bool)), ("D2", cand.d2.to_numpy(bool))):
        c = cand.loc[mask].reset_index(drop=True)
        b = base.loc[mask].reset_index(drop=True)
        if c.empty:
            block_ids = np.array([], dtype=int)
            n_blocks = 0
        else:
            block_ids = _blocks(c.target_time, block)
            n_blocks = int(block_ids.max()) + 1
        draws = (np.random.default_rng(seed).multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=n)
                 if n_blocks else np.empty((n, 0), dtype=int))
        mae_points, peak_points = [], []
        mae_draws, peak_draws = [], []
        for horizon in HORIZONS:
            hmask = c.horizon.eq(horizon).to_numpy()
            ce = np.abs(c.pred.to_numpy(float)[hmask] - c.y.to_numpy(float)[hmask])
            be = np.abs(b.pred.to_numpy(float)[hmask] - b.y.to_numpy(float)[hmask])
            peak = c.y.to_numpy(float)[hmask] > c.tau.to_numpy(float)[hmask]
            blocks_h = block_ids[hmask]
            count = np.bincount(blocks_h, minlength=n_blocks).astype(float)
            sum_ce = np.bincount(blocks_h, weights=ce, minlength=n_blocks)
            sum_be = np.bincount(blocks_h, weights=be, minlength=n_blocks)
            peak_count = np.bincount(blocks_h, weights=peak.astype(float), minlength=n_blocks)
            peak_ce = np.bincount(blocks_h, weights=ce * peak, minlength=n_blocks)
            peak_be = np.bincount(blocks_h, weights=be * peak, minlength=n_blocks)
            with np.errstate(divide="ignore", invalid="ignore"):
                mae_point = float((sum_be.sum() - sum_ce.sum()) / count.sum())
                peak_point = float((peak_ce.sum() - peak_be.sum()) / peak_count.sum())
                sampled_n = draws @ count
                sampled_peak_n = draws @ peak_count
                mae_samples = (draws @ sum_be - draws @ sum_ce) / sampled_n
                peak_samples = (draws @ peak_ce - draws @ peak_be) / sampled_peak_n
            mae_points.append(mae_point)
            peak_points.append(peak_point)
            mae_draws.append(mae_samples)
            peak_draws.append(peak_samples)
            for metric, estimate, samples, eligible in (
                ("MAE_improvement", mae_point, mae_samples, count > 0),
                ("Peak_MAE_degradation", peak_point, peak_samples, peak_count > 0),
            ):
                rows.append(_ci_row(estimate, samples, n_blocks=int(np.count_nonzero(eligible)),
                                    dataset=dataset, candidate=cand_name, baseline=base_name,
                                    horizon=horizon, metric=metric, paired_n=int(count.sum()),
                                    peak_n=int(peak_count.sum()), bootstrap_n=n, seed=seed, block=block))
        for metric, points, samples in (("AUC_MAE_improvement", mae_points, mae_draws),
                                        ("AUC_PeakMAE_degradation", peak_points, peak_draws)):
            values = np.asarray(points, dtype=float)
            sample_matrix = np.asarray(samples, dtype=float)
            estimate = float(values.mean())
            bootstrap_values = sample_matrix.mean(axis=0)
            rows.append(_ci_row(estimate, bootstrap_values, n_blocks=n_blocks,
                                dataset=dataset, candidate=cand_name, baseline=base_name,
                                horizon=np.nan, metric=metric, paired_n=len(c),
                                peak_n=int((c.y > c.tau).sum()), bootstrap_n=n,
                                seed=seed, block=block))
    return pd.DataFrame(rows)
