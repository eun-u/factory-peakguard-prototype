"""Selection-unused symmetric peak metrics with paired date-block intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import match_episodes, score_predictions

METRICS = ("peak_mae_union", "peak_bias", "overpredict_rate", "peak_mae",
           "position_fp", "episode_f1")


def metric_daily_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep original cross-midnight matching and anchor each event to its start.

    TP/FN use the actual episode start date; unmatched FP use the alert start
    date. Thus summing daily event counts reproduces the original fold-wise
    matching exactly. Resampled dates never create new episode adjacency.
    This is an event-start date cluster interval, not a new selection metric.
    """
    part = frame.sort_values(["fold", "target_time"]).copy()
    part = part.loc[np.isfinite(part[["y", "pred", "tau"]]).all(axis=1)]
    if part.empty:
        return pd.DataFrame()
    ordered = part.sort_values("target_time")
    boundary = ordered.fold.ne(ordered.fold.shift()) & ordered.fold.shift().notna()
    if (boundary & ordered.target_time.diff().le(pd.Timedelta(minutes=15))).any():
        raise ValueError("Fold scoring boundaries must be purged before episode evaluation")
    day = pd.to_datetime(part.target_time).dt.normalize()
    actual = part.y.gt(part.tau).to_numpy()
    predicted = part.pred.gt(part.tau).to_numpy()
    alert = part.alert.to_numpy(dtype=bool) if "alert" in part else predicted
    error = (part.pred - part.y).to_numpy(dtype=float)
    union = actual | predicted
    daily = pd.DataFrame({
        "date": day.to_numpy(), "union_error": np.where(union, abs(error), 0),
        "union_n": union.astype(int), "peak_error": np.where(actual, abs(error), 0),
        "bias_sum": np.where(actual, error, 0), "peak_n": actual.astype(int),
        "overpredict_n": (~actual & predicted).astype(int),
        "nonpeak_n": (~actual).astype(int), "position_fp": (~actual & alert).astype(int),
        "n": 1,
    }).groupby("date", sort=True).sum()
    for name in ("episode_tp", "episode_fp", "episode_fn"):
        daily[name] = 0
    for _, block in part.groupby("fold", sort=True):
        block = block.sort_values("target_time")
        alarms = block.alert.to_numpy(dtype=bool) if "alert" in block else block.pred.gt(block.tau)
        matched = match_episodes(block.y.gt(block.tau), alarms, block.target_time)
        actual_matched = {a for a, _ in matched["matches"]}
        alert_matched = {b for _, b in matched["matches"]}
        for number, (start, _) in enumerate(matched["actual_episodes"]):
            column = "episode_tp" if number in actual_matched else "episode_fn"
            daily.at[pd.Timestamp(block.target_time.iloc[start]).normalize(), column] += 1
        for number, (start, _) in enumerate(matched["alert_episodes"]):
            if number not in alert_matched:
                daily.at[pd.Timestamp(block.target_time.iloc[start]).normalize(), "episode_fp"] += 1
    return daily


def _ratios(totals):
    def divide(a, b):
        a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    return {
        "peak_mae_union": divide(totals["union_error"], totals["union_n"]),
        "peak_bias": divide(totals["bias_sum"], totals["peak_n"]),
        "overpredict_rate": divide(totals["overpredict_n"], totals["nonpeak_n"]),
        "peak_mae": divide(totals["peak_error"], totals["peak_n"]),
        "position_fp": totals["position_fp"],
        "episode_f1": divide(2 * totals["episode_tp"],
                             2 * totals["episode_tp"] + totals["episode_fp"] + totals["episode_fn"]),
    }


def symmetric_peak_table(predictions, cfg):
    """One row per horizon/candidate, all six intervals using the same dates."""
    rows = []
    cutoff = pd.Timestamp(cfg["split"]["test_start_origin"])
    if (pd.to_datetime(predictions.target_time) >= cutoff).any() or (pd.to_datetime(predictions.origin) >= cutoff).any():
        raise ValueError("Only development OOF predictions are allowed")
    n_boot, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    for (horizon, model), group in predictions.groupby(["horizon", "model"], sort=True):
        group = group.copy()
        reference = str(model).startswith("lgbm_quantile_")
        if reference:
            group["pred"] = group["q50"]
        daily = metric_daily_totals(group)
        if daily.empty:
            continue
        sums = daily.sum()
        observed = _ratios(sums)
        rng = np.random.default_rng(seed)
        indices = rng.integers(len(daily), size=(n_boot, len(daily)))
        draws = daily.to_numpy(dtype=float)[indices].sum(axis=1)
        sampled = _ratios({name: draws[:, i] for i, name in enumerate(daily.columns)})
        row = {"horizon": horizon, "model": model, "reference_only": reference,
               "selection_used": False, "metric_role": "보조 지표, 선정에 미사용",
               "n": int(sums["n"]), "peak_n": int(sums["peak_n"]),
               "union_n": int(sums["union_n"]), "nonpeak_n": int(sums["nonpeak_n"]),
               "target_days": len(daily), "bootstrap_n": n_boot,
               "episode_ci_method": "original_matching_event_start_date_blocks"}
        for name in METRICS:
            finite = np.asarray(sampled[name])[np.isfinite(sampled[name])]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            row.update({name: float(observed[name]), name + "_ci_low": float(low),
                        name + "_ci_high": float(high), name + "_valid_bootstrap_n": len(finite)})
        # Guard against accidentally changing the original point metrics.
        original = score_predictions(group)
        for name in ("peak_mae", "position_fp", "episode_f1"):
            if not np.isclose(row[name], original[name], atol=1e-12, equal_nan=True):
                raise AssertionError(f"Daily accounting changed {name} for {model}")
        rows.append(row)
    return pd.DataFrame(rows)
