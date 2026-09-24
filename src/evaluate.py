"""Forecast and peak-episode evaluation on chronological scoring rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .bootstrap import day_mean_ci


def _episodes(mask, times=None):
    active = np.asarray(mask, dtype=bool)
    if times is None:
        times = pd.date_range("2000-01-01", periods=len(active), freq="15min")
    time = pd.DatetimeIndex(times)
    if len(active) != len(time):
        raise ValueError("Mask and timestamp lengths differ")
    segments = []
    start = None
    for i, flag in enumerate(active):
        adjacent = i > 0 and time[i] - time[i - 1] == pd.Timedelta(minutes=15)
        if start is not None and (not flag or not adjacent):
            segments.append((start, i - 1))
            start = None
        if flag and start is None:
            start = i
    if start is not None:
        segments.append((start, len(active) - 1))
    return segments


def match_episodes(actual_mask, alert_mask, times=None, actual_values=None, predicted_values=None):
    """Maximum-cardinality one-to-one overlap matching, then most overlap."""
    from scipy.optimize import linear_sum_assignment
    actual = _episodes(actual_mask, times)
    alerts = _episodes(alert_mask, times)
    overlap = np.zeros((len(actual), len(alerts)), dtype=int)
    for ai, (a0, a1) in enumerate(actual):
        for pi, (p0, p1) in enumerate(alerts):
            overlap[ai, pi] = max(0, min(a1, p1) - max(a0, p0) + 1)
    pairs = []
    if overlap.size:
        # Cardinality dominates total overlap. Zero-overlap pairs are discarded.
        reward = np.where(overlap > 0, 1_000_000 + overlap, 0)
        left, right = linear_sum_assignment(-reward)
        pairs = [(int(ai), int(pi)) for ai, pi in zip(left, right) if overlap[ai, pi] > 0]
    out = {"tp": len(pairs), "fp": len(alerts) - len(pairs), "fn": len(actual) - len(pairs),
           "actual_episodes": actual, "alert_episodes": alerts, "matches": pairs}
    if times is not None and actual_values is not None and predicted_values is not None:
        time = pd.DatetimeIndex(times)
        y = np.asarray(actual_values, dtype=float)
        p = np.asarray(predicted_values, dtype=float)
        timing, magnitude = [], []
        for ai, pi in pairs:
            a0, a1 = actual[ai]; p0, p1 = alerts[pi]
            ya = a0 + int(np.nanargmax(y[a0:a1 + 1]))
            pp = p0 + int(np.nanargmax(p[p0:p1 + 1]))
            timing.append(float((time[pp] - time[ya]).total_seconds() / 60))
            magnitude.append(float(p[pp] - y[ya]))
        out["timing_errors_minutes"] = timing
        out["magnitude_errors"] = magnitude
    return out


def _safe_mean(x):
    values = np.asarray(x, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def symmetric_peak_metrics(y, predicted, tau):
    """Auxiliary, selection-unused errors; predicted peak means pred > tau.

    Empty conditional populations return NaN. These measures deliberately
    do not use the separately calibrated operational/episode alert threshold.
    """
    y, predicted, tau = np.broadcast_arrays(np.asarray(y, dtype=float),
                                           np.asarray(predicted, dtype=float),
                                           np.asarray(tau, dtype=float))
    valid = np.isfinite(y) & np.isfinite(predicted) & np.isfinite(tau)
    y, predicted, tau = y[valid], predicted[valid], tau[valid]
    actual_peak, predicted_peak = y > tau, predicted > tau
    return {
        "peak_mae_union": _safe_mean(np.abs(predicted - y)[actual_peak | predicted_peak]),
        "peak_bias": _safe_mean((predicted - y)[actual_peak]),
        "overpredict_rate": _safe_mean(predicted_peak[~actual_peak]),
    }


def score_predictions(frame, threshold=None):
    """Score one model/horizon/fold without reselecting its alert threshold."""
    frame = frame.sort_values("target_time")
    y = frame.y.to_numpy(dtype=float)
    p = frame.pred.to_numpy(dtype=float)
    tau = frame.tau.to_numpy(dtype=float) if threshold is None else float(threshold)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.all():
        frame = frame.iloc[np.flatnonzero(valid)]
        y, p = y[valid], p[valid]
        tau = tau[valid] if np.ndim(tau) else tau
    if not len(y):
        return {}
    peak = y > tau
    alert = frame.alert.to_numpy(dtype=bool) if "alert" in frame else p > tau
    episodes = match_episodes(peak, alert, frame.target_time, y, p)
    n_alerts = int(alert.sum())
    tp = int((peak & alert).sum()); fp = int((~peak & alert).sum()); fn = int((peak & ~alert).sum())
    e_tp, e_fp, e_fn = (episodes[k] for k in ("tp", "fp", "fn"))
    nonzero = y != 0
    result = {
        "n": len(y), "peak_n": int(peak.sum()), "mae": _safe_mean(np.abs(y - p)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mape_nonzero": _safe_mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])) * 100 if nonzero.any() else float("nan"),
        "peak_mae": _safe_mean(np.abs(y[peak] - p[peak])),
        "position_tp": tp, "position_fp": fp, "position_fn": fn,
        "position_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "episode_tp": e_tp, "episode_fp": e_fp, "episode_fn": e_fn,
        "episode_f1": 2 * e_tp / (2 * e_tp + e_fp + e_fn) if 2 * e_tp + e_fp + e_fn else float("nan"),
        "false_alarms_positions": n_alerts - tp,
        "timing_mae_minutes": _safe_mean(np.abs(episodes.get("timing_errors_minutes", []))),
        "magnitude_mae": _safe_mean(np.abs(episodes.get("magnitude_errors", []))),
    }
    # Auxiliary only: the selection implementation consumes its original keys.
    result.update(symmetric_peak_metrics(y, p, tau))
    if "p_exceed" in frame and frame.p_exceed.notna().any():
        prob = frame.p_exceed.to_numpy(dtype=float)
        ok = np.isfinite(prob)
        if ok.any():
            result["brier"] = float(brier_score_loss(peak[ok], prob[ok]))
            result["pr_auc"] = float(average_precision_score(peak[ok], prob[ok])) if peak[ok].any() else float("nan")
    for level in (.1, .5, .9, .95, .975):
        suffix = "975" if level == .975 else str(int(level * 100))
        calibrated = f"q{suffix}_cal"
        raw = f"q{suffix}"
        # Mixed-model tables have calibration columns globally. A raw-model
        # group still has those columns, filled with NaN, so use them only
        # when this group's calibrated predictions are actually finite.
        col = calibrated if calibrated in frame and np.isfinite(frame[calibrated].to_numpy(dtype=float)).any() else raw
        if col in frame and np.isfinite(frame[col].to_numpy(dtype=float)).any():
            q = frame[col].to_numpy(dtype=float)
            mask = np.isfinite(q)
            result[f"coverage_{level}"] = _safe_mean(y[mask] <= q[mask])
            if "q50" in frame and frame.q50.notna().any():
                if "q50_top_edge" in frame and frame.q50_top_edge.notna().any():
                    hi = frame.q50.to_numpy(dtype=float) >= frame.q50_top_edge.to_numpy(dtype=float)
                else:
                    hi = frame.q50.to_numpy(dtype=float) >= np.nanquantile(frame.q50, .9)
                result[f"top_coverage_{level}"] = _safe_mean(y[mask & hi] <= q[mask & hi])
            result[f"pinball_{level}"] = _safe_mean(np.maximum(level * (y[mask] - q[mask]), (level - 1) * (y[mask] - q[mask])))
    losses = [result[f"pinball_{level}"] for level in (.1, .5, .9, .95, .975)
              if f"pinball_{level}" in result]
    if losses:
        result["pinball_mean"] = _safe_mean(losses)
    return result


def evaluate_all(pred_frame, tau=None, cfg=None):
    """A tidy metric table with date-block uncertainty for primary errors."""
    cfg = cfg or {}
    rows = []
    for key, group in pred_frame.groupby(["horizon", "model", "fold"], dropna=False, sort=True):
        score = score_predictions(group, threshold=tau)
        if not score:
            continue
        if len(group) and score["peak_n"]:
            peak = group.loc[group.y > (tau if tau is not None else group.tau)]
            score["peak_mae_ci95"] = list(day_mean_ci(peak, abs(peak.y - peak.pred),
                                                      n=int(cfg.get("bootstrap", {}).get("n", 1000)),
                                                      seed=int(cfg.get("seed", 42))))
        rows.append(dict(zip(["horizon", "model", "fold"], key), **score))
    return pd.DataFrame(rows)
