"""Cal-only Phase F ensemble and peak postprocessors.

Inputs use ``phase_f.harness.make_frame`` columns and contain full cal and
score rows for every candidate.  Fitted numbers are estimated separately per
horizon and fold from cal rows.  Score labels are checked for cohort identity
but are never used to compute a prediction or fitted value.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.holidays import HOLIDAYS_2021


KEY = ("horizon", "fold", "role", "origin", "target_time")
META = ("y", "tau", "d2", "arm", "fit_mean", "mase_scale")
REQUIRED = set(KEY) | set(META) | {"model", "pred"}
BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
HOLIDAYS = frozenset(pd.Timestamp(date) for date in HOLIDAYS_2021)


def configurations() -> list[dict]:
    """Fixed starting operators; later searches must register their params."""
    return [
        {"id": "F7-1-mean", "family": "F7-1", "method": "mean", "params": {}},
        {"id": "F7-1-median", "family": "F7-1", "method": "median", "params": {}},
        {"id": "F7-2-inverse-cal-mae", "family": "F7-2", "method": "inverse_cal_mae", "params": {}},
        {"id": "F7-3-nnls", "family": "F7-3", "method": "nnls", "params": {}},
        {"id": "F7-3-ridge-positive", "family": "F7-3", "method": "ridge_positive", "params": {"ridge": 1.0}},
        {"id": "F7-4-regime-gate", "family": "F7-4", "method": "regime_gate", "params": {"min_count": 30}},
        {"id": "F7-5-peak-gate", "family": "F7-5", "method": "peak_gate", "params": {"min_count": 30, "peak_min_count": 10, "risk_threshold": .5}},
        {"id": "F8-2-bias-hour-daytype", "family": "F8-2", "method": "bias_hour_daytype", "params": {"min_count": 8}},
        {"id": "F8-3-q90-risk-025", "family": "F8-3", "method": "quantile_risk_shift", "params": {"shift": .25, "risk_threshold": .5}},
        {"id": "F8-3-q90-risk-050", "family": "F8-3", "method": "quantile_risk_shift", "params": {"shift": .5, "risk_threshold": .5}},
    ]


def base_combinations() -> dict[str, tuple[str, ...]]:
    """Symbolic F7-6 sets; best_* IDs are resolved before combination."""
    return {
        "F7-6-b5-lgbm": ("B5", "best_lgbm"),
        "F7-6-b5-r1": ("B5", "R1"),
        "F7-6-r1-lgbm": ("R1", "best_lgbm"),
        "F7-6-b5-r1-lgbm-dl": ("B5", "R1", "best_lgbm", "best_dl"),
    }


def _frames(inputs: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if isinstance(inputs, Mapping):
        items = list(inputs.items())
    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        items = [(str(frame.model.iloc[0]), frame) for frame in inputs]
    else:
        raise TypeError("inputs must map model IDs to prediction DataFrames")
    if not items or len({name for name, _ in items}) != len(items):
        raise ValueError("candidate inputs must have distinct model IDs")
    prepared = {}
    reference = None
    for name, frame in items:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("candidate model IDs must be safe and nonempty")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must have a nonempty prediction frame")
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{name} misses required columns: {sorted(missing)}")
        if frame.model.nunique() != 1 or frame.model.iloc[0] != name:
            raise ValueError(f"frame/model ID mismatch for {name}")
        ordered = frame.sort_values(list(KEY), kind="stable").reset_index(drop=True).copy()
        if ordered.duplicated(list(KEY)).any():
            raise ValueError(f"duplicate prediction keys for {name}")
        if not ordered.role.isin(("cal", "score")).all() or not {"cal", "score"} <= set(ordered.role):
            raise ValueError(f"{name} needs both cal and score roles")
        if not ordered.loc[ordered.role.eq("cal"), "arm"].eq("CAL").all():
            raise ValueError("cal rows must carry CAL arm")
        if not ordered.loc[ordered.role.eq("score"), "arm"].isin(("EXPLORE", "CONFIRM")).all():
            raise ValueError("score rows need pre-locked EXPLORE/CONFIRM arm")
        score = ordered.loc[ordered.role.eq("score")]
        expected_arm = np.where(pd.DatetimeIndex(score.target_time).isocalendar().week.to_numpy(dtype=int) % 2 == 0,
                                "EXPLORE", "CONFIRM")
        if not np.array_equal(score.arm.to_numpy(), expected_arm):
            raise ValueError("score arm differs from target-calendar ISO-week lock")
        if not np.isfinite(ordered[["pred", "y", "tau", "fit_mean", "mase_scale"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains nonfinite prediction/metadata")
        if (ordered.target_time >= BOUNDARY).any() or (ordered.origin >= BOUNDARY).any():
            raise ValueError("sealed boundary violation")
        groups = ordered.groupby(["horizon", "fold"], sort=False)
        for _, part in groups:
            cal = part.loc[part.role.eq("cal")]
            score = part.loc[part.role.eq("score")]
            if cal.empty or score.empty:
                raise ValueError("every horizon-fold needs cal and score")
            if cal.target_time.max() >= score.origin.min():
                raise ValueError("cal target/score origin embargo violated")
        if reference is not None:
            if not pd.MultiIndex.from_frame(ordered[list(KEY)]).equals(pd.MultiIndex.from_frame(reference[list(KEY)])):
                raise ValueError(f"{name} has mismatched cal/score cohort")
            for column in META:
                if not np.array_equal(ordered[column].to_numpy(), reference[column].to_numpy()):
                    raise ValueError(f"{name} has mismatched {column}")
        else:
            reference = ordered
        prepared[name] = ordered
    return prepared


def _daytype(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    return np.where(index.normalize().isin(list(HOLIDAYS)), 3,
                    np.where(index.dayofweek == 5, 1, np.where(index.dayofweek == 6, 2, 0))).astype(int)


def _hour_daytype(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    return index.hour.to_numpy(dtype=int) * 4 + _daytype(index)


def _regime(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    return (index.hour.to_numpy(dtype=int) // 8) * 4 + _daytype(index)


def _inverse_mae(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    errors = np.mean(np.abs(x - y[:, None]), axis=0)
    inverse = 1 / np.maximum(errors, 1e-8)
    return inverse / inverse.sum()


def _stack(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    n = x.shape[1]
    if n == 1:
        return np.ones(1)
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be nonnegative and finite")

    def objective(w):
        err = x @ w - y
        return float(np.mean(err**2) + ridge * np.sum(w**2))

    result = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                      options={"ftol": 1e-10, "maxiter": 1000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"constrained stack optimization failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def _risk(source: pd.DataFrame, threshold: float) -> tuple[np.ndarray, str]:
    if not 0 <= threshold <= 1:
        raise ValueError("risk_threshold must be within [0,1]")
    if "p_peak" in source and np.isfinite(source.p_peak.to_numpy(dtype=float)).all():
        probability = source.p_peak.to_numpy(dtype=float)
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("p_peak must be in [0,1]")
        return probability >= threshold, "p_peak"
    if "q90" in source and np.isfinite(source.q90.to_numpy(dtype=float)).all():
        return source.q90.to_numpy(dtype=float) > source.tau.to_numpy(dtype=float), "q90_gt_tau"
    raise ValueError("peak gate needs finite p_peak or q90 from risk_source")


def combine_predictions(inputs: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame], method: str,
                        model_id: str, *, params: Mapping | None = None) -> tuple[pd.DataFrame, dict]:
    """Return aligned cal+score predictions plus an auditable cal fit record.

    ``model_id`` names the result.  For bias correction use one source or set
    ``base_id``; for quantile shift set ``base_id`` and ``risk_source`` as needed.
    No score metric, arm ranking or score label participates in any fit.
    """
    frames = _frames(inputs)
    names = list(frames)
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("model_id must be a safe nonempty ID")
    if model_id in frames:
        raise ValueError("combined model_id must differ from source models")
    config = dict(params or {})
    reference = next(iter(frames.values()))
    matrix = np.column_stack([frame.pred.to_numpy(dtype=float) for frame in frames.values()])
    out = reference[[*KEY, *META]].copy()
    out.insert(0, "model", model_id)
    out["pred"] = np.nan
    out["development_only"] = True
    audits = []
    base_id = str(config.get("base_id", names[0]))
    risk_id = str(config.get("risk_source", base_id))
    if base_id not in frames or risk_id not in frames:
        raise ValueError("base_id and risk_source must name supplied models")
    base_index = names.index(base_id)
    for (horizon, fold), group in reference.groupby(["horizon", "fold"], sort=True):
        ix = group.index.to_numpy(dtype=int)
        cal = group.role.eq("cal").to_numpy(dtype=bool)
        cal_ix = ix[cal]
        y = reference.loc[cal_ix, "y"].to_numpy(dtype=float)
        x = matrix[cal_ix]
        all_x = matrix[ix]
        decision = {"horizon": int(horizon), "fold": int(fold), "cal_n": int(len(cal_ix)),
                    "cal_first_origin": str(reference.loc[cal_ix, "origin"].min()),
                    "cal_last_origin": str(reference.loc[cal_ix, "origin"].max()),
                    "cal_last_target": str(reference.loc[cal_ix, "target_time"].max()),
                    "score_first_origin": str(group.loc[~group.role.eq("cal"), "origin"].min())}
        if method == "mean":
            weights = np.full(len(names), 1 / len(names))
            pred = all_x @ weights
            decision["weights"] = dict(zip(names, weights.tolist()))
        elif method == "median":
            pred = np.median(all_x, axis=1)
        elif method in {"inverse_cal_mae", "nnls", "ridge_positive"}:
            weights = (_inverse_mae(x, y) if method == "inverse_cal_mae" else
                       _stack(x, y, 0.0 if method == "nnls" else float(config.get("ridge", 1.0))))
            pred = all_x @ weights
            decision["weights"] = dict(zip(names, weights.tolist()))
        elif method == "regime_gate":
            minimum = int(config.get("min_count", 30))
            if minimum < 1:
                raise ValueError("min_count must be positive")
            global_weights = _inverse_mae(x, y)
            category = _regime(group.target_time)
            pred = np.empty(len(ix))
            specific = {}
            for value in np.unique(category):
                positions = category == value
                training = positions[cal]
                weight = _inverse_mae(x[training], y[training]) if training.sum() >= minimum else global_weights
                pred[positions] = all_x[positions] @ weight
                specific[str(int(value))] = {"n_cal": int(training.sum()), "fallback": bool(training.sum() < minimum),
                                              "weights": dict(zip(names, weight.tolist()))}
            decision.update({"minimum": minimum, "global_weights": dict(zip(names, global_weights.tolist())),
                             "regimes": specific})
        elif method == "peak_gate":
            minimum = int(config.get("min_count", 30))
            peak_minimum = int(config.get("peak_min_count", 10))
            if minimum < 1 or peak_minimum < 1:
                raise ValueError("gate minimum counts must be positive")
            threshold = float(config.get("risk_threshold", .5))
            risk, source_kind = _risk(frames[risk_id].loc[ix], threshold)
            actual_peak = y > reference.loc[cal_ix, "tau"].to_numpy(dtype=float)
            global_weights = _inverse_mae(x, y)
            high_training = risk[cal] & actual_peak
            low_training = (~risk[cal]) & (~actual_peak)
            if high_training.sum() < peak_minimum:
                high_training = actual_peak
            if low_training.sum() < minimum:
                low_training = ~actual_peak
            high_weights = _inverse_mae(x[high_training], y[high_training]) if high_training.sum() >= peak_minimum else global_weights
            low_weights = _inverse_mae(x[low_training], y[low_training]) if low_training.sum() >= minimum else global_weights
            pred = np.empty(len(ix))
            pred[risk] = all_x[risk] @ high_weights
            pred[~risk] = all_x[~risk] @ low_weights
            decision.update({"risk_source": risk_id, "risk_kind": source_kind, "risk_threshold": threshold,
                             "high_cal_n": int(high_training.sum()), "low_cal_n": int(low_training.sum()),
                             "high_weights": dict(zip(names, high_weights.tolist())),
                             "low_weights": dict(zip(names, low_weights.tolist()))})
        elif method == "bias_hour_daytype":
            minimum = int(config.get("min_count", 8))
            if minimum < 1:
                raise ValueError("min_count must be positive")
            residual = y - x[:, base_index]
            global_bias = float(np.median(residual))
            category = _hour_daytype(group.target_time)
            pred = all_x[:, base_index].copy()
            corrections = {}
            for value in np.unique(category):
                positions = category == value
                train = positions[cal]
                bias = float(np.median(residual[train])) if train.sum() >= minimum else global_bias
                pred[positions] += bias
                corrections[str(int(value))] = {"n_cal": int(train.sum()), "fallback": bool(train.sum() < minimum),
                                                 "bias": bias}
            decision.update({"base_id": base_id, "minimum": minimum, "global_bias": global_bias,
                             "corrections": corrections, "cal_predictions_in_sample": True})
        elif method == "quantile_risk_shift":
            shift = float(config.get("shift", .25))
            if not 0 <= shift <= 1 or not np.isfinite(shift):
                raise ValueError("shift must be a fixed number in [0,1]")
            threshold = float(config.get("risk_threshold", .5))
            source = frames[risk_id].loc[ix]
            if "q90" not in source or not np.isfinite(source.q90.to_numpy(dtype=float)).all():
                raise ValueError("quantile risk shift needs finite q90 from risk_source")
            risk, source_kind = _risk(source, threshold)
            pred = all_x[:, base_index] + shift * risk * np.maximum(source.q90.to_numpy(dtype=float) - all_x[:, base_index], 0)
            decision.update({"base_id": base_id, "risk_source": risk_id, "risk_kind": source_kind,
                             "risk_threshold": threshold, "fixed_shift": shift})
        else:
            raise ValueError(f"unknown ensemble method: {method}")
        out.loc[ix, "pred"] = pred
        audits.append(decision)
    if not np.isfinite(out.pred.to_numpy(dtype=float)).all():
        raise RuntimeError("combined prediction contains a missing value")
    audit = {"method": method, "model_id": model_id, "base_ids": names,
             "fit_roles": ["cal"], "score_labels_used_for_fit": False, "score_arms_read_for_metrics": False,
             "params": config, "groups": audits}
    return out, audit
