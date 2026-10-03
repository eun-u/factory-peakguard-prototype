"""F9-4 post-CONFIRM weekly refit diagnostic on one sealed development fold.

This is a separate operating simulation. Its refreshed fits never replace
frozen-fold candidate selection or Phase F CONFIRM results. At each checkpoint,
training labels have become observable by that checkpoint (target_time <=
update_time), and both the primary regression and B5 receive the same update.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

import numpy as np
import pandas as pd

from phase_c.statistical import fit_kalman, predict_kalman
from phase_f.models.regression import KINDS as REGRESSION_KINDS
from phase_f.models.regression import fit_model as fit_regression
from phase_f.models.regression import predict_model as predict_regression
from phase_f.registry import config_hash


BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
SUPPORTED_KINDS = REGRESSION_KINDS - {"two_stage"}


def simulate_weekly_refit(history: pd.DataFrame, context: Mapping,
                          score_origins: pd.DatetimeIndex, *, candidate: str,
                          kind: str, frozen_config: Mapping, lock: Mapping,
                          horizon: int = 16, interval_days: int = 7) -> dict[str, pd.DataFrame]:
    """Return paired online candidate/B5 predictions and a leakage audit.

    `lock` must contain the fixed primary and completed CONFIRM record plus a
    SHA-256 digest of the exact ordered `score_origins` list and the frozen
    candidate config. Other model families require separate explicit update
    semantics and are reported unsupported by raising NotImplementedError.
    """
    if kind not in SUPPORTED_KINDS or frozen_config.get("target") == "b5_residual":
        raise NotImplementedError("F9-4 currently supports independent tabular ridge/GBDT point regressors; two-stage, ensemble, foundation and B5-residual update rules need separate locks")
    if lock.get("confirm_complete") is not True or not lock.get("confirm_once_sha256") or not lock.get("selection_lock_sha256"):
        raise ValueError("Weekly refit requires completed one-time CONFIRM and fixed selection")
    if candidate != lock.get("weekly_diagnostic_candidate",lock.get("primary_candidate")) or config_hash(frozen_config) != lock.get("frozen_config_hash"):
        raise ValueError("Weekly refit candidate/config differs from frozen primary")
    if "horizon" not in frozen_config:
        raise ValueError("Frozen weekly refit config must contain the horizon")
    if horizon != int(frozen_config.get("horizon", horizon)) or not 4 <= horizon <= 16:
        raise ValueError("Weekly refit horizon conflicts with frozen configuration")
    if interval_days != 7:
        raise ValueError("F9-4 interval is fixed at seven days")
    if not isinstance(history, pd.DataFrame) or "power" not in history or not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Sealed development power history is required")
    if history.empty or history.index.max() >= BOUNDARY or history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("History is not a sorted sealed development prefix")
    score = pd.DatetimeIndex(score_origins, name="origin")
    if score.empty or score.has_duplicates or not score.is_monotonic_increasing or not score.isin(history.index).all():
        raise ValueError("Score origins must be nonempty, unique, sorted and in history")
    if config_hash([str(item) for item in score]) != lock.get("score_origin_sha256"):
        raise ValueError("Weekly score origins differ from their fixed diagnostic lock")
    targets = context.get("target_time")
    y = context.get("y")
    if not isinstance(targets, pd.Series) or not isinstance(y, pd.Series) or "tau" not in context:
        raise ValueError("Context requires origin-indexed targets, truth and fit-only tau")
    target_time = pd.to_datetime(targets, errors="raise")
    if target_time.isna().any() or not target_time.index.equals(y.index):
        raise ValueError("Context target/truth index differs")
    if not score.isin(target_time.index).all():
        raise ValueError("A fixed score origin is missing from context")
    if not target_time.loc[score].eq(pd.Series(score + timedelta(minutes=15 * horizon), index=score)).all():
        raise ValueError("Score target times conflict with direct horizon")
    if target_time.loc[score].ge(BOUNDARY).any():
        raise ValueError("A diagnostic score target reaches the sealed boundary")

    # No query of score y enters candidate fitting: known uses only already
    # arrived targets. y is joined for diagnostics only after all forecasts.
    all_origins = pd.DatetimeIndex(target_time.index)
    if all_origins.has_duplicates or not all_origins.is_monotonic_increasing:
        raise ValueError("Context origins must be unique and chronological")
    if not all_origins.isin(history.index).all():
        raise ValueError("Context origin is not in history")
    represented = []
    audits = []
    first = score.min()
    checkpoints = pd.date_range(first, score.max(), freq=f"{interval_days}D")
    for checkpoint in checkpoints:
        following = checkpoint + timedelta(days=interval_days)
        batch = score[(score >= checkpoint) & (score < following)]
        if batch.empty:
            continue
        known_mask = ((all_origins < checkpoint) & target_time.le(checkpoint).to_numpy() &
                      np.isfinite(pd.to_numeric(y, errors="coerce").to_numpy(float)))
        known = all_origins[known_mask]
        if known.empty:
            raise ValueError("No labels available before first weekly update")
        if target_time.loc[known].max() > checkpoint or known.max() >= checkpoint:
            raise AssertionError("Weekly update saw an unarrived target")
        fit_context = {"fit": known, "stop": pd.DatetimeIndex([]), "y": y,
                       "target_time": target_time, "tau": float(context["tau"]),
                       "horizon": horizon}
        cfg = dict(frozen_config)
        fitted = fit_regression(kind, history, fit_context, cfg)
        b5 = fit_kalman(history.power, known)
        candidate_pred = predict_regression(fitted, history, batch, horizon)
        b5_pred = predict_kalman(b5, history.power, batch, horizon)
        if not np.isfinite(candidate_pred).all() or not np.isfinite(b5_pred).all():
            raise ValueError("Weekly refit produced incomplete fixed-cohort predictions")
        represented.append(pd.DataFrame({"candidate": candidate, "baseline": "B5", "kind": kind,
                                         "horizon": horizon, "origin": batch,
                                         "target_time": target_time.loc[batch].to_numpy(),
                                         "candidate_pred": candidate_pred, "b5_pred": b5_pred,
                                         "update_time": checkpoint,
                                         "max_train_target_time": target_time.loc[known].max()}))
        audits.append({"update_time": checkpoint, "forecast_first": batch.min(),
                       "forecast_last": batch.max(), "forecast_n": len(batch),
                       "train_n": len(known), "max_train_origin": known.max(),
                       "max_train_target_time": target_time.loc[known].max(),
                       "candidate_fit_end": fitted["fit_end"],
                       "b5_fit_n": b5.get("n_fit_updates", np.nan),
                       "config_hash": lock["frozen_config_hash"],
                       "arrived_score_labels_used_at_update": int(score.isin(known).sum()),
                       "unarrived_score_labels_used_at_update": False})
    if not represented:
        raise ValueError("No weekly diagnostic forecasts")
    predictions = pd.concat(represented, ignore_index=True).sort_values("origin").reset_index(drop=True)
    if not pd.DatetimeIndex(predictions.origin).equals(score):
        raise AssertionError("Weekly diagnostic lost or duplicated fixed score origins")
    # The result may contain truth for an after-CONFIRM diagnostic report; it
    # was never handed to a fit before its target time had passed.
    predictions["y"] = y.loc[score].to_numpy(float)
    if not np.isfinite(predictions.y).all():
        raise ValueError("Diagnostic truth is incomplete")
    predictions["tau"] = float(context["tau"])
    if "d2" in context:
        d2 = context["d2"].reindex(score)
        if d2.isna().any() or not d2.isin((True, False)).all():
            raise ValueError("D2 status is incomplete on fixed score origins")
        predictions["d2"] = d2.to_numpy(bool)
    return {"predictions": predictions, "updates": pd.DataFrame(audits)}
