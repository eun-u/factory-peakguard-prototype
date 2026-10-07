"""Fixed CBL residual LightGBM candidate for preregistered Phase 2 M1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cbl import cbl_predict
from .lgbm_point import fit_point
from ..session_data import SEALED_BOUNDARY
from ..training import _cutoff, _row


MODEL_NAME = "lgbm_residual_cbl"
BASELINE_BY_HORIZON = {4: "c2a_max_4_5_adjusted", 16: "c3_holiday_hybrid",
                       96: "c3_holiday_hybrid"}
HOLIDAY_FEATURES = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")


def residual_baseline(history: pd.DataFrame, origins: pd.DatetimeIndex,
                      horizon: int) -> pd.Series:
    """Recreate the original CBL representative without filling missing rows.

    ``latest_observation`` is conservative: adjusted CBL can use readings
    through the origin, and both reference methods prohibit later readings.
    """
    if horizon not in BASELINE_BY_HORIZON:
        raise ValueError("No preregistered CBL residual baseline for horizon")
    if (not isinstance(history.index, pd.DatetimeIndex) or history.empty or
            history.index.max() >= SEALED_BOUNDARY or history.index.has_duplicates or
            not history.index.is_monotonic_increasing):
        raise ValueError("CBL residual baseline requires sealed sorted history")
    origins = pd.DatetimeIndex(origins, name="origin")
    if origins.empty or origins.max() >= SEALED_BOUNDARY or not origins.isin(history.index).all():
        raise ValueError("Residual origins must be on sealed observed history grid")
    targets = origins + pd.Timedelta(minutes=15*horizon)
    if horizon == 4:
        values = cbl_predict(history, targets, "max_4_5", True, origins)
    else:
        holiday = cbl_predict(history, targets, "holiday_mid_4_6", False, origins)
        workday = cbl_predict(history, targets, "mid_6_10", False, origins)
        values = holiday.combine_first(workday)
    values = pd.Series(values.to_numpy(dtype=float), index=origins,
                       name="residual_baseline")
    values.attrs["latest_observation"] = pd.Series(origins, index=origins)
    values.attrs["baseline_method"] = BASELINE_BY_HORIZON[horizon]
    return values


@dataclass
class ResidualFoldResult:
    model: object
    predictions: pd.DataFrame
    audit: pd.DataFrame
    metadata: dict


def fit_residual_fold(history: pd.DataFrame, cfg: dict, context: dict,
                      horizon: int, fold: int) -> ResidualFoldResult:
    """Fit one fixed-parameter residual model on the original fold partitions."""
    recorded = context["expected"]
    if (int(recorded["horizon"]) != horizon or int(recorded["fold"]) != fold or
            context["tau"] != recorded["tau"]):
        raise ValueError("Context differs from original fold manifest")
    if recorded["chosen_params"] not in cfg["lgbm"]["grid"]:
        raise ValueError("Original chosen hyperparameters are outside the fixed grid")
    origins = context["fit"].append(context["stop"]).append(context["cal"]).append(context["score"])
    if origins.has_duplicates:
        raise ValueError("Residual partitions overlap")
    b = residual_baseline(history, origins, horizon)
    x = context["x"]
    columns = [c for c in x.columns if c not in HOLIDAY_FEATURES] + ["residual_baseline"]
    augmented = x.loc[origins].copy()
    augmented["residual_baseline"] = b.reindex(origins).to_numpy(dtype=float)
    provenance = x.attrs.get("latest_observation_by_feature", {})
    if any((pd.DatetimeIndex(stamps) > x.index).any() for stamps in provenance.values()):
        raise AssertionError("Original feature provenance reached after origin")
    if (pd.DatetimeIndex(b.attrs["latest_observation"]) > origins).any():
        raise AssertionError("CBL residual baseline uses after-origin observation")
    partition: dict[str, pd.DatetimeIndex] = {}
    excluded = []
    for name in ("fit", "stop", "cal", "score"):
        original = context[name]
        finite = np.isfinite(b.reindex(original).to_numpy(dtype=float))
        kept = original[finite]
        if len(kept) < 30:
            raise ValueError(f"h{horizon} fold{fold} {name} has <30 finite baseline rows")
        partition[name] = kept
        excluded.append({"horizon": horizon, "fold": fold, "partition": name,
                         "baseline_model": BASELINE_BY_HORIZON[horizon],
                         "original_rows": len(original), "finite_baseline_rows": len(kept),
                         "excluded_baseline_rows": len(original)-len(kept),
                         "excluded_baseline_fraction": (len(original)-len(kept))/len(original)})
    fit, stop, cal, score = (partition[name] for name in ("fit", "stop", "cal", "score"))
    targets = context["targets"]
    y = targets.y
    fit_y = y.loc[fit]-b.loc[fit]
    stop_y = y.loc[stop]-b.loc[stop]
    model = fit_point(augmented.loc[fit, columns], fit_y,
                      augmented.loc[stop, columns], stop_y, cfg,
                      peak_threshold=None, peak_weight=1., params=recorded["chosen_params"])
    cal_pred = b.loc[cal].to_numpy(float)+model.predict(augmented.loc[cal, columns])
    cutoff = _cutoff(y.loc[cal].to_numpy(float), cal_pred, context["tau"],
                     targets.loc[cal, "target_time"])
    score_pred = b.loc[score].to_numpy(float)+model.predict(augmented.loc[score, columns])
    row = _row(score, horizon, fold, MODEL_NAME, targets, context["tau"], score_pred, cutoff,
               residual_baseline=b.loc[score].to_numpy(float),
               baseline_model=BASELINE_BY_HORIZON[horizon],
               reference_only=(horizon == 4), selection_used=False)
    metadata = {"horizon": horizon, "fold": fold, "model_name": MODEL_NAME,
                "baseline_model": BASELINE_BY_HORIZON[horizon],
                "feature_names": columns, "n_features": len(columns),
                "chosen_params": dict(recorded["chosen_params"]),
                "fit_end": str(fit.max()), "stop_end": str(stop.max()),
                "cal_end": str(cal.max()), "score_start": str(score.min()),
                "score_end": str(score.max()),
                "n_fit": len(fit), "n_stop": len(stop), "n_cal": len(cal), "n_score": len(score),
                "original_n_fit": len(context["fit"]), "original_n_stop": len(context["stop"]),
                "original_n_cal": len(context["cal"]), "original_n_score": len(context["score"]),
                "best_iteration": int(model.best_iteration_),
                "peak_weight": 1., "reference_only": horizon == 4}
    return ResidualFoldResult(model, row, pd.DataFrame(excluded), metadata)
