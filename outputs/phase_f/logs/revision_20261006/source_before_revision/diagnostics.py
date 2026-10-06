"""Descriptive Phase F error slices on one explicitly chosen score arm.

These tables are post hoc evidence, not model training data.  CONFIRM is
unavailable until the caller states that the primary candidate was frozen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.holidays import HOLIDAYS_2021


HOLIDAYS = frozenset(pd.Timestamp(date) for date in HOLIDAYS_2021)
LATE_JULY_START = pd.Timestamp("2021-07-19")
LATE_JULY_END = pd.Timestamp("2021-08-02")


def _summary(data: pd.DataFrame, label: str) -> dict:
    if data.empty:
        return {"group": label, "n": 0, "mae": np.nan, "rmse": np.nan, "bias_pred_minus_actual": np.nan,
                "peak_n": 0, "peak_mae": np.nan, "actual_peak_rate": np.nan, "pred_peak_rate": np.nan,
                "false_positive_n": 0, "false_negative_n": 0, "actual_mean": np.nan, "pred_mean": np.nan}
    actual = data.y.to_numpy(dtype=float)
    prediction = data.pred.to_numpy(dtype=float)
    tau = data.tau.to_numpy(dtype=float)
    err = prediction - actual
    actual_peak = actual > tau
    predicted_peak = prediction > tau
    return {"group": label, "n": int(len(data)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))), "bias_pred_minus_actual": float(np.mean(err)),
            "peak_n": int(actual_peak.sum()),
            "peak_mae": float(np.mean(np.abs(err[actual_peak]))) if actual_peak.any() else np.nan,
            "actual_peak_rate": float(actual_peak.mean()), "pred_peak_rate": float(predicted_peak.mean()),
            "false_positive_n": int((predicted_peak & ~actual_peak).sum()),
            "false_negative_n": int((actual_peak & ~predicted_peak).sum()),
            "actual_mean": float(actual.mean()), "pred_mean": float(prediction.mean())}


def _groups(data: pd.DataFrame, key: pd.Series | np.ndarray, label: str) -> pd.DataFrame:
    keyed = data.assign(_group=np.asarray(key))
    rows = []
    for value, part in keyed.groupby("_group", sort=True, dropna=False):
        rows.append({"slice": label, **_summary(part, str(value))})
    return pd.DataFrame(rows)


def _holiday_context(times: pd.DatetimeIndex) -> np.ndarray:
    dates = times.normalize()
    holiday = dates.isin(list(HOLIDAYS))
    before = (dates + pd.Timedelta(days=1)).isin(list(HOLIDAYS))
    after = (dates - pd.Timedelta(days=1)).isin(list(HOLIDAYS))
    return np.where(holiday, "holiday", np.where(before, "pre_holiday",
                    np.where(after, "post_holiday", "other")))


def summarize_errors(frame: pd.DataFrame, *, arm: str = "EXPLORE", selected: bool = False,
                     production_col: str | None = None) -> dict[str, pd.DataFrame | dict]:
    """Return F10 slices without touching the other score arm's outcomes.

    ``selected=True`` means candidate selection was completed before opening
    CONFIRM.  The caller must record that decision separately.  Production
    values, when present, are used only for descriptive post hoc bins.
    """
    if arm not in ("EXPLORE", "CONFIRM"):
        raise ValueError("arm must be EXPLORE or CONFIRM")
    if arm == "CONFIRM" and not selected:
        raise PermissionError("CONFIRM diagnostics require a frozen selected candidate")
    required = {"model", "horizon", "fold", "origin", "target_time", "y", "pred", "tau", "d2", "role", "arm"}
    if not isinstance(frame, pd.DataFrame) or required - set(frame):
        raise ValueError(f"diagnostic frame misses columns: {sorted(required - set(frame.columns))}")
    data = frame.loc[frame.role.eq("score") & frame.arm.eq(arm)].copy()
    if data.empty:
        raise ValueError(f"no score rows in {arm}")
    if data.model.nunique() != 1 or data.duplicated(["horizon", "fold", "origin", "target_time"]).any():
        raise ValueError("diagnostics require one candidate and unique keys")
    expected = np.where(pd.DatetimeIndex(data.target_time).isocalendar().week.to_numpy(dtype=int) % 2 == 0,
                        "EXPLORE", "CONFIRM")
    if not np.array_equal(data.arm.to_numpy(), expected):
        raise ValueError("score arm differs from target-calendar ISO-week lock")
    if not np.isfinite(data[["y", "pred", "tau"]].to_numpy(dtype=float)).all():
        raise ValueError("diagnostics require finite y/pred/tau")
    target = pd.DatetimeIndex(data.target_time)
    actual_peak = data.y.to_numpy(dtype=float) > data.tau.to_numpy(dtype=float)
    tables: dict[str, pd.DataFrame | dict] = {
        "by_hour": _groups(data, target.hour.to_numpy(dtype=int), "hour"),
        "by_weekday": _groups(data, target.dayofweek.to_numpy(dtype=int), "weekday_0_mon"),
        "by_holiday_context": _groups(data, _holiday_context(target), "holiday_context"),
        "by_actual_peak": _groups(data, np.where(actual_peak, "peak", "nonpeak"), "actual_peak"),
        "by_fold": _groups(data, data.fold.to_numpy(dtype=int), "fold"),
        "by_domain": pd.DataFrame([{"slice": "domain", **_summary(data, "D1_all")},
                                   {"slice": "domain", **_summary(data.loc[data.d2.astype(bool)], "D2_novel_profiles")}]),
    }
    # Two complete calendar weeks surrounding the reported late-July collapse.
    date = target.normalize()
    late = (date >= LATE_JULY_START) & (date < LATE_JULY_END)
    preceding = (date >= LATE_JULY_START - pd.Timedelta(days=14)) & (date < LATE_JULY_START)
    rows = [{"slice": "late_july", **_summary(data.loc[preceding], "preceding_two_weeks")},
            {"slice": "late_july", **_summary(data.loc[late], "late_july_two_weeks")}]
    if "augmented" in data:
        rows.append({"slice": "late_july", **_summary(data.loc[late & data.augmented.astype(bool)], "late_july_augmented")})
    if "quality_bad" in data:
        rows.append({"slice": "late_july", **_summary(data.loc[late & data.quality_bad.astype(bool)], "late_july_quality_bad")})
    tables["late_july"] = pd.DataFrame(rows)
    if production_col is not None:
        if production_col not in data:
            raise ValueError(f"production diagnostic column unavailable: {production_col}")
        values = pd.to_numeric(data[production_col], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        if finite.sum() >= 4 and values[finite].nunique() >= 2:
            bins = pd.qcut(values[finite], q=4, duplicates="drop")
            tables["by_production_posthoc"] = _groups(data.loc[finite], bins.astype(str), "production_posthoc")
        else:
            tables["by_production_posthoc"] = pd.DataFrame()
    tables["audit"] = {"model": str(data.model.iloc[0]), "arm": arm, "selected_before_confirm": bool(selected),
                       "score_n": int(len(data)), "fold_1_n": int(data.fold.eq(1).sum()),
                       "fold_1_d2_n": int((data.fold.eq(1) & data.d2.astype(bool)).sum()),
                       "late_july_start_inclusive": str(LATE_JULY_START.date()),
                       "late_july_end_exclusive": str(LATE_JULY_END.date()),
                       "late_july_augmented_status": "observed" if "augmented" in data else "UNKNOWN",
                       "late_july_quality_status": "observed" if "quality_bad" in data else "UNKNOWN",
                       "production_use": "posthoc_only" if production_col else "not_requested",
                       "selection_use": False}
    return tables
