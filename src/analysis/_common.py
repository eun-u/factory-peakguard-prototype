"""Validated inputs and statistical helpers for post-hoc analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def canonical_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts_end" in out.columns:
        out["ts_end"] = pd.to_datetime(out["ts_end"])
        out = out.set_index("ts_end")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("History requires a ts_end DatetimeIndex or column")
    if out.index.has_duplicates:
        raise ValueError("History has duplicate ts_end values")
    out.index.name = "ts_end"
    return out.sort_index()


def canonical_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"origin", "target_time", "horizon", "fold", "model", "y", "pred", "tau"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame lacks {sorted(missing)}")
    out = frame.copy()
    out["origin"] = pd.to_datetime(out["origin"])
    out["target_time"] = pd.to_datetime(out["target_time"])
    for col in ("y", "pred", "tau", "horizon"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["origin", "target_time", "y", "pred", "tau", "horizon"])
    if (out["origin"] >= out["target_time"]).any():
        raise ValueError("Prediction target must follow its origin")
    if (out["horizon"] <= 0).any():
        raise ValueError("Horizons must be positive")
    return out.sort_values(["horizon", "model", "fold", "target_time"]).reset_index(drop=True)


def selected_rows(predictions: pd.DataFrame, horizon: int, model: str | None = None) -> pd.DataFrame:
    subset = predictions.loc[predictions.horizon.eq(horizon)]
    if model is None:
        models = subset.model.astype(str).unique()
        model = "lgbm" if "lgbm" in models else (models[0] if len(models) else None)
    out = subset.loc[subset.model.eq(model)].copy()
    if out.empty:
        raise ValueError(f"No OOF predictions for horizon={horizon}, model={model}")
    if out.duplicated(["target_time"]).any():
        raise ValueError("Selected OOF prediction has duplicate target_time")
    return out.sort_values("target_time").reset_index(drop=True)


def attach_history(pred: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Target-time actuals are attached only for after-the-fact analysis."""
    return pred.join(history, on="target_time", rsuffix="_actual")


def block_ci(frame: pd.DataFrame, metric, n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Day-block percentile interval; returns estimate and lower/upper bounds."""
    if frame.empty:
        return (float("nan"),) * 3
    observed = float(metric(frame))
    days = pd.to_datetime(frame["target_time"]).dt.normalize()
    blocks = [frame.loc[days.eq(day)] for day in days.drop_duplicates()]
    if len(blocks) < 2:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        sample = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))], ignore_index=True)
        val = float(metric(sample))
        if np.isfinite(val):
            values.append(val)
    if not values:
        return observed, float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return observed, float(low), float(high)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path
