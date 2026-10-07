"""Descriptive distance between training and OOF-origin feature distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ._common import write_table


def _snapshot(history: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    power = history.power.astype(float)
    if "time_repaired" in history:
        power = power.mask(history.time_repaired.astype(bool))
    known = pd.to_numeric(history.get("production_known", pd.Series(index=history.index, dtype=float)), errors="coerce")
    if "production_known_bad" in history:
        known = known.mask(history.production_known_bad.astype(bool))
    x = pd.DataFrame(index=history.index)
    x["power_now"] = power
    x["power_1h_mean"] = power.rolling(4, min_periods=4).mean()
    x["power_1d_mean"] = power.rolling(96, min_periods=96).mean()
    x["known_production"] = known
    x["hour_sin"] = np.sin(2*np.pi*(x.index.hour+x.index.minute/60)/24)
    x["hour_cos"] = np.cos(2*np.pi*(x.index.hour+x.index.minute/60)/24)
    return x.reindex(origins).dropna()


def run_applicability(history: pd.DataFrame, train: pd.DataFrame, pred: pd.DataFrame,
                      outdir: Path) -> dict:
    train_x = _snapshot(history, pd.DatetimeIndex(train.index))
    score_x = _snapshot(history, pd.DatetimeIndex(pred.origin))
    if len(train_x) < 2 or score_x.empty:
        return {"status": "unsupported", "reason": "Insufficient observed-origin rows for distance comparison"}
    median = train_x.median()
    scale = (train_x.quantile(.75)-train_x.quantile(.25)).replace(0, 1).fillna(1)
    train_scaled = ((train_x-median)/scale).to_numpy(float)
    score_scaled = ((score_x-median)/scale).to_numpy(float)
    nn = NearestNeighbors(n_neighbors=2).fit(train_scaled)
    train_dist = nn.kneighbors(train_scaled)[0][:, 1]
    score_dist = nn.kneighbors(score_scaled, n_neighbors=1)[0][:, 0]
    reference_95 = float(np.quantile(train_dist, .95))
    detail = pd.DataFrame({"origin": score_x.index, "nearest_train_distance": score_dist,
                           "beyond_training_p95": score_dist > reference_95})
    path = write_table(detail, outdir/"tables"/"applicability_distance.csv")
    return {"status": "ok", "path": str(path), "train_distance_p95": reference_95,
            "oof_fraction_beyond_p95": float((score_dist > reference_95).mean()),
            "warning": "거리 계산은 관측 가능한 일부 특징의 개발구간 진단입니다. 12월 미관측의 외삽 위험을 직접 입증하거나 제거하지 않습니다."}
