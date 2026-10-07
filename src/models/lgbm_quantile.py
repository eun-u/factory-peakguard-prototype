"""LightGBM 분위수 예측, 분위수 보간 초과확률, split conformal 보정."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lgbm_point import regression_model


def fit_quantiles(cfg: dict, x: pd.DataFrame, y: pd.Series, quantiles) -> dict:
    return {q: regression_model(cfg, alpha=q).fit(x, y) for q in quantiles}


def predict_quantiles(models: dict, x: pd.DataFrame) -> np.ndarray:
    return np.column_stack([models[q].predict(x) for q in models])


def quantile_event_probability(preds: np.ndarray, threshold: float, quantiles) -> tuple[np.ndarray, int]:
    """분위수 4개 사이를 선형 보간해 P(y > threshold)를 근사한다. 교차는 정렬로 해소한다."""
    q = tuple(quantiles)
    crossings = int(np.sum(np.any(np.diff(preds, axis=1) < 0, axis=1)))
    p = np.sort(preds, axis=1)
    cdf = np.full(len(p), 0.95)
    cdf[threshold <= p[:, 0]] = 0.10
    for j in range(3):
        mask = (threshold > p[:, j]) & (threshold <= p[:, j + 1])
        width = np.maximum(p[mask, j + 1] - p[mask, j], 1e-9)
        cdf[mask] = q[j] + (q[j + 1] - q[j]) * (threshold - p[mask, j]) / width
    return np.clip(1.0 - cdf, 0, 1), crossings


def conformal_offset(y_cal: np.ndarray, q_cal: np.ndarray, level: float) -> float:
    """상단 분위수 q̂의 split conformal 보정량 δ: P(y ≤ q̂ + δ) ≥ level (교환가능성 가정)."""
    scores = np.sort(y_cal - q_cal)
    n = len(scores)
    rank = int(np.ceil((n + 1) * level))
    return float(scores[min(rank, n) - 1])
