"""기준선: 계절 나이브 3종과 persistence 3종. 각 계열의 대표는 검증 MAE 최소로 시험 전에 고정한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import seasonal_columns

PERSISTENCE_COLUMNS = {
    "latest_15m": "current",              # P-a 원점의 최근 15분값
    "previous_hour_slot": "lag_4",        # P-b 원점 1시간 전 같은 위치값
    "recent_hour_mean": "recent_hour_mean",  # P-c 최근 네 15분값 평균
}


def naive_predict(x: pd.DataFrame, version: str, horizon: int = 4) -> np.ndarray:
    cols = seasonal_columns(horizon)
    day = x[cols["day"]].to_numpy()
    week = x[cols["week"]].to_numpy()
    return {"day": day, "week": week, "average": (day + week) / 2}[version]


def persistence_predict(x: pd.DataFrame, version: str) -> np.ndarray:
    return x[PERSISTENCE_COLUMNS[version]].to_numpy(dtype=float)
