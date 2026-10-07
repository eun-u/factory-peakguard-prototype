"""비용-손실(cost-loss) 의사결정과 상대 경제가치(REV).

C/L = r일 때 초과확률 > r이면 조치한다. L로 나눈 실현 비용은 r×조치 + 실제 초과×(1−조치),
완전예측 비용은 r×실제 초과다. REV = (기후평균 비용 − 모델 비용) / (기후평균 비용 − 완전예측 비용).
기후평균은 학습 구간에서 같은 요일·15분 위치의 초과 빈도를 라플라스 평활한 값이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def slot(times: pd.Series) -> np.ndarray:
    return (times.dt.dayofweek.to_numpy() * 96 + times.dt.hour.to_numpy() * 4
            + times.dt.minute.to_numpy() // 15)


def climatology(train_meta: pd.DataFrame, train_event: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    tr = pd.DataFrame({"slot": slot(train_meta["target_time"]), "event": train_event})
    grouped = tr.groupby("slot")["event"].agg(["sum", "count"])
    rate = (grouped["sum"] + 1) / (grouped["count"] + 2)
    return (pd.Series(slot(meta["target_time"])).map(rate)
            .fillna((train_event.sum() + 1) / (len(train_event) + 2)).to_numpy())


def rev(y: np.ndarray, probability: np.ndarray, climate: np.ndarray, ratio: float) -> float:
    model_cost = ratio * np.mean(probability > ratio) + np.mean(y & (probability <= ratio))
    climate_cost = ratio * np.mean(climate > ratio) + np.mean(y & (climate <= ratio))
    perfect_cost = ratio * np.mean(y)
    denom = climate_cost - perfect_cost
    return float((climate_cost - model_cost) / denom) if denom > 1e-12 else float("nan")
