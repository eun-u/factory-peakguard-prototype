"""예측 원점 t의 특징 생성. 모든 특징은 t 시점 이전에 관측된 값만 쓴다.

특징별 가용 시점 (원점 t, 예측거리 h, 목표 시각 t + h·15분):

| 특징 | 값의 시각 | 가용 시점 |
| --- | --- | --- |
| current | t에 끝난 15분 구간 전력 | t |
| lag_k | t − k·15분에 끝난 구간 전력 | t − k·15분 (< t) |
| recent_hour_mean | t−45분 ~ t에 끝난 4개 구간 평균 | t |
| recent_day_mean | t 이전 96개 구간 평균 | t |
| known_production | 마지막으로 끝난 시간의 생산량. HH시 생산량은 (HH+1):00에 관측(가정 A8) | t |
| known_production_prev_day | 위 값의 하루 전 | t − 1일 |
| target_hour, target_quarter, target_weekday, target_month, target_day_of_year, target_is_weekend | 목표 시각의 달력 | 달력은 미리 알 수 있음 |

입력에서 제외: 같은 시간 평균 전력(목표 구간 포함), 목표 시간의 생산량, 실측 기상, 공장인원.
"""

from __future__ import annotations

import pandas as pd


def lag_set(base_lags: list[int], horizon: int) -> tuple[int, ...]:
    """1시간(h=4) 설정의 시차 [1,2,4,8,92,96,668,672]는 전일·전주 같은 시각(96−h, 672−h)을 포함한다.
    다른 예측거리에서는 목표 시각 기준 전일·전주 시차를 같은 규칙으로 바꾼다."""
    if horizon == 4:
        return tuple(base_lags)
    lags = {1, 2, 4, 8, 96, 672, 672 - horizon}
    if 96 - horizon > 0:
        lags.add(96 - horizon)
    return tuple(sorted(lags))


def seasonal_columns(horizon: int) -> dict[str, str]:
    """목표 시각과 같은 15분 위치의 전일·전주 값이 들어 있는 열."""
    day = 96 - horizon
    return {"day": "current" if day == 0 else f"lag_{day}", "week": f"lag_{672 - horizon}"}


def feature_frame(series: pd.DataFrame, horizon: int = 4,
                  base_lags: list[int] = (1, 2, 4, 8, 92, 96, 668, 672)) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """h=4이면 사전 검증 t5_power.feature_frame과 같은 행·열·값을 만든다."""
    p = series["power"]
    lags = lag_set(list(base_lags), horizon)
    target_time = series.index.shift(horizon, freq="15min")
    x = pd.DataFrame(index=series.index)
    x["current"] = p
    for lag in lags:
        x[f"lag_{lag}"] = p.shift(lag)
    x["recent_hour_mean"] = p.rolling(4, min_periods=4).mean()
    x["recent_day_mean"] = p.rolling(96, min_periods=96).mean()
    x["known_production"] = series["production_known"]
    x["known_production_prev_day"] = series["production_known"].shift(96)
    x["target_hour"] = target_time.hour
    x["target_quarter"] = target_time.minute // 15
    x["target_weekday"] = target_time.dayofweek
    x["target_month"] = target_time.month
    x["target_day_of_year"] = target_time.dayofyear
    x["target_is_weekend"] = (target_time.dayofweek >= 5).astype(int)
    y = p.shift(-horizon)
    bad = series["recovered"] | p.isna() | series["production_known_bad"]
    eligible = y.notna() & ~series["recovered"].shift(-horizon, fill_value=True)
    for lag in sorted({0, 1, 2, 3} | set(lags)):
        eligible &= ~bad.shift(lag, fill_value=True)
    meta = pd.DataFrame(index=series.index)
    meta["target_time"] = target_time
    meta["production_target"] = series["production_target"].shift(-horizon)
    meta["date"] = target_time.date
    eligible &= x.notna().all(axis=1)
    return x.loc[eligible], y.loc[eligible], meta.loc[eligible]
