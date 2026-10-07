"""생산량 시간대 이동 시뮬레이션 (가정 기반, 인과 효과 아님).

1) 학습 구간 시간 단위 자료로 전력 = a + b·생산량 + 시각 고정효과 + 요일 고정효과 를 최소제곱 추정한다.
   b는 같은 시각·요일 안에서 생산량이 한 단위 늘 때의 평균 전력 차이이며 인과 계수로 해석하지 않는다.
2) 시험 구간 각 날짜에서 '이동 대상 시간'의 생산량 비율 f를 같은 날 다른 가동 시간(생산량 > 0)으로
   옮긴다고 가정한다. 받는 시간은 계획 전력이 피크 경계의 95%를 넘지 않는 여유만큼만 받는다.
   각 시간의 4개 15분 위치 실측 전력에 b·Δ생산량을 더한다. 밤 시간대 신규 가동은 가정하지 않는다.
3) 이동 전후의 최대 15분 전력과 피크 경계 초과 위치 수를 비교한다.

이동 대상 시간을 정하는 세 방식:
- forecast: 1시간 후 점예측 경보가 하나라도 있는 시간. 받는 시간의 여유도 예측값으로 판단 (운영 가능 방식)
- forecast_q95: forecast와 같은 이동 대상. 받는 시간의 여유를 95% 분위수 예측으로 보수적으로 판단
- static: 학습 구간에서 피크율이 전체의 2배 이상인 시각. 여유는 예측값으로 판단 (고정 일정)
- oracle: 실제로 피크가 발생한 시간. 여유도 실측으로 판단 (사후 정보, 효과의 상한)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_frame(series: pd.DataFrame) -> pd.DataFrame:
    s = series.loc[~series["recovered"] & series["power"].notna()].copy()
    s["hour_start"] = (s.index - pd.Timedelta(minutes=15)).floor("h")
    g = s.groupby("hour_start").agg(power=("power", "mean"), production=("production_target", "first"),
                                    positions=("power", "size"))
    g = g[g["positions"] == 4]
    g["hour"] = g.index.hour
    g["weekday"] = g.index.dayofweek
    g["date"] = g.index.date
    return g


def _design(g: pd.DataFrame) -> np.ndarray:
    hour = pd.get_dummies(g["hour"].astype(pd.CategoricalDtype(range(24))), drop_first=True).to_numpy(float)
    wd = pd.get_dummies(g["weekday"].astype(pd.CategoricalDtype(range(7))), drop_first=True).to_numpy(float)
    return np.column_stack([np.ones(len(g)), g["production"].to_numpy(float), hour, wd])


def fit_slope(hourly_train: pd.DataFrame, n: int = 1000, seed: int = 42) -> dict:
    X, y = _design(hourly_train), hourly_train["power"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    days = np.array(sorted(set(hourly_train["date"])))
    index = {d: np.where(hourly_train["date"].to_numpy() == d)[0] for d in days}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        take = np.concatenate([index[d] for d in rng.choice(days, len(days))])
        draws.append(np.linalg.lstsq(X[take], y[take], rcond=None)[0][1])
    low, high = np.quantile(draws, [.025, .975])
    return {"slope": float(beta[1]), "ci_low": float(low), "ci_high": float(high), "hours": len(hourly_train)}


def simulate(series_test: pd.DataFrame, plan_power: pd.Series, shift_hours: dict, slope: float,
             fraction: float, peak: float, cap_ratio: float = 0.95) -> dict:
    """series_test: 시험 구간 15분 실측(power, production_target). plan_power: 계획에 쓰는 전력(예측 또는 실측).

    각 날짜에서 이동 대상 시간의 생산량 f 비율을 같은 날 다른 가동 시간으로 옮기되, 받는 시간의 계획 전력이
    피크 경계의 cap_ratio 배를 넘지 않는 만큼만 옮긴다(여유가 적은 시간부터 채우지 않고 여유가 큰 시간부터).
    전력 변화는 b × Δ생산량을 그 시간의 4개 15분 위치에 더해 실측 전력에 적용한다.
    """
    s = series_test.copy()
    s["hour_start"] = (s.index - pd.Timedelta(minutes=15)).floor("h")
    s["date"] = s["hour_start"].dt.date
    adjusted = s["power"].to_numpy(float).copy()
    production = s.groupby("hour_start")["production_target"].first()
    plan_max = plan_power.reindex(s.index).groupby(s["hour_start"].to_numpy()).max()
    desired_total = placed_total = 0.0
    days_with_sources = days_shifted = 0
    for date, hours in shift_hours.items():
        day_hours = production[production.index.date == date]
        sources = [h for h in hours if h in day_hours.index and day_hours[h] > 0]
        if not sources or slope <= 0:
            continue
        days_with_sources += 1
        sinks = [h for h in day_hours.index if h not in hours and day_hours[h] > 0]
        desired = {h: fraction * day_hours[h] for h in sources}
        want = sum(desired.values())
        capacity = {h: max(0.0, (cap_ratio * peak - plan_max.get(h, np.inf)) / slope) for h in sinks}
        room = sum(capacity.values())
        placed = min(want, room)
        desired_total += want
        if placed <= 0:
            continue
        days_shifted += 1
        placed_total += placed
        delta = {h: -d * placed / want for h, d in desired.items()}
        remaining = placed
        for h in sorted(capacity, key=capacity.get, reverse=True):
            take = min(capacity[h], remaining)
            if take <= 0:
                break
            delta[h] = delta.get(h, 0.0) + take
            remaining -= take
        for h, d in delta.items():
            adjusted[(s["hour_start"] == h).to_numpy()] += slope * d
    before, after = s["power"].to_numpy(float), adjusted
    months = s.index.to_period("M")
    monthly = {str(m): (float(before[months == m].max()), float(after[months == m].max())) for m in months.unique()}
    daily_before = pd.Series(before).groupby(s["date"].to_numpy()).max()
    daily_after = pd.Series(after).groupby(s["date"].to_numpy()).max()
    return {"fraction": fraction, "slope": slope, "desired_production": desired_total,
            "moved_production": placed_total,
            "placed_share": placed_total / desired_total if desired_total else np.nan,
            "days_with_sources": days_with_sources, "days_shifted": days_shifted,
            "max_before": float(np.nanmax(before)), "max_after": float(np.nanmax(after)),
            "exceed_before": int(np.sum(before > peak)), "exceed_after": int(np.sum(after > peak)),
            "daily_max_mean_before": float(daily_before.mean()), "daily_max_mean_after": float(daily_after.mean()),
            "daily_max_reduced_days": int(np.sum(daily_after < daily_before - 1e-9)),
            "monthly_max": monthly}


def shift_targets(frame: pd.DataFrame, peak: float, cutoff: float, static_hours: set[int]) -> dict[str, dict]:
    """세 방식의 날짜별 이동 대상 시간(hour_start) 집합."""
    f = frame.copy()
    f["hour_start"] = (f["target_time"] - pd.Timedelta(minutes=15)).dt.floor("h")
    f["date"] = f["hour_start"].dt.date
    out = {"forecast": {}, "oracle": {}, "static": {}}
    for date, g in f.groupby("date"):
        out["forecast"][date] = set(g.loc[g["lgb"] >= cutoff, "hour_start"])
        out["oracle"][date] = set(g.loc[g["y"] > peak, "hour_start"])
        out["static"][date] = set(h for h in g["hour_start"].unique() if h.hour in static_hours)
    return out
