"""피크 발생 조건 (출제 요구 3). 시험 구간 이전(학습+검증)의 관측값만 기술적으로 분석한다.

피크 = 15분 전력 > 학습 구간 목표값의 상위 5% 경계. 생산량·기온은 같은 시간의 사후 관측값이며
모델 입력이 아니라 조건 설명용이다. 시각대·요일·생산량 구간은 교대·설비 동시가동·제품의
대리변수다(가정 A6). 인과 효과가 아니며 다중 비교 보정이 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import runs

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def position_table(series: pd.DataFrame, raw: pd.DataFrame, end_time: pd.Timestamp, peak: float,
                   cfg: dict, train_end: pd.Timestamp) -> pd.DataFrame:
    """시험 이전 15분 위치 표. 위치의 시작 시각으로 시간대를 정한다."""
    s = series.loc[(series.index <= end_time) & ~series["recovered"] & series["power"].notna()].copy()
    start = s.index - pd.Timedelta(minutes=15)
    s["hour"] = start.hour
    s["weekday"] = start.dayofweek
    s["month"] = start.month
    s["date"] = start.date
    s["shift_proxy"] = pd.cut(s["hour"], cfg["conditions"]["shift_proxy_bins"],
                              labels=cfg["conditions"]["shift_proxy_labels"]).astype(str)
    s["peak"] = s["power"] > peak
    hourly = raw.loc[~raw["recovered"], ["date", "hour", "기온"]].copy()
    hourly["key"] = hourly["date"] + pd.to_timedelta(hourly["hour"], unit="h")
    temp = hourly.set_index("key")["기온"]
    s["temperature"] = temp.reindex(start.floor("h")).to_numpy()
    s["trend_1h"] = s["power"] - series["power"].shift(4).reindex(s.index)
    train_mask = s.index <= train_end
    prod_bins = np.unique(np.nanquantile(s.loc[train_mask, "production_target"],
                                         cfg["conditions"]["production_quantiles"]))
    prod_bins[0] -= 1e-9
    prod_bins[-1] = np.inf
    s["production_band"] = pd.cut(s["production_target"], prod_bins, include_lowest=True).astype(str)
    temp_bins = np.unique(np.nanquantile(s.loc[train_mask, "temperature"], [0, .25, .5, .75, 1]))
    temp_bins[0] -= 1e-9
    temp_bins[-1] = np.inf
    s["temperature_band"] = pd.cut(s["temperature"], temp_bins, include_lowest=True).astype(str)
    return s


def rate_by_condition(s: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    overall = s["peak"].mean()
    rng = np.random.default_rng(seed)
    rows = []
    for factor in ("hour", "weekday", "month", "shift_proxy", "production_band", "temperature_band"):
        for value, g in s.groupby(factor, observed=True):
            per_day = g.groupby("date")["peak"].agg(["sum", "size"])
            draw = rng.integers(0, len(per_day), size=(n, len(per_day)))
            rate = per_day["sum"].to_numpy()[draw].sum(axis=1) / per_day["size"].to_numpy()[draw].sum(axis=1)
            label = WEEKDAY_KO[int(value)] if factor == "weekday" else str(value)
            rows.append({"factor": factor, "value": label, "positions": len(g), "peaks": int(g["peak"].sum()),
                         "peak_rate": float(g["peak"].mean()),
                         "rate_ci_low": float(np.quantile(rate, .025)), "rate_ci_high": float(np.quantile(rate, .975)),
                         "lift": float(g["peak"].mean() / overall) if overall > 0 else np.nan,
                         "share_of_peaks": float(g["peak"].sum() / s["peak"].sum()) if s["peak"].sum() else np.nan})
    return pd.DataFrame(rows)


def interaction_table(s: pd.DataFrame) -> pd.DataFrame:
    """요일 × 시각대 피크율 (교차 조건)."""
    table = s.pivot_table(index="weekday", columns="shift_proxy", values="peak", aggfunc="mean", observed=True)
    table.index = [WEEKDAY_KO[int(i)] for i in table.index]
    return table


def peak_vs_nonpeak(s: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """피크 위치와 같은 시각대(08–16시) 비피크 위치의 평균 차이. 시간대 효과를 일부 통제한다."""
    rows = []
    rng = np.random.default_rng(seed)
    for scope, part in (("전체", s), ("08-16시 내부", s[s["shift_proxy"] == "08-16"])):
        for column, label in (("production_target", "같은 시간 생산량"), ("temperature", "기온"),
                              ("trend_1h", "직전 1시간 전력 변화")):
            per_day = part.groupby("date").apply(
                lambda g: pd.Series({"ps": g.loc[g["peak"], column].sum(), "pn": g["peak"].sum(),
                                     "ns": g.loc[~g["peak"], column].sum(), "nn": (~g["peak"]).sum()}),
                include_groups=False)
            values = per_day.to_numpy()
            point = values[:, 0].sum() / values[:, 1].sum() - values[:, 2].sum() / values[:, 3].sum()
            draw = rng.integers(0, len(values), size=(n, len(values)))
            sums = values[draw].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                diffs = sums[:, 0] / sums[:, 1] - sums[:, 2] / sums[:, 3]
            diffs = diffs[np.isfinite(diffs)]
            rows.append({"scope": scope, "variable": label,
                         "peak_mean": float(values[:, 0].sum() / values[:, 1].sum()),
                         "nonpeak_mean": float(values[:, 2].sum() / values[:, 3].sum()),
                         "difference": float(point),
                         "ci_low": float(np.quantile(diffs, .025)), "ci_high": float(np.quantile(diffs, .975))})
    return pd.DataFrame(rows)


def episode_table(s: pd.DataFrame) -> pd.DataFrame:
    """연속 초과 에피소드의 시작 시각·길이."""
    rows = []
    for date, g in s.groupby("date", sort=True):
        times = g.index.to_numpy(dtype="datetime64[ns]")
        for a, b in runs(g["peak"].to_numpy(), times):
            rows.append({"date": str(date), "start": str(g.index[a] - pd.Timedelta(minutes=15)),
                         "start_hour": int(g["hour"].iloc[a]), "weekday": WEEKDAY_KO[int(g["weekday"].iloc[a])],
                         "positions": b - a + 1, "max_power": float(g["power"].iloc[a:b + 1].max())})
    return pd.DataFrame(rows)
