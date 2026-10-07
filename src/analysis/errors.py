"""조건별 오차와 미탐(FN)·오경보(FP) 집계. 날짜 블록 부트스트랩 95% CI.

조건은 모두 대리변수다(가정 A6): 시각대(00–08, 08–16, 16–24)는 실제 교대 기록이 아니고,
생산량 구간은 목표 시간의 사후 관측 생산량이다(모델 입력이 아닌 분석용).
여러 조건을 동시에 보므로 다중 비교 보정이 없는 탐색적 결과다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def error_conditions(frame: pd.DataFrame, peak: float, alarm: np.ndarray, train_prod: pd.Series,
                     cfg: dict, prediction: str = "lgb", n: int = 1000, seed: int = 42) -> pd.DataFrame:
    f = frame.copy()
    t = f["target_time"].dt
    f["weekday"] = t.day_name()
    f["hour"] = t.hour
    f["shift_proxy"] = pd.cut(t.hour, cfg["conditions"]["shift_proxy_bins"],
                              labels=cfg["conditions"]["shift_proxy_labels"])
    f["season"] = t.month.map(SEASON)
    bins = np.unique(np.nanquantile(train_prod, cfg["conditions"]["production_quantiles"]))
    if len(bins) < 2:
        f["production_band"] = "one value"
    else:
        bins[0] -= 1e-9
        bins[-1] = np.inf
        f["production_band"] = pd.cut(f["production_target"], bins=bins, include_lowest=True).astype(str)
    event = f["y"] > peak
    alarm = pd.Series(alarm, index=f.index)
    f["abs_error"] = (f["y"] - f[prediction]).abs()
    f["residual"] = f["y"] - f[prediction]
    f["fn"] = event & ~alarm
    f["fp"] = ~event & alarm
    f["event"] = event
    f["alarm"] = alarm
    rows = []
    rng = np.random.default_rng(seed)
    for factor in ("weekday", "hour", "shift_proxy", "season", "production_band"):
        for value, g in f.groupby(factor, observed=True, dropna=False):
            row = {"factor": factor, "value": str(value), "n": len(g),
                   "mae": float(g["abs_error"].mean()), "mean_residual": float(g["residual"].mean()),
                   "events": int(g["event"].sum()), "fn": int(g["fn"].sum()), "fp": int(g["fp"].sum()),
                   "recall": float(1 - g["fn"].sum() / g["event"].sum()) if g["event"].sum() else np.nan,
                   "fp_rate": float(g["fp"].sum() / (~g["event"]).sum()) if (~g["event"]).sum() else np.nan}
            per_day = g.groupby("date").agg(n=("y", "size"), abs_error=("abs_error", "sum"),
                                            events=("event", "sum"), fn=("fn", "sum"), fp=("fp", "sum"))
            count = len(per_day)
            draw = rng.integers(0, count, size=(n, count))
            totals = {key: per_day[key].to_numpy()[draw].sum(axis=1) for key in per_day.columns}
            samples = {"mae": totals["abs_error"] / totals["n"],
                       "recall": np.divide(totals["events"] - totals["fn"], totals["events"],
                                           out=np.full(n, np.nan), where=totals["events"] > 0),
                       "fp_rate": np.divide(totals["fp"], totals["n"] - totals["events"],
                                            out=np.full(n, np.nan), where=(totals["n"] - totals["events"]) > 0)}
            for name, values in samples.items():
                valid = values[np.isfinite(values)]
                low, high = (np.quantile(valid, [.025, .975]) if len(valid) >= n // 2 else (np.nan, np.nan))
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def evening_miss_cases(frame: pd.DataFrame, peak: float, alarm_column: str, hours=(16, 24), top: int = 3) -> list[dict]:
    """16–24시 실제 피크 에피소드 중 경보가 한 번도 겹치지 않은 사례를 길이 순으로 고른다."""
    cases = []
    for date, g in frame.groupby("date", sort=True):
        g = g.sort_values("target_time")
        hour = g["target_time"].dt.hour.to_numpy()
        event = (g["y"].to_numpy() > peak) & (hour >= hours[0]) & (hour < hours[1])
        alarm = g[alarm_column].to_numpy(dtype=bool)
        start = None
        for i, flag in enumerate(np.r_[event, False]):
            if flag and start is None:
                start = i
            if not flag and start is not None:
                if not alarm[start:i].any():
                    cases.append({"date": str(date), "start": g["target_time"].iloc[start],
                                  "end": g["target_time"].iloc[i - 1], "positions": i - start,
                                  "max_excess": float(g["y"].iloc[start:i].max() - peak)})
                start = None
    cases.sort(key=lambda c: (-c["positions"], -c["max_excess"]))
    return cases[:top]
