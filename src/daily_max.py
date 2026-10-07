"""보조 과제(A1): 전날 23:45 기준 다음 달력날 최대 15분 전력. 사전 검증 daily_q1과 같은 계산."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .evaluate import block_bootstrap
from .models.lgbm_point import regression_model


def daily_max_task(cfg: dict, series: pd.DataFrame, raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    grouped = series["power"].resample("D").agg(["max", "mean", "count"])
    grouped["bad"] = series["recovered"].resample("D").max().fillna(True).astype(bool)
    # 하루 96개 15분 끝 시각이 모두 있는 날만 사용. 경계의 부분 날짜는 제외.
    good = (grouped["count"] == 96) & ~grouped["bad"]
    x = pd.DataFrame(index=grouped.index)
    x["current_day_max"] = grouped["max"]
    x["current_day_mean"] = grouped["mean"]
    x["prev_day_max"] = grouped["max"].shift(1)
    x["target_week_last_max"] = grouped["max"].shift(6)
    x["prev_week_mean"] = grouped["mean"].shift(7)
    next_day = x.index.shift(1, freq="D")
    x["target_weekday"] = next_day.dayofweek
    x["target_month"] = next_day.month
    x["target_day_of_year"] = next_day.dayofyear
    # 확실히 알려진 생산량은 직전 달력날 합계까지다.
    production_by_day = raw.groupby("date")["생산량"].sum()
    x["previous_day_production_total"] = production_by_day.reindex(x.index).shift(1)
    y = grouped["max"].shift(-1)
    eligible = (good & good.shift(-1, fill_value=False) & good.shift(1, fill_value=False)
                & good.shift(6, fill_value=False) & good.shift(7, fill_value=False))
    x, y = x.loc[eligible], y.loc[eligible]
    n = len(x)
    f1, f2 = cfg["daily_max"]["split_fractions"]
    a, b = int(f1 * n), int(f2 * n)
    model = regression_model(cfg)
    model.fit(x.iloc[:a], y.iloc[:a])
    test = pd.DataFrame(index=x.index[b:])
    test["y"] = y.iloc[b:]
    daily_naives = {"day": x["current_day_max"], "week": x["target_week_last_max"],
                    "average": (x["current_day_max"] + x["target_week_last_max"]) / 2}
    daily_choice = min(daily_naives, key=lambda name: mean_absolute_error(y.iloc[a:b], daily_naives[name].iloc[a:b]))
    test["naive"] = daily_naives[daily_choice].iloc[b:]
    test["lgb"] = model.predict(x.iloc[b:])
    test["date"] = test.index.date
    threshold = float(y.iloc[:a].quantile(cfg["daily_max"]["peak_quantile"]))

    def metrics(f):
        truth = f["y"].to_numpy()
        peak = truth > threshold
        out = {}
        for name in ("naive", "lgb"):
            err = truth - f[name].to_numpy()
            out[f"{name}_mae"] = float(np.mean(np.abs(err)))
            out[f"{name}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
            out[f"{name}_mape"] = float(np.mean(np.abs(err[truth != 0] / truth[truth != 0])) * 100)
            out[f"{name}_peak_mae"] = float(np.mean(np.abs(err[peak]))) if peak.any() else float("nan")
        return out

    point, ci = block_bootstrap(test, metrics, cfg["bootstrap_draws"], cfg["seed"])
    info = {"target": "next calendar-day max of 15-minute positions, forecast at previous day 23:45",
            "rows": {"train": a, "validation": b - a, "test": n - b},
            "range": {"train": [str(x.index[0].date()), str(x.index[a - 1].date())],
                      "validation": [str(x.index[a].date()), str(x.index[b - 1].date())],
                      "test": [str(x.index[b].date()), str(x.index[-1].date())]},
            "train_daily_max_95pct": threshold, "test_daily_peak_days": int((test["y"] > threshold).sum()),
            "naive_selected_on_validation": daily_choice, "metrics": point, "ci95": ci}
    return info, test
