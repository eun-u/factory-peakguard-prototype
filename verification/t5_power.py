"""Task ⑤, reproducible pre-feasibility checks on the supplied power CSV.

Run from anywhere: ``python verification/t5_power.py``. All fitted quantities use
the training split; only the F1 decision cutoff uses validation. The test split
is read once for evaluation. A timestamp reconstructed from row order is marked
provisional and excluded from fitting and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    precision_recall_curve,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
SEED = 42
BOOT = 1000
MINUTES = (15, 30, 45, 60)
Q = (0.1, 0.5, 0.9, 0.95)


def locate_source() -> Path:
    """Identify the power CSV by schema within the canonical task ⑤ raw folder."""
    source_dir = ROOT / "data" / "raw" / "task05_power"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Task ⑤ raw data directory is missing: {source_dir}")
    matches = []
    for path in source_dir.rglob("*.csv"):
        try:
            columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        except (UnicodeError, pd.errors.ParserError):
            continue
        if {"날짜", "시간", "15분", "30분", "45분", "60분"}.issubset(columns):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one task ⑤ CSV with date/hour/four power columns, found {len(matches)}"
        )
    return matches[0]


def load_series() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = locate_source()
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d", errors="raise")
    raw["source_hour"] = pd.to_numeric(raw["시간"], errors="coerce")
    raw["hour"] = -1
    raw["recovered"] = False
    for _, rows in raw.groupby("date", sort=False):
        expected = np.arange(24)
        given = rows["source_hour"].to_numpy()
        good = np.isfinite(given) & (given >= 0) & (given <= 23)
        if len(rows) != 24 or not np.array_equal(given[good], expected[good]):
            raise ValueError("Hour recovery requires exactly 24 ordered rows per day")
        raw.loc[rows.index, "hour"] = expected
        raw.loc[rows.index, "recovered"] = ~good

    chunks = []
    for minute in MINUTES:
        chunks.append(pd.DataFrame({
            "timestamp": raw["date"] + pd.to_timedelta(raw["hour"] * 60 + minute, unit="m"),
            "power": pd.to_numeric(raw[f"{minute}분"], errors="coerce"),
            "recovered": raw["recovered"],
            "production_target": raw["생산량"],
        }))
    series = pd.concat(chunks, ignore_index=True).set_index("timestamp").sort_index()
    if series.index.has_duplicates:
        raise ValueError("Duplicate 15-minute timestamp after provisional recovery")
    full = pd.date_range(series.index.min(), series.index.max(), freq="15min", name="timestamp")
    series = series.reindex(full)
    series.loc[series["power"] < 0, "power"] = np.nan
    series["recovered"] = series["recovered"].fillna(False).astype(bool)

    # Production for hour HH is treated as observed only at (HH+1):00.
    hourly = raw[["date", "hour", "생산량", "recovered"]].copy()
    hourly.index = hourly["date"] + pd.to_timedelta(hourly["hour"] + 1, unit="h")
    hourly = hourly.sort_index()
    series["production_known"] = hourly["생산량"].reindex(series.index).ffill()
    series["production_known_bad"] = hourly["recovered"].reindex(series.index).astype("boolean").ffill().fillna(True).astype(bool)
    series["production_target"] = series["production_target"].astype(float)

    all_power = raw[[f"{m}분" for m in MINUTES]].to_numpy(dtype=float).ravel()
    q1, q3 = np.nanquantile(all_power, [0.25, 0.75])
    complete_summer = any(series.index.min() <= pd.Timestamp(f"{year}-07-01") and
                          series.index.max() >= pd.Timestamp(f"{year}-09-30 23:45")
                          for year in range(series.index.min().year, series.index.max().year + 1))
    complete_winter = any(series.index.min() <= pd.Timestamp(f"{year}-12-01") and
                          series.index.max() >= pd.Timestamp(f"{year+1}-02-28 23:45")
                          for year in range(series.index.min().year - 1, series.index.max().year + 1))
    info = {
        "source": str(path.relative_to(ROOT)), "source_rows": len(raw),
        "source_columns": list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns),
        "15min_rows": len(series), "start": str(series.index.min()), "end": str(series.index.max()),
        "missing_15min_power": int(series["power"].isna().sum()),
        "negative_raw_power": int((all_power < 0).sum()), "zero_power": int((all_power == 0).sum()),
        "iqr_outliers": int(((all_power < q1 - 1.5 * (q3 - q1)) | (all_power > q3 + 1.5 * (q3 - q1))).sum()),
        "invalid_hours": int(raw["recovered"].sum()),
        "recovered_dates": [str(d.date()) for d in raw.loc[raw["recovered"], "date"].unique()],
        "recovered_15min_rows_excluded": int(series["recovered"].sum()),
        "months": sorted(set(series.index.month.tolist())),
        "summer_present": bool(set(series.index.month) & {7, 8, 9}),
        "winter_present": bool(set(series.index.month) & {12, 1, 2}),
        "summer_month_count": len(set(series.index.month) & {7, 8, 9}),
        "winter_month_count": len(set(series.index.month) & {12, 1, 2}),
        "complete_summer_window": bool(complete_summer),
        "complete_winter_window": bool(complete_winter),
        "production_missing": int(raw["생산량"].isna().sum()),
        "source_missing_by_column": {col: int(n) for col, n in raw.isna().sum().items() if n and col in raw.columns[:18]},
        "weather_columns": ["기온", "풍속", "습도", "강수량"],
        "other_source_columns": ["전기요금(계절)", "day", "d", "m", "공장인원", "인건비"],
        "raw_average_matches_rounded_four_values_fraction": float(np.mean(
            np.abs(raw["평균"].to_numpy() - raw[[f"{m}분" for m in MINUTES]].mean(axis=1).to_numpy()) <= 0.5)),
    }
    return series, raw, info


def feature_frame(series: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """One-hour-ahead 15-minute target; features at origin t use values at/before t."""
    p = series["power"]
    target_time = series.index.shift(4, freq="15min")
    x = pd.DataFrame(index=series.index)
    x["current"] = p
    for lag in (1, 2, 4, 8, 92, 96, 668, 672):
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
    y = p.shift(-4)
    bad = series["recovered"] | p.isna() | series["production_known_bad"]
    eligible = y.notna() & ~series["recovered"].shift(-4, fill_value=True)
    for lag in (0, 1, 2, 3, 4, 8, 92, 96, 668, 672):
        eligible &= ~bad.shift(lag, fill_value=True)
    meta = pd.DataFrame(index=series.index)
    meta["target_time"] = target_time
    meta["production_target"] = series["production_target"].shift(-4)
    meta["date"] = target_time.date
    eligible &= x.notna().all(axis=1)
    return x.loc[eligible], y.loc[eligible], meta.loc[eligible]


def split_time(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    # Purge four origins: last fit/calibration label then precedes next origin.
    n = len(x)
    cut1, cut2 = int(n * 0.70), int(n * 0.85)
    sl = {"train": slice(0, cut1-4), "validation": slice(cut1, cut2-4), "test": slice(cut2, n)}
    return {k: (x.iloc[s], y.iloc[s], meta.iloc[s]) for k, s in sl.items()}


def regression_model(objective: str = "regression", alpha: float | None = None):
    kw = dict(n_estimators=180, learning_rate=0.05, num_leaves=15, min_child_samples=35,
              colsample_bytree=0.9, reg_lambda=1.0, max_depth=6, random_state=SEED,
              n_jobs=4, verbosity=-1)
    if alpha is not None:
        kw.update(objective="quantile", alpha=alpha)
    else:
        kw.update(objective=objective)
    return lgb.LGBMRegressor(**kw)


def naive_predict(x: pd.DataFrame, version: str) -> np.ndarray:
    day = x["lag_92"].to_numpy()
    week = x["lag_668"].to_numpy()
    return {"day": day, "week": week, "average": (day + week) / 2}[version]


def quantile_event_probability(preds: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    crossings = int(np.sum(np.any(np.diff(preds, axis=1) < 0, axis=1)))
    p = np.sort(preds, axis=1)  # monotone rearrangement, set before test inspection
    cdf = np.full(len(p), 0.95)
    cdf[threshold <= p[:, 0]] = 0.10
    for j in range(3):
        mask = (threshold > p[:, j]) & (threshold <= p[:, j + 1])
        width = np.maximum(p[mask, j + 1] - p[mask, j], 1e-9)
        cdf[mask] = Q[j] + (Q[j + 1] - Q[j]) * (threshold - p[mask, j]) / width
    return np.clip(1.0 - cdf, 0, 1), crossings


def val_f1_cutoff(y: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def climatology(train_meta: pd.DataFrame, train_event: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    tr = pd.DataFrame({"slot": train_meta["target_time"].dt.dayofweek.to_numpy() * 96
                       + train_meta["target_time"].dt.hour.to_numpy() * 4
                       + train_meta["target_time"].dt.minute.to_numpy() // 15,
                       "event": train_event})
    grouped = tr.groupby("slot")["event"].agg(["sum", "count"])
    rate = (grouped["sum"] + 1) / (grouped["count"] + 2)  # Laplace smoothing
    slot = (meta["target_time"].dt.dayofweek.to_numpy() * 96
            + meta["target_time"].dt.hour.to_numpy() * 4
            + meta["target_time"].dt.minute.to_numpy() // 15)
    return pd.Series(slot).map(rate).fillna((train_event.sum() + 1) / (len(train_event) + 2)).to_numpy()


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan")


def rev(y: np.ndarray, probability: np.ndarray, climate: np.ndarray, ratio: float) -> float:
    model_cost = ratio * np.mean(probability > ratio) + np.mean(y & (probability <= ratio))
    climate_cost = ratio * np.mean(climate > ratio) + np.mean(y & (climate <= ratio))
    perfect_cost = ratio * np.mean(y)
    denom = climate_cost - perfect_cost
    return float((climate_cost - model_cost) / denom) if denom > 1e-12 else float("nan")


def metric_bundle(frame: pd.DataFrame, peak: float, cutoffs: dict[str, float], ratios: np.ndarray) -> dict:
    y = frame["y"].to_numpy()
    event = y > peak
    out = {}
    for model in ("day", "week", "average", "naive", "lgb"):
        pred = frame[model].to_numpy()
        err = y - pred
        out[f"{model}_mae"] = float(np.mean(np.abs(err)))
        out[f"{model}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        nz = y != 0
        out[f"{model}_mape"] = float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else float("nan")
        out[f"{model}_peak_mae"] = float(np.mean(np.abs(err[event]))) if event.any() else float("nan")
    out["peak_mae_improvement_fraction"] = (float(1 - out["lgb_peak_mae"] / out["naive_peak_mae"])
                                             if out["naive_peak_mae"] > 0 else float("nan"))
    for q in Q:
        name = f"q{int(q * 100)}"
        pred = frame[name].to_numpy()
        out[f"{name}_pinball"] = float(mean_pinball_loss(y, pred, alpha=q))
        out[f"{name}_coverage"] = float(np.mean(y <= pred))
    out["mean_pinball"] = float(np.mean([out[f"q{int(q * 100)}_pinball"] for q in Q]))
    for model in ("classifier", "quantile"):
        prob = frame[f"{model}_prob"].to_numpy()
        out[f"{model}_f1"] = float(f1_score(event, prob >= cutoffs[model], zero_division=0))
        out[f"{model}_prauc"] = safe_ap(event, prob)
        out[f"{model}_brier"] = float(brier_score_loss(event, prob))
        action = prob >= cutoffs[model]
        out[f"{model}_precision"] = float(np.sum(action & event) / np.sum(action)) if action.any() else float("nan")
        out[f"{model}_recall"] = float(np.sum(action & event) / np.sum(event)) if event.any() else float("nan")
    for i, ratio in enumerate(ratios):
        out[f"rev_{i}"] = rev(event, frame["classifier_prob"].to_numpy(), frame["climate_prob"].to_numpy(), ratio)
    return out


def block_bootstrap(frame: pd.DataFrame, fn, n: int = BOOT) -> tuple[dict, dict]:
    point = fn(frame)
    groups = [np.where(frame["date"].to_numpy() == day)[0] for day in pd.unique(frame["date"])]
    rng = np.random.default_rng(SEED)
    values = {k: [] for k in point}
    for _ in range(n):
        sampled = rng.integers(0, len(groups), len(groups))
        take = np.concatenate([groups[i] for i in sampled])
        result = fn(frame.iloc[take])
        for key, value in result.items():
            values[key].append(value)
    ci = {}
    for key, arr in values.items():
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        ci[key] = ([float(v) for v in np.quantile(finite, [0.025, 0.975])]
                   if len(finite) >= n // 2 else [None, None])
    return point, ci


def daily_q1(series: pd.DataFrame, raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """At day D 23:45 predict D+1 calendar-day maximum 15-min value."""
    grouped = series["power"].resample("D").agg(["max", "mean", "count"])
    grouped["bad"] = series["recovered"].resample("D").max().fillna(True).astype(bool)
    # A full calendar day has 96 quarter-hour end timestamps; partial boundary days excluded.
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
    # Only the previous calendar day's completed production is certainly known.
    production_by_day = raw.groupby("date")["생산량"].sum()
    x["previous_day_production_total"] = production_by_day.reindex(x.index).shift(1)
    y = grouped["max"].shift(-1)
    eligible = (good & good.shift(-1, fill_value=False) & good.shift(1, fill_value=False)
                & good.shift(6, fill_value=False) & good.shift(7, fill_value=False))
    x, y = x.loc[eligible], y.loc[eligible]
    n = len(x)
    a, b = int(0.70 * n), int(0.85 * n)
    model = regression_model()
    model.fit(x.iloc[:a], y.iloc[:a])
    test = pd.DataFrame(index=x.index[b:])
    test["y"] = y.iloc[b:]
    daily_naives = {"day": x["current_day_max"], "week": x["target_week_last_max"],
                    "average": (x["current_day_max"] + x["target_week_last_max"]) / 2}
    daily_choice = min(daily_naives, key=lambda name: mean_absolute_error(y.iloc[a:b], daily_naives[name].iloc[a:b]))
    test["naive"] = daily_naives[daily_choice].iloc[b:]
    test["lgb"] = model.predict(x.iloc[b:])
    test["date"] = test.index.date
    threshold = float(y.iloc[:a].quantile(0.95))

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

    point, ci = block_bootstrap(test, metrics)
    info = {"target": "next calendar-day max of 15-minute positions, forecast at previous day 23:45",
            "rows": {"train": a, "validation": b-a, "test": n-b},
            "range": {"train": [str(x.index[0].date()), str(x.index[a-1].date())],
                      "validation": [str(x.index[a].date()), str(x.index[b-1].date())],
                      "test": [str(x.index[b].date()), str(x.index[-1].date())]},
            "train_daily_max_95pct": threshold, "test_daily_peak_days": int((test["y"] > threshold).sum()),
            "naive_selected_on_validation": daily_choice,
            "metrics": point, "ci95": ci}
    return info, test


def rolling_origin(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> list[dict]:
    """Three strictly earlier train-prefix checks, all before the final 15% holdout."""
    rows = []
    n = len(x)
    for fold, fraction in enumerate((.30, .45, .60), start=1):
        boundary = int(n * fraction)
        end = int(n * (fraction + .10))
        train_x, train_y = x.iloc[:boundary-4], y.iloc[:boundary-4]
        fold_x, fold_y, fold_meta = x.iloc[boundary:end], y.iloc[boundary:end], meta.iloc[boundary:end]
        if meta.iloc[boundary-5]["target_time"] >= fold_x.index.min():
            raise AssertionError("Rolling-origin training label overlaps evaluation origin")
        model = regression_model().fit(train_x, train_y)
        peak = float(train_y.quantile(.95))
        frame = pd.DataFrame({"y": fold_y, "date": fold_meta["date"],
                              "day": naive_predict(fold_x, "day"),
                              "week": naive_predict(fold_x, "week"),
                              "average": naive_predict(fold_x, "average"),
                              "lgb": model.predict(fold_x)})

        def metrics(f):
            actual = f["y"].to_numpy()
            event = actual > peak
            answer = {}
            for name in ("day", "week", "average", "lgb"):
                e = np.abs(actual - f[name].to_numpy())
                answer[f"{name}_mae"] = float(np.mean(e))
                answer[f"{name}_peak_mae"] = float(np.mean(e[event])) if event.any() else float("nan")
            return answer

        point, ci = block_bootstrap(frame, metrics)
        for name in ("day", "week", "average", "lgb"):
            rows.append({"fold": fold, "model": name, "training_rows": len(train_x),
                         "evaluation_rows": len(fold_x), "evaluation_first": str(fold_x.index.min()),
                         "evaluation_last": str(fold_x.index.max()), "peak_threshold": peak,
                         "peak_events": int((fold_y > peak).sum()),
                         "mae": point[f"{name}_mae"], "mae_ci_low": ci[f"{name}_mae"][0],
                         "mae_ci_high": ci[f"{name}_mae"][1],
                         "peak_mae": point[f"{name}_peak_mae"],
                         "peak_mae_ci_low": ci[f"{name}_peak_mae"][0],
                         "peak_mae_ci_high": ci[f"{name}_peak_mae"][1]})
    return rows


def error_conditions(frame: pd.DataFrame, peak: float, cutoff: float, train_prod: pd.Series) -> pd.DataFrame:
    f = frame.copy()
    t = f["target_time"].dt
    f["weekday"] = t.day_name()
    f["hour"] = t.hour
    f["shift_proxy"] = pd.cut(t.hour, [-1, 7, 15, 23], labels=["00-08", "08-16", "16-24"])
    f["season"] = t.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
                                 6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
    bins = np.unique(np.nanquantile(train_prod, [0, .5, .75, 1]))
    if len(bins) < 2:
        f["production_band"] = "one value"
    else:
        bins[0] -= 1e-9
        bins[-1] = np.inf
        f["production_band"] = pd.cut(f["production_target"], bins=bins, include_lowest=True).astype(str)
    event = f["y"] > peak
    alarm = f["classifier_prob"] >= cutoff
    f["abs_error"] = (f["y"] - f["lgb"]).abs()
    f["residual"] = f["y"] - f["lgb"]
    f["fn"] = event & ~alarm
    f["fp"] = ~event & alarm
    f["event"] = event
    f["alarm"] = alarm
    rows = []
    rng = np.random.default_rng(SEED)
    for factor in ("weekday", "hour", "shift_proxy", "season", "production_band"):
        for value, g in f.groupby(factor, observed=True, dropna=False):
            row = {"factor": factor, "value": str(value), "n": len(g),
                         "mae": float(g["abs_error"].mean()), "mean_residual": float(g["residual"].mean()),
                         "events": int(g["event"].sum()), "fn": int(g["fn"].sum()), "fp": int(g["fp"].sum()),
                         "recall": float(1-g["fn"].sum()/g["event"].sum()) if g["event"].sum() else np.nan,
                         "fp_rate": float(g["fp"].sum()/(~g["event"]).sum()) if (~g["event"]).sum() else np.nan}
            per_day = g.groupby("date").agg(n=("y", "size"), abs_error=("abs_error", "sum"),
                                             events=("event", "sum"), fn=("fn", "sum"), fp=("fp", "sum"))
            count = len(per_day)
            draw = rng.integers(0, count, size=(BOOT, count))
            totals = {key: per_day[key].to_numpy()[draw].sum(axis=1) for key in per_day.columns}
            samples = {"mae": totals["abs_error"] / totals["n"],
                       "recall": np.divide(totals["events"] - totals["fn"], totals["events"],
                                           out=np.full(BOOT, np.nan), where=totals["events"] > 0),
                       "fp_rate": np.divide(totals["fp"], totals["n"] - totals["events"],
                                            out=np.full(BOOT, np.nan), where=(totals["n"] - totals["events"]) > 0)}
            for name, values in samples.items():
                valid = values[np.isfinite(values)]
                low, high = (np.quantile(valid, [.025, .975]) if len(valid) >= BOOT//2 else (np.nan, np.nan))
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(frame: pd.DataFrame, ratios: np.ndarray, point: dict, ci: dict, peak: float) -> None:
    plt.figure(figsize=(10, 3.5))
    sample = frame.iloc[:min(96 * 7, len(frame))]
    for col, label in (("y", "observed"), ("naive", "day/week naive"), ("lgb", "LightGBM")):
        plt.plot(sample["target_time"], sample[col], label=label, linewidth=1)
    plt.axhline(peak, color="black", linestyle="--", linewidth=0.8, label="train p95")
    plt.legend(ncol=4, fontsize=8); plt.ylabel("Source power value (unit unknown)")
    plt.tight_layout(); plt.savefig(OUT / "05_forecasts.png", dpi=140); plt.close()

    plt.figure(figsize=(5, 4))
    observed = frame["y"].to_numpy() > peak
    for name in ("classifier", "quantile", "climate"):
        p = frame[f"{name}_prob"].to_numpy()
        edges = np.linspace(0, 1, 11)
        bins = np.digitize(p, edges[1:-1])
        means = [(p[bins == i].mean(), observed[bins == i].mean(), (bins == i).sum())
                 for i in range(10) if (bins == i).any()]
        plt.plot([x[0] for x in means], [x[1] for x in means], marker="o", label=name)
    plt.plot([0,1],[0,1], color="black", linestyle="--", linewidth=.8)
    plt.xlabel("Predicted exceedance probability"); plt.ylabel("Observed event rate")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "05_reliability.png", dpi=140); plt.close()

    revs = [point[f"rev_{i}"] for i in range(len(ratios))]
    lo = [ci[f"rev_{i}"][0] for i in range(len(ratios))]
    hi = [ci[f"rev_{i}"][1] for i in range(len(ratios))]
    plt.figure(figsize=(6, 4)); plt.plot(ratios, revs, label="LightGBM classifier vs slot climatology")
    plt.fill_between(ratios, lo, hi, alpha=.2); plt.axhline(0, color="black", linewidth=.8)
    plt.xlabel("Assumed action cost / missed-peak loss, C/L"); plt.ylabel("Relative economic value")
    plt.tight_layout(); plt.savefig(OUT / "05_rev.png", dpi=140); plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series, raw, structure = load_series()
    x, y, meta = feature_frame(series)
    split = split_time(x, y, meta)
    xt, yt, mt = split["train"]
    xv, yv, mv = split["validation"]
    xs, ys, ms = split["test"]
    if not (mt["target_time"].max() < xv.index.min() and mv["target_time"].max() < xs.index.min()):
        raise AssertionError("A fitted/calibrated label overlaps a later split's forecast origin")
    peak = float(yt.quantile(.95))
    model = regression_model(); model.fit(xt, yt)
    quantiles = {}
    for q in Q:
        quantiles[q] = regression_model(alpha=q).fit(xt, yt)
    classifier = lgb.LGBMClassifier(n_estimators=180, learning_rate=.05, num_leaves=15,
                                   min_child_samples=35, colsample_bytree=.9, reg_lambda=1,
                                   max_depth=6, random_state=SEED, n_jobs=4, verbosity=-1)
    classifier.fit(xt, yt > peak)

    val_q = np.column_stack([quantiles[q].predict(xv) for q in Q])
    test_q = np.column_stack([quantiles[q].predict(xs) for q in Q])
    val_qprob, val_cross = quantile_event_probability(val_q, peak)
    test_qprob, test_cross = quantile_event_probability(test_q, peak)
    val_cprob = classifier.predict_proba(xv)[:, 1]
    test_cprob = classifier.predict_proba(xs)[:, 1]
    cutoffs = {"classifier": val_f1_cutoff((yv > peak).to_numpy(), val_cprob),
               "quantile": val_f1_cutoff((yv > peak).to_numpy(), val_qprob)}

    frame = pd.DataFrame(index=xs.index)
    frame["y"] = ys
    for version in ("day", "week", "average"):
        frame[version] = naive_predict(xs, version)
    val_naive_mae = {version: mean_absolute_error(yv, naive_predict(xv, version))
                     for version in ("day", "week", "average")}
    naive_choice = min(val_naive_mae, key=val_naive_mae.get)
    frame["naive"] = frame[naive_choice]
    frame["lgb"] = model.predict(xs)
    for i, q in enumerate(Q):
        frame[f"q{int(q * 100)}"] = test_q[:, i]
    frame["classifier_prob"] = test_cprob
    frame["quantile_prob"] = test_qprob
    frame["climate_prob"] = climatology(mt, (yt > peak).to_numpy(), ms)
    frame["target_time"] = ms["target_time"]
    frame["date"] = ms["date"]
    frame["production_target"] = ms["production_target"]
    ratios = np.r_[.01, np.arange(.05, .51, .05)]
    point, ci = block_bootstrap(frame, lambda f: metric_bundle(f, peak, cutoffs, ratios))
    daily, daily_test = daily_q1(series, raw)
    rolling = rolling_origin(x, y, meta)
    pd.DataFrame(rolling).to_csv(OUT / "05_rolling_origin.csv", index=False, encoding="utf-8-sig")
    conditions = error_conditions(frame, peak, cutoffs["classifier"], mt["production_target"])
    conditions.to_csv(OUT / "05_error_conditions.csv", index=False, encoding="utf-8-sig")
    plot_outputs(frame, ratios, point, ci, peak)
    frame.drop(columns=["date"]).to_csv(OUT / "05_test_predictions.csv", index_label="forecast_origin", encoding="utf-8-sig")
    daily_test.drop(columns=["date"]).to_csv(OUT / "05_daily_test_predictions.csv", index_label="forecast_origin_day", encoding="utf-8-sig")

    summary = {
        "structure": structure,
        "assumptions": [
            "15/30/45/60분 열을 HH:15/30/45/(HH+1):00에 끝나는 연속 15분 위치로 간주한다. 제공 문서에서 경계 의미와 kW/kWh 단위를 확인하지 못했다.",
            "잘못된 시간 값은 해당 날짜의 행 순서대로 00~23시를 임시 부여한다. 영향 받은 위치와 이들에 의존하는 특징 행은 학습·평가에서 제외한다.",
            "시간당 생산량은 해당 시간이 끝난 뒤에만 알려진다고 가정한다. 익일 예측에는 전전날이 아닌 직전 달력날의 완료 생산량 합계를 사용한다. 미래의 실제 생산량·날씨·인력·요금과 같은 시간 평균 전력은 입력에서 제외한다.",
            "계절 나이브는 예측 대상과 같은 15분 시각의 전일값, 전주값, 두 값의 평균을 후보로 두고 검증 MAE 최저 후보를 시험 전에 고정한다.",
            "00~08, 08~16, 16~24시는 시계 기반 교대 대용 구간이며 실제 교대기록은 아니다. 제품 ID가 없어 제품 전환 조건은 분석할 수 없다.",
            "피크는 학습 전력값의 95백분위 초과로 정의하며 계약수요·요금 초과가 아니다. C/L은 가상 비용비이고 조치가 비용 C를 들여 피크 손실을 회피한다고 가정한다.",
        ],
        "q0": {"15min_reshape_possible_under_assumption": True,
               "k5a": "triggered" if not (structure["complete_summer_window"] and structure["complete_winter_window"]) else "not_triggered_under_time_alignment_assumption",
               "power_unit_confirmed": False, "interval_boundary_confirmed": False,
               "source_coverage": "2021 Jan-Sep 14 only; partial summer and Jan-Feb winter months, no complete Jul-Sep or Dec-Feb window"},
        "one_hour": {
            "target": "15-minute value at origin + 1 hour",
            "rows": {name: len(item[0]) for name, item in split.items()},
            "range": {name: [str(item[0].index[0]), str(item[0].index[-1])] for name, item in split.items()},
            "train_peak_95pct": peak,
            "events": {name: int((item[1] > peak).sum()) for name, item in split.items()},
            "test_peak_days": int(pd.Series(ms.loc[ys > peak, "date"]).nunique()),
            "test_peak_episodes": int(np.sum((ys > peak).to_numpy() &
                (np.r_[True, ~(ys > peak).to_numpy()[:-1]] |
                 np.r_[True, np.diff(ms["target_time"].to_numpy()).astype("timedelta64[m]") != np.timedelta64(15, "m")]))),
            "f1_cutoff_from_validation": cutoffs,
            "naive_validation_mae": val_naive_mae,
            "naive_selected_on_validation": naive_choice,
            "quantile_crossings": {"validation": val_cross, "test": test_cross},
            "metrics": point, "ci95": ci,
            "k5b": "triggered" if not np.isfinite(point["peak_mae_improvement_fraction"]) or point["peak_mae_improvement_fraction"] < .10 else "not_triggered",
            "k5c": "triggered" if int((ys > peak).sum()) < 30 else "not_triggered",
            "rev_cost_ratios": ratios.tolist(),
        },
        "daily": daily,
        "rolling_origin": rolling,
        "q5_unavailable": ["actual_shift", "product_id", "product_transition"],
        "bootstrap": "1000 paired calendar-day block resamples of the held-out test period; percentile 95% CI. Does not address one-season external validity.",
        "q4_cost_formula": "r=C/L일 때 초과확률 > r이면 조치. L로 나눈 실현 비용은 r×조치 + 실제 초과×(1-조치), 완전예측 비용은 r×실제 초과. REV=(동시각 기후평균 비용-모델 비용)/(동시각 기후평균 비용-완전예측 비용); 분모가 0 이하면 정의하지 않는다. 기후평균은 학습구간의 같은 요일·15분 위치 초과 빈도를 라플라스 평활한 값이다.",
    }
    summary["criteria"] = {
        "K5-a": "해당" if summary["q0"]["k5a"] == "triggered" else "비해당",
        "K5-b": "해당" if summary["one_hour"]["k5b"] == "triggered" else "비해당",
        "K5-c": "해당" if summary["one_hour"]["k5c"] == "triggered" else "비해당",
    }
    summary["headline"] = (
        f'1h peak MAE naive {point["naive_peak_mae"]:.2f} '
        f'[{ci["naive_peak_mae"][0]:.2f}, {ci["naive_peak_mae"][1]:.2f}] vs '
        f'LightGBM {point["lgb_peak_mae"]:.2f} '
        f'[{ci["lgb_peak_mae"][0]:.2f}, {ci["lgb_peak_mae"][1]:.2f}]; '
        f'peak classifier F1 {point["classifier_f1"]:.3f} '
        f'[{ci["classifier_f1"][0]:.3f}, {ci["classifier_f1"][1]:.3f}]; '
        f'test peak events {int((ys > peak).sum())}'
    )
    (OUT / "05_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    write_section(summary, conditions)
    print(json.dumps({"one_hour_test_rows": len(xs), "test_events": int((ys > peak).sum()),
                      "k5a": summary["q0"]["k5a"], "k5b": summary["one_hour"]["k5b"],
                      "k5c": summary["one_hour"]["k5c"]}, ensure_ascii=False))


def write_section(s: dict, conditions: pd.DataFrame) -> None:
    h = s["one_hour"]; m = h["metrics"]; ci = h["ci95"]
    metric = lambda key: f'{m[key]:.3f} [{ci[key][0]:.3f}, {ci[key][1]:.3f}]' if ci[key][0] is not None else f'{m[key]:.3f} [CI unavailable]'
    d = s["daily"]
    daily_metric = lambda key: f'{d["metrics"][key]:.3f} [{d["ci95"][key][0]:.3f}, {d["ci95"][key][1]:.3f}]' if d["ci95"][key][0] is not None else f'{d["metrics"][key]:.3f} [CI unavailable]'
    lines = ["## ⑤ Q0–Q5 검증 요약", "",
             f'- Q0: {s["structure"]["source_rows"]:,}개 시간 행 → 가정상 {s["structure"]["15min_rows"]:,}개 15분 위치. 기간 {s["structure"]["start"]} ~ {s["structure"]["end"]}. 잘못된 시간 {s["structure"]["invalid_hours"]}행; 행 순서로 임시 복원한 15분 위치 {s["structure"]["recovered_15min_rows_excluded"]}개와 연관 특징 행을 제외. 전력 단위/구간 경계는 미확인.',
             f'- 품질: 전력 결측 {s["structure"]["missing_15min_power"]}개, 원자료 음수 {s["structure"]["negative_raw_power"]}개·0값 {s["structure"]["zero_power"]}개, IQR 1.5배 기준 이상치 {s["structure"]["iqr_outliers"]}개. 원자료 결측 {s["structure"]["source_missing_by_column"]}. 0은 계측·비가동 여부가 불명확하여 임의 제거하지 않았다.',
             '- 변수: 4개 분 단위 전력, 같은 시간 평균, 생산량, 기온·풍속·습도·강수량, 요금/달력/공장인원/인건비. 모델에는 과거 전력, 완료된 시간의 생산량, 달력만 사용했다. 설비 ID·제품·실제 교대는 없다.',
             f'- 계절: 하계 월 {s["structure"]["summer_month_count"]}개, 동계 월 {s["structure"]["winter_month_count"]}개가 일부 포함된다. 하지만 9월은 14일까지이고 12월이 없어 완전한 7~9월·12~2월 창은 둘 다 없다. K5-a **{s["criteria"]["K5-a"]}**. 15분 열의 시간 의미도 미확인이다.',
             f'- Q1 1시간 후 15분값: 학습/검증/테스트 {h["rows"]["train"]:,}/{h["rows"]["validation"]:,}/{h["rows"]["test"]:,}행; 학습 p95={h["train_peak_95pct"]:.2f}.',
             f'- 나이브 후보 전일/전주/두 값의 평균 중 검증 MAE 최저 **{h["naive_selected_on_validation"]}**를 주 비교로 고정했다. 학습·검증/검증·시험 경계는 예측거리 1시간만큼 비웠다. 이전 관측값의 시각은 모두 미래 목표와 같은 15분 시각이다.',
             '', '| 지표 | 선택된 계절 나이브 | LightGBM |', '|---|---:|---:|']
    for label, suffix in (("전체 MAE", "mae"), ("전체 RMSE", "rmse"), ("전체 MAPE (0 제외, %)", "mape"), ("실제 피크 구간 MAE", "peak_mae")):
        lines.append(f'| {label} | {metric("naive_"+suffix)} | {metric("lgb_"+suffix)} |')
    lines += [f'- 피크 MAE 개선률 {metric("peak_mae_improvement_fraction")}; K5-b **{s["criteria"]["K5-b"]}**. 검증 전 고정된 LightGBM 설정의 단일 시험기간 결과이며 계절 외삽 근거는 아니다.',
              f'- 익일 일간 최대 15분값 Q1: 전일 23:45 기준 다음 달력날 최대값, 시험 {d["rows"]["test"]}일. 이 역시 원자료 분 열의 시간 해석이 맞다는 가정에 의존한다.',
              '', '| 일간 최대 지표 | 선택된 계절 나이브 | LightGBM |', '|---|---:|---:|']
    for label, suffix in (("MAE", "mae"), ("RMSE", "rmse"), ("MAPE (%)", "mape"), ("피크 일자 MAE", "peak_mae")):
        lines.append(f'| {label} | {daily_metric("naive_"+suffix)} | {daily_metric("lgb_"+suffix)} |')
    lines += ['- 익일 최대값에서 LightGBM은 MAE를 줄였지만 MAPE는 나이브보다 컸다. 값이 작은 날짜의 상대오차 개선 근거는 없다.',
              '- 시간순 3회 rolling-origin 점검은 최종 15% 시험구간 이전에만 수행했고, 각 폴드의 피크 경계는 해당 학습 앞부분에서 새로 정했다. 전일·전주·평균·LightGBM의 MAE/피크 MAE와 95% CI는 `05_rolling_origin.csv`에 수록.',
              f'- Q2: 평균 pinball {metric("mean_pinball")}. 실제 누적 커버리지: ' + ', '.join(f'{int(q*100)}%→{metric("q"+str(int(q*100))+"_coverage")}' for q in Q) + '.',
              '- 명목 95% 분위수의 실제 커버리지는 더 낮아 고위험 측면에서 과소포착 가능성이 있다. 외부 계절에서는 다시 보정해야 한다.',
              f'- Q3: 테스트 15분 피크 위치 {h["events"]["test"]}개, 연속 초과 에피소드 {h["test_peak_episodes"]}개, 피크 발생일 {h["test_peak_days"]}일; K5-c 판정은 **{s["criteria"]["K5-c"]}**이며 사전 기준대로 위치 수만 사용했다. 위치들은 서로 독립 사건이 아니다. 검증구간에서만 F1 cutoff 선정.',
              '', '| 피크 확률 방식 | F1 | PR-AUC | Brier |', '|---|---:|---:|---:|',
              f'| LightGBM 분류기 | {metric("classifier_f1")} | {metric("classifier_prauc")} | {metric("classifier_brier")} |',
              f'| 분위수 보간 | {metric("quantile_f1")} | {metric("quantile_prauc")} | {metric("quantile_brier")} |',
              f'- 분류기 정밀도 {metric("classifier_precision")}, 재현율 {metric("classifier_recall")}. 검증에서 선택된 확률 cutoff는 {h["f1_cutoff_from_validation"]}.',
              f'- Q4: 가정상 C/L=0.01~0.5. {s["q4_cost_formula"]} 실제 요금 절감액 또는 인과적 부하 이전 효과는 계산하지 않았다. 분모가 0 이하인 지점은 REV 정의 불가. `05_rev.png`, `05_summary.json`에 95% CI 포함.',
              '', '| C/L | REV (일자 블록 95% CI) |', '|---:|---:|']
    for ratio in (.05, .10, .20, .30, .50):
        i = int(np.argmin(np.abs(np.asarray(h["rev_cost_ratios"]) - ratio)))
        lines.append(f'| {h["rev_cost_ratios"][i]:.2f} | {metric("rev_"+str(i))} |')
    lines += ['- Q5: `05_error_conditions.csv`에 요일, 시간, 시각대 기반 교대 대용, 계절, **사후 관측 생산량** 구간별 잔차·FN·FP, 주요 비율의 날짜 블록 95% CI를 집계했다. 제품/전환 정보와 실제 교대는 없어 판정불가.']
    for factor, value, field, label in (("shift_proxy", "16-24", "recall", "16~24시 피크 재현율"),
                                        ("shift_proxy", "08-16", "fp_rate", "08~16시 비피크 오경보율")):
        g = conditions.loc[(conditions["factor"] == factor) & (conditions["value"] == value)].iloc[0]
        lines.append(f'- 탐색상 {label}: {g[field]:.3f} [{g[field+"_ci_low"]:.3f}, {g[field+"_ci_high"]:.3f}], FN {int(g["fn"])}·FP {int(g["fp"])}; 해당 시각대와 실제 교대의 일치는 미확인이다.')
    lines += ['- 위 조건 집계는 다중 비교를 조정한 인과·일반화 결론이 아니며, 특히 16~24시 피크 수가 적다.',
              '- 그림: `05_forecasts.png`, `05_reliability.png`, `05_rev.png`. 예측 원자료: `05_test_predictions.csv`, `05_daily_test_predictions.csv`.',
              '- 신뢰구간은 테스트 날짜를 묶어 1000회 재표집한 백분위 95% 구간이다. 계절 외삽·현장 적용 불확실성은 포함하지 않는다.',
              '', '### ⑤ 가정', '']
    lines += [f'- {item}' for item in s["assumptions"]]
    (OUT / "05_section.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
