"""Small, time-ordered forecasting pipeline for the supplied CSV."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


HORIZONS = (1, 2, 4)  # 15, 30, 60 minutes
MODEL_NAMES = ("Persistence", "Same time last week", "HistGradientBoosting")
MIN_HISTORY_DAYS = 45


def load_series(zip_path: Path) -> pd.DataFrame:
    """Expand hourly rows into a continuous 15-minute index without editing the ZIP.

    An hourly row's 15/30/45/60-minute fields are placed at HH:15/HH:30/
    HH:45/(HH+1):00. The two days with invalid hours are positioned by their
    original 24-row order and explicitly flagged as provisional.
    """
    with ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith("okm_augumented_2021.csv")]
        if len(names) != 1:
            raise ValueError("ZIP 안에서 okm_augumented_2021.csv를 하나만 찾을 수 있어야 합니다.")
        with archive.open(names[0]) as source:
            raw = pd.read_csv(source, encoding="utf-8-sig", dtype={"날짜": str})

    required = {"날짜", "시간", "15분", "30분", "45분", "60분"}
    if not required.issubset(raw.columns):
        raise ValueError(f"필수 열이 없습니다: {sorted(required - set(raw.columns))}")

    raw["date"] = pd.to_datetime(raw["날짜"], format="%Y%m%d", errors="raise")
    raw["source_hour"] = pd.to_numeric(raw["시간"], errors="coerce")
    raw["hour"] = -1
    raw["hour_recovered"] = False
    for _, day in raw.groupby("date", sort=False):
        expected = np.arange(24)
        given = day["source_hour"].to_numpy()
        in_range = np.isfinite(given) & (given >= 0) & (given <= 23)
        if len(day) != 24 or not np.array_equal(given[in_range], expected[in_range]):
            raise ValueError("행 순서만으로 시간을 안전하게 복원할 수 없는 날짜가 있습니다.")
        raw.loc[day.index, "hour"] = expected
        raw.loc[day.index, "hour_recovered"] = ~in_range

    parts = []
    for minutes, column in ((15, "15분"), (30, "30분"), (45, "45분"), (60, "60분")):
        part = pd.DataFrame(
            {
                "timestamp": raw["date"] + pd.to_timedelta(raw["hour"] * 60 + minutes, unit="m"),
                "power": pd.to_numeric(raw[column], errors="coerce"),
                "hour_recovered": raw["hour_recovered"],
            }
        )
        parts.append(part)

    series = pd.concat(parts, ignore_index=True).set_index("timestamp").sort_index()
    if series.index.has_duplicates:
        raise ValueError("복원된 15분 시계열에 중복 시각이 있습니다.")
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="15min", name="timestamp")
    series = series.reindex(full_index)
    series.loc[series["power"] < 0, "power"] = np.nan
    series["hour_recovered"] = series["hour_recovered"].fillna(False).astype(bool)
    series["missing_power"] = series["power"].isna()
    return series


def make_features(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Every feature at origin t is observable by t (or is calendar-only)."""
    if horizon not in HORIZONS:
        raise ValueError("지원하지 않는 예측 거리입니다.")
    power = history["power"]
    target_time = history.index + timedelta(minutes=15 * horizon)
    features = pd.DataFrame(index=history.index)
    features["current"] = power
    for lag in (1, 2, 4, 8, 96):
        features[f"lag_{lag}"] = power.shift(lag)
    # The value at (t + horizon - 7 days) is already in the past at origin t.
    features["week_target"] = power.shift(7 * 96 - horizon)
    features["recent_hour_mean"] = power.rolling(4, min_periods=1).mean()
    features["recent_day_mean"] = power.rolling(96, min_periods=12).mean()
    features["target_hour"] = target_time.hour
    features["target_quarter"] = target_time.minute // 15
    features["target_weekday"] = target_time.dayofweek
    features["target_month"] = target_time.month
    return features


def _baseline_predictions(features: pd.DataFrame) -> dict[str, np.ndarray]:
    current = features["current"].to_numpy(dtype=float)
    weekly = features["week_target"].to_numpy(dtype=float)
    return {
        "Persistence": current,
        "Same time last week": np.where(np.isfinite(weekly), weekly, current),
    }


def run_snapshot(series: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Fit and calibrate using only observations available at `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    history = series.loc[:cutoff].copy()
    if cutoff not in history.index or pd.isna(history.at[cutoff, "power"]):
        raise ValueError("선택 시각에 실제 전력값이 없습니다.")
    if len(history) < MIN_HISTORY_DAYS * 96:
        raise ValueError(f"최소 {MIN_HISTORY_DAYS}일의 과거 데이터가 필요합니다.")

    train_end = history.index[int(len(history) * 0.8)]
    clean_training_power = history.loc[:train_end].loc[
        lambda frame: (~frame["missing_power"]) & (~frame["hour_recovered"]), "power"
    ]
    threshold = float(clean_training_power.quantile(0.95))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("피크 임계값을 계산할 학습 전력값이 부족합니다.")

    recent = history.tail(5)
    quality_reasons = []
    if recent["hour_recovered"].any():
        quality_reasons.append("최근 1시간에 행 순서로 복원한 시각이 있습니다")
    if recent["missing_power"].any():
        quality_reasons.append("최근 1시간에 전력 결측이 있습니다")

    results: dict[str, dict[int, dict]] = {name: {} for name in MODEL_NAMES}
    metrics = []
    recent_bad = (history["hour_recovered"] | history["missing_power"]).rolling(
        5, min_periods=1
    ).max().astype(bool)

    for horizon in HORIZONS:
        features = make_features(history, horizon)
        labels = history["power"].shift(-horizon)
        target_bad = (history["hour_recovered"] | history["missing_power"]).shift(
            -horizon, fill_value=True
        )
        eligible = (
            labels.notna()
            & ~recent_bad
            & ~target_bad
            & (features.index >= history.index[0] + timedelta(days=7))
        )
        train_mask = eligible & (features.index + timedelta(minutes=15 * horizon) <= train_end)
        calibration_mask = eligible & (features.index > train_end)
        x_train, y_train = features.loc[train_mask], labels.loc[train_mask]
        x_cal, y_cal = features.loc[calibration_mask], labels.loc[calibration_mask]
        if len(x_train) < 500 or len(x_cal) < 100:
            raise ValueError("시간순 학습·검증에 필요한 정상 관측이 부족합니다.")

        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        model.fit(x_train, y_train)
        calibration_predictions = _baseline_predictions(x_cal)
        calibration_predictions["HistGradientBoosting"] = model.predict(x_cal)
        now_features = features.loc[[cutoff]]
        now_predictions = _baseline_predictions(now_features)
        now_predictions["HistGradientBoosting"] = model.predict(now_features)

        for name in MODEL_NAMES:
            errors = y_cal.to_numpy(dtype=float) - calibration_predictions[name]
            radius = float(np.quantile(np.abs(errors), 0.90))
            estimate = max(0.0, float(now_predictions[name][0]))
            results[name][horizon] = {
                "prediction": estimate,
                "lower": max(0.0, estimate - radius),
                "upper": estimate + radius,
                "radius": radius,
                "weekly_fallback": bool(
                    name == "Same time last week" and now_features["week_target"].isna().iloc[0]
                ),
            }
            metrics.append(
                {
                    "model": name,
                    "minutes": horizon * 15,
                    "mae": float(np.mean(np.abs(errors))),
                    "calibration_rows": len(x_cal),
                }
            )

    return {
        "cutoff": cutoff,
        "train_end": train_end,
        "threshold": threshold,
        "current": float(history.at[cutoff, "power"]),
        "quality_reasons": quality_reasons,
        "results": results,
        "metrics": pd.DataFrame(metrics),
        "history": history.tail(97),
    }


def decide(snapshot: dict, model_name: str) -> dict:
    """Transparent review rules; no equipment action is produced."""
    forecasts = snapshot["results"][model_name]
    threshold = snapshot["threshold"]
    current = snapshot["current"]
    high_risk = current >= threshold or any(
        item["prediction"] >= threshold for item in forecasts.values()
    )
    wide = any(2 * item["radius"] >= 0.30 * threshold for item in forecasts.values())
    reasons = list(snapshot["quality_reasons"])
    if any(item["weekly_fallback"] for item in forecasts.values()):
        reasons.append("지난주 같은 시각 값이 없어 현재값으로 대체했습니다")
    if wide:
        reasons.append("한 예측구간의 전체 폭이 피크 임계값의 30% 이상입니다")

    if reasons:
        status = "REVIEW_REQUIRED"
    elif max(current, *(item["prediction"] for item in forecasts.values())) >= 1.05 * threshold:
        status = "LOAD_SHIFT_REVIEW"
        reasons.append("현재 또는 예측 전력이 임시 피크 기준을 5% 이상 넘습니다")
    elif high_risk:
        status = "PEAK_WATCH"
        reasons.append("현재 또는 미래 예측값이 임시 피크 기준에 도달합니다")
    else:
        status = "NORMAL"
        reasons.append("현재와 15·30·60분 예측값이 임시 피크 기준 아래입니다")

    return {"status": status, "reasons": reasons, "wide": wide, "high_risk": high_risk}
