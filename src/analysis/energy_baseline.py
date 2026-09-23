"""Explanatory energy baseline and descriptive peak episode types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ._common import block_ci, write_table
from .errors import episodes


@dataclass
class EnergyBaseline:
    model: Ridge
    temperature_median: float
    production_coefficient_per_hour: float
    train_end: pd.Timestamp

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_matrix(frame, self.temperature_median))


def _matrix(frame: pd.DataFrame, temperature_median: float) -> np.ndarray:
    idx = pd.DatetimeIndex(frame.index)
    production = pd.to_numeric(frame["production_target"], errors="coerce").fillna(0).to_numpy()
    temperature = pd.to_numeric(frame.get("temperature", temperature_median), errors="coerce")
    if np.isscalar(temperature):
        temperature = np.full(len(frame), float(temperature))
    else:
        temperature = temperature.fillna(temperature_median).to_numpy()
    hours = idx.hour + idx.minute/60
    weekday = idx.dayofweek
    return np.column_stack([
        production/4,  # hourly production allocated uniformly to 15-minute analysis bins
        np.maximum(0, temperature-18),
        np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
        np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7),
        (weekday >= 5).astype(float),
    ])


def fit_energy_baseline(train: pd.DataFrame) -> EnergyBaseline:
    if "production_target" not in train:
        raise ValueError("Production data is required for the explanatory baseline")
    clean = train.loc[train.power.notna() & train.production_target.notna()].copy()
    if "time_repaired" in clean:
        clean = clean.loc[~clean.time_repaired.astype(bool)]
    if len(clean) < 100:
        raise ValueError("Insufficient clean training rows for energy baseline")
    temperature = pd.to_numeric(clean.get("temperature", pd.Series(dtype=float)), errors="coerce")
    temp_med = float(temperature.median()) if len(temperature) and temperature.notna().any() else 18.0
    model = Ridge(alpha=100.0).fit(_matrix(clean, temp_med), clean.power.to_numpy(float))
    return EnergyBaseline(model, temp_med, float(model.coef_[0]), pd.Timestamp(clean.index.max()))


def energy_intensity_proxy(history: pd.DataFrame) -> pd.DataFrame:
    """Ratio of four 15-minute power values to the same hour's production.

    The numerator keeps source units. This is a descriptive proxy, not kWh/unit.
    """
    if "production_target" not in history:
        return pd.DataFrame(columns=["hour", "production", "power_sum", "sec_proxy"])
    x = history.copy()
    if "time_repaired" in x:
        x = x.loc[~x.time_repaired.astype(bool)]
    x["hour"] = (x.index-pd.Timedelta(nanoseconds=1)).floor("h")
    hourly = x.groupby("hour").agg(production=("production_target", "first"),
                                   power_sum=("power", "sum"), n=("power", "count"))
    hourly = hourly.loc[hourly.production.gt(0) & hourly.n.eq(4)].copy()
    hourly["sec_proxy"] = hourly.power_sum/hourly.production
    return hourly.reset_index()[["hour", "production", "power_sum", "sec_proxy"]]


def classify_peak_types(pred: pd.DataFrame, history: pd.DataFrame, baseline: EnergyBaseline,
                        n_boot: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = history.reindex(pd.DatetimeIndex(pred.target_time))
    baseline_pred = baseline.predict(target)
    x = pred[["target_time", "y", "tau"]].copy()
    x["baseline"] = baseline_pred
    rows = []
    for start, end in episodes(x.target_time, x.y > x.tau):
        episode = x.loc[x.target_time.between(start, end)]
        peak_row = episode.loc[episode.y.idxmax()]
        kind = "production_explained" if peak_row.baseline > peak_row.tau else "excess_residual"
        rows.append({"start": start, "end": end, "peak_time": peak_row.target_time,
                     "hour": peak_row.target_time.hour, "type": kind,
                     "actual_max": peak_row.y, "baseline_at_max": peak_row.baseline,
                     "residual_at_max": peak_row.y-peak_row.baseline, "tau": peak_row.tau})
    event_table = pd.DataFrame(rows)
    if event_table.empty:
        return event_table, pd.DataFrame()
    summary = []
    for kind, group in event_table.groupby("type"):
        prevalence, low, high = block_ci(event_table.rename(columns={"peak_time": "target_time"}),
            lambda s: (s.type == kind).mean(), n=n_boot, seed=seed)
        summary.append({"type": kind, "episodes": len(group), "share": prevalence,
                        "share_ci_low": low, "share_ci_high": high,
                        "median_hour": float(group.hour.median())})
    return event_table, pd.DataFrame(summary)


def run_energy_baseline(pred: pd.DataFrame, history: pd.DataFrame, train: pd.DataFrame,
                        outdir: Path, cfg: dict) -> tuple[dict, EnergyBaseline | None]:
    try:
        model = fit_energy_baseline(train)
    except ValueError as exc:
        return {"status": "unsupported", "reason": str(exc)}, None
    if model.train_end >= pred.target_time.min():
        return {"status": "unsupported", "reason": "Training data overlaps OOF analysis targets"}, None
    n_boot = int(cfg.get("bootstrap", {}).get("n", 1000))
    events, summary = classify_peak_types(pred, history, model, n_boot, int(cfg.get("seed", 42)))
    sec = energy_intensity_proxy(history)
    base_mask = pd.to_numeric(history.production_target, errors="coerce").eq(0)
    if "time_repaired" in history:
        base_mask &= ~history.time_repaired.astype(bool)
    base = history.loc[base_mask, "power"]
    base_summary = pd.DataFrame([{"n": len(base), "p10": base.quantile(0.1),
                                  "median": base.median(), "p90": base.quantile(0.9)}])
    tables = outdir/"tables"
    paths = {"peak_types": str(write_table(events, tables/"peak_types.csv")),
             "peak_type_summary": str(write_table(summary, tables/"peak_type_summary.csv")),
             "energy_intensity_proxy": str(write_table(sec, tables/"energy_intensity_proxy.csv")),
             "base_load_proxy": str(write_table(base_summary, tables/"base_load_proxy.csv"))}
    return {"status": "ok", "paths": paths,
            "production_coefficient_per_hour": model.production_coefficient_per_hour,
            "train_end": str(model.train_end),
            "warning": "동시각 생산량과 실측 기상을 사용한 사후 설명 모형입니다. 생산량 4등분 및 SEC는 단위 미확인 대리 지표이며 인과 효과가 아닙니다."}, model
