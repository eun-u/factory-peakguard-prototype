"""Counterfactual tariff band classification and explicitly labelled relative weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ._common import write_table


def read_tariff(value: dict | str | Path | None) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    with Path(value).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _minute(text: str) -> int:
    hour, minute = map(int, str(text).split(":"))
    if hour < 0 or hour > 24 or minute < 0 or minute > 59 or (hour == 24 and minute != 0):
        raise ValueError(f"Invalid tariff time {text}")
    return 60*hour+minute


def _season_schedule(tariff: dict, month: int) -> list[dict]:
    seasons = tariff.get("bands", tariff.get("seasons", {}))
    for value in seasons.values() if isinstance(seasons, dict) else seasons:
        if month in value.get("months", []):
            return value.get("intervals", value.get("schedule", []))
    raise ValueError(f"No tariff season covers month {month}")


def classify_tariff(times, tariff: dict) -> pd.Series:
    """Use target interval start, because timestamps are interval ends."""
    dates = pd.DatetimeIndex(pd.to_datetime(times))
    holidays = set(pd.to_datetime(tariff.get("holiday_dates_2021", [])).normalize())
    bands = []
    for stamp in dates:
        if pd.isna(stamp):
            bands.append(pd.NA)
            continue
        interval_start = stamp-pd.Timedelta(minutes=15)
        minute = 60*interval_start.hour+interval_start.minute
        schedules = _season_schedule(tariff, interval_start.month)
        found = [item["band"] for item in schedules if _minute(item["start"]) <= minute < _minute(item["end"])]
        if len(found) != 1:
            raise ValueError(f"Tariff interval missing or ambiguous at {stamp}")
        band = found[0]
        if tariff.get("sunday_holiday_off_peak", False) and (
            interval_start.dayofweek == 6 or interval_start.normalize() in holidays
        ):
            band = "off_peak"
        elif tariff.get("saturday_peak_to_mid", False) and interval_start.dayofweek == 5 and band == "peak":
            band = "mid_peak"
        bands.append(band)
    return pd.Series(bands, index=dates)


def tariff_weight_basis(tariff: dict) -> tuple[dict | None, str]:
    discount = bool(tariff.get("weekend_day_discount"))
    rates = tariff.get("rates", {})
    if rates and all(isinstance(v, (int, float)) and np.isfinite(v) for v in rates.values()) and tariff.get("rates_confirmed", False):
        return {str(k): float(v) for k, v in rates.items()}, "official_rate_with_discount" if discount else "official_rate"
    weights = tariff.get("scenario_weights", {})
    if weights and all(isinstance(v, (int, float)) and np.isfinite(v) for v in weights.values()):
        return {str(k): float(v) for k, v in weights.items()}, "illustrative_ordinal_scenario_with_official_discount" if discount else "illustrative_ordinal_scenario"
    return None, "bands_only"


def discount_factor(times, tariff: dict) -> np.ndarray:
    """Official weekend/holiday daytime energy-charge factor, if configured."""
    rule = tariff.get("weekend_day_discount")
    dates = pd.DatetimeIndex(pd.to_datetime(times))
    if not rule:
        return np.ones(len(dates))
    holidays = set(pd.to_datetime(tariff.get("holiday_dates_2021", [])).normalize())
    factor = np.ones(len(dates))
    for i, stamp in enumerate(dates):
        start = stamp-pd.Timedelta(minutes=15)
        minute = start.hour*60+start.minute
        if (start.month in rule.get("months", []) and
            (start.dayofweek >= 5 or start.normalize() in holidays) and
            _minute(rule["start"]) <= minute < _minute(rule["end"])):
            factor[i] = float(rule["factor"])
    return factor


def compare_tariff_periods(pred: pd.DataFrame, tariff_2021: dict, tariff_2026: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pred.copy()
    for year, tariff in ((2021, tariff_2021), (2026, tariff_2026)):
        x[f"band_{year}"] = classify_tariff(x.target_time, tariff).to_numpy()
    x["abs_error"] = (x.y-x.pred).abs()
    x["missed_peak"] = x.y.gt(x.tau) & ~pd.to_numeric(x.get("alert", x.pred.gt(x.tau)), errors="coerce").astype(bool)
    x["missed_excess"] = (x.y-x.tau).clip(lower=0).where(x.missed_peak, 0)
    x["evening_18_21"] = x.target_time.dt.hour.between(18, 20)
    interval_start = x.target_time-pd.Timedelta(minutes=15)
    winter_inferred = interval_start.dt.month.isin([11, 12, 1, 2]) & (not bool(tariff_2026.get("winter_time_bands_confirmed", True)))
    x["band_provenance_2026"] = np.where(winter_inferred, "unverified_winter_inference", "verified_schedule")
    rows = []
    for year, tariff in ((2021, tariff_2021), (2026, tariff_2026)):
        weights, basis = tariff_weight_basis(tariff)
        band_col = f"band_{year}"
        x[f"discount_factor_{year}"] = discount_factor(x.target_time, tariff)
        x[f"effective_weight_{year}"] = x[band_col].map(weights)*x[f"discount_factor_{year}"] if weights else np.nan
        for band, g in x.groupby(band_col):
            record = {"tariff_year": year, "band": band, "basis": basis,
                      "missed_weight_metric": "excess_above_tau_times_band_weight",
                      "n": len(g), "mae": float(g.abs_error.mean()),
                      "missed_peak_n": int(g.missed_peak.sum()),
                      "evening_18_21_n": int(g.evening_18_21.sum()),
                      "unverified_winter_n": int(g.band_provenance_2026.eq("unverified_winter_inference").sum()) if year == 2026 else 0}
            if weights is not None:
                if band not in weights:
                    raise ValueError(f"No weight/rate for tariff band {band}")
                wcol = f"effective_weight_{year}"
                total = float((x.abs_error*x[wcol]).sum())
                missed_total = float((x.missed_excess*x[wcol]).sum())
                record["weighted_abs_error_share"] = float((g.abs_error*g[wcol]).sum()/total) if total else np.nan
                record["weighted_missed_peak_share"] = float((g.missed_excess*g[wcol]).sum()/missed_total) if missed_total else np.nan
                record["evening_weighted_missed_share"] = float((g.missed_excess.where(g.evening_18_21, 0)*g[wcol]).sum()/missed_total) if missed_total else np.nan
            rows.append(record)
    return x, pd.DataFrame(rows)


def run_tariff(pred: pd.DataFrame, outdir: Path, tariff_2021: dict | None,
               tariff_2026: dict | None) -> tuple[dict, pd.Series | None, dict | None]:
    if not tariff_2021 or not tariff_2026:
        return {"status": "unsupported", "reason": "Both official band schedules are required"}, None, None
    detail, summary = compare_tariff_periods(pred, tariff_2021, tariff_2026)
    paths = {"tariff_detail": str(write_table(detail, outdir/"tables"/"tariff_counterfactual_detail.csv")),
             "tariff_summary": str(write_table(summary, outdir/"tables"/"tariff_counterfactual_summary.csv"))}
    bands = classify_tariff(pred.target_time, tariff_2026)
    basis = tariff_weight_basis(tariff_2026)[1]
    return {"status": "ok", "paths": paths, "weight_basis": basis,
            "winter_schedule_confirmed": bool(tariff_2026.get("winter_time_bands_confirmed", True)),
            "warning": "2021 관측에 2026 요금 시간대를 적용한 반사실적 분류입니다. 미확인 단가 대신 순서 가중치를 쓰면 금액과 절감액으로 해석할 수 없습니다. 2026 겨울 시간대는 원문 확인 전 추정값으로 별도 표시됩니다."}, bands, tariff_2026
