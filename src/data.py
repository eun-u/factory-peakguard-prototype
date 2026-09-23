"""Lossless source-to-15-minute conversion with explicit data-quality evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = {"날짜", "시간", "15분", "30분", "45분", "60분"}
MINUTES = (15, 30, 45, 60)
ANALYSIS_COLUMNS = {
    "기온": "temperature", "풍속": "wind_speed", "습도": "humidity",
    "강수량": "precipitation", "공장인원": "headcount",
    "인건비": "labor_cost", "전기요금(계절)": "tariff_2021_raw",
}


def find_source(raw_dir: str | Path = "data/raw/task05_power") -> Path:
    matches: list[Path] = []
    for path in Path(raw_dir).rglob("*.csv"):
        try:
            columns = set(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
        except (UnicodeError, pd.errors.ParserError):
            continue
        if SOURCE_COLUMNS <= columns:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one task ⑤ CSV in {raw_dir}; found {len(matches)}")
    return matches[0]


def load_power_data(
    path: str | Path | None = None, output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read source, repair only derived timestamps, and optionally persist Parquet.

    ``ts_end`` means the end of a 15-minute interval. The raw file is read-only.
    Production and weather repeated on a row are analysis-only until their
    availability has been established separately.
    """
    path = Path(path) if path is not None else find_source()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"날짜": str})
    absent = SOURCE_COLUMNS - set(raw.columns)
    if absent:
        raise ValueError(f"Missing source columns: {sorted(absent)}")
    raw["source_date"] = pd.to_datetime(raw["날짜"], format="%Y%m%d", errors="raise")
    raw["source_hour"] = pd.to_numeric(raw["시간"], errors="coerce")
    raw["hour"] = -1
    raw["time_repaired"] = False
    if raw["source_date"].duplicated().all():
        raise ValueError("No unique date anchor for hourly timestamp repair")
    for _, day in raw.groupby("source_date", sort=False):
        expected = np.arange(24)
        observed = day["source_hour"].to_numpy(dtype=float)
        plausible = np.isfinite(observed) & (observed >= 0) & (observed <= 23)
        if len(day) != 24 or not np.array_equal(observed[plausible], expected[plausible]):
            raise ValueError("Cannot repair hours safely: require 24 ordered rows per date")
        raw.loc[day.index, "hour"] = expected
        raw.loc[day.index, "time_repaired"] = ~plausible

    hourly_end = raw["source_date"] + pd.to_timedelta(raw["hour"] + 1, unit="h")
    parts: list[pd.DataFrame] = []
    for minute in MINUTES:
        row = pd.DataFrame({
            "ts_end": raw["source_date"] + pd.to_timedelta(raw["hour"] * 60 + minute, unit="m"),
            "power": pd.to_numeric(raw[f"{minute}분"], errors="coerce"),
            "time_repaired": raw["time_repaired"].to_numpy(dtype=bool),
            # Same-hour production is recorded only for retrospective analysis.
            "production_target": pd.to_numeric(raw.get("생산량"), errors="coerce"),
        })
        for source, dest in ANALYSIS_COLUMNS.items():
            if source in raw:
                row[dest] = pd.to_numeric(raw[source], errors="coerce").to_numpy()
        parts.append(row)
    df = pd.concat(parts, ignore_index=True).set_index("ts_end").sort_index()
    df.index.name = "ts_end"
    if df.index.has_duplicates:
        raise ValueError("Duplicate 15-minute timestamp after repair")
    regular = pd.date_range(df.index.min(), df.index.max(), freq="15min", name="ts_end")
    df = df.reindex(regular)
    # Negative readings are treated as missing but preserved as raw evidence.
    negatives = int((df["power"] < 0).sum())
    df.loc[df["power"] < 0, "power"] = np.nan
    df["time_repaired"] = df["time_repaired"].fillna(False).astype(bool)
    df["missing_power"] = df["power"].isna()
    df["zero_power"] = df["power"].eq(0)
    clean = df.loc[~df["time_repaired"], "power"].dropna()
    q1, q3 = clean.quantile([.25, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    df["iqr_outlier"] = df["power"].lt(low) | df["power"].gt(high)

    # Each production observation becomes available only when its hour ends.
    completed = pd.Series(pd.to_numeric(raw.get("생산량"), errors="coerce").to_numpy(), index=pd.DatetimeIndex(hourly_end))
    repaired_hour = pd.Series(raw["time_repaired"].to_numpy(dtype=bool), index=pd.DatetimeIndex(hourly_end))
    if completed.index.has_duplicates:
        raise ValueError("Duplicate hourly production completion time")
    df["production_completed"] = completed.reindex(df.index)
    df["production_completed_bad"] = repaired_hour.reindex(df.index).fillna(False).astype(bool)
    df["production_known"] = completed.reindex(df.index).ffill()
    df["production_known_bad"] = repaired_hour.reindex(df.index).astype("boolean").ffill().fillna(True).astype(bool)
    df["production_hour_date"] = (df.index - pd.Timedelta(nanoseconds=1)).normalize()
    for field in ("production_target", *ANALYSIS_COLUMNS.values()):
        if field in df:
            df[f"missing_{field}"] = df[field].isna()

    # A zero is a valid observation; run length is descriptive and never a
    # forecast input because its complete run may end after the origin.
    zero_group = df["zero_power"].ne(df["zero_power"].shift()).cumsum()
    df["zero_run_length"] = df["zero_power"].groupby(zero_group).transform("sum").where(df["zero_power"], 0).astype(int)
    zero_rows = df.loc[df["zero_power"]]
    zero_runs = df.loc[df["zero_power"]].groupby(zero_group.loc[df["zero_power"]]).size()
    hourly_values = raw[[f"{minute}분" for minute in MINUTES]].apply(pd.to_numeric, errors="coerce")
    source_average = pd.to_numeric(raw.get("평균"), errors="coerce")
    average_error = (source_average - hourly_values.mean(axis=1)).abs()
    sum_error = (source_average - hourly_values.sum(axis=1)).abs()
    metadata = {
        "source": str(path), "sha256": source_hash, "source_rows": int(len(raw)),
        "source_columns": list(raw.columns[:18]), "rows_15min": int(len(df)),
        "start": str(df.index.min()), "end": str(df.index.max()),
        "time_repaired_hour_rows": int(raw["time_repaired"].sum()),
        "time_repaired_15min_rows": int(df["time_repaired"].sum()),
        "time_repaired_dates": sorted({str(day.date()) for day in raw.loc[raw["time_repaired"], "source_date"]}),
        "missing_power": int(df["missing_power"].sum()),
        "negative_source_power": negatives, "zero_power": int(df["zero_power"].sum()),
        "zero_run_count": int(len(zero_runs)),
        "max_zero_run_intervals": int(zero_runs.max()) if len(zero_runs) else 0,
        "zero_by_hour": {str(k): int(v) for k, v in zero_rows.groupby(zero_rows.index.hour).size().items()},
        "zero_by_weekday": {str(k): int(v) for k, v in zero_rows.groupby(zero_rows.index.dayofweek).size().items()},
        "iqr_outlier": int(df["iqr_outlier"].sum()),
        "average_column_mean_match_fraction": float(average_error.le(.5).mean()),
        "average_column_sum_match_fraction": float(sum_error.le(.5).mean()),
        "unit_inference_only": True,
    }
    df.attrs["raw_metadata"] = metadata
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        output_path.with_suffix(".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, metadata
