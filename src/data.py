"""원자료 로딩, 시간 오류 복원, 15분 전개, 품질 점검.

load_series의 동작은 사전 검증(verification/t5_power.py)과 같다. 원자료는 읽기만 하며
복원·전개 결과를 data/raw/에 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, resolve

POWER_KEYS = {"날짜", "시간", "15분", "30분", "45분", "60분"}

# 변수 사전. 의미는 열 이름과 값 범위로 추정한 것이며 주최측 확인 전이다.
VARIABLES = [
    ("날짜", "측정 날짜 (YYYYMMDD)", "달력", "예측 대상 시각의 달력 특징으로만 사용"),
    ("시간", "시(0~23). 2개 날짜 48행은 범위 밖 값", "달력", "행 순서로 복원 후 사용"),
    ("15분", "HH:15에 끝나는 15분 구간 전력 (가정 A4, 단위 미확인)", "목표·시차", "원점 이전 값만 사용"),
    ("30분", "HH:30에 끝나는 15분 구간 전력", "목표·시차", "원점 이전 값만 사용"),
    ("45분", "HH:45에 끝나는 15분 구간 전력", "목표·시차", "원점 이전 값만 사용"),
    ("60분", "(HH+1):00에 끝나는 15분 구간 전력", "목표·시차", "원점 이전 값만 사용"),
    ("평균", "같은 시간 네 값의 평균(반올림)", "제외", "목표 시점 정보를 포함하므로 입력 제외 (A8)"),
    ("생산량", "시간당 생산량 (실적으로 가정)", "완료 시차", "해당 시간이 끝난 (HH+1):00 이후에만 사용 (A8)"),
    ("기온", "기온 실측", "분석 전용", "미래 실측 기상은 예측 시점에 알 수 없어 입력 제외"),
    ("풍속", "풍속 실측", "분석 전용", "입력 제외"),
    ("습도", "습도 실측", "분석 전용", "입력 제외"),
    ("강수량", "강수량 실측", "분석 전용", "입력 제외"),
    ("전기요금(계절)", "계절별 요금 구분 값", "분석 전용", "달력(월)과 중복되어 입력 제외"),
    ("day", "요일 코드 (원자료 표기)", "분석 전용", "목표 시각에서 다시 계산한 요일을 사용"),
    ("d", "일(1~31)", "분석 전용", "목표 시각에서 다시 계산"),
    ("m", "월(1~9)", "분석 전용", "목표 시각에서 다시 계산"),
    ("공장인원", "공장 인원", "분석 전용", "기록 시점이 불명확하여 입력 제외"),
    ("인건비", "인건비 구분 값", "분석 전용", "입력 제외"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_source(cfg: dict) -> Path:
    """설정 경로를 우선 사용하고, 없으면 열 이름으로 과제 ⑤ CSV 한 개를 찾는다."""
    direct = resolve(cfg["data"]["path"])
    if direct.is_file():
        return direct
    source_dir = resolve(cfg["data"]["search_dir"])
    if not source_dir.is_dir():
        raise FileNotFoundError(f"원자료 폴더가 없습니다: {source_dir}")
    matches = []
    for path in source_dir.rglob("*.csv"):
        try:
            columns = pd.read_csv(path, nrows=0, encoding=cfg["data"]["encoding"]).columns
        except (UnicodeError, pd.errors.ParserError):
            continue
        if POWER_KEYS.issubset(columns):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"날짜·시간·전력 4열을 가진 CSV가 정확히 1개여야 합니다. 찾은 수: {len(matches)}")
    return matches[0]


def load_series(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = locate_source(cfg)
    minutes = tuple(cfg["data"]["minute_columns"])
    raw = pd.read_csv(path, encoding=cfg["data"]["encoding"])
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
    for minute in minutes:
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

    # 가정 A8: HH시 생산량은 (HH+1):00에 관측된다.
    hourly = raw[["date", "hour", "생산량", "recovered"]].copy()
    hourly.index = hourly["date"] + pd.to_timedelta(hourly["hour"] + 1, unit="h")
    hourly = hourly.sort_index()
    series["production_known"] = hourly["생산량"].reindex(series.index).ffill()
    series["production_known_bad"] = hourly["recovered"].reindex(series.index).astype("boolean").ffill().fillna(True).astype(bool)
    series["production_target"] = series["production_target"].astype(float)

    all_power = raw[[f"{m}분" for m in minutes]].to_numpy(dtype=float).ravel()
    q1, q3 = np.nanquantile(all_power, [0.25, 0.75])
    complete_summer = any(series.index.min() <= pd.Timestamp(f"{year}-07-01") and
                          series.index.max() >= pd.Timestamp(f"{year}-09-30 23:45")
                          for year in range(series.index.min().year, series.index.max().year + 1))
    complete_winter = any(series.index.min() <= pd.Timestamp(f"{year}-12-01") and
                          series.index.max() >= pd.Timestamp(f"{year+1}-02-28 23:45")
                          for year in range(series.index.min().year - 1, series.index.max().year + 1))
    try:
        source = str(path.relative_to(ROOT))
    except ValueError:
        source = path.name
    info = {
        "source": source, "source_sha256": sha256(path),
        "source_rows": len(raw),
        "source_columns": list(pd.read_csv(path, nrows=0, encoding=cfg["data"]["encoding"]).columns),
        "15min_rows": len(series), "start": str(series.index.min()), "end": str(series.index.max()),
        "missing_15min_power": int(series["power"].isna().sum()),
        "negative_raw_power": int((all_power < 0).sum()), "zero_power": int((all_power == 0).sum()),
        "iqr_outliers": int(((all_power < q1 - 1.5 * (q3 - q1)) | (all_power > q3 + 1.5 * (q3 - q1))).sum()),
        "iqr_bounds": [float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))],
        "invalid_hours": int(raw["recovered"].sum()),
        "recovered_dates": [str(d.date()) for d in raw.loc[raw["recovered"], "date"].unique()],
        "recovered_15min_rows_excluded": int(series["recovered"].sum()),
        "months": sorted(set(series.index.month.tolist())),
        "complete_summer_window": bool(complete_summer),
        "complete_winter_window": bool(complete_winter),
        "production_missing": int(raw["생산량"].isna().sum()),
        "source_missing_by_column": {col: int(n) for col, n in raw.isna().sum().items() if n and col in raw.columns[:18]},
        "full_duplicate_rows": int(raw.iloc[:, :18].duplicated().sum()),
        "duplicate_date_hour_after_recovery": int(raw.duplicated(["date", "hour"]).sum()),
        "raw_average_matches_rounded_four_values_fraction": float(np.mean(
            np.abs(raw["평균"].to_numpy() - raw[[f"{m}분" for m in minutes]].mean(axis=1).to_numpy()) <= 0.5)),
    }
    expected_hash = cfg["data"].get("expected_sha256")
    info["source_sha256_matches_expected"] = bool(expected_hash) and info["source_sha256"] == expected_hash
    return series, raw, info


def variable_dictionary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, meaning, role, availability in VARIABLES:
        if column not in raw.columns:
            continue
        values = raw[column]
        numeric = pd.to_numeric(values, errors="coerce")
        rows.append({"변수": column, "의미(추정)": meaning, "역할": role, "가용 시점·사용 방식": availability,
                     "자료형": str(values.dtype), "결측": int(values.isna().sum()),
                     "고유값": int(values.nunique()),
                     "최소": float(numeric.min()) if numeric.notna().any() else np.nan,
                     "최대": float(numeric.max()) if numeric.notna().any() else np.nan})
    return pd.DataFrame(rows)


def zero_value_runs(series: pd.DataFrame) -> pd.DataFrame:
    """전력 0값의 연속 구간. 비가동 추정의 근거(시간대·요일·생산량)를 함께 남긴다."""
    zero = (series["power"] == 0).to_numpy()
    times = series.index
    rows, start = [], None
    for i, flag in enumerate(np.r_[zero, False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            part = series.iloc[start:i]
            rows.append({"start": str(times[start]), "end": str(times[i - 1]), "positions": i - start,
                         "start_hour": int(times[start].hour), "weekday": int(times[start].dayofweek),
                         "production_sum_during": float(part["production_target"].sum()),
                         "recovered_overlap": bool(part["recovered"].any())})
            start = None
    return pd.DataFrame(rows)


def calendar_profiles(series: pd.DataFrame, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """월·요일·시간대별 전력 분포와, 시간 단위 전력과 다른 변수의 관계(분석용, 모델 입력 아님)."""
    valid = series.loc[~series["recovered"] & series["power"].notna(), ["power"]].copy()
    position_start = valid.index - pd.Timedelta(minutes=15)
    valid["month"] = position_start.month
    valid["weekday"] = position_start.dayofweek
    valid["hour"] = position_start.hour
    out = {}
    for key in ("month", "weekday", "hour"):
        out[key] = valid.groupby(key)["power"].agg(
            n="size", mean="mean", median="median",
            p95=lambda s: s.quantile(0.95), max="max").reset_index()
    hourly = raw.loc[~raw["recovered"]].copy()
    hourly["power_mean"] = hourly[["15분", "30분", "45분", "60분"]].mean(axis=1)
    relations = []
    for column in ("생산량", "기온", "습도", "공장인원"):
        if column in hourly:
            pair = hourly[["power_mean", column]].dropna()
            relations.append({"변수": column, "n": len(pair),
                              "spearman_rho": float(pair.corr(method="spearman").iloc[0, 1]),
                              "pearson_r": float(pair.corr().iloc[0, 1])})
    out["relations"] = pd.DataFrame(relations)
    return out
