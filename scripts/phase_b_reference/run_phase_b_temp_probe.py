#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporary Phase B probe reproduced from the ChatGPT-session analysis.

IMPORTANT
---------
This file reproduces the *temporary* Phase B analysis that was run in chat.
It is NOT the later/final Phase B protocol.

The temporary D2 sensitivity uses the OLD conservative rule: keep only the
first date for each exact 24-hour power profile in BOTH training and scoring.
Later discussion replaced this with a novel-profile score-subset definition.

What this script does
---------------------
1) Verify the original CSV SHA-256.
2) Restore invalid hour fields on 2021-07-13 and 2021-07-15 by row order.
3) Expand 15/30/45/60-minute columns to a 15-minute power series.
4) Build G0/G1/G2/G3/G4/G5/G7 feature groups used in the temporary probe.
5) Evaluate 13 direct horizons: 60, 75, ..., 240 minutes.
6) Use a fixed standardized Ridge probe (alpha=100).
7) Run Forward contribution, LOFO, and OLD D2 representative-profile tests.
8) Use 3 chronological development folds and date-block bootstrap (1000).
9) Never access the locked final evaluation period beginning 2021-08-09 09:45.

Example
-------
python run_phase_b_temp_probe.py \
  --input data/raw/task05_power/okm_augumented_2021.csv \
  --outdir outputs/phase_b_temp
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
DEV_BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
RIDGE_ALPHA = 100.0
BOOTSTRAP_N = 1000
SEED = 42
BAD_DATE_STRINGS = {"20210713", "20210715"}

FOLDS = [
    dict(train_end=pd.Timestamp("2021-03-15 10:45:00"), score_start=pd.Timestamp("2021-04-14 15:00:00"), score_end=pd.Timestamp("2021-05-13 05:15:00")),
    dict(train_end=pd.Timestamp("2021-04-19 16:45:00"), score_start=pd.Timestamp("2021-05-28 16:30:00"), score_end=pd.Timestamp("2021-06-26 06:45:00")),
    dict(train_end=pd.Timestamp("2021-05-24 22:45:00"), score_start=pd.Timestamp("2021-07-07 21:30:00"), score_end=pd.Timestamp("2021-08-09 08:30:00")),
]

HOLIDAYS = {
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13",
    "2021-03-01", "2021-05-05", "2021-05-19", "2021-06-06",
    "2021-08-15", "2021-08-16",
}

GROUPS: Dict[str, List[str]] = {
    "G0": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", "holiday"],
    "G1": ["current", "lag4", "slot1d", "slot7d"],
    "G2": ["r4_mean", "r4_max", "r4_std", "r16_max", "r96_max"],
    "G3": ["prod_last", "prod_prevday", "prod_samehour7"],
    "G4": ["head_last", "labor_target"],
    "G5": ["temp_last", "humid_last", "wind_last", "rain_last"],
    "G7": ["tariff_target"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
CORE_FEATURES = GROUPS["G0"] + GROUPS["G1"] + GROUPS["G2"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rounded_four_mean_integrity(df: pd.DataFrame) -> float:
    vals = df[["15분", "30분", "45분", "60분"]].mean(axis=1)
    return float((np.abs(vals - df["평균"]) <= 0.5).mean())


def load_and_prepare_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        "날짜", "시간", "15분", "30분", "45분", "60분", "평균",
        "생산량", "기온", "풍속", "습도", "강수량", "전기요금(계절)",
        "day", "d", "m", "공장인원", "인건비",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["날짜_str"] = df["날짜"].astype(int).astype(str).str.zfill(8)
    df["date"] = pd.to_datetime(df["날짜_str"], format="%Y%m%d")
    df["_row_in_date"] = df.groupby("날짜_str").cumcount()

    invalid = ~df["시간"].between(0, 23)
    damaged_dates = set(df.loc[invalid, "날짜_str"])
    if damaged_dates != BAD_DATE_STRINGS:
        raise ValueError(f"Unexpected damaged dates: {sorted(damaged_dates)}")

    df["hour_repaired"] = df["시간"].astype(int)
    bad_mask = df["날짜_str"].isin(BAD_DATE_STRINGS)
    df.loc[bad_mask, "hour_repaired"] = df.loc[bad_mask, "_row_in_date"].astype(int)
    df["time_repaired"] = bad_mask
    df["hour_start"] = df["date"] + pd.to_timedelta(df["hour_repaired"], unit="h")
    return df


def exact_profile_representatives(df: pd.DataFrame) -> Tuple[set[int], int, int]:
    reps: set[int] = set()
    seen: dict[Tuple[float, ...], int] = {}
    for date_int, g in df.sort_values(["date", "_row_in_date"]).groupby("날짜", sort=True):
        g = g.sort_values("hour_repaired")
        if len(g) != 24:
            continue
        sig = tuple(g[["15분", "30분", "45분", "60분"]].to_numpy(dtype=float).reshape(-1).tolist())
        if sig not in seen:
            seen[sig] = int(date_int)
            reps.add(int(date_int))
    total_days = int(df["날짜"].nunique())
    unique_profiles = len(seen)
    return reps, unique_profiles, total_days - unique_profiles


def expand_15min(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        vals = [row["15분"], row["30분"], row["45분"], row["60분"]]
        for j, value in enumerate(vals, start=1):
            rows.append({
                "timestamp": pd.Timestamp(row["hour_start"]) + pd.Timedelta(minutes=15 * j),
                "power": float(value),
                "source_date": int(row["날짜"]),
                "bad_time": bool(row["time_repaired"]),
            })
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ValueError("15-minute reconstruction produced duplicate timestamps.")
    return out


@dataclass
class DataContext:
    raw: pd.DataFrame
    power: pd.DataFrame
    power_map: Mapping[pd.Timestamp, Tuple[float, bool, int]]
    hour_map: Mapping[pd.Timestamp, pd.Series]
    daily_prod: Mapping[pd.Timestamp, float]
    representative_dates: set[int]


def build_context(raw: pd.DataFrame, p15: pd.DataFrame, representative_dates: set[int]) -> DataContext:
    power_map = {
        pd.Timestamp(r.timestamp): (float(r.power), bool(r.bad_time), int(r.source_date))
        for r in p15.itertuples(index=False)
    }
    hour_map = {pd.Timestamp(row["hour_start"]): row for _, row in raw.iterrows()}
    daily_prod = raw.groupby("date")["생산량"].sum().astype(float).to_dict()
    return DataContext(raw, p15, power_map, hour_map, daily_prod, representative_dates)


def clean_power(ctx: DataContext, ts: pd.Timestamp) -> float | None:
    x = ctx.power_map.get(pd.Timestamp(ts))
    if x is None:
        return None
    value, bad, _ = x
    if bad or not np.isfinite(value):
        return None
    return float(value)


def valid_hour_row(row: pd.Series | None) -> bool:
    return row is not None and str(int(row["날짜"])).zfill(8) not in BAD_DATE_STRINGS


def latest_completed_hour(ctx: DataContext, origin: pd.Timestamp) -> pd.Series | None:
    return ctx.hour_map.get(origin.floor("h") - pd.Timedelta(hours=1))


def tariff_proxy(ts: pd.Timestamp) -> float:
    # Historical temporary-probe proxy only; not an authoritative tariff definition.
    m = ts.month
    if m <= 2:
        return 109.8
    if 6 <= m <= 8:
        return 191.6
    return 167.2


def build_feature_vector(ctx: DataContext, origin: pd.Timestamp, target: pd.Timestamp) -> dict | None:
    current = clean_power(ctx, origin)
    lag4 = clean_power(ctx, origin - pd.Timedelta(hours=1))
    slot1d = clean_power(ctx, target - pd.Timedelta(days=1))
    slot7d = clean_power(ctx, target - pd.Timedelta(days=7))
    if any(v is None for v in (current, lag4, slot1d, slot7d)):
        return None

    recent96 = []
    for k in range(96):
        v = clean_power(ctx, origin - pd.Timedelta(minutes=15 * k))
        if v is None:
            return None
        recent96.append(v)
    recent4, recent16 = recent96[:4], recent96[:16]

    er = latest_completed_hour(ctx, origin)
    if not valid_hour_row(er):
        return None

    prev_day = target.normalize() - pd.Timedelta(days=1)
    if prev_day.strftime("%Y%m%d") in BAD_DATE_STRINGS:
        return None

    same_hour7 = []
    for d in range(1, 8):
        day = target.normalize() - pd.Timedelta(days=d)
        rr = ctx.hour_map.get(day + pd.Timedelta(hours=target.hour))
        if valid_hour_row(rr) and pd.notna(rr["생산량"]):
            same_hour7.append(float(rr["생산량"]))
    if not same_hour7:
        return None

    hour_float = target.hour + target.minute / 60.0
    sunday_based_dow = (target.dayofweek + 1) % 7

    return {
        "hour_sin": np.sin(2 * np.pi * hour_float / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_float / 24.0),
        "dow_sin": np.sin(2 * np.pi * sunday_based_dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * sunday_based_dow / 7.0),
        "weekend": float(target.weekday() >= 5),
        "holiday": float(target.strftime("%Y-%m-%d") in HOLIDAYS),
        "current": float(current),
        "lag4": float(lag4),
        "slot1d": float(slot1d),
        "slot7d": float(slot7d),
        "r4_mean": float(np.mean(recent4)),
        "r4_max": float(np.max(recent4)),
        "r4_std": float(np.std(recent4, ddof=1)),
        "r16_max": float(np.max(recent16)),
        "r96_max": float(np.max(recent96)),
        "prod_last": float(er["생산량"]) if pd.notna(er["생산량"]) else np.nan,
        "prod_prevday": float(ctx.daily_prod.get(prev_day, np.nan)),
        "prod_samehour7": float(np.mean(same_hour7)),
        "head_last": float(er["공장인원"]) if pd.notna(er["공장인원"]) else np.nan,
        "labor_target": 1.0 if 9 <= target.hour <= 17 else 1.5,
        "temp_last": float(er["기온"]) if pd.notna(er["기온"]) else np.nan,
        "humid_last": float(er["습도"]) if pd.notna(er["습도"]) else np.nan,
        "wind_last": float(er["풍속"]) if pd.notna(er["풍속"]) else np.nan,
        "rain_last": float(er["강수량"]) if pd.notna(er["강수량"]) else np.nan,
        "tariff_target": float(tariff_proxy(target)),
    }


def build_samples(ctx: DataContext, horizon_min: int, d2_old: bool = False) -> pd.DataFrame:
    rows = []
    h = pd.Timedelta(minutes=horizon_min)
    for r in ctx.power.itertuples(index=False):
        origin = pd.Timestamp(r.timestamp)
        if origin >= DEV_BOUNDARY:
            break
        target = origin + h
        info = ctx.power_map.get(target)
        if info is None:
            continue
        y, bad, source_date = info
        if bad or target >= DEV_BOUNDARY:
            continue
        if d2_old and int(source_date) not in ctx.representative_dates:
            continue
        f = build_feature_vector(ctx, origin, target)
        if f is None:
            continue
        rows.append({"origin": origin, "target": target, "y": float(y), "date": int(source_date), **f})
    return pd.DataFrame(rows)


def ridge_prepare(train: pd.DataFrame, score: pd.DataFrame) -> dict:
    Xtr = train[ALL_FEATURES].to_numpy(float)
    Xsc = score[ALL_FEATURES].to_numpy(float)
    ytr = train["y"].to_numpy(float)

    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
    Xsc = np.where(np.isfinite(Xsc), Xsc, mu)

    sd = Xtr.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-9), sd, 1.0)
    Ztr = (Xtr - mu) / sd
    Zsc = (Xsc - mu) / sd

    y_mean = float(ytr.mean())
    yc = ytr - y_mean
    return {"Zsc": Zsc, "ZZ": Ztr.T @ Ztr, "Zy": Ztr.T @ yc, "y_mean": y_mean}


def ridge_predict(prep: dict, selected: Sequence[str], alpha: float = RIDGE_ALPHA) -> np.ndarray:
    if not selected:
        return np.full(prep["Zsc"].shape[0], prep["y_mean"], dtype=float)
    ids = [ALL_FEATURES.index(f) for f in selected]
    A = prep["ZZ"][np.ix_(ids, ids)].copy()
    A.flat[:: len(ids) + 1] += alpha
    b = prep["Zy"][ids]
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ b
    return prep["y_mean"] + prep["Zsc"][:, ids] @ beta


def fold_threshold(ctx: DataContext, train_end: pd.Timestamp, representative_only: bool = False) -> float:
    p = ctx.power
    m = (p["timestamp"] <= train_end) & (~p["bad_time"])
    if representative_only:
        m &= p["source_date"].isin(ctx.representative_dates)
    return float(np.quantile(p.loc[m, "power"].to_numpy(float), 0.95))


def metrics(score: pd.DataFrame, pred: np.ndarray, tau: float) -> dict:
    y = score["y"].to_numpy(float)
    e = np.abs(y - pred)
    peak = y > tau
    return {
        "n": len(y),
        "peak_n": int(peak.sum()),
        "mae": float(e.mean()),
        "peak_mae": float(e[peak].mean()) if peak.any() else np.nan,
    }


def date_block_stats(score: pd.DataFrame, p0: np.ndarray, p1: np.ndarray, tau: float) -> pd.DataFrame:
    y = score["y"].to_numpy(float)
    dates = score["date"].astype(str).to_numpy()
    rows = []
    for d in pd.unique(dates):
        m = dates == d
        peak = y[m] > tau
        rows.append({
            "date": d,
            "n": int(m.sum()),
            "base_abs_sum": float(np.abs(y[m] - p0[m]).sum()),
            "alt_abs_sum": float(np.abs(y[m] - p1[m]).sum()),
            "peak_n": int(peak.sum()),
            "base_peak_abs_sum": float(np.abs(y[m][peak] - p0[m][peak]).sum()) if peak.any() else 0.0,
            "alt_peak_abs_sum": float(np.abs(y[m][peak] - p1[m][peak]).sum()) if peak.any() else 0.0,
        })
    return pd.DataFrame(rows)


def bootstrap_pooled_delta(blocks_by_fold: Sequence[pd.DataFrame], seed: int) -> dict:
    blocks = pd.concat([b.assign(fold=i) for i, b in enumerate(blocks_by_fold)], ignore_index=True)

    def calc(sample: pd.DataFrame) -> Tuple[float, float]:
        n = sample["n"].sum()
        d_mae = sample["base_abs_sum"].sum() / n - sample["alt_abs_sum"].sum() / n
        pn = sample["peak_n"].sum()
        d_peak = (
            sample["base_peak_abs_sum"].sum() / pn - sample["alt_peak_abs_sum"].sum() / pn
            if pn else np.nan
        )
        return float(d_mae), float(d_peak)

    d_mae, d_peak = calc(blocks)
    rng = np.random.default_rng(seed)
    mb, pb = [], []
    for _ in range(BOOTSTRAP_N):
        sample = blocks.iloc[rng.integers(0, len(blocks), size=len(blocks))]
        dm, dp = calc(sample)
        mb.append(dm)
        if np.isfinite(dp):
            pb.append(dp)

    return {
        "d_mae": d_mae,
        "d_mae_ci_low": float(np.quantile(mb, 0.025)),
        "d_mae_ci_high": float(np.quantile(mb, 0.975)),
        "d_peak_mae": d_peak,
        "d_peak_mae_ci_low": float(np.quantile(pb, 0.025)) if pb else np.nan,
        "d_peak_mae_ci_high": float(np.quantile(pb, 0.975)) if pb else np.nan,
        "days": int(len(blocks)),
        "n": int(blocks["n"].sum()),
        "peak_n": int(blocks["peak_n"].sum()),
    }


def run_forward(ctx: DataContext) -> pd.DataFrame:
    defs = {
        "G0": ([], GROUPS["G0"]),
        "G1": (GROUPS["G0"], GROUPS["G0"] + GROUPS["G1"]),
        "G2": (GROUPS["G0"] + GROUPS["G1"], CORE_FEATURES),
        "G3": (CORE_FEATURES, CORE_FEATURES + GROUPS["G3"]),
        "G4": (CORE_FEATURES, CORE_FEATURES + GROUPS["G4"]),
        "G5": (CORE_FEATURES, CORE_FEATURES + GROUPS["G5"]),
        "G7": (CORE_FEATURES, CORE_FEATURES + GROUPS["G7"]),
    }
    rows = []
    for h in range(60, 241, 15):
        data = build_samples(ctx, h)
        for group, (base, alt) in defs.items():
            blocks, pos_m, pos_p = [], 0, 0
            for fold in FOLDS:
                train = data[data["target"] <= fold["train_end"]]
                score = data[(data["target"] >= fold["score_start"]) & (data["target"] <= fold["score_end"])]
                tau = fold_threshold(ctx, fold["train_end"])
                prep = ridge_prepare(train, score)
                p0, p1 = ridge_predict(prep, base), ridge_predict(prep, alt)
                m0, m1 = metrics(score, p0, tau), metrics(score, p1, tau)
                pos_m += int(m0["mae"] - m1["mae"] > 0)
                pos_p += int(m0["peak_mae"] - m1["peak_mae"] > 0)
                blocks.append(date_block_stats(score, p0, p1, tau))
            rows.append({"horizon_min": h, "group": group, "analysis": "forward", **bootstrap_pooled_delta(blocks, SEED + h + ord(group[-1])), "positive_mae_folds": pos_m, "positive_peak_folds": pos_p})
    return pd.DataFrame(rows)


def run_lofo(ctx: DataContext) -> pd.DataFrame:
    rows = []
    full = ALL_FEATURES
    for h in range(60, 241, 15):
        data = build_samples(ctx, h)
        for group, gf in GROUPS.items():
            reduced = [f for f in full if f not in gf]
            blocks, pos_m, pos_p = [], 0, 0
            for fold in FOLDS:
                train = data[data["target"] <= fold["train_end"]]
                score = data[(data["target"] >= fold["score_start"]) & (data["target"] <= fold["score_end"])]
                tau = fold_threshold(ctx, fold["train_end"])
                prep = ridge_prepare(train, score)
                p0, p1 = ridge_predict(prep, reduced), ridge_predict(prep, full)
                m0, m1 = metrics(score, p0, tau), metrics(score, p1, tau)
                pos_m += int(m0["mae"] - m1["mae"] > 0)
                pos_p += int(m0["peak_mae"] - m1["peak_mae"] > 0)
                blocks.append(date_block_stats(score, p0, p1, tau))
            rows.append({"horizon_min": h, "group": group, "analysis": "lofo", **bootstrap_pooled_delta(blocks, 777 + h + ord(group[-1])), "positive_mae_folds": pos_m, "positive_peak_folds": pos_p})
    return pd.DataFrame(rows)


def run_old_d2_anchor(ctx: DataContext) -> pd.DataFrame:
    defs = {
        "G0": ([], GROUPS["G0"]),
        "G1": (GROUPS["G0"], GROUPS["G0"] + GROUPS["G1"]),
        "G2": (GROUPS["G0"] + GROUPS["G1"], CORE_FEATURES),
        "G3": (CORE_FEATURES, CORE_FEATURES + GROUPS["G3"]),
        "G4": (CORE_FEATURES, CORE_FEATURES + GROUPS["G4"]),
        "G5": (CORE_FEATURES, CORE_FEATURES + GROUPS["G5"]),
        "G7": (CORE_FEATURES, CORE_FEATURES + GROUPS["G7"]),
    }
    rows = []
    for h in (60, 120, 180, 240):
        data = build_samples(ctx, h, d2_old=True)
        for group, (base, alt) in defs.items():
            bsum = asum = bpsum = apsum = 0.0
            n = pn = pos_m = pos_p = 0
            for fold in FOLDS:
                train = data[data["target"] <= fold["train_end"]]
                score = data[(data["target"] >= fold["score_start"]) & (data["target"] <= fold["score_end"])]
                if len(train) < 100 or len(score) < 20:
                    continue
                tau = fold_threshold(ctx, fold["train_end"], representative_only=True)
                prep = ridge_prepare(train, score)
                p0, p1 = ridge_predict(prep, base), ridge_predict(prep, alt)
                m0, m1 = metrics(score, p0, tau), metrics(score, p1, tau)
                pos_m += int(m0["mae"] - m1["mae"] > 0)
                pos_p += int(m0["peak_mae"] - m1["peak_mae"] > 0)
                y = score["y"].to_numpy(float)
                e0, e1 = np.abs(y - p0), np.abs(y - p1)
                peak = y > tau
                bsum += e0.sum(); asum += e1.sum(); n += len(y)
                bpsum += e0[peak].sum(); apsum += e1[peak].sum(); pn += int(peak.sum())
            rows.append({
                "horizon_min": h,
                "group": group,
                "analysis": "old_d2_representative_profile_forward",
                "d_mae": bsum / n - asum / n,
                "d_peak_mae": bpsum / pn - apsum / pn if pn else np.nan,
                "positive_mae_folds": pos_m,
                "positive_peak_folds": pos_p,
                "n": n,
                "peak_n": pn,
            })
    return pd.DataFrame(rows)


def descriptive_characteristics(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["평균", "생산량", "기온", "풍속", "습도", "강수량", "전기요금(계절)", "day", "d", "m", "공장인원", "인건비"]
    rows = []
    for c in cols:
        v = pd.to_numeric(raw[c], errors="coerce")
        rows.append({
            "variable": c,
            "n": int(v.notna().sum()),
            "missing": int(v.isna().sum()),
            "unique": int(v.nunique(dropna=True)),
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "corr_with_avg_power": 1.0 if c == "평균" else float(raw[[c, "평균"]].corr().iloc[0, 1]),
            "corr_with_production": 1.0 if c == "생산량" else float(raw[[c, "생산량"]].corr().iloc[0, 1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/raw/task05_power/okm_augumented_2021.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/phase_b_temp"))
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    actual_sha = sha256_file(args.input)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(f"Raw SHA-256 mismatch. expected={EXPECTED_SHA256} actual={actual_sha}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_and_prepare_raw(args.input)
    reps, unique_profiles, duplicate_surplus = exact_profile_representatives(raw)
    p15 = expand_15min(raw)
    ctx = build_context(raw, p15, reps)

    characteristics = descriptive_characteristics(raw)
    forward = run_forward(ctx)
    lofo = run_lofo(ctx)
    d2 = run_old_d2_anchor(ctx)

    characteristics.to_csv(args.outdir / "temp_phase_b_characteristics.csv", index=False, encoding="utf-8-sig")
    forward.to_csv(args.outdir / "temp_phase_b_forward.csv", index=False, encoding="utf-8-sig")
    lofo.to_csv(args.outdir / "temp_phase_b_lofo.csv", index=False, encoding="utf-8-sig")
    d2.to_csv(args.outdir / "temp_phase_b_old_d2_anchor.csv", index=False, encoding="utf-8-sig")

    meta = {
        "status": "temporary_session_probe_reproduction",
        "raw_sha256": actual_sha,
        "raw_rows": int(len(raw)),
        "expanded_15min_rows": int(len(p15)),
        "development_boundary_exclusive": str(DEV_BOUNDARY),
        "horizons_minutes": list(range(60, 241, 15)),
        "probe_model": "standardized closed-form Ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_n": BOOTSTRAP_N,
        "feature_groups": GROUPS,
        "core_groups": ["G0", "G1", "G2"],
        "unique_daily_power_profiles": unique_profiles,
        "duplicate_profile_surplus_days": duplicate_surplus,
        "raw_average_integrity_fraction": rounded_four_mean_integrity(raw),
        "warning": "OLD D2 physically removes duplicate-profile dates. It was later replaced by a novel-profile score-subset definition and is not the final Phase B D2.",
    }
    (args.outdir / "temp_phase_b_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Temporary Phase B probe complete.")
    print(f"raw_sha256={actual_sha}")
    print(f"raw_rows={len(raw)}")
    print(f"15min_rows={len(p15)}")
    print(f"unique_profiles={unique_profiles}")
    print(f"duplicate_surplus={duplicate_surplus}")
    print(f"output_dir={args.outdir.resolve()}")
    print("IMPORTANT: OLD D2 output is historical/temporary, not the final D2 definition.")


if __name__ == "__main__":
    main()
