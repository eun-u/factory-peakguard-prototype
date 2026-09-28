"""Shared fold contexts, row construction and caching for Phase 4."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.research_cv import build_contexts
from src.session_data import SEALED_BOUNDARY, load_development_history
from src.training import _cutoff, _row

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 4, 16, 96)
HOLIDAY_COLUMNS = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")
CBL_REP = {1: "c2a_max_4_5_adjusted", 4: "c2a_max_4_5_adjusted",
           16: "c3_holiday_hybrid", 96: "c3_holiday_hybrid"}
INCUMBENT = {1: "p1_latest", 4: "lgbm_no_holiday_weight_2",
             16: "lgbm_residual_cbl", 96: "c3_holiday_hybrid"}
CACHE = ROOT / "_validation" / "p4_cache"
OUT = ROOT / "outputs" / "p4"


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))


def history() -> pd.DataFrame:
    df = load_development_history(ROOT)
    if df.index.max() >= SEALED_BOUNDARY:
        raise AssertionError("Sealed boundary violated")
    return df


def contexts(df: pd.DataFrame, cfg: dict) -> dict:
    """(horizon, fold) -> original CV context, verified against the manifest."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "contexts.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    manifest = json.loads((ROOT / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    ctx = build_contexts(df, cfg, manifest, horizons=HORIZONS)
    path.write_bytes(pickle.dumps(ctx))
    return ctx


def feature_columns(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if c not in HOLIDAY_COLUMNS]


def safe_power(df: pd.DataFrame) -> pd.Series:
    repaired = df.get("time_repaired", pd.Series(False, index=df.index)).fillna(True).astype(bool)
    return pd.to_numeric(df["power"], errors="coerce").mask(repaired)


def make_rows(ctx: dict, horizon: int, fold: int, name: str,
              pred_cal: np.ndarray, pred_score: np.ndarray, **extra) -> pd.DataFrame:
    """Cutoff on calibration rows only (existing rule), score rows in OOF schema."""
    targets, tau, cal, score = ctx["targets"], ctx["tau"], ctx["cal"], ctx["score"]
    y_cal = targets.loc[cal, "y"].to_numpy(dtype=float)
    pc = np.asarray(pred_cal, dtype=float)
    ok = np.isfinite(pc) & np.isfinite(y_cal)
    if ok.sum() < 30:
        cutoff = float("inf")
    else:
        cutoff = _cutoff(y_cal[ok], pc[ok], tau, targets.loc[cal[ok], "target_time"])
    return _row(score, horizon, fold, name, targets, tau, np.asarray(pred_score, dtype=float), cutoff, **extra)


def save(rows: list[pd.DataFrame], name: str, meta: dict | None = None) -> Path:
    OUT.mkdir(exist_ok=True)
    frame = pd.concat(rows, ignore_index=True)
    path = OUT / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    if meta is not None:
        (OUT / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def __call__(self):
        return round(time.perf_counter() - self.start, 1)
