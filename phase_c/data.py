"""Sealed development data and exact Phase B core features for Phase C.

This module builds candidate arrays only. It does not fit a model, choose a
threshold from stop/calibration/score rows, or read a final-evaluation value.
The full source file is streamed solely to verify its pinned byte hash;
``load_development_history`` decodes only the sealed development prefix.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.features import build_features
from src.holidays import HOLIDAYS_2021
from src.session_data import SEALED_BOUNDARY, load_development_history
from src.split import make_splits
from src.targets import point_targets
from src.training import _partition


EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
SOURCE_RELATIVE = Path("data/raw/task05_power/okm_augumented_2021.csv")
HORIZONS = tuple(range(4, 17))
GROUPS = {
    "G0": ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", "holiday"),
    "G1": ("current", "lag4", "slot1d", "slot7d"),
    "G2": ("r4_mean", "r4_max", "r4_std", "r16_max", "r96_max"),
}
CORE_FEATURES = tuple(name for group in GROUPS.values() for name in group)
SEQUENCE_LENGTH = 96


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gap_hours(frame: pd.DataFrame) -> pd.Series:
    """Flag only complete source hours with four zeros and absent headcount."""
    interval_hour = (frame.index - pd.Timedelta(nanoseconds=1)).floor("h")
    power = pd.to_numeric(frame["power"], errors="coerce")
    head_missing = frame["headcount"].isna()
    group = pd.DataFrame({
        "hour": interval_hour,
        "zero": power.eq(0).to_numpy(dtype=bool),
        "head_missing": head_missing.to_numpy(dtype=bool),
    }, index=frame.index)
    by_hour = group.groupby("hour", sort=False).agg(
        slots=("zero", "size"),
        four_zero=("zero", "all"),
        erp_missing=("head_missing", "all"),
    )
    bad_hours = by_hour.index[
        by_hour["slots"].eq(4) & by_hour["four_zero"] & by_hour["erp_missing"]
    ]
    return pd.Series(interval_hour.isin(bad_hours), index=frame.index, name="gap_hour")


def _prepare_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("History needs a DatetimeIndex of interval-end timestamps")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("History timestamps must be unique and chronological")
    if frame.empty or frame.index.max() >= SEALED_BOUNDARY:
        raise ValueError("History is empty or crosses the sealed development boundary")
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq="15min", name=frame.index.name)):
        raise ValueError("History must cover a complete 15-minute grid")
    if not {"power", "headcount", "time_repaired"} <= set(frame.columns):
        raise ValueError("History lacks required power/headcount/repair fields")

    history = frame.copy()
    history["power_raw"] = pd.to_numeric(history["power"], errors="coerce")
    history["time_repaired"] = history["time_repaired"].fillna(True).astype(bool)
    history["erp_missing"] = history["headcount"].isna() | history["time_repaired"]
    history["gap_hour"] = _gap_hours(history)
    history["measurement_gap"] = history["gap_hour"]
    history["quality_bad"] = history["time_repaired"] | history["gap_hour"]
    history["power"] = history["power_raw"].mask(history["quality_bad"])
    history.attrs.update(frame.attrs)
    history.attrs["phase_c_power_cleaning"] = "time_repaired_or_complete_hour_four_zero_with_missing_headcount"
    audit = {
        "sealed_boundary": str(SEALED_BOUNDARY),
        "history_first": str(history.index.min()),
        "history_last": str(history.index.max()),
        "history_intervals": int(len(history)),
        "time_repaired_intervals": int(history["time_repaired"].sum()),
        "erp_missing_intervals": int(history["erp_missing"].sum()),
        "raw_zero_power_intervals": int(history["power_raw"].eq(0).sum()),
        "gap_hours": int(history.loc[history["gap_hour"]].index.to_series().sub(pd.Timedelta(nanoseconds=1)).dt.floor("h").nunique()),
        "gap_intervals": int(history["gap_hour"].sum()),
        "clean_power_finite_intervals": int(np.isfinite(history["power"]).sum()),
    }
    return history, audit


def load_history(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the pinned raw CSV hash, then load only its sealed prefix.

    The raw byte hash checks identity without decoding future fields. The
    sealed loader stops before 2021-08-09 09:45, including within the partial
    boundary hour. ``power_raw`` preserves observed values; ``power`` masks
    repaired timestamps and complete four-zero/missing-headcount gap hours.
    ``erp_missing`` and ``time_repaired`` are separate audit fields.
    """
    root = Path(root).resolve()
    source = root / SOURCE_RELATIVE
    config = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    if (root / config["source"]).resolve() != source.resolve():
        raise ValueError("Configured sealed loader source differs from pinned Phase B CSV")
    digest = _sha256_file(source)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Source CSV hash mismatch: expected {EXPECTED_SHA256}, got {digest}")
    history, audit = _prepare_history(load_development_history(root))
    audit.update({
        "raw_source": SOURCE_RELATIVE.as_posix(),
        "raw_sha256": digest,
        "source_mode": history.attrs.get("source_mode"),
        "boundary_partial_hour": bool(history.attrs.get("boundary_partial_hour")),
        "development_only": True,
    })
    return history, audit


def _core_features(
    history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Map source causal lag/rolling values to the 15 Phase B core features."""
    if history.get("gap_hour", pd.Series(False, index=history.index)).any():
        raise ValueError("Complete-hour gaps require an origin-specific quality-availability adapter; retrospective masking is forbidden for features")
    target = origins + pd.Timedelta(minutes=15 * horizon)
    # Deliberately ignore build_features.valid_mask: it includes production,
    # weather, CBL and extra lags outside this core-feature experiment.
    source, _ = build_features(history, origins, horizon, {"_include_cbl": False, "_tau": None})
    hour_float = target.hour.to_numpy() + target.minute.to_numpy() / 60.0
    sunday_dow = (target.dayofweek.to_numpy() + 1) % 7
    x = pd.DataFrame(index=origins)
    x["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    x["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    x["dow_sin"] = np.sin(2 * np.pi * sunday_dow / 7.0)
    x["dow_cos"] = np.cos(2 * np.pi * sunday_dow / 7.0)
    x["weekend"] = (target.dayofweek >= 5).astype(float)
    x["holiday"] = target.strftime("%Y-%m-%d").isin(HOLIDAYS_2021).astype(float)
    mapped = {
        "current": "current",
        "lag4": "lag_4",
        "slot1d": "target_slot_1d_ago",
        "slot7d": "target_slot_7d_ago",
        "r4_mean": "recent_4_mean",
        "r4_max": "recent_4_max",
        "r16_max": "recent_16_max",
        "r96_max": "recent_96_max",
    }
    for dest, name in mapped.items():
        x[dest] = source[name].to_numpy(dtype=float)
    # The Phase B probe used sample std (ddof=1); src.features uses ddof=0.
    x["r4_std"] = source["recent_4_std"].to_numpy(dtype=float) * np.sqrt(4 / 3)
    x = x.loc[:, CORE_FEATURES]

    calendar = pd.Series(pd.NaT, index=origins, dtype="datetime64[ns]")
    provenance = {name: calendar.copy() for name in GROUPS["G0"]}
    used_at = {
        "current": origins,
        "lag4": origins - pd.Timedelta(hours=1),
        "slot1d": target - pd.Timedelta(days=1),
        "slot7d": target - pd.Timedelta(days=7),
    }
    for name, timestamps in used_at.items():
        provenance[name] = pd.Series(timestamps, index=origins)
    for name in GROUPS["G2"]:
        provenance[name] = pd.Series(origins, index=origins)
    if any((used > origins).fillna(False).any() for used in provenance.values()):
        raise AssertionError("Core feature accesses an observation after its origin")
    return x, provenance


def _sequences(history: pd.DataFrame) -> pd.DataFrame:
    """Return chronological 96-slot windows, oldest column first."""
    power = pd.to_numeric(history["power"], errors="coerce").to_numpy(dtype=float)
    if len(power) < SEQUENCE_LENGTH:
        raise ValueError("Fewer than 96 development power intervals")
    windows = np.lib.stride_tricks.sliding_window_view(power, SEQUENCE_LENGTH)
    columns = [f"seq_{position:02d}" for position in range(SEQUENCE_LENGTH)]
    return pd.DataFrame(windows, index=history.index[SEQUENCE_LENGTH - 1:], columns=columns)


def _daily_profiles(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Hash complete clean interval days as little-endian float64 bytes."""
    interval_day = (history.index - pd.Timedelta(nanoseconds=1)).normalize()
    values = pd.to_numeric(history["power"], errors="coerce")
    rows = []
    incomplete = 0
    for day, block in values.groupby(interval_day, sort=True):
        expected = pd.date_range(day + pd.Timedelta(minutes=15), periods=96, freq="15min")
        if len(block) != 96 or not block.index.equals(expected) or not np.isfinite(block.to_numpy()).all():
            incomplete += 1
            continue
        raw_bytes = np.asarray(block.to_numpy(dtype=float), dtype="<f8").tobytes(order="C")
        rows.append({"day": day, "last_timestamp": expected[-1], "sha256": hashlib.sha256(raw_bytes).hexdigest()})
    profiles = pd.DataFrame(rows, columns=["day", "last_timestamp", "sha256"]).set_index("day")
    return profiles, {"complete_profile_days": len(profiles), "incomplete_profile_days": incomplete}


def _d2_score_mask(
    profiles: pd.DataFrame,
    target_time: pd.Series,
    fit: pd.DatetimeIndex,
    score: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, int]]:
    """Retrospective score subset: complete profile unseen in fit history.

    The full target interval day is consulted only for this score-subset flag.
    It is never a model input and does not remove fit, stop, or calibration
    rows. Training reference hashes must end by max(fit target_time).
    """
    fitted_through = pd.Timestamp(target_time.loc[fit].max())
    known = set(profiles.loc[profiles["last_timestamp"] <= fitted_through, "sha256"])
    score_day = (pd.DatetimeIndex(target_time.loc[score]) - pd.Timedelta(nanoseconds=1)).normalize()
    score_hash = profiles["sha256"].reindex(score_day)
    novel = score_hash.notna().to_numpy() & ~score_hash.isin(known).to_numpy()
    mask = pd.Series(False, index=target_time.index, dtype=bool, name="d2")
    mask.loc[score] = novel
    return mask, {
        "d2_fit_profile_hashes": len(known),
        "d2_score_complete": int(score_hash.notna().sum()),
        "d2_score_incomplete": int(score_hash.isna().sum()),
        "d2_score_novel": int(novel.sum()),
    }


def build_contexts(history: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    """Build 13 horizons × 3 purged development folds without model fitting.

    Splits use the full sealed 15-minute grid. Only afterwards are rows with a
    clean target, all 15 finite core features and a complete 96-slot causal
    history retained. Fold keys are ``(horizon_quarters, zero_based_fold)``.
    Each context contains ``x``, ``y``, ``target_time``, ``seq``, ``fit``,
    ``stop``, ``cal``, ``score``, fit-only ``tau``, a score-only ``d2`` mask,
    ``summary``, and per-feature timestamp ``provenance``.
    """
    if "power" not in history:
        raise ValueError("History must have a clean power column")
    history, _ = _prepare_history(history) if "quality_bad" not in history else (history, None)
    if history["gap_hour"].any():
        raise ValueError("Retrospective gap masks cannot enter forecast features; fail closed pending origin-specific quality handling")
    all_origins = pd.DatetimeIndex(history.index, name="origin")
    if all_origins.max() >= SEALED_BOUNDARY:
        raise ValueError("History reaches the sealed boundary")
    all_sequences = _sequences(history)
    sequence_valid = pd.Series(np.isfinite(all_sequences.to_numpy()).all(axis=1), index=all_sequences.index)
    profiles, profile_summary = _daily_profiles(history)
    contexts = {}

    for horizon in HORIZONS:
        folds, frozen = make_splits(all_origins, horizon, dev_frac=1.0, n_folds=3)
        if len(frozen):
            raise AssertionError("A frozen test origin entered development splits")
        x_all, provenance_all = _core_features(history, all_origins, horizon)
        targets = point_targets(history, all_origins, horizon)
        target_before_boundary = pd.DatetimeIndex(targets["target_time"]).to_numpy() < np.datetime64(SEALED_BOUNDARY)
        eligible = (
            x_all.notna().all(axis=1).to_numpy()
            & targets["valid_target"].to_numpy(dtype=bool)
            & target_before_boundary
            & sequence_valid.reindex(all_origins, fill_value=False).to_numpy(dtype=bool)
        )
        eligible_origins = all_origins[eligible]
        x = x_all.loc[eligible_origins]
        y = targets.loc[eligible_origins, "y"].rename("y")
        target_time = targets.loc[eligible_origins, "target_time"]
        seq = all_sequences.loc[eligible_origins]
        provenance = {name: used.loc[eligible_origins] for name, used in provenance_all.items()}

        for fold_id, fold in enumerate(folds):
            train = fold.train.intersection(eligible_origins)
            validation = fold.validation.intersection(eligible_origins)
            fit, stop, cal, score = _partition(train, validation, horizon)
            tau = float(np.quantile(y.loc[fit].to_numpy(dtype=float), .95))
            d2, d2_summary = _d2_score_mask(profiles, target_time, fit, score)
            gap = pd.Timedelta(minutes=15 * horizon)
            if not (target_time.loc[fit].max() < stop.min()
                    and target_time.loc[stop].max() < cal.min()
                    and target_time.loc[cal].max() < score.min()):
                raise AssertionError("Target-time embargo failed")
            summary = {
                "horizon_quarters": horizon,
                "horizon_minutes": 15 * horizon,
                "fold": fold_id,
                "full_grid_origins": len(all_origins),
                "eligible_origins": len(eligible_origins),
                "fit_count": len(fit),
                "stop_count": len(stop),
                "cal_count": len(cal),
                "score_count": len(score),
                "fit_first": str(fit.min()),
                "fit_last_target": str(target_time.loc[fit].max()),
                "stop_first": str(stop.min()),
                "cal_first": str(cal.min()),
                "score_first": str(score.min()),
                "score_last_target": str(target_time.loc[score].max()),
                "embargo_minutes": int(gap / pd.Timedelta(minutes=1)),
                **profile_summary,
                **d2_summary,
            }
            contexts[(horizon, fold_id)] = {
                "x": x,
                "y": y,
                "target_time": target_time,
                "seq": seq,
                "fit": fit,
                "stop": stop,
                "cal": cal,
                "score": score,
                "tau": tau,
                "d2": d2,
                "summary": summary,
                "provenance": provenance,
            }
    return contexts
