"""Frozen FIT-day analogs for EXPLORE point forecasts.

The case library is built once per weekly fold from common, embargoed FIT
origins. STOP chooses only a fixed R1/analog mixing weight; CAL and SCORE
provide query histories but never enter the library or its checkpoint identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from phase_f.features_ext import _power
from phase_f.harness import make_frame
from phase_f.models.r1_residual import (
    HORIZONS, ROLES, _common_roles, _digest_frame, _path_values, _paths_index,
)
from phase_f.registry import config_hash, sha256, write_json


_SLOT = pd.Timedelta(minutes=15)
_ID = re.compile(r"[A-Za-z0-9_.+-]+\Z")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PREFIX_LENGTHS = (16, 96)
TOP_K = (3, 5)
BANK_DAYS = 56
_EPSILON = 1e-6  # Physical power unit is unconfirmed; this is a numerical floor.
_SOURCES = (
    "phase_f/models/day_analog.py", "phase_f/models/r1_residual.py",
    "phase_f/features_ext.py", "phase_f/harness.py", "phase_f/registry.py",
    "phase_c/data.py", "src/session_data.py",
)


def _prefixes(history: pd.DataFrame, origins: pd.DatetimeIndex,
              length: int) -> np.ndarray:
    """Read exact preceding quarter-hour timestamps, never positional rows."""
    power = _power(history)
    origins = pd.DatetimeIndex(origins)
    times = origins.asi8[:, None] - (
        np.arange(length - 1, -1, -1, dtype=np.int64)[None, :]
        * int(_SLOT.value))
    positions = power.index.get_indexer(pd.DatetimeIndex(times.ravel())).reshape(times.shape)
    values = np.full(times.shape, np.nan, dtype=np.float64)
    valid = positions >= 0
    values[valid] = power.to_numpy(dtype=np.float64)[positions[valid]]
    return values


def _slots(origins: pd.DatetimeIndex) -> np.ndarray:
    origins = pd.DatetimeIndex(origins)
    if not ((origins.minute % 15 == 0) & (origins.second == 0)
            & (origins.microsecond == 0) & (origins.nanosecond == 0)).all():
        raise ValueError("Analog origins must be exact interval-end quarters")
    return (origins.hour.to_numpy(dtype=np.int16) * 4
            + origins.minute.to_numpy(dtype=np.int16) // 15)


def _array_digest(parts: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(parts.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bank_digest(bank: dict) -> str:
    return _array_digest({name: bank[name] for name in
                          ("origins_ns", "slots", "prefixes", "levels", "suffixes")})


def _build_bank(view, fold: int, fit: pd.DatetimeIndex,
                length: int) -> tuple[dict, dict]:
    fit = pd.DatetimeIndex(fit)
    if fit.empty or fit.has_duplicates or not fit.is_monotonic_increasing:
        raise ValueError("Analog requires chronological, unique common FIT origins")
    # Inclusive timestamp cutoff is frozen relative to the pooled FIT end.
    recent = fit[fit >= fit.max() - pd.Timedelta(days=BANK_DAYS)]
    prefix = _prefixes(view.history, recent, length)
    clean_prefix = np.isfinite(prefix).all(axis=1)
    masked_power = _power(view.history)
    targets = []
    finite_context_y = np.ones(len(recent), dtype=bool)
    clean_target_power = np.ones(len(recent), dtype=bool)
    for h in HORIZONS:
        context = view.contexts[(h, fold)]
        target_time = pd.DatetimeIndex(context["target_time"].loc[recent])
        expected = recent + h * _SLOT
        if not target_time.equals(expected):
            raise ValueError(f"Analog FIT target timestamp changed for h={h}")
        if not (target_time < pd.DatetimeIndex(context["stop"]).min()).all():
            raise ValueError("Analog FIT case target crosses the STOP embargo")
        y = context["y"].loc[recent].to_numpy(dtype=np.float64)
        finite_context_y &= np.isfinite(y)
        clean_target_power &= np.isfinite(
            masked_power.reindex(target_time).to_numpy(dtype=np.float64))
        targets.append(y)
    keep = clean_prefix & finite_context_y & clean_target_power
    selected = recent[keep]
    values = prefix[keep]
    bank = {
        "origins_ns": selected.asi8.copy(),
        "slots": _slots(selected).copy(),
        "prefixes": np.ascontiguousarray(values),
        "levels": np.ascontiguousarray(values[:, -1]),
        "suffixes": np.ascontiguousarray(np.column_stack(targets)[keep]),
    }
    if not np.isfinite(bank["prefixes"]).all() or not np.isfinite(bank["suffixes"]).all():
        raise AssertionError("Analog library contains a nonfinite prefix or FIT suffix")
    audit = {"common_fit_origins": len(fit), "recent_fit_origins": len(recent),
             "bank_cases": len(selected),
             "excluded_quality_or_missing_prefix": int((~clean_prefix).sum()),
             "excluded_nonfinite_fit_target": int((~finite_context_y).sum()),
             "excluded_bad_or_missing_target_observation": int((~clean_target_power).sum()),
             "bank_first_origin": str(selected.min()) if len(selected) else None,
             "bank_last_origin": str(selected.max()) if len(selected) else None,
             "bank_digest": _bank_digest(bank),
             "bank_days_cutoff": str(fit.max() - pd.Timedelta(days=BANK_DAYS)),
             "same_slot_case_deduplication": False}
    return bank, audit


def _rank(history: pd.DataFrame, bank: dict, origins: pd.DatetimeIndex,
          length: int, top_k: int, *, role: str,
          horizon: int | None = None) -> dict:
    """Rank once per query; FIT tests use a target-matured past-only library."""
    origins = pd.DatetimeIndex(origins)
    if role not in ROLES:
        raise ValueError("Unrecognized analog query role")
    if role == "fit" and horizon not in HORIZONS:
        raise ValueError("FIT analog queries require a horizon for target maturity")
    queries = _prefixes(history, origins, length)
    candidate = np.full((len(origins), top_k), -1, dtype=np.int32)
    distances = np.full((len(origins), top_k), np.nan, dtype=np.float64)
    support = np.zeros(len(origins), dtype=np.int32)
    finite = np.isfinite(queries).all(axis=1)
    query_slots = _slots(origins)
    bank_shape = bank["prefixes"] - bank["prefixes"].mean(axis=1, keepdims=True)
    for i in np.flatnonzero(finite):
        eligible = np.flatnonzero(bank["slots"] == query_slots[i])
        if role == "fit":
            eligible = eligible[
                (bank["origins_ns"][eligible] < origins.asi8[i])
                & (bank["origins_ns"][eligible]
                   + int(horizon * _SLOT.value) <= origins.asi8[i])]
        elif len(eligible) and (bank["origins_ns"][eligible]
                                + int(max(HORIZONS) * _SLOT.value) > origins.asi8[i]).any():
            raise ValueError("Analog case target was not mature by query origin")
        support[i] = len(eligible)
        if len(eligible) < top_k:
            continue
        query_shape = queries[i] - queries[i].mean()
        distance = np.abs(bank_shape[eligible] - query_shape).mean(axis=1)
        order = np.lexsort((-bank["origins_ns"][eligible], distance))[:top_k]
        candidate[i] = eligible[order]
        distances[i] = distance[order]
    return {"origins": origins, "indices": candidate, "distances": distances,
            "support": support, "query_levels": queries[:, -1],
            "query_prefix_clean": finite}


def _analog_points(bank: dict, ranked: dict, h: int) -> np.ndarray:
    indices = ranked["indices"]
    valid = (indices >= 0).all(axis=1)
    points = np.full(len(indices), np.nan, dtype=np.float64)
    if not valid.any():
        return points
    selected = indices[valid]
    shifted = (bank["suffixes"][selected, h - min(HORIZONS)]
               + ranked["query_levels"][valid, None] - bank["levels"][selected])
    shifted = np.maximum(0.0, shifted)
    weights = 1.0 / np.maximum(ranked["distances"][valid], _EPSILON)
    order = np.argsort(shifted, axis=1, kind="stable")
    sorted_values = np.take_along_axis(shifted, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = sorted_weights.cumsum(axis=1)
    position = (cumulative >= sorted_weights.sum(axis=1, keepdims=True) / 2).argmax(axis=1)
    points[valid] = sorted_values[np.arange(len(position)), position]
    return points


def _blend(r1: np.ndarray, analog: np.ndarray, alpha: float) -> np.ndarray:
    baseline = np.asarray(r1, dtype=np.float64)
    if not np.isfinite(baseline).all() or not 0 <= alpha <= 1:
        raise ValueError("Analog requires finite signed R1 forecasts and alpha")
    forecast = baseline.copy()
    valid = np.isfinite(analog)
    forecast[valid] = np.maximum(0.0, (1 - alpha) * baseline[valid] + alpha * analog[valid])
    return forecast


def _choose_alpha(truth: np.ndarray, r1: np.ndarray, analog: np.ndarray,
                  peaks: np.ndarray) -> tuple[float, dict]:
    if (not len(truth) or not np.isfinite(truth).all()
            or not np.isfinite(r1).all() or not peaks.any()):
        raise ValueError("STOP needs finite truth/R1 and observed peak cases")
    metrics = {}
    for alpha in ALPHAS:
        forecast = _blend(r1, analog, alpha)
        error = np.abs(truth - forecast)
        mae = float(error.mean())
        peak = float(error[peaks].mean())
        metrics[str(alpha)] = {"MAE": mae, "PeakMAE": peak,
                               "objective": mae + .25 * peak}
    winner = min(ALPHAS, key=lambda value: (metrics[str(value)]["objective"], value))
    return winner, metrics


def _runtime() -> dict:
    return {"python": sys.version, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__}


def _identity(view, spec: dict, paths_sha: str, fold: int,
              fit: pd.DatetimeIndex, stop: pd.DatetimeIndex,
              bank: dict, bank_audit: dict) -> str:
    root = Path(view.root)
    roles = {f"h{h}_{role}": _digest_frame(pd.DataFrame({
        "origin": view.contexts[(h, fold)][role]}))
        for h in HORIZONS for role in ROLES}
    fit_targets = {}
    stop_targets = {}
    stop_prefix = _prefixes(view.history, stop, int(spec["prefix_length"]))
    stop_prefix_digest = _array_digest({"origins_ns": stop.asi8,
                                        "masked_prefixes": stop_prefix})
    for h in HORIZONS:
        context = view.contexts[(h, fold)]
        fit_targets[str(h)] = _digest_frame(pd.DataFrame({
            "y": context["y"].loc[fit].to_numpy(float),
            "target_time": pd.DatetimeIndex(context["target_time"].loc[fit])}))
        stop_targets[str(h)] = _digest_frame(pd.DataFrame({
            "y": context["y"].loc[stop].to_numpy(float),
            "target_time": pd.DatetimeIndex(context["target_time"].loc[stop])}))
    # The signed normalized paths digest covers R1 in every role; STOP values
    # are additionally fixed through the same digest, without SCORE outcomes.
    return config_hash({"spec": spec, "normalized_paths_sha256": paths_sha,
                        "fold": fold, "roles": roles,
                        "pooled_fit": fit.asi8.tolist(),
                        "pooled_stop": stop.asi8.tolist(),
                        "fit_targets": fit_targets, "stop_targets": stop_targets,
                        "stop_masked_query_prefix_sha256": stop_prefix_digest,
                        "taus": {str(h): float(view.contexts[(h, fold)]["tau"])
                                 for h in HORIZONS},
                        "bank_digest": _bank_digest(bank),
                        "bank_audit": bank_audit,
                        "split_sha256": view.split_lock["lock_sha256"],
                        "raw_sha256": view.seal["raw_sha256"],
                        "runtime": _runtime(),
                        "sources": {name: sha256(root / name) for name in _SOURCES}})


def _preserve(path: Path) -> Path | None:
    if not path.exists():
        return None
    digest = sha256(path)[:12]
    target = path.with_name(f"{path.stem}.orphan-{digest}{path.suffix}")
    number = 0
    while target.exists():
        number += 1
        target = path.with_name(f"{path.stem}.orphan-{digest}-{number}{path.suffix}")
    path.rename(target)
    return target


def _reject(view, model_path: Path, meta_path: Path,
            identity: str, reason: str) -> None:
    model_sha = sha256(model_path) if model_path.is_file() else None
    meta_sha = sha256(meta_path) if meta_path.is_file() else None
    model_saved = _preserve(model_path)
    meta_saved = _preserve(meta_path)
    if ((model_saved and sha256(model_saved) != model_sha)
            or (meta_saved and sha256(meta_saved) != meta_sha)):
        raise RuntimeError("Analog checkpoint changed while preserving corrupt bytes")
    incident = Path(view.out) / "logs/checkpoint_corruption" / \
        f"{model_path.stem}-{config_hash([identity, model_sha, meta_sha, reason])[:20]}.json"
    write_json(incident, {"reason": reason, "expected_identity": identity,
                          "model_original": str(model_path),
                          "model_sha256": model_sha,
                          "model_preserved": str(model_saved) if model_saved else None,
                          "metadata_original": str(meta_path),
                          "metadata_sha256": meta_sha,
                          "metadata_preserved": str(meta_saved) if meta_saved else None,
                          "refit_performed": False}, exclusive=True)
    raise RuntimeError(f"Analog checkpoint rejected ({reason}); preserved physical evidence: {incident}")


def _checkpoint_paths(view, spec: dict, fold: int) -> tuple[Path, Path]:
    directory = Path(view.out) / "models/day_analog" / spec["id"]
    model_path = directory / f"f{fold}.joblib"
    return model_path, model_path.with_suffix(".json")


def _read_checkpoint(view, model_path: Path, meta_path: Path,
                     identity: str, bank_audit: dict, spec: dict,
                     fold: int, stop_selection: dict) -> tuple[dict, dict]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _reject(view, model_path, meta_path, identity, "metadata_json_corrupt")
    if not isinstance(meta, dict) or meta.get("identity") != identity:
        _reject(view, model_path, meta_path, identity, "identity_changed")
    alpha = meta.get("alpha")
    if (meta.get("spec") != spec or meta.get("fold") != fold
            or meta.get("n_stochastic_seeds") != 0
            or meta.get("n_deterministic_runs") != 1
            or type(alpha) not in (int, float) or alpha not in ALPHAS
            or alpha != stop_selection["alpha"]
            or meta.get("stop_objectives") != stop_selection["objectives"]
            or meta.get("stop_peak_count") != stop_selection["peak_count"]
            or meta.get("stop_query_count") != stop_selection["query_count"]
            or meta.get("stop_fallback_count") != stop_selection["fallback_count"]
            or meta.get("stop_analog_coverage") != stop_selection["analog_coverage"]
            or meta.get("bank_audit") != bank_audit
            or meta.get("fit_library_frozen_through_score") is not True):
        _reject(view, model_path, meta_path, identity, "stop_selection_or_spec_changed")
    if meta.get("model_sha256") != sha256(model_path):
        _reject(view, model_path, meta_path, identity, "model_sha256_changed")
    try:
        bundle = joblib.load(model_path)
    except Exception:
        _reject(view, model_path, meta_path, identity, "payload_unreadable")
    if (not isinstance(bundle, dict) or set(bundle) != {"bank", "alpha", "spec"}
            or not isinstance(bundle["bank"], dict)
            or set(bundle["bank"]) != {"origins_ns", "slots", "prefixes", "levels", "suffixes"}
            or bundle["spec"] != spec
            or type(bundle["alpha"]) not in (int, float)
            or bundle["alpha"] not in ALPHAS
            or bundle["alpha"] != stop_selection["alpha"]):
        _reject(view, model_path, meta_path, identity, "payload_shape_changed")
    try:
        actual_digest = _bank_digest(bundle["bank"])
    except Exception:
        _reject(view, model_path, meta_path, identity, "bank_payload_unreadable")
    if actual_digest != bank_audit["bank_digest"] or actual_digest != meta.get("bank_digest"):
        _reject(view, model_path, meta_path, identity, "bank_payload_changed")
    return bundle, meta


def _stop_selection(view, spec: dict, paths: pd.DataFrame,
                    fold: int, stop: pd.DatetimeIndex,
                    bank: dict) -> dict:
    rank = _rank(view.history, bank, stop, spec["prefix_length"],
                 spec["top_k"], role="stop")
    truth, r1, analog, peak = [], [], [], []
    for h in HORIZONS:
        context = view.contexts[(h, fold)]
        truth.append(context["y"].loc[stop].to_numpy(dtype=np.float64))
        r1.append(_path_values(paths, stop, h).r1.to_numpy(dtype=np.float64))
        analog.append(_analog_points(bank, rank, h))
        peak.append(truth[-1] > float(context["tau"]))
    alpha, objectives = _choose_alpha(np.concatenate(truth), np.concatenate(r1),
                                      np.concatenate(analog), np.concatenate(peak))
    return {"alpha": float(alpha), "objectives": objectives,
            "peak_count": int(np.concatenate(peak).sum()),
            "query_count": len(stop),
            "analog_coverage": float((rank["indices"] >= 0).all(axis=1).mean()),
            "fallback_count": int((rank["indices"] < 0).any(axis=1).sum())}


def _fit_or_load(view, spec: dict, paths_sha: str, fold: int,
                 fit: pd.DatetimeIndex, stop: pd.DatetimeIndex,
                 bank: dict, bank_audit: dict, paths: pd.DataFrame) -> tuple[dict, dict]:
    start = perf_counter()
    model_path, meta_path = _checkpoint_paths(view, spec, fold)
    identity = _identity(view, spec, paths_sha, fold, fit, stop, bank, bank_audit)
    selection = _stop_selection(view, spec, paths, fold, stop, bank)
    if model_path.exists() != meta_path.exists():
        _reject(view, model_path, meta_path, identity, "checkpoint_pair_incomplete")
    if model_path.exists():
        bundle, meta = _read_checkpoint(view, model_path, meta_path,
                                        identity, bank_audit,
                                        spec, fold, selection)
        return bundle, {**meta, "cache_reused": True}
    alpha, objectives = selection["alpha"], selection["objectives"]
    bundle = {"bank": bank, "alpha": float(alpha), "spec": dict(spec)}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp = model_path.with_suffix(".joblib.tmp")
    if temp.exists():
        _reject(view, temp, temp.with_suffix(".json"), identity, "stale_temporary_checkpoint")
    joblib.dump(bundle, temp)
    os.replace(temp, model_path)
    metadata = {"identity": identity, "model_sha256": sha256(model_path),
                "bank_digest": bank_audit["bank_digest"], "bank_audit": bank_audit,
                "spec": dict(spec), "fold": fold, "alpha": float(alpha),
                "stop_objectives": objectives,
                "stop_peak_count": selection["peak_count"],
                "stop_query_count": selection["query_count"],
                "stop_analog_coverage": selection["analog_coverage"],
                "stop_fallback_count": selection["fallback_count"],
                "train_seconds": perf_counter() - start,
                "n_stochastic_seeds": 0, "n_deterministic_runs": 1,
                "fit_library_frozen_through_score": True,
                "cache_reused": False}
    write_json(meta_path, metadata)
    return bundle, metadata


def _future_probe(view, bundle: dict, spec: dict, paths: pd.DataFrame,
                  fold: int) -> dict:
    context = view.contexts[(16, fold)]
    origin = pd.Timestamp(context["score"][0])
    query = pd.DatetimeIndex([origin])
    bank = bundle["bank"]
    before = _rank(view.history, bank, query, spec["prefix_length"],
                   spec["top_k"], role="score")
    altered = view.history.copy()
    future = altered.index > origin
    if "power" not in altered or not future.any():
        raise ValueError("Analog future perturbation has no observable future power")
    altered.loc[future, "power"] = 10000.0
    touched = ["power"]
    for flag in ("quality_bad", "time_repaired"):
        if flag in altered:
            altered.loc[future, flag] = ~altered.loc[future, flag].fillna(True).astype(bool)
            touched.append(flag)
    after = _rank(altered, bank, query, spec["prefix_length"],
                  spec["top_k"], role="score")
    r1 = _path_values(paths, query, 16).r1.to_numpy(dtype=np.float64)
    first_analog = _analog_points(bank, before, 16)
    second_analog = _analog_points(bank, after, 16)
    first = _blend(r1, first_analog, float(bundle["alpha"]))
    second = _blend(r1, second_analog, float(bundle["alpha"]))
    difference = float(np.max(np.abs(first - second)))
    if (difference != 0
            or not np.array_equal(first_analog, second_analog, equal_nan=True)
            or not np.array_equal(before["indices"], after["indices"])):
        raise AssertionError("Future observed power or quality changed analog prediction")
    return {"future_perturbation_max_abs_difference": difference,
            "future_perturbation_analog_equal": True,
            "future_perturbation_touched_columns": touched}


def _validate_spec(prepared, spec: dict, rolling_paths: pd.DataFrame) -> tuple[pd.DataFrame, str, str, list[int]]:
    if (not _ID.fullmatch(str(spec.get("id", ""))) or spec.get("adapter") != "day_analog"
            or spec.get("arm") != "EXPLORE" or spec.get("seed") is not None
            or type(spec.get("prefix_length")) is not int
            or spec["prefix_length"] not in PREFIX_LENGTHS
            or type(spec.get("top_k")) is not int or spec["top_k"] not in TOP_K
            or type(spec.get("bank_days")) is not int or spec["bank_days"] != BANK_DAYS):
        raise ValueError("Analog requires one of four fixed deterministic EXPLORE specs")
    if not prepared.split_lock.get("lock_sha256") or not prepared.seal.get("raw_sha256"):
        raise ValueError("Analog requires sealed split and raw source")
    if {h for h, _ in prepared.contexts} != set(HORIZONS):
        raise ValueError("Analog requires thirteen horizons")
    folds = sorted({fold for _, fold in prepared.contexts})
    if not folds or any({h for h, f in prepared.contexts if f == fold} != set(HORIZONS)
                        for fold in folds):
        raise ValueError("Analog requires a complete weekly horizon grid")
    if any(prepared.contexts[(4, fold)].get("summary", {}).get("arm") not in (None, "EXPLORE")
           for fold in folds):
        raise ValueError("Analog received non-EXPLORE weekly context")
    paths, paths_sha = _paths_index(rolling_paths, prepared.contexts)
    proof = rolling_paths.attrs.get("causal_provenance", {})
    upstream = spec.get("rolling_audit_sha256")
    if (not isinstance(upstream, str) or not re.fullmatch(r"[0-9a-f]{64}", upstream)
            or not isinstance(proof, dict) or config_hash(proof) != upstream
            or proof.get("leakage_test") != "passed"
            or proof.get("input_cutoff_rule") != "history.index <= origin"
            or proof.get("future_perturbation_max_abs_difference") != 0.0):
        raise ValueError("Analog requires signed causal R1 producer provenance")
    return paths, paths_sha, upstream, folds


def run(prepared, spec: dict, rolling_paths: pd.DataFrame,
        *, smoke_first_fold: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fit/reuse one deterministic library per fold; return CAL/SCORE keys."""
    paths, paths_sha, upstream, folds = _validate_spec(prepared, spec, rolling_paths)
    selected_folds = folds[:1] if smoke_first_fold else folds
    frames, cells = [], []
    for fold in selected_folds:
        fold_start = perf_counter()
        fit, stop, roles = _common_roles(prepared.contexts, fold)
        bank, bank_audit = _build_bank(prepared, fold, fit, spec["prefix_length"])
        bank_build_seconds = perf_counter() - fold_start
        bundle, cell = _fit_or_load(prepared, spec, paths_sha, fold, fit, stop,
                                    bank, bank_audit, paths)
        prediction_start = perf_counter()
        model_path, meta_path = _checkpoint_paths(prepared, spec, fold)
        cell["checkpoint_metadata_sha256"] = sha256(meta_path)
        future_probe = _future_probe(prepared, bundle, spec, paths, fold)
        coverage = []
        for role in ("cal", "score"):
            all_origins = pd.DatetimeIndex(sorted(set.union(*(
                set(prepared.contexts[(h, fold)][role]) for h in HORIZONS))))
            ranked = _rank(prepared.history, bundle["bank"], all_origins,
                           spec["prefix_length"], spec["top_k"], role=role)
            locations = pd.Series(np.arange(len(all_origins)), index=all_origins)
            for h in HORIZONS:
                context = prepared.contexts[(h, fold)]
                origins = pd.DatetimeIndex(context[role])
                selected = locations.loc[origins].to_numpy(dtype=np.int64)
                local = {"indices": ranked["indices"][selected],
                         "distances": ranked["distances"][selected],
                         "support": ranked["support"][selected],
                         "query_levels": ranked["query_levels"][selected]}
                analog = _analog_points(bundle["bank"], local, h)
                r1 = _path_values(paths, origins, h).r1.to_numpy(dtype=np.float64)
                forecast = _blend(r1, analog, bundle["alpha"])
                fallback = ~np.isfinite(analog)
                frames.append(make_frame(context, origins, h, fold, forecast,
                                         spec["id"], role, r1=r1,
                                         analog_point=analog,
                                         analog_support=local["support"],
                                         analog_fallback=fallback,
                                         analog_alpha=float(bundle["alpha"])))
                coverage.append({"horizon": h, "role": role, "n": len(origins),
                                 "fallback_count": int(fallback.sum()),
                                 "quality_missing_query_count": int((~ranked["query_prefix_clean"][selected]).sum()),
                                 "mean_eligible_same_slot": float(local["support"].mean())})
        cells.append({**roles, **cell, "fold": fold,
                      "checkpoint_model_sha256": sha256(model_path),
                      **future_probe,
                      "bank_build_seconds": bank_build_seconds,
                      "prediction_seconds": perf_counter() - prediction_start,
                      "fold_wall_seconds": perf_counter() - fold_start,
                      "coverage": coverage})
        if (Path(prepared.out) / "logs/stop_requested.json").exists():
            raise RuntimeError(f"Analog stop requested after fold {fold} checkpoint")
    frame = pd.concat(frames, ignore_index=True)
    expected = sum(len(prepared.contexts[(h, fold)][role])
                   for fold in selected_folds for h in HORIZONS
                   for role in ("cal", "score"))
    if len(frame) != expected or frame.duplicated(["horizon", "fold", "role", "origin"]).any():
        raise ValueError("Analog changed frozen CAL/SCORE keys")
    if not np.isfinite(frame.pred.to_numpy(dtype=np.float64)).all():
        raise ValueError("Analog returned nonfinite forecast")
    return frame, {"adapter": "day_analog", "paths_sha256": paths_sha,
                   "rolling_audit_sha256": upstream,
                   "rolling_upstream_causal_manifest_verified": True,
                   "cells": cells, "n_fold_models": len(cells),
                   "n_stochastic_seeds": 0, "n_deterministic_runs": 1,
                   "seed": None, "smoke_first_fold": smoke_first_fold,
                   "future_perturbation_max_abs_difference": 0.0,
                   "leakage_test": "passed", "development_only": True,
                   "holdout_read": False}


def verify_cells(prepared, spec: dict, rolling_paths: pd.DataFrame,
                 audit: dict, *, smoke_first_fold: bool = False) -> None:
    """Recompute every identity and verify existing physical models, no refit."""
    paths, paths_sha, upstream, folds = _validate_spec(prepared, spec, rolling_paths)
    selected_folds = folds[:1] if smoke_first_fold else folds
    if (audit.get("adapter") != "day_analog" or audit.get("paths_sha256") != paths_sha
            or audit.get("rolling_audit_sha256") != upstream
            or audit.get("n_fold_models") != len(selected_folds)
            or audit.get("n_stochastic_seeds") != 0
            or audit.get("n_deterministic_runs") != 1
            or audit.get("future_perturbation_max_abs_difference") != 0.0
            or audit.get("holdout_read") is not False
            or len(audit.get("cells", [])) != len(selected_folds)):
        raise ValueError("Analog run audit differs from frozen deterministic design")
    for fold, recorded in zip(selected_folds, audit["cells"]):
        fit, stop, _ = _common_roles(prepared.contexts, fold)
        bank, bank_audit = _build_bank(prepared, fold, fit, spec["prefix_length"])
        identity = _identity(prepared, spec, paths_sha, fold, fit, stop, bank, bank_audit)
        model_path, meta_path = _checkpoint_paths(prepared, spec, fold)
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Analog checkpoint missing for fold {fold}")
        selection = _stop_selection(prepared, spec, paths, fold, stop, bank)
        _, meta = _read_checkpoint(prepared, model_path, meta_path,
                                   identity, bank_audit,
                                   spec, fold, selection)
        if (recorded.get("fold") != fold or recorded.get("identity") != identity
                or recorded.get("model_sha256") != sha256(model_path)
                or recorded.get("checkpoint_model_sha256") != sha256(model_path)
                or recorded.get("checkpoint_metadata_sha256") != sha256(meta_path)
                or recorded.get("bank_digest") != bank_audit["bank_digest"]
                or recorded.get("alpha") != meta["alpha"]
                or recorded.get("future_perturbation_max_abs_difference") != 0.0
                or recorded.get("future_perturbation_analog_equal") is not True
                or recorded.get("future_perturbation_touched_columns") != [
                    "power", *[flag for flag in ("quality_bad", "time_repaired")
                               if flag in prepared.history]]):
            raise ValueError(f"Analog physical checkpoint/audit changed for fold {fold}")
