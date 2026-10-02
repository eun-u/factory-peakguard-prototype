"""Read-only Phase C artifact audit; writes only two audit JSON files.

Run after ``predictions/main_predictions.parquet`` exists. The full raw CSV is
never opened: its opaque preregistered digest is reported as a locked claim,
while the development context cache, non-raw sealed inputs, source code, model
files, and prediction parquets are rehashed. This script does not train, score,
or evaluate model performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ("B0", "B1", "B2", "B3", "B4", "B5", "M1", "M1-W", "M2", "C1")
HORIZONS = tuple(range(4, 17))
FOLDS = (0, 1, 2)
ANCHORS = (4, 8, 12, 16)
REQUIRED = {"model", "horizon", "fold", "origin", "target_time", "y", "pred", "tau",
            "d2", "train_seconds", "inference_seconds", "selected_config", "development_only"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    _require(path.is_file(), f"Required lock missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                         encoding="utf-8")
    os.replace(temporary, path)


def _created_at(path: Path) -> float | None:
    stat = path.stat()
    if hasattr(stat, "st_birthtime"):
        return stat.st_birthtime
    if os.name == "nt":
        return stat.st_ctime
    return None


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _clock(path: Path) -> float:
    """Creation time on Windows; modification time is a weaker fallback."""
    created = _created_at(path)
    return created if created is not None else _mtime(path)


def _entry(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "created_utc": datetime.fromtimestamp(_clock(path), timezone.utc).isoformat(),
            "modified_utc": datetime.fromtimestamp(_mtime(path), timezone.utc).isoformat(),
            "creation_time_available": _created_at(path) is not None}


def _seal_hashes(root: Path, out: Path, prereg: Mapping, model_lock: Mapping) -> dict:
    context_path = out / "models/development_contexts.joblib"
    _require(context_path.is_file(), "Development context cache missing")
    context_digest = _sha256(context_path)
    _require(context_digest == prereg["context_cache_sha256"], "Development context hash changed")
    _require(prereg.get("development_only") is True, "Preregistration is not development-only")
    nonraw_inputs, raw_locked = {}, {}
    for relative, expected in prereg["bound_inputs"].items():
        path = root / relative
        if relative.startswith("data/raw/"):
            # The locked value is preserved without reopening raw holdout bytes.
            raw_locked[relative] = {"sha256_at_preregistration": expected,
                                    "current_bytes_rehashed": False}
            continue
        _require(path.is_file(), f"Sealed input missing: {relative}")
        current = _sha256(path)
        _require(current == expected, f"Sealed input changed: {relative}")
        nonraw_inputs[relative] = current
    implementation = {}
    for relative, expected in prereg["implementation_hashes"].items():
        path = root / relative
        _require(path.is_file(), f"Sealed implementation missing: {relative}")
        current = _sha256(path)
        _require(current == expected, f"Sealed implementation changed: {relative}")
        implementation[relative] = current
    config = root / "configs/phase_c.json"
    prereg_text = out / "logs/preregistration_phase_c.md"
    _require(model_lock.get("config_sha256") == _sha256(config), "Model lock config hash mismatch")
    _require(model_lock.get("preregistration_sha256") == _sha256(prereg_text),
             "Model lock preregistration hash mismatch")
    _require(model_lock.get("locked_before_score") is True and model_lock.get("development_only") is True,
             "Model configuration lock lacks sealed development status")
    return {"context_cache": {"path": context_path.relative_to(root).as_posix(),
                               "sha256": context_digest},
            "nonraw_bound_inputs": nonraw_inputs,
            "raw_source_locked_without_reread": raw_locked,
            "implementation_hashes": implementation,
            "verifier_script_sha256": _sha256(Path(__file__))}


def _locked_parts(lock: Mapping, cfg: Mapping) -> dict:
    baseline = lock.get("baseline", lock)
    cbl = baseline["cbl_by_horizon"]
    _require(baseline.get("baseline_status") == "prepared", "B2 selection not prepared")
    _require(set(cbl) == {str(h) for h in HORIZONS}, "B2 horizon lock incomplete")
    for h in HORIZONS:
        _require(cbl[str(h)] in cfg["cbl_candidates"], f"Unregistered CBL at horizon {h}")
    for family in ("lgbm", "tcn"):
        detail = lock[family]
        selected = [row for row in cfg[family]["candidates"] if row["id"] == detail.get("selected_id")]
        _require(detail.get("source") == "stop_only" and len(selected) == 1,
                 f"{family} setting was not selected on stop only")
        expected = {key: value for key, value in selected[0].items() if key != "id"}
        _require(detail.get("selected_config") == expected, f"{family} parameters differ from fixed candidates")
    return {"cbl": cbl, "lgbm": lock["lgbm"], "tcn": lock["tcn"]}


def _expected_config(model: str, horizon: int, selected: Mapping, cfg: Mapping) -> dict:
    if model in ("B0", "B1"):
        return {"feature": "current" if model == "B0" else "slot7d"}
    if model == "B2":
        return {"selected_cbl": selected["cbl"][str(horizon)]}
    if model == "B3":
        return dict(cfg["mstl"])
    if model == "B4":
        return dict(cfg["ridge"])
    if model == "B5":
        return dict(cfg["kalman"])
    if model in ("M1", "M1-W", "C1"):
        return {"selected_id": selected["lgbm"]["selected_id"],
                "model_params": selected["lgbm"]["selected_config"],
                "peak_weight": float(cfg["peak_weight"]) if model == "M1-W" else 1.0,
                "target_transform": "weekly_residual" if model == "C1" else "direct"}
    return {"selected_id": selected["tcn"]["selected_id"],
            "model_params": selected["tcn"]["selected_config"],
            "peak_weight": 1.0, "target_transform": "direct"}


def certify_b2_missingness(predictions: pd.DataFrame, history: pd.DataFrame,
                           contexts: Mapping, lock: Mapping, cfg: Mapping) -> dict:
    """Recompute the locked CBL exactly; certify only structural B2 NaNs.

    This calls the sealed original CBL source on development score origins. It
    does not fit, choose another CBL candidate, fill gaps, or read a holdout.
    Every saved B2 value, including every NaN position, must match the source.
    """
    from src.models.cbl import cbl_all_predictions

    selected = _locked_parts(lock, cfg)
    certified = {}
    for h in HORIZONS:
        method = selected["cbl"][str(h)]
        for f in FOLDS:
            score = pd.DatetimeIndex(contexts[(h, f)]["score"])
            rows = predictions.loc[predictions.model.eq("B2") & predictions.horizon.eq(h)
                                   & predictions.fold.eq(f)].sort_values("origin")
            got_origins = pd.DatetimeIndex(pd.to_datetime(rows.origin, errors="raise"))
            _require(len(rows) == len(score) and got_origins.equals(score),
                     f"B2 raw score keys differ before CBL certification at h{h} f{f}")
            recomputed = cbl_all_predictions(history, score, h)[method].to_numpy(dtype=float)
            saved = pd.to_numeric(rows.pred, errors="coerce").to_numpy(dtype=float)
            _require(not np.isinf(recomputed).any() and not np.isinf(saved).any(),
                     f"B2 has infinite CBL prediction at h{h} f{f}")
            _require(np.array_equal(saved, recomputed, equal_nan=True),
                     f"B2 saved prediction or missing mask differs from sealed CBL source at h{h} f{f}")
            certified[(h, f)] = score[np.isnan(recomputed)]
    return certified


def validate_predictions(predictions: pd.DataFrame, contexts: Mapping,
                         lock: Mapping, cfg: Mapping,
                         certified_b2_missing: Mapping | None = None) -> dict:
    """Keep exact raw keys; only source-certified B2 NaNs may leave pairing."""
    certified_b2_missing = certified_b2_missing or {}
    _require(REQUIRED <= set(predictions.columns), f"Prediction columns missing: {sorted(REQUIRED - set(predictions.columns))}")
    _require(not predictions.empty, "Main prediction parquet is empty")
    data = predictions.copy()
    data["origin"] = pd.to_datetime(data["origin"], errors="raise")
    data["target_time"] = pd.to_datetime(data["target_time"], errors="raise")
    selected = _locked_parts(lock, cfg)
    _require(set(data.model) == set(MAIN), "Main parquet must contain exactly MAIN10")
    _require(set(zip(data.horizon, data.fold)) == {(h, f) for h in HORIZONS for f in FOLDS},
             "Main parquet lacks a horizon/fold group or contains an extra one")
    _require(not data.duplicated(["model", "horizon", "fold", "origin"]).any(),
             "Duplicate model/horizon/fold/origin key")
    _require(data.development_only.eq(True).all(), "Prediction row is not development-only")
    _require(pd.api.types.is_bool_dtype(data.d2), "d2 must be a boolean score-subset flag")
    for column in ("y", "tau", "train_seconds", "inference_seconds"):
        _require(np.isfinite(pd.to_numeric(data[column], errors="coerce")).all(),
                 f"Nonfinite {column}; score truth and timing must be complete")
    _require(pd.api.types.is_numeric_dtype(data.pred), "Prediction column must be numeric")
    expected_missing = np.zeros(len(data), dtype=bool)
    for (h, f), origins in certified_b2_missing.items():
        _require((h, f) in {(horizon, fold) for horizon in HORIZONS for fold in FOLDS},
                 f"Unexpected B2 missingness cell: {h}, {f}")
        score = pd.DatetimeIndex(contexts[(h, f)]["score"])
        origins = pd.DatetimeIndex(origins)
        _require(origins.is_unique and origins.isin(score).all(),
                 f"Certified B2 missing origins are outside score at h{h} f{f}")
        expected_missing |= (data.model.eq("B2") & data.horizon.eq(h) & data.fold.eq(f)
                             & data.origin.isin(origins)).to_numpy(dtype=bool)
    prediction_values = data.pred.to_numpy(dtype=float)
    _require(np.array_equal(np.isnan(prediction_values), expected_missing),
             "Prediction NaN mask differs from source-certified B2 structural missingness")
    _require(np.isfinite(prediction_values[~expected_missing]).all(),
             "Unexpected nonfinite prediction outside certified B2 missingness")
    _require((data[["train_seconds", "inference_seconds"]] >= 0).all().all(),
             "Negative training or inference time")
    boundary = pd.Timestamp(cfg["boundary"])
    _require(data.origin.lt(boundary).all() and data.target_time.lt(boundary).all(),
             "Prediction reaches the sealed holdout boundary")
    _require((data.target_time == data.origin + pd.to_timedelta(data.horizon * 15, unit="m")).all(),
             "Target timestamps differ from direct horizon")
    _require(set(contexts) == {(h, f) for h in HORIZONS for f in FOLDS},
             "Development context cache does not contain exactly 39 cells")
    expected_rows = 0
    coverage = []
    certified_total = 0
    for h in HORIZONS:
        for f in FOLDS:
            c = contexts[(h, f)]
            score = pd.DatetimeIndex(c["score"])
            _require(score.is_unique and score.is_monotonic_increasing and len(score) > 0,
                     f"Context score origins invalid at h{h} f{f}")
            non_score = c["fit"].append(c["stop"]).append(c["cal"])
            _require(score.intersection(non_score).empty,
                     f"Score overlaps fit/stop/cal at h{h} f{f}")
            expected_rows += len(score) * len(MAIN)
            group = data.loc[data.horizon.eq(h) & data.fold.eq(f)]
            _require(len(group) == len(score) * len(MAIN), f"Wrong MAIN10 row count at h{h} f{f}")
            truth_target = pd.DatetimeIndex(c["target_time"].loc[score])
            truth_y = c["y"].loc[score].to_numpy(dtype=float)
            truth_d2 = c["d2"].loc[score].to_numpy(dtype=bool)
            truth_tau = float(c["tau"])
            _require(np.isfinite(truth_y).all() and np.isfinite(truth_tau),
                     f"Context truth is nonfinite at h{h} f{f}")
            missing_origins = pd.DatetimeIndex(certified_b2_missing.get((h, f), []))
            valid_origins = score[~score.isin(missing_origins)]
            _require(len(valid_origins) > 0, f"No finite paired MAIN10 score origins at h{h} f{f}")
            certified_total += len(missing_origins)
            for model in MAIN:
                rows = group.loc[group.model.eq(model)].sort_values("origin")
                got = pd.DatetimeIndex(rows.origin)
                _require(len(rows) == len(score) and got.equals(score),
                         f"{model} has missing, duplicate, or extra score origins at h{h} f{f}")
                _require(pd.DatetimeIndex(rows.target_time).equals(truth_target),
                         f"{model} target_time differs from development context at h{h} f{f}")
                _require(np.array_equal(rows.y.to_numpy(dtype=float), truth_y),
                         f"{model} y differs from development context at h{h} f{f}")
                _require(np.array_equal(rows.tau.to_numpy(dtype=float), np.full(len(score), truth_tau)),
                         f"{model} tau differs from fit-only context at h{h} f{f}")
                _require(np.array_equal(rows.d2.to_numpy(dtype=bool), truth_d2),
                         f"{model} d2 differs from score context at h{h} f{f}")
                coverage.append({"model": model, "horizon": h, "fold": f,
                                 "D1_expected": len(score), "D1_actual": len(rows),
                                 "D2_expected": int(truth_d2.sum()),
                                 "D2_actual": int(rows.d2.sum()),
                                 "D1_paired": len(valid_origins),
                                 "D2_paired": int(truth_d2[~score.isin(missing_origins)].sum()),
                                 "removed_for_pairing": len(missing_origins)})
                try:
                    parsed = [json.loads(value) for value in rows.selected_config]
                except (TypeError, ValueError) as exc:
                    raise AssertionError(f"{model} selected_config is not JSON at h{h} f{f}") from exc
                expected_config = _expected_config(model, h, selected, cfg)
                _require(all(value == expected_config for value in parsed),
                         f"{model} selected_config conflicts with locked settings at h{h} f{f}")
                for field in ("train_seconds", "inference_seconds"):
                    _require(rows[field].nunique() == 1,
                             f"{model} {field} changes within one fitted cell")
            finite = group.loc[np.isfinite(group.pred.to_numpy(dtype=float))]
            widths = finite.groupby("origin").model.nunique()
            _require(all(int(widths.get(origin, 0)) == (9 if origin in missing_origins else 10)
                         for origin in score),
                     f"Finite MAIN10 intersection differs from certified B2 mask at h{h} f{f}")
    _require(len(data) == expected_rows, "Main parquet has extra or missing rows")
    raw_per_model = expected_rows // len(MAIN)
    paired_per_model = raw_per_model - certified_total
    removal = {model: {"input": raw_per_model, "union_keys": raw_per_model,
                       "raw_missing": 0,
                       "nonfinite": certified_total if model == "B2" else 0,
                       "removed_for_pairing": 0 if model == "B2" else certified_total,
                       "common_finite": paired_per_model,
                       "removed": certified_total} for model in MAIN}
    return {"MAIN10_models": len(MAIN), "horizon_fold_cells": len(HORIZONS) * len(FOLDS),
            "score_origin_keys_per_model": raw_per_model,
            "main_prediction_rows": expected_rows,
            "paired_prediction_rows": paired_per_model * len(MAIN),
            "paired_score_origin_keys_per_model": paired_per_model,
            "certified_B2_structural_missing_keys": certified_total,
            "only_certified_B2_NaNs": True,
            "paired_cohort_shrinkage": certified_total,
            "per_model_removal_counts": removal,
            "context_truth_match": True, "calibration_origins_in_predictions": False,
            "holdout_timestamps_in_predictions": False,
            "locked_config_match": True,
            "D1_D2_coverage_by_model_horizon_fold": coverage}


def _compare_evaluator_removal(out: Path, expected: Mapping) -> dict:
    """The evaluator must exclude exactly the independently certified keys."""
    selection = _read_json(out / "logs/model_selection.json")
    observed = selection.get("removed_counts")
    _require(isinstance(observed, dict) and set(observed) == set(MAIN),
             "Evaluator removal metadata is absent or lacks MAIN10")
    for model in MAIN:
        _require(observed[model] == expected[model],
                 f"Evaluator paired-cohort removal differs from certified B2 mask for {model}")
    return {"source": "logs/model_selection.json", "MAIN10_removed_counts_match": True,
            "per_model": observed}


def _reference_coverage(out: Path, contexts: Mapping, cfg: Mapping) -> dict:
    """Report optional R1 coverage separately; R1 is never a MAIN10 candidate."""
    path = out / "predictions/reference_predictions.parquet"
    rows = pd.read_parquet(path) if path.is_file() else None
    if rows is not None:
        _require(REQUIRED <= set(rows.columns), "R1 reference parquet has incomplete schema")
        _require(set(rows.model) == {"R1"}, "Reference parquet contains a main candidate or unknown model")
        _require(rows.development_only.eq(True).all(), "R1 reference is not development-only")
        _require(pd.api.types.is_bool_dtype(rows.d2), "R1 d2 must be boolean")
        _require(not rows.duplicated(["horizon", "fold", "origin"]).any(), "R1 reference keys duplicate")
        rows = rows.copy()
        rows["origin"] = pd.to_datetime(rows.origin, errors="raise")
        rows["target_time"] = pd.to_datetime(rows.target_time, errors="raise")
        boundary = pd.Timestamp(cfg["boundary"])
        _require(rows.origin.lt(boundary).all() and rows.target_time.lt(boundary).all(),
                 "R1 reference reaches holdout boundary")
    details = []
    for h in HORIZONS:
        for f in FOLDS:
            c = contexts[(h, f)]
            score = pd.DatetimeIndex(c["score"])
            part = rows.loc[rows.horizon.eq(h) & rows.fold.eq(f)].sort_values("origin") if rows is not None else None
            actual = len(part) if part is not None else 0
            finite = int(np.isfinite(pd.to_numeric(part.pred, errors="coerce")).sum()) if part is not None else 0
            if part is not None:
                _require(actual == len(score) and pd.DatetimeIndex(part.origin).equals(score),
                         f"R1 missing or extra reference origins at h{h} f{f}")
                _require(pd.DatetimeIndex(part.target_time).equals(pd.DatetimeIndex(c["target_time"].loc[score])),
                         f"R1 target_time differs from context at h{h} f{f}")
                _require(np.array_equal(part.y.to_numpy(dtype=float), c["y"].loc[score].to_numpy(dtype=float)),
                         f"R1 truth differs from context at h{h} f{f}")
                _require(np.array_equal(part.tau.to_numpy(dtype=float), np.full(len(score), float(c["tau"]))),
                         f"R1 tau differs from context at h{h} f{f}")
                _require(np.array_equal(part.d2.to_numpy(dtype=bool), c["d2"].loc[score].to_numpy(dtype=bool)),
                         f"R1 d2 differs from context at h{h} f{f}")
                _require(finite == actual, f"R1 nonfinite reference prediction at h{h} f{f}")
            details.append({"model": "R1", "horizon": h, "fold": f,
                            "expected": len(score), "actual": actual, "finite": finite})
    if rows is not None:
        _require(len(rows) == sum(item["expected"] for item in details), "R1 extra reference rows")
        runtime_path = out / "logs/chronos_runtime.json"
        if runtime_path.is_file():
            runtime = _read_json(runtime_path)
            _require(runtime.get("reference_only") is True and runtime.get("local_fit_seconds") == 0.0,
                     "R1 runtime does not mark the model as reference-only")
    return {"reference_only": bool(cfg["chronos"]["reference_only"]),
            "status": "complete" if rows is not None else "not_available",
            "expected_total": sum(item["expected"] for item in details),
            "actual_total": sum(item["actual"] for item in details),
            "finite_total": sum(item["finite"] for item in details),
            "by_horizon_fold": details}


def _expected_artifacts(out: Path, cfg: Mapping, selected: Mapping) -> set[Path]:
    models = out / "models"
    expected = set()
    for h in HORIZONS:
        for f in FOLDS:
            key = f"h{h:02d}_f{f}"
            expected.update(models / f"{model}_{key}.joblib" for model in MAIN[:6])
            expected.add(models / f"CBL_stop_{key}.parquet")
            expected.add(models / f"score_{key}.parquet")
            expected.add(models / f"score_lgbm_M1-W_{key}.joblib")
            expected.add(models / f"score_lgbm_C1_{key}.joblib")
            if h not in ANCHORS:
                expected.add(models / f"score_lgbm_M1_{key}.joblib")
                expected.add(models / f"score_tcn_M2_{key}.pt")
    for h in ANCHORS:
        for f in FOLDS:
            key = f"h{h:02d}_f{f}"
            for candidate in cfg["lgbm"]["candidates"]:
                name = candidate["id"]
                expected.add(models / f"tune_lgbm_{name}_{key}.joblib")
                expected.add(models / f"tune_lgbm_{name}_{key}_stop.parquet")
            for candidate in cfg["tcn"]["candidates"]:
                name = candidate["id"]
                expected.add(models / f"tune_tcn_{name}_{key}.pt")
                expected.add(models / f"tune_tcn_{name}_{key}_stop.parquet")
            # A selected anchor bundle is also the M1/M2 scoring bundle.
            expected.add(models / f"tune_lgbm_{selected['lgbm']['selected_id']}_{key}.joblib")
            expected.add(models / f"tune_tcn_{selected['tcn']['selected_id']}_{key}.pt")
    expected.update((models / "lgbm_anchor_stop_predictions.parquet",
                     models / "tcn_anchor_stop_predictions.parquet",
                     out / "predictions/main_predictions.parquet"))
    return expected


def _compare_score_chunks(out: Path, predictions: pd.DataFrame) -> int:
    count = 0
    columns = list(predictions.columns)
    sort_key = ["model", "horizon", "fold", "origin"]
    for h in HORIZONS:
        for f in FOLDS:
            path = out / "models" / f"score_h{h:02d}_f{f}.parquet"
            chunk = pd.read_parquet(path)
            _require(set(chunk.columns) == set(columns), f"Chunk schema differs at h{h} f{f}")
            expected = predictions.loc[predictions.horizon.eq(h) & predictions.fold.eq(f), columns]
            actual = chunk.loc[:, columns]
            expected = expected.sort_values(sort_key).reset_index(drop=True)
            actual = actual.sort_values(sort_key).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            except AssertionError as exc:
                raise AssertionError(f"Score chunk differs from main parquet at h{h} f{f}") from exc
            count += 1
    return count


def _cache_files(out: Path) -> list[Path]:
    """Enumerate local artifacts without descending into pretrained weights."""
    result = []
    for directory in (out / "models", out / "predictions"):
        for current, subdirs, filenames in os.walk(directory):
            subdirs[:] = [name for name in subdirs if name != "hf_cache"]
            result.extend(Path(current) / name for name in filenames)
    return sorted(result)


def _verify_persisted_models(out: Path, cfg: Mapping, selected: Mapping) -> dict:
    """Inspect fitted bundle settings without retraining or replaying scores."""
    models = out / "models"
    verified = {"B0_B5_baseline_bundles": 0, "LightGBM_fitted_bundles": 0,
                "TCN_fitted_bundles": 0, "LightGBM_booster_parameter_fields_seen": 0}
    feature_order = [name for group in cfg["feature_groups"].values() for name in group]
    for h in HORIZONS:
        for f in FOLDS:
            key = f"h{h:02d}_f{f}"
            for name in MAIN[:6]:
                artifact = joblib.load(models / f"{name}_{key}.joblib")
                _require(artifact.get("development_only") is True and artifact.get("fit_parameters_only") is True,
                         f"{name} fitted bundle lacks development/fit-only flags at {key}")
                _require(np.isfinite(artifact.get("train_seconds", np.nan)) and artifact["train_seconds"] >= 0,
                         f"{name} training time invalid at {key}")
                bundle = artifact["bundle"]
                if name in ("B0", "B1"):
                    expected = "current" if name == "B0" else "slot7d"
                    _require(bundle.get("feature") == expected, f"{name} persisted feature differs at {key}")
                elif name == "B2":
                    _require(bundle.get("selected_cbl") == selected["cbl"][str(h)],
                             f"B2 persisted CBL choice differs at {key}")
                elif name == "B3":
                    _require(bundle.get("fit_parameters_only") is True and bundle.get("development_only") is True,
                             f"B3 MSTL fit scope missing at {key}")
                    _require(0 <= float(bundle["phi"]) <= .99 and isinstance(bundle["base_path"], pd.Series),
                             f"B3 persisted MSTL settings invalid at {key}")
                elif name == "B4":
                    _require(float(bundle["alpha"]) == float(cfg["ridge"]["alpha"])
                             and bundle["columns"] == feature_order,
                             f"B4 persisted Ridge settings differ at {key}")
                    _require(np.isfinite(np.asarray(bundle["mean"])).all()
                             and np.isfinite(np.asarray(bundle["scale"])).all()
                             and (np.asarray(bundle["scale"]) > 0).all(),
                             f"B4 persisted normalization invalid at {key}")
                else:
                    _require(bundle.get("fit_parameters_only") is True and bundle.get("development_only") is True,
                             f"B5 Kalman fit scope missing at {key}")
                    _require(-.99 <= float(bundle["phi"]) <= .99
                             and float(bundle["q"]) > 0 and float(bundle["r"]) > 0,
                             f"B5 persisted state parameters invalid at {key}")
                verified["B0_B5_baseline_bundles"] += 1

    def check_lgbm(path: Path, candidate: Mapping) -> None:
        artifact = joblib.load(path)
        _require(artifact.get("development_only") is True and artifact.get("fit_parameters_only") is True,
                 f"LightGBM fit scope missing: {path.name}")
        model = artifact["model"]
        params = model.get_params(deep=False)
        expected = {"objective": cfg["lgbm"]["objective"],
                    "learning_rate": cfg["lgbm"]["learning_rate"],
                    "n_estimators": cfg["lgbm"]["n_estimators"],
                    "n_jobs": cfg["lgbm"]["n_jobs"], "random_state": cfg["seed"],
                    "num_leaves": candidate["num_leaves"],
                    "min_child_samples": candidate["min_child_samples"]}
        for field, value in expected.items():
            _require(params.get(field) == value, f"LightGBM persisted {field} differs: {path.name}")
        booster_params = getattr(model.booster_, "params", {})
        for field in ("num_leaves", "min_child_samples", "learning_rate", "objective"):
            if field in booster_params:
                _require(booster_params[field] == expected[field],
                         f"LightGBM booster {field} differs: {path.name}")
                verified["LightGBM_booster_parameter_fields_seen"] += 1
        _require(np.isfinite(artifact.get("train_seconds", np.nan)) and artifact["train_seconds"] >= 0,
                 f"LightGBM training time invalid: {path.name}")
        verified["LightGBM_fitted_bundles"] += 1

    for h in ANCHORS:
        for f in FOLDS:
            key = f"h{h:02d}_f{f}"
            for candidate in cfg["lgbm"]["candidates"]:
                check_lgbm(models / f"tune_lgbm_{candidate['id']}_{key}.joblib", candidate)
    chosen_lgbm = {"id": selected["lgbm"]["selected_id"], **selected["lgbm"]["selected_config"]}
    for h in HORIZONS:
        for f in FOLDS:
            key = f"h{h:02d}_f{f}"
            for name in ("M1-W", "C1"):
                check_lgbm(models / f"score_lgbm_{name}_{key}.joblib", chosen_lgbm)
            if h not in ANCHORS:
                check_lgbm(models / f"score_lgbm_M1_{key}.joblib", chosen_lgbm)

    import torch

    def check_tcn(path: Path, candidate: Mapping) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        _require(artifact.get("development_only") is True and artifact.get("fit_parameters_only") is True,
                 f"TCN fit scope missing: {path.name}")
        bundle = artifact["model"]
        params = bundle["config"]
        expected = {"channels": candidate["channels"], "dropout": candidate["dropout"],
                    "max_epochs": cfg["tcn"]["max_epochs"], "patience": cfg["tcn"]["patience"],
                    "kernel_size": cfg["tcn"]["kernel_size"],
                    "dilations": cfg["tcn"]["dilations"],
                    "batch_size": cfg["tcn"]["batch_size"],
                    "learning_rate": cfg["tcn"]["learning_rate"],
                    "weight_decay": cfg["tcn"]["weight_decay"]}
        for field, value in expected.items():
            _require(params.get(field) == value, f"TCN persisted {field} differs: {path.name}")
        _require(bundle["context_dim"] == len(cfg["tcn"]["context_columns"]),
                 f"TCN context width differs: {path.name}")
        _require(bundle["metadata"].get("development_only") is True,
                 f"TCN development flag missing: {path.name}")
        _require(np.isfinite(artifact.get("train_seconds", np.nan)) and artifact["train_seconds"] >= 0,
                 f"TCN training time invalid: {path.name}")
        verified["TCN_fitted_bundles"] += 1

    for h in ANCHORS:
        for f in FOLDS:
            key = f"h{h:02d}_f{f}"
            for candidate in cfg["tcn"]["candidates"]:
                check_tcn(models / f"tune_tcn_{candidate['id']}_{key}.pt", candidate)
    chosen_tcn = {"id": selected["tcn"]["selected_id"], **selected["tcn"]["selected_config"]}
    for h in HORIZONS:
        if h in ANCHORS:
            continue
        for f in FOLDS:
            check_tcn(models / f"score_tcn_M2_h{h:02d}_f{f}.pt", chosen_tcn)
    verified["identity_limitations"] = [
        "Persisted LightGBM booster parameters do not encode sample weights; M1-W weighting is supported by sealed training code and score-row metadata, not independently recoverable from the booster.",
        "C1 weekly-residual target transformation is supported by sealed training code and score-row metadata; the fitted booster does not encode original target construction.",
        "Persisted Ridge scalers do not encode their ddof; ddof=1 is supported by sealed training code and config.",
    ]
    return verified


def _assert_existing_manifest_unchanged(path: Path, current: Mapping) -> bool:
    """An existing cache manifest is an immutable comparison baseline."""
    if not path.is_file():
        return False
    previous = _read_json(path)

    def digests(value: Mapping) -> dict[str, str]:
        artifacts = value["artifacts"]
        return {entry["path"]: entry["sha256"] for entry in artifacts}

    _require(digests(previous) == digests(current),
             "Existing output_cache_manifest.json conflicts with current model/parquet hashes or file set")
    _require(previous["sealed"]["context_cache"] == current["sealed"]["context_cache"],
             "Existing manifest context hash changed")
    _require(previous["sealed"]["implementation_hashes"] == current["sealed"]["implementation_hashes"],
             "Existing manifest sealed code hashes changed")
    _require(previous["sealed"]["nonraw_bound_inputs"] == current["sealed"]["nonraw_bound_inputs"],
             "Existing manifest sealed nonraw input hashes changed")
    _require(previous["sealed"]["raw_source_locked_without_reread"] ==
             current["sealed"]["raw_source_locked_without_reread"],
             "Existing manifest preregistered raw digest claim changed")
    _require(previous["preregistration_lock"]["sha256"] == current["preregistration_lock"]["sha256"],
             "Existing manifest preregistration lock hash changed")
    _require(previous["model_config_lock"]["sha256"] == current["model_config_lock"]["sha256"],
             "Existing manifest model lock hash changed")
    _require(previous["sealed"]["verifier_script_sha256"] == current["sealed"]["verifier_script_sha256"],
             "Existing manifest verifier script changed")
    return True


def verify(root: Path, out: Path, *, require_current_lock_order: bool = True) -> tuple[dict, dict]:
    root, out = root.resolve(), out.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    prediction_path = out / "predictions/main_predictions.parquet"
    _require(prediction_path.is_file(), "Main prediction parquet is not complete")
    prereg_path = out / "logs/preregistration_lock.json"
    model_lock_path = out / "logs/model_config_lock.json"
    prereg = _read_json(prereg_path)
    model_lock = _read_json(model_lock_path)
    cfg = _read_json(root / "configs/phase_c.json")
    sealed = _seal_hashes(root, out, prereg, model_lock)
    selected = _locked_parts(model_lock, cfg)
    cache = joblib.load(out / "models/development_contexts.joblib")
    contexts = cache["contexts"]
    predictions = pd.read_parquet(prediction_path)
    certified = certify_b2_missingness(predictions, cache["history"], contexts, model_lock, cfg)
    row_audit = validate_predictions(predictions, contexts, model_lock, cfg,
                                     certified_b2_missing=certified)
    evaluator_removal = _compare_evaluator_removal(out, row_audit["per_model_removal_counts"])
    reference_audit = _reference_coverage(out, contexts, cfg)
    expected = _expected_artifacts(out, cfg, selected)
    missing = sorted(path.relative_to(root).as_posix() for path in expected if not path.is_file())
    _require(not missing, f"Expected model or score artifacts missing: {missing[:10]} (total={len(missing)})")
    chunks = _compare_score_chunks(out, predictions)
    _require(chunks == 39, "Incomplete score chunk comparison")
    persisted_models = _verify_persisted_models(out, cfg, selected)
    seal_time = max(_clock(prereg_path), datetime.fromisoformat(prereg["registered_at"]).timestamp())
    lock_time = _clock(model_lock_path)
    _require(lock_time >= seal_time, "Full model lock predates preregistration seal")
    cache_files = _cache_files(out)
    _require(all(not path.name.endswith(".tmp") for path in cache_files), "Incomplete temporary cache file remains")
    context_path = out / "models/development_contexts.joblib"
    score_paths = []
    for path in cache_files:
        if path == context_path:
            continue
        _require(_clock(path) >= seal_time, f"Output artifact predates preregistration seal: {path}")
        if path.name.startswith("score_h") and path.suffix == ".parquet":
            score_paths.append(path)
    current_lock_precedes_all_score_files = all(_clock(path) >= lock_time for path in
                                                 [*score_paths, prediction_path])
    if require_current_lock_order:
        _require(current_lock_precedes_all_score_files,
                 "Score artifact predates the current full model lock; current-lock order is unproven on resume")
    manifest = {"development_only": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root), "output_root": str(out),
                "sealed": sealed,
                "preregistration_lock": _entry(prereg_path, root),
                "model_config_lock": _entry(model_lock_path, root),
                "artifacts": [_entry(path, root) for path in cache_files],
                "hf_cache_pretrained_weights_included": False,
                "raw_holdout_bytes_reopened": False}
    audit = {"status": "passed" if current_lock_precedes_all_score_files else "passed_with_unproven_lock_order",
             "development_only": True,
             "checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "checks": row_audit | {"score_chunks_equal_main_parquet": chunks,
                                    "expected_model_and_parquet_artifacts_present": len(expected),
                                    "all_output_artifacts_post_preregistration": True,
                                    "current_lock_file_precedes_all_score_files": current_lock_precedes_all_score_files,
                                    "context_cache_sha256_matches_preregistration": True,
                                    "sealed_code_and_nonraw_input_hashes_match": True},
             "artifact_files_hashed": len(cache_files),
             "persisted_model_metadata": persisted_models,
             "evaluator_paired_cohort_removal": evaluator_removal,
             "R1_reference_coverage": reference_audit,
             "reporting_note": "d2_rank_reversal compares the sign of each model's difference versus the strongest baseline; actual D1/D2 ranks are separate fields.",
             "context_cache_reloaded": True,
             "raw_source_hash_status": "locked_at_preregistration_not_rehashed_by_verifier",
             "B2_missingness_rule": "Only NaNs exactly reproduced by the locked original CBL on sealed history and score origins are excluded from the common finite MAIN10 cohort; no fill or fallback is applied.",
             "timestamp_provenance_limit": "Filesystem creation times plus a fresh-namespace claim are not cryptographic proof of no prior fit.",
             "model_lock_timeline_limit": "The runner can rewrite the full config lock during resume; a score chunk predating the current lock file is reported, while locked row settings are checked against the current lock.",
             "manifest": "logs/output_cache_manifest.json"}
    return audit, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-resumed-lock-order", action="store_true",
                        help="Report current lock order as unproven if the runner rewrote it during resume")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "outputs/phase_c").resolve()
    audit_path = out / "logs/artifact_integrity_audit.json"
    try:
        audit, manifest = verify(root, out,
                                 require_current_lock_order=not args.allow_resumed_lock_order)
        manifest_path = out / "logs/output_cache_manifest.json"
        existing_manifest = _assert_existing_manifest_unchanged(manifest_path, manifest)
    except Exception as exc:
        _write_json(audit_path, {"status": "failed", "development_only": True,
                                 "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "error": f"{type(exc).__name__}: {exc}"})
        raise
    if not existing_manifest:
        _write_json(manifest_path, manifest)
    _write_json(audit_path, audit)
    print(f"Phase C artifact integrity passed: {audit['checks']['main_prediction_rows']} MAIN10 rows, "
          f"{audit['artifact_files_hashed']} cache files hashed", flush=True)


if __name__ == "__main__":
    main()
