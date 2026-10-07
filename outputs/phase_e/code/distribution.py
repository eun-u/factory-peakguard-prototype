"""Development-only h16 B5 predictive distribution and Phase C source parity.

The point model is the saved Phase C Kalman fit. No fitting, model selection,
or new partition occurs here. Source CSV bytes are hashed before the sealed
prefix reader decodes observations, and score artifacts are checked before use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from phase_c.data import (
    _core_features,
    _d2_score_mask,
    _daily_profiles,
    _sequences,
    load_history,
)
from phase_c.evaluation import MAIN
from phase_c.statistical import WEEK, _grid, _origins, predict_kalman
from src.session_data import SEALED_BOUNDARY
from src.split import make_splits
from src.targets import point_targets
from src.training import _partition


HORIZON = 16
MODEL = "B5"
KEY = ["fold", "origin", "target_time"]
PARITY_TOLERANCE = 1e-8
MANIFEST = Path("outputs/phase_c/logs/output_cache_manifest.json")
COMPLETION = Path("outputs/phase_c/logs/phase_c_completion_manifest.json")
FOLD_MANIFEST = Path("outputs/phase_c/tables/fold_manifest.csv")
FOLD_METRICS = Path("outputs/phase_c/tables/model_horizon_fold_metrics.csv")
POOLED_METRICS = Path("outputs/phase_c/tables/model_horizon_pooled_metrics.csv")
PREDICTIONS = Path("outputs/phase_c/predictions/main_predictions.parquet")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_manifest(root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    completion = json.loads((root / COMPLETION).read_text(encoding="utf-8"))
    if completion.get("development_only") is not True or completion.get("selected_M_star") != MODEL:
        raise ValueError("Phase C completion is not the locked development B5 selection")
    completion_hashes = completion["artifacts_sha256"]
    for relative in (MANIFEST, FOLD_METRICS, POOLED_METRICS):
        key = relative.relative_to("outputs/phase_c").as_posix()
        if _sha256(root / relative) != completion_hashes[key]:
            raise ValueError(f"Phase C source artifact hash changed: {relative}")
    cache = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if cache.get("development_only") is not True or not isinstance(cache.get("sealed"), dict):
        raise ValueError("Phase C cache manifest is not sealed development evidence")
    for relative in ("phase_c/data.py", "phase_c/statistical.py", "src/split.py", "src/training.py"):
        expected = cache["sealed"].get("implementation_hashes", {}).get(relative)
        if expected is not None and _sha256(root / relative) != expected:
            raise ValueError(f"Phase C implementation changed: {relative}")
    artifacts = {item["path"]: item for item in cache["artifacts"]}
    if len(artifacts) != len(cache["artifacts"]):
        raise ValueError("Duplicate paths in Phase C cache manifest")
    return artifacts, completion_hashes


def _checked_artifact(root: Path, relative: Path, artifacts: dict[str, dict]) -> str:
    key = relative.as_posix()
    entry = artifacts.get(key)
    if entry is None:
        raise FileNotFoundError(f"Missing Phase C cache manifest entry: {key}")
    path = root / relative
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Phase C cached artifact size changed: {key}")
    digest = _sha256(path)
    if digest != entry["sha256"]:
        raise ValueError(f"Phase C cached artifact hash changed: {key}")
    return digest


def _validate_parameters(bundle: dict) -> tuple[float, float, float]:
    try:
        phi, q, r = (float(bundle[key]) for key in ("phi", "q", "r"))
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("B5 Kalman parameters are missing or invalid") from exc
    if not (np.isfinite([phi, q, r]).all() and -0.99 <= phi <= 0.99 and q > 0 and r > 0):
        raise ValueError("B5 Kalman parameters must be finite with positive variances")
    return phi, q, r


def predict_kalman_distribution(
    bundle: dict, power: pd.Series, origins: pd.DatetimeIndex, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the existing B5 mean and its causal Gaussian variance at h16.

    The posterior variance is propagated on the same 15-minute grid as
    ``phase_c.statistical._filter_deviations``. Missing deviations advance
    the prior without an observation update. Only data at/before each origin
    can affect its posterior; the weekly target anchor is also in that past.
    """
    if horizon != HORIZON:
        raise ValueError("Phase E is locked to h16 (240 minutes)")
    phi, q, r = _validate_parameters(bundle)
    grid = _grid(power)
    origins = _origins(grid, origins, horizon)
    if horizon > WEEK:
        raise ValueError("The weekly target anchor would be future information")
    # Calling the sealed point implementation preserves its exact arithmetic.
    mu = predict_kalman(bundle, power, origins, horizon)
    if not len(origins):
        return mu, np.empty(0, dtype=float)
    positions = grid.index.get_indexer(origins)
    last = int(positions.max())
    values = grid.to_numpy(dtype=float)
    deviations = values[: last + 1].copy()
    deviations[:WEEK] = np.nan
    if last >= WEEK:
        deviations[WEEK:] -= values[: last + 1 - WEEK]
    posterior = np.empty(last + 1, dtype=float)
    variance = q / (1.0 - phi * phi)
    for pos, observation in enumerate(deviations):
        if pos:
            variance = phi * phi * variance + q
        if np.isfinite(observation):
            gain = variance / (variance + r)
            variance *= 1.0 - gain
        posterior[pos] = variance
    factor2 = phi ** (2 * horizon)
    forecast_variance = factor2 * posterior[positions] + q * (1.0 - factor2) / (1.0 - phi * phi) + r
    sigma = np.sqrt(forecast_variance)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("B5 predictive mean/sigma must be finite and sigma positive")
    return mu, sigma


def _h16_contexts(history: pd.DataFrame, manifest: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Replay Phase C's exact h16 branch without constructing other horizons."""
    all_origins = pd.DatetimeIndex(history.index, name="origin")
    if all_origins.empty or all_origins.max() >= SEALED_BOUNDARY:
        raise ValueError("Phase E history crosses the sealed boundary")
    if history["gap_hour"].any():
        raise ValueError("Phase C origin-specific gap adapter is unavailable")
    folds, frozen = make_splits(all_origins, HORIZON, dev_frac=1.0, n_folds=3)
    if len(frozen):
        raise AssertionError("Frozen origins entered h16 development splits")
    x, provenance = _core_features(history, all_origins, HORIZON)
    targets = point_targets(history, all_origins, HORIZON)
    sequences = _sequences(history)
    sequence_valid = pd.Series(np.isfinite(sequences.to_numpy()).all(axis=1), index=sequences.index)
    before = pd.DatetimeIndex(targets["target_time"]).to_numpy() < np.datetime64(SEALED_BOUNDARY)
    eligible = (x.notna().all(axis=1).to_numpy()
                & targets["valid_target"].to_numpy(dtype=bool)
                & before
                & sequence_valid.reindex(all_origins, fill_value=False).to_numpy(dtype=bool))
    eligible_origins = all_origins[eligible]
    y = targets.loc[eligible_origins, "y"].rename("y")
    target_time = targets.loc[eligible_origins, "target_time"]
    profiles, profile_summary = _daily_profiles(history)
    context: dict[int, dict[str, Any]] = {}
    for fold_id, fold in enumerate(folds):
        train = fold.train.intersection(eligible_origins)
        validation = fold.validation.intersection(eligible_origins)
        fit, stop, cal, score = _partition(train, validation, HORIZON)
        tau = float(np.quantile(y.loc[fit].to_numpy(dtype=float), .95))
        d2, d2_summary = _d2_score_mask(profiles, target_time, fit, score)
        summary = {
            "horizon_quarters": HORIZON,
            "horizon_minutes": 240,
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
            "embargo_minutes": 240,
            **profile_summary,
            **d2_summary,
            "tau": tau,
        }
        expected = manifest.loc[manifest.fold.eq(fold_id)]
        if len(expected) != 1:
            raise ValueError(f"Phase C h16 fold manifest is missing fold {fold_id}")
        row = expected.iloc[0]
        for name, value in summary.items():
            recorded = row[name]
            if isinstance(value, (int, float, np.number)):
                equal = bool(np.isclose(float(value), float(recorded), rtol=0, atol=1e-10))
            else:
                equal = str(value) == str(recorded)
            if not equal:
                raise AssertionError(f"Phase C h16 fold {fold_id} {name} changed: {value} vs {recorded}")
        if not (target_time.loc[fit].max() < stop.min()
                and target_time.loc[stop].max() < cal.min()
                and target_time.loc[cal].max() < score.min()):
            raise AssertionError("Phase C h16 target-time embargo failed")
        if any((pd.DatetimeIndex(used.loc[eligible_origins]) > eligible_origins).any()
               for used in provenance.values()):
            raise AssertionError("Phase B core feature uses a future observation")
        context[fold_id] = {"y": y, "target_time": target_time, "fit": fit,
                            "stop": stop, "cal": cal, "score": score, "tau": tau,
                            "d2": d2, "summary": summary}
    return context


def _read_phase_c_h16(root: Path, artifacts: dict[str, dict]) -> tuple[pd.DataFrame, str]:
    digest = _checked_artifact(root, PREDICTIONS, artifacts)
    path = root / PREDICTIONS
    parquet = pq.ParquetFile(path)
    required = {"model", "horizon", "fold", "origin", "target_time", "y", "pred", "tau", "d2", "development_only"}
    if not required <= set(parquet.schema_arrow.names):
        raise ValueError("Phase C prediction schema lacks required columns")
    # Metadata is read and checked before any target/prediction numeric values.
    metadata = pq.read_table(path, columns=["model", "horizon", "fold", "origin", "target_time", "development_only"]).to_pandas()
    if (metadata.empty or metadata[["origin", "target_time"]].isna().any().any()
            or metadata.origin.ge(SEALED_BOUNDARY).any()
            or metadata.target_time.ge(SEALED_BOUNDARY).any()
            or not metadata.development_only.eq(True).all()):
        raise ValueError("Phase C prediction metadata crosses the sealed development boundary")
    if not metadata.horizon.isin(range(4, 17)).all() or not metadata.model.isin((*MAIN, "R1")).all():
        raise ValueError("Unexpected Phase C model/horizon metadata")
    frame = pq.read_table(path, columns=sorted(required), filters=[("horizon", "=", HORIZON)]).to_pandas()
    if len(frame) != int(metadata.horizon.eq(HORIZON).sum()):
        raise AssertionError("Phase C prediction artifact changed between metadata and value reads")
    if not frame.target_time.eq(frame.origin + pd.Timedelta(minutes=240)).all():
        raise ValueError("h16 targets are not four hours after origins")
    if frame.duplicated(["model", *KEY]).any():
        raise ValueError("Duplicate Phase C h16 prediction key")
    return frame, digest


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    error = frame.mu.to_numpy(dtype=float) - frame.y.to_numpy(dtype=float)
    peak = frame.is_peak.to_numpy(dtype=bool)
    return {"n": int(len(frame)), "peak_n": int(peak.sum()),
            "MAE": float(np.abs(error).mean()),
            "Peak_MAE": float(np.abs(error[peak]).mean()) if peak.any() else float("nan"),
            "RMSE": float(np.sqrt(np.square(error).mean()))}


def _same_metrics(got: dict, reference: pd.Series, label: str) -> None:
    for key, value in got.items():
        expected = float(reference[key])
        if not (np.isnan(value) and np.isnan(expected)) and not np.isclose(value, expected, rtol=0, atol=PARITY_TOLERANCE):
            raise AssertionError(f"Phase C B5 h16 {label} {key} parity failed: {value} vs {expected}")


def prepare_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return h16 B5 calibration/score rows after strict Phase C parity.

    D1 retains every finite B5 score row. The Phase C reported metrics used
    MAIN10's common finite cohort; both cohorts are audited separately.
    """
    root = Path(root).resolve()
    artifacts, completion_hashes = _checked_manifest(root)
    history, history_audit = load_history(root)
    fold_manifest = pd.read_csv(root / FOLD_MANIFEST)
    if set(fold_manifest.horizon_quarters.unique()) != set(range(4, 17)):
        raise ValueError("Phase C fold manifest horizon set changed")
    h16_manifest = fold_manifest.loc[fold_manifest.horizon_quarters.eq(HORIZON)]
    if len(h16_manifest) != 3 or set(h16_manifest.fold) != {0, 1, 2}:
        raise ValueError("Phase C h16 fold manifest must contain exactly three folds")
    contexts = _h16_contexts(history, h16_manifest)
    reference, predictions_sha = _read_phase_c_h16(root, artifacts)
    cal_parts, score_parts, parameter_audit = [], [], []
    power = history["power"]
    for fold, c in contexts.items():
        relative = Path(f"outputs/phase_c/models/B5_h16_f{fold}.joblib")
        digest = _checked_artifact(root, relative, artifacts)
        saved = joblib.load(root / relative)
        if saved.get("development_only") is not True or saved.get("fit_parameters_only") is not True:
            raise ValueError("Cached B5 fit was not marked fit-only development")
        bundle = saved["bundle"]
        phi, q, r = _validate_parameters(bundle)
        if bundle.get("development_only") is not True or bundle.get("fit_parameters_only") is not True:
            raise ValueError("B5 parameters lack fit-only development provenance")
        prevalence = float((c["y"].loc[c["fit"]] > c["tau"]).mean())
        if not 0 < prevalence < 1:
            raise ValueError("Fit-only peak prevalence is degenerate")
        for role, destination in (("cal", cal_parts), ("score", score_parts)):
            origins = c[role]
            mu, sigma = predict_kalman_distribution(bundle, power, origins)
            y = c["y"].loc[origins].to_numpy(dtype=float)
            target = pd.DatetimeIndex(c["target_time"].loc[origins])
            frame = pd.DataFrame({"fold": fold, "origin": origins, "target_time": target,
                                  "y": y, "tau": c["tau"], "mu": mu, "sigma": sigma,
                                  "p_raw": norm.sf((c["tau"] - mu) / sigma),
                                  "is_peak": y > c["tau"],
                                  "is_d2_novel_profile": c["d2"].loc[origins].to_numpy(dtype=bool),
                                  "p_climatology": prevalence, "partition": role,
                                  "model": MODEL, "horizon": HORIZON})
            if (not np.isfinite(frame[["y", "tau", "mu", "sigma", "p_raw", "p_climatology"]].to_numpy()).all()
                    or not frame.p_raw.between(0, 1).all()):
                raise ValueError(f"Nonfinite or invalid B5 distribution row in {role} fold {fold}")
            destination.append(frame)
        parameter_audit.append({"fold": fold, "path": relative.as_posix(), "sha256": digest,
                                "phi": phi, "q": q, "r": r, "tau_fit_only": c["tau"],
                                "prevalence_fit": prevalence,
                                "fit_count": len(c["fit"]), "cal_count": len(c["cal"]),
                                "score_count": len(c["score"]), "partition": c["summary"]})
    cal_frame = pd.concat(cal_parts, ignore_index=True)
    score_frame = pd.concat(score_parts, ignore_index=True)
    b5 = reference.loc[reference.model.eq(MODEL)].copy()
    joined = score_frame.merge(b5[[*KEY, "y", "pred", "tau", "d2"]], on=KEY,
                               how="outer", indicator=True, validate="one_to_one", suffixes=("", "_phase_c"))
    if not joined._merge.eq("both").all() or len(joined) != len(score_frame):
        raise AssertionError("Phase C B5 h16 score keys differ from Phase E")
    for left, right in (("y", "y_phase_c"), ("tau", "tau_phase_c"), ("mu", "pred")):
        delta = np.abs(joined[left].to_numpy(dtype=float) - joined[right].to_numpy(dtype=float))
        if not np.isfinite(delta).all() or delta.max(initial=0) > PARITY_TOLERANCE:
            raise AssertionError(f"Phase C B5 h16 prediction-level {left} parity failed")
    if not joined.is_d2_novel_profile.eq(joined.d2).all():
        raise AssertionError("Phase C h16 D2 score mask changed")
    main = reference.loc[reference.model.isin(MAIN)]
    finite = main.loc[np.isfinite(main.y) & np.isfinite(main.pred)]
    widths = finite.groupby(KEY, sort=False).model.nunique()
    common_keys = widths[widths.eq(len(MAIN))].index
    common = score_frame.set_index(KEY).loc[common_keys].reset_index()
    if len(common) != len(common_keys):
        raise AssertionError("Phase C MAIN10 common finite cohort is not unique")
    fold_reference = pd.read_csv(root / FOLD_METRICS)
    pooled_reference = pd.read_csv(root / POOLED_METRICS)
    common_metrics: dict[str, Any] = {}
    for dataset, subset in (("D1", common), ("D2", common.loc[common.is_d2_novel_profile])):
        per_fold = {}
        for fold in (0, 1, 2):
            got = _metrics(subset.loc[subset.fold.eq(fold)])
            expected = fold_reference.loc[(fold_reference.dataset.eq(dataset))
                                          & (fold_reference.model.eq(MODEL))
                                          & (fold_reference.horizon.eq(HORIZON))
                                          & (fold_reference.fold.eq(fold))]
            if len(expected) != 1:
                raise ValueError(f"Missing Phase C B5 h16 {dataset} fold metric")
            _same_metrics(got, expected.iloc[0], f"{dataset} fold {fold}")
            per_fold[str(fold)] = got
        pooled = _metrics(subset)
        expected = pooled_reference.loc[(pooled_reference.dataset.eq(dataset))
                                        & (pooled_reference.model.eq(MODEL))
                                        & (pooled_reference.horizon.eq(HORIZON))]
        if len(expected) != 1:
            raise ValueError(f"Missing Phase C B5 h16 {dataset} pooled metric")
        _same_metrics(pooled, expected.iloc[0], f"{dataset} pooled")
        common_metrics[dataset] = {"pooled": pooled, "folds": per_fold}
    audit = {"development_only": True, "model": MODEL, "horizon": HORIZON,
             "horizon_minutes": 240, "point_prediction_max_abs_difference": float(
                 np.abs(joined.mu.to_numpy(dtype=float) - joined.pred.to_numpy(dtype=float)).max()),
             "point_prediction_tolerance": PARITY_TOLERANCE,
             "all_b5_score_metrics": _metrics(score_frame),
             "all_b5_d2_metrics": _metrics(score_frame.loc[score_frame.is_d2_novel_profile]),
             "phase_c_main10_common_metrics": common_metrics,
             "all_b5_score_count": int(len(score_frame)),
             "phase_c_main10_common_count": int(len(common)),
             "history": history_audit, "kalman_fit_artifacts": parameter_audit,
             "source_sha256": {"raw": history_audit["raw_sha256"],
                               "fold_manifest": _sha256(root / FOLD_MANIFEST),
                               "phase_c_predictions": predictions_sha,
                               "phase_c_cache_manifest": completion_hashes[MANIFEST.relative_to("outputs/phase_c").as_posix()],
                               "phase_c_fold_metrics": completion_hashes[FOLD_METRICS.relative_to("outputs/phase_c").as_posix()],
                               "phase_c_pooled_metrics": completion_hashes[POOLED_METRICS.relative_to("outputs/phase_c").as_posix()]},
             "d2_refit": False, "historical_final_artifact_read": False,
             "final_holdout_parsed": False, "parity_passed": True}
    return cal_frame, score_frame, audit
