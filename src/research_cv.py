"""Reconstruct the sealed development CV partitions for Phase 2 research.

This mirrors ``training._fold`` up to fitting, using the original all-safe-origin
grid before ``make_splits``. Passing only origin values with observable targets
would move the folds and is explicitly forbidden here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features
from .session_data import SEALED_BOUNDARY
from .split import make_splits
from .targets import point_targets, training_peak_threshold
from .training import _partition, _valid_mask


REQUIRED_FOLD = ("fold", "horizon", "tau", "n_fit", "n_stop", "n_cal", "n_score",
                 "fit_end", "stop_end", "cal_end", "score_start", "score_end")


def build_contexts(history: pd.DataFrame, cfg: dict, manifest: dict,
                   horizons=(4, 16, 96)) -> dict[tuple[int, int], dict]:
    """Return ``(horizon, fold) -> context`` on the original sealed CV grid.

    Keys include ``x``, ``x0``, ``all_origins``, ``fit``, ``stop``, ``cal``,
    ``score``, ``targets``, ``tau``, ``expected`` (including chosen_params),
    ``summary``, ``valid`` and ``valid0``. Measured feature provenance remains
    in ``x.attrs['latest_observation_by_feature']``.
    """
    if "selection" not in manifest or "folds" not in manifest:
        raise ValueError("Original development selection and fold manifest required")
    if "ts_end" in history.columns:
        history = history.set_index("ts_end")
    if (not isinstance(history.index, pd.DatetimeIndex) or history.empty or
            not history.index.is_unique or not history.index.is_monotonic_increasing or
            history.index.max() >= SEALED_BOUNDARY):
        raise ValueError("Phase 2 requires sorted, unique, sealed development history")
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    if boundary != SEALED_BOUNDARY:
        raise ValueError("Configured test boundary differs from sealed boundary")
    requested = tuple(int(h) for h in horizons)
    if len(requested) != len(set(requested)) or any(h not in (1, 4, 16, 96) for h in requested):
        raise ValueError("Unexpected or repeated horizon")
    expected = {(int(row["horizon"]), int(row["fold"])): row for row in manifest["folds"]}
    if len(expected) != len(manifest["folds"]):
        raise ValueError("Duplicate original fold metadata")
    # Do not trim the final h origin positions first: make_splits owns the
    # horizon embargo and the original score grid depends on these positions.
    origins = history.index[history.index < boundary]
    out: dict[tuple[int, int], dict] = {}
    for horizon in requested:
        folds, sealed_test = make_splits(origins, horizon, dev_frac=1.0,
                                         n_folds=int(cfg["split"]["n_folds"]))
        if len(sealed_test):
            raise AssertionError("Development CV yielded a test split")
        for fold_id, fold in enumerate(folds):
            recorded = expected.get((horizon, fold_id))
            if recorded is None:
                raise ValueError(f"No original metadata for h{horizon} fold{fold_id}")
            raw_train, raw_validation = fold.train, fold.validation
            all_origins = raw_train.append(raw_validation)
            targets = point_targets(history, all_origins, horizon)
            x0, used0 = build_features(history, all_origins, horizon, {**cfg, "_tau": None})
            if not x0.index.equals(all_origins):
                x0 = x0.reindex(all_origins)
            valid0 = _valid_mask(x0, used0, all_origins, targets, history)
            good = all_origins[valid0]
            train, validation = raw_train.intersection(good), raw_validation.intersection(good)
            fit, stop, cal, score = _partition(train, validation, horizon)
            tau = training_peak_threshold(history, fit, cfg.get("peak", {}).get("tau_quantile", .95))
            for _ in range(3):
                x, used = build_features(history, all_origins, horizon, {**cfg, "_tau": tau})
                valid = _valid_mask(x, used, all_origins, targets, history)
                good = all_origins[valid]
                fit, stop, cal, score = (part.intersection(good) for part in (fit, stop, cal, score))
                if min(map(len, (fit, stop, cal, score))) < 30:
                    raise ValueError("Original fold lost too many rows after peak-history mask")
                exact = training_peak_threshold(history, fit, cfg.get("peak", {}).get("tau_quantile", .95))
                if exact == tau:
                    break
                tau = exact
            else:
                raise AssertionError("Original peak threshold failed to stabilize")
            summary = {"fold": fold_id, "horizon": horizon, "tau": tau,
                       "n_fit": len(fit), "n_stop": len(stop), "n_cal": len(cal), "n_score": len(score),
                       "fit_end": str(fit.max()), "stop_end": str(stop.max()),
                       "cal_end": str(cal.max()), "score_start": str(score.min()),
                       "score_end": str(score.max())}
            different = [key for key in REQUIRED_FOLD if summary[key] != recorded[key]]
            if different:
                raise AssertionError(f"Original fold grid mismatch h{horizon}/f{fold_id}: {different}")
            if not np.all(pd.DatetimeIndex(targets.loc[score, "target_time"]) < boundary):
                raise AssertionError("Score target reached sealed holdout boundary")
            provenance = x.attrs.get("latest_observation_by_feature", {})
            for name, stamps in provenance.items():
                available = pd.DatetimeIndex(stamps)
                if len(available) != len(all_origins) or (available > all_origins).any():
                    raise AssertionError(f"Existing feature {name} used future observation")
            out[(horizon, fold_id)] = {
                "raw_train": raw_train, "raw_validation": raw_validation,
                "all_origins": all_origins, "x0": x0, "x": x,
                "fit": fit, "stop": stop, "cal": cal, "score": score,
                "targets": targets, "tau": tau, "valid0": valid0, "valid": valid,
                "expected": recorded, "summary": summary,
            }
    return out


def _paired_residual_gain(original: pd.DataFrame, candidate: pd.DataFrame,
                          incumbent_name: str, horizon: int) -> dict:
    """Peak-only paired improvement and conservative centered day-bootstrap p."""
    keys = ["origin", "target_time", "horizon", "fold"]
    incumbent = original.loc[original.horizon.eq(horizon) & original.model.eq(incumbent_name),
                             keys+["y", "pred", "tau"]].rename(columns={"pred": "incumbent_pred"})
    challenger = candidate.loc[candidate.horizon.eq(horizon),
                               keys+["pred"]].rename(columns={"pred": "candidate_pred"})
    if incumbent.duplicated(keys).any() or challenger.duplicated(keys).any():
        raise ValueError("Residual comparison requires unique original OOF keys")
    pair = incumbent.merge(challenger, on=keys, how="inner", validate="one_to_one")
    pair = pair.loc[np.isfinite(pair.y) & np.isfinite(pair.tau) &
                    np.isfinite(pair.incumbent_pred) & np.isfinite(pair.candidate_pred)].copy()
    paired_all = len(pair)
    pair["incumbent_abs_error"] = (pair.y-pair.incumbent_pred).abs()
    pair["candidate_abs_error"] = (pair.y-pair.candidate_pred).abs()
    peaks = pair.loc[pair.y.gt(pair.tau)].copy()
    peaks["gain"] = peaks.incumbent_abs_error-peaks.candidate_abs_error
    effect = float(peaks.gain.mean()) if len(peaks) else np.nan
    fold_rows = []
    for fold in range(3):
        part = peaks.loc[peaks.fold.eq(fold)]
        fold_rows.append({"fold": fold, "n_peak": len(part),
                          "incumbent_peak_mae": float(part.incumbent_abs_error.mean()) if len(part) else np.nan,
                          "candidate_peak_mae": float(part.candidate_abs_error.mean()) if len(part) else np.nan,
                          "peak_mae_gain": float(part.gain.mean()) if len(part) else np.nan})
    final = fold_rows[2]
    final_ratio = (final["candidate_peak_mae"]/final["incumbent_peak_mae"]
                   if final["n_peak"] and final["incumbent_peak_mae"] > 0 else np.nan)
    stable = bool(np.isfinite(final_ratio) and final_ratio < 1.10)
    ci_low = ci_high = np.nan
    p = 1.
    valid_draws = 0
    if len(peaks):
        dates = pd.DatetimeIndex(peaks.target_time).normalize()
        by_day = peaks.assign(day=dates).groupby("day", sort=True).gain.agg(["sum", "count"])
        sums, counts = by_day["sum"].to_numpy(float), by_day["count"].to_numpy(int)
        rng = np.random.default_rng(42)
        chosen = rng.integers(0, len(sums), size=(1000, len(sums)))
        denominator = counts[chosen].sum(axis=1)
        sampled = np.divide(sums[chosen].sum(axis=1), denominator,
                            out=np.full(1000, np.nan), where=denominator > 0)
        finite = sampled[np.isfinite(sampled)]
        valid_draws = len(finite)
        if valid_draws >= 2 and np.isfinite(effect):
            ci_low, ci_high = map(float, np.quantile(finite, [.025, .975]))
            # Undefined draws count conservatively as non-rejections.
            p = float((1 + np.count_nonzero(finite-effect >= effect) + 1000-valid_draws)/1001)
    return {"analysis_id": "M1", "hypothesis_id": f"M1_h{horizon}", "horizon": horizon,
            "incumbent_model": incumbent_name, "candidate_model": "lgbm_residual_cbl",
            "estimate": effect, "ci_low": ci_low, "ci_high": ci_high, "p_raw": p,
            "n_positive_folds": sum(bool(np.isfinite(row["peak_mae_gain"]) and row["peak_mae_gain"] > 0)
                                    for row in fold_rows),
            "eligible": bool(np.isfinite(effect) and np.isfinite(ci_low)), "null_value": 0.,
            "n_paired_all": paired_all, "n_paired_peak": len(peaks),
            "last_fold_peak_mae_ratio": final_ratio,
            "last_fold_stability_pass": stable,
            "reference_only": horizon == 4, "selection_used": False,
            "bootstrap_valid_draws": valid_draws,
            "bootstrap_undefined_draws": 1000-valid_draws,
            "fold_effects": fold_rows}


def fit_residual_candidates(history: pd.DataFrame, cfg: dict, manifest: dict,
                            original_oof: pd.DataFrame, output_dir="outputs",
                            contexts: dict[tuple[int, int], dict] | None = None) -> dict:
    """Fit the 9 fixed M1 folds and return reusable development OOF/evidence.

    Writes only development candidate artifacts under ``output_dir``. The
    original OOF/selection are read-only; parent integration decides adoption.
    """
    from pathlib import Path
    import joblib

    from .models.residual import MODEL_NAME, fit_residual_fold

    boundary = SEALED_BOUNDARY
    if any(pd.to_datetime(original_oof[column]).ge(boundary).any()
           for column in ("origin", "target_time")):
        raise ValueError("Original OOF crosses sealed development boundary")
    if int(cfg.get("bootstrap", {}).get("n", 1000)) != 1000 or int(cfg.get("seed", 42)) != 42:
        raise ValueError("Phase 2 registered 1000 date blocks and seed 42")
    selected_horizons = (4, 16, 96)
    contexts = contexts if contexts is not None else build_contexts(history, cfg, manifest, selected_horizons)
    out = Path(output_dir)
    (out/"analysis_p2").mkdir(parents=True, exist_ok=True)
    (out/"predictions").mkdir(parents=True, exist_ok=True)
    (out/"models").mkdir(parents=True, exist_ok=True)
    frames, audit_parts, metadata = [], [], []
    model_paths = []
    for horizon in selected_horizons:
        for fold in range(3):
            context = contexts.get((horizon, fold))
            if context is None:
                raise ValueError(f"Missing reconstructed h{horizon} fold{fold} context")
            fitted = fit_residual_fold(history, cfg, context, horizon, fold)
            frames.append(fitted.predictions)
            audit_parts.append(fitted.audit)
            metadata.append(fitted.metadata)
            path = out/"models"/f"development_residual_h{horizon}_fold{fold}.joblib"
            joblib.dump({"model": fitted.model, **fitted.metadata}, path)
            model_paths.append(str(path))
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.model.nunique() != 1 or predictions.model.iloc[0] != MODEL_NAME:
        raise AssertionError("Unexpected M1 model name")
    if predictions.target_time.ge(boundary).any() or predictions.origin.ge(boundary).any():
        raise AssertionError("M1 wrote a sealed target")
    audit = pd.concat(audit_parts, ignore_index=True)
    comparisons = []
    for horizon in selected_horizons:
        incumbent_name = manifest["selection"]["by_horizon"][str(horizon)]["point_model"]
        comparisons.append(_paired_residual_gain(original_oof, predictions, incumbent_name, horizon))
    comparison_frame = pd.DataFrame(comparisons)
    fold_comparison = pd.DataFrame(
        {"horizon": row["horizon"], "incumbent_model": row["incumbent_model"],
         "candidate_model": row["candidate_model"], **fold_row}
        for row in comparisons for fold_row in row["fold_effects"])
    comparison_frame = comparison_frame.drop(columns="fold_effects")
    meta_frame = pd.DataFrame(metadata)
    paths = {"predictions": str(out/"predictions/p2_residual_oof.csv"),
             "audit": str(out/"analysis_p2/M1_baseline_exclusion.csv"),
             "comparisons": str(out/"analysis_p2/M1_comparisons.csv"),
             "fold_comparison": str(out/"analysis_p2/M1_fold_comparison.csv"),
             "fit_metadata": str(out/"analysis_p2/M1_fit_metadata.csv"),
             "model_bundles": model_paths}
    predictions.to_csv(paths["predictions"], index=False)
    audit.to_csv(paths["audit"], index=False)
    comparison_frame.to_csv(paths["comparisons"], index=False)
    fold_comparison.to_csv(paths["fold_comparison"], index=False)
    meta_frame.to_csv(paths["fit_metadata"], index=False)
    return {"predictions": predictions, "audit": audit, "comparisons": comparison_frame,
            "fold_comparison": fold_comparison, "fit_metadata": meta_frame,
            "paths": paths, "contexts": contexts}
