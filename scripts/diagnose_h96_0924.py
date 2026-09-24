"""Development-only audit of the 24-hour point forecast on the frozen CV grid.

Run from any directory with ``python scripts/diagnose_h96_0924.py``. This tool
never loads the unrestricted source frame, changes model selection, or opens a
final-test artifact. Its sole data entry point is load_development_history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.bootstrap import paired_mae_improvement
from src.evaluate import score_predictions
from src.features import build_features
from src.models.lgbm_point import fit_point
from src.split import make_splits
from src.targets import point_targets, training_peak_threshold
from src.training import _partition, _valid_mask, _cutoff


FOUR = ["target_slot_1d_ago", "target_slot_7d_ago", "target_weekday", "target_hour"]
KEY = ["target_slot_1d_ago", "target_slot_7d_ago", "lag_96", "lag_672"]
REQUIRED_FOLD = ["fold", "horizon", "tau", "n_fit", "n_stop", "n_cal", "n_score",
                 "fit_end", "stop_end", "cal_end", "score_start", "score_end"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary(cfg: dict) -> pd.Timestamp:
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    if boundary != pd.Timestamp("2021-08-09 09:45:00"):
        raise AssertionError("The preregistered frozen-test boundary changed")
    return boundary


def _assert_development_frame(df: pd.DataFrame, boundary: pd.Timestamp) -> None:
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise AssertionError("Safe loader must return sorted unique timestamp-indexed development history")
    if df.empty or df.index.max() >= boundary:
        raise AssertionError("Development loader exposed the frozen test period")


def _read_oof(path: Path, boundary: pd.Timestamp) -> pd.DataFrame:
    # Inspect timestamps without target values first. Do not deserialize the
    # full prediction table unless every stored origin and target is pre-test.
    stamps = pd.read_csv(path, usecols=["origin", "target_time"], parse_dates=["origin", "target_time"])
    if stamps.empty or (stamps.origin >= boundary).any() or (stamps.target_time >= boundary).any():
        raise AssertionError("Development OOF contains a frozen-test timestamp")
    columns = ["origin", "target_time", "horizon", "fold", "model", "y", "pred", "tau", "alert"]
    oof = pd.read_csv(path, usecols=columns, parse_dates=["origin", "target_time"], low_memory=False)
    if (oof.origin >= boundary).any() or (oof.target_time >= boundary).any():
        raise AssertionError("OOF changed between timestamp preflight and data read")
    return oof


def _reconstruct(df: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
                 fold_id: int, cfg: dict, expected: dict) -> dict:
    """Mirror the original fold's partition and masks without fitting models."""
    folds, test = make_splits(origins, horizon, dev_frac=1.0,
                              n_folds=int(cfg["split"]["n_folds"]))
    if len(test):
        raise AssertionError("A test grid was returned from development-only splitting")
    raw_train, raw_validation = folds[fold_id].train, folds[fold_id].validation
    all_origins = raw_train.append(raw_validation)
    targets = point_targets(df, all_origins, horizon)
    x0, used0 = build_features(df, all_origins, horizon, {**cfg, "_tau": None})
    if not x0.index.equals(all_origins):
        x0 = x0.reindex(all_origins)
    valid0 = _valid_mask(x0, used0, all_origins, targets, df)
    good = all_origins[valid0]
    train, validation = raw_train.intersection(good), raw_validation.intersection(good)
    fit, stop, cal, score = _partition(train, validation, horizon)
    tau = training_peak_threshold(df, fit, cfg.get("peak", {}).get("tau_quantile", .95))
    for _ in range(3):
        x, used = build_features(df, all_origins, horizon, {**cfg, "_tau": tau})
        valid = _valid_mask(x, used, all_origins, targets, df)
        available = all_origins[valid]
        fit, stop, cal, score = (part.intersection(available) for part in (fit, stop, cal, score))
        exact_tau = training_peak_threshold(df, fit, cfg.get("peak", {}).get("tau_quantile", .95))
        if exact_tau == tau:
            break
        tau = exact_tau
    else:
        raise AssertionError(f"h={horizon} fold={fold_id} threshold failed to stabilize")
    actual = {"fold": fold_id, "horizon": horizon, "tau": tau,
              "n_fit": len(fit), "n_stop": len(stop), "n_cal": len(cal), "n_score": len(score),
              "fit_end": str(fit.max()), "stop_end": str(stop.max()),
              "cal_end": str(cal.max()), "score_start": str(score.min()),
              "score_end": str(score.max())}
    if any(actual[key] != expected[key] for key in REQUIRED_FOLD):
        raise AssertionError(f"Fold grid differs from original training: {actual} versus {expected}")
    if (targets.loc[score, "target_time"] >= _boundary(cfg)).any():
        raise AssertionError("Score target crossed the frozen-test boundary")
    return {"raw_train": raw_train, "raw_validation": raw_validation,
            "fit": fit, "stop": stop, "cal": cal, "score": score,
            "all_origins": all_origins, "x0": x0, "x": x, "targets": targets,
            "tau": tau, "valid0": valid0, "valid": valid,
            "summary": actual, "expected": expected}


def _missing_rows(context: dict, horizon: int, fold_id: int) -> list[dict]:
    rows = []
    x = context["x"]
    for name in ("raw_train", "raw_validation", "fit", "stop", "cal", "score"):
        indices = context[name]
        block = x.loc[indices]
        for feature in x.columns:
            count = int(block[feature].isna().sum())
            rows.append({"horizon": horizon, "fold": fold_id, "partition": name,
                         "feature": feature, "n": len(indices), "missing_n": count,
                         "missing_rate": count / len(indices) if len(indices) else np.nan})
    return rows


def _fold_summary(context: dict, df: pd.DataFrame) -> dict:
    row = dict(context["summary"])
    raw_train, raw_val = context["raw_train"], context["raw_validation"]
    targets, x = context["targets"], context["x"]
    repaired = df.get("time_repaired", pd.Series(False, index=df.index)).fillna(True).astype(bool)
    direct = ["current", "lag_96", "lag_672", "target_slot_1d_ago", "target_slot_7d_ago"]
    all_origins = context["all_origins"]
    dependency = np.zeros(len(all_origins), dtype=bool)
    for name in direct:
        stamps = x.attrs["latest_observation_by_feature"][name]
        masked = repaired.reindex(pd.DatetimeIndex(stamps)).fillna(False).to_numpy(dtype=bool)
        if np.isfinite(x[name].to_numpy(dtype=float)[masked]).any():
            raise AssertionError(f"Repaired observation leaked into {name}")
        dependency |= masked
    for width, feature in ((96, "recent_96_mean"), (672, "peak_count_7d")):
        repaired_in_window = repaired.rolling(width, min_periods=width).sum().reindex(all_origins).fillna(0).gt(0).to_numpy()
        if np.isfinite(x[feature].to_numpy(dtype=float)[repaired_in_window]).any():
            raise AssertionError(f"Repaired interval leaked into {feature}")
    for label, index in (("raw_train", raw_train), ("raw_validation", raw_val)):
        locs = all_origins.get_indexer(index)
        row[f"n_{label}"] = len(index)
        row[f"n_{label}_initial_valid"] = int(context["valid0"][locs].sum())
        row[f"n_{label}_final_valid"] = int(context["valid"][locs].sum())
        row[f"n_{label}_target_repaired"] = int(targets.loc[index, "target_repaired"].sum())
        row[f"n_{label}_direct_repaired_dependency"] = int(dependency[locs].sum())
        row[f"n_{label}_key_missing"] = int(x.loc[index, KEY].isna().any(axis=1).sum())
    return row


def _uniform_origins(origins: pd.DatetimeIndex, count: int = 20) -> pd.DatetimeIndex:
    if len(origins) < count:
        raise ValueError("At least 20 h96 score origins are required for a uniform manual audit")
    return origins[np.linspace(0, len(origins) - 1, count, dtype=int)]


def _manual_audit(df: pd.DataFrame, contexts: dict[int, dict], oof: pd.DataFrame,
                  boundary: pd.Timestamp) -> pd.DataFrame:
    reference = oof.loc[(oof.horizon == 96) & (oof.model == "lgbm")].sort_values("origin")
    if reference.origin.duplicated().any():
        raise AssertionError("h96 LightGBM OOF has duplicate score origins")
    picks = _uniform_origins(pd.DatetimeIndex(reference.origin))
    records = []
    for origin in picks:
        rec = reference.loc[reference.origin == origin].iloc[0]
        context = contexts[int(rec.fold)]
        target_time = origin + pd.Timedelta(days=1)
        if target_time >= boundary or target_time != rec.target_time:
            raise AssertionError("Manual target timestamp does not match the h96 grid")
        xrow = context["x"].loc[origin]
        actual = float(df.loc[target_time, "power"])
        if not np.isclose(float(rec.y), actual, rtol=0, atol=1e-10):
            raise AssertionError("Stored OOF h96 label differs from source power at origin+24h")
        item = {"fold": int(rec.fold), "origin": str(origin), "target_time": str(target_time),
                "stored_y": float(rec.y), "source_y": actual,
                "target_repaired": bool(df.loc[target_time, "time_repaired"]),
                "target_weekday": int(xrow.target_weekday), "target_hour": int(xrow.target_hour)}
        if item["target_repaired"] or item["target_weekday"] != target_time.dayofweek or item["target_hour"] != target_time.hour:
            raise AssertionError("Target quality/calendar mismatch in the manual audit")
        times = {"current": origin, "lag_96": origin - pd.Timedelta(days=1),
                 "lag_672": origin - pd.Timedelta(days=7),
                 "target_slot_1d_ago": target_time - pd.Timedelta(days=1),
                 "target_slot_7d_ago": target_time - pd.Timedelta(days=7)}
        for feature, stamp in times.items():
            if stamp > origin or stamp >= boundary:
                raise AssertionError(f"Future observation in {feature}")
            observed = float(df.loc[stamp, "power"])
            item[f"{feature}_time"] = str(stamp)
            item[f"{feature}_feature"] = float(xrow[feature])
            item[f"{feature}_source"] = observed
            if bool(df.loc[stamp, "time_repaired"]) or not np.isclose(float(xrow[feature]), observed, rtol=0, atol=1e-10):
                raise AssertionError(f"{feature} value/masking mismatch at {origin}")
        if item["target_slot_1d_ago_time"] != item["origin"]:
            raise AssertionError("h96 yesterday-target slot must equal the known current reading")
        records.append(item)
    return pd.DataFrame(records)


def _prediction_frame(context: dict, predictions: np.ndarray, name: str,
                      cutoff: float, fold_id: int) -> pd.DataFrame:
    score = context["score"]
    target = context["targets"].loc[score]
    return pd.DataFrame({"origin": score, "target_time": target.target_time.to_numpy(),
                         "horizon": 96, "fold": fold_id, "model": name,
                         "y": target.y.to_numpy(dtype=float), "pred": np.asarray(predictions, dtype=float),
                         "tau": context["tau"], "alert": np.asarray(predictions, dtype=float) > cutoff})


def _compare_models(contexts: dict[int, dict], oof: pd.DataFrame, cfg: dict,
                    root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, curves, all_prediction_frames = [], [], {}
    for fold_id, context in contexts.items():
        x, target, tau = context["x"], context["targets"], context["tau"]
        fit, stop, cal, score = (context[name] for name in ("fit", "stop", "cal", "score"))
        y = target.y
        original_oof = oof.loc[(oof.horizon == 96) & (oof.fold == fold_id) & (oof.model == "lgbm")].sort_values("origin")
        if not pd.DatetimeIndex(original_oof.origin).equals(score):
            raise AssertionError(f"Original h96 fold {fold_id} score grid changed")
        bundle_path = root / f"outputs/models/development_h96_fold{fold_id}_lgbm.joblib"
        bundle = joblib.load(bundle_path)
        if bundle["horizon"] != 96 or bundle["fold"] != fold_id or not pd.DatetimeIndex(bundle["score_origins"]).equals(score):
            raise AssertionError("Saved model metadata differs from reconstructed score grid")
        original = bundle["model"]
        original_pred = original.predict(x.loc[score, bundle["feature_names"]])
        max_abs = float(np.max(np.abs(original_pred - original_oof.pred.to_numpy(dtype=float))))
        if max_abs > 1e-10:
            raise AssertionError(f"Original h96 model prediction does not reproduce its OOF: {max_abs}")
        original_fresh = fit_point(x.loc[fit, bundle["feature_names"]], y.loc[fit],
                                   x.loc[stop, bundle["feature_names"]], y.loc[stop], cfg,
                                   peak_threshold=tau, params=context["expected"]["chosen_params"])
        fresh_max_abs = float(np.max(np.abs(original_fresh.predict(x.loc[score, bundle["feature_names"]]) -
                                            original_oof.pred.to_numpy(dtype=float))))
        if fresh_max_abs > 1e-10 or original_fresh.best_iteration_ != original.best_iteration_:
            raise AssertionError(f"Fresh full h96 fit differs from stored OOF/model, fold={fold_id}, "
                                 f"max_abs={fresh_max_abs}, best={original_fresh.best_iteration_}")
        full = original_oof[["origin", "target_time", "horizon", "fold", "model", "y", "pred", "tau", "alert"]].copy()
        all_prediction_frames.setdefault("original_full", []).append(full)
        curve = original.evals_result_["valid_0"]["l1"]
        curves.extend({"fold": fold_id, "model": "original_full", "iteration": i + 1,
                       "stop_l1": float(loss), "best_iteration": int(original.best_iteration_)}
                      for i, loss in enumerate(curve))
        rows.append({"fold": fold_id, "model": "original_full", "best_iteration": int(original.best_iteration_),
                     "recorded_iterations": len(curve), "fit_n": len(fit), "stop_n": len(stop),
                     "cal_n": len(cal), "score_n": len(score), "oof_max_abs_difference": max_abs,
                     "fresh_refit_max_abs_difference": fresh_max_abs,
                     "fresh_best_iteration": int(original_fresh.best_iteration_),
                     **score_predictions(full)})

        plain_path = root / f"outputs/models/development_h96_fold{fold_id}_lgbm_no_holiday.joblib"
        plain_bundle = joblib.load(plain_path)
        plain_oof = oof.loc[(oof.horizon == 96) & (oof.fold == fold_id) &
                            (oof.model == "lgbm_no_holiday")].sort_values("origin")
        if plain_bundle["horizon"] != 96 or plain_bundle["fold"] != fold_id or not pd.DatetimeIndex(
                plain_bundle["score_origins"]).equals(score) or not pd.DatetimeIndex(plain_oof.origin).equals(score):
            raise AssertionError("Saved no-holiday model metadata differs from reconstructed score grid")
        plain_model = plain_bundle["model"]
        plain_pred = plain_model.predict(x.loc[score, plain_bundle["feature_names"]])
        plain_max_abs = float(np.max(np.abs(plain_pred - plain_oof.pred.to_numpy(dtype=float))))
        if plain_max_abs > 1e-10:
            raise AssertionError(f"No-holiday h96 model prediction does not reproduce its OOF: {plain_max_abs}")
        plain_fresh = fit_point(x.loc[fit, plain_bundle["feature_names"]], y.loc[fit],
                                x.loc[stop, plain_bundle["feature_names"]], y.loc[stop], cfg,
                                peak_threshold=tau, params=context["expected"]["chosen_params"])
        plain_fresh_max_abs = float(np.max(np.abs(plain_fresh.predict(x.loc[score, plain_bundle["feature_names"]]) -
                                                  plain_oof.pred.to_numpy(dtype=float))))
        if plain_fresh_max_abs > 1e-10 or plain_fresh.best_iteration_ != plain_model.best_iteration_:
            raise AssertionError(f"Fresh no-holiday h96 fit differs from stored OOF/model, fold={fold_id}, "
                                 f"max_abs={plain_fresh_max_abs}, best={plain_fresh.best_iteration_}")
        plain = plain_oof[["origin", "target_time", "horizon", "fold", "model", "y", "pred", "tau", "alert"]].copy()
        all_prediction_frames.setdefault("original_no_holiday", []).append(plain)
        plain_curve = plain_model.evals_result_["valid_0"]["l1"]
        curves.extend({"fold": fold_id, "model": "original_no_holiday", "iteration": i + 1,
                       "stop_l1": float(loss), "best_iteration": int(plain_model.best_iteration_)}
                      for i, loss in enumerate(plain_curve))
        rows.append({"fold": fold_id, "model": "original_no_holiday",
                     "best_iteration": int(plain_model.best_iteration_),
                     "recorded_iterations": len(plain_curve), "fit_n": len(fit), "stop_n": len(stop),
                     "cal_n": len(cal), "score_n": len(score), "oof_max_abs_difference": plain_max_abs,
                     "fresh_refit_max_abs_difference": plain_fresh_max_abs,
                     "fresh_best_iteration": int(plain_fresh.best_iteration_),
                     **score_predictions(plain)})

        # A: use the target's preceding-week reading as-is, including its
        # original calibration/score boundary for the episode-alert cutoff.
        weekly_cal = x.loc[cal, "target_slot_7d_ago"].to_numpy(dtype=float)
        weekly_score = x.loc[score, "target_slot_7d_ago"].to_numpy(dtype=float)
        cutoff_a = _cutoff(y.loc[cal].to_numpy(dtype=float), weekly_cal, tau,
                           target.loc[cal, "target_time"])
        weekly = _prediction_frame(context, weekly_score, "A_weekly_slot", cutoff_a, fold_id)
        seasonal_oof = oof.loc[(oof.horizon == 96) & (oof.fold == fold_id) & (oof.model == "s2_week")].sort_values("origin")
        if not pd.DatetimeIndex(seasonal_oof.origin).equals(score) or not np.allclose(
                weekly_score, seasonal_oof.pred.to_numpy(dtype=float), rtol=0, atol=1e-10):
            raise AssertionError("Weekly feature differs from the existing weekly-naive OOF")
        all_prediction_frames.setdefault("A_weekly_slot", []).append(weekly)
        rows.append({"fold": fold_id, "model": "A_weekly_slot", "best_iteration": np.nan,
                     "recorded_iterations": 0, "fit_n": len(fit), "stop_n": len(stop),
                     "cal_n": len(cal), "score_n": len(score), "oof_max_abs_difference": 0.0,
                     **score_predictions(weekly)})

        # B: no new tuning. Reuse the fold's already selected parameters,
        # sample grid, target, early-stop/calibration/scoring partitions.
        params = context["expected"]["chosen_params"]
        reduced = fit_point(x.loc[fit, FOUR], y.loc[fit], x.loc[stop, FOUR], y.loc[stop],
                            cfg, peak_threshold=tau, params=params)
        reduced_cal = reduced.predict(x.loc[cal, FOUR])
        cutoff_b = _cutoff(y.loc[cal].to_numpy(dtype=float), reduced_cal, tau,
                           target.loc[cal, "target_time"])
        reduced_score = reduced.predict(x.loc[score, FOUR])
        four = _prediction_frame(context, reduced_score, "B_four_features", cutoff_b, fold_id)
        all_prediction_frames.setdefault("B_four_features", []).append(four)
        curve = reduced.evals_result_["valid_0"]["l1"]
        curves.extend({"fold": fold_id, "model": "B_four_features", "iteration": i + 1,
                       "stop_l1": float(loss), "best_iteration": int(reduced.best_iteration_)}
                      for i, loss in enumerate(curve))
        rows.append({"fold": fold_id, "model": "B_four_features", "best_iteration": int(reduced.best_iteration_),
                     "recorded_iterations": len(curve), "fit_n": len(fit), "stop_n": len(stop),
                     "cal_n": len(cal), "score_n": len(score), "oof_max_abs_difference": np.nan,
                     **score_predictions(four)})
        print(f"[h96] fold={fold_id} original iteration={original.best_iteration_}, "
              f"four-feature iteration={reduced.best_iteration_}", flush=True)

    all_frames = {name: pd.concat(parts, ignore_index=True) for name, parts in all_prediction_frames.items()}
    paired = {}
    for name in ("A_weekly_slot", "B_four_features"):
        paired[name] = paired_mae_improvement(all_frames["original_full"], all_frames[name],
                                               peak_only=True, n=int(cfg["bootstrap"]["n"]),
                                               seed=int(cfg["seed"]))
    pooled_rows = []
    for name, frame in all_frames.items():
        pooled_rows.append({"fold": "pooled", "model": name, "best_iteration": np.nan,
                            "recorded_iterations": np.nan, "fit_n": sum(len(c["fit"]) for c in contexts.values()),
                            "stop_n": sum(len(c["stop"]) for c in contexts.values()),
                            "cal_n": sum(len(c["cal"]) for c in contexts.values()),
                            "score_n": len(frame), "oof_max_abs_difference": np.nan,
                            **score_predictions(frame)})
    return pd.DataFrame(rows + pooled_rows), pd.DataFrame(curves), paired


def _existing_candidates(oof: pd.DataFrame) -> pd.DataFrame:
    h96 = oof.loc[(oof.horizon == 96) & oof.model.str.startswith("lgbm") &
                  ~oof.model.str.contains("quantile")]
    return pd.DataFrame([{"model_id": name, "n": len(part),
                          "overall_mae": float(np.mean(np.abs(part.y - part.pred))),
                          "peak_mae": float(np.mean(np.abs(part.loc[part.y > part.tau, "y"] -
                                                           part.loc[part.y > part.tau, "pred"]))) }
                         for name, part in h96.groupby("model", sort=True)])


def audit_early_stopping(root: Path) -> pd.DataFrame:
    """Read the six saved h96 point models' stop-window metric histories."""
    root = Path(root).resolve()
    rows = []
    for fold in range(3):
        for name in ("lgbm", "lgbm_no_holiday"):
            path = root / f"outputs/models/development_h96_fold{fold}_{name}.joblib"
            bundle = joblib.load(path)
            if bundle["horizon"] != 96 or bundle["fold"] != fold:
                raise AssertionError(f"Saved early-stop model metadata mismatch: {path}")
            model = bundle["model"]
            monitored = model.evals_result_["valid_0"]
            if not {"l1", "l2"}.issubset(monitored):
                raise AssertionError(f"Both l1 and l2 histories are required: {path}")
            l1 = np.asarray(monitored["l1"], dtype=float)
            l2 = np.asarray(monitored["l2"], dtype=float)
            best = int(model.best_iteration_)
            if not len(l1) or len(l1) != len(l2) or best < 1 or best > len(l1) or not (
                    np.isfinite(l1).all() and np.isfinite(l2).all()):
                raise AssertionError(f"Invalid early-stop history: {path}")
            rows.append({"model": name, "fold": fold, "best_iteration": best,
                         "min_l1_iteration": int(np.argmin(l1) + 1),
                         "min_l2_iteration": int(np.argmin(l2) + 1),
                         "monitored_metrics": ",".join(monitored),
                         "l1_at_best": float(l1[best - 1]),
                         "l2_at_best": float(l2[best - 1])})
    return pd.DataFrame(rows)


def _checked_output(root: Path, output: Path | None) -> Path:
    root = Path(root).resolve()
    destination = Path(output).resolve() if output is not None else root / "outputs"
    if not any(destination.is_relative_to(root / name) for name in ("outputs", "_validation")):
        raise ValueError("Diagnostic output must stay under outputs/ or _validation/")
    return destination


def run(root: Path = ROOT, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    output = _checked_output(root, output)
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    boundary = _boundary(cfg)
    # The guarded loader is deliberately the only route to historical labels.
    from src.session_data import load_development_history
    df = load_development_history(root)
    _assert_development_frame(df, boundary)
    oof_path = root / "outputs/predictions/development_oof.csv"
    oof = _read_oof(oof_path, boundary)
    manifest = json.loads((root / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    expected = {(int(row["horizon"]), int(row["fold"])): row for row in manifest["folds"]}
    if len(expected) != len(cfg["horizons"]) * int(cfg["split"]["n_folds"]):
        raise AssertionError("Development manifest fold count is incomplete")
    fold_rows, missing, contexts = [], [], {}
    origins = df.index[df.index < boundary]
    for horizon in cfg["horizons"]:
        for fold_id in range(int(cfg["split"]["n_folds"])):
            context = _reconstruct(df, origins, int(horizon), fold_id, cfg,
                                   expected[(int(horizon), fold_id)])
            fold_rows.append(_fold_summary(context, df))
            missing.extend(_missing_rows(context, int(horizon), fold_id))
            oof_rows = oof.loc[(oof.horizon == horizon) & (oof.fold == fold_id) & (oof.model == "lgbm")]
            if not pd.DatetimeIndex(oof_rows.sort_values("origin").origin).equals(context["score"]):
                raise AssertionError(f"h={horizon} fold={fold_id} OOF score origins differ from the original fold")
            if int(horizon) == 96:
                contexts[fold_id] = context
            print(f"[grid] h={horizon} fold={fold_id} fit={len(context['fit'])} "
                  f"stop={len(context['stop'])} cal={len(context['cal'])} score={len(context['score'])}", flush=True)
    audit = _manual_audit(df, contexts, oof, boundary)
    comparison, learning, paired = _compare_models(contexts, oof, cfg, root)
    early_stopping = audit_early_stopping(root)
    # These later Task-2 metrics are intentionally outside this preregistered Task-1 artifact.
    comparison = comparison.drop(columns=["peak_mae_union", "peak_bias", "overpredict_rate"], errors="ignore")
    existing_candidates = _existing_candidates(oof)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    products = {"fold_grid": output / "tables/h96_diagnostic_fold_grid.csv",
                "feature_missing": output / "tables/h96_diagnostic_feature_missing.csv",
                "audit_20": output / "tables/h96_diagnostic_audit_20.csv",
                "existing_candidates": output / "tables/h96_diagnostic_existing_candidates.csv",
                "model_compare": output / "tables/h96_diagnostic_model_compare.csv",
                "learning_curve": output / "tables/h96_diagnostic_learning_curve.csv",
                "early_stopping_metrics": output / "tables/h96_diagnostic_early_stopping_metrics.csv"}
    pd.DataFrame(fold_rows).to_csv(products["fold_grid"], index=False)
    pd.DataFrame(missing).to_csv(products["feature_missing"], index=False)
    audit.to_csv(products["audit_20"], index=False)
    existing_candidates.to_csv(products["existing_candidates"], index=False)
    comparison.to_csv(products["model_compare"], index=False)
    learning.to_csv(products["learning_curve"], index=False)
    early_stopping.to_csv(products["early_stopping_metrics"], index=False)
    summary = {"scope": "development_oof_only", "test_boundary_exclusive": str(boundary),
               "data_max_timestamp": str(df.index.max()), "original_selection_unchanged": True,
               "fold_grid_exactly_reproduced": True, "manual_audit_rows": len(audit),
               "manual_audit_all_passed": True,
               "key_missing": pd.DataFrame(missing).loc[lambda part: (part.horizon == 96) &
                   part.partition.isin(["raw_train", "raw_validation"]) & part.feature.isin(KEY),
                   ["fold", "partition", "feature", "n", "missing_n", "missing_rate"]].to_dict("records"),
               "h96_model_metrics": comparison.loc[comparison.fold.astype(str) == "pooled",
                   ["model", "n", "mae", "peak_mae", "episode_f1", "false_alarms_positions"]].to_dict("records"),
               "reference_model_mapping": {"original_full": "lgbm (holiday-feature model)",
                                           "original_no_holiday": "lgbm_no_holiday (31.2 overall-MAE comparison model)"},
               "fresh_refit_models": 6,
               "fresh_refit_max_abs_difference": float(comparison.fresh_refit_max_abs_difference.max()),
               "existing_point_candidates": existing_candidates.to_dict("records"),
               "paired_peak_mae_gain_over_original": paired,
               "repaired_masking_checked": True,
               "learning_curve_metric": "LightGBM stop-window validation L1, one value per boosting iteration",
               "early_stopping_first_metric_only": False,
               "learning_curve_scope": "L1 validation curve only; L2 also monitored for early stopping",
               "sources_sha256": {"development_oof": _digest(oof_path),
                                  "development_selection": _digest(root / "outputs/logs/development_selection.json")},
               "products_sha256": {name: _digest(path) for name, path in products.items()},
               "interpretation": "No fold-grid, feature-timestamp, repaired-mask, target-alignment, or saved-model reproduction bug found. "
                                 "The weekly slot and four-feature controls both improve paired h96 peak MAE over the "
                                 "original full LightGBM on the same score rows; this is evidence of a generalization/complexity "
                                 "problem, not proof of its cause. The preregistered original selection is unchanged."}
    log = output / "logs/h96_diagnostic_0924.json"
    log.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Separate diagnostic output directory for a cache-free repeat")
    arguments = parser.parse_args()
    result = run(ROOT, output=arguments.output)
    print(json.dumps({"status": "complete", "manual_audit_rows": result["manual_audit_rows"],
                      "h96_model_metrics": result["h96_model_metrics"]}, ensure_ascii=False, indent=2))
