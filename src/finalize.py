"""Dated, one-time model freeze and holdout evaluation.

All fitting and calibration is completed and hashed before a holdout label is
read. An interrupted reservation stays locked for manual audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluate import evaluate_all
from .features import build_features
from .models.baselines import baseline_predictions
from .models.cbl import cbl_all_predictions
from .models.conformal import apply_conformal, assign_mondrian_bins, fit_conformal, fit_mondrian_edges
from .models.expected_exceedance import fit_exceedance, predict_exceedance
from .models.lgbm_point import fit_point
from .models.lgbm_quantile import fit_quantiles, predict_quantiles
from .models.peak_prob import exceedance_from_quantiles
from .targets import next_day_max_targets, point_targets, training_peak_threshold
from .training import _as_index, _cutoff, _ordered_quantiles, _row, _valid_mask


HOLIDAY_COLUMNS = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprints(cfg, manifest_path):
    base = Path(__file__).resolve().parent
    from .workflow import model_code_digest
    source_path = Path(cfg["source"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Frozen source file is missing: {source_path}")
    processed_path = Path("data/processed/power_15min.parquet")
    processed_hash = _sha(processed_path.read_bytes()) if processed_path.is_file() else None
    return {"code_sha256": model_code_digest(base),
            "config_sha256": _sha(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")),
            "selection_sha256": _sha(manifest_path.read_bytes()),
            "source_sha256": _sha(source_path.read_bytes()),
            "processed_sha256": processed_hash}


def _write_lock(path, record):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _prepare_horizon(df_dev, boundary, horizon, cfg, development):
    delta = pd.Timedelta(minutes=15 * horizon)
    origins = df_dev.index[df_dev.index + delta < boundary]
    if len(origins) < 1000:
        raise ValueError(f"Insufficient development origins for h={horizon}")
    target = point_targets(df_dev, origins, horizon)
    cal_start = int(len(origins) * .80)
    cal_raw, train_raw = origins[cal_start:], origins[:cal_start]
    stop_start = int(len(train_raw) * .80)
    fit_raw, stop_raw = train_raw[:stop_start], train_raw[stop_start:]
    x, latest = build_features(df_dev, origins, horizon, {**cfg, "_tau": None})
    good = origins[_valid_mask(x, latest, origins, target, df_dev)]
    fit, stop, cal = (part.intersection(good) for part in (fit_raw, stop_raw, cal_raw))
    fit = fit[fit + delta < stop.min()]
    stop = stop[stop + delta < cal.min()]
    if min(map(len, (fit, stop, cal))) < 40:
        raise ValueError(f"Insufficient clean fit/stop/calibration rows for h={horizon}")
    tau = training_peak_threshold(df_dev, fit, cfg.get("peak", {}).get("tau_quantile", .95))
    for _ in range(3):
        x, latest = build_features(df_dev, origins, horizon, {**cfg, "_tau": tau})
        good = origins[_valid_mask(x, latest, origins, target, df_dev)]
        fit, stop, cal = (part.intersection(good) for part in (fit, stop, cal))
        if min(map(len, (fit, stop, cal))) < 40:
            raise ValueError("Final fit lost too many rows after peak-history validation")
        exact_tau = training_peak_threshold(df_dev, fit, cfg.get("peak", {}).get("tau_quantile", .95))
        if exact_tau == tau:
            break
        tau = exact_tau
    else:
        raise AssertionError("Final peak threshold did not stabilize")
    y = target.y
    choice = development["selection"]["by_horizon"][str(horizon)]
    bcal = pd.concat([baseline_predictions(df_dev, cal, horizon),
                      cbl_all_predictions(df_dev, cal, horizon)], axis=1)
    bcal["c3_holiday_hybrid"] = bcal["c3_holiday_mid_4_6"].combine_first(bcal["c1_mid_6_10"])
    baseline_cutoffs = {}
    for name in bcal.columns:
        values = bcal[name].to_numpy(dtype=float)
        good_cal = np.isfinite(values)
        if good_cal.sum() >= 30:
            baseline_cutoffs[name] = _cutoff(y.loc[cal].to_numpy()[good_cal],
                                             values[good_cal], tau,
                                             target.loc[cal[good_cal], "target_time"])
    params = next((item["chosen_params"] for item in reversed(development["folds"])
                   if item["horizon"] == horizon), {})
    point = None
    selected = choice["point_model"]
    if selected.startswith("lgbm"):
        cols = list(x.columns)
        if "no_holiday" in selected:
            cols = [col for col in cols if col not in HOLIDAY_COLUMNS]
        weight = float(selected.rsplit("_weight_", 1)[1]) if "_weight_" in selected else 1.0
        model = fit_point(x.loc[fit, cols], y.loc[fit], x.loc[stop, cols], y.loc[stop], cfg,
                          peak_threshold=tau, peak_weight=weight, params=params)
        cutoff = _cutoff(y.loc[cal].to_numpy(), model.predict(x.loc[cal, cols]), tau,
                         target.loc[cal, "target_time"])
        point = {"model": model, "features": cols, "cutoff": cutoff, "name": selected}
    qcols = [col for col in x.columns if col not in HOLIDAY_COLUMNS]
    qmodels = fit_quantiles(x.loc[fit, qcols], y.loc[fit], x.loc[stop, qcols], y.loc[stop], cfg,
                            params=params)
    qc = predict_quantiles(qmodels, x.loc[cal, qcols])
    method = choice.get("conformal", "a")
    edges = fit_mondrian_edges(qc[.5], cfg.get("conformal", {}).get("mondrian_bins", [.5, .9]))
    bc = assign_mondrian_bins(qc[.5], edges) if method == "b" else None
    midpoint = len(cal) // 2
    cutoff_start = midpoint + horizon + 1
    if cutoff_start >= len(cal):
        raise ValueError("Independent alarm-cutoff calibration block is empty")
    qct = dict(qc); corrections = {}
    for alpha in (.9, .95, .975):
        correction = fit_conformal(y.loc[cal].to_numpy()[:midpoint], qc[alpha][:midpoint],
                                   alpha, None if bc is None else bc[:midpoint])
        corrections[str(alpha)] = correction
        qct[alpha] = apply_conformal(qc[alpha], correction, bc)
    qct = _ordered_quantiles(qct)
    pcal = exceedance_from_quantiles(qct, tau)
    probability_cutoff = _cutoff(y.loc[cal].to_numpy()[cutoff_start:], pcal[cutoff_start:],
                                  tau, target.loc[cal[cutoff_start:], "target_time"])
    expected = None
    if development["selection"]["adoption"].get(f"h{horizon}_expected_exceedance", {}).get("adopted", False):
        expected = fit_exceedance(x.loc[fit, qcols], y.loc[fit], tau, cfg)
    return {"horizon": horizon, "tau": tau, "baseline_cutoffs": baseline_cutoffs,
            "point": point, "quantile_models": qmodels, "quantile_features": qcols,
            "conformal": method, "mondrian_edges": edges, "corrections": corrections,
            "probability_cutoff": probability_cutoff, "expected_model": expected,
            "fit_end": str(fit.max()), "cal_end": str(cal.max())}


def _prepare_t2(df_dev, boundary, cfg):
    origins = df_dev.index[(df_dev.index.hour == 23) & (df_dev.index.minute == 45)]
    target = next_day_max_targets(df_dev, origins)
    origins = origins[(target.valid_target & (target.target_time < boundary)).to_numpy()]
    if len(origins) < 100:
        raise ValueError("Insufficient T2 development days")
    x, _ = build_features(df_dev, origins, 96, {**cfg, "_include_cbl": False})
    good = np.asarray(x.attrs["valid_mask"], dtype=bool) & target.loc[origins, "valid_target"].to_numpy()
    origins = origins[good]
    target = target.loc[origins]
    train, stop = origins[:int(len(origins) * .75)], origins[int(len(origins) * .75):]
    train = train[target.loc[train, "target_time"].to_numpy() < stop.min()]
    model = fit_point(x.loc[train], target.loc[train, "y"], x.loc[stop], target.loc[stop, "y"], cfg)
    return {"model": model, "features": list(x.columns), "fit_end": str(train.max())}


def _known_day_max(df, origins):
    result = []
    for origin in origins:
        window = df.loc[(df.index > origin - pd.Timedelta(days=1)) & (df.index <= origin)]
        repaired = window.get("time_repaired", pd.Series(False, index=window.index)).fillna(True).astype(bool)
        result.append(float(window.power.max()) if len(window) == 96 and window.power.notna().all()
                      and not repaired.any() else np.nan)
    return np.asarray(result)


def freeze_and_evaluate(df, cfg, output_dir="outputs"):
    deadline = pd.Timestamp(cfg["freeze"]["not_before"])
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if now < deadline:
        raise RuntimeError(f"Holdout is sealed until {deadline}; current time is {now}")
    out = Path(output_dir)
    from .workflow import require_freeze_approval
    # The old YAML switch is historical configuration, never an approval.
    # This check precedes all holdout artifact reads and model fitting.
    require_freeze_approval(Path.cwd(), cfg, out)
    for folder in ("logs", "tables", "predictions", "models"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest_path = out / "logs" / "development_selection.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Development selection manifest is required before freeze")
    signatures = _fingerprints(cfg, manifest_path)
    lock = out / "logs" / "freeze_record.json"
    final_path = out / "tables" / "final_test.csv"
    pred_path = out / "predictions" / "final_test_predictions.csv"
    if lock.exists():
        record = json.loads(lock.read_text(encoding="utf-8"))
        model_file = out / "models" / "frozen_final.joblib"
        t2_metrics_path = out / "tables" / "t2_final_test.csv"
        t2_predictions_path = out / "predictions" / "t2_final_test_predictions.csv"
        complete_and_matching = (record.get("status") == "completed"
                                 and record.get("fingerprints") == signatures
                                 and final_path.exists() and pred_path.exists() and model_file.exists()
                                 and t2_metrics_path.exists() and t2_predictions_path.exists()
                                 and record.get("model_sha256") == _sha(model_file.read_bytes())
                                 and record.get("final_metrics_sha256") == _sha(final_path.read_bytes())
                                 and record.get("holdout_prediction_sha256") == _sha(pred_path.read_bytes())
                                 and record.get("t2_metrics_sha256") == _sha(t2_metrics_path.read_bytes())
                                 and record.get("t2_predictions_sha256") == _sha(t2_predictions_path.read_bytes()))
        if complete_and_matching:
            return {"predictions": pd.read_csv(pred_path, parse_dates=["origin", "target_time"]),
                    "metrics": pd.read_csv(final_path),
                    "paths": {"metrics": str(final_path), "predictions": str(pred_path)},
                    "cached": True}
        raise RuntimeError("Freeze reservation already exists; inspect it before any retry")
    if final_path.exists() or pred_path.exists():
        raise RuntimeError("Holdout artifacts exist without a matching freeze record")
    record = {"status": "reserved", "reserved_at_kst": str(now), "fingerprints": signatures,
              "test_start_origin": cfg["split"]["test_start_origin"]}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)

    # No holdout labels or holdout feature values are read in this phase.
    df = _as_index(df)
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    df_dev = df.loc[df.index < boundary].copy()
    development = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = {int(h): _prepare_horizon(df_dev, boundary, int(h), cfg, development)
                for h in cfg.get("horizons", [1, 4, 16, 96])}
    prepared["T2"] = _prepare_t2(df_dev, boundary, cfg)
    frozen_path = out / "models" / "frozen_final.joblib"
    joblib.dump(prepared, frozen_path)
    record.update(status="models_frozen", model_sha256=_sha(frozen_path.read_bytes()),
                  models_frozen_at_kst=str(pd.Timestamp.now(tz="Asia/Seoul")))
    _write_lock(lock, record)

    # The single holdout read begins only after all selected models are frozen.
    all_test = df.index[df.index >= boundary]
    predictions = []
    for horizon, bundle in prepared.items():
        if horizon == "T2":
            continue
        delta = pd.Timedelta(minutes=15 * horizon)
        origins = all_test[all_test + delta <= df.index.max()]
        target = point_targets(df, origins, horizon)
        x, latest = build_features(df, origins, horizon, {**cfg, "_tau": bundle["tau"]})
        good = origins[_valid_mask(x, latest, origins, target, df)]
        x = x.loc[good]
        tau = bundle["tau"]
        btest = pd.concat([baseline_predictions(df, good, horizon),
                           cbl_all_predictions(df, good, horizon)], axis=1)
        btest["c3_holiday_hybrid"] = btest["c3_holiday_mid_4_6"].combine_first(btest["c1_mid_6_10"])
        for name, cutoff in bundle["baseline_cutoffs"].items():
            predictions.append(_row(good, horizon, -1, name, target, tau,
                                    btest[name].to_numpy(dtype=float), cutoff))
        point = bundle["point"]
        if point is not None:
            predictions.append(_row(good, horizon, -1, point["name"], target, tau,
                                    point["model"].predict(x[point["features"]]), point["cutoff"]))
        qt = predict_quantiles(bundle["quantile_models"], x[bundle["quantile_features"]])
        bt = (assign_mondrian_bins(qt[.5], bundle["mondrian_edges"])
              if bundle["conformal"] == "b" else None)
        corrected = dict(qt)
        for alpha in (.9, .95, .975):
            corrected[alpha] = apply_conformal(qt[alpha], bundle["corrections"][str(alpha)], bt)
        corrected = _ordered_quantiles(corrected)
        p = exceedance_from_quantiles(corrected, tau)
        extras = {f"q{int(alpha*1000) if alpha == .975 else int(alpha*100)}": qt[alpha] for alpha in sorted(qt)}
        extras.update(q90_cal=corrected[.9], q95_cal=corrected[.95], q975_cal=corrected[.975],
                      p_exceed=p, q50_top_edge=float(bundle["mondrian_edges"][-1]),
                      conformal_method=bundle["conformal"])
        if bundle["expected_model"] is not None:
            extras["exp_exceed"] = predict_exceedance(bundle["expected_model"],
                                                       x[bundle["quantile_features"]], p)
        row = _row(good, horizon, -1, f"lgbm_quantile_{bundle['conformal']}", target, tau,
                   qt[.5], float("inf"), **extras)
        row["alert"] = p > bundle["probability_cutoff"]
        row["alert_cutoff"] = bundle["probability_cutoff"]
        predictions.append(row)
    final_pred = pd.concat(predictions, ignore_index=True)
    final_metrics = evaluate_all(final_pred, cfg=cfg)

    t2_origins = all_test[(all_test.hour == 23) & (all_test.minute == 45)]
    t2_target = next_day_max_targets(df, t2_origins)
    t2_origins = t2_origins[t2_target.valid_target.to_numpy()]
    t2_x, _ = build_features(df, t2_origins, 96, {**cfg, "_include_cbl": False})
    t2_good = t2_origins[np.asarray(t2_x.attrs["valid_mask"], dtype=bool)]
    t2_actual = t2_target.loc[t2_good, "y"].to_numpy(dtype=float)
    t2_models = {"t2_recent_24h_max": _known_day_max(df, t2_good),
                 "t2_last_week_max": _known_day_max(df, t2_good - pd.Timedelta(days=7)),
                 "t2_lgbm": prepared["T2"]["model"].predict(t2_x.loc[t2_good, prepared["T2"]["features"]])}
    t2_rows = []
    for name, values in t2_models.items():
        for origin, target_time, actual, pred in zip(t2_good, t2_target.loc[t2_good, "target_time"], t2_actual, values):
            t2_rows.append({"origin": origin, "target_time": target_time, "model": name,
                            "y": actual, "pred": pred})
    t2_pred = pd.DataFrame(t2_rows)
    t2_metrics = pd.DataFrame([{"model": name, "n": len(part),
                                "mae": float(np.nanmean(abs(part.y - part.pred)))}
                               for name, part in t2_pred.groupby("model")])
    t2_pred.to_csv(out / "predictions" / "t2_final_test_predictions.csv", index=False)
    t2_metrics.to_csv(out / "tables" / "t2_final_test.csv", index=False)
    final_pred.to_csv(pred_path, index=False)
    temporary = final_path.with_suffix(".csv.tmp")
    final_metrics.to_csv(temporary, index=False)
    temporary.replace(final_path)
    record.update(status="completed", evaluated_at_kst=str(pd.Timestamp.now(tz="Asia/Seoul")),
                  holdout_prediction_sha256=_sha(pred_path.read_bytes()),
                  final_metrics_sha256=_sha(final_path.read_bytes()),
                  t2_metrics_sha256=_sha((out / "tables" / "t2_final_test.csv").read_bytes()),
                  t2_predictions_sha256=_sha((out / "predictions" / "t2_final_test_predictions.csv").read_bytes()))
    _write_lock(lock, record)
    return {"predictions": final_pred, "metrics": final_metrics,
            "paths": {"metrics": str(final_path), "predictions": str(pred_path),
                      "t2_metrics": str(out / "tables" / "t2_final_test.csv")}, "cached": False}
