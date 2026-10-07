"""Fit-only Phase C model preparation, stop-only tuning, and sealed scoring.

The caller supplies development contexts and controls the config/input/code
hashes protecting local resume files. This module never opens a raw source or
final holdout. ``prepare_baselines`` and ``tune_family`` do not inspect score
targets or predictions. ``score_main`` requires the saved full configuration
lock before consulting score rows.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from phase_c.statistical import fit_kalman, fit_mstl, predict_kalman, predict_mstl
from src.models.cbl import cbl_all_predictions
from src.models.lgbm_point import fit_point


HORIZONS = tuple(range(4, 17))
FOLDS = (0, 1, 2)
ANCHORS = (4, 8, 12, 16)
MAIN = ("B0", "B1", "B2", "B3", "B4", "B5", "M1", "M1-W", "M2", "C1")
BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")


def _paths(out: str | Path) -> tuple[Path, Path, Path]:
    root = Path(out)
    models, tables, logs = (root / folder for folder in ("models", "tables", "logs"))
    for folder in (models, tables, logs):
        folder.mkdir(parents=True, exist_ok=True)
    return models, tables, logs


def _atomic_joblib(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _key(horizon: int, fold: int) -> str:
    return f"h{horizon:02d}_f{fold}"


def _context(contexts: Mapping, horizon: int, fold: int) -> Mapping:
    return contexts[(horizon, fold)]


def _validate_setup(history: pd.DataFrame, contexts: Mapping, cfg: Mapping) -> None:
    if cfg.get("development_only") is not True:
        raise ValueError("Phase C requires development_only=True")
    if not isinstance(history.index, pd.DatetimeIndex) or history.empty or history.index.max() >= BOUNDARY:
        raise ValueError("History must be a sealed development DatetimeIndex")
    if "power" not in history or not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("History must have chronological clean power")
    expected = {(h, f) for h in HORIZONS for f in FOLDS}
    if set(contexts) != expected:
        raise ValueError("Exactly 13 horizons by three development folds are required")


def _frame(c: Mapping, role: str) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    index = pd.DatetimeIndex(c[role])
    x = c["x"].loc[index]
    y = c["y"].loc[index]
    if x.empty or not np.isfinite(x.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(f"{role} must contain finite model rows")
    return x, y, index


def _ridge_fit(x: pd.DataFrame, y: pd.Series, alpha: float) -> dict:
    if alpha != 100.0:
        raise ValueError("B4 Ridge alpha is fixed at 100")
    matrix = x.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    centered = (matrix - mean) / scale
    target_mean = float(target.mean())
    coef = np.linalg.solve(centered.T @ centered + alpha * np.eye(matrix.shape[1]),
                           centered.T @ (target - target_mean))
    return {"columns": list(x.columns), "mean": mean, "scale": scale,
            "target_mean": target_mean, "coef": coef, "alpha": alpha,
            "development_only": True, "fit_parameters_only": True}


def _ridge_predict(bundle: Mapping, x: pd.DataFrame) -> np.ndarray:
    matrix = x.loc[:, bundle["columns"]].to_numpy(dtype=float)
    return bundle["target_mean"] + ((matrix - bundle["mean"]) / bundle["scale"]) @ bundle["coef"]


def _save_baseline(models: Path, model: str, h: int, f: int, bundle: Any, seconds: float) -> None:
    _atomic_joblib(models / f"{model}_{_key(h, f)}.joblib",
                   {"bundle": bundle, "train_seconds": float(seconds),
                    "development_only": True, "fit_parameters_only": True})


def _load_baseline(models: Path, model: str, h: int, f: int) -> dict:
    artifact = joblib.load(models / f"{model}_{_key(h, f)}.joblib")
    if artifact.get("development_only") is not True or artifact.get("fit_parameters_only") is not True:
        raise ValueError(f"Invalid cached {model} fit artifact")
    return artifact


def prepare_baselines(history: pd.DataFrame, contexts: Mapping, cfg: Mapping, out: str | Path) -> dict:
    """Prepare B0..B5 fit artifacts and choose B2 on stop rows only."""
    _validate_setup(history, contexts, cfg)
    models, _, _ = _paths(out)
    candidates = list(cfg["cbl_candidates"])
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("B2 requires five ordered CBL candidates")
    cbl_by_horizon: dict[str, str] = {}
    for h in HORIZONS:
        stop_mae: dict[str, list[float]] = {name: [] for name in candidates}
        for f in FOLDS:
            c = _context(contexts, h, f)
            key = _key(h, f)
            x_fit, y_fit, fit = _frame(c, "fit")
            _, y_stop, stop = _frame(c, "stop")
            cbl_path = models / f"CBL_stop_{key}.parquet"
            if cbl_path.exists():
                cbl = pd.read_parquet(cbl_path).set_index("origin")
                if not cbl.index.equals(stop) or not set(candidates) <= set(cbl.columns):
                    raise ValueError(f"Stale CBL stop cache: {cbl_path}")
                cbl = cbl.loc[:, candidates]
            else:
                started = perf_counter()
                cbl = cbl_all_predictions(history, stop, h).loc[:, candidates]
                _atomic_parquet(cbl_path, cbl.rename_axis("origin").reset_index())
                print(f"CBL stop {key} seconds={perf_counter() - started:.1f}", flush=True)
            valid = np.isfinite(cbl.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y_stop.to_numpy(dtype=float))
            if not valid.any():
                raise ValueError(f"No common finite five-CBL stop rows for {key}")
            errors = np.abs(cbl.to_numpy(dtype=float)[valid] - y_stop.to_numpy(dtype=float)[valid, None])
            for column, name in enumerate(candidates):
                stop_mae[name].append(float(errors[:, column].mean()))

            for model in ("B0", "B1", "B3", "B4", "B5"):
                path = models / f"{model}_{key}.joblib"
                if path.exists():
                    continue
                started = perf_counter()
                if model == "B0":
                    bundle = {"feature": "current"}
                elif model == "B1":
                    bundle = {"feature": "slot7d"}
                elif model == "B3":
                    # Calendar metadata is safe here; score values are untouched.
                    end = pd.Timestamp(c["target_time"].loc[c["score"]].max())
                    bundle = fit_mstl(history["power"], fit, end)
                elif model == "B4":
                    bundle = _ridge_fit(x_fit, y_fit, float(cfg["ridge"]["alpha"]))
                else:
                    bundle = fit_kalman(history["power"], fit)
                _save_baseline(models, model, h, f, bundle, perf_counter() - started)
                print(f"Fit {model} {key} seconds={perf_counter() - started:.1f}", flush=True)
        # An equal mean of three stop MAEs, each on its common five-CBL cohort.
        selected = min(candidates, key=lambda name: (float(np.mean(stop_mae[name])), candidates.index(name)))
        cbl_by_horizon[str(h)] = selected
        for f in FOLDS:
            path = models / f"B2_{_key(h, f)}.joblib"
            if not path.exists():
                _save_baseline(models, "B2", h, f,
                               {"selected_cbl": selected, "stop_mae_by_candidate":
                                {name: stop_mae[name][f] for name in candidates}}, 0.0)
            elif _load_baseline(models, "B2", h, f)["bundle"]["selected_cbl"] != selected:
                raise ValueError(f"Cached B2 selection conflicts at horizon {h}")
        print(f"B2 selected h{h:02d}: {selected}", flush=True)
    return {"cbl_by_horizon": cbl_by_horizon, "baseline_status": "prepared",
            "development_only": True}


def _candidate_list(cfg: Mapping, family: str) -> list[dict]:
    if family not in ("lgbm", "tcn"):
        raise ValueError("family must be lgbm or tcn")
    candidates = list(cfg[family]["candidates"])
    expected = 3 if family == "lgbm" else 4
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"{family} requires {expected} distinct ordered candidates")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("Candidate IDs must be safe filename components")
    return candidates


def _tcn_inputs(c: Mapping, role: str, cfg: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = pd.DatetimeIndex(c[role])
    seq = c["seq"].loc[index].to_numpy(dtype=np.float32)
    context = c["x"].loc[index, cfg["tcn"]["context_columns"]].to_numpy(dtype=np.float32)
    target = c["y"].loc[index].to_numpy(dtype=np.float32)
    return seq, context, target


def _tcn_config(candidate: Mapping, cfg: Mapping) -> dict:
    return {"channels": int(candidate["channels"]), "dropout": float(candidate["dropout"]),
            "max_epochs": int(cfg["tcn"]["max_epochs"]), "patience": int(cfg["tcn"]["patience"])}


def _fit_family(family: str, c: Mapping, cfg: Mapping, candidate: Mapping,
                *, peak_weight: float = 1.0, residual: bool = False) -> tuple[Any, float, dict]:
    if residual and family != "lgbm":
        raise ValueError("Weekly residual C1 must use LightGBM")
    started = perf_counter()
    if family == "lgbm":
        x_fit, y_fit, _ = _frame(c, "fit")
        x_stop, y_stop, _ = _frame(c, "stop")
        if residual:
            y_fit = y_fit - x_fit["slot7d"]
            y_stop = y_stop - x_stop["slot7d"]
        params = {name: value for name, value in candidate.items() if name != "id"}
        model = fit_point(x_fit, y_fit, x_stop, y_stop, cfg,
                          peak_threshold=float(c["tau"]) if peak_weight != 1 else None,
                          peak_weight=peak_weight, params=params)
        effective = {"best_iteration": int(getattr(model, "best_iteration_", 0) or 0)}
    else:
        from phase_c.tcn import fit_tcn

        fit_seq, fit_context, fit_y = _tcn_inputs(c, "fit", cfg)
        stop_seq, stop_context, stop_y = _tcn_inputs(c, "stop", cfg)
        model = fit_tcn(fit_seq, fit_context, fit_y, stop_seq, stop_context, stop_y,
                        _tcn_config(candidate, cfg))
        effective = {"best_epoch": int(model["metadata"]["best_epoch"]),
                     "epochs_ran": int(model["metadata"]["epochs_ran"])}
    return model, perf_counter() - started, effective


def _predict_family(family: str, model: Any, c: Mapping, role: str, cfg: Mapping,
                    *, residual: bool = False) -> np.ndarray:
    if residual and family != "lgbm":
        raise ValueError("Weekly residual C1 must use LightGBM")
    if family == "lgbm":
        prediction = np.asarray(model.predict(c["x"].loc[c[role]]), dtype=float)
    else:
        from phase_c.tcn import predict_tcn

        seq, context, _ = _tcn_inputs(c, role, cfg)
        prediction = predict_tcn(model, seq, context)
    if residual:
        prediction = prediction + c["x"].loc[c[role], "slot7d"].to_numpy(dtype=float)
    return prediction


def _family_path(models: Path, family: str, candidate_id: str, h: int, f: int,
                 *, tuning: bool) -> Path:
    prefix = "tune" if tuning else "score"
    suffix = ".pt" if family == "tcn" else ".joblib"
    return models / f"{prefix}_{family}_{candidate_id}_{_key(h, f)}{suffix}"


def _save_family(path: Path, family: str, model: Any, seconds: float, effective: Mapping) -> None:
    artifact = {"model": model, "train_seconds": float(seconds),
                "effective": dict(effective), "development_only": True,
                "fit_parameters_only": True}
    temporary = path.with_name(path.name + ".tmp")
    if family == "tcn":
        import torch
        torch.save(artifact, temporary)
        temporary.replace(path)
    else:
        _atomic_joblib(path, artifact)


def _load_family(path: Path, family: str) -> dict:
    if family == "tcn":
        import torch
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    else:
        artifact = joblib.load(path)
    if artifact.get("development_only") is not True or artifact.get("fit_parameters_only") is not True:
        raise ValueError(f"Invalid cached fitted family artifact: {path}")
    return artifact


def _anchor_bootstrap(cell_data: list[dict], candidate_ids: list[str], draws_n: int,
                      seed: int) -> tuple[dict[str, float], dict[str, dict], dict[str, dict]]:
    """One target-day draw is shared by all candidates, horizons and folds."""
    dates = pd.Index(sorted({day for cell in cell_data for day in cell["days"]}))
    n_days, n_cells, n_candidates = len(dates), len(cell_data), len(candidate_ids)
    error_sum = np.zeros((n_days, n_cells, n_candidates), dtype=float)
    counts = np.zeros((n_days, n_cells, n_candidates), dtype=float)
    cell_means: dict[str, dict] = {name: {} for name in candidate_ids}
    for cell_index, cell in enumerate(cell_data):
        day_index = dates.get_indexer(cell["days"])
        errors = np.abs(cell["predictions"] - cell["y"][:, None])
        for candidate_index, name in enumerate(candidate_ids):
            np.add.at(error_sum[:, cell_index, candidate_index], day_index, errors[:, candidate_index])
            np.add.at(counts[:, cell_index, candidate_index], day_index, 1)
            cell_means[name][cell["key"]] = float(errors[:, candidate_index].mean())
    means = {name: float(np.mean(list(cell_means[name].values()))) for name in candidate_ids}
    draws = np.random.default_rng(seed).multinomial(n_days, np.full(n_days, 1 / n_days), size=draws_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        sampled = np.einsum("bd,dcj->bcj", draws, error_sum) / np.einsum("bd,dcj->bcj", draws, counts)
    sampled_equal_mean = np.mean(sampled, axis=1)
    primary = min(candidate_ids, key=lambda name: (means[name], candidate_ids.index(name)))
    primary_index = candidate_ids.index(primary)
    intervals: dict[str, dict] = {}
    for index, name in enumerate(candidate_ids):
        differences = sampled_equal_mean[:, index] - sampled_equal_mean[:, primary_index]
        if np.isfinite(differences).all():
            low, high = np.percentile(differences, [2.5, 97.5])
            intervals[name] = {"low": float(low), "high": float(high), "status": "available"}
        else:
            intervals[name] = {"low": None, "high": None, "status": "unavailable_zero_cell_draw"}
    return means, cell_means, intervals


def tune_family(family: str, history: pd.DataFrame, contexts: Mapping, cfg: Mapping,
                out: str | Path) -> dict:
    """Select M1 or M2 settings from fixed anchor stop cells, never score."""
    _validate_setup(history, contexts, cfg)
    candidates = _candidate_list(cfg, family)
    ids = [candidate["id"] for candidate in candidates]
    models, tables, logs = _paths(out)
    cell_data: list[dict] = []
    table_rows: list[dict] = []
    for h in ANCHORS:
        for f in FOLDS:
            c = _context(contexts, h, f)
            _, stop_y, stop = _frame(c, "stop")
            predictions = []
            for candidate in candidates:
                name = candidate["id"]
                model_path = _family_path(models, family, name, h, f, tuning=True)
                stop_path = models / f"tune_{family}_{name}_{_key(h, f)}_stop.parquet"
                new_fit = not model_path.exists()
                if not new_fit:
                    artifact = _load_family(model_path, family)
                else:
                    model, seconds, effective = _fit_family(family, c, cfg, candidate)
                    _save_family(model_path, family, model, seconds, effective)
                    artifact = _load_family(model_path, family)
                    print(f"Tune {family} {name} {_key(h, f)} seconds={seconds:.1f} effective={effective}", flush=True)
                if stop_path.exists() and not new_fit:
                    frame = pd.read_parquet(stop_path)
                    if (not pd.DatetimeIndex(frame["origin"]).equals(stop)
                            or not np.array_equal(frame["y"].to_numpy(dtype=float), stop_y.to_numpy(dtype=float))
                            or not pd.DatetimeIndex(frame["target_time"]).equals(
                                pd.DatetimeIndex(c["target_time"].loc[stop]))):
                        raise ValueError(f"Stale stop cache: {stop_path}")
                    forecast = frame["pred"].to_numpy(dtype=float)
                else:
                    forecast = _predict_family(family, artifact["model"], c, "stop", cfg)
                    if len(forecast) != len(stop) or not np.isfinite(forecast).all():
                        raise RuntimeError(f"Nonfinite {family} stop predictions at {_key(h, f)}")
                    _atomic_parquet(stop_path, pd.DataFrame({"origin": stop,
                                                            "target_time": c["target_time"].loc[stop].to_numpy(),
                                                            "y": stop_y.to_numpy(dtype=float),
                                                            "pred": forecast}))
                if len(forecast) != len(stop) or not np.isfinite(forecast).all():
                    raise ValueError(f"Invalid cached stop predictions: {stop_path}")
                predictions.append(forecast)
                table_rows.append({"family": family, "candidate_id": name, "horizon": h, "fold": f,
                                   "stop_mae": float(np.mean(np.abs(forecast - stop_y.to_numpy(dtype=float)))),
                                   "stop_rows": len(stop), "development_only": True})
            days = pd.DatetimeIndex(c["target_time"].loc[stop]).normalize()
            cell_data.append({"key": _key(h, f), "days": days, "y": stop_y.to_numpy(dtype=float),
                              "predictions": np.column_stack(predictions)})
    draws_n = int(cfg["bootstrap_n"])
    if draws_n != 1000:
        raise ValueError("Anchor selection requires 1000 frozen bootstrap draws")
    means, cell_means, intervals = _anchor_bootstrap(cell_data, ids, draws_n, int(cfg["seed"]))
    primary = min(ids, key=lambda name: (means[name], ids.index(name)))
    tied = [name for name in ids if name == primary or
            (intervals[name]["status"] == "available" and intervals[name]["low"] <= 0 <= intervals[name]["high"])]
    selected = tied[0]  # cfg candidate order is the frozen simplicity order.
    chosen_config = next({key: value for key, value in row.items() if key != "id"}
                         for row in candidates if row["id"] == selected)
    detail = {"selected_id": selected, "selected_config": chosen_config,
              "primary_min_mean_id": primary, "tieids": tied,
              "allcandidatecellmeans": cell_means, "equal_mean_12_cells": means,
              "ci_vs_primary": intervals, "source": "stop_only", "development_only": True}
    _atomic_parquet(models / f"{family}_anchor_stop_predictions.parquet",
                    pd.concat([pd.read_parquet(models / f"tune_{family}_{name}_{_key(h, f)}_stop.parquet")
                               .assign(candidate_id=name, horizon=h, fold=f)
                               for h in ANCHORS for f in FOLDS for name in ids], ignore_index=True))
    table_path = tables / f"{family}_anchor_tuning.csv"
    temporary_table = table_path.with_name(table_path.name + ".tmp")
    pd.DataFrame(table_rows).assign(equal_mean_12_cells=lambda frame: frame.candidate_id.map(means)).to_csv(
        temporary_table, index=False)
    temporary_table.replace(table_path)
    _atomic_json(logs / f"{family}_config_lock.json", detail)
    print(f"Locked {family}: selected={selected} primary={primary} ties={tied}", flush=True)
    return {family: detail}


def _full_lock(lock: Mapping, cfg: Mapping, logs: Path) -> dict:
    path = logs / "model_config_lock.json"
    if not path.is_file():
        raise FileNotFoundError("score_main requires a saved logs/model_config_lock.json")
    saved = json.loads(path.read_text(encoding="utf-8"))

    def fields(value: Mapping) -> tuple[Mapping, Mapping, Mapping, str]:
        baseline = value.get("baseline", value)
        return baseline["cbl_by_horizon"], value["lgbm"], value["tcn"], baseline["baseline_status"]

    try:
        supplied_fields = fields(lock)
        saved_fields = fields(saved)
    except (KeyError, TypeError) as exc:
        raise ValueError("Full lock lacks baseline, lgbm or tcn selection") from exc
    if json.dumps(supplied_fields, sort_keys=True) != json.dumps(saved_fields, sort_keys=True):
        raise ValueError("Passed model lock differs from saved model_config_lock.json")
    cbl_by_horizon, lgbm, tcn, status = supplied_fields
    if status != "prepared" or set(cbl_by_horizon) != {str(h) for h in HORIZONS}:
        raise ValueError("Incomplete baseline selection in full lock")
    if any(cbl_by_horizon[str(h)] not in cfg["cbl_candidates"] for h in HORIZONS):
        raise ValueError("Locked CBL candidate is not in the fixed config")
    for family, detail in (("lgbm", lgbm), ("tcn", tcn)):
        candidates = _candidate_list(cfg, family)
        matched = [row for row in candidates if row["id"] == detail.get("selected_id")]
        if detail.get("source") != "stop_only" or len(matched) != 1:
            raise ValueError(f"{family} lock is not a valid stop-only selection")
        expected = {key: value for key, value in matched[0].items() if key != "id"}
        if detail.get("selected_config") != expected:
            raise ValueError(f"{family} selected settings differ from config")
    return {"cbl_by_horizon": dict(cbl_by_horizon), "lgbm": dict(lgbm), "tcn": dict(tcn)}


def _score_family(models: Path, family: str, model_name: str, h: int, f: int,
                  c: Mapping, cfg: Mapping, candidate: Mapping,
                  *, peak_weight: float = 1.0, residual: bool = False) -> tuple[np.ndarray, dict]:
    selected_id = candidate["id"]
    tuned = model_name in ("M1", "M2") and h in ANCHORS
    if tuned:
        path = _family_path(models, family, selected_id, h, f, tuning=True)
        if not path.exists():
            raise FileNotFoundError(f"Missing locked anchor fit: {path}")
    else:
        path = _family_path(models, family, model_name, h, f, tuning=False)
        if not path.exists():
            model, seconds, effective = _fit_family(family, c, cfg, candidate,
                                                     peak_weight=peak_weight, residual=residual)
            _save_family(path, family, model, seconds, effective)
            print(f"Fit {model_name} {_key(h, f)} seconds={seconds:.1f} effective={effective}", flush=True)
    artifact = _load_family(path, family)
    started = perf_counter()
    prediction = _predict_family(family, artifact["model"], c, "score", cfg, residual=residual)
    inference = perf_counter() - started
    return prediction, {"train_seconds": float(artifact["train_seconds"]),
                        "inference_seconds": inference, "effective": artifact["effective"],
                        "fit_artifact": str(path)}


def _score_baseline(models: Path, model: str, h: int, f: int, c: Mapping,
                    history: pd.DataFrame, chosen_cbl: str) -> tuple[np.ndarray, dict]:
    artifact = _load_baseline(models, model, h, f)
    bundle = artifact["bundle"]
    score = pd.DatetimeIndex(c["score"])
    started = perf_counter()
    if model in ("B0", "B1"):
        prediction = c["x"].loc[score, bundle["feature"]].to_numpy(dtype=float)
    elif model == "B2":
        if bundle["selected_cbl"] != chosen_cbl:
            raise ValueError(f"B2 selected method mismatch at {_key(h, f)}")
        prediction = cbl_all_predictions(history, score, h)[chosen_cbl].to_numpy(dtype=float)
    elif model == "B3":
        prediction = predict_mstl(bundle, history["power"], score, h)
    elif model == "B4":
        prediction = _ridge_predict(bundle, c["x"].loc[score])
    elif model == "B5":
        prediction = predict_kalman(bundle, history["power"], score, h)
    else:
        raise ValueError(f"Unknown baseline: {model}")
    return np.asarray(prediction, dtype=float), {
        "train_seconds": float(artifact["train_seconds"]),
        "inference_seconds": perf_counter() - started,
        "effective": {}, "fit_artifact": str(models / f"{model}_{_key(h, f)}.joblib")}


def _config_for_model(model: str, h: int, lock: Mapping, cfg: Mapping) -> str:
    if model in ("B0", "B1"):
        value = {"feature": "current" if model == "B0" else "slot7d"}
    elif model == "B2":
        value = {"selected_cbl": lock["cbl_by_horizon"][str(h)]}
    elif model == "B3":
        value = dict(cfg["mstl"])
    elif model == "B4":
        value = dict(cfg["ridge"])
    elif model == "B5":
        value = dict(cfg["kalman"])
    elif model in ("M1", "M1-W", "C1"):
        value = {"selected_id": lock["lgbm"]["selected_id"],
                 "model_params": lock["lgbm"]["selected_config"],
                 "peak_weight": float(cfg["peak_weight"]) if model == "M1-W" else 1.0,
                 "target_transform": "weekly_residual" if model == "C1" else "direct"}
    else:
        value = {"selected_id": lock["tcn"]["selected_id"],
                 "model_params": lock["tcn"]["selected_config"], "peak_weight": 1.0,
                 "target_transform": "direct"}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_chunk(frame: pd.DataFrame, c: Mapping, h: int, f: int) -> None:
    score = pd.DatetimeIndex(c["score"])
    if len(frame) != len(score) * len(MAIN) or set(frame["model"]) != set(MAIN):
        raise ValueError(f"Incomplete cached score chunk {_key(h, f)}")
    if not frame.development_only.eq(True).all():
        raise ValueError("Cached score chunk is not development-only")
    for model in MAIN:
        rows = frame.loc[frame.model.eq(model)]
        if not pd.DatetimeIndex(rows.origin).equals(score):
            raise ValueError(f"Cached {model} score origins differ at {_key(h, f)}")
    if not frame.target_time.lt(BOUNDARY).all():
        raise ValueError("Cached score chunk crosses sealed boundary")


def score_main(history: pd.DataFrame, contexts: Mapping, cfg: Mapping,
               lock: Mapping, out: str | Path) -> pd.DataFrame:
    """Score MAIN10 after all hyperparameters are locked from stop only.

    Each horizon/fold parquet is durable before the next fit. Existing chunks
    are validated and reused; unfinished chunks reuse individual fitted models.
    No calibration labels are read for model fitting or this point evaluation.
    """
    _validate_setup(history, contexts, cfg)
    models, _, logs = _paths(out)
    frozen = _full_lock(lock, cfg, logs)  # Must precede every score read.
    lgbm_choice = {"id": frozen["lgbm"]["selected_id"], **frozen["lgbm"]["selected_config"]}
    tcn_choice = {"id": frozen["tcn"]["selected_id"], **frozen["tcn"]["selected_config"]}
    chunks = []
    for h in HORIZONS:
        for f in FOLDS:
            c = _context(contexts, h, f)
            score = pd.DatetimeIndex(c["score"])
            targets = pd.DatetimeIndex(c["target_time"].loc[score])
            if score.empty or score.max() >= BOUNDARY or targets.max() >= BOUNDARY:
                raise ValueError("Score context reaches sealed boundary")
            chunk_path = models / f"score_{_key(h, f)}.parquet"
            if chunk_path.exists():
                frame = pd.read_parquet(chunk_path)
                _validate_chunk(frame, c, h, f)
                chunks.append(frame)
                print(f"Resume score {_key(h, f)} rows={len(frame)}", flush=True)
                continue
            truth = c["y"].loc[score].to_numpy(dtype=float)
            tau = float(c["tau"])
            d2 = c["d2"].loc[score].to_numpy(dtype=bool)
            predictions: dict[str, np.ndarray] = {}
            runtime: dict[str, dict] = {}
            for model in MAIN:
                if model.startswith("B"):
                    prediction, record = _score_baseline(models, model, h, f, c, history,
                                                         frozen["cbl_by_horizon"][str(h)])
                elif model in ("M1", "M1-W"):
                    prediction, record = _score_family(models, "lgbm", model, h, f, c, cfg, lgbm_choice,
                                                       peak_weight=float(cfg["peak_weight"]) if model == "M1-W" else 1.0)
                elif model == "C1":
                    prediction, record = _score_family(models, "lgbm", model, h, f, c, cfg, lgbm_choice,
                                                       residual=True)
                else:
                    prediction, record = _score_family(models, "tcn", model, h, f, c, cfg, tcn_choice,
                                                       residual=False)
                if len(prediction) != len(score):
                    raise ValueError(f"{model} prediction count mismatch at {_key(h, f)}")
                predictions[model] = prediction
                runtime[model] = {**record, "selected_config": json.loads(_config_for_model(model, h, frozen, cfg)),
                                  "horizon": h, "fold": f, "development_only": True}
                print(f"Score {model} {_key(h, f)} seconds={record['inference_seconds']:.1f}", flush=True)
            frames = [pd.DataFrame({"model": model, "horizon": h, "fold": f,
                                    "origin": score, "target_time": targets, "y": truth,
                                    "pred": predictions[model], "tau": tau, "d2": d2,
                                    "train_seconds": runtime[model]["train_seconds"],
                                    "inference_seconds": runtime[model]["inference_seconds"],
                                    "selected_config": _config_for_model(model, h, frozen, cfg),
                                    "development_only": True}) for model in MAIN]
            frame = pd.concat(frames, ignore_index=True)
            _validate_chunk(frame, c, h, f)
            _atomic_parquet(chunk_path, frame)
            _atomic_json(logs / f"runtime_{_key(h, f)}.json", runtime)
            print(f"Completed score {_key(h, f)} rows={len(frame)}", flush=True)
            chunks.append(frame)
    return pd.concat(chunks, ignore_index=True)
