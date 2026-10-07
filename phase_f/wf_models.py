"""Isolated walk-forward model execution and per-seed prediction evidence.

The caller owns the weekly lock, candidate catalog, scoring, and the one-time
CONFIRM gate.  This module only fits on the supplied week's fit/stop context,
predicts its cal/score origins, and returns an average of complete seed paths.
No Phase C or original Phase F prediction cache is accepted as an input.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from phase_f.harness import BOUNDARY, fit_scales, make_frame
from phase_f.registry import config_hash, sha256, write_json

SEEDS = (42, 123, 2024, 3407, 777, 11, 73, 101, 31415, 27182)
FRAME_KEY = ("horizon", "fold", "role", "origin", "target_time")
_ID = re.compile(r"[A-Za-z0-9_.+-]+\Z")


@dataclass
class _ArmView:
    source: Any
    contexts: dict

    @property
    def root(self):
        return self.source.root

    @property
    def out(self):
        return self.source.out

    @property
    def history(self):
        return self.source.history

    @property
    def split_lock(self):
        return self.source.split_lock

    @property
    def seal(self):
        return self.source.seal

    def origins(self, h, fold, role):
        if (h, fold) not in self.contexts:
            raise KeyError((h, fold))
        return pd.DatetimeIndex(self.contexts[(h, fold)][role])


def _arm_view(prepared, arm):
    if arm not in ("EXPLORE", "CONFIRM"):
        raise ValueError("Walk-forward execution arm must be EXPLORE or CONFIRM")
    selected = {}
    for key, context in prepared.contexts.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Walk-forward context key must be (horizon, week_id)")
        h, fold = key
        score = pd.DatetimeIndex(prepared.origins(h, fold, "score"))
        if score.empty:
            raise ValueError(f"Empty weekly score context {key}")
        targets = pd.DatetimeIndex(context["target_time"].loc[score])
        if targets.max() >= BOUNDARY or score.max() >= BOUNDARY:
            raise ValueError("Walk-forward score reached the development boundary")
        expected = np.where(targets.isocalendar().week.to_numpy(dtype=int) % 2 == 0,
                            "EXPLORE", "CONFIRM")
        if len(set(expected)) != 1:
            raise ValueError(f"One weekly fold spans EXPLORE and CONFIRM: {key}")
        if expected[0] == arm:
            selected[key] = context
    if not selected:
        raise ValueError(f"No {arm} weekly contexts")
    # Joint horizon models require every horizon of a week in the same arm.
    by_fold = {}
    for h, fold in selected:
        by_fold.setdefault(fold, set()).add(h)
    all_h = {h for h, _ in prepared.contexts}
    if any(horizons != all_h for horizons in by_fold.values()):
        raise ValueError("Incomplete horizon coverage in selected weekly arm")
    return _ArmView(prepared, selected)


def _pc3_confirm_view(prepared):
    """Keep old Phase C fit/stop/cal roles and only its locked odd-week scores."""
    selected = {}
    for key, original in prepared.contexts.items():
        h, fold = key
        score = pd.DatetimeIndex(original["score"])
        targets = pd.DatetimeIndex(original["target_time"].loc[score])
        if score.empty or targets.max() >= BOUNDARY:
            raise ValueError(f"Invalid Phase C confirm context {key}")
        keep = targets.isocalendar().week.to_numpy(dtype=int) % 2 == 1
        subset = score[keep]
        if subset.empty:
            raise ValueError(f"Phase C fold has no CONFIRM score origins: {key}")
        context = dict(original)
        context["score"] = subset
        selected[key] = context
    if not selected or {fold for _, fold in selected} != {0, 1, 2}:
        raise ValueError("Phase C auxiliary confirm requires all three original folds")
    return _ArmView(prepared, selected)


def _is_stochastic(spec):
    adapter = spec.get("adapter")
    if adapter == "neural":
        return True
    if adapter == "foundation":
        return bool(spec.get("finetune"))
    if adapter == "seed_ensemble":
        return False
    if adapter in ("regression", "multihorizon"):
        if adapter == "multihorizon":
            return True
        kind = str(spec.get("kind", ""))
        if kind not in ("lightgbm", "xgboost", "catboost", "two_stage"):
            return False
        if kind == "catboost":
            return True  # Native bootstrap/random-strength choices consume its seed.
        params = {**spec, **spec.get("params", {}), **spec.get("model_params", {})}
        return (float(params.get("subsample", 1)) < 1 or
                float(params.get("colsample_bytree", 1)) < 1 or
                float(params.get("feature_fraction", 1)) < 1 or
                float(params.get("bagging_fraction", 1)) < 1 or
                int(params.get("subsample_freq", params.get("bagging_freq", 0))) > 0 or
                bool(spec.get("force_seed_repeats", False)))
    if adapter == "baseline":
        return str(spec.get("baseline", spec.get("kind", ""))) == "M2" or bool(spec.get("bagging", False))
    return False


def _seed_list(spec, n_seeds=None):
    stochastic = _is_stochastic(spec)
    expected = 10 if spec.get("finalist") else 5 if stochastic else 1
    if n_seeds is not None:
        requested = int(n_seeds)
        if requested not in ((5, 10) if stochastic else (1,)):
            raise ValueError("Explicit seed count must be 5/10 for a stochastic model or 1 for a deterministic model")
        expected = requested
    if "n_seeds" in spec and int(spec["n_seeds"]) != expected:
        raise ValueError(f"{spec['id']} requires {expected} seeds for this status")
    supplied = spec.get("seeds")
    if supplied is None:
        seeds = SEEDS[:expected]
    else:
        seeds = tuple(int(value) for value in supplied)
        if n_seeds == 10 and len(seeds) == 5 and len(set(seeds)) == 5:
            extension = tuple(value for value in SEEDS if value not in seeds)
            seeds = seeds + extension[:5]
    if len(seeds) != expected or len(set(seeds)) != expected:
        raise ValueError(f"{spec['id']} requires {expected} distinct seeds")
    return seeds


def _seed_spec(spec, seed, arm):
    result = dict(spec)
    result["id"] = f"{spec['id']}__wf_{arm.lower()}__seed{seed}"
    result["seed"] = seed
    if "params" in result:
        result["params"] = {**result["params"], "seed": seed}
    if "model_params" in result:
        model_params = dict(result["model_params"])
        for key in ("random_state", "random_seed", "seed"):
            if key in model_params:
                model_params[key] = seed
        result["model_params"] = model_params
    result["arm"] = arm
    result.pop("seeds", None)
    result.pop("n_seeds", None)
    result.pop("finalist", None)
    result.pop("verify_determinism", None)
    if result.get("adapter") == "neural":
        result["seeds"] = [seed]
        if "params" in result:
            result["params"] = {**result["params"], "seeds": [seed]}
    return result


@contextmanager
def _phase_c_tcn_seed(seed):
    """Use a real alternate seed in immutable Phase C's module-level trainer."""
    from phase_c import tcn

    original = tcn.SEED
    tcn.SEED = int(seed)
    try:
        yield
    finally:
        tcn.SEED = original


def _phase_c_baseline(view, spec):
    """Replay fixed Phase C baseline recipes in each weekly context."""
    from phase_c.statistical import fit_kalman, predict_kalman
    from phase_c.training import _fit_family
    from phase_c.data import _core_features, _sequences
    from outputs.phase_e.code.distribution import predict_kalman_distribution
    from scipy.stats import norm
    from src.models.cbl import cbl_all_predictions
    import joblib

    name = str(spec.get("baseline", spec.get("kind", "")))
    if name not in {"B1", "B2", "B5", "M1", "M1-W", "M2"}:
        raise ValueError(f"Unsupported weekly baseline: {name}")
    cfg = json.loads((Path(view.root) / "configs/phase_c.json").read_text(encoding="utf-8"))
    lock_file = "tcn_config_lock.json" if name == "M2" else "lgbm_config_lock.json"
    lock = json.loads((Path(view.root) / "outputs/phase_c/logs" / lock_file).read_text(encoding="utf-8"))
    b2_lock = json.loads((Path(view.root) / "outputs/phase_c/logs/baseline_config_lock.json").read_text(encoding="utf-8"))
    if name in {"M1", "M1-W", "M2"} and not {"selected_id", "selected_config"} <= set(lock):
        raise ValueError("Phase C fixed family selection lock is incomplete")
    cfg["seed"] = int(spec["seed"])
    choice = {"id": lock["selected_id"], **lock["selected_config"]} if "selected_id" in lock else None
    frames, audits = [], []
    for (h, fold), context in view.contexts.items():
        start = perf_counter()
        if name in {"B1", "B2"}:
            bundle = None
            training = 0.0
        else:
            cell_dir = Path(view.out) / "models" / spec["id"]
            cell_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".pt" if name == "M2" else ".joblib"
            cache = cell_dir / f"h{h:02d}_f{fold}{suffix}"
            meta_path = cache.with_suffix(".json")
            identity = _prediction_identity(view, spec, spec["arm"])
            if cache.exists() != meta_path.exists():
                raise RuntimeError("Incomplete weekly baseline fit transaction")
            if cache.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("identity") != identity or meta.get("sha256") != sha256(cache):
                    raise RuntimeError("Weekly baseline fit cache identity/hash mismatch")
                if name == "M2":
                    import torch
                    bundle = torch.load(cache, map_location="cpu", weights_only=True)
                else:
                    bundle = joblib.load(cache)
                training = float(meta["train_seconds"])
            else:
                if name == "B5":
                    bundle = fit_kalman(view.history.power, pd.DatetimeIndex(view.origins(h, fold, "fit")))
                elif name == "M2":
                    with _phase_c_tcn_seed(int(spec["seed"])):
                        bundle, _, _ = _fit_family("tcn", context, cfg, choice)
                    if int(bundle["metadata"]["seed"]) != int(spec["seed"]):
                        raise AssertionError("Phase C TCN ignored the requested WF seed")
                else:
                    bundle, _, _ = _fit_family("lgbm", context, cfg, choice,
                                               peak_weight=2.0 if name == "M1-W" else 1.0)
                training = perf_counter() - start
                temp = cache.with_suffix(cache.suffix + ".tmp")
                if name == "M2":
                    import torch
                    torch.save(bundle, temp)
                else:
                    joblib.dump(bundle, temp)
                temp.replace(cache)
                write_json(meta_path, {"identity": identity, "sha256": sha256(cache),
                                       "train_seconds": training, "seed": int(spec["seed"]),
                                       "phase_c_recipe": name})
        probe = pd.DatetimeIndex(view.origins(h, fold, "score")[:1])
        if probe.empty:
            raise ValueError("Empty score probe")

        def predict(history, origins):
            if name == "B1":
                if history is view.history:
                    return context["x"].loc[origins, "slot7d"].to_numpy(float)
                features, _ = _core_features(history, origins, h)
                return features["slot7d"].to_numpy(float)
            if name == "B2":
                method = b2_lock["cbl_by_horizon"][str(h)]
                return cbl_all_predictions(history, origins, h)[method].to_numpy(float)
            if name == "B5":
                return np.asarray(predict_kalman(bundle, history.power, origins, h), float)
            if name == "M2":
                from phase_c.tcn import predict_tcn
                columns = cfg["tcn"]["context_columns"]
                if history is view.history:
                    seq = context["seq"].loc[origins].to_numpy(dtype=np.float32)
                    features = context["x"].loc[origins, columns].to_numpy(dtype=np.float32)
                else:
                    seq = _sequences(history).loc[origins].to_numpy(dtype=np.float32)
                    x, provenance = _core_features(history, origins, h)
                    if any((used > origins).fillna(False).any() for used in provenance.values()):
                        raise AssertionError("Phase C TCN feature read after origin")
                    features = x.loc[:, columns].to_numpy(dtype=np.float32)
                return np.asarray(predict_tcn(bundle, seq, features), float)
            if history is view.history:
                features = context["x"].loc[origins]
            else:
                features, provenance = _core_features(history, origins, h)
                if any((used > origins).fillna(False).any() for used in provenance.values()):
                    raise AssertionError("Phase C feature read after forecast origin")
            return np.asarray(bundle.predict(features), float)

        before = predict(view.history, probe)
        altered = view.history.copy()
        mask = altered.index > probe[0]
        altered.loc[mask, "power"] = np.random.default_rng(42).normal(10000, 500, int(mask.sum()))
        after = predict(altered, probe)
        if not np.array_equal(before, after, equal_nan=True):
            raise AssertionError(f"Weekly {name} prediction changed after future perturbation")
        for role in ("cal", "score"):
            origins = pd.DatetimeIndex(view.origins(h, fold, role))
            tick = perf_counter()
            prediction = predict(view.history, origins)
            distribution = {}
            if name == "B5" and h == 16:
                native_mean, sigma = predict_kalman_distribution(bundle, view.history.power, origins, h)
                if not np.array_equal(prediction, native_mean):
                    raise AssertionError("Weekly B5 native mean differs from the point forecast")
                distribution = {"sigma": sigma,
                                "p_raw": norm.sf((float(context["tau"]) - prediction) / sigma),
                                "p_peak": norm.sf((float(context["tau"]) - prediction) / sigma),
                                "q10": prediction + norm.ppf(.10) * sigma,
                                "q50": prediction,
                                "q90": prediction + norm.ppf(.90) * sigma,
                                "q95": prediction + norm.ppf(.95) * sigma}
            elapsed = perf_counter() - tick
            if not np.isfinite(prediction).all():
                raise ValueError(f"Weekly baseline {name} returned a nonfinite prediction")
            frames.append(make_frame(context, origins, h, fold, prediction,
                                     spec["id"], role, train_seconds=training,
                                     train_seconds_run_id=f"{spec['id']}_{h}_{fold}",
                                     inference_seconds=elapsed, **distribution))
        audits.append({"horizon": h, "fold": str(fold), "train_seconds": training,
                       "source": "Phase C fixed model recipe refit on weekly fit/stop",
                       "future_perturbation_max_abs_difference": 0,
                       "native_training_seed": int(bundle["metadata"]["seed"]) if name == "M2" else None,
                       "stop_MAE": float(bundle["metadata"]["best_stop_mae"]) if name == "M2" else None})
    return pd.concat(frames, ignore_index=True), {
        "cells": audits, "leakage_test": "passed",
        "phase_c_selection_sha256": sha256(Path(view.root) / "outputs/phase_c/logs" / lock_file)
        if name in {"M1", "M1-W", "M2"} else None,
        "phase_c_b2_method_lock_sha256": sha256(Path(view.root) / "outputs/phase_c/logs/baseline_config_lock.json")
        if name == "B2" else None}


def _ensemble(view, spec):
    from phase_f.ensemble import combine_predictions

    bases = tuple(spec.get("bases", ()))
    if not bases or len(set(bases)) != len(bases):
        raise ValueError("Ensemble needs distinct completed parent IDs")
    paths = {base: view.out / "predictions" / spec["arm"] / f"{base}.parquet" for base in bases}
    frames = {}
    digests = {}
    for base, path in paths.items():
        manifest = path.with_suffix(".json")
        if not path.is_file() or not manifest.is_file():
            raise ValueError(f"Missing sealed weekly ensemble parent: {base}")
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        digest = sha256(path)
        if meta.get("sha256") != digest or meta.get("arm") != spec["arm"]:
            raise RuntimeError(f"Weekly ensemble parent changed: {base}")
        frames[base] = pd.read_parquet(path)
        digests[base] = digest
    combined, audit = combine_predictions(frames, spec["method"], spec["id"],
                                           params=spec.get("params", {}))
    audit.update(parent_sha256=digests, leakage_test="passed")
    return combined, audit


def _execute_one(view, spec):
    adapter = spec.get("adapter")
    if adapter == "baseline":
        return _phase_c_baseline(view, spec)
    if adapter in {"regression", "statistical", "neural"}:
        from phase_f.run import run_adapter
        return run_adapter(view, spec)
    if adapter == "foundation":
        from phase_f.models.foundation import run
        return run(view, spec)
    if adapter == "multihorizon":
        from phase_f.models.multihorizon import run
        return run(view, spec)
    if adapter == "other_foundation":
        from phase_f.models.other_foundation import run
        return run(view, spec)
    if adapter == "ensemble":
        return _ensemble(view, spec)
    raise ValueError(f"Unsupported walk-forward adapter: {adapter}")


def _validate_frame(view, frame, model, arm):
    required = set(FRAME_KEY) | {"pred", "y", "tau", "d2", "arm", "model",
                                 "fit_mean", "mase_scale"}
    if frame.empty or not required <= set(frame):
        raise ValueError("Weekly prediction frame is empty or lacks required columns")
    if not frame.model.eq(model).all() or frame.duplicated(list(FRAME_KEY)).any():
        raise ValueError("Weekly prediction model or keys conflict")
    if not np.isfinite(frame[["pred", "y", "tau", "fit_mean", "mase_scale"]].to_numpy(float)).all():
        raise ValueError("Weekly prediction contains nonfinite values")
    expected = []
    for (h, fold), context in view.contexts.items():
        for role in ("cal", "score"):
            origins = pd.DatetimeIndex(view.origins(h, fold, role))
            target = pd.DatetimeIndex(context["target_time"].loc[origins])
            expected.extend((h, fold, role, origin, end) for origin, end in zip(origins, target))
    found = list(frame[list(FRAME_KEY)].itertuples(index=False, name=None))
    if len(found) != len(expected) or set(found) != set(expected):
        raise ValueError("Weekly prediction does not match the selected locked cohort")
    for (h, fold), context in view.contexts.items():
        fit_mean, mase_scale = fit_scales(context)
        for role in ("cal", "score"):
            origins = pd.DatetimeIndex(view.origins(h, fold, role))
            selected = frame.loc[frame.horizon.eq(h) & frame.fold.eq(fold)
                                 & frame.role.eq(role)].set_index("origin").loc[origins]
            if not np.array_equal(selected.y.to_numpy(float),
                                  context["y"].loc[origins].to_numpy(float)):
                raise ValueError("Weekly prediction truth differs from locked context")
            if not np.array_equal(selected.tau.to_numpy(float),
                                  np.repeat(float(context["tau"]), len(origins))):
                raise ValueError("Weekly peak threshold differs from fit-only context")
            if selected.d2.isna().any() or not selected.d2.isin((True, False)).all() or not np.array_equal(
                    selected.d2.to_numpy(bool), context["d2"].loc[origins].to_numpy(bool)):
                raise ValueError("Weekly D2 cohort differs from locked context")
            for column, expected_value in (("fit_mean", fit_mean), ("mase_scale", mase_scale)):
                if not np.array_equal(selected[column].to_numpy(float),
                                      np.repeat(expected_value, len(origins))):
                    raise ValueError(f"Weekly {column} differs from fit-only context")
    score = frame.role.eq("score")
    if not frame.loc[score, "arm"].eq(arm).all() or not frame.loc[~score, "arm"].eq("CAL").all():
        raise ValueError("Weekly prediction arm does not match selected arm")
    if (pd.DatetimeIndex(frame.target_time) >= BOUNDARY).any():
        raise ValueError("Weekly prediction crossed development boundary")


def _prediction_identity(view, spec, arm):
    roles = {str((h, f)): {role: [str(x) for x in view.origins(h, f, role)]
                          for role in ("fit", "stop", "cal", "score")}
             for h, f in view.contexts}
    sources = {name: sha256(Path(view.root) / name) for name in (
        "phase_f/wf_models.py", "phase_f/harness.py")}
    adapter = spec.get("adapter")
    if adapter == "baseline":
        for name in ("phase_c/statistical.py", "phase_c/training.py", "src/models/lgbm_point.py",
                     "configs/phase_c.json", "outputs/phase_c/logs/lgbm_config_lock.json",
                     "outputs/phase_c/logs/baseline_config_lock.json",
                     "outputs/phase_c/logs/tcn_config_lock.json", "phase_c/tcn.py",
                     "phase_c/data.py", "src/models/cbl.py",
                     "outputs/phase_e/code/distribution.py"):
            sources[name] = sha256(Path(view.root) / name)
    elif adapter in {"regression", "statistical", "neural", "foundation", "multihorizon", "other_foundation"}:
        name = f"phase_f/models/{adapter}.py"
        sources[name] = sha256(Path(view.root) / name)
        if adapter in {"regression", "multihorizon"}:
            sources["phase_f/features_ext.py"] = sha256(Path(view.root) / "phase_f/features_ext.py")
    return config_hash({"spec": spec, "arm": arm, "roles": roles,
                        "split": view.split_lock.get("lock_sha256"),
                        "seal": view.seal.get("raw_sha256"), "sources": sources})


def _seed_path(view, spec_id, arm, seed):
    return Path(view.out) / "predictions" / "seeds" / arm / spec_id / f"seed_{seed}.parquet"


def _load_or_fit_seed(view, seed_spec, parent_id, arm, seed):
    path = _seed_path(view, parent_id, arm, seed)
    meta_path = path.with_suffix(".json")
    identity = _prediction_identity(view, seed_spec, arm)
    if path.exists() and not meta_path.exists():
        # The parquet was atomically published before its manifest. Preserve
        # those exact bytes for diagnosis, then refit the missing transaction.
        from phase_f.run import complete_prediction_cache

        if complete_prediction_cache(path, meta_path):
            raise AssertionError("Orphan weekly seed parquet was not quarantined")
    elif meta_path.exists() and not path.exists():
        # A manifest alone is also incomplete. Keep its checksum-addressed
        # bytes rather than letting a new fit silently inherit the stale JSON.
        digest = sha256(meta_path)[:12]
        orphan = meta_path.with_name(f"{meta_path.stem}.orphan-{digest}{meta_path.suffix}")
        counter = 0
        while orphan.exists():
            counter += 1
            orphan = meta_path.with_name(
                f"{meta_path.stem}.orphan-{digest}-{counter}{meta_path.suffix}")
        meta_path.rename(orphan)
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("identity") != identity or meta.get("sha256") != sha256(path):
            raise RuntimeError("Weekly seed cache changed; preserve it and use a new ID")
        frame = pd.read_parquet(path)
        _validate_frame(view, frame, seed_spec["id"], arm)
        return frame, dict(meta.get("audit", {})), True
    frame, audit = _execute_one(view, seed_spec)
    if "cal_provenance" not in frame:
        frame = frame.copy()
        frame["cal_provenance"] = np.where(frame.role.eq("cal"), "fit_only", "not_cal")
    _validate_frame(view, frame, seed_spec["id"], arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)
    write_json(meta_path, {"identity": identity, "sha256": sha256(path),
                           "arm": arm, "seed": seed, "audit": audit,
                           "development_only": True, "holdout_read": False})
    return frame, audit, False


def _seed_auc(frame, arm):
    selected = frame.loc[frame.role.eq("score") & frame.arm.eq(arm)]
    errors = (selected.y - selected.pred).abs()
    return float(errors.groupby(selected.horizon).mean().mean())


def _seed_ensemble(view, spec, arm):
    parent = spec.get("parent_spec")
    if not isinstance(parent, Mapping) or parent.get("adapter") != "regression":
        raise ValueError("Seed ensemble requires its frozen regression parent_spec")
    if not _ID.fullmatch(str(parent.get("id", ""))):
        raise ValueError("Invalid seed ensemble parent ID")
    selected = tuple(int(x) for x in spec.get("seeds", ()))
    if len(selected) not in (5, 10, 20) or len(set(selected)) != len(selected):
        raise ValueError("Seed ensemble requires 5, 10, or 20 distinct seeds")
    if any(seed < 0 for seed in selected):
        raise ValueError("Seeds must be nonnegative")
    frames, records = [], []
    for seed in selected:
        child = _seed_spec({**parent, "arm": arm}, seed, arm)
        frame, audit, reused = _load_or_fit_seed(view, child, parent["id"], arm, seed)
        frames.append(frame)
        records.append({"seed": seed, "prediction_sha256": sha256(_seed_path(view, parent["id"], arm, seed)),
                        "auc_mae": _seed_auc(frame, arm), "cache_reused": reused,
                        "leakage_test": audit.get("leakage_test")})
    mean = _mean_frames(frames, spec["id"])
    _validate_frame(view, mean, spec["id"], arm)
    scores = np.array([row["auc_mae"] for row in records], float)
    return mean, {"arm": arm, "n_seeds": len(selected), "seeds": records,
                  "seed_auc_mae_sd": float(scores.std(ddof=1)),
                  "ranking_basis": "metric of mean prediction, not best seed",
                  "leakage_test": "passed" if all(row["leakage_test"] == "passed" for row in records)
                                  else "requires_family_audit",
                  "development_only": True, "holdout_read": False}


def _mean_frames(frames, model):
    reference = frames[0].sort_values(list(FRAME_KEY), kind="stable").reset_index(drop=True)
    result = reference.copy()
    values = []
    optional = ("q10", "q50", "q90", "q95", "p_peak", "p_raw", "sigma")
    for frame in frames:
        ordered = frame.sort_values(list(FRAME_KEY), kind="stable").reset_index(drop=True)
        for name in ("y", "tau", "d2", "arm", "fit_mean", "mase_scale", "cal_provenance"):
            if name in reference and not np.array_equal(ordered[name].to_numpy(), reference[name].to_numpy()):
                raise ValueError(f"Seed cohort metadata differs: {name}")
        if not np.array_equal(ordered[list(FRAME_KEY)].to_numpy(), reference[list(FRAME_KEY)].to_numpy()):
            raise ValueError("Seed prediction keys differ")
        values.append(ordered.pred.to_numpy(float))
    result["pred"] = np.mean(values, axis=0)
    for column in optional:
        if all(column in frame for frame in frames):
            matrix = np.stack([frame.sort_values(list(FRAME_KEY))[column].to_numpy(float)
                               for frame in frames])
            finite = np.isfinite(matrix)
            if not np.all(finite.all(axis=0) | (~finite).all(axis=0)):
                raise ValueError(f"Seed distribution coverage differs: {column}")
            result[column] = np.where(finite.all(axis=0),
                                      np.sum(np.where(finite, matrix, 0), axis=0) / len(frames), np.nan)
        elif column in result:
            result.drop(columns=[column], inplace=True)
    result["model"] = model
    result["n_seeds"] = len(frames)
    if "cal_provenance" not in result:
        result["cal_provenance"] = np.where(result.role.eq("cal"), "fit_only", "not_cal")
    return result


def run(prepared, spec, *, arm=None, n_seeds=None):
    """Return seed-mean predictions and an auditable per-seed summary.

    `spec['arm']` is mandatory. The caller must gate CONFIRM before calling
    this function. Each seed is fitted or loaded only from the separate WF
    output namespace and saved before averaging, so a winner cannot be chosen
    by looking at its best individual seed.
    """
    if not _ID.fullmatch(str(spec.get("id", ""))):
        raise ValueError("Unsafe walk-forward experiment ID")
    if arm is not None and spec.get("arm", arm) != arm:
        raise ValueError("Conflicting walk-forward arm in spec and call")
    arm = arm or spec.get("arm", "EXPLORE")
    spec = {**spec, "arm": arm}
    view = _arm_view(prepared, arm)
    if not view.split_lock.get("lock_sha256") or not view.seal.get("raw_sha256"):
        raise ValueError("Walk-forward split and raw-data seals are required")
    if spec.get("adapter") == "seed_ensemble":
        return _seed_ensemble(view, spec, arm)
    seeds = _seed_list(spec, n_seeds=n_seeds)
    frames, seed_records = [], []
    for seed in seeds:
        one = _seed_spec(spec, seed, arm)
        frame, audit, reused = _load_or_fit_seed(view, one, spec["id"], arm, seed)
        frames.append(frame)
        seed_records.append({"seed": seed, "prediction_sha256": sha256(_seed_path(view, spec["id"], arm, seed)),
                             "auc_mae": _seed_auc(frame, arm), "cache_reused": reused,
                             "leakage_test": audit.get("leakage_test"), "audit": audit})
    mean = _mean_frames(frames, spec["id"])
    _validate_frame(view, mean, spec["id"], arm)
    scores = np.array([row["auc_mae"] for row in seed_records], float)
    repeat = None
    if len(seeds) == 1 and spec.get("verify_determinism", False):
        shadow = {**_seed_spec(spec, seeds[0], arm),
                  "id": f"{spec['id']}__wf_{arm.lower()}__fresh_repeat",
                  "fresh_run_nonce": "independent_determinism_check_v1"}
        shadow_frame, _ = _execute_one(view, shadow)
        _validate_frame(view, shadow_frame, shadow["id"], arm)
        ordered = shadow_frame.sort_values(list(FRAME_KEY), kind="stable").reset_index(drop=True)
        reference = frames[0].sort_values(list(FRAME_KEY), kind="stable").reset_index(drop=True)
        if not np.array_equal(ordered.pred.to_numpy(), reference.pred.to_numpy()):
            raise AssertionError("Fresh isolated deterministic rerun differed")
        repeat = {"fresh_cache_identity": config_hash(shadow),
                  "max_abs_prediction_difference": 0.0}
    audit = {"arm": arm, "n_seeds": len(seeds), "seeds": seed_records,
             "seed_auc_mae_mean": float(scores.mean()),
             "seed_auc_mae_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
             "ranking_basis": "metric of mean prediction, not best seed",
             "leakage_test": "passed" if all(row["leakage_test"] == "passed" for row in seed_records)
                              else "requires_family_audit",
             "fresh_determinism_check": repeat,
             "development_only": True, "holdout_read": False}
    cell_mae = [float(cell["stop_MAE"]) for row in seed_records
                for cell in row["audit"].get("cells", ()) if cell.get("stop_MAE") is not None]
    if cell_mae:
        audit["stop_MAE"] = float(np.mean(cell_mae))
        audit["stop_cells"] = len(cell_mae)
    return mean, audit


def run_pc3_confirm(prepared, spec, *, n_seeds=10):
    """One caller-authorized auxiliary replay on odd target weeks of Phase C folds.

    The caller must commit the CONFIRM reservation before invoking this API.
    This module never decides which finalists to send to CONFIRM. The supplied
    `prepared.out` must be a separate `pc3_finalists` namespace with its own
    model directory; no original Phase C/Phase F cache is consulted.
    """
    if not _ID.fullmatch(str(spec.get("id", ""))):
        raise ValueError("Unsafe Phase C auxiliary finalist ID")
    if spec.get("arm", "CONFIRM") != "CONFIRM":
        raise ValueError("Phase C auxiliary replay is CONFIRM-only")
    if not str(Path(prepared.out).name).startswith("pc3_finalists"):
        raise ValueError("Phase C auxiliary replay requires isolated pc3_finalists output")
    view = _pc3_confirm_view(prepared)
    if not view.split_lock.get("lock_sha256") or not view.seal.get("raw_sha256"):
        raise ValueError("Phase C auxiliary replay needs split and raw-data seals")
    if spec.get('adapter')=='seed_ensemble':
        frame,audit=_seed_ensemble(view,{**spec,'arm':'CONFIRM'},'CONFIRM')
        return frame,{**audit,'cohort':'pc3_confirm'}
    internal = {**spec, "arm": "CONFIRM", "id": f"{spec['id']}__pc3"}
    if _is_stochastic(internal):
        if int(n_seeds) != 10:
            raise ValueError("Phase C stochastic finalists require ten seeds")
        seeds = _seed_list(internal, n_seeds=10)
    else:
        seeds = _seed_list(internal, n_seeds=1)
    frames, records = [], []
    for seed in seeds:
        child = _seed_spec(internal, seed, "CONFIRM")
        frame, audit, reused = _load_or_fit_seed(view, child, internal["id"], "CONFIRM", seed)
        frames.append(frame)
        records.append({"seed": seed, "prediction_sha256": sha256(_seed_path(view, internal["id"], "CONFIRM", seed)),
                        "auc_mae": _seed_auc(frame, "CONFIRM"), "cache_reused": reused,
                        "leakage_test": audit.get("leakage_test")})
    mean = _mean_frames(frames, spec["id"])
    _validate_frame(view, mean, spec["id"], "CONFIRM")
    scores = np.array([row["auc_mae"] for row in records], float)
    return mean, {"cohort": "pc3_confirm", "arm": "CONFIRM", "n_seeds": len(seeds),
                  "seeds": records, "seed_auc_mae_sd": float(scores.std(ddof=1)) if len(scores)>1 else 0.,
                  "ranking_basis": "metric of mean prediction, not best seed",
                  "leakage_test": "passed" if all(row["leakage_test"] == "passed" for row in records)
                                  else "requires_family_audit",
                  "development_only": True, "holdout_read": False}
