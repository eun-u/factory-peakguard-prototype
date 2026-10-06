"""Causal Chronos-2 R1 paths for an EXPLORE-only residual experiment.

``prepared`` is the EXPLORE arm view, not the full weekly object. The anchor
is the completed F0-1-R1 EXPLORE prediction frame; only its selected cells'
forecast columns are read. Cached paths contain no targets or score metrics.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from phase_c.data import HORIZONS
from phase_f.harness import BOUNDARY
from phase_f.models.foundation import QUANTILES, causal_input
from phase_f.registry import config_hash, sha256, write_json


PATH_COLUMNS = ("origin", "horizon", "r1", "q10", "q50", "q90", "q95")
FORECAST_COLUMNS = ("q10", "q50", "q90", "q95")
ROLES = ("fit", "stop", "cal", "score")
CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 96
BATCH_SIZE = 32
SEED = 42
CHUNK_ORIGINS = 320  # Ten model batches per durable checkpoint.
ANCHOR_RECIPE_ATOL = 1e-4


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g",
                           date_format="%Y-%m-%dT%H:%M:%S.%f").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sort_paths(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, PATH_COLUMNS].copy()
    result["origin"] = pd.to_datetime(result["origin"])
    result["horizon"] = result["horizon"].astype("int16")
    for col in PATH_COLUMNS[2:]:
        result[col] = result[col].astype("float64")
    return result.sort_values(["origin", "horizon"]).reset_index(drop=True)


def _required(prepared) -> pd.DataFrame:
    if not prepared.contexts:
        raise ValueError("R1 paths require selected EXPLORE weekly contexts")
    rows = []
    for (h, fold), context in sorted(prepared.contexts.items()):
        if int(h) not in HORIZONS:
            raise ValueError("R1 path horizon differs from Phase C horizons")
        if context.get("summary", {}).get("arm", "EXPLORE") != "EXPLORE":
            raise ValueError("R1 path context includes a non-EXPLORE arm")
        score = pd.DatetimeIndex(prepared.origins(h, fold, "score"))
        if score.empty:
            raise ValueError("Empty R1 EXPLORE score context")
        targets = pd.DatetimeIndex(context["target_time"].loc[score])
        if np.any(targets.isocalendar().week.to_numpy(dtype=int) % 2 != 0):
            raise ValueError("R1 path context contains CONFIRM score targets")
        for role in ROLES:
            origins = pd.DatetimeIndex(prepared.origins(h, fold, role))
            if origins.empty:
                raise ValueError(f"Empty R1 {role} context")
            if origins.max() >= BOUNDARY:
                raise ValueError("R1 origin reached sealed boundary")
            rows.append(pd.DataFrame({"origin": origins, "horizon": int(h)}))
    required = pd.concat(rows, ignore_index=True).drop_duplicates()
    return required.sort_values(["origin", "horizon"]).reset_index(drop=True)


def _anchor_paths(prepared, anchor: pd.DataFrame, required: pd.DataFrame) -> pd.DataFrame:
    # Select by metadata before touching y/pred/quantiles from an unselected arm.
    columns = set(anchor.columns)
    needed = {"horizon", "fold", "origin", "role", "pred", *FORECAST_COLUMNS}
    if not needed <= columns:
        raise ValueError(f"R1 anchor is missing {sorted(needed - columns)}")
    selected = pd.MultiIndex.from_tuples(prepared.contexts, names=["horizon", "fold"])
    metadata = anchor.loc[:, ["horizon", "fold"]]
    mask = pd.MultiIndex.from_frame(metadata).isin(selected)
    selected_rows = anchor.loc[mask, ["horizon", "fold", "origin", "role", "pred",
                                      *FORECAST_COLUMNS]].copy()
    if selected_rows.empty:
        raise ValueError("R1 anchor has no selected EXPLORE cells")
    if not selected_rows.role.isin(("cal", "score")).all():
        raise ValueError("R1 anchor has unsupported selected roles")
    selected_rows["origin"] = pd.to_datetime(selected_rows["origin"])
    for (h, fold), group in selected_rows.groupby(["horizon", "fold"], sort=False):
        context = prepared.contexts[(int(h), int(fold))]
        for role, role_rows in group.groupby("role", sort=False):
            actual = pd.DatetimeIndex(role_rows.origin)
            expected = pd.DatetimeIndex(context[role])
            if not actual.isin(expected).all():
                raise ValueError("R1 anchor origin differs from selected cohort")
    values = selected_rows.loc[:, ["pred", *FORECAST_COLUMNS]].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.array_equal(values[:, 0], values[:, 2]):
        raise ValueError("R1 anchor is nonfinite or its median differs from pred")
    paths = selected_rows.rename(columns={"pred": "r1"}).loc[:, PATH_COLUMNS]
    # The same causal origin/horizon can recur in several selected week contexts.
    # Repeated predictions must be byte-identical, not averaged.
    repeated = paths.loc[paths.duplicated(["origin", "horizon"], keep=False)]
    if (not repeated.empty and repeated.groupby(["origin", "horizon"], sort=False)
            .agg({column: "nunique" for column in PATH_COLUMNS[2:]})
            .gt(1).any().any()):
        raise ValueError("Duplicate R1 anchor predictions disagree")
    paths = _sort_paths(paths.drop_duplicates(["origin", "horizon"]))
    keys = pd.MultiIndex.from_frame(paths.loc[:, ["origin", "horizon"]])
    required_keys = pd.MultiIndex.from_frame(required)
    if not keys.isin(required_keys).all():
        raise ValueError("R1 anchor includes keys outside selected roles")
    return paths


def _verify_anchor_source(prepared, anchors: pd.DataFrame, revision: str) -> dict:
    """Prove prefills are exactly the already-audited zero-shot R1 EXPLORE run."""
    root = Path(prepared.root)
    base = root / "outputs/phase_f/walkforward_v2"
    parquet = base / "predictions/EXPLORE/F0-1-R1.parquet"
    metadata = parquet.with_suffix(".json")
    experiment = base / "logs/experiments/F0-1-R1.json"
    if not all(path.is_file() for path in (parquet, metadata, experiment)):
        raise FileNotFoundError("Completed R1 EXPLORE evidence is missing")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    exp = json.loads(experiment.read_text(encoding="utf-8"))
    physical_sha = sha256(parquet)
    if (manifest.get("arm") != "EXPLORE" or manifest.get("sha256") != physical_sha
            or manifest.get("prediction_sha256") != physical_sha
            or exp.get("prediction_sha256") != physical_sha
            or exp.get("status") != "completed"
            or exp.get("leakage_test") != "passed"
            or exp.get("holdout_read") is not False):
        raise RuntimeError("R1 anchor physical prediction or completion evidence changed")
    recipe = json.loads(exp["config_json"])
    if (recipe.get("adapter") != "foundation" or recipe.get("context_length") != CONTEXT_LENGTH
            or recipe.get("finetune") is not False or recipe.get("point") != "median"
            or recipe.get("verify_determinism") is not True
            or manifest.get("config_hash") != config_hash(recipe)):
        raise RuntimeError("R1 anchor is not the pinned zero-shot median recipe")
    run = manifest["audit"]
    seeds = run.get("seeds", [])
    if len(seeds) != 1 or seeds[0].get("seed") != SEED:
        raise RuntimeError("R1 anchor seed evidence changed")
    model_audit = seeds[0].get("audit", {})
    inputs = model_audit.get("input_spec", {})
    cells = model_audit.get("fit_cells", [])
    if (model_audit.get("model_revision") != revision
            or inputs.get("model_revision") != revision
            or inputs.get("split_sha256") != prepared.split_lock["lock_sha256"]
            or inputs.get("quantiles") != QUANTILES
            or inputs.get("context_length") != CONTEXT_LENGTH
            or inputs.get("finetune") is not False
            or inputs.get("seed") != SEED
            or model_audit.get("holdout_read") is not False
            or len(cells) != 1 or cells[0].get("fold") != -1
            or cells[0].get("future_perturbation_max_abs_difference") != 0.0
            or run.get("fresh_determinism_check", {}).get("max_abs_prediction_difference") != 0.0):
        raise RuntimeError("R1 anchor causal and determinism audits are incomplete")
    # Read only metadata and predictions from the selected physical source.
    columns = ["horizon", "fold", "origin", "role", "pred", *FORECAST_COLUMNS]
    source = pd.read_parquet(parquet, columns=columns)
    physical_paths = _anchor_paths(prepared, source, _required(prepared))
    if not physical_paths.equals(anchors):
        raise RuntimeError("R1 anchor values differ from the sealed EXPLORE file")
    return {"prediction_file_sha256": physical_sha,
            "prediction_manifest_sha256": sha256(metadata),
            "anchor_prediction_sha256": _hash_frame(anchors),
            "anchor_original_future_perturbation_max_abs_difference": 0.0,
            "anchor_original_fresh_repeat_max_abs_difference": 0.0}


def _snapshot(prepared) -> tuple[Path, dict]:
    root = Path(prepared.root)
    metadata = root / "outputs/phase_c/logs/chronos_model_download.json"
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if info.get("model_id") != "amazon/chronos-2" or info.get("development_only") is not True:
        raise ValueError("Pinned Chronos-2 metadata is not development-only")
    revision = str(info["revision"])
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Invalid pinned Chronos-2 revision")
    snapshot = root / "outputs/phase_c/models/hf_cache/models--amazon--chronos-2/snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned local Chronos-2 snapshot is missing")
    files = {p.relative_to(snapshot).as_posix(): sha256(p)
             for p in sorted(snapshot.rglob("*")) if p.is_file()}
    if not files:
        raise FileNotFoundError("Pinned Chronos-2 snapshot has no files")
    return snapshot, {"metadata_sha256": sha256(metadata), "model_id": info["model_id"],
                      "revision": revision, "snapshot_files": files}


def _execution_fingerprint(device: str) -> dict:
    """Bind checkpoint reuse to the exact installed inference implementation."""
    import torch

    distribution = importlib.metadata.distribution("chronos-forecasting")
    package = Path(distribution.locate_file("chronos"))
    sources = {path.relative_to(package).as_posix(): sha256(path)
               for path in sorted(package.rglob("*.py")) if path.is_file()}
    if not sources or "chronos2/pipeline.py" not in sources:
        raise FileNotFoundError("Installed Chronos-2 pipeline source is unavailable")
    return {"device": device, "chronos_version": distribution.version,
            "chronos_source_sha256": sources,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda}


def _identity(prepared, required: pd.DataFrame, anchors: pd.DataFrame,
              model: dict, anchor_source: dict, execution: dict) -> dict:
    root = Path(prepared.root)
    history = prepared.history
    if (history.empty or history.index.has_duplicates or not history.index.is_monotonic_increasing
            or history.index.max() >= BOUNDARY):
        raise ValueError("R1 history is not sealed development history")
    if "power" not in history:
        raise ValueError("R1 history lacks the observed power series")
    hashed_history = pd.util.hash_pandas_object(history[["power"]], index=True).to_numpy()
    sources = {name: sha256(root / name) for name in
               ("phase_f/goal_r1_paths.py", "phase_f/models/foundation.py",
                "phase_f/wf_harness.py")}
    return {"schema": 1, "arm": "EXPLORE", "model": model,
            "execution": execution,
            "config": {"context_length": CONTEXT_LENGTH,
                       "prediction_length": PREDICTION_LENGTH,
                       "quantiles": QUANTILES, "batch_size": BATCH_SIZE,
                       "seed": SEED, "dtype": "float32", "covariates": False},
            "split_sha256": prepared.split_lock["lock_sha256"],
            "seal_sha256": config_hash(prepared.seal),
            "history_sha256": hashlib.sha256(hashed_history.tobytes()).hexdigest(),
            "required_sha256": _hash_frame(required),
            "anchor_sha256": _hash_frame(anchors),
            "anchor_source": anchor_source,
            "source_sha256": sources}


def _write_manifest(path: Path, state: dict) -> None:
    state = {key: value for key, value in state.items() if key != "manifest_sha256"}
    state["manifest_sha256"] = config_hash(state)
    write_json(path, state)


def _quarantine(path: Path, reason: str) -> Path:
    digest = sha256(path)[:12]
    target = path.with_name(f"{path.name}.{reason}-{digest}-{uuid.uuid4().hex[:8]}")
    os.replace(path, target)
    return target


def _read_chunks(cache: Path, state: dict) -> tuple[pd.DataFrame, bool]:
    changed = False
    good = []
    frames = []
    for entry in state["parts"]:
        path = cache / entry["file"]
        valid = path.is_file() and sha256(path) == entry["sha256"]
        if valid:
            try:
                frame = _sort_paths(pd.read_parquet(path))
                valid = (len(frame) == entry["rows"] and not frame.duplicated(["origin", "horizon"]).any()
                         and _hash_frame(frame) == entry["content_sha256"])
            except Exception:
                valid = False
        if valid:
            good.append(entry)
            frames.append(frame)
        else:
            if path.exists():
                _quarantine(path, "corrupt")
            changed = True
    state["parts"] = good
    known = {entry["file"] for entry in good}
    for path in cache.glob("part_*.parquet"):
        if path.name not in known:
            _quarantine(path, "orphan")
            changed = True
    inferred = (pd.concat(frames, ignore_index=True) if frames else
                pd.DataFrame(columns=PATH_COLUMNS))
    if not inferred.empty and inferred.duplicated(["origin", "horizon"]).any():
        raise RuntimeError("R1 cache chunks have duplicate forecast keys")
    if changed:
        state["complete"] = False
    return inferred, changed


def _load_pipeline(snapshot: Path, device: str):
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    pipe = Chronos2Pipeline.from_pretrained(str(snapshot), device_map=device,
                                           torch_dtype=torch.float32)
    if CONTEXT_LENGTH > pipe.model_context_length:
        raise ValueError("Pinned Chronos-2 model cannot support context length 2048")
    return pipe


def _quantile_array(prediction, n: int) -> np.ndarray:
    # Chronos2Pipeline.predict_quantiles returns one (1, horizon, quantile)
    # tensor per input in its public list result. Some test backends return a
    # pre-stacked tensor, so accept that shape without changing the task axis.
    if isinstance(prediction, (list, tuple)):
        if len(prediction) != n:
            raise ValueError("Chronos quantile list length differs from input batch")
        result = np.stack([item.detach().cpu().numpy() for item in prediction], axis=0)
    else:
        result = prediction.detach().cpu().numpy()
    if result.shape != (n, 1, PREDICTION_LENGTH, len(QUANTILES)):
        raise ValueError(f"Unexpected Chronos quantile shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Chronos quantiles")
    return result


def _future_probe(pipe, history: pd.DataFrame, origin: pd.Timestamp) -> float:
    import torch

    altered = history.copy()
    mask = altered.index > origin
    altered.loc[mask, "power"] = np.random.default_rng(SEED).normal(10000, 1000, int(mask.sum()))
    before = causal_input(history, origin, CONTEXT_LENGTH, PREDICTION_LENGTH, False)
    after = causal_input(altered, origin, CONTEXT_LENGTH, PREDICTION_LENGTH, False)
    if not np.array_equal(before, after, equal_nan=True):
        raise AssertionError("Causal input read future observations")
    with torch.inference_mode():
        left, _ = pipe.predict_quantiles([before], prediction_length=PREDICTION_LENGTH,
                                         quantile_levels=QUANTILES, context_length=CONTEXT_LENGTH)
        right, _ = pipe.predict_quantiles([after], prediction_length=PREDICTION_LENGTH,
                                          quantile_levels=QUANTILES, context_length=CONTEXT_LENGTH)
    left_array = _quantile_array(left, 1)
    right_array = _quantile_array(right, 1)
    difference = float(np.max(np.abs(left_array - right_array)))
    if difference != 0:
        raise AssertionError(f"R1 future perturbation changed predictions: {difference}")
    return difference


def _anchor_recipe_check(pipe, history: pd.DataFrame,
                         anchors: pd.DataFrame) -> dict:
    """Check current weights/runtime against old R1 forecasts before extension."""
    import torch

    horizons = tuple(sorted(set(anchors.horizon.astype(int))))
    grouped = anchors.groupby("origin", sort=True).horizon.apply(
        lambda values: tuple(sorted(set(values.astype(int)))))
    complete = pd.DatetimeIndex(grouped.loc[grouped.map(lambda found: found == horizons)].index)
    if complete.empty:
        raise ValueError("No R1 anchor origin covers all selected horizons")
    positions = sorted({0, len(complete) // 2, len(complete) - 1})
    probes = [pd.Timestamp(complete[position]) for position in positions]
    inputs = [causal_input(history, origin, CONTEXT_LENGTH,
                           PREDICTION_LENGTH, False) for origin in probes]
    with torch.inference_mode():
        prediction, _ = pipe.predict_quantiles(
            inputs, prediction_length=PREDICTION_LENGTH,
            quantile_levels=QUANTILES, context_length=CONTEXT_LENGTH,
            batch_size=BATCH_SIZE)
    arrays = _quantile_array(prediction, len(probes))
    lookup = anchors.set_index(["origin", "horizon"])
    qindex = [QUANTILES.index(level) for level in (.1, .5, .9, .95)]
    max_abs = 0.0
    compared = 0
    for origin, array in zip(probes, arrays):
        for horizon in horizons:
            actual = array[0, horizon - 1, qindex].astype("float64")
            prior = lookup.loc[(origin, horizon), list(FORECAST_COLUMNS)].to_numpy(dtype="float64")
            difference = float(np.max(np.abs(actual - prior)))
            max_abs = max(max_abs, difference)
            compared += len(FORECAST_COLUMNS)
    if max_abs > ANCHOR_RECIPE_ATOL:
        raise RuntimeError(f"Current Chronos pipeline disagrees with old R1 anchor: "
                           f"max_abs={max_abs:.9g}, atol={ANCHOR_RECIPE_ATOL}")
    return {"status": "passed", "origin_count": len(probes),
            "origins": [origin.isoformat() for origin in probes],
            "horizons": list(horizons), "compared_quantiles": compared,
            "max_abs_difference": max_abs, "absolute_tolerance": ANCHOR_RECIPE_ATOL,
            "relative_tolerance": 0.0}


def _infer_chunk(pipe, history: pd.DataFrame, origins: list[pd.Timestamp],
                 required_by_origin: dict, device: str) -> pd.DataFrame:
    import torch

    rows = []
    for offset in range(0, len(origins), BATCH_SIZE):
        batch = origins[offset:offset + BATCH_SIZE]
        inputs = [causal_input(history, origin, CONTEXT_LENGTH, PREDICTION_LENGTH, False)
                  for origin in batch]
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.inference_mode():
            prediction, _ = pipe.predict_quantiles(
                inputs, prediction_length=PREDICTION_LENGTH,
                quantile_levels=QUANTILES, context_length=CONTEXT_LENGTH,
                batch_size=BATCH_SIZE)
        if device == "cuda":
            torch.cuda.synchronize()
        arrays = _quantile_array(prediction, len(batch))
        for origin, array in zip(batch, arrays):
            for h in required_by_origin[origin]:
                values = array[0, h - 1]
                rows.append((origin, h, float(values[QUANTILES.index(.5)]),
                             float(values[QUANTILES.index(.1)]),
                             float(values[QUANTILES.index(.5)]),
                             float(values[QUANTILES.index(.9)]),
                             float(values[QUANTILES.index(.95)])))
    return _sort_paths(pd.DataFrame(rows, columns=PATH_COLUMNS))


def load_or_build(prepared, out, *, anchor: pd.DataFrame, device=None):
    """Return required (origin,horizon) R1 paths and an immutable-input audit.

    ``out/r1_paths`` holds resumable inferred chunks. Existing EXPLORE CAL and
    score predictions are prefills; fit/stop origins are zero-shot inference.
    No model is trained. The returned table excludes labels by construction.
    """
    import torch

    required = _required(prepared)
    anchors = _anchor_paths(prepared, anchor, required)
    snapshot, model = _snapshot(prepared)
    anchor_source = _verify_anchor_source(prepared, anchors, model["revision"])
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device not in ("cuda", "cpu"):
        raise ValueError("R1 device must be cpu or cuda")
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    execution = _execution_fingerprint(resolved_device)
    identity = _identity(prepared, required, anchors, model, anchor_source, execution)
    identity_hash = config_hash(identity)
    cache = Path(out) / "r1_paths"
    cache.mkdir(parents=True, exist_ok=True)
    for orphan in (*cache.glob("part_*.parquet.tmp"), *cache.glob("manifest.json.tmp")):
        _quarantine(orphan, "orphan")
    manifest = cache / "manifest.json"
    if manifest.exists():
        try:
            state = json.loads(manifest.read_text(encoding="utf-8"))
            expected_hash = state.pop("manifest_sha256")
            if config_hash(state) != expected_hash:
                raise ValueError("Manifest checksum mismatch")
            state["manifest_sha256"] = expected_hash
        except (ValueError, KeyError, json.JSONDecodeError):
            _quarantine(manifest, "corrupt")
            state = None
        if state is not None and state.get("identity_hash") != identity_hash:
            raise RuntimeError("R1 path cache identity changed; use a new output directory")
    else:
        state = None
    if state is None:
        state = {"identity": identity, "identity_hash": identity_hash, "parts": [],
                 "complete": False, "future_perturbation_max_abs_difference": None,
                 "anchor_recipe_check": None,
                 "holdout_read": False, "historical_final_artifact_read": False}
        _write_manifest(manifest, state)
    inferred, changed = _read_chunks(cache, state)
    if changed:
        _write_manifest(manifest, state)
    anchor_keys = pd.MultiIndex.from_frame(anchors.loc[:, ["origin", "horizon"]])
    inferred_keys = pd.MultiIndex.from_frame(inferred.loc[:, ["origin", "horizon"]])
    if inferred_keys.isin(anchor_keys).any():
        left = inferred.set_index(["origin", "horizon"])
        right = anchors.set_index(["origin", "horizon"])
        overlap = left.index.intersection(right.index)
        if not np.array_equal(left.loc[overlap, PATH_COLUMNS[2:]].to_numpy(),
                              right.loc[overlap, PATH_COLUMNS[2:]].to_numpy()):
            raise RuntimeError("Inferred R1 paths disagree with anchored predictions")
        inferred = inferred.loc[~inferred_keys.isin(anchor_keys)].reset_index(drop=True)
        inferred_keys = pd.MultiIndex.from_frame(inferred.loc[:, ["origin", "horizon"]])
    all_keys = pd.MultiIndex.from_frame(required)
    done_keys = anchor_keys.append(inferred_keys)
    if not done_keys.isin(all_keys).all():
        raise RuntimeError("R1 cache includes keys outside required selected roles")
    missing = required.loc[~all_keys.isin(done_keys)]
    required_by_origin = {origin: tuple(group.horizon.astype(int))
                          for origin, group in missing.groupby("origin", sort=True)}
    todo = list(required_by_origin)
    if todo:
        pipe = _load_pipeline(snapshot, resolved_device)
        if state.get("anchor_recipe_check") is None:
            state["anchor_recipe_check"] = _anchor_recipe_check(
                pipe, prepared.history, anchors)
            _write_manifest(manifest, state)
        if state["future_perturbation_max_abs_difference"] is None:
            state["future_perturbation_max_abs_difference"] = _future_probe(
                pipe, prepared.history, todo[0])
            _write_manifest(manifest, state)
        for offset in range(0, len(todo), CHUNK_ORIGINS):
            chunk = _infer_chunk(pipe, prepared.history, todo[offset:offset + CHUNK_ORIGINS],
                                 required_by_origin, resolved_device)
            if chunk.empty:
                raise RuntimeError("Empty R1 inference chunk")
            index = max((int(entry["file"][5:11]) for entry in state["parts"]),
                        default=-1) + 1
            name = f"part_{index:06d}.parquet"
            path = cache / name
            if path.exists():
                raise RuntimeError("R1 chunk path unexpectedly exists")
            temp = path.with_suffix(".parquet.tmp")
            chunk.to_parquet(temp, index=False)
            os.replace(temp, path)
            state["parts"].append({"file": name, "rows": len(chunk),
                                   "sha256": sha256(path),
                                   "content_sha256": _hash_frame(chunk)})
            _write_manifest(manifest, state)
            inferred = (chunk if inferred.empty else pd.concat([inferred, chunk], ignore_index=True))
            print(f"R1 causal paths: {min(offset + CHUNK_ORIGINS, len(todo))}/{len(todo)} "
                  f"missing origins ({len(state['parts'])} checkpoint chunks)", flush=True)
        del pipe
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
    paths = _sort_paths(pd.concat([anchors, inferred], ignore_index=True))
    if paths.duplicated(["origin", "horizon"]).any():
        raise RuntimeError("R1 path result has duplicate keys")
    if not pd.MultiIndex.from_frame(paths.loc[:, ["origin", "horizon"]]).equals(all_keys):
        raise RuntimeError("R1 path result lacks required role coverage")
    path_sha = _hash_frame(paths)
    if state.get("complete") and state.get("paths_sha256") != path_sha:
        raise RuntimeError("Completed R1 paths differ from prior committed cache")
    state["complete"] = True
    state["path_rows"] = len(paths)
    state["paths_sha256"] = path_sha
    _write_manifest(manifest, state)
    audit = {"identity_hash": identity_hash, "manifest": str(manifest),
             "path_rows": len(paths), "anchor_rows": len(anchors),
             "inferred_rows": len(inferred),
             "inferred_origins": int(inferred.origin.nunique()),
             "future_perturbation_max_abs_difference": state["future_perturbation_max_abs_difference"],
             "anchor_recipe_check": state["anchor_recipe_check"],
             "leakage_test": ("passed" if state["future_perturbation_max_abs_difference"] == 0.0
                              and anchor_source["anchor_original_future_perturbation_max_abs_difference"] == 0.0
                              and state["anchor_recipe_check"] is not None
                              and state["anchor_recipe_check"].get("status") == "passed"
                              else "unverified"),
             "input_cutoff_rule": "history.index <= origin",
             "anchor_prediction_sha256": anchor_source["anchor_prediction_sha256"],
             "anchor_source": anchor_source,
             "source_sha256": identity["source_sha256"],
             "execution": execution,
             "cache_identity_hash": identity_hash,
             "holdout_read": False, "historical_final_artifact_read": False,
             "model_revision": model["revision"], "local_training": False}
    paths.attrs["causal_provenance"] = audit
    return paths, audit
