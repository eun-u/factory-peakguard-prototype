"""FG-R5 stateless recent-day analogs from quality-masked observed power.

Reference cases are derived afresh from raw history at each query origin.
No FIT bank or evaluation-period model state is built. STOP alone selects a
weekly mixture with the frozen R1 path; inference never receives context.y.
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
from phase_f.wf_harness import FIRST_SCORE_WEEK


_SLOT = pd.Timedelta(minutes=15)
_DAY = pd.Timedelta(days=1)
_ID = re.compile(r"[A-Za-z0-9_.+-]+\Z")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PREFIX_LENGTHS = (16, 96)
TOP_K = (3, 5)
RECENT_DAYS = 56
CHUNK_ROWS = 64
_EPSILON = 1e-6
_SOURCES = (
    "phase_f/models/recent_day_analog.py", "phase_f/models/r1_residual.py",
    "phase_f/features_ext.py", "phase_f/harness.py", "phase_f/registry.py",
    "phase_f/wf_harness.py",
    "phase_c/data.py", "src/session_data.py",
)


def _runtime() -> dict:
    return {"python": sys.version, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__}


def _update_digest(digest, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = np.where(np.isfinite(array), array, np.nan).astype("<f8")
    else:
        array = array.astype("<i8")
    array = np.ascontiguousarray(array)
    digest.update(name.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _take_exact(power: pd.Series, timestamps_ns: np.ndarray) -> np.ndarray:
    """Resolve timestamps against the index; missing grid entries remain NaN."""
    positions = power.index.get_indexer(
        pd.DatetimeIndex(timestamps_ns.ravel())).reshape(timestamps_ns.shape)
    result = np.full(timestamps_ns.shape, np.nan, dtype=np.float64)
    valid = positions >= 0
    result[valid] = power.to_numpy(dtype=np.float64)[positions[valid]]
    return result


def _check_origins(origins: pd.DatetimeIndex) -> pd.DatetimeIndex:
    origins = pd.DatetimeIndex(origins)
    if (origins.has_duplicates or not origins.is_monotonic_increasing
            or origins.tz is not None
            or not ((origins.minute % 15 == 0) & (origins.second == 0)
                    & (origins.microsecond == 0) & (origins.nanosecond == 0)).all()):
        raise ValueError("Recent analog requires unique chronological quarter-hour origins")
    return origins


def _input_arrays(power: pd.Series, origins: pd.DatetimeIndex,
                  length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
    """At most 64 x 56 x 96 values per call; no full-cohort tensor."""
    origin_ns = origins.asi8
    lags = np.arange(length - 1, -1, -1, dtype=np.int64) * int(_SLOT.value)
    ref_ns = origin_ns[:, None] - np.arange(1, RECENT_DAYS + 1,
                                           dtype=np.int64)[None, :] * int(_DAY.value)
    query_times = origin_ns[:, None] - lags[None, :]
    ref_prefix_times = ref_ns[:, :, None] - lags[None, None, :]
    suffix_times = ref_ns[:, :, None] + np.asarray(HORIZONS,
        dtype=np.int64)[None, None, :] * int(_SLOT.value)
    if ((query_times > origin_ns[:, None]).any()
            or (ref_prefix_times > origin_ns[:, None, None]).any()
            or (suffix_times > origin_ns[:, None, None]).any()
            or (ref_ns >= origin_ns[:, None]).any()):
        raise AssertionError("Recent analog attempted to read after its origin")
    return (ref_ns, query_times, ref_prefix_times, suffix_times,
            _take_exact(power, query_times),
            _take_exact(power, ref_prefix_times))


def _retrieve(power: pd.Series, origins: pd.DatetimeIndex, length: int,
              top_k: int, *, score_start: pd.Timestamp | None = None) -> dict:
    """Inference uses raw power only; elapsed-read counts exclude current t.

    Current origin power is used for shape and level shift, and disclosed
    separately. Its observation is already available at the interval end.
    """
    origins = _check_origins(origins)
    if length not in PREFIX_LENGTHS or top_k not in TOP_K:
        raise ValueError("Recent analog recipe changed")
    if not isinstance(power.index, pd.DatetimeIndex) or power.index.has_duplicates:
        raise ValueError("Recent analog power requires unique timestamps")
    n = len(origins)
    points = np.full((n, len(HORIZONS)), np.nan, dtype=np.float64)
    support = np.zeros(n, dtype=np.int16)
    selected_refs = np.full((n, top_k), -1, dtype=np.int64)
    selected_dist = np.full((n, top_k), np.nan, dtype=np.float64)
    latest_source = np.full(n, -1, dtype=np.int64)
    elapsed_reads = np.zeros((n, len(HORIZONS)), dtype=np.int32)
    query_clean = np.zeros(n, dtype=bool)
    input_digest = hashlib.sha256()
    score_ns = None if score_start is None else pd.Timestamp(score_start).value
    for start in range(0, n, CHUNK_ROWS):
        end = min(n, start + CHUNK_ROWS)
        chosen = origins[start:end]
        (ref_ns, query_times, ref_prefix_times, suffix_times,
         query, ref_prefix) = _input_arrays(power, chosen, length)
        suffix = _take_exact(power, suffix_times)
        _update_digest(input_digest, "origin_ns", chosen.asi8)
        _update_digest(input_digest, "query_power", query)
        _update_digest(input_digest, "reference_prefix_power", ref_prefix)
        _update_digest(input_digest, "reference_suffix_power", suffix)
        clean_query = np.isfinite(query).all(axis=1)
        clean_refs = (np.isfinite(ref_prefix).all(axis=2)
                      & np.isfinite(suffix).all(axis=2))
        query_clean[start:end] = clean_query
        centered_query = query - query.mean(axis=1, keepdims=True)
        centered_refs = ref_prefix - ref_prefix.mean(axis=2, keepdims=True)
        distance = np.abs(centered_refs - centered_query[:, None, :]).mean(axis=2)
        distance[~clean_refs | ~clean_query[:, None]] = np.inf
        count = clean_refs.sum(axis=1).astype(np.int16)
        count[~clean_query] = 0
        support[start:end] = count
        order = np.argsort(distance, axis=1, kind="stable")[:, :top_k]
        row = np.arange(end - start)[:, None]
        eligible = count >= top_k
        selected_ref = ref_ns[row, order]
        chosen_dist = distance[row, order]
        selected_ref[~eligible] = -1
        chosen_dist[~eligible] = np.nan
        selected_refs[start:end] = selected_ref
        selected_dist[start:end] = chosen_dist
        if not eligible.any():
            continue
        selected_suffix = suffix[row, order, :]
        selected_level = ref_prefix[:, :, -1][row, order]
        shifted = np.maximum(0.0, selected_suffix
                             + query[:, -1, None, None] - selected_level[:, :, None])
        weights = 1.0 / np.maximum(chosen_dist, _EPSILON)
        # Keep invalid rows harmless during vectorized ordering, then mask them.
        weights[~eligible] = 0.0
        shifted[~eligible] = 0.0
        value_order = np.argsort(shifted, axis=1, kind="stable")
        values_sorted = np.take_along_axis(shifted, value_order, axis=1)
        weights_sorted = np.take_along_axis(weights[:, :, None]
                              * np.ones((1, 1, len(HORIZONS))), value_order, axis=1)
        cumulative = weights_sorted.cumsum(axis=1)
        at_half = (cumulative >= weights_sorted.sum(axis=1, keepdims=True) / 2).argmax(axis=1)
        median = np.take_along_axis(values_sorted, at_half[:, None, :], axis=1)[:, 0, :]
        median[~eligible] = np.nan
        points[start:end] = median
        latest_source[start:end][eligible] = chosen.asi8[eligible]
        if score_ns is not None:
            past_query = ((query_times >= score_ns)
                          & (query_times < chosen.asi8[:, None])).sum(axis=1)
            selected_prefix_times = ref_prefix_times[row, order, :]
            past_ref_prefix = (selected_prefix_times >= score_ns).sum(axis=(1, 2))
            selected_suffix_times = suffix_times[row, order, :]
            past_ref_suffix = (selected_suffix_times >= score_ns).sum(axis=1)
            reads = past_query[:, None] + past_ref_prefix[:, None] + past_ref_suffix
            reads[~eligible] = 0
            elapsed_reads[start:end] = reads
    if ((latest_source >= 0) & (latest_source > origins.asi8)).any():
        raise AssertionError("Recent analog latest observed source exceeds origin")
    return {"analog": points, "support": support,
            "fallback": ~np.isfinite(points).all(axis=1),
            "selected_refs_ns": selected_refs, "distances": selected_dist,
            "latest_source_ns": latest_source,
            "elapsed_eval_observation_reads": elapsed_reads,
            "query_prefix_clean": query_clean,
            "input_sha256": input_digest.hexdigest(),
            "max_chunk_rows": CHUNK_ROWS}


def _blend(r1: np.ndarray, analog: np.ndarray, alpha: float) -> np.ndarray:
    baseline = np.asarray(r1, dtype=np.float64)
    if (not np.isfinite(baseline).all() or type(alpha) not in (int, float)
            or alpha not in ALPHAS):
        raise ValueError("Recent analog requires finite frozen R1 and fixed alpha")
    result = baseline.copy()
    valid = np.isfinite(analog)
    result[valid] = np.maximum(0.0, (1 - alpha) * baseline[valid] + alpha * analog[valid])
    return result


def _choose_alpha(truth: np.ndarray, r1: np.ndarray, analog: np.ndarray,
                  peak: np.ndarray) -> tuple[float, dict]:
    if (not len(truth) or not np.isfinite(truth).all()
            or not np.isfinite(r1).all() or not peak.any()):
        raise ValueError("STOP requires finite truth/R1 and observed peak cases")
    objectives = {}
    for alpha in ALPHAS:
        error = np.abs(truth - _blend(r1, analog, alpha))
        mae = float(error.mean())
        peak_mae = float(error[peak].mean())
        objectives[str(alpha)] = {"MAE": mae, "PeakMAE": peak_mae,
                                  "objective": mae + .25 * peak_mae}
    winner = min(ALPHAS, key=lambda value: (objectives[str(value)]["objective"], value))
    return winner, objectives


def _stop_selection(view, spec: dict, paths: pd.DataFrame, fold: int,
                    stop: pd.DatetimeIndex, retrieved: dict) -> dict:
    truth, r1, analog, peak = [], [], [], []
    for i, h in enumerate(HORIZONS):
        context = view.contexts[(h, fold)]
        truth.append(context["y"].loc[stop].to_numpy(dtype=np.float64))
        r1.append(_path_values(paths, stop, h).r1.to_numpy(dtype=np.float64))
        analog.append(retrieved["analog"][:, i])
        peak.append(truth[-1] > float(context["tau"]))
    alpha, objectives = _choose_alpha(np.concatenate(truth), np.concatenate(r1),
                                      np.concatenate(analog), np.concatenate(peak))
    return {"alpha": float(alpha), "objectives": objectives,
            "peak_count": int(np.concatenate(peak).sum()),
            "query_count": len(stop),
            "analog_coverage": float((~retrieved["fallback"]).mean()),
            "fallback_count": int(retrieved["fallback"].sum()),
            "stop_input_sha256": retrieved["input_sha256"]}


def _identity(view, spec: dict, paths_sha: str, fold: int,
              fit: pd.DatetimeIndex, stop: pd.DatetimeIndex,
              stop_input_sha: str) -> str:
    root = Path(view.root)
    roles = {f"h{h}_{role}": _digest_frame(pd.DataFrame({
        "origin": view.contexts[(h, fold)][role]}))
        for h in HORIZONS for role in ROLES}
    targets = {}
    for h in HORIZONS:
        context = view.contexts[(h, fold)]
        for role, origins in (("fit", fit), ("stop", stop)):
            targets[f"h{h}_{role}"] = _digest_frame(pd.DataFrame({
                "y": context["y"].loc[origins].to_numpy(dtype=np.float64),
                "target_time": pd.DatetimeIndex(context["target_time"].loc[origins])}))
    contract_files = (
        "outputs/phase_f/goal_r1_recent_analog_v1/RECENT_ANALOG_PLAN.md",
        "outputs/phase_f/goal_r1_recent_analog_v1/CAUSAL_INPUT_CONTRACT.json",
    )
    return config_hash({"spec": spec, "normalized_paths_sha256": paths_sha,
                        "fold": fold, "roles": roles,
                        "pooled_fit": fit.asi8.tolist(),
                        "pooled_stop": stop.asi8.tolist(),
                        "fit_stop_targets": targets,
                        "taus": {str(h): float(view.contexts[(h, fold)]["tau"])
                                 for h in HORIZONS},
                        "stop_historical_input_sha256": stop_input_sha,
                        "split_sha256": view.split_lock["lock_sha256"],
                        "raw_sha256": view.seal["raw_sha256"],
                        "runtime": _runtime(),
                        "sources": {name: sha256(root / name) for name in _SOURCES},
                        "scientific_contracts": {name: sha256(root / name)
                                                 for name in contract_files}})


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
    metadata_sha = sha256(meta_path) if meta_path.is_file() else None
    model_saved = _preserve(model_path)
    metadata_saved = _preserve(meta_path)
    if ((model_saved and sha256(model_saved) != model_sha)
            or (metadata_saved and sha256(metadata_saved) != metadata_sha)):
        raise RuntimeError("Recent analog evidence changed during preservation")
    incident = Path(view.out) / "logs/checkpoint_corruption" / \
        f"{model_path.stem}-{config_hash([identity, model_sha, metadata_sha, reason])[:20]}.json"
    write_json(incident, {"reason": reason, "expected_identity": identity,
                          "model_original": str(model_path),
                          "model_sha256": model_sha,
                          "model_preserved": str(model_saved) if model_saved else None,
                          "metadata_original": str(meta_path),
                          "metadata_sha256": metadata_sha,
                          "metadata_preserved": str(metadata_saved) if metadata_saved else None,
                          "refit_performed": False}, exclusive=True)
    raise RuntimeError(f"Recent analog checkpoint rejected ({reason}); preserved physical evidence: {incident}")


def _checkpoint_paths(view, spec: dict, fold: int) -> tuple[Path, Path]:
    path = Path(view.out) / "models/recent_day_analog" / spec["id"] / f"f{fold}.joblib"
    return path, path.with_suffix(".json")


def _read_checkpoint(view, model_path: Path, meta_path: Path,
                     identity: str, spec: dict, fold: int,
                     selection: dict) -> tuple[dict, dict]:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _reject(view, model_path, meta_path, identity, "metadata_json_corrupt")
    if not isinstance(metadata, dict) or metadata.get("identity") != identity:
        _reject(view, model_path, meta_path, identity, "identity_changed")
    if (metadata.get("spec") != spec or metadata.get("fold") != fold
            or metadata.get("n_stochastic_seeds") != 0
            or metadata.get("n_deterministic_runs") != 1
            or metadata.get("selection") != selection
            or metadata.get("alpha") != selection["alpha"]
            or metadata.get("stop_input_sha256") != selection["stop_input_sha256"]
            or metadata.get("prediction_state_updates") != "none"
            or metadata.get("historical_input_mode") != "observed_raw_power_through_origin"
            or metadata.get("historical_final_artifact_read") is not False
            or metadata.get("holdout_read") is not False
            or metadata.get("model_sha256") != sha256(model_path)):
        _reject(view, model_path, meta_path, identity, "metadata_or_stop_selection_changed")
    # Only private, locally generated checkpoints are deserialized, after
    # physical-byte verification and an independently recomputed identity.
    try:
        bundle = joblib.load(model_path)
    except Exception:
        _reject(view, model_path, meta_path, identity, "payload_unreadable")
    if (not isinstance(bundle, dict)
            or set(bundle) != {"spec", "alpha", "stop_input_sha256"}
            or bundle["spec"] != spec
            or type(bundle["alpha"]) not in (int, float)
            or bundle["alpha"] not in ALPHAS
            or bundle["alpha"] != selection["alpha"]
            or bundle["stop_input_sha256"] != selection["stop_input_sha256"]):
        _reject(view, model_path, meta_path, identity, "payload_or_alpha_changed")
    return bundle, metadata


def _fit_or_load(view, spec: dict, paths_sha: str, fold: int,
                 fit: pd.DatetimeIndex, stop: pd.DatetimeIndex,
                 selection: dict, stop_selection_seconds: float) -> tuple[dict, dict]:
    start = perf_counter()
    identity = _identity(view, spec, paths_sha, fold, fit, stop,
                         selection["stop_input_sha256"])
    model_path, meta_path = _checkpoint_paths(view, spec, fold)
    if model_path.exists() != meta_path.exists():
        _reject(view, model_path, meta_path, identity, "checkpoint_pair_incomplete")
    if model_path.exists():
        bundle, metadata = _read_checkpoint(view, model_path, meta_path,
                                             identity, spec, fold, selection)
        return bundle, {**metadata, "cache_reused": True}
    bundle = {"spec": dict(spec), "alpha": selection["alpha"],
              "stop_input_sha256": selection["stop_input_sha256"]}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp = model_path.with_suffix(".joblib.tmp")
    if temp.exists():
        _reject(view, temp, temp.with_suffix(".json"), identity,
                "stale_temporary_checkpoint")
    joblib.dump(bundle, temp)
    os.replace(temp, model_path)
    metadata = {"identity": identity, "model_sha256": sha256(model_path),
                "spec": dict(spec), "fold": fold, "selection": selection,
                "alpha": selection["alpha"],
                "stop_input_sha256": selection["stop_input_sha256"],
                "n_stochastic_seeds": 0, "n_deterministic_runs": 1,
                "prediction_state_updates": "none",
                "historical_input_mode": "observed_raw_power_through_origin",
                "historical_final_artifact_read": False,
                "holdout_read": False,
                "train_seconds": stop_selection_seconds + perf_counter() - start,
                "cache_reused": False}
    write_json(meta_path, metadata)
    return bundle, metadata


def _validate_spec(prepared, spec: dict, rolling_paths: pd.DataFrame) -> tuple[pd.DataFrame, str, str, list[int]]:
    length, top_k = spec.get("prefix_length"), spec.get("top_k")
    if (type(length) is not int or length not in PREFIX_LENGTHS
            or type(top_k) is not int or top_k not in TOP_K
            or type(spec.get("bank_days")) is not int or spec["bank_days"] != RECENT_DAYS
            or spec.get("id") != f"FG-R5-recent-w{length}-k{top_k}"
            or spec.get("adapter") != "recent_day_analog"
            or spec.get("arm") != "EXPLORE" or spec.get("seed") is not None
            or not _ID.fullmatch(str(spec.get("id", "")))):
        raise ValueError("Recent analog requires one of four fixed deterministic EXPLORE specs")
    if not prepared.split_lock.get("lock_sha256") or not prepared.seal.get("raw_sha256"):
        raise ValueError("Recent analog requires sealed raw source and weekly split")
    if {h for h, _ in prepared.contexts} != set(HORIZONS):
        raise ValueError("Recent analog requires all thirteen horizons")
    folds = sorted({fold for _, fold in prepared.contexts})
    if not folds or any({h for h, f in prepared.contexts if f == fold} != set(HORIZONS)
                        for fold in folds):
        raise ValueError("Recent analog requires complete weekly horizon grids")
    if any(prepared.contexts[(4, fold)].get("summary", {}).get("arm") not in (None, "EXPLORE")
           for fold in folds):
        raise ValueError("Recent analog requires EXPLORE weekly context")
    paths, paths_sha = _paths_index(rolling_paths, prepared.contexts)
    proof = rolling_paths.attrs.get("causal_provenance", {})
    upstream = spec.get("rolling_audit_sha256")
    if (not isinstance(upstream, str) or not re.fullmatch(r"[0-9a-f]{64}", upstream)
            or not isinstance(proof, dict) or config_hash(proof) != upstream
            or proof.get("leakage_test") != "passed"
            or proof.get("input_cutoff_rule") != "history.index <= origin"
            or proof.get("future_perturbation_max_abs_difference") != 0.0):
        raise ValueError("Recent analog requires signed causal R1 path provenance")
    return paths, paths_sha, upstream, folds


def _future_probe(view, spec: dict, paths: pd.DataFrame,
                  fold: int, alpha: float) -> dict:
    """Probe all 13 unblended outputs at each horizon's first SCORE origin."""
    touched_columns = ["power", *[flag for flag in ("quality_bad", "time_repaired")
                                   if flag in view.history]]
    power = _power(view.history)
    maximum = 0.0
    n_origins = 0
    for h in HORIZONS:
        origin = pd.Timestamp(view.contexts[(h, fold)]["score"][0])
        chosen = pd.DatetimeIndex([origin])
        original = _retrieve(power, chosen, spec["prefix_length"],
                             spec["top_k"], score_start=FIRST_SCORE_WEEK)
        altered_history = view.history.copy()
        future = altered_history.index > origin
        if not future.any():
            raise ValueError("Recent analog future probe has no later observed rows")
        altered_history.loc[future, "power"] = 10000.0
        for flag in touched_columns[1:]:
            altered_history.loc[future, flag] = \
                ~altered_history.loc[future, flag].fillna(True).astype(bool)
        changed = _retrieve(_power(altered_history), chosen,
                            spec["prefix_length"], spec["top_k"],
                            score_start=FIRST_SCORE_WEEK)
        if (not np.array_equal(original["analog"], changed["analog"], equal_nan=True)
                or not np.array_equal(original["selected_refs_ns"],
                                      changed["selected_refs_ns"])
                or not np.array_equal(original["distances"],
                                      changed["distances"], equal_nan=True)
                or not np.array_equal(original["support"], changed["support"])
                or not np.array_equal(original["latest_source_ns"],
                                      changed["latest_source_ns"])):
            raise AssertionError("Future raw observations changed unblended analog or rank")
        r1 = _path_values(paths, chosen, h).r1.to_numpy(dtype=np.float64)
        before = _blend(r1, original["analog"][:, h - min(HORIZONS)], alpha)
        after = _blend(r1, changed["analog"][:, h - min(HORIZONS)], alpha)
        difference = float(np.max(np.abs(before - after)))
        if difference != 0:
            raise AssertionError("Future raw observations changed fitted analog blend")
        maximum = max(maximum, difference)
        n_origins += 1
    return {"future_perturbation_max_abs_difference": maximum,
            "future_perturbation_unblended_all13_equal": True,
            "future_perturbation_reference_rank_equal": True,
            "future_perturbation_probe_origins": n_origins,
            "future_perturbation_touched_columns": touched_columns}


def _datetime_or_nat(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.where(values >= 0, values, pd.NaT.value))


def run(prepared, spec: dict, rolling_paths: pd.DataFrame,
        *, smoke_first_fold: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fit/reuse STOP alpha; infer CAL/SCORE using each origin's raw past."""
    paths, paths_sha, upstream, folds = _validate_spec(prepared, spec, rolling_paths)
    selected_folds = folds[:1] if smoke_first_fold else folds
    frames, cells = [], []
    for fold in selected_folds:
        fold_start = perf_counter()
        fit, stop, roles = _common_roles(prepared.contexts, fold)
        power = _power(prepared.history)
        stop_retrieval_start = perf_counter()
        stop_retrieved = _retrieve(power, stop, spec["prefix_length"],
                                   spec["top_k"], score_start=FIRST_SCORE_WEEK)
        selection = _stop_selection(prepared, spec, paths, fold, stop, stop_retrieved)
        stop_selection_seconds = perf_counter() - stop_retrieval_start
        bundle, metadata = _fit_or_load(prepared, spec, paths_sha, fold,
                                         fit, stop, selection,
                                         stop_selection_seconds)
        model_path, meta_path = _checkpoint_paths(prepared, spec, fold)
        metadata["checkpoint_metadata_sha256"] = sha256(meta_path)
        probe = _future_probe(prepared, spec, paths, fold, float(bundle["alpha"]))
        predict_start = perf_counter()
        coverage = []
        for role in ("cal", "score"):
            all_origins = pd.DatetimeIndex(sorted(set.union(*(
                set(prepared.contexts[(h, fold)][role]) for h in HORIZONS))))
            retrieved = _retrieve(power, all_origins, spec["prefix_length"],
                                  spec["top_k"], score_start=FIRST_SCORE_WEEK)
            locations = pd.Series(np.arange(len(all_origins)), index=all_origins)
            for i, h in enumerate(HORIZONS):
                context = prepared.contexts[(h, fold)]
                origins = pd.DatetimeIndex(context[role])
                selected = locations.loc[origins].to_numpy(dtype=np.int64)
                analog = retrieved["analog"][selected, i]
                support = retrieved["support"][selected]
                fallback = retrieved["fallback"][selected]
                source_ns = retrieved["latest_source_ns"][selected]
                elapsed = retrieved["elapsed_eval_observation_reads"][selected, i]
                reference_ns = retrieved["selected_refs_ns"][selected]
                r1 = _path_values(paths, origins, h).r1.to_numpy(dtype=np.float64)
                forecast = _blend(r1, analog, float(bundle["alpha"]))
                aux = {"r1": r1, "recent_analog_point": analog,
                       "recent_analog_support": support,
                       "recent_analog_fallback": fallback,
                       "recent_analog_alpha": float(bundle["alpha"]),
                       "recent_analog_latest_observed": _datetime_or_nat(source_ns),
                       "recent_analog_elapsed_eval_reads": elapsed}
                for j in range(spec["top_k"]):
                    aux[f"recent_analog_ref{j + 1}"] = _datetime_or_nat(reference_ns[:, j])
                frames.append(make_frame(context, origins, h, fold, forecast,
                                         spec["id"], role, **aux))
                d2 = context["d2"].loc[origins].to_numpy(dtype=bool)
                later_count = int(((source_ns >= 0) & (source_ns > origins.asi8)).sum())
                if later_count:
                    raise AssertionError("Recent analog selected future input")
                coverage.append({"horizon": h, "role": role, "n": len(origins),
                                 "fallback_count": int(fallback.sum()),
                                 "mean_eligible_same_slot": float(support.mean()),
                                 "latest_source_exceeds_origin_count": later_count,
                                 "earlier_elapsed_evaluation_observation_reads": int(elapsed.sum()),
                                 "rows_using_earlier_elapsed_evaluation_observation": int((elapsed > 0).sum()),
                                 "d2_n": int(d2.sum()),
                                 "d2_fallback_count": int((fallback & d2).sum())})
        cells.append({**roles, **metadata, **probe, "fold": fold,
                      "checkpoint_model_sha256": sha256(model_path),
                      "stop_selection_seconds": stop_selection_seconds,
                      "prediction_seconds": perf_counter() - predict_start,
                      "fold_wall_seconds": perf_counter() - fold_start,
                      "coverage": coverage,
                      "historical_input_disclosure": {
                          "score_observation_threshold": str(FIRST_SCORE_WEEK),
                          "current_origin_power_used_when_analog_available": True,
                          "earlier_elapsed_reads_exclude_current_origin": True,
                          "input_source": "quality_masked_observed_raw_power",
                          "prediction_state_updates": "none"}})
        if (Path(prepared.out) / "logs/stop_requested.json").exists():
            raise RuntimeError(f"Recent analog stop requested after fold {fold}")
    frame = pd.concat(frames, ignore_index=True)
    expected = sum(len(prepared.contexts[(h, fold)][role])
                   for fold in selected_folds for h in HORIZONS
                   for role in ("cal", "score"))
    if len(frame) != expected or frame.duplicated(["horizon", "fold", "role", "origin"]).any():
        raise ValueError("Recent analog changed the frozen CAL/SCORE cohort")
    if not np.isfinite(frame.pred.to_numpy(dtype=np.float64)).all():
        raise ValueError("Recent analog emitted nonfinite point forecast")
    return frame, {"adapter": "recent_day_analog",
                   "paths_sha256": paths_sha,
                   "rolling_audit_sha256": upstream,
                   "rolling_upstream_causal_manifest_verified": True,
                   "n_stochastic_seeds": 0, "n_deterministic_runs": 1,
                   "seed": None, "cells": cells, "n_fold_models": len(cells),
                   "smoke_first_fold": smoke_first_fold,
                   "first_score_week_for_observed_input_disclosure": str(FIRST_SCORE_WEEK),
                   "future_perturbation_max_abs_difference": 0.0,
                   "leakage_test": "passed", "development_only": True,
                   "holdout_read": False,
                   "historical_final_artifact_read": False}


def verify_cells(prepared, spec: dict, rolling_paths: pd.DataFrame,
                 audit: dict, *, smoke_first_fold: bool = False) -> None:
    """Reconstruct STOP-only selection and validate every physical model."""
    paths, paths_sha, upstream, folds = _validate_spec(prepared, spec, rolling_paths)
    selected_folds = folds[:1] if smoke_first_fold else folds
    if (audit.get("adapter") != "recent_day_analog"
            or audit.get("paths_sha256") != paths_sha
            or audit.get("rolling_audit_sha256") != upstream
            or audit.get("n_fold_models") != len(selected_folds)
            or audit.get("n_stochastic_seeds") != 0
            or audit.get("n_deterministic_runs") != 1
            or audit.get("first_score_week_for_observed_input_disclosure") != str(FIRST_SCORE_WEEK)
            or audit.get("future_perturbation_max_abs_difference") != 0.0
            or audit.get("holdout_read") is not False
            or audit.get("historical_final_artifact_read") is not False
            or len(audit.get("cells", [])) != len(selected_folds)):
        raise ValueError("Recent analog audit differs from deterministic design")
    power = _power(prepared.history)
    for fold, recorded in zip(selected_folds, audit["cells"]):
        fit, stop, _ = _common_roles(prepared.contexts, fold)
        retrieved = _retrieve(power, stop, spec["prefix_length"],
                              spec["top_k"], score_start=FIRST_SCORE_WEEK)
        selection = _stop_selection(prepared, spec, paths, fold, stop, retrieved)
        identity = _identity(prepared, spec, paths_sha, fold, fit, stop,
                             selection["stop_input_sha256"])
        model_path, meta_path = _checkpoint_paths(prepared, spec, fold)
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Recent analog checkpoint missing for fold {fold}")
        bundle, metadata = _read_checkpoint(prepared, model_path, meta_path,
                                            identity, spec, fold, selection)
        if (recorded.get("fold") != fold
                or recorded.get("identity") != identity
                or recorded.get("selection") != selection
                or recorded.get("stop_input_sha256") != selection["stop_input_sha256"]
                or recorded.get("model_sha256") != sha256(model_path)
                or recorded.get("checkpoint_model_sha256") != sha256(model_path)
                or recorded.get("checkpoint_metadata_sha256") != sha256(meta_path)
                or recorded.get("alpha") != bundle["alpha"]
                or recorded.get("future_perturbation_max_abs_difference") != 0.0
                or recorded.get("future_perturbation_unblended_all13_equal") is not True
                or recorded.get("future_perturbation_reference_rank_equal") is not True
                or recorded.get("future_perturbation_probe_origins") != len(HORIZONS)
                or recorded.get("future_perturbation_touched_columns") != [
                    "power", *[flag for flag in ("quality_bad", "time_repaired")
                               if flag in prepared.history]]
                or recorded.get("historical_input_disclosure") != {
                    "score_observation_threshold": str(FIRST_SCORE_WEEK),
                    "current_origin_power_used_when_analog_available": True,
                    "earlier_elapsed_reads_exclude_current_origin": True,
                    "input_source": "quality_masked_observed_raw_power",
                    "prediction_state_updates": "none"}):
            raise ValueError(f"Recent analog checkpoint/audit changed for fold {fold}")
