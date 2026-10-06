"""Phase F zero-shot foundation comparisons on the sealed development cohort.

Only observed power through each origin is sent to the model worker. Optional
packages are installed in Phase F's own directory and never modify the shared
Phase C/F interpreter. Public weights are resolved to one immutable HF commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from phase_f.harness import make_frame
from phase_f.registry import config_hash, sha256, write_json


LEVELS = np.arange(1, 10, dtype=float) / 10
PREDICTION_LENGTH = 16
MODELS = {
    "chronos_bolt": {
        "repo": "amazon/chronos-bolt-small", "package": "chronos-forecasting",
        "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
        "site": None, "commercial_license": "apache-2.0",
        "source": "https://huggingface.co/amazon/chronos-bolt-small",
    },
    "timesfm2p5": {
        "repo": "google/timesfm-2.5-200m-pytorch", "package": "timesfm",
        "revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        "site": "timesfm", "commercial_license": "apache-2.0",
        "source": "https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/api_reference.md",
    },
    "moirai2": {
        "repo": "Salesforce/moirai-2.0-R-small", "package": "uni2ts",
        "revision": "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
        "site": "moirai", "commercial_license": "cc-by-nc-4.0",
        "source": "https://github.com/SalesforceAIResearch/uni2ts",
    },
    "tirex": {
        "repo": "NX-AI/TiRex", "package": "tirex-ts",
        "revision": "63c740922493f5fbe60b277609ec62babfba2762",
        "site": "tirex", "commercial_license": "other: NXAI community license; review required",
        "source": "https://nx-ai.github.io/tirex/how-to/forecasting/practice",
    },
}


def configurations() -> list[dict]:
    """Initial cross-family sweep; later contexts get new experiment IDs."""
    return [
        {"id": f"F6-5-{kind}-c{length}", "family": "F6", "tier": 2,
         "adapter": "other_foundation", "kind": kind, "context_length": length,
         "point": "median", "prediction_length": PREDICTION_LENGTH}
        for kind in MODELS for length in (512, 2048)
    ]


def causal_context(history: pd.DataFrame, origin: pd.Timestamp, length: int) -> tuple[np.ndarray, int]:
    """Return a fixed-size input using only values available by *origin*.

    Missing interior observations use the last observed value. For an origin
    with less than `length` history, the earliest known value pads the left.
    """
    origin = pd.Timestamp(origin)
    past = history.power.loc[:origin].iloc[-length:].astype("float32")
    if past.empty or past.index[-1] != origin:
        raise ValueError("Origin absent from observed history")
    values = past.to_numpy(copy=True)
    missing = int(np.count_nonzero(~np.isfinite(values)))
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        raise ValueError("No observed power in causal context")
    # The first observed value is known by the forecast origin.
    values[:valid[0]] = values[valid[0]]
    values = pd.Series(values).ffill().to_numpy(dtype="float32")
    if len(values) < length:
        values = np.pad(values, (length - len(values), 0), constant_values=values[0])
    if values.shape != (length,) or not np.isfinite(values).all():
        raise AssertionError("Invalid causal context")
    return values, missing


def causal_matrix(history: pd.DataFrame, origins: pd.DatetimeIndex,
                  length: int) -> tuple[np.ndarray, int]:
    arrays, missing = [], 0
    for origin in origins:
        values, count = causal_context(history, origin, length)
        arrays.append(values)
        missing += count
    if not arrays:
        raise ValueError("No calibration/score origins")
    return np.stack(arrays), missing


def _snapshot(prepared, kind: str, spec: dict) -> tuple[Path, dict]:
    """Download official public weights only, at a recorded immutable commit."""
    definition = MODELS[kind]
    repo = definition["repo"]
    folder = prepared.out / "models" / "hf_optional"
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = prepared.out / "logs" / f"{spec['id']}_weights.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("repo") != repo or lock.get("spec_hash") != config_hash(spec):
            raise RuntimeError("Weight lock conflicts with experiment ID")
        revision = lock["revision"]
    else:
        revision = str(spec.get("model_revision") or definition["revision"])
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Model revision must be a full immutable SHA")
        lock = {"repo": repo, "revision": revision, "spec_hash": config_hash(spec),
                "license": definition["commercial_license"],
                "commercially_eligible": definition["commercial_license"] == "apache-2.0",
                "source": definition["source"], "training_on_local_data": False,
                "development_only": True, "holdout_read": False}
        write_json(lock_path, lock, exclusive=True)
    patterns = (["model.ckpt", "LICENSE", "README.md"] if kind == "tirex" else None)
    try:
        snapshot = Path(snapshot_download(repo_id=repo, revision=revision,
                                          cache_dir=str(folder), allow_patterns=patterns,
                                          local_files_only=True))
    except LocalEntryNotFoundError:
        snapshot = Path(snapshot_download(repo_id=repo, revision=revision,
                                          cache_dir=str(folder), allow_patterns=patterns))
    weights = [p for p in snapshot.iterdir() if p.suffix in (".safetensors", ".ckpt", ".bin")]
    if not weights or not (snapshot / "config.json").exists() and kind != "tirex":
        raise RuntimeError("Official model snapshot lacks required weights/config")
    hashes = {p.name: sha256(p) for p in sorted(weights + [snapshot / "config.json"]
              if (snapshot / "config.json").exists() else weights)}
    return snapshot, {**lock, "files_sha256": hashes,
                      "snapshot_hash": config_hash(hashes)}


def _optional_site(prepared, kind: str) -> Path | None:
    dirname = MODELS[kind]["site"]
    if dirname is None:
        return None
    site = prepared.out / "optional_envs" / dirname / "site"
    if not site.is_dir():
        raise RuntimeError(f"Optional dependency not installed in {site}")
    return site


def _quarantine_orphan(path: Path) -> Path:
    """Preserve an unpaired transaction file without replacing prior evidence."""
    digest = sha256(path)
    for number in range(10000):
        suffix = "" if number == 0 else f"-{number}"
        orphan = path.with_name(f"{path.stem}.orphan-{digest}{suffix}{path.suffix}")
        if not orphan.exists():
            path.rename(orphan)
            return orphan
    raise RuntimeError("Too many orphan files for input transaction")


def _write_request(path: Path, matrix: np.ndarray, origins: pd.DatetimeIndex,
                   probe: np.ndarray) -> str:
    """Preserve an input transaction; never silently replace an earlier run."""
    meta = path.with_suffix(".json")
    payload_hash = hashlib.sha256(matrix.tobytes() + origins.asi8.tobytes() + probe.tobytes()).hexdigest()
    if path.exists() != meta.exists():
        # A crash can leave either half of the transaction behind. Keep its
        # original bytes under a checksum name, then build the same causal
        # request afresh. A complete pair still has strict identity checks.
        unpaired = path if path.exists() else meta
        if not unpaired.is_file():
            raise RuntimeError("Incomplete input transaction is not a file")
        if unpaired == path:
            try:
                with np.load(path, allow_pickle=False) as data:
                    prior = (data["context"], data["origin_ns"], data["probe"])
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                prior = None  # Retain unreadable bytes, then reconstruct.
            if prior is not None and not (
                np.array_equal(prior[0], matrix) and
                np.array_equal(prior[1], origins.asi8) and
                np.array_equal(prior[2], probe)
            ):
                raise RuntimeError("Input transaction changed; use a new experiment ID")
        else:
            try:
                prior_meta = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                prior_meta = None
            if isinstance(prior_meta, dict) and prior_meta.get("payload_hash") not in (None, payload_hash):
                raise RuntimeError("Input transaction changed; use a new experiment ID")
        _quarantine_orphan(unpaired)
    if path.exists() or meta.exists():
        if not path.is_file() or not meta.is_file():
            raise RuntimeError("Incomplete input transaction")
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old.get("payload_hash") != payload_hash or old.get("sha256") != sha256(path):
            raise RuntimeError("Input transaction changed; use a new experiment ID")
        return old["sha256"]
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, context=matrix, origin_ns=origins.asi8, probe=probe)
    digest = sha256(temp)
    temp.replace(path)
    write_json(meta, {"sha256": digest, "payload_hash": payload_hash,
                      "origin_count": len(origins), "context_length": matrix.shape[1]}, exclusive=True)
    return digest


def _worker(prepared, spec: dict, snapshot: Path, request: Path, cache_id: str,
            optional_site: Path | None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    work = prepared.out / "models" / spec["id"] / "worker"
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(prepared.out / "models" / "hf_optional")
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if optional_site is not None:
        env["PYTHONPATH"] = str(optional_site) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "phase_f.models.other_foundation_worker",
           "--kind", spec["kind"], "--snapshot", str(snapshot),
           "--request", str(request), "--output", str(work),
           "--cache-id", cache_id, "--batch-size", str(spec.get("batch_size", 8))]
    if spec.get('actual_cuda_required'):cmd.append('--require-cuda')
    subprocess.run(cmd, cwd=prepared.root, env=env, check=True)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("cache_id") != cache_id:
        raise RuntimeError("Worker cache identity mismatch")
    if spec.get('actual_cuda_required') and not str(manifest.get('model_device','')).startswith('cuda'):
        raise RuntimeError('Required actual CUDA model placement was not recorded')
    chunks = []
    for row in manifest["chunks"]:
        file = work / row["file"]
        if not file.is_file() or sha256(file) != row["sha256"]:
            raise RuntimeError("Worker output checksum mismatch")
        with np.load(file, allow_pickle=False) as data:
            chunks.append((data["quantiles"], data["mean"], data["levels"]))
    quantiles = np.concatenate([x[0] for x in chunks], axis=0)
    means = np.concatenate([x[1] for x in chunks], axis=0)
    levels = chunks[0][2].tolist()
    if any(not np.array_equal(x[2], chunks[0][2]) for x in chunks):
        raise RuntimeError("Worker quantile levels changed between batches")
    return quantiles, means, levels


def _peak_probability(quantiles: np.ndarray, tau: float) -> np.ndarray:
    ordered = np.maximum.accumulate(quantiles, axis=1)
    # Native models return q10..q90. Tail probabilities are capped, not invented.
    return np.asarray([1 - np.interp(tau, row, LEVELS, left=.1, right=.9)
                       for row in ordered], dtype=float)


def run(prepared, spec: dict) -> tuple[pd.DataFrame, dict]:
    kind = spec["kind"]
    if kind not in MODELS or spec.get("point", "median") not in ("median", "mean"):
        raise ValueError("Unknown foundation model or point rule")
    length = int(spec["context_length"])
    if length <= 0 or int(spec.get("prediction_length", PREDICTION_LENGTH)) != PREDICTION_LENGTH:
        raise ValueError("Only fixed 16-step paths are comparable")
    origins = pd.DatetimeIndex(sorted(set().union(*[
        set(prepared.origins(h, f, role)) for h, f in prepared.contexts
        for role in ("cal", "score")])))
    matrix, n_missing = causal_matrix(prepared.history, origins, length)
    altered = prepared.history.copy()
    future = altered.index > origins[0]
    altered.loc[future, "power"] = np.random.default_rng(42).normal(10000, 1000, int(future.sum()))
    probe, _ = causal_context(altered, origins[0], length)
    if not np.array_equal(matrix[0], probe):
        raise AssertionError("Future perturbation changed causal model input")
    work = prepared.out / "models" / spec["id"]
    work.mkdir(parents=True, exist_ok=True)
    request = work / "causal_inputs.npz"
    input_hash = _write_request(request, matrix, origins, probe)
    snapshot, weight_lock = _snapshot(prepared, kind, spec)
    optional_site = _optional_site(prepared, kind)
    code_hashes = {"adapter": sha256(Path(__file__)),
                   "worker": sha256(Path(__file__).with_name("other_foundation_worker.py"))}
    cache_id = config_hash({"spec": spec, "weights": weight_lock["snapshot_hash"],
                            "revision": weight_lock["revision"], "input": input_hash,
                            "split": prepared.split_lock["lock_sha256"], "code": code_hashes,
                            "dependencies": sha256(prepared.out / "logs" / "other_foundation_dependencies.json")})
    tick = perf_counter()
    paths, means, levels = _worker(prepared, spec, snapshot, request, cache_id, optional_site)
    if levels != LEVELS.tolist() or paths.shape != (len(origins), PREDICTION_LENGTH, 9):
        raise RuntimeError("Unexpected native quantile levels or path shape")
    if means.shape != (len(origins), PREDICTION_LENGTH):
        raise RuntimeError("Unexpected mean path shape")
    if not np.isfinite(paths).all():
        raise RuntimeError("Foundation output has nonfinite quantiles")
    if spec.get("point", "median") == "mean" and not np.isfinite(means).all():
        raise RuntimeError("Native mean is unavailable for this candidate")
    index = {origin: i for i, origin in enumerate(origins)}
    manifest = json.loads((work / "worker" / "manifest.json").read_text(encoding="utf-8"))
    per_origin = np.empty(len(origins), dtype=float)
    for row in manifest["chunks"]:
        per_origin[row["start"]:row["stop"]] = row["forecast_seconds"] / (row["stop"] - row["start"])
    frames = []
    for (h, f), context in prepared.contexts.items():
        for role in ("cal", "score"):
            idx = prepared.origins(h, f, role)
            positions = np.asarray([index[o] for o in idx])
            q = paths[positions, h - 1, :]
            pred = q[:, 4] if spec.get("point", "median") == "median" else means[positions, h - 1]
            frames.append(make_frame(context, idx, h, f, pred, spec["id"], role,
                train_seconds=0., train_seconds_run_id=f"{cache_id}_zero_shot",
                inference_seconds=per_origin[positions],
                inference_seconds_run_id=[f"{cache_id}_{o.value}" for o in idx],
                q10=q[:, 0], q50=q[:, 4], q90=q[:, 8],
                p_peak=_peak_probability(q, float(context["tau"]))))
    audit = {"model": kind, "model_id": MODELS[kind]["repo"],
             "model_revision": weight_lock["revision"], "model_files_sha256": weight_lock["files_sha256"],
             "cache_id": cache_id, "input_sha256": input_hash,
             "causal_gap_values_filled": n_missing,
             "future_perturbation_max_abs_difference": manifest["future_perturbation_max_abs_difference"],
             "quantiles": levels, "q95": "unavailable from native model",
             "peak_probability": "linear interpolation q10..q90 with capped tails; uncalibrated",
             "inference_seconds": sum(row["forecast_seconds"] for row in manifest["chunks"]),
             "process_seconds": perf_counter() - tick,
             "optional_site": None if optional_site is None else optional_site.relative_to(prepared.root).as_posix(),
             "license": weight_lock["license"],
             "commercially_eligible": weight_lock["commercially_eligible"],
             "training_on_local_data": False, "leakage_test": "passed",
             "historical_final_artifact_read": False, "holdout_read": False}
    write_json(prepared.out / "logs" / f"{spec['id']}_other_foundation.json", audit)
    return pd.concat(frames, ignore_index=True), audit
