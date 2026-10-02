"""Fresh Chronos-2 reference paths; never an M* selection candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd


def cache_equivalence(root: Path, contexts: dict) -> dict:
    """Inspect only development cache keys, not historical prediction values."""
    import pyarrow.parquet as pq
    checks = []
    required_horizons = set(range(4, 17))
    for relative in ("outputs/p4/p4_chronos2_ref.parquet", "outputs/p4/p5_chronos2.parquet"):
        path = root / relative
        check = {"path": relative, "exists": path.exists(), "equivalent": False}
        if path.exists():
            schema = pq.read_schema(path)
            check["schema_columns"] = schema.names
            check["reason"] = "Historical scalar prediction schema has no sealed 13-horizon path cache bound to this origin and input-quality contract; no data columns opened"
        checks.append(check)
    return {"development_only": True, "reference_only": True,
            "equivalent_cache_found": False, "new_inference_required": True,
            "required_horizons": sorted(required_horizons), "checks": checks,
            "historical_predictions_used": False}


def download_model(out: Path, config: dict) -> dict:
    """Pin the public weight revision and download without uploading any data."""
    from huggingface_hub import HfApi, snapshot_download
    log = out / "logs/chronos_model_download.json"
    if log.exists():
        prior = json.loads(log.read_text(encoding="utf-8"))
        if Path(prior["local_path"]).is_dir():
            return prior
    model_id = config["chronos"]["model_id"]
    revision = HfApi().model_info(model_id).sha
    started = perf_counter()
    path = snapshot_download(model_id, revision=revision,
                             cache_dir=str(out / "models/hf_cache"),
                             allow_patterns=["*.json", "*.safetensors"])
    metadata = {"model_id": model_id, "revision": revision, "local_path": path,
                "download_seconds": perf_counter()-started, "development_only": True,
                "training_data_uploaded": False, "reference_only": True}
    log.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_reference(history: pd.DataFrame, contexts: dict, cfg: dict, out: Path) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline
    spec = cfg["chronos"]
    info = json.loads((out / "logs/chronos_model_download.json").read_text(encoding="utf-8"))
    power = history.power.astype(np.float32)
    origins = pd.DatetimeIndex(sorted(set().union(*(set(c["score"]) for c in contexts.values()))))
    if power.index.max() >= pd.Timestamp(cfg["boundary"]) or origins.max() >= pd.Timestamp(cfg["boundary"]):
        raise ValueError("Chronos reference crossed development boundary")
    fingerprint = hashlib.sha256(power.to_numpy().tobytes() + power.index.asi8.tobytes()
                                 + json.dumps(spec, sort_keys=True).encode()
                                 + info["revision"].encode()).hexdigest()
    cache_path = out / "models/chronos_paths.joblib"
    state = joblib.load(cache_path) if cache_path.exists() else {
        "fingerprint": fingerprint, "paths": {}, "inference_seconds": 0.0}
    if state["fingerprint"] != fingerprint:
        raise ValueError("Chronos cache input/specification fingerprint mismatch")
    todo = [origin for origin in origins if origin not in state["paths"]]
    if todo:
        torch.set_num_threads(2)
        torch.manual_seed(cfg["seed"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = BaseChronosPipeline.from_pretrained(info["local_path"], device_map=device,
                                                  torch_dtype=torch.float32)
        positions = pd.Series(np.arange(len(power)), index=power.index)
        values = power.to_numpy()
        for start in range(0, len(todo), spec["batch_size"]):
            batch = todo[start:start+spec["batch_size"]]
            arrays = []
            for origin in batch:
                end = int(positions[origin]) + 1
                arrays.append(torch.tensor(values[max(0, end-spec["context_length"]):end]))
            if device == "cuda":
                torch.cuda.synchronize()
            tick = perf_counter()
            with torch.inference_mode():
                quantiles, _ = pipe.predict_quantiles(arrays,
                    prediction_length=spec["prediction_length"], quantile_levels=[spec["quantile"]])
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = perf_counter()-tick
            state["inference_seconds"] += seconds
            for origin, quantile in zip(batch, quantiles):
                array = quantile.detach().cpu().numpy()
                if array.shape != (1, spec["prediction_length"], 1):
                    raise ValueError(f"Unexpected Chronos quantile shape {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError("Chronos produced a nonfinite reference forecast")
                state["paths"][origin] = {"pred": array[0,:,0], "seconds": seconds/len(batch)}
            joblib.dump(state, cache_path)
            if start % (spec["batch_size"]*10) == 0:
                print(f"Chronos reference {start+len(batch)}/{len(todo)} origins", flush=True)
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for (horizon, fold), context in contexts.items():
        score = context["score"]
        prediction = np.asarray([state["paths"][o]["pred"][horizon-1] for o in score])
        rows.append(pd.DataFrame({"model":"R1", "horizon":horizon,"fold":fold,
            "origin":score,"target_time":context["target_time"].loc[score].to_numpy(),
            "y":context["y"].loc[score].to_numpy(),"pred":prediction,"tau":context["tau"],
            "d2":context["d2"].loc[score].to_numpy(),"train_seconds":0.0,
            "inference_seconds":sum(state["paths"][o]["seconds"] for o in score),
            "selected_config":json.dumps({**spec,"revision":info["revision"]},sort_keys=True),
            "development_only":True}))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(out / "predictions/reference_predictions.parquet", index=False)
    (out / "logs/chronos_runtime.json").write_text(json.dumps({
        "development_only":True,"reference_only":True,"unique_origins":len(origins),
        "inference_seconds":state["inference_seconds"],"prediction_paths_shared_across_horizons":True,
        "pretrained_training_seconds":None,"local_fit_seconds":0.0,
        "input_fingerprint":fingerprint,"model_revision":info["revision"],
        "missing_context_policy":"Native Chronos observed-value mask for NaN; no interpolation or future fill"},indent=2),encoding="utf-8")
    return predictions
