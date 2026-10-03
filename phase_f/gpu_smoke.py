"""Explicit, synthetic-only GPU readiness check for Phase F model adapters.

Run after the normal tests with::

    outputs/phase_f/env/Scripts/python.exe -m phase_f.gpu_smoke --root .

This is deliberately separate from the experiment registry. Passing it is not
plant-data validation, an Optuna trial, or evidence of commercial accuracy.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from phase_f.registry import sha256, write_json
from phase_f.runtime import activate_optional_dependencies


NEURAL_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "dlinear": {}, "nlinear": {}, "tcn": {}, "lstm": {}, "gru": {},
    "nhits": {"n_blocks": [1, 1, 1], "mlp_units": [[16, 16]] * 3},
    "nbeats": {"n_blocks": [1, 1, 1], "mlp_units": [[16, 16]] * 3},
    "patchtst": {"encoder_layers": 1, "n_heads": 2, "hidden_size": 16,
                 "linear_hidden_size": 32, "patch_len": 4, "stride": 2},
    "tide": {"hidden_size": 32, "decoder_output_dim": 8,
             "temporal_decoder_dim": 16, "temporal_width": 4},
    "tsmixer": {"n_block": 1, "ff_dim": 16, "dropout": 0.1},
    "timesnet": {"hidden_size": 8, "conv_hidden_size": 8, "top_k": 2,
                 "num_kernels": 1, "encoder_layers": 1},
    "itransformer": {"hidden_size": 32, "n_heads": 2,
                     "e_layers": 1, "d_layers": 1, "d_ff": 64},
    "xlstm": {},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _synthetic_case() -> tuple[pd.DataFrame, dict[str, Any], pd.DatetimeIndex]:
    index = pd.date_range("2021-04-01 00:15", periods=16 * 96,
                          freq="15min", name="ts_end")
    slot = np.arange(len(index), dtype=float)
    power = 50 + 5 * np.sin(2 * np.pi * slot / 96) + 0.002 * slot
    history = pd.DataFrame({"power": power, "time_repaired": False}, index=index)
    fit = pd.DatetimeIndex(index[950:1010:2], name="origin")
    stop = pd.DatetimeIndex(index[1030:1050:2], name="origin")
    origins = pd.DatetimeIndex(index[1200:1203], name="origin")
    target_time = pd.Series(index, index=index) + pd.Timedelta(hours=1)
    y = pd.Series(power, index=index).reindex(index + pd.Timedelta(hours=1)).to_numpy()
    context = {"fit": fit, "stop": stop, "y": pd.Series(y, index=index),
               "target_time": target_time, "tau": 58.0,
               "summary": {"horizon_quarters": 4}}
    return history, context, origins


def _assert_causal_equal(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise AssertionError("Invalid or mismatched prediction shape/values")
    maximum = float(np.max(np.abs(before - after)))
    if maximum > 1e-5:
        raise AssertionError(f"Future perturbation changed predictions by {maximum:g}")
    return {"future_perturbation_max_abs_difference": maximum,
            "future_perturbation_exact": bool(np.array_equal(before, after)),
            "numeric_tolerance_used": maximum != 0,
            "numeric_tolerance_reason": ("Repeated GPU inference may differ by at most 1e-5"
                                         if maximum != 0 else None)}


def _neural_check(kind: str, case: tuple[pd.DataFrame, dict[str, Any], pd.DatetimeIndex]) -> dict[str, Any]:
    from phase_f.models import neural

    history, context, origins = case
    cfg: dict[str, Any] = {
        "context_length": 16, "horizon": 4, "seeds": [42],
        "max_epochs": 1, "patience": 1, "batch_size": 8,
        "architecture_kwargs": NEURAL_ARCHITECTURES[kind],
        "use_calendar_exog": False, "exog_columns": [], "device": "cuda",
    }
    if kind == "tcn":
        cfg.update(channels=4, depth=2)
    elif kind in ("lstm", "gru"):
        cfg["hidden_size"] = 8
    if kind == "nhits":
        # Default linear interpolation has no deterministic CUDA backward.
        # Keep the canonical architecture; the production plan records this mode.
        cfg["determinism_mode"] = "warn_only"
    if kind == "xlstm":
        cfg.update(xlstm_embedding_dim=32, xlstm_num_blocks=2, xlstm_num_heads=4)
    bundle = neural.fit_model(kind, history, context, cfg)
    if bundle["config"]["device"] != "cuda" or bundle["fit_count"] != 30 or bundle["stop_count"] != 10:
        raise AssertionError("Neural smoke used unexpected device or fit/stop cohort")
    before = np.asarray(neural.predict_model(bundle, history, origins, 4), dtype=float)
    changed = history.copy()
    changed.loc[changed.index > origins.max(), "power"] += 50_000
    after = np.asarray(neural.predict_model(bundle, changed, origins, 4), dtype=float)
    if before.shape != (3,):
        raise AssertionError(f"Expected three neural predictions, got {before.shape}")
    result = _assert_causal_equal(before, after)
    result.update(kind=kind, seed=42, epochs=1, fit_origins=30, stop_origins=10,
                  prediction_origins=3, horizon_quarters=4, device="cuda",
                  fitted_epoch=int(bundle["seeds"][0]["best_epoch"]),
                  determinism_mode=bundle["determinism_mode"],
                  nondeterministic_operations=bundle["nondeterministic_operations"])
    return result


def _chronos_windows(history: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    values = history["power"].to_numpy(dtype=np.float32)
    length, horizon = 512, 16

    def window(origin_position: int) -> np.ndarray:
        result = values[origin_position + 1 - length:origin_position + 1 + horizon].copy()
        if result.shape != (length + horizon,) or not np.isfinite(result).all():
            raise AssertionError("Synthetic Chronos window is incomplete")
        return result

    # Seeded random synthetic windows; training targets end before validation.
    rng = np.random.default_rng(42)
    train_positions = np.sort(rng.choice(np.arange(700, 821), size=4, replace=False))
    valid_positions = np.sort(rng.choice(np.arange(1080, 1121), size=2, replace=False))
    train = [window(int(position)) for position in train_positions]
    valid = [window(int(position)) for position in valid_positions]
    query = values[1201 - length:1201].copy()
    if query.shape != (length,):
        raise AssertionError("Synthetic Chronos query is incomplete")
    return train, valid, query


def _chronos_check(root: Path, mode: str, run_dir: Path,
                   case: tuple[pd.DataFrame, dict[str, Any], pd.DatetimeIndex]) -> dict[str, Any]:
    import torch
    from chronos import Chronos2Pipeline
    from phase_f.models.foundation import QUANTILES, require_finetune_mode

    require_finetune_mode(mode)
    log = root / "outputs/phase_c/logs/chronos_model_download.json"
    info = json.loads(log.read_text(encoding="utf-8"))
    if info.get("model_id") != "amazon/chronos-2" or info.get("development_only") is not True:
        raise RuntimeError("Protected Chronos snapshot metadata is not development-only Chronos-2")
    revision = str(info["revision"])
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Invalid pinned Chronos revision")
    snapshot = root / "outputs/phase_c/models/hf_cache/models--amazon--chronos-2/snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned local Chronos-2 snapshot is missing")

    history, _, _ = case
    train, valid, query = _chronos_windows(history)
    output_dir = run_dir / mode  # A new run directory; never reuse a prior checkpoint.
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke checkpoint: {output_dir}")
    torch.manual_seed(42)
    pipe = Chronos2Pipeline.from_pretrained(str(snapshot), device_map="cuda", torch_dtype=torch.float32)
    if int(pipe.model_context_length) < 512:
        raise RuntimeError("Pinned Chronos-2 does not support context length 512")
    pipe = pipe.fit(
        inputs=train, prediction_length=16, validation_inputs=valid,
        finetune_mode=mode, context_length=512, min_past=512,
        learning_rate=1e-5 if mode == "lora" else 1e-6,
        num_steps=2, batch_size=2, output_dir=output_dir,
        eval_strategy="steps", eval_steps=1,
        save_strategy="steps", save_steps=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=2,
        report_to="none", dataloader_num_workers=0, seed=42, disable_tqdm=True,
    )
    checkpoint = output_dir / "finetuned-ckpt"
    if not checkpoint.is_dir() or not any(path.is_file() for path in checkpoint.rglob("*")):
        raise AssertionError("Chronos fit did not save a reloadable checkpoint")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map="cuda",
                                                torch_dtype=torch.float32)
    with torch.inference_mode():
        quantiles, _ = reloaded.predict_quantiles([query], prediction_length=16,
            quantile_levels=QUANTILES, context_length=512)
    forecast = quantiles[0].detach().cpu().numpy()
    if forecast.shape != (1, 16, len(QUANTILES)) or not np.isfinite(forecast).all():
        raise AssertionError(f"Invalid Chronos quantile forecast: {forecast.shape}")

    # The history perturbation occurs strictly after the forecast origin.
    changed = history.copy()
    origin = history.index[1200]
    changed.loc[changed.index > origin, "power"] += 50_000
    changed_query = changed["power"].loc[:origin].iloc[-512:].to_numpy(dtype=np.float32)
    if not np.array_equal(query, changed_query):
        raise AssertionError("Chronos query construction read future history")
    with torch.inference_mode():
        altered_quantiles, _ = reloaded.predict_quantiles([changed_query], prediction_length=16,
            quantile_levels=QUANTILES, context_length=512)
    result = _assert_causal_equal(forecast, altered_quantiles[0].detach().cpu().numpy())
    result.update(mode=mode, device="cuda", seed=42, training_windows=4,
                  validation_windows=2, context_length=512, prediction_length=16,
                  quantile_count=len(QUANTILES), quantile_shape=list(forecast.shape),
                  fit_steps=2, checkpoint_dir=checkpoint.relative_to(root).as_posix(),
                  checkpoint_files={path.relative_to(checkpoint).as_posix(): sha256(path)
                                    for path in sorted(checkpoint.rglob("*")) if path.is_file()},
                  protected_snapshot_revision=revision, protected_snapshot_log_sha256=sha256(log))
    return result


def run_gpu_synthetic_smoke(root: str | Path) -> dict[str, Any]:
    """Run each GPU adapter sequentially, auditing each result before the next."""
    import torch

    root = Path(root).absolute()
    expected_python = (root / "outputs/phase_f/env/Scripts/python.exe").absolute()
    if Path(sys.executable).absolute() != expected_python:
        raise RuntimeError("Invoke with outputs/phase_f/env/Scripts/python.exe")
    out = root / "outputs/phase_f"
    log_path = out / "logs/gpu_synthetic_smoke.json"
    run_id = _utc_now().replace(":", "").replace("+00:00", "Z") + "-" + uuid.uuid4().hex[:8]
    record: dict[str, Any] = {
        "run_id": run_id, "started_at": _utc_now(), "status": "running",
        "synthetic_only": True, "not_plant": True, "not_search_trials": True,
        "holdout_read": False, "historical_final_artifact_read": False,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "packages": {name: _version(name) for name in
                         ("chronos-forecasting", "transformers", "peft", "neuralforecast", "xlstm")}},
        "checks": [],
    }
    if log_path.exists():
        document = json.loads(log_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or not isinstance(document.get("runs"), list):
            raise RuntimeError("Existing GPU smoke log has an incompatible schema")
    else:
        document = {"schema_version": 1, "runs": []}
    document["runs"].append(record)

    def persist() -> None:
        write_json(log_path, document)

    def check(name: str, function: Callable[[], dict[str, Any]]) -> None:
        item: dict[str, Any] = {"name": name, "status": "running", "started_at": _utc_now()}
        record["checks"].append(item)
        persist()
        started = time.perf_counter()
        try:
            item.update(function())
            item["status"] = "passed"
        except BaseException as error:
            item.update(status="failed", error_type=type(error).__name__,
                        error=str(error)[:1000])
            raise
        finally:
            item["finished_at"] = _utc_now()
            item["duration_seconds"] = time.perf_counter() - started
            persist()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    persist()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU smoke has no CPU fallback")
        record["environment"].update(device_name=torch.cuda.get_device_name(0),
                                     device_count=torch.cuda.device_count(),
                                     cuda_available=True)
        activate_optional_dependencies(root)
        from phase_f.models.neural import NATIVE_KINDS, OPTIONAL_NIXTLA_KINDS, OPTIONAL_XLSTM_KINDS
        kinds = (*NATIVE_KINDS, *OPTIONAL_NIXTLA_KINDS, *OPTIONAL_XLSTM_KINDS)
        if len(kinds) != 13 or set(kinds) != set(NEURAL_ARCHITECTURES):
            raise RuntimeError("Neural smoke does not cover exactly the 13 supported kinds")
        case = _synthetic_case()
        for kind in kinds:
            check(f"neural_{kind}", lambda kind=kind: _neural_check(kind, case))
        run_dir = out / "models/gpu_smoke" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for mode in ("full", "lora"):
            check(f"chronos2_{mode}", lambda mode=mode: _chronos_check(root, mode, run_dir, case))
        record["status"] = "passed"
        record["finished_at"] = _utc_now()
        persist()
        return record
    except BaseException as error:
        record.update(status="failed", finished_at=_utc_now(),
                      error_type=type(error).__name__, error=str(error)[:1000])
        persist()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Phase C workspace root containing outputs/phase_f/env")
    args = parser.parse_args()
    result = run_gpu_synthetic_smoke(args.root)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "checks": len(result["checks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
