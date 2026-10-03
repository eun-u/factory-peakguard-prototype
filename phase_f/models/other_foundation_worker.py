"""Isolated optional-dependency worker for public zero-shot model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


LEVELS = np.arange(1, 10, dtype=float) / 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quarantine_orphan(path: Path) -> Path:
    """Keep bytes from a chunk written before its manifest update."""
    digest = _sha256(path)
    for number in range(10000):
        suffix = "" if number == 0 else f"-{number}"
        orphan = path.with_name(f"{path.stem}.orphan-{digest}{suffix}{path.suffix}")
        if not orphan.exists():
            path.rename(orphan)
            return orphan
    raise RuntimeError("Too many orphan files for worker chunk")


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _load(kind: str, snapshot: Path, context_length: int, batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if kind == "chronos_bolt":
        from chronos import ChronosBoltPipeline
        model = ChronosBoltPipeline.from_pretrained(str(snapshot), device_map=device,
                                                    torch_dtype=torch.float32)
        if context_length > model.model_context_length:
            raise ValueError("Chronos-Bolt context exceeds supported length")
    elif kind == "timesfm2p5":
        import timesfm
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(snapshot),
                                                                force_download=False)
        model.compile(timesfm.ForecastConfig(
            max_context=context_length, max_horizon=16,
            normalize_inputs=True, per_core_batch_size=min(batch_size, 8),
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))
    elif kind == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(str(snapshot)),
            prediction_length=16, context_length=context_length, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(device).eval()
    elif kind == "tirex":
        from tirex import load_model
        # The public loader requires a repository ID; local_files_only ensures
        # the already pinned snapshot is used without another network request.
        model = load_model("NX-AI/TiRex", device=device,
                           hf_kwargs={"revision": snapshot.name,
                                      "cache_dir": str(snapshot.parents[2]),
                                      "local_files_only": True})
        model.eval()
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return model


def _forecast(kind: str, model, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(inputs)
    if kind == "chronos_bolt":
        with torch.inference_mode():
            output = model.predict([torch.from_numpy(x) for x in inputs], prediction_length=16)
        quantiles = output.detach().cpu().numpy().transpose(0, 2, 1)
        mean = np.full((n, 16), np.nan, dtype="float32")
    elif kind == "timesfm2p5":
        with torch.inference_mode():
            point, distribution = model.forecast(horizon=16, inputs=[x for x in inputs])
        distribution = np.asarray(distribution)
        quantiles = distribution[:, :, 1:]
        mean = distribution[:, :, 0]
        if not np.allclose(point, quantiles[:, :, 4], rtol=1e-5, atol=1e-5):
            raise RuntimeError("TimesFM point and q50 disagree")
    elif kind == "moirai2":
        with torch.inference_mode():
            output = model.predict([np.asarray(x) for x in inputs])
        quantiles = np.asarray(output.detach().cpu() if hasattr(output, "detach") else output)
        quantiles = quantiles.transpose(0, 2, 1)
        mean = np.full((n, 16), np.nan, dtype="float32")
    elif kind == "tirex":
        with torch.inference_mode():
            q, m = model.forecast(torch.from_numpy(inputs), prediction_length=16)
        quantiles = np.asarray(q.detach().cpu() if hasattr(q, "detach") else q)
        mean = np.asarray(m.detach().cpu() if hasattr(m, "detach") else m)
    else:
        raise ValueError(kind)
    quantiles = np.asarray(quantiles, dtype="float32")
    mean = np.asarray(mean, dtype="float32")
    if quantiles.shape != (n, 16, 9) or mean.shape != (n, 16):
        raise RuntimeError(f"Invalid model output shape: {quantiles.shape}, {mean.shape}")
    if not np.isfinite(quantiles).all():
        raise RuntimeError("Nonfinite zero-shot quantile")
    return quantiles, mean


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("kind", "snapshot", "request", "output", "cache-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument('--require-cuda',action='store_true')
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA execution required by this diagnostic')
    request, output = Path(args.request), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(request, allow_pickle=False) as data:
        contexts = data["context"]
        probe = data["probe"]
    if contexts.ndim != 2 or contexts.shape[0] == 0 or probe.shape != contexts[0].shape:
        raise RuntimeError("Invalid causal input transaction")
    if not np.array_equal(contexts[0], probe):
        raise AssertionError("Future perturbation changed causal input")
    batch_size = max(1, args.batch_size)
    manifest_path = output / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists()
                else {"cache_id": args.cache_id, "request_sha256": _sha256(request),
                      "chunks": [], "future_perturbation_max_abs_difference": None})
    if manifest["cache_id"] != args.cache_id or manifest["request_sha256"] != _sha256(request):
        raise RuntimeError("Worker cache identity mismatch")
    expected = 0
    for row in manifest["chunks"]:
        path = output / row["file"]
        if row["start"] != expected or row["stop"] > len(contexts) or not path.exists() or _sha256(path) != row["sha256"]:
            raise RuntimeError("Prior worker chunk failed integrity check")
        expected = row["stop"]
    if expected == len(contexts):
        return
    model = _load(args.kind, Path(args.snapshot), contexts.shape[1], batch_size)
    module=model if hasattr(model,'parameters') else getattr(model,'model',None)
    parameter=next(module.parameters(),None) if module is not None and hasattr(module,'parameters') else None
    manifest['model_device']=str(parameter.device) if parameter is not None else 'unknown'
    if args.require_cuda and not manifest['model_device'].startswith('cuda'):
        raise RuntimeError('Loaded model parameters are not on CUDA')
    if expected == 0:
        left, _ = _forecast(args.kind, model, contexts[:1])
        right, _ = _forecast(args.kind, model, probe[None, :])
        difference = float(np.max(np.abs(left - right)))
        if not np.array_equal(left, right):
            raise AssertionError(f"Future perturbation changed forecast: {difference}")
        manifest["future_perturbation_max_abs_difference"] = difference
    for start in range(expected, len(contexts), batch_size):
        stop = min(start + batch_size, len(contexts))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tick = perf_counter()
        quantiles, mean = _forecast(args.kind, model, contexts[start:stop])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = perf_counter() - tick
        file = f"chunk_{start:06d}_{stop:06d}.npz"
        path = output / file
        if path.exists():
            if not path.is_file():
                raise RuntimeError("Unregistered worker output is not a file")
            # The chunk rename may have succeeded immediately before a crash.
            # Never trust an unregistered output; preserve and recompute it.
            _quarantine_orphan(path)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, quantiles=quantiles, mean=mean, levels=LEVELS)
        digest = _sha256(temp)
        temp.replace(path)
        manifest["chunks"].append({"file": file, "start": start, "stop": stop,
                                    "sha256": digest, "forecast_seconds": seconds})
        _write_json(manifest_path, manifest)
        if start == 0 or stop % (batch_size * 10) == 0 or stop == len(contexts):
            print(f"{args.kind}: {stop}/{len(contexts)} origins", flush=True)


if __name__ == "__main__":
    main()
