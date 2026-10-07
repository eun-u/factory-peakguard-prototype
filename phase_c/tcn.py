"""Isolated, direct TCN regressor for one Phase C forecast horizon.

This module accepts already separated fit and stop arrays. It never reads data,
chooses a split, or evaluates a final holdout. Train one independent bundle per
forecast horizon. A bundle contains CPU tensors and Python primitives, so it can
be saved with ``torch.save(bundle, path)`` and restored with
``torch.load(path, map_location="cpu", weights_only=True)``. The same bundle can
be passed to :func:`predict_tcn` on another machine with PyTorch installed.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


SEQUENCE_LENGTH = 96
KERNEL_SIZE = 3
DILATIONS = (1, 2, 4, 8, 16, 32)
SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.left_padding = (KERNEL_SIZE - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, KERNEL_SIZE, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, channels, dilation)
        self.conv2 = _CausalConv1d(channels, channels, dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == channels
            else nn.Conv1d(in_channels, channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))
        return F.relu(out + self.residual(x))


class _TCNDirect(nn.Module):
    def __init__(self, context_dim: int, channels: int, dropout: float):
        super().__init__()
        blocks = []
        in_channels = 1
        for dilation in DILATIONS:
            blocks.append(_ResidualBlock(in_channels, channels, dilation, dropout))
            in_channels = channels
        self.encoder = nn.Sequential(*blocks)
        self.output = nn.Linear(channels + context_dim, 1)

    def forward(self, seq: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # seq: [batch, 1, 96]. Only the final causal state predicts the target.
        last_state = self.encoder(seq)[:, :, -1]
        return self.output(torch.cat((last_state, context), dim=1)).squeeze(1)


def _validated_inputs(
    seq: np.ndarray, context: np.ndarray, y: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    seq = np.asarray(seq, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != SEQUENCE_LENGTH or not len(seq):
        raise ValueError(f"seq must have nonempty shape [N, {SEQUENCE_LENGTH}]")
    if context.ndim != 2 or context.shape[0] != len(seq) or context.shape[1] < 1:
        raise ValueError("context must have shape [N, K] with K >= 1")
    if not np.isfinite(seq).all() or not np.isfinite(context).all():
        raise ValueError("seq and context must contain only finite values")
    if y is not None:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or len(y) != len(seq) or not np.isfinite(y).all():
            raise ValueError("y must be a finite vector of length N")
    return seq, context, y


def _training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    channels = int(config.get("channels", 32))
    dropout = float(config.get("dropout", 0.1))
    max_epochs = int(config.get("max_epochs", MAX_EPOCHS))
    patience = int(config.get("patience", PATIENCE))
    if channels not in (32, 64):
        raise ValueError("channels must be 32 or 64")
    if dropout not in (0.1, 0.2):
        raise ValueError("dropout must be 0.1 or 0.2")
    if not 1 <= max_epochs <= MAX_EPOCHS:
        raise ValueError(f"max_epochs must be between 1 and {MAX_EPOCHS}")
    if not 1 <= patience <= PATIENCE:
        raise ValueError(f"patience must be between 1 and {PATIENCE}")
    return {
        "channels": channels,
        "dropout": dropout,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "kernel_size": KERNEL_SIZE,
        "dilations": list(DILATIONS),
    }


def _fit_normalization(
    seq_fit: np.ndarray, context_fit: np.ndarray, y_fit: np.ndarray
) -> dict[str, Any]:
    # Population statistics of fit rows only. Constant columns retain scale 1.
    seq_mean = float(np.mean(seq_fit, dtype=np.float64))
    seq_std = float(np.std(seq_fit, dtype=np.float64))
    y_mean = float(np.mean(y_fit, dtype=np.float64))
    y_std = float(np.std(y_fit, dtype=np.float64))
    scaler = StandardScaler().fit(context_fit)
    return {
        "seq_mean": seq_mean,
        "seq_scale": seq_std if seq_std > 1e-12 else 1.0,
        "context_mean": scaler.mean_.astype(float).tolist(),
        "context_scale": scaler.scale_.astype(float).tolist(),
        "y_mean": y_mean,
        "y_scale": y_std if y_std > 1e-12 else 1.0,
    }


def _normalize(
    seq: np.ndarray, context: np.ndarray, norm: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_norm = (seq.astype(np.float64) - norm["seq_mean"]) / norm["seq_scale"]
    context_norm = (
        context.astype(np.float64) - np.asarray(norm["context_mean"])
    ) / np.asarray(norm["context_scale"])
    return (
        torch.from_numpy(np.ascontiguousarray(seq_norm[:, None, :], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(context_norm, dtype=np.float32)),
    )


def _device(device: str | torch.device | None) -> torch.device:
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def _seed_training() -> None:
    # Set before the first CUDA operation. Deterministic mode may reject an
    # unsupported CUDA kernel rather than silently produce different weights.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _elapsed(device: torch.device, start: float) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def fit_tcn(
    seq_fit: np.ndarray,
    context_fit: np.ndarray,
    y_fit: np.ndarray,
    seq_stop: np.ndarray,
    context_stop: np.ndarray,
    y_stop: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Fit a direct scalar TCN on fit rows; select an epoch with stop MAE.

    ``seq_*`` has shape ``[N, 96]`` in time order, ``context_*`` has shape
    ``[N, K]``, and ``y_*`` has shape ``[N]``. Inputs must already be finite and
    split by the caller. Sequence and target statistics plus a context
    ``StandardScaler`` are fit on fit rows only. Stop rows affect only the
    selected epoch; they never update weights or normalization statistics.

    ``config`` selects ``channels`` (32 or 64) and ``dropout`` (0.1 or 0.2).
    Optional ``max_epochs`` and ``patience`` can reduce the fixed ceilings of
    100 and 10 for smoke tests. Other optimizer and architecture settings are
    fixed. The returned bundle has CPU ``state_dict`` tensors, Python-only
    normalization/config dictionaries, and metadata with per-epoch history,
    best epoch, stop MAE, timings, device, and package version. Save/load via
    ``torch.save`` / ``torch.load(..., weights_only=True)``.
    """
    cfg = _training_config(config)
    seq_fit, context_fit, y_fit = _validated_inputs(seq_fit, context_fit, y_fit)
    seq_stop, context_stop, y_stop = _validated_inputs(seq_stop, context_stop, y_stop)
    if context_fit.shape[1] != context_stop.shape[1]:
        raise ValueError("fit and stop context dimensions differ")
    assert y_fit is not None and y_stop is not None
    _seed_training()
    dev = _device(device)
    norm = _fit_normalization(seq_fit, context_fit, y_fit)
    fit_seq, fit_context = _normalize(seq_fit, context_fit, norm)
    stop_seq, stop_context = _normalize(seq_stop, context_stop, norm)
    fit_y = torch.from_numpy(np.ascontiguousarray(
        (y_fit.astype(np.float64) - norm["y_mean"]) / norm["y_scale"],
        dtype=np.float32,
    ))
    stop_y = torch.from_numpy(np.ascontiguousarray(y_stop, dtype=np.float32))

    # Small windows can remain on the GPU. Larger windows are transferred in
    # batches so the same implementation also works when GPU memory is limited.
    byte_count = sum(x.numel() * x.element_size() for x in (
        fit_seq, fit_context, fit_y, stop_seq, stop_context, stop_y
    ))
    preload = dev.type == "cuda" and byte_count <= 256 * 1024 * 1024
    if preload:
        fit_seq, fit_context, fit_y, stop_seq, stop_context, stop_y = (
            x.to(dev) for x in (
                fit_seq, fit_context, fit_y, stop_seq, stop_context, stop_y
            )
        )

    model = _TCNDirect(context_fit.shape[1], cfg["channels"], cfg["dropout"]).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start_total = time.perf_counter()

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        start_train = time.perf_counter()
        order = torch.randperm(len(seq_fit), generator=generator)
        loss_sum = 0.0
        for offset in range(0, len(order), BATCH_SIZE):
            idx = order[offset : offset + BATCH_SIZE].to(fit_seq.device)
            batch_seq = fit_seq[idx].to(dev)
            batch_context = fit_context[idx].to(dev)
            batch_y = fit_y[idx].to(dev)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_seq, batch_context)
            loss = F.l1_loss(prediction, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(idx)
        train_seconds = _elapsed(dev, start_train)

        model.eval()
        start_stop = time.perf_counter()
        absolute_error = 0.0
        with torch.inference_mode():
            for offset in range(0, len(seq_stop), BATCH_SIZE):
                end = offset + BATCH_SIZE
                prediction = model(
                    stop_seq[offset:end].to(dev), stop_context[offset:end].to(dev)
                )
                raw_prediction = prediction * norm["y_scale"] + norm["y_mean"]
                absolute_error += float(
                    torch.abs(raw_prediction - stop_y[offset:end].to(dev)).sum().item()
                )
        stop_seconds = _elapsed(dev, start_stop)
        stop_mae = absolute_error / len(seq_stop)
        history.append({
            "epoch": epoch,
            "train_l1_normalized": loss_sum / len(seq_fit),
            "stop_mae": stop_mae,
            "train_seconds": train_seconds,
            "stop_seconds": stop_seconds,
        })
        if stop_mae < best_mae:
            best_mae = stop_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg["patience"]:
                break

    assert best_state is not None
    metadata = {
        "development_only": True,
        "seed": SEED,
        "device": str(dev),
        "torch_version": str(torch.__version__),
        "fit_rows": len(seq_fit),
        "stop_rows": len(seq_stop),
        "best_epoch": best_epoch,
        "best_stop_mae": best_mae,
        "epochs_ran": len(history),
        "early_stopped": len(history) < cfg["max_epochs"],
        "training_schedule": {
            "max_epochs": cfg["max_epochs"],
            "patience": cfg["patience"],
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "L1_normalized_target",
            "stop_metric": "MAE_original_target_units",
        },
        "train_seconds": sum(row["train_seconds"] for row in history),
        "stop_seconds": sum(row["stop_seconds"] for row in history),
        "total_seconds": _elapsed(dev, start_total),
        "history": history,
    }
    return {
        "model_type": "tcn_direct",
        "context_dim": context_fit.shape[1],
        "config": cfg,
        "normalization": norm,
        "state_dict": best_state,
        "metadata": metadata,
    }


def predict_tcn(
    bundle: Mapping[str, Any],
    seq: np.ndarray,
    context: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Return one raw-unit prediction per row from a fitted TCN bundle.

    A loaded bundle runs on CUDA when available and otherwise on CPU. The
    saved fit-only normalization is reused without fitting or mutation.
    Returns a float64 NumPy vector of shape ``[N]``.
    """
    if bundle.get("model_type") != "tcn_direct":
        raise ValueError("bundle is not a tcn_direct model")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    seq, context, _ = _validated_inputs(seq, context)
    if context.shape[1] != bundle["context_dim"]:
        raise ValueError("context dimension differs from the fitted model")
    dev = _device(None)
    cfg = bundle["config"]
    model = _TCNDirect(bundle["context_dim"], cfg["channels"], cfg["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model = model.to(dev).eval()
    norm = bundle["normalization"]
    seq_tensor, context_tensor = _normalize(seq, context, norm)
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(seq), batch_size):
            end = offset + batch_size
            normalized = model(
                seq_tensor[offset:end].to(dev), context_tensor[offset:end].to(dev)
            )
            chunks.append(normalized.cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) * norm["y_scale"] + norm["y_mean"]
