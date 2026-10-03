"""Phase F neural forecasts with sealed, per-horizon fit/stop roles.

Only the requested Phase C fit origins update weights or scalers. Stop labels
select the epoch. The caller owns score/CONFIRM access and candidate selection.
Each bundle predicts one horizon and stores CPU weights for reproducible replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from phase_f.features_ext import build_features, causal_sequences
from src.holidays import HOLIDAYS_2021
from src.session_data import SEALED_BOUNDARY


NATIVE_KINDS = ("dlinear", "nlinear", "tcn", "lstm", "gru")
OPTIONAL_NIXTLA_KINDS = (
    "nhits", "nbeats", "patchtst", "tide", "tsmixer", "timesnet", "itransformer",
)
CONTEXT_GRID = (96, 192, 336, 672, 1344, 2016, 2688)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
QUANTILES = (0.1, 0.5, 0.9, 0.95)
_CALENDAR_NAMES = ("slot_sin", "slot_cos", "dow_sin", "dow_cos", "holiday")


def configurations() -> list[dict[str, Any]]:
    """F5 native and Nixtla candidates; runtime import decides availability."""
    plans = []
    for context in CONTEXT_GRID:
        for kind, exp_id in (("tcn", "F5-1"), ("dlinear", "F5-2"), ("nlinear", "F5-2")):
            plans.append({"exp_id": exp_id, "kind": kind, "context_length": context,
                          "status": "supported_native"})
    for kind in ("lstm", "gru"):
        plans.append({"exp_id": "F5-5", "kind": kind, "context_length": 96,
                      "status": "supported_native"})
    for kind in OPTIONAL_NIXTLA_KINDS:
        exp_id = "F5-3" if kind in ("nhits", "nbeats") else (
            "F5-4" if kind in ("patchtst", "tide") else "F5-7"
        )
        for context in CONTEXT_GRID:
            plans.append({"exp_id": exp_id, "kind": kind, "context_length": context,
                          "status": "supported_optional_neuralforecast"})
    return plans


class _DLinear(nn.Module):
    """Moving-average decomposition with distinct seasonal/trend linear heads."""

    def __init__(self, length: int, output_dim: int, exog_dim: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.seasonal = nn.Linear(length, output_dim)
        self.trend = nn.Linear(length, output_dim)
        self.exog = nn.Linear(exog_dim, output_dim, bias=False) if exog_dim else None

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel - 1) // 2
        moving = F.avg_pool1d(F.pad(seq[:, None, :], (padding, padding), mode="replicate"),
                              kernel_size=self.kernel, stride=1).squeeze(1)
        prediction = self.seasonal(seq - moving) + self.trend(moving)
        return prediction + self.exog(exog) if self.exog is not None else prediction


class _NLinear(nn.Module):
    """Last-value subtraction and one direct linear horizon head."""

    def __init__(self, length: int, output_dim: int, exog_dim: int):
        super().__init__()
        self.linear = nn.Linear(length, output_dim)
        self.exog = nn.Linear(exog_dim, output_dim, bias=False) if exog_dim else None

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        last = seq[:, -1:]
        prediction = self.linear(seq - last) + last
        return prediction + self.exog(exog) if self.exog is not None else prediction


class _CausalConv(nn.Module):
    def __init__(self, input_dim: int, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(input_dim, channels, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, input_dim: int, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.first = _CausalConv(input_dim, channels, kernel, dilation)
        self.second = _CausalConv(channels, channels, kernel, dilation)
        self.skip = nn.Identity() if input_dim == channels else nn.Conv1d(input_dim, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(F.relu(self.first(x)))
        z = self.drop(F.relu(self.second(z)))
        return F.relu(z + self.skip(x))


class _TCN(nn.Module):
    def __init__(self, output_dim: int, exog_dim: int, channels: int, depth: int,
                 kernel: int, dropout: float):
        super().__init__()
        blocks = []
        input_dim = 2  # normalized power plus observed-value mask
        for level in range(depth):
            blocks.append(_TCNBlock(input_dim, channels, kernel, 2 ** level, dropout))
            input_dim = channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(channels + exog_dim, output_dim)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        value = torch.stack((seq, mask), dim=1)
        state = self.encoder(value)[:, :, -1]
        return self.head(torch.cat((state, exog), dim=1))


class _Recurrent(nn.Module):
    def __init__(self, kind: str, output_dim: int, exog_dim: int, hidden: int,
                 layers: int, dropout: float):
        super().__init__()
        cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.encoder = cell(2, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden + exog_dim, output_dim)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        value = torch.stack((seq, mask), dim=2)
        states, _ = self.encoder(value)
        return self.head(torch.cat((states[:, -1, :], exog), dim=1))


def _model(kind: str, config: Mapping[str, Any], output_dim: int, exog_dim: int) -> nn.Module:
    length = int(config["context_length"])
    if kind == "dlinear":
        kernel = int(config.get("moving_average", 25))
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("moving_average must be a positive odd number")
        return _DLinear(length, output_dim, exog_dim, kernel)
    if kind == "nlinear":
        return _NLinear(length, output_dim, exog_dim)
    if kind == "tcn":
        return _TCN(output_dim, exog_dim, int(config.get("channels", 32)),
                    int(config.get("depth", 6)), int(config.get("kernel_size", 3)),
                    float(config.get("dropout", 0.1)))
    if kind in ("lstm", "gru"):
        return _Recurrent(kind, output_dim, exog_dim, int(config.get("hidden_size", 64)),
                          int(config.get("num_layers", 1)), float(config.get("dropout", 0.1)))
    if kind in OPTIONAL_NIXTLA_KINDS:
        try:
            from neuralforecast import models as nixtla_models
            from neuralforecast.losses.pytorch import MAE, MQLoss
        except ImportError as error:
            raise ImportError(f"{kind} requires the neuralforecast package") from error
        class_name = {"nhits": "NHITS", "nbeats": "NBEATS", "patchtst": "PatchTST",
                      "tide": "TiDE", "tsmixer": "TSMixer", "timesnet": "TimesNet",
                      "itransformer": "iTransformer"}[kind]
        kwargs = dict(config.get("architecture_kwargs", {}))
        kwargs.update({"h": int(config["horizon"]), "input_size": length,
                       "random_seed": int(config.get("model_seed", 42)),
                       "loss": MQLoss(quantiles=config["quantiles"])
                       if config["loss"] == "quantile" else MAE(),
                       "max_steps": 1, "scaler_type": "identity"})
        if kind in ("tsmixer", "itransformer"):
            kwargs["n_series"] = 1
        if config.get("use_calendar_exog", False):
            kwargs["futr_exog_list"] = list(_CALENDAR_NAMES)
        return getattr(nixtla_models, class_name)(**kwargs)
    raise ValueError(f"Unsupported native kind: {kind}")


def _config(kind: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    if kind not in (*NATIVE_KINDS, *OPTIONAL_NIXTLA_KINDS):
        raise ValueError(f"Unknown neural kind: {kind}")
    cfg.setdefault("context_length", 96)
    cfg.setdefault("seeds", list(DEFAULT_SEEDS))
    cfg.setdefault("loss", "mae")
    cfg.setdefault("target_baseline", "direct")
    cfg.setdefault("exog_columns", [])
    cfg.setdefault("exog_groups", [])
    cfg.setdefault("max_epochs", 100)
    cfg.setdefault("patience", 10)
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("learning_rate", 1e-3)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("quantiles", list(QUANTILES))
    cfg.setdefault("point_quantile", 0.5)
    cfg.setdefault("architecture_kwargs", {})
    cfg.setdefault("use_calendar_exog", kind == "tide")
    if not isinstance(cfg["architecture_kwargs"], Mapping):
        raise ValueError("architecture_kwargs must be a mapping")
    unsupported_exog_keys = {"futr_exog_list", "hist_exog_list", "stat_exog_list", "cat_exog_list"}
    if kind in OPTIONAL_NIXTLA_KINDS and unsupported_exog_keys & set(cfg["architecture_kwargs"]):
        raise ValueError("Nixtla architecture exogenous lists are managed by the causal adapter")
    if not 1 <= int(cfg["context_length"]) <= 8192:
        raise ValueError("context_length must be in 1..8192")
    if not 1 <= int(cfg["max_epochs"]) <= 10000 or not 1 <= int(cfg["patience"]) <= int(cfg["max_epochs"]):
        raise ValueError("Invalid max_epochs or patience")
    if int(cfg["batch_size"]) < 1 or float(cfg["learning_rate"]) <= 0:
        raise ValueError("Invalid batch_size or learning_rate")
    seeds = tuple(int(seed) for seed in cfg["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty unique list")
    cfg["seeds"] = list(seeds)
    if cfg["loss"] not in ("mae", "huber", "peak_weighted_mae", "quantile"):
        raise ValueError("Unsupported neural loss")
    if cfg["target_baseline"] not in ("direct", "weekly"):
        raise ValueError("target_baseline must be direct or weekly")
    if kind == "nlinear" and cfg["target_baseline"] != "direct":
        raise ValueError("NLinear's last-value skip requires a direct power target")
    if kind in OPTIONAL_NIXTLA_KINDS and cfg["exog_columns"]:
        raise ValueError("Nixtla models use known-future calendar exog; origin feature columns are unsupported")
    if kind in OPTIONAL_NIXTLA_KINDS and cfg["target_baseline"] != "direct":
        raise ValueError("Nixtla adapter currently supports direct power targets only")
    if kind in ("nbeats", "patchtst", "tsmixer", "itransformer") and cfg["use_calendar_exog"]:
        raise ValueError(f"{kind} forward does not consume a calendar exogenous tensor")
    if cfg["loss"] == "peak_weighted_mae" and float(cfg.get("peak_weight", 2)) <= 1:
        raise ValueError("peak_weighted_mae requires peak_weight > 1")
    if cfg["loss"] == "quantile":
        levels = tuple(float(q) for q in cfg["quantiles"])
        if not levels or any(not 0 < q < 1 for q in levels) or levels != tuple(sorted(set(levels))):
            raise ValueError("quantiles must be strictly increasing in (0, 1)")
        if float(cfg["point_quantile"]) not in levels:
            raise ValueError("point_quantile must be one of the fitted quantiles")
        cfg["quantiles"] = list(levels)
    if not isinstance(cfg["exog_columns"], (list, tuple)) or not isinstance(cfg["exog_groups"], (list, tuple)):
        raise ValueError("exog_columns and exog_groups must be sequences")
    cfg["exog_columns"] = list(cfg["exog_columns"])
    cfg["exog_groups"] = list(cfg["exog_groups"])
    cfg["kind"] = kind
    if kind in OPTIONAL_NIXTLA_KINDS and "horizon" not in cfg:
        raise ValueError("Nixtla candidates require config.horizon to construct the output head")
    # The architecture is constructed here to validate its parameter bounds.
    _model(kind, cfg, len(cfg["quantiles"]) if cfg["loss"] == "quantile" else 1,
           2 * len(cfg["exog_columns"]))
    return cfg


def _seed(seed: int, device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _history_guard(history: pd.DataFrame) -> None:
    if not isinstance(history.index, pd.DatetimeIndex) or history.empty:
        raise ValueError("History needs a nonempty DatetimeIndex")
    if history.index.max() >= SEALED_BOUNDARY:
        raise ValueError("Phase F neural adapter cannot read beyond the sealed development boundary")


def _horizon(context: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    summary = context.get("summary", {})
    horizon = int(summary.get("horizon_quarters", config.get("horizon", 0)))
    if horizon not in range(4, 17):
        raise ValueError("A single Phase C horizon 4..16 is required")
    if "horizon" in config and int(config["horizon"]) != horizon:
        raise ValueError("Configured horizon differs from context")
    return horizon


def _roles(context: Mapping[str, Any], horizon: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    fit = pd.DatetimeIndex(context["fit"], name="origin")
    stop = pd.DatetimeIndex(context["stop"], name="origin")
    if len(fit) == 0 or len(stop) == 0 or fit.has_duplicates or stop.has_duplicates:
        raise ValueError("Nonempty unique fit and stop origins are required")
    if not fit.is_monotonic_increasing or not stop.is_monotonic_increasing or fit.max() >= stop.min():
        raise ValueError("Fit and stop must be chronological and separate")
    target_times = pd.to_datetime(context["target_time"])
    expected_fit = fit + pd.Timedelta(minutes=15 * horizon)
    expected_stop = stop + pd.Timedelta(minutes=15 * horizon)
    if not pd.DatetimeIndex(target_times.loc[fit]).equals(expected_fit) or not pd.DatetimeIndex(target_times.loc[stop]).equals(expected_stop):
        raise ValueError("Target timestamps must match the fitted horizon")
    if expected_fit.max() >= stop.min():
        raise ValueError("Fit target labels overlap the stop interval")
    return fit, stop


def _baseline(history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
              baseline: str) -> np.ndarray:
    if baseline == "direct":
        return np.zeros(len(origins), dtype=float)
    target = origins + pd.Timedelta(minutes=15 * horizon)
    reference = target - pd.Timedelta(days=7)
    if (reference > origins).any():
        raise AssertionError("Weekly reference follows the origin")
    power = pd.to_numeric(history["power"], errors="coerce")
    for flag in ("time_repaired", "quality_bad"):
        if flag in history:
            power = power.mask(history[flag].fillna(True).astype(bool))
    return power.reindex(reference).to_numpy(dtype=float)


def _exog(history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
          tau: float, config: Mapping[str, Any]) -> np.ndarray:
    columns = config["exog_columns"]
    if not columns:
        return np.empty((len(origins), 0), dtype=float)
    features, provenance = build_features(history, origins, horizon, tau,
                                          groups=config["exog_groups"])
    missing = set(columns) - set(features)
    if missing:
        raise ValueError(f"Unknown or unselected exogenous features: {sorted(missing)}")
    if any((provenance[column] > origins).fillna(False).any() for column in columns):
        raise AssertionError("Exogenous feature reads after origin")
    return features.loc[:, columns].to_numpy(dtype=float)


def _raw_inputs(history: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
                tau: float, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values, mask = causal_sequences(history, origins, int(cfg["context_length"]))
    exog = _exog(history, origins, horizon, tau, cfg)
    baseline = _baseline(history, origins, horizon, str(cfg["target_baseline"]))
    return values, mask, exog, baseline


def _calendar_covariates(origins: pd.DatetimeIndex, context_length: int,
                         horizon: int) -> torch.Tensor:
    """Known calendar at each past and future slot, with no measured fields."""
    offsets = np.arange(1 - context_length, horizon + 1, dtype=np.int64)
    stamps = origins.to_numpy(dtype="datetime64[ns]")[:, None] + offsets[None, :] * np.timedelta64(15, "m")
    flat = pd.DatetimeIndex(stamps.reshape(-1))
    slot = flat.hour.to_numpy() * 4 + flat.minute.to_numpy() / 15
    dow = flat.dayofweek.to_numpy()
    holiday = flat.strftime("%Y-%m-%d").isin(HOLIDAYS_2021).astype(float)
    values = np.column_stack((np.sin(2 * np.pi * slot / 96), np.cos(2 * np.pi * slot / 96),
                              np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), holiday))
    return torch.from_numpy(np.ascontiguousarray(values.reshape(len(origins), len(offsets), 5), dtype=np.float32))


def _forward(model: nn.Module, kind: str, seq: torch.Tensor, mask: torch.Tensor,
             exog: torch.Tensor, future_calendar: torch.Tensor | None,
             horizon: int) -> torch.Tensor:
    if kind in NATIVE_KINDS:
        return model(seq, mask, exog)
    windows = {"insample_y": seq[:, :, None], "insample_mask": mask[:, :, None],
               "hist_exog": None, "futr_exog": future_calendar, "stat_exog": None}
    output = model(windows)
    if output.ndim != 3 or output.shape[1] != horizon:
        raise ValueError(f"{kind} returned an incompatible native forecast shape {tuple(output.shape)}")
    return output[:, horizon - 1, :]


def _scalers(seq_fit: np.ndarray, y_fit: np.ndarray, exog_fit: np.ndarray,
             kind: str, baseline: str) -> dict[str, Any]:
    finite = seq_fit[np.isfinite(seq_fit)]
    if len(finite) == 0:
        raise ValueError("Fit windows have no observed power")
    mean = float(finite.mean())
    scale = float(finite.std()) or 1.0
    # NLinear's last-value skip requires power and target on the same scale.
    if kind == "nlinear" and baseline == "direct":
        target_mean, target_scale = mean, scale
    else:
        target_mean = float(y_fit.mean())
        target_scale = float(y_fit.std()) or 1.0
    if exog_fit.shape[1]:
        counts = np.isfinite(exog_fit).sum(axis=0)
        ex_mean = np.divide(np.nansum(exog_fit, axis=0), counts,
                            out=np.zeros(exog_fit.shape[1]), where=counts > 0)
        centered = np.where(np.isfinite(exog_fit), exog_fit - ex_mean, 0)
        ex_scale = np.sqrt(np.divide((centered ** 2).sum(axis=0), counts,
                                      out=np.ones(exog_fit.shape[1]), where=counts > 0))
        ex_scale = np.where(ex_scale > 1e-12, ex_scale, 1.0)
    else:
        ex_mean, ex_scale = np.empty(0), np.empty(0)
    return {"seq_mean": mean, "seq_scale": scale,
            "target_mean": target_mean, "target_scale": target_scale,
            "exog_mean": ex_mean.tolist(), "exog_scale": ex_scale.tolist()}


def _transform(seq: np.ndarray, mask: np.ndarray, exog: np.ndarray,
               norm: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Causal forward fill within each window, then fit-only mean for left pad.
    n, length = seq.shape
    observed_at = np.where(mask, np.arange(length)[None, :], -1)
    last_at = np.maximum.accumulate(observed_at, axis=1)
    filled = np.where(last_at >= 0, np.take_along_axis(seq, np.maximum(last_at, 0), axis=1),
                      norm["seq_mean"])
    filled = np.where(np.isfinite(filled), filled, norm["seq_mean"])
    value = ((filled - norm["seq_mean"]) / norm["seq_scale"]).astype(np.float32)
    means = np.asarray(norm["exog_mean"], dtype=float)
    scales = np.asarray(norm["exog_scale"], dtype=float)
    finite_ex = np.isfinite(exog)
    ex_scaled = np.where(finite_ex, (exog - means) / scales, 0) if exog.shape[1] else exog
    ex_input = np.concatenate((ex_scaled, finite_ex.astype(float)), axis=1).astype(np.float32)
    return (torch.from_numpy(np.ascontiguousarray(value)),
            torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(ex_input)))


def _point(output: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    if cfg["loss"] != "quantile":
        return output[:, 0]
    position = cfg["quantiles"].index(cfg["point_quantile"])
    return torch.sort(output, dim=1).values[:, position]


def _loss(output: torch.Tensor, target: torch.Tensor, raw_target: torch.Tensor,
          tau: float, cfg: Mapping[str, Any]) -> torch.Tensor:
    if cfg["loss"] == "quantile":
        levels = torch.tensor(cfg["quantiles"], device=output.device, dtype=output.dtype)
        residual = target[:, None] - output
        return torch.maximum(levels * residual, (levels - 1) * residual).mean()
    point = output[:, 0]
    if cfg["loss"] == "huber":
        return F.huber_loss(point, target, delta=float(cfg.get("huber_delta", 1.0)))
    error = torch.abs(point - target)
    if cfg["loss"] == "peak_weighted_mae":
        weights = torch.where(raw_target > tau, float(cfg.get("peak_weight", 2.0)), 1.0)
        error = error * weights
    return error.mean()


def _checkpoint(cfg: Mapping[str, Any], seed: int) -> Path | None:
    if not cfg.get("checkpoint_dir"):
        return None
    root = Path(__file__).resolve().parents[2] / "outputs" / "phase_f" / "models"
    path = Path(cfg["checkpoint_dir"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("checkpoint_dir must be inside outputs/phase_f/models")
    run_id = str(cfg.get("run_id", "neural"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, numbers, dash, underscore")
    return path / f"{run_id}_seed_{seed}.pt"


def _fingerprint(fit: pd.DatetimeIndex, stop: pd.DatetimeIndex,
                 fit_arrays: tuple[np.ndarray, ...], stop_arrays: tuple[np.ndarray, ...],
                 cfg: Mapping[str, Any], horizon: int) -> str:
    digest = hashlib.sha256()
    stable_cfg = {k: v for k, v in cfg.items() if k not in ("checkpoint_dir", "run_id")}
    digest.update(json.dumps({"config": stable_cfg, "horizon": horizon}, sort_keys=True, default=str).encode())
    for index in (fit, stop):
        digest.update(index.asi8.astype("<i8").tobytes())
    for array in (*fit_arrays, *stop_arrays):
        raw = np.asarray(array)
        digest.update(str(raw.shape).encode())
        digest.update(np.ascontiguousarray(raw).tobytes())
    return digest.hexdigest()


def _train_seed(seed: int, kind: str, cfg: Mapping[str, Any], norm: Mapping[str, Any],
                fit_x: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                stop_x: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                fit_calendar: torch.Tensor | None, stop_calendar: torch.Tensor | None,
                fit_y: np.ndarray, stop_y: np.ndarray, fit_raw_y: np.ndarray,
                stop_baseline: np.ndarray, tau: float) -> dict[str, Any]:
    device = _device(str(cfg["device"]))
    _seed(seed, device)
    output_dim = len(cfg["quantiles"]) if cfg["loss"] == "quantile" else 1
    model_cfg = {**cfg, "model_seed": seed}
    model = _model(kind, model_cfg, output_dim, fit_x[2].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    fit_target = torch.tensor((fit_y - norm["target_mean"]) / norm["target_scale"], dtype=torch.float32)
    raw_target = torch.tensor(fit_raw_y, dtype=torch.float32)
    stop_target = torch.tensor(stop_y + stop_baseline, dtype=torch.float32)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    best, epoch_best, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(fit_y), generator=rng)
        loss_sum = 0.0
        for offset in range(0, len(order), int(cfg["batch_size"])):
            selected = order[offset:offset + int(cfg["batch_size"])]
            seq, mask, exog = (arr[selected].to(device) for arr in fit_x)
            calendar = fit_calendar[selected].to(device) if fit_calendar is not None else None
            prediction = _forward(model, kind, seq, mask, exog, calendar, int(cfg["horizon"]))
            loss = _loss(prediction, fit_target[selected].to(device),
                         raw_target[selected].to(device), tau, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(selected)
        model.eval()
        absolute = 0.0
        with torch.inference_mode():
            for offset in range(0, len(stop_y), int(cfg["batch_size"])):
                end = offset + int(cfg["batch_size"])
                seq, mask, exog = (arr[offset:end].to(device) for arr in stop_x)
                calendar = stop_calendar[offset:end].to(device) if stop_calendar is not None else None
                prediction = _forward(model, kind, seq, mask, exog, calendar, int(cfg["horizon"]))
                raw = _point(prediction, cfg) * norm["target_scale"] + norm["target_mean"]
                raw = raw + torch.tensor(stop_baseline[offset:end], device=device, dtype=torch.float32)
                absolute += torch.abs(raw - stop_target[offset:end].to(device)).sum().item()
        stop_mae = absolute / len(stop_y)
        history.append({"epoch": epoch, "fit_loss": loss_sum / len(fit_y), "stop_mae": stop_mae})
        if stop_mae < best - 1e-12:
            best, epoch_best, stale = stop_mae, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise AssertionError("Training did not produce a checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"seed": seed, "state_dict": best_state, "best_epoch": epoch_best,
            "stop_mae": best, "epochs_ran": len(history), "history": history,
            "train_seconds": time.perf_counter() - start}


def fit_model(kind: str, history: pd.DataFrame, context: Mapping[str, Any],
              config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit one genuine neural architecture for one Phase C horizon/fold.

    Defaults to five seeds. Passing ``seeds=[42]`` is for targeted smoke tests;
    trial registries must record the actual seed count. Checkpoints, if enabled,
    are immutable per data/config fingerprint and stored under Phase F outputs.
    """
    kind = kind.lower()
    config_with_horizon = dict(config or {})
    config_with_horizon.setdefault("horizon", _horizon(context, config_with_horizon))
    cfg = _config(kind, config_with_horizon)
    _history_guard(history)
    horizon = _horizon(context, cfg)
    fit, stop = _roles(context, horizon)
    tau = float(context["tau"])
    if not np.isfinite(tau):
        raise ValueError("A finite fit-only peak threshold is required")
    fit_seq, fit_mask, fit_exog, fit_base = _raw_inputs(history, fit, horizon, tau, cfg)
    stop_seq, stop_mask, stop_exog, stop_base = _raw_inputs(history, stop, horizon, tau, cfg)
    fit_raw = pd.to_numeric(context["y"].loc[fit], errors="coerce").to_numpy(dtype=float)
    stop_raw = pd.to_numeric(context["y"].loc[stop], errors="coerce").to_numpy(dtype=float)
    fit_valid = np.isfinite(fit_raw) & np.isfinite(fit_base)
    stop_valid = np.isfinite(stop_raw) & np.isfinite(stop_base)
    if not fit_valid.any() or not stop_valid.any():
        raise ValueError("No finite fit or stop labels under the selected target baseline")
    fit_seq, fit_mask, fit_exog, fit_base, fit_raw = (
        arr[fit_valid] for arr in (fit_seq, fit_mask, fit_exog, fit_base, fit_raw)
    )
    stop_seq, stop_mask, stop_exog, stop_base, stop_raw = (
        arr[stop_valid] for arr in (stop_seq, stop_mask, stop_exog, stop_base, stop_raw)
    )
    fit_target, stop_target = fit_raw - fit_base, stop_raw - stop_base
    norm = _scalers(fit_seq, fit_target, fit_exog, kind, cfg["target_baseline"])
    fit_x = _transform(fit_seq, fit_mask, fit_exog, norm)
    stop_x = _transform(stop_seq, stop_mask, stop_exog, norm)
    fit_calendar = stop_calendar = None
    if kind in OPTIONAL_NIXTLA_KINDS and cfg["use_calendar_exog"]:
        fit_calendar = _calendar_covariates(fit[fit_valid], int(cfg["context_length"]), horizon)
        stop_calendar = _calendar_covariates(stop[stop_valid], int(cfg["context_length"]), horizon)
    fingerprint = _fingerprint(fit[fit_valid], stop[stop_valid],
                               (fit_seq, fit_mask, fit_exog, fit_raw, fit_base),
                               (stop_seq, stop_mask, stop_exog, stop_raw, stop_base), cfg, horizon)
    results = []
    for seed in cfg["seeds"]:
        path = _checkpoint(cfg, seed)
        if path is not None and path.exists():
            result = torch.load(path, map_location="cpu", weights_only=True)
            if result.get("fingerprint") != fingerprint:
                raise ValueError(f"Stale or conflicting neural checkpoint: {path}")
            results.append(result["result"])
            continue
        result = _train_seed(seed, kind, cfg, norm, fit_x, stop_x,
                             fit_calendar, stop_calendar, fit_target,
                             stop_target, fit_raw, stop_base, tau)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            torch.save({"fingerprint": fingerprint, "result": result}, temporary)
            temporary.replace(path)
        results.append(result)
    return {"kind": kind, "horizon": horizon, "config": cfg, "norm": norm,
            "tau": tau, "seeds": results, "fit_count": int(fit_valid.sum()),
            "stop_count": int(stop_valid.sum()), "fit_last_target": str((fit[fit_valid] + horizon * pd.Timedelta(minutes=15)).max()),
            "stop_first": str(stop[stop_valid].min()), "fingerprint": fingerprint,
            "fit_observed_context_min": int(fit_mask.sum(axis=1).min()),
            "fit_observed_context_max": int(fit_mask.sum(axis=1).max()),
            "development_only": True, "fit_parameters_only": True}


def predict_model(bundle: Mapping[str, Any], history: pd.DataFrame,
                  origins: pd.DatetimeIndex, horizon: int,
                  *, return_quantiles: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict without labels; output keys and order match supplied origins."""
    if bundle.get("development_only") is not True or int(bundle["horizon"]) != horizon:
        raise ValueError("Bundle is not a development fit for this horizon")
    _history_guard(history)
    origins = pd.DatetimeIndex(origins, name="origin")
    cfg, norm = bundle["config"], bundle["norm"]
    if return_quantiles and cfg["loss"] != "quantile":
        raise ValueError("This bundle has no quantile head")
    seq, mask, exog, baseline = _raw_inputs(history, origins, horizon, float(bundle["tau"]), cfg)
    transformed = _transform(seq, mask, exog, norm)
    calendar = (_calendar_covariates(origins, int(cfg["context_length"]), horizon)
                if bundle["kind"] in OPTIONAL_NIXTLA_KINDS and cfg["use_calendar_exog"] else None)
    device = _device(str(cfg["device"]))
    output_dim = len(cfg["quantiles"]) if cfg["loss"] == "quantile" else 1
    ensemble = []
    with torch.inference_mode():
        for seed_result in bundle["seeds"]:
            model = _model(str(bundle["kind"]), {**cfg, "model_seed": seed_result["seed"]},
                           output_dim, transformed[2].shape[1]).to(device)
            model.load_state_dict(seed_result["state_dict"])
            model.eval()
            pieces = []
            for offset in range(0, len(origins), int(cfg["batch_size"])):
                end = offset + int(cfg["batch_size"])
                batch_seq, batch_mask, batch_exog = (arr[offset:end].to(device) for arr in transformed)
                batch_calendar = calendar[offset:end].to(device) if calendar is not None else None
                output = _forward(model, str(bundle["kind"]), batch_seq, batch_mask,
                                  batch_exog, batch_calendar, horizon)
                if cfg["loss"] == "quantile":
                    output = torch.sort(output, dim=1).values
                pieces.append(output.cpu().numpy())
            ensemble.append(np.concatenate(pieces, axis=0) if pieces else np.empty((0, output_dim)))
    combined = np.mean(ensemble, axis=0) * norm["target_scale"] + norm["target_mean"]
    combined = combined + baseline[:, None]
    if cfg["loss"] == "quantile":
        quantiles = {str(q): combined[:, i].astype(float) for i, q in enumerate(cfg["quantiles"])}
        point = quantiles[str(cfg["point_quantile"])]
        return (point, quantiles) if return_quantiles else point
    return combined[:, 0].astype(float)
