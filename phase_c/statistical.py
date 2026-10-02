"""Fixed statistical benchmarks for Phase C development folds.

Both parameter fits use only the supplied fit timestamps. Predictions may use
observations through each forecast origin, but never observations after it.

B3 intentionally differs from the historical ``research_p4/stat_models.py``
daily-refit MSTL. That implementation fits a new 28-day model every day. Phase
C reserves calibration and score data from parameter fitting, so this wrapper
fits one 28-day MSTL at the fold's fit endpoint and projects its base path.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize


STEP = pd.Timedelta(minutes=15)
WEEK = 672
DAY = 96
MSTL_WINDOW = 28 * DAY


def _grid(power: pd.Series) -> pd.Series:
    """Keep gaps explicit; position 672 must mean exactly one week earlier."""
    if not isinstance(power, pd.Series) or not isinstance(power.index, pd.DatetimeIndex):
        raise TypeError("power must be a Series with a DatetimeIndex")
    if power.empty or not power.index.is_unique or not power.index.is_monotonic_increasing:
        raise ValueError("power must have a nonempty, unique, sorted index")
    if (power.index.asi8 % STEP.value != 0).any():
        raise ValueError("power timestamps must be on the 15-minute grid")
    index = pd.date_range(power.index[0], power.index[-1], freq=STEP)
    return pd.to_numeric(power, errors="coerce").reindex(index).astype(float)


def _fit_index(grid: pd.Series, fit: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(fit, pd.DatetimeIndex) or fit.empty:
        raise ValueError("fit must be a nonempty DatetimeIndex")
    if not fit.is_unique or not fit.is_monotonic_increasing:
        raise ValueError("fit must be unique and sorted")
    if fit.tz != grid.index.tz or not fit.isin(grid.index).all():
        raise ValueError("fit timestamps must belong to the power grid")
    return fit


def _origins(grid: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    if not isinstance(origins, pd.DatetimeIndex):
        raise TypeError("origins must be a DatetimeIndex")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive number of 15-minute steps")
    if origins.tz != grid.index.tz or not origins.isin(grid.index).all():
        raise ValueError("origins must belong to the power grid")
    return origins


def _transition(mean: float, variance: float, phi: float, q: float, steps: int) -> tuple[float, float]:
    """Advance over missing observations using exact AR(1) step powers."""
    factor = phi ** steps
    process_variance = q * (1.0 - factor * factor) / (1.0 - phi * phi)
    return factor * mean, factor * factor * variance + process_variance


def _filter_deviations(
    deviations: np.ndarray,
    selected: np.ndarray,
    phi: float,
    q: float,
    r: float,
    *,
    save_states: bool = False,
) -> tuple[float, int, np.ndarray | None]:
    """Gaussian innovations; unselected slots never contribute likelihood."""
    mean, variance = 0.0, q / (1.0 - phi * phi)
    nll, n_updates, last = 0.0, 0, -1
    states = np.full(len(deviations), np.nan) if save_states else None
    # The likelihood traverses only observed fit slots. A skipped span advances
    # the state and its process variance by the exact number of 15-minute steps.
    positions = range(len(deviations)) if save_states else np.flatnonzero(selected & np.isfinite(deviations))
    for pos in positions:
        value = deviations[pos]
        if last >= 0:
            mean, variance = _transition(mean, variance, phi, q, pos - last)
        last = pos
        if selected[pos] and np.isfinite(value):
            innovation_variance = variance + r
            innovation = value - mean
            nll += 0.5 * (np.log(2.0 * np.pi * innovation_variance) + innovation * innovation / innovation_variance)
            gain = variance / innovation_variance
            mean += gain * innovation
            variance *= 1.0 - gain
            n_updates += 1
        if states is not None:
            states[pos] = mean
    return float(nll), n_updates, states


def fit_kalman(power: pd.Series, fit: pd.DatetimeIndex) -> dict:
    """Fit B5's fixed weekly-deviation AR(1) state-space specification.

    The seasonal reference at t is the observed power at t-672. Only deviations
    whose timestamp is in ``fit`` enter the Gaussian likelihood. ``q`` and
    ``r`` are positive process and observation variances, respectively.
    """
    started = perf_counter()
    grid = _grid(power)
    fit = _fit_index(grid, fit)
    grid = grid.loc[: fit.max()]
    deviations = (grid - grid.shift(WEEK)).to_numpy(dtype=float)
    selected = grid.index.isin(fit) & np.isfinite(deviations)
    n_updates = int(selected.sum())
    if n_updates < 20:
        raise ValueError("at least 20 finite fit deviations are required")
    sample = deviations[selected]
    scale = max(float(np.var(sample)), 1e-8)
    adjacent = selected[1:] & selected[:-1]
    if int(adjacent.sum()) >= 2:
        previous, current = deviations[:-1][adjacent], deviations[1:][adjacent]
        correlation = np.corrcoef(previous, current)[0, 1]
        initial_phi = float(np.clip(correlation, -0.9, 0.9)) if np.isfinite(correlation) else 0.0
    else:
        initial_phi = 0.0
    log_scale = float(np.log(scale))

    def objective(parameters: np.ndarray) -> float:
        phi, log_q, log_r = parameters
        value, _, _ = _filter_deviations(deviations, selected, float(phi), float(np.exp(log_q)), float(np.exp(log_r)))
        return value

    result = minimize(
        objective,
        np.array([initial_phi, np.log(scale * 0.3), np.log(scale * 0.7)]),
        method="L-BFGS-B",
        bounds=[(-0.99, 0.99), (log_scale - 18, log_scale + 7), (log_scale - 18, log_scale + 7)],
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Kalman likelihood optimization failed: {result.message}")
    phi, log_q, log_r = result.x
    return {
        "phi": float(phi),
        "q": float(np.exp(log_q)),
        "r": float(np.exp(log_r)),
        "fit_nll": float(result.fun),
        "n_fit_updates": n_updates,
        "optimizer_success": True,
        "optimizer_message": str(result.message),
        "seconds": perf_counter() - started,
        "development_only": True,
        "fit_parameters_only": True,
    }


def predict_kalman(bundle: dict, power: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Filter through each origin and add its known prior-week target anchor."""
    grid = _grid(power)
    origins = _origins(grid, origins, horizon)
    if horizon > WEEK:
        raise ValueError("weekly target anchor would be after the forecast origin")
    phi, q, r = (float(bundle[key]) for key in ("phi", "q", "r"))
    if not -0.99 <= phi <= 0.99 or q <= 0 or r <= 0:
        raise ValueError("invalid fitted Kalman parameters")
    deviations = (grid - grid.shift(WEEK)).to_numpy(dtype=float)
    last_position = int(grid.index.get_indexer(origins).max()) if len(origins) else -1
    if last_position < 0:
        return np.empty(0, dtype=float)
    selected = np.isfinite(deviations[: last_position + 1])
    _, _, states = _filter_deviations(deviations[: last_position + 1], selected, phi, q, r, save_states=True)
    positions = grid.index.get_indexer(origins)
    anchor_positions = positions + horizon - WEEK
    anchors = np.full(len(origins), np.nan)
    available = anchor_positions >= 0
    anchors[available] = grid.to_numpy(dtype=float)[anchor_positions[available]]
    return anchors + phi ** horizon * states[positions]


def fit_mstl(power: pd.Series, fit: pd.DatetimeIndex, forecast_end: pd.Timestamp) -> dict:
    """Fit B3 once on the final 28 fit days and project its fixed base path.

    Gaps are filled only from earlier fit observations. Fitted in-window trend
    plus seasonal components supply residuals for phi; out-of-window baseline
    values come solely from the one fitted model's forward projection.
    """
    from statsforecast.models import AutoETS, MSTL

    started = perf_counter()
    grid = _grid(power)
    fit = _fit_index(grid, fit)
    end = fit.max()
    forecast_end = pd.Timestamp(forecast_end)
    if forecast_end.tz != end.tz or forecast_end < end or (forecast_end - end) % STEP != pd.Timedelta(0):
        raise ValueError("forecast_end must be on the power grid at or after fit end")
    fit_only = grid.where(grid.index.isin(fit))
    filled = fit_only.loc[:end].ffill()
    window = filled.iloc[-MSTL_WINDOW:]
    if len(window) != MSTL_WINDOW or not np.isfinite(window.to_numpy(dtype=float)).all():
        raise ValueError("MSTL needs 28 complete days after fit-only forward fill")
    model = MSTL(season_length=[DAY, WEEK], trend_forecaster=AutoETS(model="ZZN"))
    model.fit(window.to_numpy(dtype=float))
    fitted_components = model.model_
    seasonal_columns = [column for column in fitted_components.columns if column.startswith("seasonal")]
    if not seasonal_columns:
        raise RuntimeError("MSTL did not return seasonal fitted components")
    fitted_base = fitted_components["trend"].to_numpy(dtype=float)
    fitted_base = fitted_base + fitted_components[seasonal_columns].sum(axis=1).to_numpy(dtype=float)
    n_future = int((forecast_end - end) / STEP)
    if n_future:
        future = np.asarray(model.predict(h=n_future)["mean"], dtype=float)
        if len(future) != n_future or not np.isfinite(future).all():
            raise RuntimeError("MSTL produced an incomplete future base path")
    else:
        future = np.empty(0, dtype=float)
    base_path = pd.Series(
        np.concatenate((fitted_base, future)),
        index=pd.date_range(window.index[0], forecast_end, freq=STEP),
        name="mstl_base",
    )
    residual = (grid.reindex(window.index) - base_path.reindex(window.index)).where(window.index.isin(fit))
    lagged = residual.shift(1)
    valid = residual.notna() & lagged.notna()
    n_resid = int(valid.sum())
    if n_resid > 100:
        phi = np.corrcoef(residual[valid], lagged[valid])[0, 1]
        phi = float(np.clip(phi, 0.0, 0.99)) if np.isfinite(phi) else 0.0
    else:
        phi = 0.0
    return {
        "base_path": base_path,
        "phi": phi,
        "n_resid": n_resid,
        "fit_end": end,
        "fit_window_start": window.index[0],
        "fit_forward_filled": int(fit_only.reindex(window.index).isna().sum()),
        "seconds": perf_counter() - started,
        "development_only": True,
        "fit_parameters_only": True,
    }


def predict_mstl(bundle: dict, power: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Apply the audited AR(1) origin correction to the fixed B3 base path."""
    grid = _grid(power)
    origins = _origins(grid, origins, horizon)
    base = bundle["base_path"]
    if not isinstance(base, pd.Series) or not isinstance(base.index, pd.DatetimeIndex):
        raise TypeError("bundle.base_path must be a datetime-indexed Series")
    targets = origins + horizon * STEP
    if not origins.isin(base.index).all() or not targets.isin(base.index).all():
        raise ValueError("MSTL base path does not cover all origins and targets")
    origin_base = base.reindex(origins).to_numpy(dtype=float)
    target_base = base.reindex(targets).to_numpy(dtype=float)
    observed = grid.reindex(origins).to_numpy(dtype=float)
    return target_base + float(bundle["phi"]) ** horizon * (observed - origin_base)
