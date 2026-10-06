"""Phase F statistical candidates with fit-only parameters and causal updates.

``context['fit']`` is the exact Phase C fit-origin index.  A fit may inspect
earlier observations for lag construction, but every estimated parameter and
seasonal template is based on the selected fit observations alone.  During
prediction observations through each origin are available; later values are
never used, including when several origins are passed at once.

Native statsforecast models are labelled separately from our state filters.
Their ``forward`` method applies fitted parameters to observed prefixes; it
does not optimize parameters on stop/calibration/score observations.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from phase_c.statistical import DAY, STEP, WEEK, _filter_deviations, _grid, _origins, fit_kalman, predict_kalman
from src.holidays import HOLIDAYS_2021


HOLIDAYS = frozenset(pd.Timestamp(date) for date in HOLIDAYS_2021)


def configurations() -> list[dict]:
    """Named, auditable initial grid; later search can add explicit configs."""
    rows = []
    for k in (2, 3, 4):
        for statistic in ("mean", "median"):
            rows.append({"id": f"F0-3-k{k}-{statistic}", "family": "F0-3", "kind": "seasonal_kweek",
                         "params": {"weeks": k, "statistic": statistic}})
    rows.append({"id": "F0-3-daytype", "family": "F0-3", "kind": "daytype_profile",
                 "params": {"statistic": "median", "level_span": DAY}})
    rows.append({"id": "F0-2-fixed", "family": "F0-2", "kind": "mstl_stationary", "params": {}})
    rows.extend([
        {"id": "F4-1-profile-k2", "family": "F4-1", "kind": "kalman_profile", "params": {"weeks": 2}},
        {"id": "F4-1-hour-noise", "family": "F4-1", "kind": "kalman_hour_noise", "params": {}},
        {"id": "F4-1-daytype-phi", "family": "F4-1", "kind": "kalman_daytype_phi", "params": {}},
        {"id": "F4-1-two-state", "family": "F4-1", "kind": "kalman_two_state", "params": {}},
        {"id": "F4-1-robust", "family": "F4-1", "kind": "kalman_robust", "params": {"nu": 4.0}},
        {"id": "F4-2-ets-daily", "family": "F4-2", "kind": "autoets", "params": {"season_length": DAY}},
        {"id": "F4-2-dshw", "family": "F4-2", "kind": "dshw", "params": {}},
        {"id": "F4-2-tbats-daily-weekly", "family": "F4-2", "kind": "tbats", "params": {}},
        {"id": "F4-2-arima-fourier", "family": "F4-2", "kind": "autoarima_fourier", "params": {"harmonics": (2, 2)}},
        {"id": "F4-4-theta", "family": "F4-4", "kind": "autotheta", "params": {"season_length": DAY}},
    ])
    return rows


def _power(history: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(history, pd.DataFrame):
        if "power" not in history.columns:
            raise ValueError("history requires clean power")
        return _grid(history["power"])
    return _grid(history)


def _fit(power: pd.Series, context: Mapping) -> pd.DatetimeIndex:
    if "fit" not in context:
        raise ValueError("context requires fit origins")
    fit = pd.DatetimeIndex(context["fit"])
    if fit.empty or not fit.is_unique or not fit.is_monotonic_increasing or not fit.isin(power.index).all():
        raise ValueError("fit origins must be nonempty, unique, chronological and in history")
    if fit.tz != power.index.tz:
        raise ValueError("fit timezone does not match history")
    return fit


def _masked_fit(power: pd.Series, fit: pd.DatetimeIndex) -> pd.Series:
    return power.loc[:fit.max()].where(power.loc[:fit.max()].index.isin(fit))


def _daytype(index: pd.DatetimeIndex) -> np.ndarray:
    date = index.normalize()
    return np.where(date.isin(list(HOLIDAYS)), 3,
                    np.where(index.dayofweek == 5, 1, np.where(index.dayofweek == 6, 2, 0))).astype(int)


def _slot(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 4 + index.minute // 15).to_numpy(dtype=int)


def _weekly_reference(values: np.ndarray, weeks: int, statistic: str) -> np.ndarray:
    n = len(values)
    anchors = np.full((weeks, n), np.nan)
    for k in range(1, weeks + 1):
        anchors[k - 1, k * WEEK:] = values[:-k * WEEK]
    valid = np.isfinite(anchors).all(axis=0)
    out = np.full(n, np.nan)
    if np.any(valid):
        if statistic == "median":
            out[valid] = np.median(anchors[:, valid], axis=0)
        elif statistic == "mean":
            out[valid] = np.mean(anchors[:, valid], axis=0)
        else:
            raise ValueError("statistic must be mean or median")
    return out


def _profile_fit(power: pd.Series, fit: pd.DatetimeIndex, statistic: str) -> np.ndarray:
    index = fit
    y = power.reindex(index).to_numpy(dtype=float)
    labels = _daytype(index) * DAY + _slot(index)
    profile = np.full(4 * DAY, np.nan)
    for category in range(4 * DAY):
        sample = y[(labels == category) & np.isfinite(y)]
        if len(sample):
            profile[category] = float(np.median(sample) if statistic == "median" else np.mean(sample))
    # A category absent from fit has no fitted value; prediction returns NaN.
    return profile


def _profile_values(profile: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    return profile[_daytype(index) * DAY + _slot(index)]


def _fit_ar1(deviation: np.ndarray, selected: np.ndarray) -> dict:
    sample = deviation[selected & np.isfinite(deviation)]
    if len(sample) < 20:
        raise ValueError("at least 20 finite fit deviations required")
    scale = max(float(np.var(sample)), 1e-8)
    adjacent = selected[1:] & selected[:-1] & np.isfinite(deviation[1:]) & np.isfinite(deviation[:-1])
    initial = float(np.corrcoef(deviation[1:][adjacent], deviation[:-1][adjacent])[0, 1]) if adjacent.sum() > 2 else 0.0
    if not np.isfinite(initial):
        initial = 0.0
    lower = np.log(scale) - 18
    upper = np.log(scale) + 7

    def objective(par: np.ndarray) -> float:
        value, _, _ = _filter_deviations(deviation, selected, float(par[0]), float(np.exp(par[1])), float(np.exp(par[2])))
        return value

    result = minimize(objective, np.array([np.clip(initial, -.9, .9), np.log(scale * .3), np.log(scale * .7)]),
                      method="L-BFGS-B", bounds=[(-.99, .99), (lower, upper), (lower, upper)])
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"AR1 likelihood optimization failed: {result.message}")
    return {"phi": float(result.x[0]), "q": float(np.exp(result.x[1])), "r": float(np.exp(result.x[2])),
            "fit_nll": float(result.fun), "n_fit_updates": int(len(sample))}


def _seasonal_fit(power: pd.Series, fit: pd.DatetimeIndex, params: Mapping) -> dict:
    weeks = int(params.get("weeks", 2))
    statistic = str(params.get("statistic", "median"))
    if weeks not in (2, 3, 4):
        raise ValueError("weeks must be 2, 3 or 4")
    if statistic not in ("mean", "median"):
        raise ValueError("invalid statistic")
    return {"weeks": weeks, "statistic": statistic, "n_fit_observations": int(np.isfinite(power.reindex(fit)).sum())}


def _level_state(observed: np.ndarray, seasonal: np.ndarray, alpha: float, *, save: bool = False):
    level = np.nan
    states = np.full(len(observed), np.nan) if save else None
    sse = 0.0
    n = 0
    for pos, (y, season) in enumerate(zip(observed, seasonal)):
        if np.isfinite(y) and np.isfinite(season):
            residual = y - season
            if np.isfinite(level):
                sse += (residual - level) ** 2
                n += 1
                level += alpha * (residual - level)
            else:
                level = residual
        if save:
            states[pos] = level
    return sse / max(1, n), states


def _mstl_fit(power: pd.Series, fit: pd.DatetimeIndex) -> dict:
    """Keep fitted periodic components; never extrapolate AutoETS trend."""
    from statsforecast.models import AutoETS, MSTL

    fit_only = _masked_fit(power, fit).ffill()
    window = fit_only.iloc[-28 * DAY:]
    if len(window) != 28 * DAY or not np.isfinite(window.to_numpy(dtype=float)).all():
        raise ValueError("MSTL needs 28 complete fit-only days")
    model = MSTL(season_length=[DAY, WEEK], trend_forecaster=AutoETS(model="ZZN"))
    model.fit(window.to_numpy(dtype=float))
    components = model.model_
    seasonal_columns = [name for name in components.columns if name.startswith("seasonal")]
    if not seasonal_columns:
        raise RuntimeError("MSTL did not expose seasonal components")
    raw = components[seasonal_columns].sum(axis=1).to_numpy(dtype=float)
    if len(raw) != len(window) or not np.isfinite(raw).all():
        raise RuntimeError("MSTL seasonal decomposition incomplete")
    # Last seven days retain the calendar phase learned in the fit window.
    weekly = raw[-WEEK:].copy()
    weekly -= weekly.mean()
    fit_end = fit.max()
    index = power.loc[:fit_end].index
    pos = (np.arange(len(index)) - (len(index) - 1 - WEEK + 1)) % WEEK
    seasonal = weekly[pos]
    selected = power.loc[:fit_end].where(index.isin(fit)).to_numpy(dtype=float)
    alpha_grid = (.005, .02, .08, .2, .5)
    alpha = min(alpha_grid, key=lambda x: _level_state(selected, seasonal, x)[0])
    return {"weekly": weekly, "fit_end": fit_end, "alpha": alpha,
            "fit_window_start": window.index[0], "fit_forward_filled": int(_masked_fit(power, fit).reindex(window.index).isna().sum()),
            "n_fit_observations": int(np.isfinite(selected).sum()), "seasonal_columns": seasonal_columns}


def _fourier(index: pd.DatetimeIndex, start: pd.Timestamp, harmonics: tuple[int, int]) -> np.ndarray:
    slots = np.asarray((index - start) / STEP, dtype=float)
    columns = []
    for period, n in zip((DAY, WEEK), harmonics):
        for harmonic in range(1, int(n) + 1):
            phase = 2 * np.pi * harmonic * slots / period
            columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns) if columns else np.empty((len(index), 0))


def _native_fit(power: pd.Series, fit: pd.DatetimeIndex, kind: str, params: Mapping) -> dict:
    try:
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, TBATS
    except ImportError as exc:
        raise RuntimeError("statsforecast is required for native statistical models") from exc
    fit_only = _masked_fit(power, fit)
    window = fit_only.loc[fit.min():fit.max()].ffill()
    if not np.isfinite(window.to_numpy(dtype=float)).all():
        raise ValueError("native model needs a finite fit-only training window")
    y = window.to_numpy(dtype=float)
    if len(y) < 2 * WEEK:
        raise ValueError("native model needs at least two fit weeks")
    x = None
    if kind == "autoets":
        model = AutoETS(season_length=int(params.get("season_length", DAY)), model="ZZZ")
    elif kind == "tbats":
        model = TBATS(season_length=[DAY, WEEK], use_boxcox=False, use_arma_errors=False)
    elif kind == "autotheta":
        model = AutoTheta(season_length=int(params.get("season_length", DAY)), decomposition_type="additive")
    elif kind == "autoarima_fourier":
        harmonics = tuple(params.get("harmonics", (2, 2)))
        if len(harmonics) != 2 or not all(0 <= int(k) <= 4 for k in harmonics):
            raise ValueError("Fourier harmonics must contain two values 0..4")
        x = _fourier(window.index, window.index[0], harmonics)
        model = AutoARIMA(season_length=1, seasonal=False, max_p=3, max_q=3, max_order=4,
                          stepwise=True, nmodels=25)
    else:
        raise ValueError(f"unknown native model {kind}")
    model.fit(y, X=x)
    return {"model": model, "fit_start": window.index[0], "fit_end": fit.max(),
            "fit_n": int(np.isfinite(fit_only).sum()), "harmonics": tuple(params.get("harmonics", (2, 2))) if x is not None else None,
            "state_update": "native_forward" if hasattr(model, "forward") else "frozen_projection"}


def fit_model(kind: str, history: pd.DataFrame | pd.Series, context: Mapping, config: Mapping | None = None) -> dict:
    """Fit a named candidate using only explicitly selected fit observations."""
    power = _power(history)
    fit = _fit(power, context)
    params = dict(config or {})
    started = perf_counter()
    if kind == "seasonal_kweek":
        content = _seasonal_fit(power, fit, params)
    elif kind == "daytype_profile":
        statistic = str(params.get("statistic", "median"))
        if statistic not in ("mean", "median"):
            raise ValueError("invalid statistic")
        level_span = int(params.get("level_span", DAY))
        if not 0 <= level_span <= 4 * WEEK:
            raise ValueError("level_span must be 0..2688")
        content = {"profile": _profile_fit(power, fit, statistic), "level_span": level_span,
                   "statistic": statistic, "n_fit_observations": int(np.isfinite(power.reindex(fit)).sum())}
    elif kind == "mstl_stationary":
        content = _mstl_fit(power, fit)
    elif kind == "kalman_b5":
        content = fit_kalman(power, fit)
    elif kind == "dshw":
        try:
            from src.models.seasonal import DSHW
        except ImportError as exc:
            raise RuntimeError("DSHW requires the numba dependency used by src.models.seasonal") from exc

        fit_only = _masked_fit(power, fit)
        if int(np.isfinite(fit_only).sum()) < 2 * WEEK:
            raise ValueError("DSHW needs at least two fit weeks")
        model = DSHW().fit(fit_only, fit.max())
        content = {"model": model, "n_fit_observations": int(np.isfinite(fit_only).sum()),
                   "fit_mse": float(model.fit_mse_), "params_fitted": model.params_}
    elif kind in {"kalman_profile", "kalman_hour_noise", "kalman_daytype_phi", "kalman_two_state", "kalman_robust"}:
        fit_grid = power.loc[:fit.max()]
        weeks = int(params.get("weeks", 2 if kind == "kalman_profile" else 1))
        if weeks not in (1, 2, 3, 4):
            raise ValueError("weeks must be 1..4")
        base = _weekly_reference(fit_grid.to_numpy(dtype=float), weeks, "mean")
        dev = fit_grid.to_numpy(dtype=float) - base
        selected = fit_grid.index.isin(fit) & np.isfinite(dev)
        content = _fit_ar1(dev, selected)
        content["weeks"] = weeks
        if kind == "kalman_hour_noise":
            # Fit a positive hour-specific observation variance multiplier.
            hour = fit_grid.index.hour.to_numpy(dtype=int)
            sample_var = max(float(np.var(dev[selected])), 1e-8)
            var = np.array([np.var(dev[selected & (hour == h)]) if (selected & (hour == h)).sum() > 4
                            else sample_var for h in range(24)], dtype=float)
            var = np.clip(var, sample_var * .1, sample_var * 10)
            content["r_by_hour"] = content["r"] * var / float(np.mean(var))
        elif kind == "kalman_daytype_phi":
            category = _daytype(fit_grid.index)
            adjacent = selected[1:] & selected[:-1]
            phis = []
            for group in range(4):
                ix = adjacent & (category[1:] == group)
                if ix.sum() >= 20 and np.var(dev[:-1][ix]) > 1e-12:
                    phi = float(np.dot(dev[:-1][ix], dev[1:][ix]) / np.dot(dev[:-1][ix], dev[:-1][ix]))
                    phis.append(float(np.clip(phi, -.99, .99)))
                else:
                    phis.append(content["phi"])
            content["phi_by_daytype"] = np.array(phis, dtype=float)
        elif kind == "kalman_two_state":
            # The slow state has a fixed persistence; its variance fraction is
            # estimated from fit innovations in a small fit-only grid.
            candidates = (.02, .1, .3)
            likelihood = [(fraction, _two_state_filter(dev, selected, fit_grid.index, content,
                           fraction, save=False)[0]) for fraction in candidates]
            content["slow_q_fraction"] = min(likelihood, key=lambda row: row[1])[0]
            content["slow_phi"] = .995
        elif kind == "kalman_robust":
            nu = float(params.get("nu", 4.0))
            if nu <= 2:
                raise ValueError("nu must exceed 2")
            content["nu"] = nu
    elif kind in {"autoets", "tbats", "autoarima_fourier", "autotheta"}:
        content = _native_fit(power, fit, kind, params)
    else:
        raise ValueError(f"unknown statistical kind: {kind}")
    return {"kind": kind, "params": params, "fit_end": fit.max(), "fit_start": fit.min(),
            "development_only": True, "fit_parameters_only": True, "seconds": perf_counter() - started,
            **content}


def _two_state_filter(dev: np.ndarray, selected: np.ndarray, index: pd.DatetimeIndex,
                      bundle: Mapping, fraction: float, *, save: bool) -> tuple[float, np.ndarray | None]:
    slow_phi = float(bundle.get("slow_phi", .995))
    phi = float(bundle["phi"])
    q = float(bundle["q"])
    r = float(bundle["r"])
    transition = np.diag([slow_phi, phi])
    noise = np.diag([q * fraction, q * (1 - fraction)])
    state = np.zeros(2)
    cov = np.diag([q * fraction / (1 - slow_phi**2), q * (1 - fraction) / (1 - phi**2)])
    states = np.full((len(dev), 2), np.nan) if save else None
    nll = 0.0
    for pos in range(len(dev)):
        if pos:
            state = transition @ state
            cov = transition @ cov @ transition.T + noise
        if selected[pos] and np.isfinite(dev[pos]):
            innovation = dev[pos] - float(state.sum())
            variance = float(cov.sum() + r)
            gain = cov @ np.ones(2) / variance
            state += gain * innovation
            cov -= np.outer(gain, np.ones(2) @ cov)
            nll += .5 * (np.log(2 * np.pi * variance) + innovation**2 / variance)
        if save:
            states[pos] = state
    return nll, states


def _dynamic_filter(dev: np.ndarray, index: pd.DatetimeIndex, bundle: Mapping, kind: str) -> np.ndarray:
    phi = float(bundle["phi"])
    q = float(bundle["q"])
    r = float(bundle["r"])
    mean = 0.0
    variance = q / (1 - phi**2)
    states = np.full(len(dev), np.nan)
    hours = index.hour.to_numpy(dtype=int)
    daytypes = _daytype(index) if kind == "kalman_daytype_phi" else None
    for pos, obs in enumerate(dev):
        if pos:
            p = float(bundle["phi_by_daytype"][daytypes[pos]]) if daytypes is not None else phi
            mean *= p
            variance = p**2 * variance + q
        if np.isfinite(obs):
            obs_r = float(bundle["r_by_hour"][hours[pos]]) if kind == "kalman_hour_noise" else r
            innovation = obs - mean
            if kind == "kalman_robust":
                z2 = innovation**2 / max(variance + obs_r, 1e-10)
                weight = (float(bundle["nu"]) + 1) / (float(bundle["nu"]) + z2)
                obs_r /= max(weight, 1e-3)
            gain = variance / (variance + obs_r)
            mean += gain * innovation
            variance *= 1 - gain
        states[pos] = mean
    return states


def _native_predict(bundle: Mapping, power: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    model = bundle["model"]
    fit_start = pd.Timestamp(bundle["fit_start"])
    fit_end = pd.Timestamp(bundle["fit_end"])
    result = np.full(len(origins), np.nan)
    for row, origin in enumerate(origins):
        if origin < fit_end:
            raise ValueError("native model predictions require origin after fitted window")
        ahead = int((origin + horizon * STEP - fit_end) / STEP)
        if bundle["state_update"] == "native_forward":
            observed = power.loc[fit_start:origin].ffill().to_numpy(dtype=float)
            if not np.isfinite(observed).all():
                continue
            x = future_x = None
            if bundle["kind"] == "autoarima_fourier":
                harmonics = tuple(bundle["harmonics"])
                x = _fourier(power.loc[fit_start:origin].index, fit_start, harmonics)
                future_x = _fourier(pd.date_range(origin + STEP, periods=horizon, freq=STEP), fit_start, harmonics)
            forecast = model.forward(observed, h=horizon, X=x, X_future=future_x)["mean"]
            result[row] = float(np.asarray(forecast)[-1])
        else:
            forecast = model.predict(h=ahead)["mean"]
            result[row] = float(np.asarray(forecast)[-1])
    return result


def predict_model(bundle: Mapping, history: pd.DataFrame | pd.Series,
                  origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Forecast one target per origin, never reading observations after origin."""
    power = _power(history)
    origins = _origins(power, pd.DatetimeIndex(origins), horizon)
    kind = bundle["kind"]
    if len(origins) == 0:
        return np.empty(0, dtype=float)
    if kind != "seasonal_kweek" and (origins < pd.Timestamp(bundle["fit_end"])).any():
        raise ValueError("prediction origin precedes fitted parameter endpoint")
    if kind == "kalman_b5":
        return predict_kalman(bundle, power, origins, horizon)
    if kind == "dshw":
        if horizon >= DAY:
            raise ValueError("DSHW causal state indexing requires horizon < one day")
        if (origins < bundle["fit_end"]).any():
            raise ValueError("DSHW predictions require origin after fitted window")
        return bundle["model"].predict(power.loc[:origins.max()], origins, horizon)
    if kind in {"autoets", "tbats", "autoarima_fourier", "autotheta"}:
        return _native_predict(bundle, power, origins, horizon)
    grid = power.loc[:origins.max()]
    positions = grid.index.get_indexer(origins)
    targets = origins + horizon * STEP
    if kind == "seasonal_kweek":
        if horizon > WEEK:
            raise ValueError("weekly targets must be at/before origin")
        anchor = _weekly_reference(grid.to_numpy(dtype=float), int(bundle["weeks"]), str(bundle["statistic"]))
        # Targets after the last origin still reference positions in the known past.
        lookup = positions + horizon - WEEK
        return np.where(lookup >= 0, anchor[np.maximum(lookup, 0)], np.nan)
    if kind == "daytype_profile":
        target_profile = _profile_values(np.asarray(bundle["profile"]), targets)
        span = int(bundle["level_span"])
        if span == 0:
            return target_profile
        known_profile = _profile_values(np.asarray(bundle["profile"]), grid.index)
        residual = grid.to_numpy(dtype=float) - known_profile
        level = np.full(len(origins), np.nan)
        for row, pos in enumerate(positions):
            recent = residual[max(0, pos - span + 1):pos + 1]
            sample = recent[np.isfinite(recent)]
            if len(sample):
                level[row] = float(np.median(sample))
        return target_profile + level
    if kind == "mstl_stationary":
        weekly = np.asarray(bundle["weekly"], dtype=float)
        fit_end = pd.Timestamp(bundle["fit_end"])
        fit_pos = grid.index.get_loc(fit_end)
        phase = (np.arange(len(grid)) - (fit_pos - WEEK + 1)) % WEEK
        origin_seasonal = weekly[phase]
        _, states = _level_state(grid.to_numpy(dtype=float), origin_seasonal, float(bundle["alpha"]), save=True)
        target_positions = positions + horizon
        target_phase = (target_positions - (fit_pos - WEEK + 1)) % WEEK
        return states[positions] + weekly[target_phase]
    if kind.startswith("kalman_"):
        weeks = int(bundle["weeks"])
        if horizon > WEEK:
            raise ValueError("weekly reference target must be at/before origin")
        base = _weekly_reference(grid.to_numpy(dtype=float), weeks, "mean")
        dev = grid.to_numpy(dtype=float) - base
        lookup = positions + horizon - WEEK
        # Multi-week profile target requires each prior-week target anchor.
        target_base = np.full(len(origins), np.nan)
        for row, target_week_pos in enumerate(lookup):
            anchors = [target_week_pos - (k - 1) * WEEK for k in range(1, weeks + 1)]
            if min(anchors) >= 0 and np.isfinite(grid.to_numpy(dtype=float)[anchors]).all():
                target_base[row] = float(np.mean(grid.to_numpy(dtype=float)[anchors]))
        if kind == "kalman_two_state":
            _, states = _two_state_filter(dev, np.isfinite(dev), grid.index, bundle,
                                          float(bundle["slow_q_fraction"]), save=True)
            correction = float(bundle["slow_phi"]) ** horizon * states[positions, 0] + float(bundle["phi"]) ** horizon * states[positions, 1]
        elif kind in {"kalman_hour_noise", "kalman_daytype_phi", "kalman_robust"}:
            states = _dynamic_filter(dev, grid.index, bundle, kind)
            if kind == "kalman_daytype_phi":
                correction = np.empty(len(origins))
                full_index = pd.date_range(grid.index[0], targets.max(), freq=STEP)
                types = _daytype(full_index)
                for row, pos in enumerate(positions):
                    factor = float(np.prod(np.asarray(bundle["phi_by_daytype"])[types[pos + 1:pos + horizon + 1]]))
                    correction[row] = factor * states[pos]
            else:
                correction = float(bundle["phi"]) ** horizon * states[positions]
        else:
            _, _, states = _filter_deviations(dev, np.isfinite(dev), float(bundle["phi"]),
                                               float(bundle["q"]), float(bundle["r"]), save_states=True)
            correction = float(bundle["phi"]) ** horizon * states[positions]
        return target_base + correction
    raise ValueError(f"unknown statistical kind: {kind}")


def predict_state(bundle: Mapping, history: pd.DataFrame | pd.Series,
                  origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Expose causal Kalman state mean/variance for a separately fitted hybrid.

    These are origin-time features only.  The hybrid must be fitted on fit
    targets with its own causal B5 predictions; no score label enters here.
    """
    kind = str(bundle.get("kind", "kalman_b5"))
    allowed = {"kalman_b5", "kalman_profile", "kalman_hour_noise", "kalman_daytype_phi", "kalman_robust"}
    if kind not in allowed:
        raise ValueError(f"state export is unsupported for {kind}")
    power = _power(history)
    origins = _origins(power, pd.DatetimeIndex(origins), 1)
    if len(origins) == 0:
        return pd.DataFrame({"state_mean": [], "state_variance": []}, index=origins)
    if (origins < pd.Timestamp(bundle.get("fit_end", origins.min()))).any():
        raise ValueError("state origin precedes fitted parameter endpoint")
    grid = power.loc[:origins.max()]
    values = grid.to_numpy(dtype=float)
    base = _weekly_reference(values, int(bundle.get("weeks", 1)), "mean")
    deviation = values - base
    phi = float(bundle["phi"])
    q = float(bundle["q"])
    r = float(bundle["r"])
    if not -.99 <= phi <= .99 or q <= 0 or r <= 0:
        raise ValueError("invalid Kalman parameters")
    mean = 0.0
    variance = q / (1 - phi**2)
    means = np.full(len(grid), np.nan)
    variances = np.full(len(grid), np.nan)
    hours = grid.index.hour.to_numpy(dtype=int)
    daytypes = _daytype(grid.index) if kind == "kalman_daytype_phi" else None
    for pos, obs in enumerate(deviation):
        if pos:
            current_phi = float(bundle["phi_by_daytype"][daytypes[pos]]) if daytypes is not None else phi
            mean *= current_phi
            variance = current_phi**2 * variance + q
        if np.isfinite(obs):
            observation_r = float(bundle["r_by_hour"][hours[pos]]) if kind == "kalman_hour_noise" else r
            innovation = obs - mean
            if kind == "kalman_robust":
                nu = float(bundle["nu"])
                z2 = innovation**2 / max(variance + observation_r, 1e-10)
                observation_r /= max((nu + 1) / (nu + z2), 1e-3)
            gain = variance / (variance + observation_r)
            mean += gain * innovation
            variance *= 1 - gain
        means[pos] = mean
        variances[pos] = variance
    positions = grid.index.get_indexer(origins)
    return pd.DataFrame({"state_mean": means[positions], "state_variance": variances[positions]}, index=origins)
