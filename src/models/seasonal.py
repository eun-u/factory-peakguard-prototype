"""Seasonal statistical forecasters adopted in P4 (preregistration_0928_P4.md).

* Taylor (2003) additive double-seasonal Holt-Winters with AR(1) error term
  (day 96, week 672 quarter-hours).
* Daily-refit MSTL (statsforecast) with an AR(1) intraday correction.

Availability: a forecast issued at origin t reads only observations stamped
<= t. Missing or time-repaired observations never update a state; the one-step
forecast stands in for them. MSTL day D is fitted on the 28 days ending at D
00:00 and used only for origins in (D, D+1 day].
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize

M1, M2, INIT = 96, 672, 1344
MSTL_WINDOW_DAYS = 28


def safe_grid(df: pd.DataFrame) -> pd.Series:
    """Clean 15-minute power on a gap-free grid; repaired readings become NaN."""
    repaired = df.get("time_repaired", pd.Series(False, index=df.index)).fillna(True).astype(bool)
    power = pd.to_numeric(df["power"], errors="coerce").mask(repaired)
    grid = pd.date_range(power.index.min(), power.index.max(), freq="15min")
    return power.reindex(grid)


def _init_states(y: np.ndarray):
    level = np.nanmean(y[:INIT])
    day = np.array([np.nanmean(y[i:INIT:M1]) for i in range(M1)]) - level
    week = np.array([np.nanmean(y[i:INIT:M2]) for i in range(M2)]) - level - day[np.arange(M2) % M1]
    return float(level), np.nan_to_num(day), np.nan_to_num(week)


@numba.njit(cache=True)
def _filter(y, a, d, w, phi, level0, day0, week0, sse_end):
    n = len(y)
    lev = np.empty(n); sd = np.empty(n); sw = np.empty(n); err = np.zeros(n)
    for i in range(INIT):
        lev[i] = level0; sd[i] = day0[i % M1]; sw[i] = week0[i % M2]
    sse = 0.0; count = 0
    for t in range(INIT, n):
        base = lev[t - 1] + sd[t - M1] + sw[t - M2]
        pred = base + phi * err[t - 1]
        if np.isnan(y[t]):
            yt = pred
        else:
            yt = y[t]
            if t <= sse_end:
                sse += (y[t] - pred) ** 2; count += 1
        err[t] = yt - base
        lev[t] = a * (yt - sd[t - M1] - sw[t - M2]) + (1 - a) * lev[t - 1]
        sd[t] = d * (yt - lev[t] - sw[t - M2]) + (1 - d) * sd[t - M1]
        sw[t] = w * (yt - lev[t] - sd[t - M1]) + (1 - w) * sw[t - M2]
    return lev, sd, sw, err, sse / max(count, 1)


class DSHW:
    """Parameters are estimated only from observations stamped <= ``fit_end``."""

    def fit(self, series: pd.Series, fit_end: pd.Timestamp) -> "DSHW":
        y = series.to_numpy(dtype=float)
        end = int(series.index.get_loc(fit_end))
        self.init_ = _init_states(y)
        objective = lambda p: _filter(y, p[0], p[1], p[2], p[3], *self.init_, end)[4]
        result = minimize(objective, x0=np.array([.1, .2, .2, .5]), method="L-BFGS-B",
                          bounds=[(0, 1), (0, 1), (0, 1), (0, .99)])
        self.params_ = tuple(float(v) for v in result.x)
        self.fit_end_ = pd.Timestamp(fit_end)
        self.start_ = series.index[0]
        self.fit_mse_ = float(result.fun)
        return self

    def predict(self, series: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        """Filter causally over ``series`` and read the state at each origin."""
        if series.index[0] != self.start_:
            raise ValueError("DSHW must filter the same grid start used for fitting")
        y = series.to_numpy(dtype=float)
        a, d, w, phi = self.params_
        lev, sd, sw, err, _ = _filter(y, a, d, w, phi, *self.init_, -1)
        pos = series.index.get_indexer(pd.DatetimeIndex(origins))
        if (pos < INIT).any() or (pos < 0).any():
            raise ValueError("DSHW origins must lie on the grid after the initialisation window")
        return lev[pos] + sd[pos + horizon - M1] + sw[pos + horizon - M2] + phi ** horizon * err[pos]


def mstl_day_paths(series: pd.Series, days, cache: dict | None = None) -> dict:
    """day -> Series of 192 forecast values after that midnight (causal)."""
    from statsforecast.models import MSTL, AutoETS
    model = MSTL(season_length=[M1, M2], trend_forecaster=AutoETS(model="ZZN"))
    window = MSTL_WINDOW_DAYS * M1
    out = {} if cache is None else cache
    for day in pd.DatetimeIndex(days):
        if day in out:
            continue
        hist = series.loc[:day].iloc[-window:]
        if len(hist) < window or hist.index[-1] != day:
            continue
        values = hist.interpolate(limit_direction="both").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        forecast = model.forecast(y=values, h=2 * M1)["mean"]
        out[day] = pd.Series(forecast, index=pd.date_range(day + pd.Timedelta(minutes=15), periods=2 * M1, freq="15min"))
    return out


def origin_days(origins: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (pd.DatetimeIndex(origins) - pd.Timedelta(nanoseconds=1)).floor("D")


def mstl_level(paths: dict, origins: pd.DatetimeIndex, offset: int) -> np.ndarray:
    result = np.full(len(origins), np.nan)
    for i, (origin, key) in enumerate(zip(pd.DatetimeIndex(origins), origin_days(origins))):
        path = paths.get(key)
        if path is not None:
            result[i] = path.get(origin + pd.Timedelta(minutes=15 * offset), np.nan)
    return result


def mstl_phi(series: pd.Series, paths: dict, fit: pd.DatetimeIndex) -> float:
    """Lag-1 autocorrelation of (y - MSTL) on contiguous fit origins, clipped to [0, .99]."""
    fit = pd.DatetimeIndex(fit)
    residual = pd.Series(series.reindex(fit).to_numpy() - mstl_level(paths, fit, 0), index=fit)
    contiguous = (fit.to_series().diff() == pd.Timedelta(minutes=15)).to_numpy()
    lagged = residual.shift(1).where(contiguous)
    ok = residual.notna() & lagged.notna()
    if ok.sum() <= 100:
        return 0.0
    return float(np.clip(np.corrcoef(residual[ok], lagged[ok])[0, 1], 0, .99))


def mstl_forecast(series: pd.Series, paths: dict, origins: pd.DatetimeIndex, horizon: int, phi: float) -> np.ndarray:
    now = series.reindex(origins).to_numpy(dtype=float)
    return mstl_level(paths, origins, horizon) + phi ** horizon * (now - mstl_level(paths, origins, 0))


def simplex_weights(components: list[np.ndarray], y: np.ndarray, step: float = .1):
    """Grid of nonnegative weights summing to one; minimum calibration MAE, first grid order on ties."""
    import itertools
    grid = [w for w in itertools.product(np.round(np.arange(0, 1 + 1e-9, step), 1), repeat=len(components))
            if abs(sum(w) - 1) < 1e-9]
    def mae(w):
        p = sum(wi * c for wi, c in zip(w, components))
        ok = np.isfinite(p) & np.isfinite(y)
        return float(np.mean(np.abs(y[ok] - p[ok]))) if ok.any() else np.inf
    best = min(grid, key=lambda w: (mae(w), w))
    return tuple(float(v) for v in best)
