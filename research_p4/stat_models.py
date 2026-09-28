"""E1: statistical seasonal benchmarks (Taylor DSHW, daily-refit MSTL).

Both are causal: a forecast issued at origin t reads only observations with
timestamps <= t. Missing/repaired observations are never used for updates.
"""
from __future__ import annotations

import json
import pickle

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import CACHE, HORIZONS, Timer, config, contexts, history, make_rows, safe_power, save

M1, M2, INIT = 96, 672, 1344


def _grid_series(df: pd.DataFrame) -> pd.Series:
    power = safe_power(df)
    grid = pd.date_range(power.index.min(), power.index.max(), freq="15min")
    return power.reindex(grid)


def _init_states(y: np.ndarray):
    level = np.nanmean(y[:INIT])
    day = np.array([np.nanmean(y[i:INIT:M1]) for i in range(M1)]) - level
    week = np.array([np.nanmean(y[i:INIT:M2]) for i in range(M2)]) - level - day[np.arange(M2) % M1]
    return level, np.nan_to_num(day), np.nan_to_num(week)


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


def dshw_fit(y: np.ndarray, fit_end_idx: int):
    level0, day0, week0 = _init_states(y)
    obj = lambda p: _filter(y, p[0], p[1], p[2], p[3], level0, day0, week0, fit_end_idx)[4]
    res = minimize(obj, x0=np.array([.1, .2, .2, .5]), method="L-BFGS-B",
                   bounds=[(0, 1), (0, 1), (0, 1), (0, .99)])
    return res.x, float(res.fun), (level0, day0, week0)


def dshw_forecast(states, position: np.ndarray, h: int, phi: float) -> np.ndarray:
    lev, sd, sw, err = states
    return lev[position] + sd[position + h - M1] + sw[position + h - M2] + phi ** h * err[position]


def run_dshw() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    series = _grid_series(df)
    y = series.to_numpy(dtype=float)
    pos = pd.Series(np.arange(len(series)), index=series.index)
    rows, meta, store = [], [], {}
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            fit_end = int(pos[c["fit"].max()])
            params, mse, init = dshw_fit(y, fit_end)
            a, d, w, phi = params
            lev, sd, sw, err, _ = _filter(y, a, d, w, phi, *init, fit_end)
            fc = lambda idx: dshw_forecast((lev, sd, sw, err), pos[idx].to_numpy(), h, phi)
            pc, ps = fc(c["cal"]), fc(c["score"])
            store[(h, f)] = {"cal": pc, "score": ps}
            rows.append(make_rows(c, h, f, "p4_dshw", pc, ps))
            meta.append({"horizon": h, "fold": f, "alpha": a, "delta": d, "omega": w, "phi": phi,
                         "fit_mse": mse, "seconds": t()})
            print(meta[-1], flush=True)
    (CACHE / "dshw.pkl").write_bytes(pickle.dumps(store))
    save(rows, "p4_dshw", {"folds": meta})
    return {"folds": meta}


# ---------------- MSTL (daily refit) ----------------

def _mstl_daily_paths(series: pd.Series) -> dict:
    """day D -> forecast path for the 192 slots after D, fitted on the 28 days ending at D."""
    path = CACHE / "mstl_daily.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from statsforecast.models import MSTL, AutoETS
    model = MSTL(season_length=[M1, M2], trend_forecaster=AutoETS(model="ZZN"))
    window = 28 * M1
    days = pd.date_range(series.index.min().normalize() + pd.Timedelta(days=28),
                         series.index.max().normalize(), freq="D")
    out = {}
    t = Timer()
    for i, day in enumerate(days):
        hist = series.loc[:day].iloc[-window:]
        if len(hist) < window:
            continue
        values = hist.interpolate(limit_direction="both").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        fc = model.forecast(y=values, h=2 * M1)["mean"]
        out[day] = pd.Series(fc, index=pd.date_range(day + pd.Timedelta(minutes=15), periods=2 * M1, freq="15min"))
        if i % 20 == 0:
            print(f"[mstl] {i}/{len(days)} {day.date()} {t()}s", flush=True)
    path.write_bytes(pickle.dumps(out))
    return out


def _mstl_level(paths: dict, origins: pd.DatetimeIndex, offset_steps: int) -> np.ndarray:
    """MSTL value for time origin+offset using the fit of the day containing the origin."""
    result = np.full(len(origins), np.nan)
    keys = (origins - pd.Timedelta(nanoseconds=1)).floor("D")
    for i, (origin, key) in enumerate(zip(origins, keys)):
        p = paths.get(key)
        if p is not None:
            result[i] = p.get(origin + pd.Timedelta(minutes=15 * offset_steps), np.nan)
    return result


def run_mstl() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    series = _grid_series(df)
    paths = _mstl_daily_paths(series)
    rows, meta, store = [], [], {}
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            fit = c["fit"]
            resid = series.reindex(fit).to_numpy() - _mstl_level(paths, fit, 0)
            r = pd.Series(resid, index=fit)
            lagged = r.shift(1).where((fit.to_series().diff() == pd.Timedelta(minutes=15)).to_numpy())
            ok = r.notna() & lagged.notna()
            phi = float(np.clip(np.corrcoef(r[ok], lagged[ok])[0, 1], 0, .99)) if ok.sum() > 100 else 0.0

            def fc(idx):
                now = series.reindex(idx).to_numpy()
                return _mstl_level(paths, idx, h) + phi ** h * (now - _mstl_level(paths, idx, 0))
            pc, ps = fc(c["cal"]), fc(c["score"])
            store[(h, f)] = {"cal": pc, "score": ps}
            rows.append(make_rows(c, h, f, "p4_mstl_daily", pc, ps))
            meta.append({"horizon": h, "fold": f, "phi": phi, "n_resid": int(ok.sum()),
                         "score_missing": int(np.isnan(ps).sum()), "seconds": t()})
            print(meta[-1], flush=True)
    (CACHE / "mstl.pkl").write_bytes(pickle.dumps(store))
    save(rows, "p4_mstl_daily", {"folds": meta, "days_fitted": len(paths)})
    return {"folds": meta}


if __name__ == "__main__":
    import sys
    {"dshw": run_dshw, "mstl": run_mstl}[sys.argv[1]]()
