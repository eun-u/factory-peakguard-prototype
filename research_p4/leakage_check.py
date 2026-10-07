"""Future-perturbation checks: forecasts issued at an origin must not change when later data change."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import stat_models as sm
from .common import OUT, config, contexts, history


def run():
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    series = sm._grid_series(df)
    y = series.to_numpy(float)
    pos = pd.Series(np.arange(len(series)), index=series.index)
    results = []
    # DSHW with parameters fitted on fold 2's fit window (h4).
    c = ctx[(4, 2)]
    fit_end = int(pos[c["fit"].max()])
    params, _, init = sm.dshw_fit(y, fit_end)
    for cut in (c["cal"][10], c["score"][500], c["score"][1500]):
        k = int(pos[cut])
        y2 = y.copy(); y2[k + 1:] = 999.0
        out = []
        for arr in (y, y2):
            lev, sd, sw, err, _ = sm._filter(arr, *params, *init, fit_end)
            out.append(np.array([sm.dshw_forecast((lev, sd, sw, err), np.array([k]), h, params[3])[0] for h in (1, 4, 16, 96)]))
        results.append({"model": "dshw", "origin": str(cut), "max_abs_change": float(np.max(np.abs(out[0] - out[1])))})
    # MSTL: refit the day paths for the origin's day on perturbed future data.
    from statsforecast.models import MSTL, AutoETS
    model = MSTL(season_length=[96, 672], trend_forecaster=AutoETS(model="ZZN"))
    for cut in (c["score"][500], c["score"][1500]):
        day = (cut - pd.Timedelta(nanoseconds=1)).floor("D")
        s2 = series.copy(); s2[s2.index > cut] = 999.0
        vals = []
        for s in (series, s2):
            hist = s.loc[:day].iloc[-28 * 96:].interpolate(limit_direction="both").to_numpy(float)
            vals.append(model.forecast(y=hist, h=192)["mean"])
        results.append({"model": "mstl_daily", "origin": str(cut), "max_abs_change": float(np.max(np.abs(vals[0] - vals[1])))})
    (OUT / "leakage_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results)
    assert all(r["max_abs_change"] == 0 for r in results)


if __name__ == "__main__":
    run()
