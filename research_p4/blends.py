"""E3: forecast combinations with weights chosen on calibration rows only."""
from __future__ import annotations

import itertools
import pickle

import numpy as np
import pandas as pd

from src.models.cbl import cbl_all_predictions
from .common import CACHE, CBL_REP, HORIZONS, OUT, ROOT, Timer, config, contexts, history, make_rows, save


def _cbl(df, ctx, h, f):
    path = CACHE / f"cbl_h{h}_f{f}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    c = ctx[(h, f)]
    out = {}
    for part in ("cal", "score"):
        frame = cbl_all_predictions(df, c[part], h)
        frame["c3_holiday_hybrid"] = frame["c3_holiday_mid_4_6"].combine_first(frame["c1_mid_6_10"])
        out[part] = frame[CBL_REP[h]].to_numpy(dtype=float)
    path.write_bytes(pickle.dumps(out))
    return out


def _mae(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[ok] - p[ok]))) if ok.any() else np.inf


def run() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    lgbm = pickle.loads((CACHE / "lgbm_component.pkl").read_bytes())
    dshw = pickle.loads((CACHE / "dshw.pkl").read_bytes())
    oof = pd.read_csv(ROOT / "outputs/predictions/p2_combined_oof.csv",
                      usecols=["origin", "horizon", "fold", "model", "pred"], parse_dates=["origin"])
    rows2, rows3, meta = [], [], []
    grid2 = np.round(np.arange(0, 1.0001, .05), 2)
    grid3 = [w for w in itertools.product(np.round(np.arange(0, 1.0001, .1), 1), repeat=3)
             if abs(sum(w) - 1) < 1e-9]
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            cbl = _cbl(df, ctx, h, f)
            ref = oof.loc[oof.horizon.eq(h) & oof.fold.eq(f) & oof.model.eq(CBL_REP[h])].set_index("origin").pred.reindex(c["score"])
            cbl_parity = float(np.nanmax(np.abs(ref.to_numpy() - cbl["score"])))
            y_cal = c["targets"].loc[c["cal"], "y"].to_numpy(dtype=float)
            comp = {"lgbm": lgbm[(h, f)], "cbl": cbl, "dshw": dshw[(h, f)]}
            w2 = min(grid2, key=lambda w: (_mae(y_cal, w * comp["lgbm"]["cal"] + (1 - w) * comp["cbl"]["cal"]), w))
            blend = lambda part, w: w * comp["lgbm"][part] + (1 - w) * comp["cbl"][part]
            rows2.append(make_rows(c, h, f, "p4_blend2", blend("cal", w2), blend("score", w2), blend_weights=f"lgbm={w2}"))
            def b3(part, w):
                return w[0] * comp["lgbm"][part] + w[1] * comp["cbl"][part] + w[2] * comp["dshw"][part]
            w3 = min(grid3, key=lambda w: (_mae(y_cal, b3("cal", w)), w))
            rows3.append(make_rows(c, h, f, "p4_blend3", b3("cal", w3), b3("score", w3),
                                   blend_weights=f"lgbm={w3[0]},cbl={w3[1]},dshw={w3[2]}"))
            meta.append({"horizon": h, "fold": f, "w_lgbm_blend2": float(w2),
                         "w_blend3": [float(v) for v in w3], "cbl_parity_max_abs": cbl_parity,
                         "cbl_cal_missing": int(np.isnan(cbl["cal"]).sum()),
                         "cbl_score_missing": int(np.isnan(cbl["score"]).sum()), "seconds": t()})
            print(meta[-1], flush=True)
    save(rows2, "p4_blend2", {"folds": meta})
    save(rows3, "p4_blend3", {"folds": meta})
    return {"folds": meta}


if __name__ == "__main__":
    run()
