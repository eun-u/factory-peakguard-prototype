"""Refit the existing lgbm_no_holiday_weight_2 per fold and check parity with the OOF."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from src.models.lgbm_point import fit_point
from .common import CACHE, HORIZONS, OUT, ROOT, config, contexts, feature_columns, history, Timer


def run() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    oof = pd.read_csv(ROOT / "outputs/predictions/p2_combined_oof.csv",
                      usecols=["origin", "horizon", "fold", "model", "pred"],
                      parse_dates=["origin"])
    oof = oof.loc[oof.model.eq("lgbm_no_holiday_weight_2")]
    store, parity = {}, []
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            cols = feature_columns(c["x"])
            y = c["targets"].y
            params = c["expected"]["chosen_params"]
            model = fit_point(c["x"].loc[c["fit"], cols], y.loc[c["fit"]],
                              c["x"].loc[c["stop"], cols], y.loc[c["stop"]], cfg,
                              peak_threshold=c["tau"], peak_weight=2.0, params=params)
            pc = model.predict(c["x"].loc[c["cal"], cols])
            ps = model.predict(c["x"].loc[c["score"], cols])
            ref = oof.loc[oof.horizon.eq(h) & oof.fold.eq(f)].set_index("origin").pred.reindex(c["score"])
            diff = float(np.nanmax(np.abs(ref.to_numpy() - ps)))
            parity.append({"horizon": h, "fold": f, "max_abs_diff": diff,
                           "missing_ref": int(ref.isna().sum()), "n_score": len(ps), "seconds": t()})
            store[(h, f)] = {"cal": pc, "score": ps}
            print(parity[-1], flush=True)
    (CACHE / "lgbm_component.pkl").write_bytes(pickle.dumps(store))
    OUT.mkdir(exist_ok=True)
    (OUT / "parity_lgbm_component.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    return {"parity": parity}


if __name__ == "__main__":
    run()
