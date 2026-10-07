"""Shadow run of the final (freeze) path on a pseudo-holdout inside the sealed development data.

Never reads the real holdout. The pseudo-holdout starts at 2021-07-10; models are
frozen on data before it, then scored with src.finalize._predict_horizon.
A future-perturbation check confirms causal test-time predictions.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import finalize
from src.evaluate import score_predictions
from .common import OUT, ROOT, config, history

PSEUDO = pd.Timestamp("2021-07-10 00:00")


def run():
    cfg, df = config(), history()
    dev = json.loads((ROOT / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    dev["selection"]["by_horizon"]["1"]["point_model"] = "p4_mstl_daily"
    dev["selection"]["by_horizon"]["1"]["risk_model"] = "p4_quantile_dense"
    dev["selection"]["by_horizon"]["4"]["point_model"] = "p4_blend3"
    df_before = df.loc[df.index < PSEUDO]
    test_origins = df.index[df.index >= PSEUDO]
    report = {}
    for h in (1, 4):
        bundle = finalize._prepare_horizon(df_before, PSEUDO, h, cfg, dev)
        rows = pd.concat(finalize._predict_horizon(df, test_origins, h, bundle, cfg), ignore_index=True)
        summary = {}
        for name, part in rows.groupby("model"):
            if name in ("p4_mstl_daily", "p4_blend3", "lgbm_no_holiday_weight_2", "p4_dshw", "p1_latest",
                        "c2a_max_4_5_adjusted", "p4_quantile_dense", "lgbm_quantile_b"):
                s = score_predictions(part)
                summary[name] = {k: s.get(k) for k in ("n", "mae", "peak_mae", "episode_f1", "false_alarms_positions", "brier")}
        # Causality: perturb everything after a mid-window origin; earlier predictions must not move.
        cut = test_origins[len(test_origins) // 2]
        perturbed = df.copy()
        perturbed.loc[perturbed.index > cut, "power"] = 999.0
        early = test_origins[test_origins <= cut - pd.Timedelta(minutes=15 * h)]
        a = pd.concat(finalize._predict_horizon(df, early, h, bundle, cfg), ignore_index=True)
        b = pd.concat(finalize._predict_horizon(perturbed, early, h, bundle, cfg), ignore_index=True)
        keep = ["origin", "model", "pred"]
        m = a[keep].merge(b[keep], on=["origin", "model"], suffixes=("", "_p"))
        change = float(np.nanmax(np.abs(m.pred - m.pred_p)))
        point = bundle["point"]
        report[h] = {"metrics": summary, "future_perturbation_max_change": change,
                     "point": {k: v for k, v in point.items() if k in ("name", "kind", "phi", "weights", "cutoff")},
                     "dshw_params": getattr(point.get("dshw"), "params_", None),
                     "dense_cutoff": (bundle.get("dense_risk") or {}).get("cutoff")}
        print(h, json.dumps(report[h], default=float, indent=1), flush=True)
        assert change == 0.0, "Final path used future observations"
    (OUT / "shadow_finalize.json").write_text(json.dumps(report, default=float, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
