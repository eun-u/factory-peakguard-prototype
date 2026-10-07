"""E4: dense LightGBM quantiles with the existing Mondrian-B calibration and probability rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.conformal import apply_conformal, assign_mondrian_bins, fit_conformal, fit_mondrian_edges
from src.models.lgbm_quantile import fit_quantiles, predict_quantiles
from src.models.peak_prob import exceedance_from_quantiles
from src.training import _cutoff, _ordered_quantiles, _row
from .common import HORIZONS, Timer, config, contexts, feature_columns, history, save

LEVELS = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99)


def _name(level):
    return "q" + (f"{level:.3f}".rstrip("0").split(".")[1])


def run() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    rows, meta = [], []
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            cols = feature_columns(c["x"])
            x, y, tau = c["x"][cols], c["targets"].y, c["tau"]
            cal, score = c["cal"], c["score"]
            models = fit_quantiles(x.loc[c["fit"]], y.loc[c["fit"]], x.loc[c["stop"]], y.loc[c["stop"]],
                                   cfg, alphas=list(LEVELS), params=c["expected"]["chosen_params"])
            q_cal = predict_quantiles(models, x.loc[cal])
            q_score = predict_quantiles(models, x.loc[score])
            edges = fit_mondrian_edges(q_cal[.5], cfg.get("conformal", {}).get("mondrian_bins", [.5, .9]))
            cal_bins, score_bins = assign_mondrian_bins(q_cal[.5], edges), assign_mondrian_bins(q_score[.5], edges)
            half = len(cal) // 2
            cutoff_start = min(len(cal), half + h + 1)
            y_cal = y.loc[cal].to_numpy()
            qc, qs = dict(q_cal), dict(q_score)
            for level in [lv for lv in LEVELS if lv >= .9]:
                fitted = fit_conformal(y_cal[:half], q_cal[level][:half], level, cal_bins[:half])
                qc[level] = apply_conformal(q_cal[level], fitted, cal_bins)
                qs[level] = apply_conformal(q_score[level], fitted, score_bins)
            qc, qs = _ordered_quantiles(qc), _ordered_quantiles(qs)
            p_cal, p_score = exceedance_from_quantiles(qc, tau), exceedance_from_quantiles(qs, tau)
            cutoff = _cutoff(y_cal[cutoff_start:], p_cal[cutoff_start:], tau,
                             c["targets"].loc[cal[cutoff_start:], "target_time"])
            extras = {"q10": q_score[.1], "q50": q_score[.5], "q90": q_score[.9], "q95": q_score[.95],
                      "q975": q_score[.975], "q90_cal": qs[.9], "q95_cal": qs[.95], "q975_cal": qs[.975],
                      "p_exceed": p_score, "q50_top_edge": float(edges[-1]), "conformal_method": "b_dense"}
            extras.update({f"dense_{_name(lv)}": qs[lv] for lv in LEVELS})
            row = _row(score, h, f, "p4_quantile_dense", c["targets"], tau, q_score[.5], float("inf"), **extras)
            row["alert"] = p_score > cutoff
            row["alert_cutoff"] = cutoff
            rows.append(row)
            meta.append({"horizon": h, "fold": f, "levels": list(LEVELS), "fits": len(LEVELS),
                         "alert_cutoff": cutoff, "seconds": t()})
            print(meta[-1], flush=True)
    save(rows, "p4_quantile_dense", {"folds": meta})
    return {"folds": meta}


if __name__ == "__main__":
    run()
