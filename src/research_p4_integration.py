"""Integrate the P4-adopted candidates into the sealed development CV.

Adopted by the preregistered P4 rules (outputs/logs/preregistration_0928_P4.md,
scored in outputs/p4/p4_point_comparisons.csv and p4_risk_comparisons.csv):

* h1 point  ``p4_mstl_daily``   (daily-refit MSTL + AR(1) intraday correction)
* h4 point  ``p4_blend3``       (LGBM no-holiday weight 2, CBL representative, DSHW;
                                 simplex weights on calibration MAE)
* h1 risk   ``p4_quantile_dense`` (15 LightGBM quantiles + Mondrian B on levels >= .9)

P5 (preregistration_0929_P5.md) promoted Chronos-2 but no P5 candidate passed,
so it is not integrated. This module regenerates only the adopted candidates on
the original fold grid and requires parity (1e-6) with the research outputs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate_all
from .models.cbl import cbl_all_predictions
from .models.conformal import apply_conformal, assign_mondrian_bins, fit_conformal, fit_mondrian_edges
from .models.lgbm_point import fit_point
from .models.lgbm_quantile import fit_quantiles, predict_quantiles
from .models.peak_prob import exceedance_from_quantiles
from .models.seasonal import (DSHW, mstl_day_paths, mstl_forecast, mstl_phi, origin_days,
                              safe_grid, simplex_weights)
from .research_cv import build_contexts
from .session_data import SEALED_BOUNDARY
from .training import _cutoff, _ordered_quantiles, _row

HOLIDAY_COLUMNS = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")
DENSE_LEVELS = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99)
EVIDENCE = ("outputs/logs/preregistration_0928_P4.md", "outputs/logs/preregistration_0929_P5.md",
            "outputs/p4/p4_point_comparisons.csv", "outputs/p4/p4_risk_comparisons.csv",
            "outputs/p4/p4_summary.json", "outputs/p4/p5_point_comparisons.csv", "outputs/p4/p5_summary.json")
PARITY_TOLERANCE = 1e-6


def dense_name(level: float) -> str:
    return "dense_q" + f"{level:.3f}".rstrip("0").split(".")[1]


def _no_holiday(columns) -> list[str]:
    return [c for c in columns if c not in HOLIDAY_COLUMNS]


def _point_rows(ctx, horizon, fold, name, pred_cal, pred_score, **extra):
    targets, tau, cal, score = ctx["targets"], ctx["tau"], ctx["cal"], ctx["score"]
    y_cal = targets.loc[cal, "y"].to_numpy(dtype=float)
    pc = np.asarray(pred_cal, dtype=float)
    ok = np.isfinite(pc) & np.isfinite(y_cal)
    cutoff = _cutoff(y_cal[ok], pc[ok], tau, targets.loc[cal[ok], "target_time"]) if ok.sum() >= 30 else float("inf")
    return _row(score, horizon, fold, name, targets, tau, np.asarray(pred_score, dtype=float), cutoff, **extra)


def dense_quantile_bundle(x_fit, y_fit, x_stop, y_stop, x_cal, y_cal, cfg, params, horizon, tau, cal_times):
    """Fit dense quantiles, Mondrian-B correct levels >= .9 on the first calibration half."""
    models = fit_quantiles(x_fit, y_fit, x_stop, y_stop, cfg, alphas=list(DENSE_LEVELS), params=params)
    q_cal = predict_quantiles(models, x_cal)
    edges = fit_mondrian_edges(q_cal[.5], cfg.get("conformal", {}).get("mondrian_bins", [.5, .9]))
    bins = assign_mondrian_bins(q_cal[.5], edges)
    half = len(y_cal) // 2
    cutoff_start = min(len(y_cal), half + horizon + 1)
    corrections = {}
    qc = dict(q_cal)
    for level in [lv for lv in DENSE_LEVELS if lv >= .9]:
        corrections[level] = fit_conformal(y_cal[:half], q_cal[level][:half], level, bins[:half])
        qc[level] = apply_conformal(q_cal[level], corrections[level], bins)
    qc = _ordered_quantiles(qc)
    p_cal = exceedance_from_quantiles(qc, tau)
    cutoff = _cutoff(y_cal[cutoff_start:], p_cal[cutoff_start:], tau, cal_times[cutoff_start:])
    return {"models": models, "edges": edges, "corrections": corrections, "cutoff": cutoff}


def dense_quantile_predict(bundle, x, tau):
    raw = predict_quantiles(bundle["models"], x)
    bins = assign_mondrian_bins(raw[.5], bundle["edges"])
    corrected = dict(raw)
    for level, correction in bundle["corrections"].items():
        corrected[level] = apply_conformal(raw[level], correction, bins)
    corrected = _ordered_quantiles(corrected)
    p = exceedance_from_quantiles(corrected, tau)
    extras = {"q10": raw[.1], "q50": raw[.5], "q90": raw[.9], "q95": raw[.95], "q975": raw[.975],
              "q90_cal": corrected[.9], "q95_cal": corrected[.95], "q975_cal": corrected[.975],
              "p_exceed": p, "q50_top_edge": float(bundle["edges"][-1]), "conformal_method": "b_dense"}
    extras.update({dense_name(lv): corrected[lv] for lv in DENSE_LEVELS})
    return raw[.5], p, extras


def _evidence_hashes(root: Path) -> dict:
    out = {}
    for relative in EVIDENCE:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"P4/P5 adoption evidence missing: {relative}")
        data = path.read_bytes().replace(b"\r\n", b"\n")
        out[relative] = hashlib.sha256(data).hexdigest()
    summary = json.loads((root / "outputs/p4/p4_summary.json").read_text(encoding="utf-8"))
    if (summary["point_selection"].get("1") != "p4_mstl_daily" or summary["point_selection"].get("4") != "p4_blend3"
            or summary["risk_selection"].get("1") != "p4_quantile_dense"):
        raise AssertionError("P4 summary does not authorize the integrated candidates")
    p5 = json.loads((root / "outputs/p4/p5_summary.json").read_text(encoding="utf-8"))
    if p5["point_selection_after_p5"].get("1") != "p4_mstl_daily" or p5["point_selection_after_p5"].get("4") != "p4_blend3":
        raise AssertionError("P5 changed the P4 selection; integration must be re-registered")
    return out


def generate_candidates(history: pd.DataFrame, cfg: dict, selection: dict, folds: list[dict]) -> tuple[pd.DataFrame, dict]:
    contexts = build_contexts(history, cfg, {"selection": selection, "folds": folds}, horizons=(1, 4))
    series = safe_grid(history)
    rows, meta = [], {"h1_mstl": [], "h4_blend3": [], "h1_dense": []}
    # h1 MSTL: one causal daily path per needed day, shared across folds.
    needed = set()
    for fold in range(3):
        c = contexts[(1, fold)]
        for part in ("fit", "cal", "score"):
            needed |= set(origin_days(c[part]))
    paths = mstl_day_paths(series, sorted(needed))
    for fold in range(3):
        c = contexts[(1, fold)]
        phi = mstl_phi(series, paths, c["fit"])
        pc = mstl_forecast(series, paths, c["cal"], 1, phi)
        ps = mstl_forecast(series, paths, c["score"], 1, phi)
        rows.append(_point_rows(c, 1, fold, "p4_mstl_daily", pc, ps))
        meta["h1_mstl"].append({"fold": fold, "phi": phi, "days": len(paths)})
    # h4 blend of LGBM, CBL representative and DSHW.
    cbl_name = selection["by_horizon"]["4"]["cbl"]
    for fold in range(3):
        c = contexts[(4, fold)]
        cols = _no_holiday(c["x"].columns)
        y = c["targets"].y
        model = fit_point(c["x"].loc[c["fit"], cols], y.loc[c["fit"]], c["x"].loc[c["stop"], cols], y.loc[c["stop"]],
                          cfg, peak_threshold=c["tau"], peak_weight=2.0, params=c["expected"]["chosen_params"])
        dshw = DSHW().fit(series, c["fit"].max())
        comp = {}
        for part in ("cal", "score"):
            cbl = cbl_all_predictions(history, c[part], 4)
            cbl["c3_holiday_hybrid"] = cbl["c3_holiday_mid_4_6"].combine_first(cbl["c1_mid_6_10"])
            comp[part] = [model.predict(c["x"].loc[c[part], cols]), cbl[cbl_name].to_numpy(dtype=float),
                          dshw.predict(series, c[part], 4)]
        w = simplex_weights(comp["cal"], y.loc[c["cal"]].to_numpy(dtype=float))
        mix = lambda part: sum(wi * ci for wi, ci in zip(w, comp[part]))
        rows.append(_point_rows(c, 4, fold, "p4_blend3", mix("cal"), mix("score"),
                                blend_weights=f"lgbm={w[0]},cbl={w[1]},dshw={w[2]}"))
        meta["h4_blend3"].append({"fold": fold, "weights": w, "dshw_params": dshw.params_})
    # h1 dense quantile risk.
    for fold in range(3):
        c = contexts[(1, fold)]
        cols = _no_holiday(c["x"].columns)
        x, y, tau = c["x"][cols], c["targets"].y, c["tau"]
        bundle = dense_quantile_bundle(x.loc[c["fit"]], y.loc[c["fit"]], x.loc[c["stop"]], y.loc[c["stop"]],
                                       x.loc[c["cal"]], y.loc[c["cal"]].to_numpy(), cfg, c["expected"]["chosen_params"],
                                       1, tau, c["targets"].loc[c["cal"], "target_time"])
        q50, p, extras = dense_quantile_predict(bundle, x.loc[c["score"]], tau)
        row = _row(c["score"], 1, fold, "p4_quantile_dense", c["targets"], tau, q50, float("inf"), **extras)
        row["alert"] = p > bundle["cutoff"]
        row["alert_cutoff"] = bundle["cutoff"]
        rows.append(row)
        meta["h1_dense"].append({"fold": fold, "cutoff": bundle["cutoff"]})
    return pd.concat(rows, ignore_index=True, sort=False), meta


def _parity(candidates: pd.DataFrame, root: Path) -> dict:
    report = {}
    keys = ["origin", "target_time", "horizon", "fold"]
    for name, columns in (("p4_mstl_daily", ["pred"]), ("p4_blend3", ["pred"]),
                          ("p4_quantile_dense", ["pred", "p_exceed", "q95_cal"])):
        ref = pd.read_parquet(root / f"outputs/p4/{name}.parquet")
        new = candidates.loc[candidates.model.eq(name)]
        ref = ref.loc[ref.horizon.isin(new.horizon.unique())]
        m = new[keys + columns + ["alert"]].merge(ref[keys + columns + ["alert"]], on=keys, suffixes=("", "_ref"),
                                                  validate="one_to_one", how="outer", indicator=True)
        if m._merge.ne("both").any():
            raise AssertionError(f"{name}: integrated rows differ from research rows")
        diffs = {c: float(np.nanmax(np.abs(m[c].to_numpy(float) - m[c + "_ref"].to_numpy(float)))) for c in columns}
        alert_mismatch = int((m.alert.astype(bool) != m.alert_ref.astype(bool)).sum())
        if max(diffs.values()) > PARITY_TOLERANCE or alert_mismatch:
            raise AssertionError(f"{name}: parity failed {diffs}, alert mismatches {alert_mismatch}")
        report[name] = {"max_abs_diff": diffs, "alert_mismatch": alert_mismatch, "rows": len(new)}
    return report


def run_p4_integration(history, cfg, predictions, metrics, selection, folds, output_dir="outputs", root=None):
    if cfg.get("research_0928_p4", {}).get("enabled") is not True:
        raise ValueError("P4 integration requires explicit config enablement")
    if history.index.max() >= SEALED_BOUNDARY:
        raise ValueError("P4 integration requires sealed development history")
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    evidence = _evidence_hashes(root)
    before = deepcopy(selection)
    candidates, meta = generate_candidates(history, cfg, selection, folds)
    if pd.to_datetime(candidates.target_time).ge(SEALED_BOUNDARY).any():
        raise AssertionError("P4 candidate crossed the sealed boundary")
    parity = _parity(candidates, root)
    predictions = pd.concat([predictions, candidates], ignore_index=True, sort=False)
    metrics = evaluate_all(predictions, cfg=cfg)
    after = deepcopy(selection)
    after["by_horizon"]["1"]["point_model"] = "p4_mstl_daily"
    after["by_horizon"]["1"]["risk_model"] = "p4_quantile_dense"
    after["by_horizon"]["4"]["point_model"] = "p4_blend3"
    after["p4_integration"] = {"evidence_sha256": evidence, "parity": parity,
                               "before": {h: {"point_model": before["by_horizon"][h]["point_model"],
                                              "risk_model": before["by_horizon"][h].get("risk_model",
                                                                                         f"lgbm_quantile_{before['by_horizon'][h].get('conformal')}")}
                                          for h in ("1", "4", "16", "96")},
                               "chronos2": "P5 promoted, no candidate passed; not integrated"}
    logs = Path(output_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "P4_integration.json").write_text(json.dumps({"meta": meta, **after["p4_integration"]},
                                                         ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"predictions": predictions, "metrics": metrics, "selection": after, "before": before}
