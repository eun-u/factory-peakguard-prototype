"""E2: walk-forward operations replay for h4 (frozen vs weekly vs 4-weekly retraining). Development only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import match_episodes
from src.features import build_features
from src.models.cbl import cbl_all_predictions
from src.models.conformal import apply_conformal, assign_mondrian_bins, fit_conformal, fit_mondrian_edges
from src.models.lgbm_point import fit_point, point_grid
from src.models.lgbm_quantile import fit_quantiles, predict_quantiles
from src.models.peak_prob import exceedance_from_quantiles
from src.models.seasonal import DSHW, safe_grid, simplex_weights
from src.session_data import SEALED_BOUNDARY
from src.targets import point_targets, training_peak_threshold
from src.training import _cutoff, _ordered_quantiles, _valid_mask
from .common import Timer, config, history
from .p6_common import HOLIDAY, out_dir, write_json

H = 4
GAP = pd.Timedelta(minutes=15 * H)
WEEKS = pd.date_range("2021-05-03", "2021-08-02", freq="7D")


def train_at(df, cfg, series, T):
    dev = df.index[df.index + GAP < SEALED_BOUNDARY]
    origins = dev
    targets = point_targets(df, origins, H)
    x, used = build_features(df, origins, H, {**cfg, "_tau": None})
    good = origins[_valid_mask(x, used, origins, targets, df)]
    train = good[good + GAP < T]
    n = len(train)
    fit, stop, cal = train[:int(n * .64)], train[int(n * .64):int(n * .80)], train[int(n * .80):]
    fit = fit[fit + GAP < stop.min()]
    stop = stop[stop + GAP < cal.min()]
    tau = training_peak_threshold(df, fit, .95)
    for _ in range(3):
        x, used = build_features(df, origins, H, {**cfg, "_tau": tau})
        good = origins[_valid_mask(x, used, origins, targets, df)]
        fit, stop, cal = (p.intersection(good) for p in (fit, stop, cal))
        exact = training_peak_threshold(df, fit, .95)
        if exact == tau:
            break
        tau = exact
    y = targets.y
    best = min(((float(np.mean(np.abs(y.loc[stop].to_numpy() - fit_point(x.loc[fit], y.loc[fit], x.loc[stop], y.loc[stop], cfg,
                                                                            peak_threshold=tau, params=p).predict(x.loc[stop])))), i, p)
                for i, p in enumerate(point_grid(cfg))))
    params = best[2]
    cols = [c for c in x.columns if c not in HOLIDAY]
    lgbm = fit_point(x.loc[fit, cols], y.loc[fit], x.loc[stop, cols], y.loc[stop], cfg, peak_threshold=tau, peak_weight=2.0, params=params)
    dshw = DSHW().fit(series, fit.max())
    evaluation = good[good >= T]

    def components(idx):
        cb = cbl_all_predictions(df, idx, H)
        return [lgbm.predict(x.loc[idx, cols]), cb["c2a_max_4_5_adjusted"].to_numpy(float), dshw.predict(series, idx, H)]
    comp_cal, comp_eval = components(cal), components(evaluation)
    y_cal = y.loc[cal].to_numpy(float)
    w = simplex_weights(comp_cal, y_cal)
    blend_cal = sum(a * b for a, b in zip(w, comp_cal))
    blend_eval = sum(a * b for a, b in zip(w, comp_eval))
    ok = np.isfinite(blend_cal)
    point_cutoff = _cutoff(y_cal[ok], blend_cal[ok], tau, targets.loc[cal[ok], "target_time"])
    q_models = fit_quantiles(x.loc[fit, cols], y.loc[fit], x.loc[stop, cols], y.loc[stop], cfg, params=params)
    qc, qe = predict_quantiles(q_models, x.loc[cal, cols]), predict_quantiles(q_models, x.loc[evaluation, cols])
    edges = fit_mondrian_edges(qc[.5], [.5, .9])
    bc, be = assign_mondrian_bins(qc[.5], edges), assign_mondrian_bins(qe[.5], edges)
    half = len(cal) // 2
    start = half + H + 1
    qct, qet = dict(qc), dict(qe)
    for a in (.9, .95, .975):
        corr = fit_conformal(y_cal[:half], qc[a][:half], a, bc[:half])
        qct[a], qet[a] = apply_conformal(qc[a], corr, bc), apply_conformal(qe[a], corr, be)
    qct, qet = _ordered_quantiles(qct), _ordered_quantiles(qet)
    pc, pe = exceedance_from_quantiles(qct, tau), exceedance_from_quantiles(qet, tau)
    prob_cutoff = _cutoff(y_cal[start:], pc[start:], tau, targets.loc[cal[start:], "target_time"])
    frame = pd.DataFrame({"origin": evaluation, "target_time": targets.loc[evaluation, "target_time"].to_numpy(),
                          "y": y.loc[evaluation].to_numpy(float), "blend": blend_eval, "point_alert": blend_eval > point_cutoff,
                          "p_exceed": pe, "risk_alert": pe > prob_cutoff, "q50": qe[.5], "q95_cal": qet[.95],
                          "top_edge": float(edges[-1])})
    frame = frame.loc[frame.target_time < SEALED_BOUNDARY]
    meta = {"trained_at": str(T), "tau": tau, "fit_end": str(fit.max()), "cal_end": str(cal.max()), "weights": w,
            "params": params, "n_fit": len(fit), "n_cal": len(cal), "point_cutoff": point_cutoff, "prob_cutoff": prob_cutoff}
    return frame, meta


def week_metrics(part, tau_common):
    part = part.sort_values("target_time")
    y = part.y.to_numpy()
    peak = y > tau_common
    times = pd.DatetimeIndex(part.target_time)
    out = {"n": len(part), "mae": float(np.nanmean(np.abs(y - part.blend))), "peak_n": int(peak.sum()),
           "peak_mae": float(np.nanmean(np.abs(y[peak] - part.blend.to_numpy()[peak]))) if peak.any() else np.nan}
    for col in ("risk_alert", "point_alert"):
        m = match_episodes(peak, part[col].to_numpy(bool), times)
        den = 2 * m["tp"] + m["fp"] + m["fn"]
        out[f"{col}_episode_f1"] = 2 * m["tp"] / den if den else np.nan
        out[f"{col}_tp_fp_fn"] = (m["tp"], m["fp"], m["fn"])
    hi = part.q50.to_numpy() >= part.top_edge.to_numpy()
    out["top_q95_coverage"] = float(np.mean(y[hi] <= part.q95_cal.to_numpy()[hi])) if hi.any() else np.nan
    out["top_n"] = int(hi.sum())
    return out


def run():
    cfg, df = config(), history()
    series = safe_grid(df)
    out = out_dir("e2_walkforward")
    models, metas = {}, []
    for T in WEEKS:
        t = Timer()
        frame, meta = train_at(df, cfg, series, T)
        models[T] = frame
        meta["seconds"] = t()
        metas.append(meta)
        print(meta, flush=True)
        frame.assign(trained_at=T).to_parquet(out / f"pred_{T.date()}.parquet", index=False)
    tau_common = metas[0]["tau"]
    rows = []
    for k, week_start in enumerate(WEEKS):
        week_end = WEEKS[k + 1] if k + 1 < len(WEEKS) else SEALED_BOUNDARY
        for strategy, T in (("frozen", WEEKS[0]), ("weekly", week_start), ("every4weeks", WEEKS[4 * (k // 4)])):
            f = models[T]
            part = f.loc[(f.origin >= week_start) & (f.origin < week_end)]
            rows.append({"week_start": str(week_start.date()), "strategy": strategy, "trained_at": str(T.date()),
                         **week_metrics(part, tau_common)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "weekly_metrics.csv", index=False)
    pooled = {}
    for s, g in table.groupby("strategy"):
        pooled[s] = {"mae_mean": float(g.mae.mean()), "peak_mae_weighted": float(np.nansum(g.peak_mae * g.peak_n) / g.peak_n.sum()),
                     "risk_f1_mean": float(g.risk_alert_episode_f1.mean()), "top_cov_mean": float(g.top_q95_coverage.mean()),
                     "late_weeks_top_cov": float(g.loc[g.week_start >= "2021-07-05"].top_q95_coverage.mean())}
    write_json(out / "summary.json", {"tau_common": tau_common, "pooled": pooled, "trainings": metas})
    print(pooled)


if __name__ == "__main__":
    run()
