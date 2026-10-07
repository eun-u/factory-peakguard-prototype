"""FG-R8: exact-copy gate + zero-shot Chronos-2, with peak shift and conformal risk.

Per (origin, horizon):
- gate (FG-R7 rule): today's prefix matches an earlier day exactly and the target
  is on the origin's calendar day -> most recent exact day's value;
- otherwise the Chronos-2 (context 2048) median;
- B8: for non-gated forecasts within 25 of the fit-only peak threshold, add a shift
  chosen on that week's CAL rows (grid 0..15);
- risk: Chronos q05/q10/q90/q95 conformalized on CAL (CQR, pooled horizons per week);
  gated rows are treated as known; exceedance probability by quantile interpolation;
- alert threshold per (week, horizon) maximizes CAL episode F1 (eval_protocol 6).

Selection used all 16 development weeks (EXPLORE + CONFIRM). Holdout is never loaded.
Chronos forecasts come from phase_f.gate_chronos_infer (run in outputs/phase_f/env).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from phase_f.goal_full_analog import HORIZONS, ROOT, load
from phase_f.goal_full_analog_v2 import Features, gate

NAMESPACE = "goal_gate_chronos_v1"
OUT = ROOT / "outputs/phase_f" / NAMESPACE
CHRONOS = OUT / "cache" / "chronos_c2048.parquet"
CONFIG = {"id": "FG-R8-gate-chronos", "chronos_context": 2048, "point": "median",
          "shift_margin": 25.0, "shift_grid": list(range(16)), "cqr_min_cal": 30,
          "levels": [.05, .1, .5, .9, .95]}


def _frame(F, contexts, chronos, fold, role):
    parts = []
    for h in HORIZONS:
        c = contexts[(h, fold)]
        o = c[role]
        A = F.analog(o, h)
        k = pd.DataFrame({"horizon": h, "fold": fold, "role": role, "origin": o,
                          "target_time": pd.DatetimeIndex(c["target_time"].loc[o]),
                          "y": c["y"].loc[o].to_numpy(float), "tau": c["tau"],
                          "d2": c["d2"].loc[o].to_numpy(bool),
                          "fit_mean": float(c["y"].loc[c["fit"]].mean()),
                          "analog": A.a_today_pred.to_numpy(), "gated": gate(A)})
        parts.append(k)
    k = pd.concat(parts, ignore_index=True)
    k = k.merge(chronos, on=["horizon", "origin"], how="left", validate="one_to_one")
    if k.ch_q50.isna().any():
        raise ValueError("Missing Chronos forecast for an evaluation origin")
    return k


def predict(F, contexts, chronos, folds, config=CONFIG):
    frames = []
    for f in folds:
        w = pd.concat([_frame(F, contexts, chronos, f, r) for r in ("cal", "score")], ignore_index=True)
        base = np.where(w.gated, w.analog, w.ch_q50)
        hi = ~w.gated & (base > w.tau - config["shift_margin"])
        cal = hi & w.role.eq("cal")
        shift = 0.
        if cal.sum() >= 20:
            grid = np.array(config["shift_grid"], float)
            shift = float(grid[np.argmin([np.mean(np.abs(base[cal] + v - w.y[cal])) for v in grid])])
        w["point_r0"] = base
        w["pred"] = np.where(hi, base + shift, base)
        w["shift"] = shift
        calng = w.role.eq("cal") & ~w.gated
        for level, col, lower in ((.95, "ch_q95", False), (.9, "ch_q90", False),
                                  (.1, "ch_q10", True), (.05, "ch_q05", True)):
            err = ((w[col] - w.y) if lower else (w.y - w[col]))[calng].to_numpy()
            n = len(err)
            target = (1 - level) if lower else level
            adj = float(np.quantile(err, min(1., np.ceil((n + 1) * target) / n))) if n >= config["cqr_min_cal"] else 0.
            w[f"q{int(round(level * 100)):02d}"] = np.where(w.gated, w.analog, w[col] - adj if lower else w[col] + adj)
        w["q50"] = w.pred
        levels = np.array(config["levels"])
        Q = np.sort(w[["q05", "q10", "q50", "q90", "q95"]].to_numpy(), 1)
        p = np.array([1 - np.interp(t, q, levels, left=.02, right=.98) for t, q in zip(w.tau, Q)])
        w["p_exceed"] = np.where(w.gated, (w.analog > w.tau).astype(float), p)
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


def point_metrics(frame, col="pred"):
    s = frame[frame.role.eq("score")]
    rows = []
    for h, g in s.groupby("horizon"):
        e = (g[col] - g.y).abs()
        peak = g.y > g.tau
        rows.append({"MAE": e.mean(), "Peak": e[peak].mean(), "nMAE": e.mean() / g.fit_mean.mean(),
                     "D2": e[g.d2].mean()})
    r = pd.DataFrame(rows)
    return {"AUC_MAE": float(r.MAE.mean()), "AUC_PeakMAE": float(r.Peak.mean()),
            "h4_MAE": float(r.MAE.iloc[0]), "h16_MAE": float(r.MAE.iloc[-1]),
            "AUC_nMAE": float(r.nMAE.mean()), "D2_AUC_MAE": float(r.D2.mean())}


def risk_metrics(frame):
    from sklearn.metrics import average_precision_score, brier_score_loss
    s = frame[frame.role.eq("score")]
    ng = s[~s.gated]
    peak = s.y > s.tau
    return {"coverage_q90_nongated": float((ng.y <= ng.q90).mean()),
            "coverage_q95_nongated": float((ng.y <= ng.q95).mean()),
            "coverage_q90_all": float((s.y <= s.q90).mean()),
            "coverage_q95_all": float((s.y <= s.q95).mean()),
            "brier": float(brier_score_loss(peak, s.p_exceed)),
            "pr_auc": float(average_precision_score(peak, s.p_exceed))}


def alert_metrics(frame, score_col, horizons=(4, 16)):
    from src.evaluate import match_episodes, score_predictions

    def episode_f1(g, threshold):
        m = match_episodes(g.y > g.tau, g[score_col] > threshold, g.target_time)
        d = 2 * m["tp"] + m["fp"] + m["fn"]
        return 2 * m["tp"] / d if d else -1.

    rows = []
    for h in horizons:
        tot = dict(tp=0, fp=0, fn=0, etp=0, efp=0, efn=0, default_threshold_weeks=0)
        for f in sorted(frame.fold.unique()):
            cal = frame[(frame.fold == f) & (frame.horizon == h) & frame.role.eq("cal")].sort_values("target_time")
            sc = frame[(frame.fold == f) & (frame.horizon == h) & frame.role.eq("score")].sort_values("target_time").copy()
            if (cal.y > cal.tau).any():
                cands = np.unique(np.quantile(cal[score_col], np.linspace(.5, .999, 120)))
                thr = float(cands[int(np.argmax([episode_f1(cal, c) for c in cands]))])
            else:
                thr = float(sc.tau.iloc[0]) if score_col != "p_exceed" else .5
                tot["default_threshold_weeks"] += 1
            sc["alert"] = sc[score_col] > thr
            r = score_predictions(sc[["target_time", "y", "pred", "tau", "alert"]])
            for a, b in (("tp", "position_tp"), ("fp", "position_fp"), ("fn", "position_fn"),
                         ("etp", "episode_tp"), ("efp", "episode_fp"), ("efn", "episode_fn")):
                tot[a] += r[b]
        s = frame[frame.role.eq("score") & frame.horizon.eq(h)]
        rows.append({"horizon": h, "alert_score": score_col,
                     "position_F1": 2 * tot["tp"] / (2 * tot["tp"] + tot["fp"] + tot["fn"]),
                     "episode_F1": 2 * tot["etp"] / (2 * tot["etp"] + tot["efp"] + tot["efn"]),
                     "episode_TP": tot["etp"], "episode_FP": tot["efp"], "episode_FN": tot["efn"],
                     "position_false_alarms": tot["fp"],
                     "peak_location_MAE": float((s.pred - s.y).abs()[s.y > s.tau].mean()),
                     "default_threshold_weeks": tot["default_threshold_weeks"]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chronos", default=str(CHRONOS))
    args = ap.parse_args(argv)
    t0 = time.time()
    history, contexts, keys = load()
    chronos = pd.read_parquet(args.chronos)
    F = Features(history)
    folds = {arm: sorted(keys.loc[keys.arm.eq(arm), "fold"].unique()) for arm in ("EXPLORE", "CONFIRM")}
    frame = predict(F, contexts, chronos, folds["EXPLORE"] + folds["CONFIRM"])
    frame["arm"] = np.where(frame.fold.isin(folds["EXPLORE"]), "EXPLORE", "CONFIRM")
    record = {"config": CONFIG, "selection": "16 development weeks (EXPLORE+CONFIRM)", "holdout_read": False}
    for arm, sel in (("ALL16", frame), ("EXPLORE", frame[frame.arm.eq("EXPLORE")]),
                     ("CONFIRM", frame[frame.arm.eq("CONFIRM")])):
        record[arm] = {"point": point_metrics(sel), "point_without_shift": point_metrics(sel, "point_r0"),
                       "risk": risk_metrics(sel),
                       "alerts": alert_metrics(sel, "pred") + alert_metrics(sel, "p_exceed")}
    from phase_f.goal_protocol import meets_targets
    record["targets_ALL16"] = meets_targets(record["ALL16"]["point"])
    record["wall_seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DEV16_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (OUT / "predictions").mkdir(exist_ok=True)
    frame.to_parquet(OUT / "predictions" / "dev16_predictions.parquet", index=False)
    print(json.dumps({k: record[k]["point"] for k in ("ALL16", "EXPLORE", "CONFIRM")}, indent=2))
    print(record["targets_ALL16"])
    return record


if __name__ == "__main__":
    main()
