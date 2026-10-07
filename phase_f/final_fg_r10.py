"""FG-R10 third test evaluation: FG-R10 + per-horizon linear MOS on Chronos-2 realized errors.

MOS (model output statistics): y - q50 ~ Huber linear in Chronos errors already observed at the
origin (forecasts issued 4/8/16 slots earlier for the origin time), their recent means, and the
Chronos quantile spread/skew. Trained on dev non-gated rows with targets before the final CAL window.
Dev evidence: outputs/phase_f/goal_fm_ensemble_v1/MOS_RESULTS.md. Derived from final_fg_r9.py.

FG-R10 notes:

Derived from phase_f/final_fg_r8.py. Pre-registration: outputs/phase_f/goal_fm_ensemble_v1/PREREGISTRATION.md.
The test window was already evaluated once with FG-R8; this is a disclosed second look.

Original FG-R8 notes:

Stages (run in this order; each refuses to repeat):
  lock     dev data only: tau, final-CAL parameters, FG-R6 baseline models, hashes
  infer    (Chronos env) Chronos-2 forecasts for test origins and 23:45 day-ahead origins
  evaluate full data once: apply locked parameters, metrics, CIs, prediction CSVs

Test origins: 2021-08-09 09:45 onward with a valid target before the data end.
Inputs at an origin use power at timestamps <= origin only. A zero reading is
masked at its own slot (known when observed); the whole-hour gap rule is used
only to invalidate evaluation targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase_f/final_fg_r10"
R8_TEST_CHRONOS = ROOT / "outputs/phase_f/final_fg_r8/cache/chronos_test.parquet"
MIN_SLOTS = 5  # dev score gate precision >= 0.95 from 5 observed slots of the day
LOCK = OUT / "FINAL_LOCK.json"
BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
HORIZONS = tuple(range(4, 17))
Q = pd.Timedelta(minutes=15)
CAL_DAYS = 28  # revised from 7 before any test read: the last dev week had 0 peak rows
ALERT_SCORES = ("persistence", "b1_weekly", "analog_only", "chronos_alone", "fg_r6", "pred", "p_exceed")
SOURCES = ("phase_f/final_fg_r10.py", "phase_f/final_fg_r9.py", "phase_f/final_fg_r8.py", "phase_f/goal_gate_chronos.py", "phase_f/gate_chronos_infer.py",
           "phase_f/goal_full_analog.py", "phase_f/goal_full_analog_v2.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_once(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path.name} already exists; the final evaluation is one-shot")
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


MOS_COLS = ("r4", "r4_m4", "r4_m16", "r8", "r8_m4", "r16_m4", "spread", "skew")


def mos_features(chronos, power):
    """Realized Chronos errors observable at each origin (targets <= origin)."""
    piv = chronos.pivot(index="origin", columns="horizon", values="ch_q50")
    err = {}
    for k in (4, 8, 16):
        prev = piv[k].copy()
        prev.index = prev.index + Q * k
        err[k] = power.reindex(prev.index) - prev
    out = pd.DataFrame(index=piv.index)
    out["r4"] = err[4].reindex(out.index)
    out["r4_m4"] = err[4].rolling(4, min_periods=1).mean().reindex(out.index)
    out["r4_m16"] = err[4].rolling(16, min_periods=2).mean().reindex(out.index)
    out["r8"] = err[8].reindex(out.index)
    out["r8_m4"] = err[8].rolling(4, min_periods=1).mean().reindex(out.index)
    out["r16_m4"] = err[16].rolling(4, min_periods=1).mean().reindex(out.index)
    return out


def add_mos(frame, feats, coefs):
    f = frame.merge(feats, left_on="origin", right_index=True, how="left")
    f["spread"] = f.ch_q90 - f.ch_q10
    f["skew"] = (f.ch_q90 - f.ch_q50) - (f.ch_q50 - f.ch_q10)
    X = f[list(MOS_COLS)].fillna(0).to_numpy()
    adj = np.zeros(len(f))
    for h, c in coefs.items():
        s = (f.horizon == int(h)).to_numpy()
        adj[s] = c["intercept"] + X[s] @ np.array(c["coef"])
    frame = frame.copy()
    frame["base50"] = frame.ch_q50.to_numpy() + adj
    return frame


# ------------------------------------------------------------------ data
def full_history():
    """Full source with dev-equivalent cleaning; dev part must equal phase_c."""
    from phase_c.data import _gap_hours, load_history
    from src.data import load_power_data
    raw, _ = load_power_data(ROOT / "data/raw/task05_power/okm_augumented_2021.csv")
    h = raw.copy()
    h["power_raw"] = pd.to_numeric(h["power"], errors="coerce")
    h["gap_hour"] = _gap_hours(h)
    h["target_power"] = h.power_raw.mask(h.time_repaired | h.gap_hour)   # evaluation labels
    h["power"] = h.power_raw.mask(h.time_repaired | h.power_raw.eq(0))    # causal inputs
    dev, _ = load_history(ROOT)
    a = h.power.reindex(dev.index)
    b = dev.power
    same = (a.isna() & b.isna()) | (a == b)
    if not same.all():
        raise RuntimeError(f"Full-data inputs differ from sealed dev history at {int((~same).sum())} slots")
    return h, dev


def _rows(F, history, origins, h, label):
    A = F.analog(origins, h)
    t = pd.DatetimeIndex(origins) + Q * h
    from phase_f.goal_full_analog_v2 import gate
    return pd.DataFrame({"origin": origins, "horizon": h, "target_time": t,
                         "y": label.reindex(t).to_numpy(float), "analog": A.a_today_pred.to_numpy(),
                         "gated": gate(A) & (A.a_dayslot.to_numpy() >= MIN_SLOTS), "lag0": history.power.reindex(origins).to_numpy(),
                         "tw1": history.power.reindex(t - pd.Timedelta(days=7)).to_numpy()})


def cal_origins(dev):
    """Last CAL_DAYS of dev targets: origins whose every horizon target is < BOUNDARY."""
    idx = dev.index
    start = BOUNDARY - pd.Timedelta(days=CAL_DAYS) - Q * 16
    return idx[(idx >= start) & (idx + Q * 16 < BOUNDARY)]


def apply_params(frame, params, tau):
    """Locked FG-R8 post-processing for one frame (cal or test)."""
    point = frame.base50 if "base50" in frame else frame.ch_q50
    base = np.where(frame.gated, frame.analog, point)
    hi = ~frame.gated & (base > tau - params["shift_margin"])
    frame = frame.copy()
    frame["point_r0"] = base
    frame["pred"] = np.where(hi, base + params["shift"], base)
    frame["q05"] = np.where(frame.gated, frame.analog, frame.ch_q05 - params["cqr"]["q05"])
    frame["q10"] = np.where(frame.gated, frame.analog, frame.ch_q10 - params["cqr"]["q10"])
    frame["q90"] = np.where(frame.gated, frame.analog, frame.ch_q90 + params["cqr"]["q90"])
    frame["q95"] = np.where(frame.gated, frame.analog, frame.ch_q95 + params["cqr"]["q95"])
    lv = np.array([.05, .1, .5, .9, .95])
    Qm = np.sort(np.stack([frame.q05, frame.q10, frame.pred, frame.q90, frame.q95], 1), 1)
    p = np.array([1 - np.interp(tau, q, lv, left=.02, right=.98) for q in Qm])
    frame["p_exceed"] = np.where(frame.gated, (frame.analog > tau).astype(float), p)
    return frame


# ------------------------------------------------------------------ lock
def stage_lock():
    from phase_f.goal_full_analog_v2 import Features
    from phase_f import goal_full_analog as v1
    from phase_f.goal_gate_chronos import CHRONOS
    from phase_c.data import load_history
    from phase_f.wf_harness import build_weekly_contexts
    if LOCK.exists():
        raise SystemExit("Lock exists")
    dev, _ = load_history(ROOT)
    tau = float(np.nanquantile(dev.power.to_numpy(float), .95))
    F = Features(dev)
    chronos = pd.read_parquet(CHRONOS)
    co = cal_origins(dev)
    cal = pd.concat([_rows(F, dev, co, h, dev.power) for h in HORIZONS], ignore_index=True)
    cal = cal.merge(chronos, on=["origin", "horizon"], how="left")
    dropped = int((cal.ch_q50.isna() | cal.y.isna()).sum())
    cal = cal[cal.ch_q50.notna() & cal.y.notna()]
    if len(cal) < 1000:
        raise RuntimeError("Too few final CAL rows with Chronos forecasts")
    # MOS training rows: dev non-gated, targets before the final CAL window.
    from sklearn.linear_model import HuberRegressor
    feats_dev = mos_features(chronos, dev.power)
    start = cal.origin.min()
    tr_parts = []
    for h in HORIZONS:
        o = pd.DatetimeIndex(sorted(chronos.loc[chronos.horizon.eq(h), "origin"]))
        o = o[o + Q * h < start]
        tr_parts.append(_rows(F, dev, o, h, dev.power))
    tr = pd.concat(tr_parts, ignore_index=True).merge(chronos, on=["origin", "horizon"], how="left")
    tr = tr[~tr.gated & tr.y.notna() & tr.ch_q50.notna()]
    tr = tr.merge(feats_dev, left_on="origin", right_index=True, how="left")
    tr["spread"] = tr.ch_q90 - tr.ch_q10
    tr["skew"] = (tr.ch_q90 - tr.ch_q50) - (tr.ch_q50 - tr.ch_q10)
    coefs = {}
    for h in HORIZONS:
        a = tr[tr.horizon.eq(h)]
        m = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000).fit(a[list(MOS_COLS)].fillna(0), a.y - a.ch_q50)
        coefs[str(h)] = {"intercept": float(m.intercept_), "coef": [float(v) for v in m.coef_], "rows": int(len(a))}
    cal = add_mos(cal, feats_dev, coefs)
    base = np.where(cal.gated, cal.analog, cal.base50)
    margin = 25.
    hi = ~cal.gated & (base > tau - margin)
    grid = np.arange(16.)
    shift = float(grid[np.argmin([np.mean(np.abs(base[hi] + v - cal.y[hi])) for v in grid])]) if hi.sum() >= 20 else 0.
    ng = cal[~cal.gated]
    cqr = {}
    for name, col, lower, level in (("q05", "ch_q05", True, .05), ("q10", "ch_q10", True, .1),
                                    ("q90", "ch_q90", False, .9), ("q95", "ch_q95", False, .95)):
        err = ((ng[col] - ng.y) if lower else (ng.y - ng[col])).to_numpy()
        n = len(err)
        target = (1 - level) if lower else level
        cqr[name] = float(np.quantile(err, min(1., np.ceil((n + 1) * target) / n)))
    params = {"shift_margin": margin, "shift": shift, "cqr": cqr}
    cal = apply_params(cal, params, tau)
    # FG-R6 baseline: 5 seeds trained on the last development fold's fit/stop.
    contexts = build_weekly_contexts(dev)
    last = max(f for _, f in contexts)
    F1 = v1.Features(dev)
    import joblib
    import lightgbm as lgb
    Xf, yf, _ = v1._stack(F1, contexts, last, "fit")
    Xs, ys, _ = v1._stack(F1, contexts, last, "stop")
    models = []
    for seed in v1.CONFIG["seeds"]:
        m = lgb.LGBMRegressor(objective=v1.CONFIG["objective"], **v1.CONFIG["params"], verbose=-1, n_jobs=8, seed=seed)
        w = np.where(v1.gate(Xf), 1., v1.CONFIG["non_gated_weight"])
        m.fit(Xf, yf, sample_weight=w, eval_set=[(Xs, ys)],
              eval_sample_weight=[np.where(v1.gate(Xs), 1., v1.CONFIG["non_gated_weight"])],
              callbacks=[lgb.early_stopping(v1.CONFIG["early_stopping_rounds"], verbose=False)])
        models.append(m)
    cal = cal.sort_values(["horizon", "origin"]).reset_index(drop=True)
    X1 = pd.concat([F1.build(pd.DatetimeIndex(cal.loc[cal.horizon.eq(h), "origin"]), h) for h in HORIZONS], ignore_index=True)
    cal["fg_r6"] = np.where(v1.gate(X1), X1.a_today_pred, np.mean([m.predict(X1) for m in models], 0))
    fallback = float(np.nanmean(dev.power))
    cal["persistence"] = cal.lag0.fillna(fallback)
    cal["b1_weekly"] = cal.tw1.fillna(fallback)
    cal["analog_only"] = cal.analog.fillna(fallback)
    cal["chronos_alone"] = cal.ch_q50
    # Alert thresholds: CAL episode-F1 maximization per horizon, same rule for every model.
    from src.evaluate import match_episodes
    thresholds = {}
    for h in HORIZONS:
        g = cal[cal.horizon.eq(h)].sort_values("target_time")
        for col in ALERT_SCORES:
            default = .5 if col == "p_exceed" else tau
            if (g.y > tau).sum() == 0:
                thresholds[f"{col}_h{h}"] = default
                continue
            cands = np.unique(np.quantile(g[col], np.linspace(.5, .999, 120)))
            f1 = []
            for c in cands:
                m = match_episodes(g.y > tau, g[col] > c, g.target_time)
                d = 2 * m["tp"] + m["fp"] + m["fn"]
                f1.append(2 * m["tp"] / d if d else -1)
            thresholds[f"{col}_h{h}"] = float(cands[int(np.argmax(f1))])
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, OUT / "fg_r6_models.joblib")
    record = {"locked_at": datetime.now().astimezone().isoformat(), "stage": "lock",
              "tau": tau, "final_cal": {"first_origin": str(co.min()), "last_origin": str(co.max()),
                                        "rows": int(len(cal)), "gated_share": float(cal.gated.mean()),
                                        "peak_rows": int((cal.y > tau).sum()), "dropped_rows_without_chronos_or_label": dropped},
              "fg_r8_params": params, "alert_thresholds": thresholds,
              "fg_r6_last_fold": int(last), "fg_r6_models_sha256": _sha(OUT / "fg_r6_models.joblib"),
              "chronos_dev_cache_sha256": _sha(CHRONOS),
              "r8_test_chronos_sha256": _sha(R8_TEST_CHRONOS), "gate_min_slots": MIN_SLOTS,
              "mos": {"columns": list(MOS_COLS), "coefs": coefs, "train_rows": int(len(tr))},
              "source_sha256": {s: _sha(ROOT / s) for s in SOURCES},
              "chronos": {"model": "amazon/chronos-2", "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
                          "context": 2048},
              "models": ["persistence", "B1 weekly naive", "analog only", "Chronos-2 alone", "FG-R6",
                         "FG-R10 point", "FG-R10 probability alerts"],
              "holdout_read": False}
    _write_once(LOCK, record)
    print(json.dumps({k: record[k] for k in ("tau", "final_cal", "fg_r8_params")}, indent=2))


# ------------------------------------------------------------------ infer (Chronos env)
def stage_infer():
    import torch
    from chronos import Chronos2Pipeline
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha(ROOT / "phase_f/final_fg_r8.py") != lock["source_sha256"]["phase_f/final_fg_r8.py"]:
        raise SystemExit("Final source changed after lock")
    out = OUT / "cache" / "chronos_test.parquet"
    if out.exists():
        raise SystemExit("Test Chronos cache exists")
    h, _ = full_history()
    values = h.power.astype("float32").to_numpy()
    pos = pd.Series(np.arange(len(values)), index=h.index)
    end = h.index.max()
    origins = h.index[(h.index >= BOUNDARY) & (h.index + Q * 4 <= end)]
    day_origins = h.index[(h.index >= BOUNDARY) & (h.index.hour == 23) & (h.index.minute == 45)
                          & (h.index + Q * 96 <= end)]
    L = lock["chronos"]["context"]
    pipe = Chronos2Pipeline.from_pretrained(lock["chronos"]["model"], revision=lock["chronos"]["revision"],
                                            device_map="cuda", torch_dtype=torch.float32)

    def series(o, x=values):
        i = pos[o]
        return x[max(0, i - L + 1):i + 1].copy()

    probe = origins[len(origins) // 2]
    alt = values.copy()
    alt[pos[probe] + 1:] = 10000.
    lv = [.05, .1, .5, .9, .95]
    with torch.inference_mode():
        a, _ = pipe.predict_quantiles([series(probe)], prediction_length=16, quantile_levels=lv, context_length=L)
        b, _ = pipe.predict_quantiles([series(probe, alt)], prediction_length=16, quantile_levels=lv, context_length=L)
    diff = float(torch.max(torch.abs(a[0] - b[0])))
    if diff != 0:
        raise AssertionError(diff)
    rows = []
    for s in range(0, len(origins), 64):
        batch = list(origins[s:s + 64])
        with torch.inference_mode():
            q, _ = pipe.predict_quantiles([series(o) for o in batch], prediction_length=16, quantile_levels=lv,
                                          batch_size=64, context_length=L)
        for o, qq in zip(batch, q):
            arr = qq.detach().cpu().numpy()[0]
            rows += [(o, hh, *arr[hh - 1]) for hh in HORIZONS]
    pd.DataFrame(rows, columns=["origin", "horizon", "ch_q05", "ch_q10", "ch_q50", "ch_q90", "ch_q95"]).pipe(
        lambda d: (out.parent.mkdir(parents=True, exist_ok=True), d.to_parquet(out, index=False)))
    with torch.inference_mode():
        q, _ = pipe.predict_quantiles([series(o) for o in day_origins], prediction_length=96, quantile_levels=[.5],
                                      context_length=L)
    pd.DataFrame({"origin": day_origins, "chronos_next_day_max": [float(x[0, :, 0].max()) for x in q]}).to_parquet(
        OUT / "cache" / "chronos_day_ahead.parquet", index=False)
    print({"test_origins": len(origins), "day_ahead_origins": len(day_origins), "perturbation_diff": diff})


# ------------------------------------------------------------------ evaluate
def _metrics(d, col):
    rows = []
    for h, g in d.groupby("horizon"):
        e = (g[col] - g.y).abs()
        pk = g.y > g.tau
        rows.append((h, e.mean(), e[pk].mean() if pk.any() else np.nan, (g[col] - g.y).mean()))
    r = pd.DataFrame(rows, columns=["h", "MAE", "Peak", "bias"]).set_index("h")
    return {"AUC_MAE": r.MAE.mean(), "AUC_PeakMAE": r.Peak.mean(), "h4_MAE": r.MAE.loc[4],
            "h16_MAE": r.MAE.loc[16], "bias": r.bias.mean(), "per_horizon_MAE": r.MAE.round(4).to_dict()}


def _paired_ci(d, ref, col="pred", n=1000, seed=42):
    rng = np.random.default_rng(seed)
    m = d.assign(diff=(d[ref] - d.y).abs() - (d[col] - d.y).abs(), day=d.target_time.dt.normalize())
    groups = {k: g for k, g in m.groupby("day")}
    days = np.array(list(groups))

    def stat(z):
        return z.groupby("horizon")["diff"].mean().mean()
    draws = [stat(pd.concat([groups[x] for x in rng.choice(days, len(days))])) for _ in range(n)]
    return {"improvement": float(stat(m)), "ci95": np.percentile(draws, [2.5, 97.5]).round(4).tolist(),
            "n_days": int(len(days))}


def stage_evaluate():
    from phase_f.goal_full_analog_v2 import Features
    from phase_f import goal_full_analog as v1
    from src.evaluate import score_predictions
    from sklearn.metrics import average_precision_score, brier_score_loss
    import joblib
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for s, digest in lock["source_sha256"].items():
        if _sha(ROOT / s) != digest:
            raise SystemExit(f"{s} changed after lock")
    if _sha(OUT / "fg_r6_models.joblib") != lock["fg_r6_models_sha256"]:
        raise SystemExit("FG-R6 models changed after lock")
    result_path = OUT / "FINAL_TEST_RESULT.json"
    if result_path.exists():
        raise SystemExit("Final evaluation already done")
    t0 = time.time()
    h, dev = full_history()
    tau = lock["tau"]
    label = h.target_power
    if _sha(R8_TEST_CHRONOS) != lock["r8_test_chronos_sha256"]:
        raise SystemExit("Reused Chronos test cache changed")
    chronos = pd.read_parquet(R8_TEST_CHRONOS)
    F = Features(h)
    F1 = v1.Features(h)
    models = joblib.load(OUT / "fg_r6_models.joblib")
    frames = []
    for hz in HORIZONS:
        origins = pd.DatetimeIndex(sorted(chronos.loc[chronos.horizon.eq(hz), "origin"]))
        t = origins + Q * hz
        origins = origins[label.reindex(t).notna().to_numpy() & ~h.time_repaired.reindex(t).fillna(True).to_numpy()]
        d = _rows(F, h, origins, hz, label)
        X1 = F1.build(origins, hz)
        d["fg_r6"] = np.where(v1.gate(X1), X1.a_today_pred, np.mean([m.predict(X1) for m in models], 0))
        frames.append(d)
    d = pd.concat(frames, ignore_index=True).merge(chronos, on=["origin", "horizon"], how="left", validate="one_to_one")
    d["tau"] = tau
    from phase_f.goal_gate_chronos import CHRONOS as DEV_CHRONOS
    both = pd.concat([pd.read_parquet(DEV_CHRONOS), chronos], ignore_index=True).drop_duplicates(["origin", "horizon"], keep="last")
    d = add_mos(d, mos_features(both, h.power), lock["mos"]["coefs"])
    d["chronos_mos"] = d.base50
    d = apply_params(d, lock["fg_r8_params"], tau)
    d["persistence"] = d.lag0
    d["b1_weekly"] = d.tw1
    d["analog_only"] = d.analog
    d["chronos_alone"] = d.ch_q50
    for c in ("persistence", "b1_weekly", "analog_only", "chronos_alone", "fg_r6"):
        d[c] = d[c].fillna(float(np.nanmean(dev.power)))  # same fallback for every model with a missing input
    cols = {"persistence": "persistence", "B1 weekly naive": "b1_weekly", "analog only": "analog_only",
            "Chronos-2 alone": "chronos_alone", "Chronos-2 + MOS": "chronos_mos", "FG-R6": "fg_r6", "FG-R10 without peak shift": "point_r0",
            "FG-R10": "pred"}
    result = {"evaluated_at": datetime.now().astimezone().isoformat(), "lock_sha256": _sha(LOCK),
              "test_first_origin": str(d.origin.min()), "test_last_target": str(d.target_time.max()),
              "rows": int(len(d)), "test_days": int(d.target_time.dt.normalize().nunique()),
              "gated_share": float(d.gated.mean()), "peak_rows_share": float((d.y > tau).mean()),
              "tau": tau, "point": {}, "paired_ci_vs_fg_r10": {}, "alerts": {}, "risk": {}}
    for name, col in cols.items():
        result["point"][name] = _metrics(d, col)
        if col != "pred":
            result["paired_ci_vs_fg_r10"][name] = _paired_ci(d, col)
    ng = d[~d.gated]
    peak = d.y > tau
    result["risk"] = {"coverage_q90_nongated": float((ng.y <= ng.q90).mean()),
                      "coverage_q95_nongated": float((ng.y <= ng.q95).mean()),
                      "coverage_q90_all": float((d.y <= d.q90).mean()), "coverage_q95_all": float((d.y <= d.q95).mean()),
                      "brier": float(brier_score_loss(peak, d.p_exceed)),
                      "pr_auc": float(average_precision_score(peak, d.p_exceed)) if peak.any() else None}
    for hz in (4, 16):
        g = d[d.horizon.eq(hz)].sort_values("target_time")
        for name, col in (("persistence", "persistence"), ("B1 weekly naive", "b1_weekly"),
                          ("analog only", "analog_only"), ("Chronos-2 alone", "chronos_alone"),
                          ("FG-R6", "fg_r6"), ("FG-R10 point", "pred"), ("FG-R10 probability", "p_exceed")):
            thr = lock["alert_thresholds"][f"{col}_h{hz}"]
            frame = g.assign(pred=g[col] if col != "p_exceed" else g.pred, alert=g[col] > thr)
            r = score_predictions(frame[["target_time", "y", "pred", "tau", "alert"]])
            result["alerts"][f"{name}|h{hz}"] = {k: r[k] for k in (
                "position_f1", "episode_f1", "episode_tp", "episode_fp", "episode_fn",
                "false_alarms_positions", "peak_mae")} | {"threshold": thr}
    # Auxiliary A1 task: next-day maximum from 23:45.
    da = pd.read_parquet(R8_TEST_CHRONOS.parent / "chronos_day_ahead.parquet")
    rows = []
    for o, chmax in zip(da.origin, da.chronos_next_day_max):
        ends = pd.date_range(o + Q, periods=96, freq="15min")
        y = label.reindex(ends)
        if y.isna().any():
            continue
        today = h.power.loc[o - pd.Timedelta(hours=24) + Q:o]
        lastweek = h.power.reindex(ends - pd.Timedelta(days=7))
        rows.append({"day": ends[0].normalize(), "y": y.max(), "chronos": chmax,
                     "persistence_today_max": today.max(), "b1_last_week_max": lastweek.max()})
    aux = pd.DataFrame(rows)
    result["next_day_max"] = {"days": int(len(aux)), **{c: float((aux[c] - aux.y).abs().mean())
                              for c in ("chronos", "persistence_today_max", "b1_last_week_max")}} if len(aux) else {"days": 0}
    result["wall_seconds"] = round(time.time() - t0, 1)
    _write_once(result_path, result)
    pred_dir = ROOT / "outputs/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    export = d[["origin", "target_time", "horizon", "y", "pred", "q05", "q10", "q90", "q95", "p_exceed", "gated"]].copy()
    export["alert"] = [p > lock["alert_thresholds"][f"p_exceed_h{hz}"] for p, hz in zip(export.p_exceed, export.horizon)]
    export.rename(columns={"y": "actual"}).to_csv(pred_dir / "final_test_fg_r10.csv", index=False, encoding="utf-8")
    export[export.horizon.eq(4)].rename(columns={"y": "actual"}).to_csv(pred_dir / "final_test_fg_r10_h4.csv", index=False, encoding="utf-8")
    if len(aux):
        aux.to_csv(pred_dir / "final_test_next_day_max.csv", index=False, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("rows", "test_days", "gated_share")}, indent=2))
    print(pd.DataFrame(result["point"]).T[["AUC_MAE", "AUC_PeakMAE", "h4_MAE", "h16_MAE"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lock", "evaluate"))
    {"lock": stage_lock, "evaluate": stage_evaluate}[ap.parse_args().stage]()
