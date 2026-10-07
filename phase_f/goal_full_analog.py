"""FG-R6 full-history exact-day analog + gated LightGBM fallback.

Motivation (post-hoc, disclosed): in the sealed development history more than
half of the calendar days are exact 96-slot copies of an earlier day, often
59-151 days earlier. Chronos-2 (2048-slot context, ~21 days) and FG-R4/R5
(FIT bank / 56-day windows, k-median blending) cannot reach most sources.

At origin o (interval-end timestamp) only power at timestamps <= o is used:
- for every day lag d=1..D the mean |x - x(-96d)| over today's slots so far
  (and over fixed windows W) is a match distance;
- the analog value for horizon h is x[o+h-96d] (<= o since h <= 16 < 96);
- gate: today's prefix matches exactly (distance 0), all exact ties agree and
  the target is still on the origin's calendar day -> use the analog value;
- otherwise a per-week global multi-horizon LightGBM (FIT train, STOP early
  stopping) using power-only causal lags/rolling/weekly-aligned and analog
  features.

The gate rule is fixed a priori (exact identity). Loss and weighting were
chosen on EXPLORE only. CONFIRM is run once after the configuration lock.
Holdout (>= 2021-08-09 09:45) is never loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "goal_full_analog_v1"
OUT = ROOT / "outputs/phase_f" / NAMESPACE
HORIZONS = tuple(range(4, 17))
Q = pd.Timedelta(minutes=15)
WINDOWS = (4, 16, 48, 96)
DROP_SLOW = ("t_month", "rmean672", "rmax672", "rstd672", "tw4")
CONFIG = {
    "id": "FG-R6-full-analog-gated-lgbm",
    "objective": "regression_l1",
    "params": {"learning_rate": .03, "num_leaves": 63, "min_child_samples": 50,
               "colsample_bytree": .7, "subsample": .8, "subsample_freq": 1,
               "reg_lambda": 5., "n_estimators": 5000},
    "early_stopping_rounds": 200,
    "non_gated_weight": 3.0,
    "seeds": (42, 123, 2024, 3407, 777),
    "gate": "today_prefix_distance==0 & same calendar day & exact ties agree",
}


# ---------------------------------------------------------------- data
def load(root: Path = ROOT):
    from phase_c.data import load_history
    from phase_f.wf_harness import build_weekly_contexts, _score_keys
    from phase_f.registry import config_hash
    history, _ = load_history(root)
    contexts = build_weekly_contexts(history)
    keys = _score_keys(contexts)
    lock = json.loads((root / "outputs/phase_f/walkforward_v2/walkforward_lock.json").read_text(encoding="utf-8"))
    digest = config_hash(keys.astype(str).to_dict("records"))
    if digest != lock["score_key_sha256"]:
        raise RuntimeError("Weekly score cohort differs from walkforward_v2 lock")
    return history, contexts, keys


# ---------------------------------------------------------------- features
class Features:
    """All features at origin o read power at timestamps <= o only."""

    def __init__(self, history: pd.DataFrame):
        pw = history.power.astype(float)
        self.index = pw.index
        self.x = pw.to_numpy()
        self.n = len(self.x)
        self.pos = pd.Series(np.arange(self.n), index=pw.index)
        full = pd.date_range(pw.index.min() - pd.Timedelta(days=29), pw.index.max(), freq="15min")
        self.P = pw.reindex(full)
        local = pw.index - Q
        self.dayslot = (local.hour * 4 + local.minute // 15 + 1).to_numpy()
        self._origin_features()
        self._analog_distances()

    def at(self, idx):
        return self.P.reindex(idx).to_numpy()

    def _origin_features(self):
        P, idx = self.P, self.index
        og = pd.DataFrame(index=idx)
        for k in range(17):  # available at origin: x[o-k]
            og[f"lag{k}"] = P.shift(k).reindex(idx).to_numpy()
        for w in (4, 8, 16, 96, 672):  # trailing windows ending at origin
            r = P.rolling(w, min_periods=max(1, w // 2))
            og[f"rmean{w}"] = r.mean().reindex(idx).to_numpy()
            og[f"rmax{w}"] = r.max().reindex(idx).to_numpy()
            og[f"rstd{w}"] = r.std().reindex(idx).to_numpy()
        og["d1"] = og.lag0 - og.lag1
        og["d4"] = og.lag0 - og.lag4
        og["d16"] = og.lag0 - og.lag16
        for k in (1, 2, 3, 4):
            og[f"o_w{k}"] = self.at(idx - pd.Timedelta(days=7 * k))
        og["o_dev1"] = og.lag0 - og.o_w1
        og["o_devm"] = og.lag0 - og[["o_w1", "o_w2", "o_w3"]].mean(1)
        with np.errstate(all="ignore"):
            past = np.nanmean([self.at(idx - Q * k - pd.Timedelta(days=7)) for k in range(4)], 0)
        og["o_dev1h"] = og[[f"lag{k}" for k in range(4)]].mean(1) - past
        on = (P > 60).astype(float).where(P.notna())
        og["on_frac16"] = on.rolling(16, min_periods=1).mean().reindex(idx).to_numpy()
        state = on.reindex(idx).ffill()
        og["run_len"] = state.groupby(state.ne(state.shift()).cumsum()).cumcount().to_numpy()
        og["state"] = state.to_numpy()
        og["o_hour"] = idx.hour + idx.minute / 60
        og["o_dow"] = idx.dayofweek
        self.og = og

    def _analog_distances(self):
        x, n = self.x, self.n
        self.lags = np.arange(1, n // 96)
        dayid = np.cumsum(self.dayslot == 1)
        self.dist = {W: np.full((len(self.lags), n), np.nan, np.float32) for W in WINDOWS}
        self.dtoday = np.full((len(self.lags), n), np.nan, np.float32)
        for i, d in enumerate(self.lags):
            ref = np.full(n, np.nan)
            ref[96 * d:] = x[:n - 96 * d]
            ad = np.abs(x - ref)  # both terms observed at or before each index
            for W in WINDOWS:
                self.dist[W][i] = pd.Series(ad).rolling(W, min_periods=W).mean().to_numpy()
            s = pd.Series(ad).groupby(dayid).cumsum().to_numpy()
            c = pd.Series(np.isfinite(ad).astype(float)).groupby(dayid).cumsum().to_numpy()
            bad = pd.Series(np.isnan(ad).astype(float)).groupby(dayid).cumsum().to_numpy()
            v = s / np.maximum(c, 1)
            v[bad > 0] = np.nan
            self.dtoday[i] = v

    def analog(self, origins, h):
        i = self.pos.loc[pd.DatetimeIndex(origins)].to_numpy()
        cols = np.arange(len(i))
        ref = i[None, :] + h - 96 * self.lags[:, None]
        valid = ref >= 0
        vals = np.where(valid, self.x[np.clip(ref, 0, self.n - 1)], np.nan)
        crit = {"today": self.dtoday, **{f"W{W}": self.dist[W] for W in WINDOWS}}
        out = {}
        for name, D in crit.items():
            a = D[:, i]
            a = np.where(valid & np.isfinite(vals) & np.isfinite(a), a, np.inf)
            j = np.argmin(a, 0)  # ties -> most recent lag
            bd = a[j, cols]
            ok = np.isfinite(bd)
            out[f"a_{name}_pred"] = np.where(ok, vals[j, cols], np.nan)
            out[f"a_{name}_dist"] = np.where(ok, bd, np.nan)
            out[f"a_{name}_lag"] = np.where(ok, self.lags[j], np.nan)
            if name in ("today", "W16"):
                near = a <= bd[None, :] + .5
                v = np.where(near, vals, np.nan)
                with np.errstate(all="ignore"), _quiet():
                    out[f"a_{name}_nnear"] = near.sum(0)
                    out[f"a_{name}_med"] = np.nanmedian(v, 0)
                    out[f"a_{name}_mean"] = np.nanmean(v, 0)
                    out[f"a_{name}_std"] = np.nanstd(v, 0)
                    out[f"a_{name}_max"] = np.nanmax(v, 0)
                    out[f"a_{name}_min"] = np.nanmin(v, 0)
                k5 = np.argsort(a, 0)[:5]
                d5 = np.take_along_axis(a, k5, 0)
                v5 = np.take_along_axis(vals, k5, 0)
                w = np.where(np.isfinite(d5) & np.isfinite(v5), 1 / (d5 + .25), 0)
                out[f"a_{name}_w5"] = np.nansum(w * np.nan_to_num(v5), 0) / np.maximum(w.sum(0), 1e-9)
        out["a_dayslot"] = self.dayslot[i]
        out["a_cross"] = (self.dayslot[i] + h > 96).astype(int)
        return pd.DataFrame(out)

    def build(self, origins, h):
        o = pd.DatetimeIndex(origins)
        t = o + Q * h  # known calendar of the target; values read at t-k days <= o
        X = self.og.loc[o].reset_index(drop=True).copy()
        X["h"] = h
        X["t_hour"] = t.hour + t.minute / 60
        X["t_dow"] = t.dayofweek
        X["t_slotw"] = t.dayofweek * 96 + t.hour * 4 + t.minute // 15
        X["t_month"] = t.month
        for k in (1, 2, 3, 4):
            X[f"tw{k}"] = self.at(t - pd.Timedelta(days=7 * k))
            if k <= 2:
                for j in (-2, -1, 1, 2):
                    X[f"tw{k}_{j}"] = self.at(t - pd.Timedelta(days=7 * k) + Q * j)
        X["t_d1"] = self.at(t - pd.Timedelta(days=1))
        X["t_d2"] = self.at(t - pd.Timedelta(days=2))
        X["tw_mean3"] = X[["tw1", "tw2", "tw3"]].mean(1)
        X["tw_med3"] = X[["tw1", "tw2", "tw3"]].median(1)
        X["tw_max3"] = X[["tw1", "tw2", "tw3"]].max(1)
        X["tw_min3"] = X[["tw1", "tw2", "tw3"]].min(1)
        win = ["tw1", "tw1_-2", "tw1_-1", "tw1_1", "tw1_2"]
        X["tw1_win_max"] = X[win].max(1)
        X["tw1_win_mean"] = X[win].mean(1)
        X["w1_change"] = X.tw1 - X.o_w1
        X["naive_wdev"] = X.tw1 + X.o_dev1
        X["naive_mdev"] = X.tw_mean3 + X.o_devm
        X["w1_change_m"] = X.tw_mean3 - X[["o_w1", "o_w2", "o_w3"]].mean(1)
        X = X.drop(columns=list(DROP_SLOW))
        return pd.concat([X, self.analog(o, h)], axis=1)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def gate(X: pd.DataFrame) -> np.ndarray:
    return ((X.a_today_dist == 0) & (X.a_cross == 0) & (X.a_dayslot >= 1)
            & (X.a_today_min == X.a_today_max)).to_numpy()


# ---------------------------------------------------------------- model
def _stack(F, contexts, fold, role):
    Xs, ys, keys = [], [], []
    for h in HORIZONS:
        c = contexts[(h, fold)]
        o = c[role]
        Xs.append(F.build(o, h))
        ys.append(c["y"].loc[o].to_numpy(float))
        keys.append(pd.DataFrame({"horizon": h, "fold": fold, "origin": o,
                                  "target_time": pd.DatetimeIndex(c["target_time"].loc[o]),
                                  "y": ys[-1], "tau": c["tau"],
                                  "d2": c["d2"].loc[o].to_numpy(bool),
                                  "fit_mean": float(c["y"].loc[c["fit"]].mean())}))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), pd.concat(keys, ignore_index=True)


def run_fold(F, contexts, fold, seed, config=CONFIG):
    import lightgbm as lgb
    Xf, yf, _ = _stack(F, contexts, fold, "fit")
    Xs, ys, _ = _stack(F, contexts, fold, "stop")
    Xt, _, keys = _stack(F, contexts, fold, "score")
    wng = config["non_gated_weight"]
    model = lgb.LGBMRegressor(objective=config["objective"], **config["params"],
                              verbose=-1, n_jobs=8, seed=seed)
    model.fit(Xf, yf, sample_weight=np.where(gate(Xf), 1., wng),
              eval_set=[(Xs, ys)], eval_sample_weight=[np.where(gate(Xs), 1., wng)],
              callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)])
    g = gate(Xt)
    keys["lgbm"] = model.predict(Xt)
    keys["analog"] = Xt.a_today_pred.to_numpy()
    keys["gated"] = g
    keys["pred"] = np.where(g, keys.analog, keys.lgbm)
    keys["best_iteration"] = model.best_iteration_
    return keys


def metrics(frame: pd.DataFrame) -> dict:
    rows = []
    for h, part in frame.groupby("horizon"):
        e = (part.pred - part.y).abs()
        peak = part.y > part.tau
        rows.append({"h": h, "MAE": e.mean(), "Peak": e[peak].mean(),
                     "nMAE": e.mean() / part.fit_mean.mean(), "D2": e[part.d2].mean()})
    r = pd.DataFrame(rows)
    return {"AUC_MAE": r.MAE.mean(), "AUC_PeakMAE": r.Peak.mean(),
            "h4_MAE": r.MAE.iloc[0], "h16_MAE": r.MAE.iloc[-1],
            "AUC_nMAE": r.nMAE.mean(), "D2_AUC_MAE": r.D2.mean()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("EXPLORE", "CONFIRM"), default="EXPLORE")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(CONFIG["seeds"]))
    args = ap.parse_args(argv)
    if args.arm == "CONFIRM" and not (OUT / "CONFIG_LOCK.json").exists():
        raise SystemExit("CONFIRM requires CONFIG_LOCK.json written after EXPLORE selection")
    t0 = time.time()
    history, contexts, keys = load()
    folds = sorted(keys.loc[keys.arm.eq(args.arm), "fold"].unique())
    F = Features(history)
    runs = []
    for seed in args.seeds:
        frame = pd.concat([run_fold(F, contexts, f, seed) for f in folds], ignore_index=True)
        frame["seed"] = seed
        runs.append(frame)
        print(seed, {k: round(float(v), 4) for k, v in metrics(frame).items()}, flush=True)
    allp = pd.concat(runs, ignore_index=True)
    mean = (allp.groupby(["horizon", "fold", "origin"], sort=True)
            .agg({**{c: "first" for c in ("target_time", "y", "tau", "d2", "fit_mean",
                                          "analog", "gated")}, "lgbm": "mean"}).reset_index())
    mean["pred"] = np.where(mean.gated, mean.analog, mean.lgbm)
    result = {k: float(v) for k, v in metrics(mean).items()}
    from phase_f.goal_protocol import meets_targets
    record = {"config": CONFIG, "arm": args.arm, "seeds": args.seeds,
              "seed_mean_metrics": result, "targets": meets_targets(result),
              "per_seed": {int(s): {k: float(v) for k, v in metrics(r).items()}
                           for s, r in zip(args.seeds, runs)},
              "gated_share": float(mean.gated.mean()), "folds": [int(f) for f in folds],
              "holdout_read": False, "wall_seconds": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.arm}_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    mean.to_parquet(pred_dir / f"{args.arm}_seed_mean.parquet", index=False)
    print(json.dumps(record["seed_mean_metrics"], indent=2), record["targets"])
    return record


if __name__ == "__main__":
    main()
