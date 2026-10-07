"""FG-R7: FG-R6 revision after a second EXPLORE-only search.

Changes from FG-R6 (phase_f/goal_full_analog.py, kept unchanged as evidence):
- NaN-tolerant match distances: a missing slot is skipped instead of
  invalidating the rest of the day (windows need >= 75% observed slots);
- "yest" criterion: distance over the whole previous calendar day;
- deviation features: present minus matched day (last 1/4/16 slots),
  deviation-corrected analog, matched day's own change and level ratio;
- gate keeps the most recent exact match even when exact ties disagree;
- fallback = mean of two shallow Huber LightGBMs (3 and 7 leaves).
Only power at timestamps <= origin is read. Holdout is never loaded.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from phase_f import goal_full_analog as v1
from phase_f.goal_full_analog import HORIZONS, Q, ROOT, metrics, load

NAMESPACE = "goal_full_analog_v2"
OUT = ROOT / "outputs/phase_f" / NAMESPACE
WINDOWS = (4, 16, 48, 96)
CONFIG = {
    "id": "FG-R7-full-analog-gated-shallow-huber",
    "members": [{"num_leaves": 3, "min_child_samples": 300},
                {"num_leaves": 7, "min_child_samples": 300}],
    "objective": "huber", "huber_alpha": 8.0,
    "params": {"learning_rate": .03, "colsample_bytree": .7, "subsample": .8,
               "subsample_freq": 1, "reg_lambda": 5., "n_estimators": 8000},
    "early_stopping_rounds": 200,
    "non_gated_weight": 3.0,
    "seeds": (42, 123, 2024, 3407, 777),
    "gate": "today_prefix_distance==0 & same calendar day; most recent exact lag",
}


class Features(v1.Features):
    def _analog_distances(self):
        x, n = self.x, self.n
        self.lags = np.arange(1, n // 96)
        dayid = np.cumsum(self.dayslot == 1)
        shape = (len(self.lags), n)
        self.dist = {W: np.full(shape, np.nan, np.float32) for W in WINDOWS}
        self.dtoday = np.full(shape, np.nan, np.float32)
        self.dyest = np.full(shape, np.nan, np.float32)
        last_slot = self.dayslot == 96
        for i, d in enumerate(self.lags):
            ref = np.full(n, np.nan)
            ref[96 * d:] = x[:n - 96 * d]
            ad = np.abs(x - ref)
            ok = np.isfinite(ad).astype(float)
            a0 = np.nan_to_num(ad)
            for W in WINDOWS:
                s = pd.Series(a0).rolling(W, min_periods=W).sum().to_numpy()
                c = pd.Series(ok).rolling(W, min_periods=W).sum().to_numpy()
                self.dist[W][i] = s / np.where(c >= .75 * W, c, np.nan)
            s = pd.Series(a0).groupby(dayid).cumsum().to_numpy()
            c = pd.Series(ok).groupby(dayid).cumsum().to_numpy()
            today = s / np.where(c > 0, c, np.nan)
            self.dtoday[i] = today
            # Completed previous day: value at slot 96, shifted past that slot.
            self.dyest[i] = pd.Series(np.where(last_slot, today, np.nan)).shift(1).ffill().to_numpy()

    def _xa(self, p):
        p = np.asarray(p)
        return np.where((p >= 0) & (p < self.n), self.x[np.clip(p, 0, self.n - 1)], np.nan)

    def analog(self, origins, h):
        i = self.pos.loc[pd.DatetimeIndex(origins)].to_numpy()
        cols = np.arange(len(i))
        ref = i[None, :] + h - 96 * self.lags[:, None]
        valid = ref >= 0
        vals = np.where(valid, self.x[np.clip(ref, 0, self.n - 1)], np.nan)
        crit = {"today": self.dtoday, **{f"W{W}": self.dist[W] for W in WINDOWS}, "yest": self.dyest}
        out = {}
        with np.errstate(all="ignore"), v1._quiet():
            for name, D in crit.items():
                a = D[:, i]
                a = np.where(valid & np.isfinite(vals) & np.isfinite(a), a, np.inf)
                j = np.argmin(a, 0)  # ties -> most recent lag
                bd = a[j, cols]
                ok = np.isfinite(bd)
                d = self.lags[j]
                p = np.where(ok, vals[j, cols], np.nan)
                out[f"a_{name}_pred"] = p
                out[f"a_{name}_dist"] = np.where(ok, bd, np.nan)
                out[f"a_{name}_lag"] = np.where(ok, d, np.nan)
                r = np.stack([self._xa(i - m) - self._xa(i - m - 96 * d) for m in range(16)])
                out[f"a_{name}_r0"] = r[0]
                out[f"a_{name}_r4"] = np.nanmean(r[:4], 0)
                out[f"a_{name}_r16"] = np.nanmean(r, 0)
                out[f"a_{name}_corr"] = p + out[f"a_{name}_r4"]
                out[f"a_{name}_chg"] = p - self._xa(i - 96 * d)
                out[f"a_{name}_ratio"] = (self._xa(i) + 5) / (self._xa(i - 96 * d) + 5)
                if name in ("today", "W16"):
                    near = a <= bd[None, :] + .5
                    v = np.where(near, vals, np.nan)
                    out[f"a_{name}_nnear"] = near.sum(0)
                    out[f"a_{name}_med"] = np.nanmedian(v, 0)
                    out[f"a_{name}_mean"] = np.nanmean(v, 0)
                    out[f"a_{name}_std"] = np.nanstd(v, 0)
                    out[f"a_{name}_max"] = np.nanmax(v, 0)
                    out[f"a_{name}_min"] = np.nanmin(v, 0)
                    order = np.argsort(a, 0)
                    d5 = np.take_along_axis(a, order[:5], 0)
                    v5 = np.take_along_axis(vals, order[:5], 0)
                    w = np.where(np.isfinite(d5) & np.isfinite(v5), 1 / (d5 + .25), 0)
                    out[f"a_{name}_w5"] = np.nansum(w * np.nan_to_num(v5), 0) / np.maximum(w.sum(0), 1e-9)
                    v10 = np.take_along_axis(vals, order[:10], 0)
                    out[f"a_{name}_m10"] = np.nanmean(v10, 0)
                    out[f"a_{name}_q90_10"] = np.nanpercentile(v10, 90, axis=0)
        out["a_dayslot"] = self.dayslot[i]
        out["a_cross"] = (self.dayslot[i] + h > 96).astype(int)
        out["a_lag_same"] = (out["a_today_lag"] == out["a_yest_lag"]).astype(float)
        return pd.DataFrame(out)


def gate(X: pd.DataFrame) -> np.ndarray:
    return ((X.a_today_dist == 0) & (X.a_cross == 0) & (X.a_dayslot >= 1)).to_numpy()


def run_fold(F, contexts, fold, seed, config=CONFIG):
    import lightgbm as lgb
    Xf, yf, _ = v1._stack(F, contexts, fold, "fit")
    Xs, ys, _ = v1._stack(F, contexts, fold, "stop")
    Xt, _, keys = v1._stack(F, contexts, fold, "score")
    wng = config["non_gated_weight"]
    preds, iters = [], []
    for member in config["members"]:
        model = lgb.LGBMRegressor(objective=config["objective"], alpha=config["huber_alpha"],
                                  **config["params"], **member, verbose=-1, n_jobs=8, seed=seed)
        model.fit(Xf, yf, sample_weight=np.where(gate(Xf), 1., wng),
                  eval_set=[(Xs, ys)], eval_sample_weight=[np.where(gate(Xs), 1., wng)],
                  callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)])
        preds.append(model.predict(Xt))
        iters.append(model.best_iteration_)
    g = gate(Xt)
    keys["lgbm"] = np.mean(preds, 0)
    keys["analog"] = Xt.a_today_pred.to_numpy()
    keys["gated"] = g
    keys["pred"] = np.where(g, keys.analog, keys.lgbm)
    keys["best_iterations"] = str(iters)
    return keys


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
    first = ("target_time", "y", "tau", "d2", "fit_mean", "analog", "gated")
    mean = (allp.groupby(["horizon", "fold", "origin"], sort=True)
            .agg({**{c: "first" for c in first}, "lgbm": "mean"}).reset_index())
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
    (OUT / f"{args.arm}_metrics.json").write_text(json.dumps(record, indent=2, default=list), encoding="utf-8")
    (OUT / "predictions").mkdir(exist_ok=True)
    mean.to_parquet(OUT / "predictions" / f"{args.arm}_seed_mean.parquet", index=False)
    print(json.dumps(result, indent=2), record["targets"])
    return record


if __name__ == "__main__":
    main()
