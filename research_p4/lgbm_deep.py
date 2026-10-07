"""E2: tuned LightGBM (Optuna on the stop window) with seed bagging, and a delta-target variant."""
from __future__ import annotations

import json
import pickle

import lightgbm as lgb
import numpy as np
import optuna

from .common import CACHE, HORIZONS, Timer, config, contexts, feature_columns, history, make_rows, save

N_TRIALS, SEEDS, PEAK_WEIGHT = 40, (42, 43, 44, 45, 46), 2.0


def _model(params, seed, cfg):
    s = cfg.get("lgbm", {})
    return lgb.LGBMRegressor(objective="regression", n_estimators=int(s.get("n_estimators", 600)),
                             random_state=seed, n_jobs=2, verbosity=-1, subsample_freq=1, **params)


def _fit(params, seed, cfg, xf, yf, wf, xs, ys):
    m = _model(params, seed, cfg)
    m.fit(xf, yf, sample_weight=wf, eval_set=[(xs, ys)], eval_metric="l1",
          callbacks=[lgb.early_stopping(int(cfg.get("lgbm", {}).get("early_stopping", 60)), verbose=False)])
    return m


def run() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows_t, rows_d, meta, store = [], [], [], {}
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            cols = feature_columns(c["x"])
            x, y, tau = c["x"][cols], c["targets"].y, c["tau"]
            xf, yf, xs, ys = x.loc[c["fit"]], y.loc[c["fit"]], x.loc[c["stop"]], y.loc[c["stop"]]
            wf = np.where(yf.to_numpy() > tau, PEAK_WEIGHT, 1.0)

            def objective(trial):
                params = {"num_leaves": trial.suggest_int("num_leaves", 15, 127),
                          "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                          "learning_rate": trial.suggest_float("learning_rate", .02, .10, log=True),
                          "colsample_bytree": trial.suggest_float("colsample_bytree", .6, 1.0),
                          "subsample": trial.suggest_float("subsample", .6, 1.0),
                          "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0)}
                m = _fit(params, 42, cfg, xf, yf, wf, xs, ys)
                return float(np.mean(np.abs(ys.to_numpy() - m.predict(xs))))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=N_TRIALS)
            best = dict(study.best_params)
            # Level target, 5-seed average.
            level = [_fit(best, s, cfg, xf, yf, wf, xs, ys) for s in SEEDS]
            pred = lambda models, part: np.mean([m.predict(x.loc[c[part]]) for m in models], axis=0)
            pc, ps = pred(level, "cal"), pred(level, "score")
            rows_t.append(make_rows(c, h, f, "p4_lgbm_tuned", pc, ps))
            # Delta target y - current with the same settings; no extra search.
            cur = x["current"]
            delta = [_fit(best, s, cfg, xf, yf - cur.loc[c["fit"]], wf, xs, ys - cur.loc[c["stop"]]) for s in SEEDS]
            dc = pred(delta, "cal") + cur.loc[c["cal"]].to_numpy()
            ds = pred(delta, "score") + cur.loc[c["score"]].to_numpy()
            rows_d.append(make_rows(c, h, f, "p4_lgbm_delta", dc, ds))
            store[(h, f)] = {"tuned": {"cal": pc, "score": ps}, "delta": {"cal": dc, "score": ds}}
            meta.append({"horizon": h, "fold": f, "best_params": best, "best_stop_mae": study.best_value,
                         "trials": len(study.trials), "final_fits": 2 * len(SEEDS),
                         "best_iterations": [m.best_iteration_ for m in level], "seconds": t()})
            print(json.dumps(meta[-1], default=float), flush=True)
    (CACHE / "lgbm_deep.pkl").write_bytes(pickle.dumps(store))
    save(rows_t, "p4_lgbm_tuned", {"folds": meta})
    save(rows_d, "p4_lgbm_delta", {"folds": meta})
    return {"folds": meta}


if __name__ == "__main__":
    run()
