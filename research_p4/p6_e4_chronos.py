"""E4 (reference only): Chronos-2 variants with quantile outputs; never used for selection."""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
import torch

from src.holidays import calendar_flags
from src.models.peak_prob import exceedance_from_quantiles
from src.session_data import read_development_oof
from src.training import _cutoff, _row
from .common import CACHE, ROOT, Timer, config, contexts, history, safe_power
from .p6_common import compare, out_dir, write_json

MODEL, REVISION, BATCH = "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", 32
VARIANTS = {"R1_ctx2048": {"context": 2048, "covariates": False},
            "R2_ctx4096": {"context": 4096, "covariates": False},
            "R3_ctx2048_calendar": {"context": 2048, "covariates": True}}
SELECTED = {1: ("p4_mstl_daily", "p4_quantile_dense"), 4: ("p4_blend3", "lgbm_quantile_b"),
            16: ("lgbm_residual_cbl", "lgbm_quantile_b"), 96: ("c3_holiday_hybrid", "lgbm_quantile_b")}


def _calendar(times):
    flags = calendar_flags(times)
    return {"hour": times.hour.to_numpy(np.float32), "quarter": (times.minute // 15).to_numpy(np.float32),
            "weekday": times.dayofweek.to_numpy(np.float32), "offday": flags["is_offday"].to_numpy(np.float32)}


def forecast(name, spec, grid, series, origins, pipe, levels):
    path = CACHE / f"p6_{name}.pkl"
    store = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [o for o in origins if o not in store]
    pos = pd.Series(np.arange(len(grid)), index=grid)
    t = Timer()
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        inputs = []
        for o in batch:
            end = int(pos[o]) + 1
            lo = max(0, end - spec["context"])
            target = series[lo:end]
            if spec["covariates"]:
                past = _calendar(grid[lo:end])
                future = _calendar(pd.date_range(o + pd.Timedelta(minutes=15), periods=96, freq="15min"))
                inputs.append({"target": target, "past_covariates": past, "future_covariates": future})
            else:
                inputs.append(torch.tensor(target))
        q, _ = pipe.predict_quantiles(inputs, prediction_length=96, quantile_levels=levels)
        for o, item in zip(batch, q):
            store[o] = np.asarray(item, dtype=np.float32).reshape(96, len(levels))
        if (start // BATCH) % 50 == 0:
            print(f"[{name}] {start + len(batch)}/{len(todo)} {t()}s", flush=True)
            path.write_bytes(pickle.dumps(store))
    path.write_bytes(pickle.dumps(store))
    return store


def rows_from(store, ctx, levels, name):
    rows = []
    for (h, f), c in ctx.items():
        t, tau = c["targets"], c["tau"]
        def q_at(idx):
            arr = np.stack([store[o][h - 1] for o in idx])
            return {lv: arr[:, i].astype(float) for i, lv in enumerate(levels)}
        qc, qs = q_at(c["cal"]), q_at(c["score"])
        yc = t.loc[c["cal"], "y"].to_numpy(float)
        cut_point = _cutoff(yc, qc[.5], tau, t.loc[c["cal"], "target_time"])
        rows.append(_row(c["score"], h, f, f"{name}_median", t, tau, qs[.5], cut_point))
        pc, ps = exceedance_from_quantiles(qc, tau), exceedance_from_quantiles(qs, tau)
        start = len(c["cal"]) // 2 + h + 1
        cut_prob = _cutoff(yc[start:], pc[start:], tau, t.loc[c["cal"][start:], "target_time"])
        r = _row(c["score"], h, f, f"{name}_risk", t, tau, qs[.5], float("inf"), p_exceed=ps, q95_cal=qs[.95],
                 q90_cal=qs[.9], q975_cal=np.maximum(qs[.95], qs[.99]), q10=qs[.1], q50=qs[.5], reference_only=True)
        r["alert"] = ps > cut_prob
        r["alert_cutoff"] = cut_prob
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def run(only=None):
    torch.set_num_threads(2)
    from chronos import BaseChronosPipeline
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    power = safe_power(df)
    grid = pd.date_range(power.index.min(), power.index.max(), freq="15min")
    series = power.reindex(grid).to_numpy(np.float32)
    origins = sorted(set().union(*[set(c["cal"]) | set(c["score"]) for c in ctx.values()]))
    pipe = BaseChronosPipeline.from_pretrained(MODEL, revision=REVISION, device_map="cpu")
    levels = list(pipe.quantiles)
    out = out_dir("e4_chronos")
    oof = read_development_oof(ROOT / "outputs/predictions/development_oof.csv")
    table = []
    for name, spec in VARIANTS.items():
        if only and name != only:
            continue
        t = Timer()
        store = forecast(name, spec, grid, series, origins, pipe, levels)
        rows = rows_from(store, ctx, levels, name)
        rows.to_parquet(out / f"{name}_rows.parquet", index=False)
        both = pd.concat([oof, rows], ignore_index=True)
        both["target_time"] = pd.to_datetime(both.target_time)
        for h, (point, risk) in SELECTED.items():
            r = compare(both, point, f"{name}_median", h)
            r["fold_ratios"] = str(r["fold_ratios"])
            table.append({"variant": name, "role": "point_vs_selected", **r})
            r = compare(both, risk, f"{name}_risk", h)
            r["fold_ratios"] = str(r["fold_ratios"])
            table.append({"variant": name, "role": "risk_alert_vs_selected_risk", **r})
        pd.DataFrame(table).to_csv(out / "chronos_variants_table.csv", index=False)
        print(name, "done", t(), flush=True)
    write_json(out / "meta.json", {"model": MODEL, "revision": REVISION, "levels": levels, "variants": VARIANTS,
                                   "origins": len(origins), "reference_only": True})


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
