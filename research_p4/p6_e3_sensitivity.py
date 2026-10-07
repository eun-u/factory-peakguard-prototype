"""E3: data-arrival delay, peak-definition (tau) and LightGBM seed sensitivity (development only)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.lgbm_point import fit_point
from src.models.seasonal import safe_grid
from src.session_data import read_development_oof
from src.targets import training_peak_threshold
from .common import ROOT, Timer, config, contexts, history
from .p6_common import HOLIDAY, build_ctx, compare, day_boot, mstl_paths, out_dir, point_models, write_json

SELECTED = {1: "p4_mstl_daily", 4: "p4_blend3", 16: "lgbm_residual_cbl", 96: "c3_holiday_hybrid"}
ORIGINAL = {1: "p1_latest", 4: "lgbm_no_holiday_weight_2", 16: "c3_holiday_hybrid", 96: "c3_holiday_hybrid"}
CBL = {1: "c2a_max_4_5_adjusted", 4: "c2a_max_4_5_adjusted", 16: "c3_holiday_hybrid", 96: "c3_holiday_hybrid"}


def delay(df, cfg, out):
    series, paths = safe_grid(df), mstl_paths()
    rows, meta = [], []
    for target_h, delays in ((1, (1, 2)), (4, (1, 2))):
        for d in delays:
            h = target_h + d
            t = Timer()
            ctx = build_ctx(df, cfg, h)
            for f in range(3):
                rows += point_models(df, cfg, ctx[f], h, f, series, paths)
            meta.append({"target_horizon": target_h, "delay_intervals": d, "model_horizon": h, "seconds": t()})
            print(meta[-1], flush=True)
    frame = pd.concat(rows, ignore_index=True)
    frame.to_parquet(out / "delay_rows.parquet", index=False)
    table = []
    for h in sorted(frame.horizon.unique()):
        part = frame.loc[frame.horizon.eq(h)]
        for model, g in part.groupby("model"):
            peaks = g.loc[g.y > g.tau]
            err = (peaks.y - peaks.pred).abs().to_numpy()
            est, ci = day_boot(peaks, err)
            table.append({"model_horizon": int(h), "model": model, "peak_mae": est, "ci_low": ci[0], "ci_high": ci[1],
                          "mae": float((g.y - g.pred).abs().mean()), "n": len(g), "n_peak": len(peaks)})
        for a, b in (("p1_latest", "p4_mstl_daily"), ("p1_latest", "p4_blend3"), ("lgbm_no_holiday_weight_2", "p4_blend3"),
                     ("cbl_rep", "p4_blend3"), ("cbl_rep", "p4_mstl_daily")):
            table.append({"model_horizon": int(h), "model": f"gain:{b}_vs_{a}", **{k: v for k, v in compare(frame, a, b, h).items()
                                                                                 if k in ("gain", "ci_low", "ci_high", "n_peak")}})
    pd.DataFrame(table).to_csv(out / "delay_table.csv", index=False)
    write_json(out / "delay_meta.json", meta)


def tau_sensitivity(df, out):
    oof = read_development_oof(ROOT / "outputs/predictions/development_oof.csv")
    oof["target_time"] = pd.to_datetime(oof.target_time)
    manifest = json.loads((ROOT / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    fit_end = {(int(r["horizon"]), int(r["fold"])): pd.Timestamp(r["fit_end"]) for r in manifest["folds"]}
    table = []
    for q in (.90, .95, .975):
        taus = {k: training_peak_threshold(df, pd.DatetimeIndex([v]), q) for k, v in fit_end.items()}
        frame = oof.copy()
        frame["tau_q"] = [taus[(int(h), int(f))] for h, f in zip(frame.horizon, frame.fold)]
        for h in (1, 4, 16, 96):
            for label, a in (("vs_original", ORIGINAL[h]), ("vs_cbl", CBL[h])):
                if a == SELECTED[h]:
                    continue
                r = compare(frame, a, SELECTED[h], h, tau_col="tau_q")
                table.append({"tau_quantile": q, "horizon": h, "comparison": label, "a": a, "b": SELECTED[h],
                              **{k: r[k] for k in ("n_peak", "peak_mae_a", "peak_mae_b", "gain", "ci_low", "ci_high")}})
    pd.DataFrame(table).to_csv(out / "tau_table.csv", index=False)


def seeds(df, cfg, out):
    ctx = contexts(df, cfg)
    result = []
    for f in range(3):
        c = ctx[(4, f)]
        cols = [k for k in c["x"].columns if k not in HOLIDAY]
        y = c["targets"].y
        base = None
        for s in range(42, 52):
            m = fit_point(c["x"].loc[c["fit"], cols], y.loc[c["fit"]], c["x"].loc[c["stop"], cols], y.loc[c["stop"]],
                          {**cfg, "seed": s}, peak_threshold=c["tau"], peak_weight=2.0, params=c["expected"]["chosen_params"])
            p = m.predict(c["x"].loc[c["score"], cols])
            base = p if base is None else base
            result.append({"fold": f, "seed": s, "max_abs_diff_vs_seed42": float(np.max(np.abs(p - base)))})
    frame = pd.DataFrame(result)
    frame.to_csv(out / "seed_table.csv", index=False)
    return {"max_abs_diff_any_seed": float(frame.max_abs_diff_vs_seed42.max()),
            "note": "LightGBM 설정에 행·열 부분표집이 없어 seed가 적합에 영향을 주지 않는지 확인"}


def run():
    cfg, df = config(), history()
    out = out_dir("e3_sensitivity")
    t = Timer()
    seed_summary = seeds(df, cfg, out)
    print("seeds", seed_summary, t(), flush=True)
    tau_sensitivity(df, out)
    print("tau done", t(), flush=True)
    delay(df, cfg, out)
    write_json(out / "summary.json", {"seeds": seed_summary, "seconds": t()})
    print("E3 done", t())


if __name__ == "__main__":
    run()
