"""E1: does the P4 selection hold under alternative development splits? (h1, h4; selection unchanged)"""
from __future__ import annotations

import pandas as pd

from src.models.seasonal import safe_grid
from .common import Timer, config, history
from .p6_common import build_ctx, compare, mstl_paths, out_dir, point_models, write_json

SCHEMES = {"S1_5fold": {"n_folds": 5, "cal_frac": None}, "S2_3fold_cal50": {"n_folds": 3, "cal_frac": .5}}
PAIRS = {1: [("p1_latest", "p4_mstl_daily"), ("cbl_rep", "p4_mstl_daily"), ("p1_latest", "p4_dshw")],
         4: [("lgbm_no_holiday_weight_2", "p4_blend3"), ("cbl_rep", "p4_blend3"), ("p1_latest", "p4_blend3"),
             ("lgbm_no_holiday_weight_2", "p4_mstl_daily")]}


def run():
    cfg, df = config(), history()
    series, paths = safe_grid(df), mstl_paths()
    out = out_dir("e1_robustness")
    table, meta = [], []
    for scheme, spec in SCHEMES.items():
        rows = []
        for h in (1, 4):
            t = Timer()
            ctx = build_ctx(df, cfg, h, n_folds=spec["n_folds"], cal_frac=spec["cal_frac"])
            for f in sorted(ctx):
                rows += point_models(df, cfg, ctx[f], h, f, series, paths)
            meta.append({"scheme": scheme, "horizon": h, "folds": len(ctx), "seconds": t(),
                         "score_windows": [(str(ctx[f]["score"].min()), str(ctx[f]["score"].max())) for f in sorted(ctx)]})
            print(meta[-1], flush=True)
        frame = pd.concat(rows, ignore_index=True)
        frame.to_parquet(out / f"{scheme}_rows.parquet", index=False)
        for h, pairs in PAIRS.items():
            for a, b in pairs:
                r = compare(frame, a, b, h)
                r["fold_ratios"] = str(r["fold_ratios"])
                table.append({"scheme": scheme, **r})
        pd.DataFrame(table).to_csv(out / "robustness_table.csv", index=False)
    write_json(out / "meta.json", meta)
    print(pd.DataFrame(table)[["scheme", "horizon", "a", "b", "peak_mae_a", "peak_mae_b", "gain", "ci_low", "ci_high", "p4_gates_nominal"]].round(3).to_string())


if __name__ == "__main__":
    run()
