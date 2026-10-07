"""P5 (outputs/logs/preregistration_0929_P5.md): Chronos-2 promotion candidates and scoring."""
from __future__ import annotations

import itertools
import json
import pickle

import numpy as np
import pandas as pd

from src.research_stats import holm_adjust
from . import evaluate_p4 as ev
from .blends import _cbl, _mae
from .common import CACHE, HORIZONS, OUT, Timer, config, contexts, history, make_rows, save

INCUMBENT_P5 = {1: "p4_mstl_daily", 4: "p4_blend3", 16: "lgbm_residual_cbl", 96: "c3_holiday_hybrid"}
CANDIDATES = ["p5_chronos2", "p5_blend4"]


def build() -> dict:
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    lgbm = pickle.loads((CACHE / "lgbm_component.pkl").read_bytes())
    dshw = pickle.loads((CACHE / "dshw.pkl").read_bytes())
    paths = pickle.loads((CACHE / "chronos2_paths.pkl").read_bytes())
    grid = [w for w in itertools.product(np.round(np.arange(0, 1.0001, .1), 1), repeat=4) if abs(sum(w) - 1) < 1e-9]
    rows_a, rows_b, meta = [], [], []
    for h in HORIZONS:
        for f in range(3):
            t = Timer()
            c = ctx[(h, f)]
            chron = {part: np.array([paths[o][h - 1] for o in c[part]], dtype=float) for part in ("cal", "score")}
            comp = [lgbm[(h, f)], _cbl(df, ctx, h, f), dshw[(h, f)], chron]
            y_cal = c["targets"].loc[c["cal"], "y"].to_numpy(dtype=float)
            mix = lambda part, w: sum(wi * ci[part] for wi, ci in zip(w, comp))
            w = min(grid, key=lambda w: (_mae(y_cal, mix("cal", w)), w))
            rows_a.append(make_rows(c, h, f, "p5_chronos2", chron["cal"], chron["score"]))
            rows_b.append(make_rows(c, h, f, "p5_blend4", mix("cal", w), mix("score", w),
                                    blend_weights=f"lgbm={w[0]},cbl={w[1]},dshw={w[2]},chronos={w[3]}"))
            meta.append({"horizon": h, "fold": f, "weights_lgbm_cbl_dshw_chronos": [float(v) for v in w], "seconds": t()})
            print(meta[-1], flush=True)
    save(rows_a, "p5_chronos2", {"source": "p4 chronos2 cache", "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c"})
    save(rows_b, "p5_blend4", {"folds": meta})
    return {"folds": meta}


def evaluate() -> dict:
    base = pd.read_csv(ev.ROOT / "outputs/predictions/p2_combined_oof.csv", low_memory=False,
                       parse_dates=["origin", "target_time"])
    frames = [base] + [pd.read_parquet(OUT / f"{n}.parquet") for n in
                       ["p4_mstl_daily", "p4_blend3", "p5_chronos2", "p5_blend4"]]
    data = pd.concat(frames, ignore_index=True)
    data["origin"], data["target_time"] = pd.to_datetime(data.origin), pd.to_datetime(data.target_time)
    rows = []
    for h in HORIZONS:
        inc = INCUMBENT_P5[h]
        for cand in CANDIDATES:
            m = ev._pair(data, h, inc, cand)
            (effect, ci, p), peaks = ev._peak_gain(m)
            si, sc = ev._scores(m, "a"), ev._scores(m, "b")
            ratios = {}
            for f in range(3):
                pf = peaks.loc[peaks.fold.eq(f)]
                ia, cb = (pf.y - pf.pred_a).abs().mean(), (pf.y - pf.pred_b).abs().mean()
                ratios[f"fold{f}_peak_ratio"] = float(cb / ia) if ia > 0 else np.nan
            row = {"horizon": h, "candidate": cand, "incumbent": inc, "n_paired": len(m), "n_peak": len(peaks),
                   "incumbent_peak_mae": si.get("peak_mae"), "candidate_peak_mae": sc.get("peak_mae"),
                   "peak_mae_gain": effect, "ci_low": ci[0], "ci_high": ci[1], "p_raw": p,
                   "incumbent_episode_f1": si.get("episode_f1"), "candidate_episode_f1": sc.get("episode_f1"),
                   "incumbent_fp": si.get("false_alarms_positions"), "candidate_fp": sc.get("false_alarms_positions"),
                   "incumbent_mae": si.get("mae"), "candidate_mae": sc.get("mae"), **ratios}
            for label, ref in (("cbl", ev.CBL_REP[h]), ("persistence", ev.PERSISTENCE[h]), ("original", ev.INCUMBENT[h])):
                (e2, c2, _), _ = ev._peak_gain(ev._pair(data, h, ref, cand))
                row.update({f"vs_{label}_model": ref, f"vs_{label}_gain": e2, f"vs_{label}_ci_low": c2[0], f"vs_{label}_ci_high": c2[1]})
            rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) != 8:
        raise AssertionError("P5 Holm family must be 8")
    table["p_holm"] = holm_adjust(table.p_raw.to_numpy(), family_size=8)
    decisions = []
    for _, r in table.iterrows():
        fails = []
        if not (r.peak_mae_gain > 0 and r.p_holm < .05): fails.append("peak_gain_not_significant_after_holm")
        if r.candidate_episode_f1 < r.incumbent_episode_f1 - .02: fails.append("episode_f1_drop")
        if r.candidate_fp > 1.2 * r.incumbent_fp: fails.append("fp_increase")
        if not (r.fold2_peak_ratio <= 1.10): fails.append("last_fold_ratio")
        if not all(r[f"fold{f}_peak_ratio"] <= 1.50 for f in range(3)): fails.append("any_fold_ratio")
        decisions.append("pass" if not fails else ";".join(fails))
    table["decision"] = decisions
    chosen = {}
    for h in HORIZONS:
        ok = table.loc[table.horizon.eq(h) & table.decision.eq("pass")]
        ok = ok.assign(order=ok.candidate.map({"p5_chronos2": 0, "p5_blend4": 1})).sort_values(["peak_mae_gain", "order"], ascending=[False, True])
        chosen[h] = ok.candidate.iloc[0] if len(ok) else INCUMBENT_P5[h]
    table.to_csv(OUT / "p5_point_comparisons.csv", index=False)
    summary = {"point_selection_after_p5": chosen, "incumbent_p4": INCUMBENT_P5, "holm_family": 8,
               "preregistration": "outputs/logs/preregistration_0929_P5.md", "post_hoc_promotion_disclosed": True}
    (OUT / "p5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table[["horizon", "candidate", "incumbent_peak_mae", "candidate_peak_mae", "peak_mae_gain", "ci_low", "ci_high", "p_holm",
                 "incumbent_episode_f1", "candidate_episode_f1", "incumbent_fp", "candidate_fp", "incumbent_mae", "candidate_mae",
                 "fold0_peak_ratio", "fold1_peak_ratio", "fold2_peak_ratio", "vs_cbl_gain", "vs_cbl_ci_low", "decision"]].round(3).to_string())
    print(summary)
    return summary


if __name__ == "__main__":
    import sys
    {"build": build, "evaluate": evaluate}[sys.argv[1]]()
