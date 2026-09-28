"""Apply the preregistered P4 adoption rules (outputs/logs/preregistration_0928_P4.md)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluate import score_predictions
from src.research_stats import holm_adjust
from .common import CBL_REP, HORIZONS, INCUMBENT, OUT, ROOT

KEYS = ["origin", "target_time", "horizon", "fold"]
POINT = ["p4_blend2", "p4_blend3", "p4_dshw", "p4_mstl_daily", "p4_lgbm_tuned", "p4_lgbm_delta"]
REFERENCE = ["p4_chronos2_ref"]
SIMPLICITY = {name: i for i, name in enumerate(POINT)}
PERSISTENCE = {1: "p1_latest", 4: "p1_latest", 16: "p3_recent_mean", 96: "p1_latest"}
N_BOOT, SEED = 1000, 42


def _load():
    base = pd.read_csv(ROOT / "outputs/predictions/p2_combined_oof.csv", low_memory=False,
                       parse_dates=["origin", "target_time"])
    frames = [base]
    for name in POINT + REFERENCE + ["p4_quantile_dense"]:
        path = OUT / f"{name}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    data = pd.concat(frames, ignore_index=True)
    data["origin"] = pd.to_datetime(data.origin)
    data["target_time"] = pd.to_datetime(data.target_time)
    return data


def _day_boot(frame: pd.DataFrame, values: np.ndarray):
    """Centered one-sided day-block p and percentile CI for a mean effect."""
    if not len(frame):
        return np.nan, [np.nan, np.nan], 1.0
    days = pd.DatetimeIndex(frame.target_time).normalize()
    g = pd.DataFrame({"d": days, "v": values}).groupby("d").v.agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    effect = float(values.mean())
    p = (1 + int(np.count_nonzero(boot - effect >= effect))) / (N_BOOT + 1)
    return effect, [float(v) for v in np.quantile(boot, [.025, .975])], float(p)


def _pair(data, h, a, b):
    left = data.loc[data.horizon.eq(h) & data.model.eq(a)]
    right = data.loc[data.horizon.eq(h) & data.model.eq(b)]
    cols = ["y", "pred", "tau", "alert"]
    m = left[KEYS + cols].merge(right[KEYS + ["pred", "alert"]], on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    m = m.loc[np.isfinite(m.y) & np.isfinite(m.pred_a) & np.isfinite(m.pred_b)]
    return m


def _peak_gain(m):
    peaks = m.loc[m.y > m.tau]
    gain = (peaks.y - peaks.pred_a).abs() - (peaks.y - peaks.pred_b).abs()
    return _day_boot(peaks, gain.to_numpy()), peaks


def _scores(m, side):
    frame = m[KEYS + ["y", "tau"]].assign(pred=m[f"pred_{side}"], alert=m[f"alert_{side}"].astype(bool))
    return score_predictions(frame)


def point_table(data):
    rows = []
    for h in HORIZONS:
        inc = INCUMBENT[h]
        for cand in POINT + REFERENCE:
            if data.model.eq(cand).sum() == 0:
                rows.append({"horizon": h, "candidate": cand, "executed": False, "p_raw": 1.0})
                continue
            m = _pair(data, h, inc, cand)
            (effect, ci, p), peaks = _peak_gain(m)
            si, sc = _scores(m, "a"), _scores(m, "b")
            folds = {}
            for f in range(3):
                pf = peaks.loc[peaks.fold.eq(f)]
                ia, cb = (pf.y - pf.pred_a).abs().mean(), (pf.y - pf.pred_b).abs().mean()
                folds[f] = float(cb / ia) if ia > 0 else np.nan
            row = {"horizon": h, "candidate": cand, "incumbent": inc, "executed": True,
                   "reference_only": cand in REFERENCE, "n_paired": len(m), "n_peak": len(peaks),
                   "incumbent_peak_mae": si.get("peak_mae"), "candidate_peak_mae": sc.get("peak_mae"),
                   "peak_mae_gain": effect, "ci_low": ci[0], "ci_high": ci[1], "p_raw": p,
                   "incumbent_episode_f1": si.get("episode_f1"), "candidate_episode_f1": sc.get("episode_f1"),
                   "incumbent_fp": si.get("false_alarms_positions"), "candidate_fp": sc.get("false_alarms_positions"),
                   "incumbent_mae": si.get("mae"), "candidate_mae": sc.get("mae"),
                   **{f"fold{f}_peak_ratio": r for f, r in folds.items()}}
            for label, ref in (("cbl", CBL_REP[h]), ("persistence", PERSISTENCE[h])):
                mm = _pair(data, h, ref, cand)
                (e2, c2, _), _ = _peak_gain(mm)
                row.update({f"vs_{label}_model": ref, f"vs_{label}_gain": e2,
                            f"vs_{label}_ci_low": c2[0], f"vs_{label}_ci_high": c2[1]})
            rows.append(row)
    return pd.DataFrame(rows)


def risk_table(data):
    rows = []
    for h in HORIZONS:
        b = data.loc[data.horizon.eq(h) & data.model.eq("lgbm_quantile_b")]
        d = data.loc[data.horizon.eq(h) & data.model.eq("p4_quantile_dense")]
        if d.empty:
            rows.append({"horizon": h, "candidate": "p4_quantile_dense", "executed": False, "p_raw": 1.0})
            continue
        m = b[KEYS + ["y", "tau", "p_exceed"]].merge(d[KEYS + ["p_exceed"]], on=KEYS, suffixes=("_b", "_d"), validate="one_to_one")
        m = m.loc[np.isfinite(m.p_exceed_b) & np.isfinite(m.p_exceed_d)]
        event = (m.y > m.tau).astype(float)
        gain = (m.p_exceed_b - event) ** 2 - (m.p_exceed_d - event) ** 2
        effect, ci, p = _day_boot(m, gain.to_numpy())
        sb, sd = score_predictions(b), score_predictions(d)
        pin_d = np.mean([sd[f"pinball_{lv}"] for lv in (.1, .5, .9, .95, .975)])
        levels = [c for c in d.columns if c.startswith("dense_q")]
        crps = []
        for c in levels:
            lv = float("0." + c.split("dense_q")[1])
            q = d[c].to_numpy(float); y = d.y.to_numpy(float)
            crps.append(np.nanmean(np.maximum(lv * (y - q), (lv - 1) * (y - q))))
        rows.append({"horizon": h, "candidate": "p4_quantile_dense", "executed": True, "n_paired": len(m),
                     "brier_b": sb.get("brier"), "brier_dense": sd.get("brier"), "brier_gain": effect,
                     "ci_low": ci[0], "ci_high": ci[1], "p_raw": p,
                     "top_cov95_err_b": abs(sb["top_coverage_0.95"] - .95), "top_cov95_err_dense": abs(sd["top_coverage_0.95"] - .95),
                     "pinball_b": sb["pinball_mean"], "pinball_dense": pin_d,
                     "episode_f1_b": sb["episode_f1"], "episode_f1_dense": sd["episode_f1"],
                     "fp_b": sb["false_alarms_positions"], "fp_dense": sd["false_alarms_positions"],
                     "pr_auc_b": sb.get("pr_auc"), "pr_auc_dense": sd.get("pr_auc"),
                     "crps_dense_approx": float(2 * np.mean(crps))})
    return pd.DataFrame(rows)


def decide(point, risk):
    family = pd.concat([point.loc[~point.candidate.isin(REFERENCE), ["horizon", "candidate", "p_raw"]].assign(kind="point"),
                        risk[["horizon", "candidate", "p_raw"]].assign(kind="risk")], ignore_index=True)
    if len(family) != 28:
        raise AssertionError(f"Holm family must hold 28 hypotheses, got {len(family)}")
    family["p_holm"] = holm_adjust(family.p_raw.fillna(1.0).to_numpy(), family_size=28)
    point = point.merge(family.loc[family.kind.eq("point"), ["horizon", "candidate", "p_holm"]], how="left", on=["horizon", "candidate"])
    risk = risk.merge(family.loc[family.kind.eq("risk"), ["horizon", "candidate", "p_holm"]], how="left", on=["horizon", "candidate"])
    reasons = []
    for i, r in point.iterrows():
        if r.candidate in REFERENCE or not r.executed:
            reasons.append("reference_only" if r.candidate in REFERENCE else "not_executed"); continue
        fails = []
        if not (r.peak_mae_gain > 0 and r.p_holm < .05): fails.append("peak_gain_not_significant_after_holm")
        if r.candidate_episode_f1 < r.incumbent_episode_f1 - .02: fails.append("episode_f1_drop")
        if r.candidate_fp > 1.2 * r.incumbent_fp: fails.append("fp_increase")
        if not (r.fold2_peak_ratio <= 1.10): fails.append("last_fold_ratio")
        if not all(r[f"fold{f}_peak_ratio"] <= 1.50 for f in range(3)): fails.append("any_fold_ratio")
        reasons.append("pass" if not fails else ";".join(fails))
    point["decision"] = reasons
    rr = []
    for _, r in risk.iterrows():
        if not r.executed:
            rr.append("not_executed"); continue
        fails = []
        if not (r.brier_gain > 0 and r.p_holm < .05): fails.append("brier_not_significant_after_holm")
        if r.top_cov95_err_dense > r.top_cov95_err_b + .01: fails.append("top_coverage")
        if r.pinball_dense > 1.05 * r.pinball_b: fails.append("pinball")
        if r.episode_f1_dense < r.episode_f1_b - .02: fails.append("episode_f1_drop")
        rr.append("pass" if not fails else ";".join(fails))
    risk["decision"] = rr
    chosen = {}
    for h in HORIZONS:
        ok = point.loc[point.horizon.eq(h) & point.decision.eq("pass")]
        if len(ok):
            ok = ok.assign(order=ok.candidate.map(SIMPLICITY)).sort_values(["peak_mae_gain", "order"], ascending=[False, True])
            chosen[h] = ok.candidate.iloc[0]
        else:
            chosen[h] = INCUMBENT[h]
    risk_choice = {int(r.horizon): ("p4_quantile_dense" if r.decision == "pass" else "lgbm_quantile_b") for _, r in risk.iterrows()}
    return point, risk, chosen, risk_choice


def run():
    data = _load()
    point, risk = point_table(data), risk_table(data)
    point, risk, chosen, risk_choice = decide(point, risk)
    point.to_csv(OUT / "p4_point_comparisons.csv", index=False)
    risk.to_csv(OUT / "p4_risk_comparisons.csv", index=False)
    summary = {"point_selection": chosen, "risk_selection": risk_choice,
               "holm_family": 28, "preregistration": "outputs/logs/preregistration_0928_P4.md"}
    (OUT / "p4_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(point[["horizon", "candidate", "incumbent_peak_mae", "candidate_peak_mae", "peak_mae_gain", "ci_low", "ci_high",
                 "p_holm", "candidate_episode_f1", "incumbent_episode_f1", "candidate_fp", "incumbent_fp",
                 "fold0_peak_ratio", "fold1_peak_ratio", "fold2_peak_ratio", "decision"]].round(3).to_string())
    print(risk.round(4).to_string())
    print(summary)
    return summary


if __name__ == "__main__":
    run()
