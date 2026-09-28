"""Shared helpers for P6 (outputs/logs/preregistration_0929_P6.md). Development data only."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import score_predictions
from src.features import build_features
from src.models.baselines import baseline_predictions
from src.models.cbl import cbl_all_predictions
from src.models.lgbm_point import fit_point, point_grid
from src.models.seasonal import DSHW, mstl_forecast, mstl_phi, safe_grid, simplex_weights
from src.split import make_splits
from src.targets import point_targets, training_peak_threshold
from src.training import _cutoff, _partition, _row, _valid_mask
from .common import CACHE, ROOT, config, history

OUT6 = ROOT / "outputs" / "p6"
HOLIDAY = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")
KEYS = ["origin", "target_time", "horizon", "fold"]
CBL_FOR = lambda h: "c2a_max_4_5_adjusted" if h <= 8 else "c3_holiday_hybrid"


def out_dir(name: str) -> Path:
    path = OUT6 / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def mstl_paths() -> dict:
    return pickle.loads((CACHE / "mstl_daily.pkl").read_bytes())


def partition_cal(train, validation, horizon, cal_frac):
    """Variant of training._partition with a configurable calibration share of the validation block."""
    train, validation = pd.DatetimeIndex(train), pd.DatetimeIndex(validation)
    gap = pd.Timedelta(minutes=15 * horizon)
    stop_start = int(len(train) * .80)
    fit = train[train + gap < train[stop_start]]
    stop = train[stop_start:]
    cal = validation[:max(40, int(len(validation) * cal_frac))]
    score = validation[len(cal):]
    cal = cal[cal > stop.max() + gap]
    score = score[score > cal.max() + gap]
    if min(map(len, (fit, stop, cal, score))) < 30:
        raise ValueError("Partition too small")
    return fit, stop, cal, score


def build_ctx(df, cfg, horizon, n_folds=3, cal_frac=None) -> dict:
    """Fold contexts on the sealed development grid (mirrors src.research_cv without manifest checks)."""
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    origins = df.index[df.index < boundary]
    folds, test = make_splits(origins, horizon, dev_frac=1.0, n_folds=n_folds)
    assert not len(test)
    out = {}
    for fold_id, fold in enumerate(folds):
        all_origins = fold.train.append(fold.validation)
        targets = point_targets(df, all_origins, horizon)
        x0, used0 = build_features(df, all_origins, horizon, {**cfg, "_tau": None})
        x0 = x0.reindex(all_origins)
        good = all_origins[_valid_mask(x0, used0, all_origins, targets, df)]
        train, validation = fold.train.intersection(good), fold.validation.intersection(good)
        fit, stop, cal, score = (_partition(train, validation, horizon) if cal_frac is None
                                 else partition_cal(train, validation, horizon, cal_frac))
        tau = training_peak_threshold(df, fit, .95)
        for _ in range(3):
            x, used = build_features(df, all_origins, horizon, {**cfg, "_tau": tau})
            good = all_origins[_valid_mask(x, used, all_origins, targets, df)]
            fit, stop, cal, score = (p.intersection(good) for p in (fit, stop, cal, score))
            exact = training_peak_threshold(df, fit, .95)
            if exact == tau:
                break
            tau = exact
        if (pd.DatetimeIndex(targets.loc[score, "target_time"]) >= boundary).any():
            raise AssertionError("score target crossed sealed boundary")
        # compact grid choice on stop MAE, as in training._fold
        y = targets.y
        best = None
        for params in point_grid(cfg):
            m = fit_point(x.loc[fit], y.loc[fit], x.loc[stop], y.loc[stop], cfg, peak_threshold=tau, params=params)
            mae = float(np.mean(np.abs(y.loc[stop].to_numpy() - m.predict(x.loc[stop]))))
            if best is None or mae < best[0]:
                best = (mae, params)
        out[fold_id] = {"x": x, "targets": targets, "tau": tau, "fit": fit, "stop": stop, "cal": cal,
                        "score": score, "params": best[1]}
    return out


def rows_for(ctx, horizon, fold, name, pc, ps, **extra):
    t, tau, cal, score = ctx["targets"], ctx["tau"], ctx["cal"], ctx["score"]
    yc = t.loc[cal, "y"].to_numpy(float)
    pc = np.asarray(pc, float)
    ok = np.isfinite(pc) & np.isfinite(yc)
    cutoff = _cutoff(yc[ok], pc[ok], tau, t.loc[cal[ok], "target_time"]) if ok.sum() >= 30 else float("inf")
    return _row(score, horizon, fold, name, t, tau, np.asarray(ps, float), cutoff, **extra)


def point_models(df, cfg, ctx, horizon, fold, series, paths, *, include_mstl=True, seed=None) -> list[pd.DataFrame]:
    """p1_latest, CBL representative, LGBM(no-holiday, weight 2), DSHW, 3-model blend, MSTL."""
    c = ctx
    y = c["targets"].y
    cols = [k for k in c["x"].columns if k not in HOLIDAY]
    cfg_seed = cfg if seed is None else {**cfg, "seed": seed}
    lgbm = fit_point(c["x"].loc[c["fit"], cols], y.loc[c["fit"]], c["x"].loc[c["stop"], cols], y.loc[c["stop"]],
                     cfg_seed, peak_threshold=c["tau"], peak_weight=2.0, params=c["params"])
    dshw = DSHW().fit(series, c["fit"].max())
    comp, base = {}, {}
    for part in ("cal", "score"):
        idx = c[part]
        b = baseline_predictions(df, idx, horizon)
        cb = cbl_all_predictions(df, idx, horizon)
        cb["c3_holiday_hybrid"] = cb["c3_holiday_mid_4_6"].combine_first(cb["c1_mid_6_10"])
        base[part] = {"p1_latest": b["p1_latest"].to_numpy(float), "cbl": cb[CBL_FOR(horizon)].to_numpy(float)}
        comp[part] = [lgbm.predict(c["x"].loc[idx, cols]), base[part]["cbl"], dshw.predict(series, idx, horizon)]
    w = simplex_weights(comp["cal"], y.loc[c["cal"]].to_numpy(float))
    mix = lambda part: sum(wi * ci for wi, ci in zip(w, comp[part]))
    rows = [rows_for(c, horizon, fold, "p1_latest", base["cal"]["p1_latest"], base["score"]["p1_latest"]),
            rows_for(c, horizon, fold, "cbl_rep", base["cal"]["cbl"], base["score"]["cbl"]),
            rows_for(c, horizon, fold, "lgbm_no_holiday_weight_2", comp["cal"][0], comp["score"][0]),
            rows_for(c, horizon, fold, "p4_dshw", comp["cal"][2], comp["score"][2]),
            rows_for(c, horizon, fold, "p4_blend3", mix("cal"), mix("score"), blend_weights=str(w))]
    if include_mstl:
        phi = mstl_phi(series, paths, c["fit"])
        rows.append(rows_for(c, horizon, fold, "p4_mstl_daily", mstl_forecast(series, paths, c["cal"], horizon, phi),
                             mstl_forecast(series, paths, c["score"], horizon, phi)))
    return rows


def day_boot(frame: pd.DataFrame, values: np.ndarray, n=1000, seed=42):
    if not len(frame):
        return np.nan, [np.nan, np.nan]
    days = pd.DatetimeIndex(frame.target_time).normalize()
    g = pd.DataFrame({"d": days, "v": values}).groupby("d").v.agg(["sum", "count"])
    s, k = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    pick = np.random.default_rng(seed).integers(0, len(s), size=(n, len(s)))
    boot = s[pick].sum(1) / k[pick].sum(1)
    return float(values.mean()), [float(v) for v in np.quantile(boot, [.025, .975])]


def compare(rows: pd.DataFrame, a: str, b: str, horizon: int, tau_col: str = "tau") -> dict:
    """Peak MAE gain of b over a on common valid rows, plus pooled scores and fold ratios."""
    left = rows.loc[rows.horizon.eq(horizon) & rows.model.eq(a)]
    right = rows.loc[rows.horizon.eq(horizon) & rows.model.eq(b)]
    m = left[KEYS + ["y", "pred", tau_col, "alert"]].merge(right[KEYS + ["pred", "alert"]], on=KEYS, suffixes=("_a", "_b"))
    m = m.loc[np.isfinite(m.y) & np.isfinite(m.pred_a) & np.isfinite(m.pred_b)]
    peaks = m.loc[m.y > m[tau_col]]
    gain = ((peaks.y - peaks.pred_a).abs() - (peaks.y - peaks.pred_b).abs()).to_numpy()
    est, ci = day_boot(peaks, gain)
    sa = score_predictions(m[KEYS + ["y"]].assign(tau=m[tau_col], pred=m.pred_a, alert=m.alert_a.astype(bool)))
    sb = score_predictions(m[KEYS + ["y"]].assign(tau=m[tau_col], pred=m.pred_b, alert=m.alert_b.astype(bool)))
    ratios = {}
    for f in sorted(peaks.fold.unique()):
        p = peaks.loc[peaks.fold.eq(f)]
        ia = (p.y - p.pred_a).abs().mean()
        ratios[f"fold{int(f)}"] = float((p.y - p.pred_b).abs().mean() / ia) if ia > 0 else np.nan
    return {"horizon": horizon, "a": a, "b": b, "n": len(m), "n_peak": len(peaks),
            "peak_mae_a": sa.get("peak_mae"), "peak_mae_b": sb.get("peak_mae"), "gain": est,
            "ci_low": ci[0], "ci_high": ci[1], "mae_a": sa.get("mae"), "mae_b": sb.get("mae"),
            "episode_f1_a": sa.get("episode_f1"), "episode_f1_b": sb.get("episode_f1"),
            "fp_a": sa.get("false_alarms_positions"), "fp_b": sb.get("false_alarms_positions"),
            "fold_ratios": ratios,
            "p4_gates_nominal": bool(np.isfinite(ci[0]) and ci[0] > 0
                                     and sb.get("episode_f1", 0) >= sa.get("episode_f1", 0) - .02
                                     and sb.get("false_alarms_positions", 0) <= 1.2 * sa.get("false_alarms_positions", 0)
                                     and all(r <= 1.5 for r in ratios.values())
                                     and list(ratios.values())[-1] <= 1.1)}
