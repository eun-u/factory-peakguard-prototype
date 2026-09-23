"""Development-only rolling-origin model comparison for task 05.

The public entry point deliberately has no holdout argument. It returns
out-of-fold scoring rows from the first 85% only. A separate, dated freeze
gate must own any later test evaluation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import day_bootstrap, paired_mae_improvement
from .evaluate import evaluate_all, match_episodes, score_predictions
from .models.conformal import (apply_conformal, assign_mondrian_bins,
                               fit_conformal, fit_mondrian_edges)
from .models.expected_exceedance import fit_exceedance, predict_exceedance
from .models.lgbm_point import fit_point, point_grid
from .models.lgbm_quantile import fit_quantiles, predict_quantiles
from .models.peak_prob import exceedance_from_quantiles, fit_classifier, predict_classifier


def _as_index(df):
    if "ts_end" in df.columns:
        df = df.set_index("ts_end", drop=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input needs DatetimeIndex or ts_end")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("ts_end must be unique and chronological")
    return df


def _valid_mask(x, used, origins, targets, df):
    valid = np.asarray(x.attrs.get("valid_mask", np.ones(len(x), dtype=bool)), dtype=bool).copy()
    if len(valid) != len(x):
        raise ValueError("Feature validity mask has wrong length")
    used = pd.DatetimeIndex(used)
    if len(used) != len(origins) or (used > origins).any():
        raise AssertionError("Feature availability exceeds its forecast origin")
    valid &= targets.y.notna().to_numpy()
    valid &= ~targets.target_repaired.to_numpy()
    valid &= np.asarray(used <= origins)
    return valid


def _partition(train, validation, horizon):
    """Use training tail for stopping, validation head for calibration, tail for score."""
    train = pd.DatetimeIndex(train)
    validation = pd.DatetimeIndex(validation)
    if len(train) < 200 or len(validation) < 120:
        raise ValueError("Fold too small for disjoint fit, stop, calibration, scoring")
    stop_start = int(len(train) * .80)
    gap = pd.Timedelta(minutes=15 * horizon)
    fit = train[train + gap < train[stop_start]]
    stop = train[stop_start:]
    cal_end = max(40, int(len(validation) * .35))
    cal = validation[:cal_end]
    score = validation[cal_end:]
    if not len(fit) or not len(stop) or not len(cal) or not len(score):
        raise ValueError("Fold partition has an empty segment")
    if not (fit.max() + gap < stop.min() and stop.max() + gap < cal.min()
            and cal.max() + gap < score.min()):
        # Purge calibration/score boundaries without consuming future targets.
        cal = cal[cal > stop.max() + gap]
        score = score[score > cal.max() + gap]
    if not len(cal) or not len(score):
        raise ValueError("Insufficient validation rows after target embargo")
    if not (fit.max() + gap < stop.min() and stop.max() + gap < cal.min()
            and cal.max() + gap < score.min()):
        raise AssertionError("Target-label embargo failed")
    return fit, stop, cal, score


def _cutoff(y, predicted, tau, times):
    """Choose an episode-F1 cutoff using calibration rows only."""
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    grid = np.unique(np.quantile(predicted[np.isfinite(predicted)], np.linspace(.5, .99, 35)))
    if not len(grid):
        return float("inf")
    best = (-1.0, -float("inf"))
    for cutoff in grid:
        match = match_episodes(y > tau, predicted > cutoff, times)
        den = 2 * match["tp"] + match["fp"] + match["fn"]
        f1 = 2 * match["tp"] / den if den else 0.0
        key = (f1, cutoff)
        if key > best:
            best = key
    return float(best[1])


def _ordered_quantiles(values):
    """Repair crossings after independent upper-tail calibration."""
    levels = sorted(values)
    ordered = np.maximum.accumulate(np.column_stack([values[level] for level in levels]), axis=1)
    return {level: ordered[:, i] for i, level in enumerate(levels)}


def _row(origins, horizon, fold, model, targets, tau, pred, cutoff, **extra):
    from .holidays import calendar_flags
    flags = calendar_flags(pd.DatetimeIndex(targets.loc[origins, "target_time"]))
    out = pd.DataFrame({"origin": origins, "target_time": targets.loc[origins, "target_time"].to_numpy(),
                        "horizon": horizon, "fold": fold, "model": model,
                        "y": targets.loc[origins, "y"].to_numpy(dtype=float),
                        "pred": np.asarray(pred, dtype=float), "tau": tau,
                        "holiday_near": flags[["is_offday", "pre_holiday", "post_holiday", "bridge_day"]].any(axis=1).to_numpy()})
    out["alert"] = out.pred.to_numpy() > cutoff
    out["alert_cutoff"] = cutoff
    for name, value in extra.items():
        out[name] = value
    return out


def _baseline_frames(df, fit, cal, score, horizon, fold, targets, tau):
    from .models.baselines import baseline_predictions
    from .models.cbl import cbl_all_predictions
    bcal = baseline_predictions(df, cal, horizon)
    bscore = baseline_predictions(df, score, horizon)
    ccal = cbl_all_predictions(df, cal, horizon)
    cscore = cbl_all_predictions(df, score, horizon)
    cal_all = pd.concat([bcal, ccal], axis=1)
    score_all = pd.concat([bscore, cscore], axis=1)
    cal_all["c3_holiday_hybrid"] = cal_all["c3_holiday_mid_4_6"].combine_first(cal_all["c1_mid_6_10"])
    score_all["c3_holiday_hybrid"] = score_all["c3_holiday_mid_4_6"].combine_first(score_all["c1_mid_6_10"])
    rows = []
    for col in cal_all.columns.intersection(score_all.columns):
        yc = targets.loc[cal, "y"].to_numpy(dtype=float)
        pc = cal_all.loc[cal, col].to_numpy(dtype=float)
        valid = np.isfinite(pc) & np.isfinite(yc)
        if valid.sum() < 30:
            continue
        cutoff = _cutoff(yc[valid], pc[valid], tau, targets.loc[cal[valid], "target_time"])
        ps = score_all.loc[score, col].to_numpy(dtype=float)
        rows.append(_row(score, horizon, fold, col, targets, tau, ps, cutoff))
    return rows


def _fold(df, origins, horizon, fold_id, cfg, model_dir=None):
    from .features import build_features
    from .targets import point_targets, training_peak_threshold
    from .split import make_splits

    folds, _ = make_splits(origins, horizon, dev_frac=1.0,
                           n_folds=int(cfg.get("split", {}).get("n_folds", 3)))
    fold = folds[fold_id]
    raw_train = getattr(fold, "train")
    raw_validation = getattr(fold, "validation")
    all_origins = raw_train.append(raw_validation)
    targets = point_targets(df, all_origins, horizon)
    # Determine clean partition first, then estimate tau only through the
    # actual fitted prefix. Rebuild peak-history features with that tau.
    x, latest = build_features(df, all_origins, horizon, {**cfg, "_tau": None})
    if not x.index.equals(all_origins):
        x = x.reindex(all_origins)
    valid = _valid_mask(x, latest, all_origins, targets, df)
    good = all_origins[valid]
    train = raw_train.intersection(good)
    validation = raw_validation.intersection(good)
    fit, stop, cal, score = _partition(train, validation, horizon)
    tau = training_peak_threshold(df, fit, cfg.get("peak", {}).get("tau_quantile", .95))
    for _ in range(3):
        x, latest = build_features(df, all_origins, horizon, {**cfg, "_tau": tau})
        valid_with_peak = _valid_mask(x, latest, all_origins, targets, df)
        good_with_peak = all_origins[valid_with_peak]
        fit, stop, cal, score = (part.intersection(good_with_peak)
                                 for part in (fit, stop, cal, score))
        if min(map(len, (fit, stop, cal, score))) < 30:
            raise ValueError("Fold lost too many rows after peak-history validation")
        exact_tau = training_peak_threshold(df, fit, cfg.get("peak", {}).get("tau_quantile", .95))
        if exact_tau == tau:
            break
        tau = exact_tau
    else:
        raise AssertionError("Peak threshold did not stabilize on the fit prefix")
    y = targets.y
    rows = _baseline_frames(df, fit, cal, score, horizon, fold_id, targets, tau)
    x_fit, x_stop, x_cal, x_score = (x.loc[index] for index in (fit, stop, cal, score))
    y_fit, y_stop, y_cal = (y.loc[index] for index in (fit, stop, cal))
    # Select a compact hyperparameter setting on the stop window only.
    choices = []
    for params in point_grid(cfg):
        fitted = fit_point(x_fit, y_fit, x_stop, y_stop, cfg, peak_threshold=tau, params=params)
        choices.append((float(np.mean(abs(y_stop.to_numpy() - fitted.predict(x_stop)))), params, fitted))
    _, chosen_params, base_model = min(choices, key=lambda row: row[0])
    holiday_columns = [col for col in ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day") if col in x.columns]
    for ablated in (False, True):
        columns = [col for col in x.columns if col not in holiday_columns] if ablated else list(x.columns)
        point_models = [(1, fit_point(x_fit[columns], y_fit, x_stop[columns], y_stop, cfg,
                                      peak_threshold=tau, params=chosen_params) if ablated else base_model)]
        for weight in cfg.get("peak_weight", [1, 2, 4]):
            if float(weight) != 1:
                point_models.append((weight, fit_point(x_fit[columns], y_fit, x_stop[columns], y_stop, cfg,
                                                       peak_threshold=tau, peak_weight=float(weight),
                                                       params=chosen_params)))
        for weight, model in point_models:
            stem = "lgbm_no_holiday" if ablated else "lgbm"
            name = stem if float(weight) == 1 else f"{stem}_weight_{weight}"
            cutoff = _cutoff(y_cal.to_numpy(), model.predict(x_cal[columns]), tau,
                             targets.loc[cal, "target_time"])
            rows.append(_row(score, horizon, fold_id, name, targets, tau,
                             model.predict(x_score[columns]), cutoff))
            if model_dir is not None:
                import joblib
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": model, "score_origins": score, "horizon": horizon,
                             "fold": fold_id, "tau": tau, "model_name": name,
                             "feature_names": columns},
                            model_dir / f"development_h{horizon}_fold{fold_id}_{name}.joblib")
    # Holiday features need a separate acceptance test; the risk layer uses
    # the conservative feature set until that test exists for quantiles.
    quantile_columns = [col for col in x.columns if col not in holiday_columns]
    quantile_models = fit_quantiles(x_fit[quantile_columns], y_fit,
                                    x_stop[quantile_columns], y_stop, cfg, params=chosen_params)
    q_cal = predict_quantiles(quantile_models, x_cal[quantile_columns])
    q_score = predict_quantiles(quantile_models, x_score[quantile_columns])
    edges = fit_mondrian_edges(q_cal[.5], cfg.get("conformal", {}).get("mondrian_bins", [.5, .9]))
    cal_bins = assign_mondrian_bins(q_cal[.5], edges)
    score_bins = assign_mondrian_bins(q_score[.5], edges)
    expected = (fit_exceedance(x_fit[quantile_columns], y_fit, tau, cfg)
                if cfg.get("optional", {}).get("expected_exceedance", True) else None)
    variants = {}
    for method in ("a", "b"):
        grouping_cal = None if method == "a" else cal_bins
        grouping_score = None if method == "a" else score_bins
        q_calibrated_cal = dict(q_cal)
        q_calibrated_score = dict(q_score)
        # A calibrated distribution needs independent calibration information;
        # probability cutoff selection is therefore based on an inner split.
        half = len(cal) // 2
        fit_slice = slice(None, half)
        cutoff_start = min(len(cal), half + horizon + 1)
        cutoff_slice = slice(cutoff_start, None)
        if cutoff_start >= len(cal):
            raise ValueError("Calibration block too short for threshold embargo")
        for alpha in (.9, .95, .975):
            bins_fit = None if grouping_cal is None else grouping_cal[fit_slice]
            bins_all = None if grouping_cal is None else grouping_cal
            fitted = fit_conformal(y_cal.to_numpy()[fit_slice], q_cal[alpha][fit_slice], alpha, bins_fit)
            q_calibrated_cal[alpha] = apply_conformal(q_cal[alpha], fitted, bins_all)
            q_calibrated_score[alpha] = apply_conformal(q_score[alpha], fitted, grouping_score)
        q_calibrated_cal = _ordered_quantiles(q_calibrated_cal)
        q_calibrated_score = _ordered_quantiles(q_calibrated_score)
        p_cal = exceedance_from_quantiles(q_calibrated_cal, tau)
        p_score = exceedance_from_quantiles(q_calibrated_score, tau)
        cutoff = _cutoff(y_cal.to_numpy()[cutoff_slice], p_cal[cutoff_slice], tau,
                         targets.loc[cal[cutoff_slice], "target_time"])
        extras = {f"q{int(alpha*1000) if alpha == .975 else int(alpha*100)}": q_score[alpha]
                  for alpha in sorted(q_score)}
        extras.update(q90_cal=q_calibrated_score[.9], q95_cal=q_calibrated_score[.95],
                      q975_cal=q_calibrated_score[.975], p_exceed=p_score,
                      q50_top_edge=float(edges[-1]),
                      exp_exceed=predict_exceedance(expected, x_score[quantile_columns], p_score) if expected else np.nan,
                      conformal_method=method)
        # `alert` uses the calibrated probability threshold, not q50.
        row = _row(score, horizon, fold_id, f"lgbm_quantile_{method}", targets, tau,
                   q_score[.5], float("inf"), **extras)
        row["alert"] = p_score > cutoff
        row["alert_cutoff"] = cutoff
        variants[method] = row
    raw_p = exceedance_from_quantiles(q_score, tau)
    raw_cutoff = _cutoff(y_cal.to_numpy(), exceedance_from_quantiles(q_cal, tau),
                         tau, targets.loc[cal, "target_time"])
    raw_extras = {f"q{int(alpha*1000) if alpha == .975 else int(alpha*100)}": q_score[alpha]
                  for alpha in sorted(q_score)}
    raw_extras.update(p_exceed=raw_p, q50_top_edge=float(edges[-1]), conformal_method="raw")
    raw_row = _row(score, horizon, fold_id, "lgbm_quantile_raw", targets, tau,
                   q_score[.5], float("inf"), **raw_extras)
    raw_row["alert"] = raw_p > raw_cutoff
    raw_row["alert_cutoff"] = raw_cutoff
    rows.append(raw_row)
    rows.extend(variants.values())
    return rows, {"fold": fold_id, "horizon": horizon, "tau": tau, "n_fit": len(fit),
                  "n_stop": len(stop), "n_cal": len(cal), "n_score": len(score),
                  "fit_end": str(fit.max()), "stop_end": str(stop.max()),
                  "cal_end": str(cal.max()), "score_start": str(score.min()),
                  "score_end": str(score.max()),
                  "chosen_params": chosen_params}


def _selection(predictions, metrics, cfg):
    out = {"by_horizon": {}, "adoption": {}, "protocol": "development_oof_only"}
    n_boot = int(cfg.get("bootstrap", {}).get("n", 1000))
    for horizon, group in predictions.groupby("horizon"):
        ranking_ci_cache = {}
        def one(name):
            return group.loc[group.model == name]
        def pooled(name):
            return score_predictions(one(name))
        def comparison(first, second, subset=None, peak_only=False):
            a, b = one(first), one(second)
            if subset is not None:
                a, b = a.loc[a[subset]], b.loc[b[subset]]
            return paired_mae_improvement(a, b, peak_only=peak_only, n=n_boot,
                                          seed=int(cfg.get("seed", 42)))
        def family_representative(names):
            names = [name for name in names if not one(name).empty]
            if not names:
                return None
            keys = ["target_time", "horizon", "fold"]
            wide = group.loc[group.model.isin(names)].pivot(index=keys, columns="model", values="pred")
            actual = group.drop_duplicates(keys).set_index(keys)["y"]
            common = wide.join(actual).dropna(subset=names + ["y"])
            if common.empty:
                return None
            return min(names, key=lambda name: (float(np.mean(abs(common.y - common[name]))), name))
        def episode_and_alarm_ci(name):
            if name in ranking_ci_cache:
                return ranking_ci_cache[name]
            part = one(name).loc[lambda rows: np.isfinite(rows.pred)].copy()
            part["day"] = pd.to_datetime(part.target_time).dt.normalize()
            daily = []
            for _, block in part.groupby(["fold", "day"]):
                block = block.sort_values("target_time")
                actual = block.y.to_numpy(dtype=float) > block.tau.to_numpy(dtype=float)
                alert = block.alert.to_numpy(dtype=bool)
                matched = match_episodes(actual, alert, block.target_time)
                daily.append((matched["tp"], matched["fp"], matched["fn"],
                              int((~actual & alert).sum())))
            values = np.asarray(daily, dtype=float)
            if not len(values):
                result = {"episode_f1": (float("nan"), float("nan")),
                          "false_alarm_positions": (float("nan"), float("nan"))}
            else:
                rng = np.random.default_rng(int(cfg.get("seed", 42)))
                picked = rng.integers(len(values), size=(n_boot, len(values)))
                totals = values[picked].sum(axis=1)
                denom = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
                f1 = np.divide(2 * totals[:, 0], denom, out=np.zeros_like(denom), where=denom > 0)
                result = {"episode_f1": tuple(np.quantile(f1, [.025, .975])),
                          "false_alarm_positions": tuple(np.quantile(totals[:, 3], [.025, .975]))}
            ranking_ci_cache[name] = result
            return result

        holiday = comparison("lgbm_no_holiday", "lgbm", subset="holiday_near")
        overall = comparison("lgbm_no_holiday", "lgbm")
        holiday_adopt = bool(np.isfinite(holiday["ci95"][0]) and holiday["ci95"][0] > 0
                             and overall["estimate"] >= 0)
        out["adoption"][f"h{horizon}_holiday"] = {
            "adopted": holiday_adopt, "holiday_mae_improvement": holiday,
            "overall_mae_improvement": overall}
        base_name = "lgbm" if holiday_adopt else "lgbm_no_holiday"
        eligible = [base_name]
        base_fp = pooled(base_name)["false_alarms_positions"]
        for weight in cfg.get("peak_weight", [1, 2, 4]):
            if float(weight) == 1:
                continue
            name = f"{base_name}_weight_{weight}"
            gain = comparison(base_name, name, peak_only=True)
            fp = pooled(name)["false_alarms_positions"]
            adopt = bool(np.isfinite(gain["estimate"]) and gain["estimate"] > 0
                         and fp <= base_fp * 1.20)
            out["adoption"][f"h{horizon}_weight_{weight}"] = {
                "adopted": adopt, "peak_mae_improvement": gain,
                "base_false_alarms": base_fp, "weighted_false_alarms": fp}
            if adopt:
                eligible.append(name)
        # The family representative is frozen by overall development CV MAE.
        choice = {}
        for prefix, names in {"persistence": ["p1_latest", "p2_hour_slot", "p3_recent_mean"],
                              "seasonal": ["s1_day", "s2_week", "s3_day_week_mean"]}.items():
            representative = family_representative(names)
            if representative:
                choice[prefix] = representative
        cbl_names = ["c1_mid_6_10", "c2_max_4_5", "c3_holiday_hybrid",
                     "c1a_mid_6_10_adjusted", "c2a_max_4_5_adjusted"]
        representative = family_representative(cbl_names)
        if representative:
            choice["cbl"] = representative
        eligible.extend(choice[key] for key in ("cbl", "persistence", "seasonal") if key in choice)

        # Sequential preregistered ranking: peak MAE CI, episode F1,
        # false alarm count, calibration, then simplicity. The comparison is
        # paired on identical target timestamps; missing baseline rows are
        # omitted from both sides of that comparison.
        selected = eligible[0]
        evidence = []
        for challenger in eligible[1:]:
            gain = comparison(selected, challenger, peak_only=True)
            lower, upper = gain["ci95"]
            if np.isfinite(lower) and lower > 0:
                decision = "challenger_peak_mae_ci"
            elif np.isfinite(upper) and upper < 0:
                decision = "incumbent_peak_mae_ci"
            else:
                left_ci, right_ci = episode_and_alarm_ci(selected), episode_and_alarm_ci(challenger)
                if right_ci["episode_f1"][0] > left_ci["episode_f1"][1]:
                    decision = "challenger_episode_f1_ci"
                elif left_ci["episode_f1"][0] > right_ci["episode_f1"][1]:
                    decision = "incumbent_episode_f1_ci"
                elif right_ci["false_alarm_positions"][1] < left_ci["false_alarm_positions"][0]:
                    decision = "challenger_false_alarm_ci"
                elif left_ci["false_alarm_positions"][1] < right_ci["false_alarm_positions"][0]:
                    decision = "incumbent_false_alarm_ci"
                else:
                    # Point candidates have no comparable coverage estimate;
                    # uncertainty coverage is selected in the separate A/B gate.
                    complexity = lambda name: (int(name.startswith("lgbm")), int("weight" in name), name)
                    decision = ("challenger_simpler_after_ci_overlap" if complexity(challenger) < complexity(selected)
                                else "incumbent_simpler_after_ci_overlap")
            evidence.append({"incumbent": selected, "challenger": challenger,
                             "peak_mae_gain": gain, "episode_ci": episode_and_alarm_ci(challenger),
                             "decision": decision})
            if decision.startswith("challenger"):
                selected = challenger
        choice["point_model"] = selected
        choice["ranking_evidence"] = evidence
        qa = group.loc[group.model == "lgbm_quantile_a"]
        qb = group.loc[group.model == "lgbm_quantile_b"]
        if not qa.empty and not qb.empty:
            a = score_predictions(qa)
            b = score_predictions(qb)
            a_cov = float(a["top_coverage_0.95"])
            b_cov = float(b["top_coverage_0.95"])
            a_pin = float(a["pinball_mean"])
            b_pin = float(b["pinball_mean"])
            adopt = abs(b_cov - .95) < abs(a_cov - .95) and b_pin <= a_pin * 1.05
            choice["conformal"] = "b" if adopt else "a"
            out["adoption"][f"h{horizon}_mondrian"] = {"adopted": bool(adopt),
                "top_coverage_a": a_cov, "top_coverage_b": b_cov,
                "pinball_a": a_pin, "pinball_b": b_pin}
            if cfg.get("optional", {}).get("expected_exceedance", True):
                from scipy.stats import spearmanr
                chosen_q = qb if adopt else qa
                episode_rows = []
                for fold_id, part in chosen_q.groupby("fold"):
                    part = part.sort_values("target_time")
                    actual = part.y.to_numpy(dtype=float) > part.tau.to_numpy(dtype=float)
                    from .evaluate import _episodes
                    for first, last in _episodes(actual, part.target_time):
                        block = part.iloc[first:last + 1]
                        episode_rows.append({"target_time": block.target_time.iloc[0],
                                             "actual_excess": float((block.y - block.tau).max()),
                                             "predicted_excess": float(block.exp_exceed.max())})
                episodes = pd.DataFrame(episode_rows)
                if len(episodes) >= 5 and episodes.actual_excess.nunique() > 1:
                    def rank_corr(sample):
                        if sample.actual_excess.nunique() < 2 or sample.predicted_excess.nunique() < 2:
                            return float("nan")
                        return float(spearmanr(sample.actual_excess, sample.predicted_excess).statistic)
                    rank = rank_corr(episodes)
                    ci = day_bootstrap(episodes, rank_corr, n=n_boot, seed=int(cfg.get("seed", 42)))
                else:
                    rank, ci = float("nan"), (float("nan"), float("nan"))
                adopt_excess = bool(np.isfinite(ci[0]) and ci[0] > 0)
                out["adoption"][f"h{horizon}_expected_exceedance"] = {
                    "adopted": adopt_excess, "spearman": rank, "ci95": list(ci),
                    "episodes": len(episodes)}
        if choice.get("cbl"):
            comparison = paired_mae_improvement(group.loc[group.model == choice["cbl"]],
                                                 group.loc[group.model == selected],
                                                 peak_only=True, n=n_boot, seed=int(cfg.get("seed", 42)))
            choice["vs_cbl_peak_mae"] = comparison
        if choice.get("persistence"):
            choice["vs_persistence_peak_mae"] = paired_mae_improvement(
                group.loc[group.model == choice["persistence"]], group.loc[group.model == selected],
                peak_only=True, n=n_boot, seed=int(cfg.get("seed", 42)))
        out["by_horizon"][str(horizon)] = choice
    return out


def run_development(df: pd.DataFrame, cfg: dict, output_dir="outputs") -> dict:
    """Train and score all direct horizons strictly before the configured test origin."""
    df = _as_index(df)
    from .split import make_splits
    output_dir = Path(output_dir)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    all_origins = df.index[df.index < pd.Timestamp(cfg["split"]["test_start_origin"])]
    if not len(all_origins):
        raise ValueError("No development origins before fixed test boundary")
    rows, manifests = [], []
    for horizon in cfg.get("horizons", [1, 4, 16, 96]):
        # `make_splits` is called per horizon to purge target-overlap boundaries.
        folds, _ = make_splits(all_origins, horizon, dev_frac=1.0,
                               n_folds=int(cfg.get("split", {}).get("n_folds", 3)))
        for fold_id in range(len(folds)):
            started = time.perf_counter()
            print(f"[development] h={horizon} fold={fold_id + 1}/{len(folds)} fitting", flush=True)
            fold_rows, evidence = _fold(df, all_origins, int(horizon), fold_id, cfg,
                                        output_dir / "models")
            rows.extend(fold_rows)
            manifests.append(evidence)
            print(f"[development] h={horizon} fold={fold_id + 1}/{len(folds)} "
                  f"scored={evidence['n_score']} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    if (predictions.origin >= pd.Timestamp(cfg["split"]["test_start_origin"])).any():
        raise AssertionError("Development predictions include test origins")
    print("[development] evaluating date-block intervals and model selection", flush=True)
    metrics = evaluate_all(predictions, cfg=cfg)
    selection = _selection(predictions, metrics, cfg)
    paths = {"predictions": output_dir / "predictions" / "development_oof.csv",
             "metrics": output_dir / "tables" / "development_cv.csv",
             "selection": output_dir / "logs" / "development_selection.json"}
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    paths["selection"].write_text(json.dumps({"selection": selection, "folds": manifests},
                                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"predictions": predictions, "metrics": metrics, "selection": selection,
            "folds": manifests, "paths": {k: str(v) for k, v in paths.items()}}


def freeze_and_evaluate(df: pd.DataFrame, cfg: dict, output_dir="outputs") -> dict:
    """Run the one-time holdout after the configured freeze instant.

    This entry point does not select or retune from test labels. It reads the
    development decision manifest and writes final_test.csv only once.
    """
    from .finalize import freeze_and_evaluate as audited_freeze
    return audited_freeze(df, cfg, output_dir)



def run_t2_development(df: pd.DataFrame, cfg: dict, output_dir="outputs") -> dict:
    """Secondary next-day maximum task on development dates only.

    Daily targets are evaluated as daily maxima, not as 15-minute episodes.
    The first development block fits; each later block is scored once.
    """
    from .features import build_features
    from .targets import next_day_max_targets, training_peak_threshold

    df = _as_index(df)
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    origins = df.index[(df.index.hour == 23) & (df.index.minute == 45) & (df.index < boundary)]
    target = next_day_max_targets(df, origins)
    origins = origins[(target.valid_target & (target.target_time < boundary)).to_numpy()]
    target = target.loc[origins]
    if len(origins) < 100:
        raise ValueError("Too few clean T2 development days")
    x, latest = build_features(df, origins, 96, {**cfg, "_include_cbl": False})
    valid = np.asarray(x.attrs["valid_mask"], dtype=bool).copy()
    valid &= np.asarray(pd.DatetimeIndex(latest) <= origins)
    valid &= target.valid_target.to_numpy()
    origins = origins[valid]
    x, target = x.loc[origins], target.loc[origins]
    initial = int(len(origins) * .4)
    n_folds = int(cfg.get("split", {}).get("n_folds", 3))
    edges = np.linspace(initial, len(origins), n_folds + 1, dtype=int)
    rows, folds = [], []
    for fold_id in range(n_folds):
        val = origins[edges[fold_id]:edges[fold_id + 1]]
        train = origins[:edges[fold_id]]
        train = train[target.loc[train, "target_time"].to_numpy() < val.min()]
        stop_start = int(len(train) * .75)
        fit = train[:stop_start]
        stop = train[stop_start:]
        fit = fit[target.loc[fit, "target_time"].to_numpy() < stop.min()]
        cal_end = max(5, int(len(val) * .30))
        cal = val[:cal_end]
        score = val[cal_end:]
        cal = cal[target.loc[cal, "target_time"].to_numpy() < score.min()]
        stop = stop[target.loc[stop, "target_time"].to_numpy() < cal.min()]
        if min(map(len, (fit, stop, cal, score))) < 5:
            raise ValueError("T2 fold too short after daily target embargo")
        tau = training_peak_threshold(df, fit, cfg.get("peak", {}).get("tau_quantile", .95))
        fitted = fit_point(x.loc[fit], target.loc[fit, "y"], x.loc[stop],
                           target.loc[stop, "y"], cfg)
        p = fitted.predict(x.loc[score])
        days = score.normalize()
        def known_day_max(dates):
            result = []
            for origin in dates:
                window = df.loc[(df.index > origin - pd.Timedelta(days=1)) & (df.index <= origin)]
                repaired = window.get("time_repaired", pd.Series(False, index=window.index)).fillna(True).astype(bool)
                result.append(float(window.power.max()) if len(window) == 96 and window.power.notna().all()
                              and not repaired.any() else np.nan)
            return np.asarray(result)
        baselines = {"t2_recent_24h_max": known_day_max(score),
                     "t2_last_week_max": known_day_max(score - pd.Timedelta(days=7)),
                     "t2_lgbm": p}
        for name, prediction in baselines.items():
            rows.append(pd.DataFrame({"origin": score, "target_time": target.loc[score, "target_time"].to_numpy(),
                                      "horizon": "T2", "fold": fold_id, "model": name,
                                      "y": target.loc[score, "y"].to_numpy(dtype=float),
                                      "pred": prediction, "tau": tau}))
        folds.append({"fold": fold_id, "fit_end": str(fit.max()), "stop_end": str(stop.max()),
                      "cal_end": str(cal.max()), "score_start": str(score.min()), "score_end": str(score.max())})
    prediction = pd.concat(rows, ignore_index=True)
    metric_rows = []
    for name, part in prediction.groupby("model"):
        ok = np.isfinite(part.pred)
        part = part.loc[ok]
        actual = part.y.to_numpy() > part.tau.to_numpy()
        pred_peak = part.pred.to_numpy() > part.tau.to_numpy()
        tp = int((actual & pred_peak).sum()); fp = int((~actual & pred_peak).sum())
        fn = int((actual & ~pred_peak).sum())
        metric_rows.append({"model": name, "n": len(part), "mae": float(np.mean(abs(part.y - part.pred))),
                            "daily_peak_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")})
    metrics = pd.DataFrame(metric_rows)
    output_dir = Path(output_dir)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_dir / "predictions" / "t2_development_oof.csv", index=False)
    metrics.to_csv(output_dir / "tables" / "t2_development_cv.csv", index=False)
    return {"predictions": prediction, "metrics": metrics, "folds": folds}
