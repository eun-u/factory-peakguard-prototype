"""Preregistered development-only peak-threshold sensitivity (2026-09-24 A).

The original selected point model and three baseline-family representatives
are fixed. Only fold-fit peak thresholds, their history features, weighted
point fit, and calibration alert cutoffs change at q=.90/.975. No holdout is
read and the original selection artifact is never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from scripts.diagnose_h96_0924 import _assert_development_frame, _boundary, _checked_output
from src.bootstrap import paired_mae_improvement
from src.evaluate import match_episodes, score_predictions
from src.features import build_features
from src.models.baselines import baseline_predictions
from src.models.cbl import cbl_all_predictions
from src.models.lgbm_point import fit_point
from src.split import make_splits
from src.targets import point_targets, training_peak_threshold
from src.training import _cutoff, _partition, _row, _valid_mask


SENSITIVITY_QUANTILES = (.90, .975)
REFERENCE_QUANTILE = .95
N_BOOT = 1000
HOLIDAY_COLUMNS = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")
FOLD_KEYS = ("n_fit", "n_stop", "n_cal", "n_score", "fit_end", "stop_end", "cal_end", "score_start", "score_end")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_candidates(selection: dict, horizon: int) -> list[str]:
    chosen = selection["by_horizon"][str(horizon)]
    names = ([chosen["point_model"]] if chosen["point_model"].startswith("lgbm") else []) + [
        chosen["cbl"], chosen["persistence"], chosen["seasonal"]]
    names = list(dict.fromkeys(names))
    if chosen["point_model"] not in names:
        raise AssertionError("Original selected model is absent from its fixed candidate subset")
    return names


def _base_fold(df: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
               fold_id: int, cfg: dict, boundary: pd.Timestamp) -> dict:
    folds, test = make_splits(origins, horizon, dev_frac=1.0,
                              n_folds=int(cfg["split"]["n_folds"]))
    if len(test):
        raise AssertionError("Development-only split returned test origins")
    raw_train, raw_validation = folds[fold_id].train, folds[fold_id].validation
    all_origins = raw_train.append(raw_validation)
    targets = point_targets(df, all_origins, horizon)
    if (targets.target_time >= boundary).any():
        raise AssertionError("A fold target enters the frozen test period")
    x0, used0 = build_features(df, all_origins, horizon, {**cfg, "_tau": None})
    if not x0.index.equals(all_origins):
        x0 = x0.reindex(all_origins)
    valid = _valid_mask(x0, used0, all_origins, targets, df)
    good = all_origins[valid]
    fit, stop, cal, score = _partition(raw_train.intersection(good),
                                       raw_validation.intersection(good), horizon)
    return {"all_origins": all_origins, "targets": targets, "base_fit": fit,
            "base_stop": stop, "base_cal": cal, "base_score": score,
            "horizon": horizon, "fold": fold_id}


def _at_tau(df: pd.DataFrame, base: dict, cfg: dict, quantile: float,
            expected: dict, original_score: pd.DatetimeIndex) -> dict:
    """Rebuild peak features and threshold exactly as original _fold, no tuning."""
    horizon, fold_id = base["horizon"], base["fold"]
    fit, stop, cal, score = (base[f"base_{name}"] for name in ("fit", "stop", "cal", "score"))
    tau = training_peak_threshold(df, fit, quantile)
    for _ in range(3):
        x, used = build_features(df, base["all_origins"], horizon, {**cfg, "_tau": tau})
        valid = _valid_mask(x, used, base["all_origins"], base["targets"], df)
        available = base["all_origins"][valid]
        fit, stop, cal, score = (part.intersection(available) for part in (fit, stop, cal, score))
        if min(map(len, (fit, stop, cal, score))) < 30:
            raise AssertionError(f"h={horizon} fold={fold_id} q={quantile} lost a partition")
        exact_tau = training_peak_threshold(df, fit, quantile)
        if exact_tau == tau:
            break
        tau = exact_tau
    else:
        raise AssertionError("Sensitivity threshold failed to stabilize")
    actual = {"horizon": horizon, "fold": fold_id, "tau_quantile": quantile, "tau": tau,
              "n_fit": len(fit), "n_stop": len(stop), "n_cal": len(cal), "n_score": len(score),
              "fit_end": str(fit.max()), "stop_end": str(stop.max()),
              "cal_end": str(cal.max()), "score_start": str(score.min()),
              "score_end": str(score.max())}
    if any(actual[key] != expected[key] for key in FOLD_KEYS):
        raise AssertionError(f"Sensitivity partition differs from frozen fold: {actual} vs {expected}")
    if not score.equals(original_score):
        raise AssertionError(f"Sensitivity score grid differs from original OOF h={horizon} fold={fold_id}")
    if quantile == REFERENCE_QUANTILE and tau != float(expected["tau"]):
        raise AssertionError("The q=.95 reference threshold did not reproduce")
    return {**base, "x": x, "fit": fit, "stop": stop, "cal": cal, "score": score,
            "tau": tau, "summary": actual, "expected": expected}


def _baseline_values(df: pd.DataFrame, origins: pd.DatetimeIndex,
                     horizon: int, name: str) -> np.ndarray:
    if name.startswith(("p", "s")):
        frame = baseline_predictions(df, origins, horizon)
        column = name
    else:
        frame = cbl_all_predictions(df, origins, horizon)
        column = "c3_holiday_mid_4_6" if name == "c3_holiday_hybrid" else name
    if column not in frame:
        raise KeyError(f"Unknown fixed baseline: {name}")
    return frame.loc[origins, column].to_numpy(dtype=float)


def _score_fold(df: pd.DataFrame, context: dict, cfg: dict, candidates: list[str]) -> list[pd.DataFrame]:
    h, fold, tau = context["horizon"], context["fold"], context["tau"]
    fit, stop, cal, score = (context[name] for name in ("fit", "stop", "cal", "score"))
    target, x = context["targets"], context["x"]
    y = target.y
    rows = []
    for name in candidates:
        if name.startswith("lgbm"):
            if h != 4 or name != "lgbm_no_holiday_weight_2":
                raise AssertionError("Only the fixed selected h4 LightGBM may be refitted")
            columns = [col for col in x if col not in HOLIDAY_COLUMNS]
            model = fit_point(x.loc[fit, columns], y.loc[fit], x.loc[stop, columns], y.loc[stop],
                              cfg, peak_threshold=tau, peak_weight=2.0,
                              params=context["expected"]["chosen_params"])
            cal_values = model.predict(x.loc[cal, columns])
            score_values = model.predict(x.loc[score, columns])
        else:
            cal_values = _baseline_values(df, cal, h, name)
            score_values = _baseline_values(df, score, h, name)
        y_cal = y.loc[cal].to_numpy(dtype=float)
        valid_cal = np.isfinite(y_cal) & np.isfinite(cal_values)
        if valid_cal.sum() < 30:
            raise AssertionError(f"Insufficient calibration cases for {name} h={h} fold={fold}")
        cutoff = _cutoff(y_cal[valid_cal], cal_values[valid_cal], tau,
                         target.loc[cal[valid_cal], "target_time"])
        row = _row(score, h, fold, name, target, tau, score_values, cutoff)
        row["tau_quantile"] = context["summary"]["tau_quantile"]
        rows.append(row)
    return rows


def _assert_q95_parity(recomputed: list[pd.DataFrame], oof: pd.DataFrame,
                       horizon: int, fold: int) -> list[dict]:
    """Require unchanged q95 predictions, labels, alerts and cutoffs."""
    records = []
    for fresh in recomputed:
        name = str(fresh.model.iloc[0])
        old = oof.loc[(oof.horizon == horizon) & (oof.fold == fold) &
                      (oof.model == name)].sort_values("origin")
        fresh = fresh.sort_values("origin")
        if not pd.DatetimeIndex(fresh.origin).equals(pd.DatetimeIndex(old.origin)) or not pd.DatetimeIndex(
                fresh.target_time).equals(pd.DatetimeIndex(old.target_time)):
            raise AssertionError(f"q95 parity timestamps changed: h={horizon} fold={fold} {name}")
        diffs = {}
        for field in ("y", "pred", "tau", "alert_cutoff"):
            newer = fresh[field].to_numpy(dtype=float)
            older = old[field].to_numpy(dtype=float)
            if not np.array_equal(np.isfinite(newer), np.isfinite(older)):
                raise AssertionError(f"q95 parity finite mask changed: {field} {name}")
            finite = np.isfinite(newer)
            delta = float(np.max(np.abs(newer[finite] - older[finite]))) if finite.any() else 0.
            if delta > 1e-10:
                raise AssertionError(f"q95 parity {field} changed by {delta}: {name}")
            diffs[f"{field}_max_abs_difference"] = delta
        if not np.array_equal(fresh.alert.to_numpy(dtype=bool), old.alert.to_numpy(dtype=bool)):
            raise AssertionError(f"q95 alert parity changed: {name}")
        records.append({"horizon": horizon, "fold": fold, "model": name, "n": len(fresh),
                        **diffs, "alert_exact": True})
    return records


def _day_metric_ci(frame: pd.DataFrame, *, n: int = N_BOOT, seed: int = 42) -> dict:
    """Sample target dates; retain within-day fold episodes and peak counts."""
    part = frame.loc[np.isfinite(frame.y) & np.isfinite(frame.pred)].copy()
    if part.empty:
        nan = [float("nan"), float("nan")]
        return {"peak_mae_ci95": nan, "episode_f1_ci95": nan, "false_alarm_ci95": nan}
    part["date"] = pd.to_datetime(part.target_time).dt.normalize()
    if (part.groupby("date").fold.nunique() > 1).any():
        raise AssertionError("A target date spans folds; original fold/day CI would differ from day CI")
    daily = []
    for _, day in part.groupby("date", sort=True):
        peak_error, peak_n, tp, fp, fn, false_positions = 0., 0, 0, 0, 0, 0
        for _, block in day.groupby("fold", sort=True):
            block = block.sort_values("target_time")
            actual = block.y.to_numpy(dtype=float) > block.tau.to_numpy(dtype=float)
            alert = block.alert.to_numpy(dtype=bool)
            peak_error += float(np.abs(block.pred.to_numpy(dtype=float)[actual] -
                                       block.y.to_numpy(dtype=float)[actual]).sum())
            peak_n += int(actual.sum())
            matched = match_episodes(actual, alert, block.target_time)
            tp += matched["tp"]; fp += matched["fp"]; fn += matched["fn"]
            false_positions += int((~actual & alert).sum())
        daily.append((peak_error, peak_n, tp, fp, fn, false_positions))
    values = np.asarray(daily, dtype=float)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(len(values), size=(n, len(values)))
    sums = values[chosen].sum(axis=1)
    peak_mae = np.divide(sums[:, 0], sums[:, 1], out=np.full(n, np.nan), where=sums[:, 1] > 0)
    denom = 2 * sums[:, 2] + sums[:, 3] + sums[:, 4]
    # Match src.training._selection: a sampled block with no episodes has F1=0.
    episode_f1 = np.divide(2 * sums[:, 2], denom, out=np.zeros(n), where=denom > 0)
    def interval(samples):
        finite = samples[np.isfinite(samples)]
        return [float(v) for v in np.quantile(finite, [.025, .975])] if len(finite) else [float("nan")] * 2
    return {"peak_mae_ci95": interval(peak_mae), "episode_f1_ci95": interval(episode_f1),
            "false_alarm_ci95": interval(sums[:, 5])}


def _metrics(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for (quantile, horizon, name), part in predictions.groupby(["tau_quantile", "horizon", "model"], sort=True):
        metrics = score_predictions(part)
        ci = _day_metric_ci(part, n=N_BOOT, seed=seed)
        rows.append({"tau_quantile": quantile, "horizon": horizon, "model": name,
                     **{key: metrics[key] for key in ("n", "peak_n", "mae", "peak_mae",
                                                       "episode_f1", "false_alarms_positions")},
                     **{f"{key}_{end}": value[i] for key, value in ci.items()
                        for i, end in enumerate(("low", "high"))}})
    return pd.DataFrame(rows)


def _reference_ranking(group: pd.DataFrame, metrics: pd.DataFrame,
                       candidates: list[str], seed: int) -> tuple[str, list[dict]]:
    """Repeat the existing sequential CI/tie-break rule on this fixed subset."""
    indexed = {name: group.loc[group.model == name] for name in candidates}
    measured = metrics.set_index("model")
    selected, evidence = candidates[0], []
    for challenger in candidates[1:]:
        gain = paired_mae_improvement(indexed[selected], indexed[challenger], peak_only=True,
                                      n=N_BOOT, seed=seed)
        lower, upper = gain["ci95"]
        if np.isfinite(lower) and lower > 0:
            decision = "challenger_peak_mae_ci"
        elif np.isfinite(upper) and upper < 0:
            decision = "incumbent_peak_mae_ci"
        else:
            left, right = measured.loc[selected], measured.loc[challenger]
            if right.episode_f1_ci95_low > left.episode_f1_ci95_high:
                decision = "challenger_episode_f1_ci"
            elif left.episode_f1_ci95_low > right.episode_f1_ci95_high:
                decision = "incumbent_episode_f1_ci"
            elif right.false_alarm_ci95_high < left.false_alarm_ci95_low:
                decision = "challenger_false_alarm_ci"
            elif left.false_alarm_ci95_high < right.false_alarm_ci95_low:
                decision = "incumbent_false_alarm_ci"
            else:
                complexity = lambda name: (int(name.startswith("lgbm")), int("weight" in name), name)
                decision = ("challenger_simpler_after_ci_overlap"
                            if complexity(challenger) < complexity(selected)
                            else "incumbent_simpler_after_ci_overlap")
        evidence.append({"incumbent": selected, "challenger": challenger,
                         "peak_mae_gain": gain, "decision": decision})
        if decision.startswith("challenger"):
            selected = challenger
    return selected, evidence


def run(root: Path = ROOT, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    output = _checked_output(root, output)
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    boundary = _boundary(cfg)
    from src.session_data import load_development_history, load_development_oof
    df = load_development_history(root)
    _assert_development_frame(df, boundary)
    oof_path = root / "outputs/predictions/development_oof.csv"
    oof = load_development_oof(root)
    manifest_path = root / "outputs/logs/development_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    expected = {(int(fold["horizon"]), int(fold["fold"])): fold for fold in manifest["folds"]}
    if len(expected) != len(cfg["horizons"]) * int(cfg["split"]["n_folds"]):
        raise AssertionError("Original development fold manifest is incomplete")
    origins = df.index[df.index < boundary]
    all_rows, fold_rows, parity_rows = [], [], []
    for horizon in cfg["horizons"]:
        horizon = int(horizon)
        candidates = _fixed_candidates(selection, horizon)
        for fold_id in range(int(cfg["split"]["n_folds"])):
            baseline = _base_fold(df, origins, horizon, fold_id, cfg, boundary)
            original = oof.loc[(oof.horizon == horizon) & (oof.fold == fold_id) &
                               (oof.model == candidates[0])].sort_values("origin")
            if original.empty:
                raise AssertionError(f"Original OOF candidate missing: h={horizon}, fold={fold_id}")
            original_score = pd.DatetimeIndex(original.origin)
            for quantile in (REFERENCE_QUANTILE, *SENSITIVITY_QUANTILES):
                context = _at_tau(df, baseline, cfg, quantile, expected[(horizon, fold_id)], original_score)
                fold_rows.append(context["summary"])
                if quantile == REFERENCE_QUANTILE:
                    parity_rows.extend(_assert_q95_parity(_score_fold(df, context, cfg, candidates),
                                                         oof, horizon, fold_id))
                    for name in candidates:
                        part = oof.loc[(oof.horizon == horizon) & (oof.fold == fold_id) &
                                       (oof.model == name)].sort_values("origin").copy()
                        if not pd.DatetimeIndex(part.origin).equals(original_score):
                            raise AssertionError(f"Original OOF baseline grid mismatch: {name}")
                        part["tau_quantile"] = quantile
                        all_rows.append(part)
                else:
                    all_rows.extend(_score_fold(df, context, cfg, candidates))
            print(f"[sensitivity] h={horizon} fold={fold_id} candidates={','.join(candidates)}", flush=True)
    predictions = pd.concat(all_rows, ignore_index=True)
    if (predictions.origin >= boundary).any() or (predictions.target_time >= boundary).any():
        raise AssertionError("Sensitivity prediction crossed the frozen-test boundary")
    metrics = _metrics(predictions, seed=int(cfg["seed"]))
    metrics["reference_only"] = True
    metrics["selection_used"] = False
    rank_rows, evidence = [], {}
    for horizon in cfg["horizons"]:
        horizon = int(horizon)
        candidates = _fixed_candidates(selection, horizon)
        winners = {}
        for quantile in (REFERENCE_QUANTILE, *SENSITIVITY_QUANTILES):
            group = predictions.loc[(predictions.horizon == horizon) &
                                    (predictions.tau_quantile == quantile)]
            measured = metrics.loc[(metrics.horizon == horizon) &
                                   (metrics.tau_quantile == quantile)]
            winner, path = _reference_ranking(group, measured, candidates, int(cfg["seed"]))
            winners[quantile] = winner
            evidence[f"h{horizon}_q{quantile}"] = path
        for quantile in (REFERENCE_QUANTILE, *SENSITIVITY_QUANTILES):
            measured = metrics.loc[(metrics.horizon == horizon) &
                                   (metrics.tau_quantile == quantile)].sort_values(["peak_mae", "model"])
            for rank, (_, row) in enumerate(measured.iterrows(), 1):
                rank_rows.append({"horizon": horizon, "tau_quantile": quantile,
                                  "metric_peak_mae_rank": rank, "model": row.model,
                                  "peak_mae": row.peak_mae, "reference_rule_winner": winners[quantile],
                                  "reference_q95_winner": winners[REFERENCE_QUANTILE],
                                  "original_selected": selection["by_horizon"][str(horizon)]["point_model"],
                                  "reference_winner_changed_vs_q95": winners[quantile] != winners[REFERENCE_QUANTILE],
                                  "reference_winner_changed_vs_original": winners[quantile] != selection["by_horizon"][str(horizon)]["point_model"],
                                  "reference_only": True, "selection_used": False})
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    products = {"fold_grid": output / "tables/peak_sensitivity_0924_fold_grid.csv",
                "metrics": output / "tables/peak_sensitivity_0924_metrics.csv",
                "ranking": output / "tables/peak_sensitivity_0924_ranking.csv"}
    fold_grid = pd.DataFrame(fold_rows)
    fold_grid["reference_only"] = True
    fold_grid["selection_used"] = False
    fold_grid.to_csv(products["fold_grid"], index=False)
    metrics.to_csv(products["metrics"], index=False)
    pd.DataFrame(rank_rows).to_csv(products["ranking"], index=False)
    summary = {"scope": "development_oof_only", "test_boundary_exclusive": str(boundary),
               "data_max_timestamp": str(df.index.max()), "original_selection_unchanged": True,
               "sensitivity_quantiles": list(SENSITIVITY_QUANTILES),
               "reference_quantile": REFERENCE_QUANTILE, "bootstrap_date_blocks": N_BOOT,
               "candidate_scope": "original selected point model and original persistence, seasonal, CBL representatives only",
               "reference_ranking_scope": "sequential original CI rule on this fixed candidate subset; not a full reselection",
               "episode_ci_method": "original fold/day episode split and zero-F1 empty-denominator convention; one-fold-per-target-date asserted",
               "retraining": "only original h4 selected LightGBM, fixed fold parameters and weight 2; no retuning",
               "reference_only": True, "selection_used": False,
               "q95_refit_parity": {"models_checked": len(parity_rows),
                                    "max_abs_difference": max(value for row in parity_rows for key, value in row.items()
                                                              if key.endswith("_max_abs_difference")),
                                    "all_alerts_exact": all(row["alert_exact"] for row in parity_rows),
                                    "details": parity_rows},
               "reference_evidence": evidence,
               "original_selection": {str(h): selection["by_horizon"][str(h)]["point_model"] for h in cfg["horizons"]},
               "reference_winners": {f"h{h}_q{q}": next(row["reference_rule_winner"] for row in rank_rows
                                          if row["horizon"] == h and row["tau_quantile"] == q)
                                     for h in cfg["horizons"] for q in (REFERENCE_QUANTILE, *SENSITIVITY_QUANTILES)},
               "sources_sha256": {"development_oof": _digest(oof_path),
                                  "development_selection": _digest(manifest_path)},
               "products_sha256": {name: _digest(path) for name, path in products.items()}}
    log = output / "logs/peak_sensitivity_0924.json"
    log.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Results directory under outputs/ or _validation/")
    arguments = parser.parse_args()
    result = run(ROOT, output=arguments.output)
    print(json.dumps({"status": "complete", "reference_winners": result["reference_winners"]},
                     ensure_ascii=False, indent=2))
