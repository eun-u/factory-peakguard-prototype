"""Development-only, paired Phase C evaluation of precomputed predictions."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = tuple(range(4, 17))
MAIN = ("B0", "B1", "B2", "B3", "B4", "B5", "M1", "M1-W", "M2", "C1")
SIMPLICITY = ("B0", "B1", "B2", "B4", "B5", "B3", "M1", "M1-W", "C1", "M2")
BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
KEY = ["horizon", "fold", "origin", "target_time"]
REQUIRED = {"model", *KEY, "y", "pred", "tau", "d2", "train_seconds",
            "inference_seconds", "selected_config", "development_only"}


def _prepare(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    data = predictions.copy()
    if data.empty or not data.development_only.eq(True).all():
        raise ValueError("Phase C evaluation requires development_only=True on every row")
    if not data.model.isin((*MAIN, "R1")).all():
        raise ValueError("Unknown model in Phase C predictions")
    if not data.horizon.isin(HORIZONS).all() or not data.fold.isin((0, 1, 2)).all():
        raise ValueError("Expected horizons 4..16 and fold IDs 0..2")
    for column in ("origin", "target_time"):
        data[column] = pd.to_datetime(data[column], errors="raise")
        if data[column].isna().any() or data[column].ge(BOUNDARY).any():
            raise ValueError(f"{column} reaches the sealed boundary")
    if not (data.target_time > data.origin).all():
        raise ValueError("Targets must follow forecast origins")
    if not (data.target_time == data.origin + pd.to_timedelta(data.horizon * 15, unit="m")).all():
        raise ValueError("Target time does not match direct horizon")
    if data.duplicated(["model", *KEY]).any():
        raise ValueError("Duplicate model/horizon/fold/origin/target key")
    if data.d2.isna().any() or not data.d2.isin((True, False)).all():
        raise ValueError("d2 must be a nonmissing boolean")
    if data.selected_config.isna().any():
        raise ValueError("selected_config must contain JSON for every model")
    try:
        for value in data.selected_config.unique():
            json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("selected_config must be a JSON string") from exc
    for column in ("y", "pred", "tau", "train_seconds", "inference_seconds"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(data.tau).all() or not np.isfinite(data.train_seconds).all() or not np.isfinite(data.inference_seconds).all():
        raise ValueError("tau and timing must be finite")
    if (data[["train_seconds", "inference_seconds"]] < 0).any().any():
        raise ValueError("Timing cannot be negative")
    main = data[data.model.isin(MAIN)].copy()
    present = main.groupby(["model", "horizon", "fold"], observed=True).size()
    expected = pd.MultiIndex.from_product([MAIN, HORIZONS, (0, 1, 2)], names=present.index.names)
    if not expected.isin(present.index).all():
        raise ValueError("All ten main models, thirteen horizons and three folds are required")
    for column in ("y", "tau", "d2"):
        if main.groupby(KEY, sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent {column} on a paired key")
    if main.groupby(["horizon", "fold"], sort=False).tau.nunique().gt(1).any():
        raise ValueError("Fold/horizon tau is inconsistent")
    for column in ("train_seconds", "inference_seconds", "selected_config"):
        if data.groupby(["model", "horizon", "fold"], sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent {column} within one fitted model")
    finite = main[np.isfinite(main.y) & np.isfinite(main.pred)]
    widths = finite.groupby(KEY, sort=False).model.nunique()
    common_keys = widths[widths.eq(len(MAIN))].index
    if len(common_keys) == 0:
        raise ValueError("No finite paired MAIN10 intersection")
    paired = finite.set_index(KEY).loc[common_keys].reset_index()
    if not paired.groupby(KEY, sort=False).model.nunique().eq(len(MAIN)).all():
        raise AssertionError("The final paired cohort differs among MAIN10 models")
    counts = paired.groupby(["horizon", "fold"], sort=False).model.size().div(len(MAIN))
    required_groups = pd.MultiIndex.from_product([HORIZONS, (0, 1, 2)], names=["horizon", "fold"])
    if not required_groups.isin(counts.index).all():
        raise ValueError("A horizon/fold has no finite paired MAIN10 observations")
    reference = data[data.model.eq("R1") & np.isfinite(data.y) & np.isfinite(data.pred)]
    if not reference.empty:
        reference = reference.set_index(KEY).loc[lambda frame: frame.index.isin(common_keys)].reset_index()
        truth = paired[paired.model.eq("B0")][[*KEY, "y", "tau", "d2"]]
        checked = reference.merge(truth, on=KEY, how="left", suffixes=("", "_main"), validate="one_to_one")
        for column in ("y", "tau", "d2"):
            if not checked[column].eq(checked[f"{column}_main"]).all():
                raise ValueError(f"R1 {column} conflicts with the main paired cohort")
        paired = pd.concat([paired, reference], ignore_index=True)
    raw_count = main.groupby("model").size().to_dict()
    finite_count = finite.groupby("model").size().to_dict()
    paired_count = paired[paired.model.isin(MAIN)].groupby("model").size().to_dict()
    union_keys = main[KEY].drop_duplicates()
    union_count = len(union_keys)
    removal = {model: {"input": int(raw_count.get(model, 0)),
                       "union_keys": union_count,
                       "raw_missing": int(union_count - raw_count.get(model, 0)),
                       "nonfinite": int(raw_count.get(model, 0) - finite_count.get(model, 0)),
                       "removed_for_pairing": int(finite_count.get(model, 0) - paired_count.get(model, 0)),
                       "common_finite": int(paired_count.get(model, 0)),
                       "removed": int(raw_count.get(model, 0) - paired_count.get(model, 0))}
               for model in MAIN}
    return paired, removal


def _metrics(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for names, part in frame.groupby(group, sort=True, observed=True):
        if not isinstance(names, tuple):
            names = (names,)
        error = part.pred.to_numpy(float) - part.y.to_numpy(float)
        peak = part.y.to_numpy(float) > part.tau.to_numpy(float)
        run_groups = part.drop_duplicates(["model", "horizon", "fold"])
        configs = sorted(set(part.selected_config.astype(str)))
        taus = part.groupby("fold", sort=True).tau.first().to_dict()
        rows.append({**dict(zip(group, names)), "n": len(part), "peak_n": int(peak.sum()),
                     "MAE": float(np.mean(np.abs(error))),
                     "Peak_MAE": float(np.mean(np.abs(error[peak]))) if peak.any() else np.nan,
                     "RMSE": float(np.sqrt(np.mean(error ** 2))),
                     "train_seconds": float(run_groups.train_seconds.sum()),
                     "inference_seconds": float(run_groups.inference_seconds.sum()),
                     "selected_config": configs[0] if len(configs) == 1 else json.dumps(configs),
                     "score_start": part.target_time.min(), "score_end": part.target_time.max(),
                     "tau": float(part.tau.iloc[0]) if len(taus) == 1 else np.nan,
                     "tau_by_fold": json.dumps({str(k): float(v) for k, v in taus.items()}, sort_keys=True),
                     "development_only": True})
    return pd.DataFrame(rows)


def _bootstrap(frame: pd.DataFrame, n: int, seed: int) -> tuple[dict, dict]:
    """Resample target calendar days jointly across models and horizons."""
    models = list(MAIN)
    dates = pd.Index(sorted(frame.target_time.dt.normalize().unique()))
    d, h, m = len(dates), len(HORIZONS), len(models)
    date_index = dates.get_indexer(frame.target_time.dt.normalize())
    horizon_index = frame.horizon.to_numpy(int) - HORIZONS[0]
    model_index = pd.Index(models).get_indexer(frame.model)
    if (model_index < 0).any():
        raise AssertionError("Reference model reached main bootstrap")
    shape = (d, h, m)
    absolute_sum = np.zeros(shape)
    count = np.zeros(shape)
    peak_sum = np.zeros(shape)
    peak_count = np.zeros(shape)
    err = np.abs(frame.pred.to_numpy(float) - frame.y.to_numpy(float))
    is_peak = frame.y.to_numpy(float) > frame.tau.to_numpy(float)
    index = (date_index, horizon_index, model_index)
    np.add.at(absolute_sum, index, err)
    np.add.at(count, index, 1)
    np.add.at(peak_sum, index, err * is_peak)
    np.add.at(peak_count, index, is_peak.astype(float))
    draws = np.random.default_rng(seed).multinomial(d, np.full(d, 1 / d), size=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        mae = np.einsum("bd,dhm->bhm", draws, absolute_sum) / np.einsum("bd,dhm->bhm", draws, count)
        peak = np.einsum("bd,dhm->bhm", draws, peak_sum) / np.einsum("bd,dhm->bhm", draws, peak_count)
    auc_mae = np.mean(mae, axis=1)
    auc_peak = np.mean(peak, axis=1)
    point_mae = np.sum(absolute_sum, axis=0) / np.sum(count, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        point_peak = np.sum(peak_sum, axis=0) / np.sum(peak_count, axis=0)
    points = {"MAE": point_mae, "Peak_MAE": point_peak,
              "AUC_MAE": np.mean(point_mae, axis=0), "AUC_PeakMAE": np.mean(point_peak, axis=0)}
    samples = {"MAE": mae, "Peak_MAE": peak, "AUC_MAE": auc_mae, "AUC_PeakMAE": auc_peak}
    return points, samples


def _difference(points: dict, samples: dict, metric: str, a: str, b: str,
                horizon: int | None = None) -> tuple[float, float, float, str, str, int]:
    ia, ib = MAIN.index(a), MAIN.index(b)
    if horizon is None:
        value = points[metric][ia] - points[metric][ib]
        draws = samples[metric][:, ia] - samples[metric][:, ib]
    else:
        j = HORIZONS.index(horizon)
        value = points[metric][j, ia] - points[metric][j, ib]
        draws = samples[metric][:, j, ia] - samples[metric][:, j, ib]
    valid = draws[np.isfinite(draws)]
    if not np.isfinite(value) or len(valid) != len(draws):
        reason = "nonfinite_point_estimate" if not np.isfinite(value) else "nonfinite_bootstrap_draws"
        return float(value), np.nan, np.nan, "unavailable", reason, len(valid)
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(value), float(lo), float(hi), "available", "", len(valid)


def _curve(path: Path, metrics: pd.DataFrame, column: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, part in metrics.groupby("model", sort=False):
        ax.plot(part.horizon.to_numpy() * 15, part[column], label=model, marker="o", markersize=2)
    ax.set(xlabel="Horizon (minutes)", ylabel=column, title=title)
    ax.grid(alpha=.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(predictions: pd.DataFrame, output_dir: str | Path, *, bootstrap_n: int = 1000,
             seed: int = 42) -> dict:
    """Evaluate all precomputed direct forecasts and select one model family.

    The input is never trained or extended here. Every comparison uses the same
    finite MAIN10 key intersection; optional R1 cannot change that intersection.
    """
    if bootstrap_n < 1:
        raise ValueError("bootstrap_n must be positive")
    paired, removal = _prepare(predictions)
    root = Path(output_dir)
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    datasets = {"D1": paired, "D2": paired[paired.d2]}
    fold_tables, pooled_tables, auc_rows = [], [], []
    bootstrap_results = {}
    for name, sample in datasets.items():
        if sample.empty:
            continue
        fold = _metrics(sample, ["model", "horizon", "fold"])
        pooled = _metrics(sample, ["model", "horizon"])
        fold.insert(0, "dataset", name)
        pooled.insert(0, "dataset", name)
        fold_tables.append(fold)
        pooled_tables.append(pooled)
        main_sample = sample[sample.model.isin(MAIN)]
        if not main_sample.empty:
            bootstrap_results[name] = _bootstrap(main_sample, bootstrap_n, seed)
        for model, part in pooled.groupby("model", sort=False):
            all_h = set(part.horizon) == set(HORIZONS)
            auc_rows.append({"dataset": name, "model": model,
                             "AUC_MAE": float(part.MAE.mean()) if all_h else np.nan,
                             "AUC_PeakMAE": float(part.Peak_MAE.mean()) if all_h and part.Peak_MAE.notna().all() else np.nan,
                             "n_horizons": len(part), "definition": "equally spaced horizon-normalized mean error",
                             "development_only": True})
    fold_table = pd.concat(fold_tables, ignore_index=True)
    pooled_table = pd.concat(pooled_tables, ignore_index=True)
    auc = pd.DataFrame(auc_rows)
    if not set(MAIN).issubset(set(auc.loc[auc.dataset.eq("D1"), "model"])):
        raise AssertionError("Main model missing after paired intersection")
    baseline = min(("B1", "B2", "B3"), key=lambda model: float(auc.loc[(auc.dataset.eq("D1")) & (auc.model.eq(model)), "AUC_MAE"].iloc[0]))
    pair_rows = []
    for dataset, (points, samples) in bootstrap_results.items():
        for a, b in combinations(MAIN, 2):
            for metric, auc_metric in (("MAE", "AUC_MAE"), ("Peak_MAE", "AUC_PeakMAE")):
                for horizon in (*HORIZONS, None):
                    kind = metric if horizon is not None else auc_metric
                    value, lo, hi, status, reason, valid_draw_count = _difference(points, samples, kind, a, b, horizon)
                    pair_rows.append({"dataset": dataset, "model_a": a, "model_b": b,
                                      "metric": kind, "horizon": horizon, "difference_a_minus_b": value,
                                      "ci_low": lo, "ci_high": hi, "ci_status": status,
                                      "ci_reason": reason, "valid_draw_count": valid_draw_count,
                                      "bootstrap_n": bootstrap_n, "block": "target_calendar_day_joint_across_horizons",
                                      "development_only": True})
    pairwise = pd.DataFrame(pair_rows)

    d1_points, d1_samples = bootstrap_results["D1"]
    d2_result = bootstrap_results.get("D2")
    d1_auc = auc[auc.dataset.eq("D1")].set_index("model")
    d2_auc = auc[auc.dataset.eq("D2")].set_index("model") if "D2" in set(auc.dataset) else pd.DataFrame()
    fold_auc = fold_table[fold_table.dataset.eq("D1")].groupby(["model", "fold"]).MAE.mean()
    stability_rows = []
    d1_rank = d1_auc.loc[list(MAIN), "AUC_MAE"].rank(method="min")
    d2_rank = d2_auc.loc[list(MAIN), "AUC_MAE"].rank(method="min") if set(MAIN).issubset(d2_auc.index) and d2_auc.loc[list(MAIN), "AUC_MAE"].notna().all() else None
    weekly = pooled_table[(pooled_table.dataset.eq("D1")) & (pooled_table.model.eq("B1"))].set_index("horizon").MAE
    for model in MAIN:
        fold_diffs = [float(fold_auc[baseline, fold] - fold_auc[model, fold]) for fold in (0, 1, 2)]
        curve = pooled_table[(pooled_table.dataset.eq("D1")) & (pooled_table.model.eq(model))].set_index("horizon").MAE
        skills = 1 - curve / weekly
        base_d1 = float(d1_auc.loc[model, "AUC_MAE"] - d1_auc.loc[baseline, "AUC_MAE"])
        base_d2 = (float(d2_auc.loc[model, "AUC_MAE"] - d2_auc.loc[baseline, "AUC_MAE"])
                   if d2_rank is not None else np.nan)
        stability_rows.append({"model": model, "baseline": baseline,
                               "fold_0_improvement": fold_diffs[0], "fold_1_improvement": fold_diffs[1],
                               "fold_2_improvement": fold_diffs[2], "positive_fold_count": sum(v > 0 for v in fold_diffs),
                               "worst_horizon_weekly_skill": float(skills.min()),
                               "d1_rank": int(d1_rank[model]), "d2_rank": int(d2_rank[model]) if d2_rank is not None else np.nan,
                               "d2_rank_reversal": bool(np.sign(base_d1) != np.sign(base_d2)) if np.isfinite(base_d2) else pd.NA,
                               "development_only": True})
    stability = pd.DataFrame(stability_rows)

    decisions = []
    for model in MAIN:
        peak_value, peak_lo, peak_hi, peak_status, peak_reason, peak_valid_draws = _difference(d1_points, d1_samples, "AUC_PeakMAE", model, baseline)
        excluded_peak = peak_status == "available" and peak_lo > 0
        c1_reasons = []
        if model == "C1":
            _, _, c1_hi, c1_status, _, _ = _difference(d1_points, d1_samples, "AUC_MAE", "C1", "M1")
            if c1_status != "available" or not c1_hi < 0:
                c1_reasons.append("no_clear_auc_mae_gain_vs_M1")
            if sum(float(fold_auc["M1", fold] - fold_auc["C1", fold]) > 0 for fold in (0, 1, 2)) < 2:
                c1_reasons.append("fewer_than_two_positive_folds")
            if d2_rank is None or not np.isfinite(d2_auc.loc["C1", "AUC_MAE"]) or not np.isfinite(d2_auc.loc["M1", "AUC_MAE"]):
                c1_reasons.append("D2_unavailable")
            elif d2_auc.loc["C1", "AUC_MAE"] > d2_auc.loc["M1", "AUC_MAE"]:
                c1_reasons.append("D2_worse_than_M1")
        decisions.append({"model": model, "AUC_MAE": float(d1_auc.loc[model, "AUC_MAE"]),
                          "AUC_PeakMAE": float(d1_auc.loc[model, "AUC_PeakMAE"]),
                          "strongest_baseline": baseline, "peak_difference_vs_baseline": peak_value,
                          "peak_ci_low": peak_lo, "peak_ci_high": peak_hi, "peak_ci_status": peak_status,
                          "peak_ci_reason": peak_reason, "peak_valid_draw_count": peak_valid_draws,
                          "peak_safeguard_excluded": excluded_peak, "C1_rejection_reasons": ";".join(c1_reasons),
                          "eligible": not excluded_peak and not c1_reasons,
                          "complexity_rank": SIMPLICITY.index(model) + 1, "development_only": True})
    decision = pd.DataFrame(decisions)
    insufficient_peak_evidence = decision.peak_ci_status.ne("available").any()
    if insufficient_peak_evidence:
        decision["eligible"] = False
    eligible = decision.loc[decision.eligible].sort_values("AUC_MAE")
    if insufficient_peak_evidence or eligible.empty:
        selected = None
        primary = None
        tied = []
    else:
        primary = str(eligible.iloc[0].model)
        tied = [primary]
        for model in eligible.model.iloc[1:]:
            _, lo, hi, status, _, _ = _difference(d1_points, d1_samples, "AUC_MAE", model, primary)
            if status == "available" and lo <= 0 <= hi:
                tied.append(model)
        selected = min(tied, key=SIMPLICITY.index)
    decision["primary_min_auc_model"] = primary
    decision["statistical_tie_with_primary"] = decision.model.isin(tied)
    decision["selected_M_star"] = decision.model.eq(selected) if selected is not None else False
    selection_status = ("insufficient_peak_evidence" if insufficient_peak_evidence else
                        "selected" if selected is not None else "unavailable")
    decision["selection_status"] = selection_status

    paths = {}
    artifacts = {
        "model_horizon_fold_metrics.csv": fold_table,
        "model_horizon_pooled_metrics.csv": pooled_table,
        "model_pairwise_ci.csv": pairwise,
        "d2_novel_profile_metrics.csv": pooled_table[pooled_table.dataset.eq("D2")].copy(),
        "model_auc_summary.csv": auc,
        "model_stability.csv": stability,
        "model_selection_decision.csv": decision,
    }
    for name, frame in artifacts.items():
        target = tables / name
        frame.to_csv(target, index=False)
        paths[name] = str(target)
    d1_pooled = pooled_table[pooled_table.dataset.eq("D1")]
    for name, column, title in (("mae_horizon_curve.png", "MAE", "Development MAE by horizon"),
                                ("peak_mae_horizon_curve.png", "Peak_MAE", "Development peak MAE by horizon")):
        target = figures / name
        _curve(target, d1_pooled, column, title)
        paths[name] = str(target)
    skill = d1_pooled.copy()
    skill["skill"] = skill.apply(lambda row: 1 - row.MAE / weekly.loc[row.horizon], axis=1)
    _curve(figures / "skill_horizon_curve.png", skill, "skill", "Skill relative to weekly baseline")
    paths["skill_horizon_curve.png"] = str(figures / "skill_horizon_curve.png")
    comparison = auc[auc.model.isin(MAIN)].copy()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for dataset, part in comparison.groupby("dataset", sort=False):
        ax.scatter(part.model, part.AUC_MAE, label=dataset)
    ax.set(xlabel="Model", ylabel="AUC-MAE", title="D1 and novel-profile D2")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / "d1_d2_comparison.png", dpi=150)
    plt.close(fig)
    paths["d1_d2_comparison.png"] = str(figures / "d1_d2_comparison.png")
    return {"selected_M_star": selected, "primary_min_auc_model": primary,
            "strongest_baseline": baseline, "statistical_tie_models": tied,
            "removed_counts": removal, "D2_available": d2_rank is not None,
            "peak_safeguard_available": bool(decision.peak_ci_status.eq("available").all()),
            "selection_status": selection_status,
            "development_only": True, "auc_definition": "equally spaced horizon-normalized mean error",
            "paths": paths}
