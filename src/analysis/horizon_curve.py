"""Development-only horizon performance curve from matched OOF rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import configure, save
from ._common import write_table
from .errors import match_episode_table


def horizon_metrics(predictions: pd.DataFrame, selected_models: dict[int, str] | None = None,
                    n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for (horizon, model), group in predictions.groupby(["horizon", "model"]):
        group = group.sort_values("target_time").copy()
        if group.duplicated("target_time").any():
            continue
        actual_peak = group.y > group.tau
        peaks = group.loc[actual_peak].copy()
        if peaks.empty:
            peak_mae = low = high = np.nan
        else:
            peaks["abs_error"] = (peaks.y-peaks.pred).abs()
            daily = peaks.groupby(peaks.target_time.dt.normalize()).abs_error.agg(["sum", "count"])
            peak_mae = float(daily["sum"].sum()/daily["count"].sum())
            if len(daily) >= 2:
                rng = np.random.default_rng(seed)
                ids = rng.integers(0, len(daily), size=(n_boot, len(daily)))
                sums, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
                low, high = map(float, np.quantile(sums[ids].sum(axis=1)/counts[ids].sum(axis=1), [.025, .975]))
            else:
                low = high = np.nan
        group["alert_flag"] = pd.to_numeric(group.get("alert", group.pred > group.tau), errors="coerce").astype(bool)
        events = match_episode_table(group)
        tp = int(events.status.eq("TP").sum())
        fp = int(events.status.eq("FP").sum())
        fn = int(events.status.eq("FN").sum())
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
        rows.append({"horizon": int(horizon), "minutes": int(horizon*15), "model": model,
                     "selected": bool(selected_models and selected_models.get(int(horizon)) == model),
                     "n": len(group), "peak_n": int(actual_peak.sum()),
                     "mae": float((group.y-group.pred).abs().mean()),
                     "peak_mae": peak_mae, "peak_mae_ci_low": low, "peak_mae_ci_high": high,
                     "episode_f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def plot_horizon_curve(table: pd.DataFrame, path: Path,
                       representatives: dict[int, dict] | None = None) -> Path | None:
    if table.empty:
        return None
    configure()
    families = {"Persistence": r"^p[123]_", "Seasonal": r"^s[123]_", "CBL": r"^c[123]a?_"}
    series = []
    for horizon, group in table.groupby("horizon"):
        point = group.loc[group.selected & group.model.astype(str).str.startswith("lgbm")]
        if point.empty:
            point = group.loc[group.model.eq("lgbm_no_holiday")]
        if not point.empty:
            role = "selected" if bool(point.selected.iloc[0]) else "baseline_winner_comparator"
            series.append(point.iloc[[0]].assign(curve="LightGBM", comparison_role=role))
        for label, pattern in families.items():
            candidates = group.loc[group.model.astype(str).str.match(pattern)]
            if not candidates.empty:
                if representatives is not None:
                    name = representatives.get(int(horizon), {}).get(label.lower())
                    candidates = candidates.loc[candidates.model.eq(name)]
                else:
                    candidates = candidates.sort_values("mae").iloc[[0]]
                if not candidates.empty:
                    series.append(candidates.iloc[[0]].assign(curve=label))
    if not series:
        return None
    keep = pd.concat(series)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharex=True)
    for model, frame in keep.groupby("curve"):
        frame = frame.sort_values("minutes")
        axes[0].plot(frame.minutes, frame.peak_mae, marker="o", label=str(model))
        axes[1].plot(frame.minutes, frame.episode_f1, marker="o", label=str(model))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sorted(table.minutes.unique()))
        ax.set_xticklabels([f"{m//60}시간" if m >= 60 else f"{m}분" for m in sorted(table.minutes.unique())])
        ax.set_xlabel("예측거리")
        ax.grid(True)
    axes[0].set_ylabel("피크 위치 MAE (원자료 단위)")
    axes[0].set_title("피크 위치 오차")
    axes[1].set_ylabel("에피소드 F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("피크 에피소드 적중")
    axes[0].legend(fontsize=7, ncol=2)
    if keep.get("comparison_role", pd.Series(dtype=str)).eq("baseline_winner_comparator").any():
        fig.text(.5, .01, "기준선이 선택된 예측거리의 LightGBM은 공휴일 제외 개발 비교 모델",
                 ha="center", fontsize=8, color="#B8741A")
    fig.suptitle("예측거리별 성능 · 개발 교차검증")
    fig.tight_layout(rect=(0, .04, 1, .93))
    return save(fig, path)


def paired_horizon_advantage(predictions: pd.DataFrame, selected_models: dict[int, str] | None,
                             n_boot: int, seed: int, representatives: dict[int, dict] | None = None) -> pd.DataFrame:
    """Paired day-block MAE gain over each baseline family at each horizon."""
    families = {"persistence": r"^p[123]_", "seasonal": r"^s[123]_", "cbl": r"^c[123]a?_"}
    rows = []
    keys = ["target_time", "fold", "horizon"]
    for horizon, group in predictions.groupby("horizon"):
        chosen = (selected_models or {}).get(int(horizon), "lgbm")
        model = chosen if str(chosen).startswith("lgbm") else "lgbm_no_holiday"
        comparison_role = "selected_point_model" if model == chosen else "development_comparator_not_selected"
        point = group.loc[group.model.eq(model), keys+["y", "pred", "tau"]].rename(columns={"pred": "model_pred"})
        if point.empty:
            continue
        for family, pattern in families.items():
            candidates = group.loc[group.model.astype(str).str.match(pattern)]
            if candidates.empty:
                continue
            if representatives is not None:
                representative = representatives.get(int(horizon), {}).get(family)
                if representative not in set(candidates.model):
                    continue
            else:
                representative = (candidates.assign(abs_error=lambda x: (x.y-x.pred).abs())
                                  .groupby("model").abs_error.mean().idxmin())
            base = candidates.loc[candidates.model.eq(representative), keys+["pred"]].rename(columns={"pred": "base_pred"})
            pair = point.merge(base, on=keys, how="inner", validate="one_to_one")
            pair = pair.loc[pair.y.gt(pair.tau) & pair.model_pred.notna() & pair.base_pred.notna()].copy()
            if pair.empty:
                continue
            pair["gain"] = (pair.y-pair.base_pred).abs()-(pair.y-pair.model_pred).abs()
            daily = pair.groupby(pair.target_time.dt.normalize()).gain.agg(["sum", "count"])
            gain = float(daily["sum"].sum()/daily["count"].sum())
            if len(daily) >= 2:
                rng = np.random.default_rng(seed)
                ids = rng.integers(0, len(daily), size=(n_boot, len(daily)))
                samples = daily["sum"].to_numpy()[ids].sum(axis=1)/daily["count"].to_numpy()[ids].sum(axis=1)
                low, high = map(float, np.quantile(samples, [.025, .975]))
            else:
                low = high = np.nan
            rows.append({"horizon": int(horizon), "minutes": int(horizon*15), "model": model,
                         "selected_model": chosen, "comparison_role": comparison_role,
                         "baseline_family": family, "baseline_model": representative,
                         "paired_peak_n": len(pair), "peak_mae_gain": gain,
                         "ci_low": low, "ci_high": high,
                         "model_beats_baseline_ci": bool(np.isfinite(low) and low > 0)})
    return pd.DataFrame(rows)


def run_horizon_curve(predictions: pd.DataFrame, outdir: Path, cfg: dict,
                      selected_models: dict[int, str] | None = None,
                      representatives: dict[int, dict] | None = None) -> dict:
    n_boot, seed = int(cfg.get("bootstrap", {}).get("n", 1000)), int(cfg.get("seed", 42))
    table = horizon_metrics(predictions, selected_models, n_boot=n_boot, seed=seed)
    advantage = paired_horizon_advantage(predictions, selected_models, n_boot, seed, representatives)
    path = write_table(table, outdir/"tables"/"horizon_curve.csv")
    advantage_path = write_table(advantage, outdir/"tables"/"horizon_advantage.csv")
    fig = plot_horizon_curve(table, outdir/"figures"/"horizon_curve.png", representatives)
    selected_gain = (advantage.loc[advantage.comparison_role.eq("selected_point_model")]
                     if not advantage.empty else advantage)
    comparator_gain = (advantage.loc[advantage.comparison_role.eq("development_comparator_not_selected")]
                       if not advantage.empty else advantage)
    def furthest(frame: pd.DataFrame) -> dict:
        return {family: int(group.loc[group.model_beats_baseline_ci, "minutes"].max())
                for family, group in frame.groupby("baseline_family")
                if group.model_beats_baseline_ci.any()} if not frame.empty else {}
    maximum = furthest(selected_gain)
    return {"status": "ok", "paths": {"table": str(path), "advantage": str(advantage_path),
                                     "figure": str(fig) if fig else None},
            "maximum_horizon_minutes_beating_baseline_ci": maximum,
            "development_comparator_maximum_horizon_minutes_beating_baseline_ci": furthest(comparator_gain)}
