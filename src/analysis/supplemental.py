"""Probability reliability and the forecast-to-operator decision figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from scipy.stats import spearmanr

from src.viz import configure, save
from ._common import block_ci, write_table
from .errors import episodes


def reliability_table(pred: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42,
                      n_bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Fixed probability bins, with day-block uncertainty on observed rates."""
    if n_bins < 1 or n_boot < 1:
        raise ValueError("n_bins and n_boot must be positive")
    if "p_exceed" not in pred:
        return pd.DataFrame(), {"status": "unsupported", "reason": "No exceedance probabilities"}
    x = pred[["target_time", "y", "tau", "p_exceed"]].copy()
    x["p_exceed"] = pd.to_numeric(x.p_exceed, errors="coerce")
    x = x.loc[x.p_exceed.notna() & x.y.notna() & x.tau.notna()].copy()
    if x.empty:
        return pd.DataFrame(), {"status": "unsupported", "reason": "No finite probability and target pairs"}
    if x.p_exceed.lt(0).any() or x.p_exceed.gt(1).any():
        raise ValueError("Exceedance probability outside [0, 1]")
    x["event"] = x.y.gt(x.tau).astype(int)
    edges = np.linspace(0, 1, n_bins+1)
    x["bin"] = np.clip(np.searchsorted(edges, x.p_exceed, side="right")-1, 0, n_bins-1)
    x["date"] = pd.to_datetime(x.target_time).dt.normalize()
    rng = np.random.default_rng(seed)
    rows = []
    for bin_index in range(n_bins):
        group = x.loc[x["bin"].eq(bin_index)]
        if group.empty:
            continue
        daily = group.groupby("date").event.agg(["sum", "count"])
        if len(daily) >= 2:
            ids = rng.integers(0, len(daily), size=(n_boot, len(daily)))
            sampled = daily["sum"].to_numpy()[ids].sum(axis=1)/daily["count"].to_numpy()[ids].sum(axis=1)
            low, high = map(float, np.quantile(sampled, [.025, .975]))
        else:
            low = high = np.nan
        rows.append({"bin": bin_index, "bin_left": float(edges[bin_index]),
                     "bin_right": float(edges[bin_index+1]), "n": len(group),
                     "mean_predicted": float(group.p_exceed.mean()),
                     "events": int(group.event.sum()), "observed_rate": float(group.event.mean()),
                     "observed_rate_ci_low": low, "observed_rate_ci_high": high,
                     "distinct_days": len(daily)})
    y = x.event.to_numpy(int)
    p = x.p_exceed.to_numpy(float)
    summary = {"status": "ok", "n": len(x), "events": int(y.sum()),
               "event_rate": float(y.mean()), "brier": float(brier_score_loss(y, p)),
               "pr_auc": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
               "binning": f"fixed_equal_width_{n_bins}", "ci_method": "target_date_block_bootstrap"}
    return pd.DataFrame(rows), summary


def plot_reliability(table: pd.DataFrame, summary: dict, path: Path) -> Path | None:
    if table.empty:
        return None
    configure()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7D8894", label="완전 보정")
    sizes = 30+140*np.sqrt(table.n/table.n.max())
    ax.scatter(table.mean_predicted, table.observed_rate, s=sizes, color="#3E6283", zorder=3)
    ax.plot(table.mean_predicted, table.observed_rate, color="#3E6283", alpha=.6)
    valid = table.observed_rate_ci_low.notna()
    if valid.any():
        lower = np.maximum(0, table.loc[valid, "observed_rate"]-table.loc[valid, "observed_rate_ci_low"])
        upper = np.maximum(0, table.loc[valid, "observed_rate_ci_high"]-table.loc[valid, "observed_rate"])
        ax.errorbar(table.loc[valid, "mean_predicted"], table.loc[valid, "observed_rate"],
                    yerr=[lower, upper],
                    fmt="none", ecolor="#3E6283", alpha=.55, capsize=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="평균 예측 피크 확률", ylabel="실제 피크 비율",
           title=f"피크 확률 신뢰도 · Brier {summary['brier']:.3f}")
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, path)


def plot_decision_flow(path: Path) -> Path:
    """Static figure for F4-1; the output names the operator's decision gate."""
    configure()
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, detail: str,
            color: str = "#DCE6EF") -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor=color, edgecolor="#3E6283", linewidth=1.1))
        ax.text(x+w/2, y+h*.65, title, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x+w/2, y+h*.29, detail, ha="center", va="center", fontsize=8.5)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "color": "#4F5F70", "lw": 1.6})

    box(.2, 1.53, 1.7, 1.0, "관측 입력", "과거 전력·완료 생산·달력")
    box(2.25, 1.53, 1.7, 1.0, "직접 예측", "4시간·1시간 후 15분 전력")
    box(4.3, 1.53, 1.7, 1.0, "위험 산출", "피크 확률·보정 상단·여유")
    box(6.35, 2.45, 1.55, .95, "4시간 전 주의", "생산 일정 검토", "#F6E7CF")
    box(6.35, .65, 1.55, .95, "1시간 전 조치", "연속 확인 경보", "#F6E7CF")
    box(8.3, 1.53, 1.5, 1.0, "운영자 판단", "준비시간·미래 이동창 확인", "#D8EDE7")
    arrow(1.95, 2.03, 2.2, 2.03)
    arrow(4.0, 2.03, 4.25, 2.03)
    arrow(6.05, 2.1, 6.3, 2.85)
    arrow(6.05, 1.95, 6.3, 1.13)
    arrow(7.95, 2.85, 8.25, 2.1)
    arrow(7.95, 1.13, 8.25, 1.95)
    ax.text(5.4, .22, "조치 가능 시각이 지나면 실행 불가로 집계 · 설비 제어 자동화 없음",
            ha="center", fontsize=9, color="#4F5F70")
    fig.tight_layout()
    return save(fig, path)


def expected_exceedance_episodes(pred: pd.DataFrame, *, n_boot: int = 1000,
                                 seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Descriptive comparison on observed peak episodes only.

    The max of pointwise expected excess is not an expected episode maximum;
    its signed error is an operational proxy, not probability calibration.
    """
    if "exp_exceed" not in pred:
        return pd.DataFrame(), {"status": "unsupported", "reason": "No adopted expected-excess output"}
    rows = []
    for fold, group in pred.groupby("fold", sort=False):
        group = group.sort_values("target_time")
        for start, end in episodes(group.target_time, group.y.gt(group.tau)):
            block = group.loc[group.target_time.between(start, end)]
            actual = float((block.y-block.tau).max())
            predicted = pd.to_numeric(block.exp_exceed, errors="coerce")
            usable = predicted.notna().all() and np.isfinite(predicted.to_numpy(float)).all()
            rows.append({"fold": fold, "target_time": start, "episode_end": end,
                         "intervals": len(block), "actual_max_excess": actual,
                         "predicted_max_pointwise_excess": float(predicted.max()) if usable else np.nan,
                         "signed_error": float(predicted.max()-actual) if usable else np.nan,
                         "absolute_error": float(abs(predicted.max()-actual)) if usable else np.nan,
                         "status": "evaluated" if usable else "missing_prediction"})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {"status": "no_peak_episodes", "episodes": 0}
    valid = detail.loc[detail.status.eq("evaluated")].copy()
    if valid.empty:
        return detail, {"status": "no_evaluable_episodes", "episodes": len(detail)}
    bias, low, high = block_ci(valid, lambda s: s.signed_error.mean(), n=n_boot, seed=seed)
    mae, mae_low, mae_high = block_ci(valid, lambda s: s.absolute_error.mean(), n=n_boot, seed=seed)
    rank = (float(spearmanr(valid.actual_max_excess, valid.predicted_max_pointwise_excess).statistic)
            if len(valid) >= 3 and valid.actual_max_excess.nunique() > 1
            and valid.predicted_max_pointwise_excess.nunique() > 1 else np.nan)
    summary = {"status": "ok", "episodes": len(detail), "evaluated_episodes": len(valid),
               "mean_signed_error": bias, "mean_signed_error_ci95": [low, high],
               "mae": mae, "mae_ci95": [mae_low, mae_high], "spearman": rank,
               "interpretation": "Observed peak episodes only; maximum pointwise expected excess is a proxy, not calibrated episode-maximum expectation."}
    return detail, summary


def run_supplemental(pred: pd.DataFrame, outdir: Path, cfg: dict, scope: str,
                     *, expected_exceedance_adopted: bool = False) -> dict:
    outdir = Path(outdir)
    table, stats = reliability_table(pred, n_boot=int(cfg.get("bootstrap", {}).get("n", 1000)),
                                     seed=int(cfg.get("seed", 42)))
    result = {"reliability": stats, "scope": scope}
    if not table.empty:
        result["reliability_table"] = str(write_table(table, outdir/"tables"/"probability_reliability.csv"))
        fig = plot_reliability(table, stats, outdir/"figures"/"F2_probability_reliability.png")
        result["reliability_figure"] = str(fig) if fig else None
    if expected_exceedance_adopted:
        detail, excess = expected_exceedance_episodes(
            pred, n_boot=int(cfg.get("bootstrap", {}).get("n", 1000)), seed=int(cfg.get("seed", 42)))
        result["expected_exceedance"] = excess
        if not detail.empty:
            result["expected_exceedance_episodes"] = str(write_table(
                detail, outdir/"tables"/"expected_exceedance_episodes.csv"))
            result["expected_exceedance_summary"] = str(write_table(
                pd.DataFrame([excess]), outdir/"tables"/"expected_exceedance_summary.csv"))
    else:
        result["expected_exceedance"] = {"status": "not_adopted"}
    result["decision_flow_figure"] = str(plot_decision_flow(outdir/"figures"/"F4-1_forecast_alert_action.png"))
    return result
