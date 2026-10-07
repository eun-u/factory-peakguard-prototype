"""Run preregistered Phase 1 A1–A3 on sealed development history only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.predictability import analyze_a1, analyze_a2, analyze_a3
from src.session_data import SEALED_BOUNDARY, load_development_history, load_development_oof
from src.viz import configure, save


def _source_frames(root: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[tuple[int, int], dict]]:
    history = load_development_history(root)
    oof = load_development_oof(root)
    if history.index.max() >= SEALED_BOUNDARY or oof.target_time.ge(SEALED_BOUNDARY).any():
        raise AssertionError("Phase 1 source escaped development boundary")
    frozen = json.loads((root / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    meta = {(int(row["horizon"]), int(row["fold"])): row for row in frozen["folds"]}
    by_horizon = {}
    for horizon in (4, 16, 96):
        name = frozen["selection"]["by_horizon"][str(horizon)]["point_model"]
        score = oof.loc[oof.horizon.eq(horizon) & oof.model.eq(name)].copy()
        if score.empty or score.duplicated(["fold", "target_time"]).any():
            raise AssertionError(f"h{horizon} selected original OOF is empty or duplicate")
        for fid, block in score.groupby("fold"):
            required = meta[(horizon, int(fid))]
            if (len(block) != int(required["n_score"])
                    or pd.Timestamp(block.origin.min()) != pd.Timestamp(required["score_start"])
                    or pd.Timestamp(block.origin.max()) != pd.Timestamp(required["score_end"])):
                raise AssertionError(f"h{horizon} fold{fid} differs from sealed score grid")
        if score.fold.nunique() != 3:
            raise AssertionError("Expected original three-fold development OOF")
        by_horizon[horizon] = score
    return history, by_horizon, meta


def _save_tables(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for key, frame in tables.items():
        frame.to_csv(output / f"{key}.csv", index=False, encoding="utf-8-sig")


def _plot_a1(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    starts = tables["A1_peak_starts"]
    if starts.empty:
        return
    configure()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    counts = starts.groupby("slot").size().reindex(range(96), fill_value=0)
    axes[0].bar(np.arange(96) / 4, counts.to_numpy(), width=.23, color="#3E6283")
    axes[0].set(xlabel="시각(시)", ylabel="시작 에피소드(건)", xlim=(0, 24),
                title="A1. 피크 에피소드 시작 시각(15분 슬롯)")
    matrix = starts.groupby(["is_offday", "slot"]).size().unstack(fill_value=0).reindex(
        index=[False, True], columns=range(96), fill_value=0)
    heatmap = axes[1].imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    axes[1].set(yticks=[0, 1], yticklabels=["평일", "휴일"],
                xticks=np.arange(0, 96, 8), xticklabels=np.arange(0, 24, 2),
                xlabel="시각(시)", title="A1. 요일 유형별 피크 시작 횟수")
    fig.colorbar(heatmap, ax=axes[1], label="시작 횟수(건)", fraction=.025)
    save(fig, output / "A1_peak_slot_distribution.png")


def _plot_a2(output: Path, hypotheses: pd.DataFrame) -> None:
    rows = hypotheses.loc[hypotheses.analysis_id.eq("A2")].copy()
    if rows.empty:
        return
    configure()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    labels = [f"h{r.horizon} {r.feature}" for r in rows.itertuples(index=False)]
    y = np.arange(len(rows))
    x = pd.to_numeric(rows.estimate, errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(rows.ci_low, errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(rows.ci_high, errors="coerce").to_numpy(dtype=float)
    for i in range(len(rows)):
        if np.isfinite(x[i]):
            ax.plot(x[i], y[i], "o", color="#3E6283")
            if np.isfinite(low[i]) and np.isfinite(high[i]):
                ax.plot([low[i], high[i]], [y[i], y[i]], color="#3E6283", linewidth=2)
    ax.axvline(.5, color="#7D8894", linestyle="--")
    ax.axvline(.6, color="#B8741A", linestyle=":")
    if not np.isfinite(x).any():
        ax.text(.5, .5, "사전 정의 슬롯의 피크/비피크 비교 표본 부족",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "#C9D2DB", "pad": 8})
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="날짜 블록 OOF AUC와 95% 신뢰구간",
           title="A2. 같은 재가동 슬롯의 피크일 구분력")
    ax.invert_yaxis()
    save(fig, output / "A2_auc_by_horizon.png")


def _plot_a3(output: Path, table: pd.DataFrame) -> None:
    if table.empty:
        return
    configure()
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    y = np.arange(len(table))
    for i, row in enumerate(table.itertuples(index=False)):
        if np.isfinite(row.estimate):
            ax.plot(row.estimate, i, "o", color="#2B7564" if row.analysis_role == "pre_origin" else "#B8741A")
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                ax.plot([row.ci_low, row.ci_high], [i, i], color="#2B7564" if row.analysis_role == "pre_origin" else "#B8741A", linewidth=2)
    ax.axvline(0, color="#7D8894", linestyle="--")
    if not pd.to_numeric(table.estimate, errors="coerce").notna().any():
        ax.text(.5, .5, "사전 정의 슬롯에서 유효 피크 초과 사례 없음",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "#C9D2DB", "pad": 8})
    ax.set(yticks=y, yticklabels=table.feature, xlabel="표준화 OLS 계수와 날짜 블록 95% 신뢰구간",
           title="A3. 피크 초과량 관련성(사후 변수는 해석 전용)")
    ax.invert_yaxis()
    save(fig, output / "A3_peak_excess_coefficients.png")


def _episode_origin_coverage(history: pd.DataFrame, starts: pd.DataFrame,
                             oof: dict[int, pd.DataFrame],
                             a2_slots: pd.DataFrame) -> pd.DataFrame:
    """Descriptive availability for every original h4 episode at h16/h96."""
    rows = []
    for event in starts.itertuples(index=False):
        for horizon in (16, 96):
            origin = pd.Timestamp(event.start) - pd.Timedelta(minutes=15 * horizon)
            matched = oof[horizon].loc[pd.to_datetime(oof[horizon].target_time).eq(event.start)]
            if len(matched) > 1:
                raise AssertionError("Horizon OOF target time is duplicated")
            horizon_fold = int(matched.fold.iloc[0]) if len(matched) else np.nan
            on_grid = origin in history.index and origin < SEALED_BOUNDARY
            clean_power = bool(on_grid and pd.notna(history.loc[origin, "power"])
                               and not bool(history.loc[origin, "time_repaired"]))
            selected = (bool(a2_slots.loc[a2_slots.horizon.eq(horizon)
                                        & a2_slots.fold.eq(horizon_fold)
                                        & a2_slots.slot.eq(event.slot), "selected"].any())
                        if len(matched) else False)
            rows.append({"h4_fold": int(event.fold), "peak_start": event.start,
                         "peak_slot": int(event.slot), "horizon": horizon,
                         "origin": origin, "origin_on_history_grid": bool(on_grid),
                         "origin_has_clean_power": clean_power,
                         "target_in_original_horizon_score": bool(len(matched)),
                         "horizon_score_fold": horizon_fold,
                         "slot_selected_from_horizon_fold_fit": selected,
                         "analysis_role": "descriptive_coverage_only"})
    return pd.DataFrame(rows)


def run_predictability_p1(root: Path, output: Path, *, through: str = "A3") -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if through not in ("A1", "A2", "A3"):
        raise ValueError("through must be A1, A2, or A3")
    history, oof, metadata = _source_frames(root)
    summary: dict = {"scope": "development_oof_only", "sealed_boundary_exclusive": str(SEALED_BOUNDARY),
                     "through": through}
    hypotheses = []
    a1_tables, a1 = analyze_a1(history, oof[4], {fid: metadata[(4, fid)] for fid in range(3)})
    _save_tables(output, a1_tables)
    _plot_a1(output, a1_tables)
    summary["A1"] = a1
    hypotheses.append(a1["hypothesis"])
    print(f"[P1] A1 complete: peak starts={a1['n_peak_starts']}", flush=True)
    if through in ("A2", "A3"):
        a2_tables, a2, _ = analyze_a2(history, {16: oof[16], 96: oof[96]}, metadata)
        _save_tables(output, a2_tables)
        episode_coverage = _episode_origin_coverage(history, a1_tables["A1_peak_starts"],
                                                    oof, a2_tables["A2_restart_slots"])
        episode_coverage.to_csv(output / "A2_episode_origin_coverage.csv", index=False,
                                encoding="utf-8-sig")
        summary["A2"] = a2
        summary["A2"]["episode_origin_coverage"] = {
            "total_rows": len(episode_coverage),
            "origin_grid_available": int(episode_coverage.origin_on_history_grid.sum()),
            "origin_clean_power_available": int(episode_coverage.origin_has_clean_power.sum()),
            "original_horizon_score_target_present": int(episode_coverage.target_in_original_horizon_score.sum()),
            "train_selected_slot": int(episode_coverage.slot_selected_from_horizon_fold_fit.sum()),
            "analysis_role": "descriptive_only_not_a_gate"}
        hypotheses.extend(a2["hypotheses"])
        _plot_a2(output, pd.DataFrame(hypotheses))
        print(f"[P1] A2 complete: {len(a2['hypotheses'])} fixed tests", flush=True)
    if through == "A3":
        a3_tables, a3 = analyze_a3(history, a2_tables, metadata)
        _save_tables(output, a3_tables)
        summary["A3"] = a3
        hypotheses.extend(a3["hypotheses"])
        _plot_a3(output, a3_tables["A3_coefficients"])
        print(f"[P1] A3 complete: {len(a3['hypotheses'])} fixed tests", flush=True)
    pd.DataFrame(hypotheses).to_csv(output / "A1_A3_hypotheses.csv", index=False, encoding="utf-8-sig")
    (output / "A1_A3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                         default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/analysis_p1")
    parser.add_argument("--through", choices=("A1", "A2", "A3"), default="A3")
    args = parser.parse_args()
    run_predictability_p1(args.root, args.output, through=args.through)
