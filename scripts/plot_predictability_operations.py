"""Plot fixed Phase 1 error and operational scenario summaries."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.viz import configure, plt, save


def run(output=None):
    out = Path(output) if output else ROOT / "outputs/analysis_p1"
    configure()
    errors = pd.read_csv(out / "A4_hypotheses.csv")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    pos = np.arange(len(errors))
    ax.errorbar(errors.estimate, pos,
                xerr=np.vstack([errors.estimate-errors.ci_low, errors.ci_high-errors.estimate]),
                fmt="o", color="#3E6283", capsize=5)
    ax.axvline(0, color="#8995a1", linestyle="--")
    ax.set_yticks(pos, [f"h{int(r.horizon)} · 재가동 {int(r.restart_rows)}위치" for _, r in errors.iterrows()])
    ax.set_xlabel("재가동 - 비재가동의 (LightGBM - CBL) MAE 차이 · 95% CI")
    ax.set_title("재가동 구간의 상대 오차 집중: 양의 효과 미입증")
    fig.tight_layout()
    save(fig, out / "A4_error_concentration.png")

    scenarios = pd.read_csv(out / "A5_summary.csv")
    scenarios = scenarios.loc[scenarios.scenario.eq("D")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, prep in zip(axes, [30, 60]):
        part = scenarios.loc[scenarios.prep_minutes.eq(prep)]
        labels, values = [], []
        for horizon in [4, 16]:
            for source in ["original_operational", "perfect_information"]:
                found = part.loc[part.horizon.eq(horizon) & part.alarm_source.eq(source)]
                if found.empty:
                    # Oracle label is a schema name, not inferred from its outcome.
                    found = part.loc[part.horizon.eq(horizon) & ~part.alarm_source.eq("original_operational")] if source == "perfect_information" else found
                if len(found) != 1:
                    raise ValueError("Expected one fixed D scenario per horizon/alarm/preparation")
                r = found.iloc[0]
                labels.append(f"h{horizon}\n{'기존 경보' if source == 'original_operational' else '완벽정보 참고'}")
                values.append(float(r.applied_actual_peak_source_fraction) * 100)
        bars = ax.bar(np.arange(4), values, color=["#3E6283", "#8CA9C2", "#2B7564", "#90BDB0"])
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3)
        ax.set_xticks(np.arange(4), labels)
        ax.set_title(f"준비 {prep}분 · 지연 이동 20%")
        ax.set_ylim(0, 110)
    axes[0].set_ylabel("실제 피크 출발 시간 중 조치된 비율 (%)")
    fig.suptitle("사후 관측 생산량 기반 가정 · 완벽정보는 전역 최적 상한이 아님", fontsize=11)
    fig.tight_layout()
    save(fig, out / "A5_actionable_peak_source_hours.png")


if __name__ == "__main__":
    run()
