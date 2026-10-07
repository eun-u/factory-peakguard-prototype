"""그림 생성. 한글 폰트 설정은 이 파일의 setup_font 한 곳에서만 한다."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

KOREAN_FONTS = ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "UnDotum")
COLORS = {"actual": "#222222", "lgb": "#1f6fb2", "persistence": "#d97a1e", "seasonal": "#7f7f7f",
          "threshold": "#b22222", "accent": "#2a9d8f"}


def setup_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    warnings.warn("한글 폰트를 찾지 못했습니다. 그림의 한글이 깨질 수 있습니다: " + ", ".join(KOREAN_FONTS))
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    return None


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def series_overview(series: pd.DataFrame, peak: float, boundaries: dict, path) -> None:
    daily = series["power"].resample("D").agg(["max", "mean"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(daily.index, daily["max"], color=COLORS["lgb"], lw=1, label="일 최대 15분 전력")
    ax.plot(daily.index, daily["mean"], color=COLORS["seasonal"], lw=1, label="일 평균")
    ax.axhline(peak, color=COLORS["threshold"], ls="--", lw=.9, label=f"피크 경계(학습 상위 5%) {peak:.0f}")
    for name, when in boundaries.items():
        ax.axvline(when, color="black", lw=.7, ls=":")
        ax.text(when, ax.get_ylim()[1], f" {name}", va="top", fontsize=8)
    ax.set_ylabel("전력(원자료 단위)")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    _save(fig, path)


def calendar_profiles(profiles: dict, positions: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    heat = positions.pivot_table(index="weekday", columns="hour", values="power", aggfunc="mean")
    im = axes[0].imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(len(heat.index)), ["월", "화", "수", "목", "금", "토", "일"][:len(heat.index)])
    axes[0].set_xticks(range(0, 24, 2), range(0, 24, 2))
    axes[0].set_xlabel("시각(구간 시작)")
    axes[0].set_title("요일 × 시각 평균 전력")
    fig.colorbar(im, ax=axes[0], fraction=.04)
    month = profiles["month"]
    axes[1].bar(month["month"], month["mean"], color=COLORS["seasonal"], label="평균")
    axes[1].plot(month["month"], month["p95"], color=COLORS["threshold"], marker="o", label="95% 분위")
    axes[1].set_xlabel("월")
    axes[1].set_title("월별 전력 분포")
    axes[1].legend(fontsize=8)
    _save(fig, path)


def model_comparison(table: pd.DataFrame, path) -> None:
    """table: model, metric, value, ci_low, ci_high (피크 위치 MAE, 에피소드 F1)."""
    metrics = [("peak_position_mae", "피크 위치 MAE (낮을수록 좋음)"), ("episode_f1", "피크 에피소드 F1 (높을수록 좋음)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (metric, title) in zip(axes, metrics):
        part = table[table["metric"] == metric]
        ypos = np.arange(len(part))
        err = np.vstack([part["value"] - part["ci_low"], part["ci_high"] - part["value"]])
        ax.errorbar(part["value"], ypos, xerr=err, fmt="o", color=COLORS["lgb"], capsize=4)
        ax.set_yticks(ypos, part["label"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.3)
    _save(fig, path)


def forecast_week(frame: pd.DataFrame, peak: float, path, days: int = 7) -> None:
    sample = frame.iloc[:min(96 * days, len(frame))]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(sample["target_time"], sample["y"], color=COLORS["actual"], lw=1, label="실제")
    ax.plot(sample["target_time"], sample["persistence"], color=COLORS["persistence"], lw=.9, label="persistence(직전 15분)")
    ax.plot(sample["target_time"], sample["lgb"], color=COLORS["lgb"], lw=1, label="LightGBM")
    ax.axhline(peak, color=COLORS["threshold"], ls="--", lw=.8, label="피크 경계")
    ax.set_ylabel("전력(원자료 단위)")
    ax.legend(fontsize=8, ncol=4)
    _save(fig, path)


def reliability(frame: pd.DataFrame, peak: float, path) -> None:
    observed = frame["y"].to_numpy() > peak
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    labels = {"classifier": "분류기(방식 B)", "quantile": "분위수 보간(방식 A)", "climate": "기후평균"}
    for name, label in labels.items():
        p = frame[f"{name}_prob"].to_numpy()
        edges = np.linspace(0, 1, 11)
        bins = np.digitize(p, edges[1:-1])
        pts = [(p[bins == i].mean(), observed[bins == i].mean()) for i in range(10) if (bins == i).any()]
        ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o", label=label)
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=.8)
    ax.set_xlabel("예측 초과확률")
    ax.set_ylabel("실제 초과 빈도")
    ax.legend(fontsize=8)
    _save(fig, path)


def horizon(curve: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = {"persistence": "persistence", "day": "전일 같은 시각", "week": "전주 같은 시각", "lgb": "LightGBM"}
    for ax, suffix, title in ((axes[0], "_mae", "전체 MAE"), (axes[1], "_peak_mae", "피크 위치 MAE")):
        for name, label in labels.items():
            part = curve[curve["metric"] == name + suffix]
            ax.errorbar(part["horizon_minutes"], part["value"],
                        yerr=[part["value"] - part["ci_low"], part["ci_high"] - part["value"]],
                        marker="o", capsize=3, label=label)
        ax.set_xscale("log")
        ticks = sorted(curve["horizon_minutes"].unique())
        ax.set_xticks(ticks, [f"{t // 60}시간" if t >= 60 else f"{t}분" for t in ticks])
        ax.set_xlabel("예측거리")
        ax.set_title(title + " (개발 폴드 3개 통합)", fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    _save(fig, path)


def conditions_by_hour(conditions: pd.DataFrame, path) -> None:
    part = conditions[conditions["factor"] == "hour"].copy()
    part["hour"] = part["value"].astype(int)
    part = part.sort_values("hour")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    has = part["events"] > 0
    axes[0].errorbar(part.loc[has, "hour"], part.loc[has, "recall"],
                     yerr=[part.loc[has, "recall"] - part.loc[has, "recall_ci_low"],
                           part.loc[has, "recall_ci_high"] - part.loc[has, "recall"]],
                     fmt="o", capsize=3, color=COLORS["lgb"])
    for _, r in part[has].iterrows():
        axes[0].text(r["hour"], -0.08, int(r["events"]), ha="center", fontsize=7)
    axes[0].set_ylim(-0.12, 1.05)
    axes[0].set_title("시각별 피크 재현율 (아래 숫자: 실제 피크 위치 수)", fontsize=10)
    axes[0].set_xlabel("목표 시각")
    axes[1].errorbar(part["hour"], part["fp_rate"],
                     yerr=[part["fp_rate"] - part["fp_rate_ci_low"], part["fp_rate_ci_high"] - part["fp_rate"]],
                     fmt="o", capsize=3, color=COLORS["persistence"])
    axes[1].set_title("시각별 비피크 오경보율", fontsize=10)
    axes[1].set_xlabel("목표 시각")
    for ax in axes:
        ax.axvspan(15.5, 23.5, color="grey", alpha=.12)
        ax.grid(alpha=.3)
    _save(fig, path)


def peak_heatmap(positions: pd.DataFrame, path) -> None:
    heat = positions.pivot_table(index="weekday", columns="hour", values="peak", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Reds", vmin=0)
    ax.set_yticks(range(len(heat.index)), ["월", "화", "수", "목", "금", "토", "일"][:len(heat.index)])
    ax.set_xticks(range(24), range(24))
    ax.set_xlabel("시각(구간 시작)")
    ax.set_title("시험 이전 구간의 요일 × 시각 피크 발생률", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.03)
    _save(fig, path)


def importance(table: pd.DataFrame, path) -> None:
    part = table[table["kind"] == "group"].sort_values("delta_peak_mae")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(part["unit"], part["delta_peak_mae"], xerr=part["delta_peak_mae_sd"], color=COLORS["lgb"], alpha=.8,
            label="피크 위치 MAE 증가")
    ax.barh(part["unit"], part["delta_mae"], height=.35, color=COLORS["persistence"], label="전체 MAE 증가")
    ax.set_xlabel("열을 섞었을 때 오차 증가 (원자료 단위)")
    ax.legend(fontsize=8)
    _save(fig, path)


def evening_cases(frame: pd.DataFrame, cases: list[dict], peak: float, path) -> None:
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.6 * len(cases)), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        lo, hi = case["start"] - pd.Timedelta(hours=6), case["end"] + pd.Timedelta(hours=2)
        part = frame[(frame["target_time"] >= lo) & (frame["target_time"] <= hi)]
        ax.plot(part["target_time"], part["y"], color=COLORS["actual"], lw=1, label="실제")
        ax.plot(part["target_time"], part["lgb"], color=COLORS["lgb"], lw=1, label="LightGBM(1시간 전 예측)")
        ax.plot(part["target_time"], part["persistence"], color=COLORS["persistence"], lw=.8, label="persistence")
        ax.axhline(peak, color=COLORS["threshold"], ls="--", lw=.8)
        ax.axvspan(case["start"] - pd.Timedelta(minutes=15), case["end"], color=COLORS["threshold"], alpha=.1)
        ax.set_title(f"{case['date']} 미탐 에피소드 ({case['positions']}개 위치)", fontsize=9)
    axes[0, 0].legend(fontsize=7, ncol=3)
    _save(fig, path)


def rev_curve(ratios, point, low, high, path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(ratios, point, marker="o", color=COLORS["lgb"], label="분류기 확률 vs 기후평균")
    ax.fill_between(ratios, low, high, color=COLORS["lgb"], alpha=.18)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("가정한 비용비 C/L (조치 비용 / 피크 손실)")
    ax.set_ylabel("상대 경제가치 REV")
    ax.legend(fontsize=8)
    _save(fig, path)


def shift_simulation(table: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = {"forecast": "예측 경보 기반", "forecast_q95": "예측 경보 + 95% 분위 여유", "static": "고정 시간대",
              "oracle": "사후 정보(상한)"}
    for trigger, label in labels.items():
        part = table[(table["trigger"] == trigger) & (table["slope_case"] == "point")].sort_values("fraction")
        if part.empty:
            continue
        ax.plot(part["fraction"] * 100, part["exceed_reduction_share"] * 100, marker="o", label=label)
    ax.set_xlabel("이동 비율 (%)")
    ax.set_ylabel("피크 경계 초과 위치 감소율 (%)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def alert_flow(path) -> None:
    steps = ["15분 전력·완료 생산량\n(원점 t까지)", "LightGBM\n1시간 후 점·분위수 예측", "피크 초과 판정\n주의: q95 ≥ 경계\n조치: k/n 확인",
             "운영자 검토\n이동 가능 작업 확인", "생산량 일부를\n비경보 가동 시간으로 이동", "최대 15분 전력\n재측정·기록"]
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    for i, text in enumerate(steps):
        x = i / len(steps) + .005
        ax.add_patch(plt.Rectangle((x, .2), .15, .6, fill=True, color="#eef3f8", ec=COLORS["lgb"], transform=ax.transAxes))
        ax.text(x + .075, .5, text, ha="center", va="center", fontsize=8, transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + .163, .5), xytext=(x + .152, .5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="black"))
    _save(fig, path)
