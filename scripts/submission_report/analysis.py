"""제출 보고서용 FG-R11 사후 분석: 조건별 FN·FP, 피크 발생 조건, 생산량 이동 시나리오, 그림.

    python scripts/submission_report/analysis.py

입력은 이미 확정된 결과만 읽는다. 모델을 다시 학습하거나 테스트로 무엇을 고르지 않는다.
- outputs/phase_f/final_fg_r11/FINAL_TEST_RESULT.json   (FG-R11 4차 테스트 평가 결과)
- outputs/predictions/final_test_fg_r11_h4.csv           (1시간 앞 예측·경보)
- data/raw/task05_power/okm_augumented_2021.csv           (원자료, 조건 설명용 생산량·기온)
출력: outputs/tables/submission/*.csv, outputs/figures/submission/*.png, outputs/tables/submission/summary.json
마지막에 reduction.py(일·월 최대 피크 사전 포착, 목표 최대수요 운영 시나리오)를 이어서 실행한다.
조건 분석은 탐색적이며 다중 비교 보정이 없다. 시각대·휴일·생산량은 교대·가동의 대리변수다.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_power_data  # noqa: E402
from src.holidays import calendar_flags  # noqa: E402

warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta")
TAB = ROOT / "outputs/tables/submission"
FIG = ROOT / "outputs/figures/submission"
SEED, BOOT = 42, 1000
BANDS = [(0, 6), (6, 10), (10, 12), (12, 14), (14, 18), (18, 22), (22, 24)]
BAND_LABELS = [f"{a:02d}-{b:02d}" for a, b in BANDS]
KO_FONTS = ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic")


def setup_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in KO_FONTS:
        if name in names:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def band_of(hours) -> np.ndarray:
    hours = np.asarray(hours)
    out = np.empty(len(hours), dtype=object)
    for (a, b), label in zip(BANDS, BAND_LABELS):
        out[(hours >= a) & (hours < b)] = label
    return out


def day_ci(frame: pd.DataFrame, num: str, den: str, rng) -> tuple[float, float]:
    per = frame.groupby("date")[[num, den]].sum()
    if per[den].sum() == 0:
        return (np.nan, np.nan)
    draw = rng.integers(0, len(per), size=(BOOT, len(per)))
    n = per[num].to_numpy()[draw].sum(axis=1)
    d = per[den].to_numpy()[draw].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = n / d
    vals = vals[np.isfinite(vals)]
    return tuple(np.quantile(vals, [.025, .975])) if len(vals) >= BOOT // 2 else (np.nan, np.nan)


def load_inputs():
    result = json.loads((ROOT / "outputs/phase_f/final_fg_r11/FINAL_TEST_RESULT.json").read_text(encoding="utf-8"))
    h4 = pd.read_csv(ROOT / "outputs/predictions/final_test_fg_r11_h4.csv", parse_dates=["origin", "target_time"])
    df, _ = load_power_data(ROOT / "data/raw/task05_power/okm_augumented_2021.csv")
    return result, h4, df


def h4_conditions(h4: pd.DataFrame, df: pd.DataFrame, tau: float, rng) -> tuple[pd.DataFrame, dict]:
    f = h4.copy()
    # 시각대는 목표 시각(15분 구간 종료) 기준으로, FG-R11 결과 블록과 같은 규칙이다.
    f["date"] = f["target_time"].dt.date
    f["band"] = band_of(f["target_time"].dt.hour)
    f["offday"] = np.where(calendar_flags(pd.DatetimeIndex(f["target_time"]))["is_offday"].to_numpy() == 1, "휴일·주말", "평일")
    f["production"] = df["production_target"].reindex(pd.DatetimeIndex(f["target_time"])).to_numpy()
    f["event"] = f["actual"] > tau
    f["alert"] = f["alert"].astype(bool)
    f["tp"] = f["event"] & f["alert"]
    f["fn"] = f["event"] & ~f["alert"]
    f["fp"] = ~f["event"] & f["alert"]
    f["nonevent"] = ~f["event"]
    f["ae"] = (f["actual"] - f["pred"]).abs()
    f["err"] = f["actual"] - f["pred"]
    rows = []
    groups = [("시각대", "band"), ("요일 구분", "offday"), ("생산량 구간", "production_band")]
    dev_q = df.loc[df.index < f["target_time"].min() - pd.Timedelta(hours=1), "production_target"].quantile([0, .5, .75, 1]).to_numpy()
    edges = np.unique(dev_q)
    edges[0] -= 1e-9
    edges[-1] = np.inf
    f["production_band"] = pd.cut(f["production"], edges, include_lowest=True).astype(str)
    for label, col in groups:
        for value, g in f.groupby(col, sort=True):
            recall = g["tp"].sum() / g["event"].sum() if g["event"].sum() else np.nan
            fpr = g["fp"].sum() / g["nonevent"].sum() if g["nonevent"].sum() else np.nan
            rl, rh = day_ci(g.assign(tp_=g["tp"].astype(int), ev_=g["event"].astype(int)), "tp_", "ev_", rng)
            fl, fh = day_ci(g.assign(fp_=g["fp"].astype(int), ne_=g["nonevent"].astype(int)), "fp_", "ne_", rng)
            rows.append({"factor": label, "value": str(value), "rows": len(g), "peaks": int(g["event"].sum()),
                         "tp": int(g["tp"].sum()), "fn": int(g["fn"].sum()), "fp": int(g["fp"].sum()),
                         "recall": recall, "recall_ci_low": rl, "recall_ci_high": rh,
                         "fp_rate": fpr, "fp_rate_ci_low": fl, "fp_rate_ci_high": fh,
                         "mae": g["ae"].mean(),
                         "peak_bias": g.loc[g["event"], "err"].mean() if g["event"].any() else np.nan})
    table = pd.DataFrame(rows)
    restart = f[f["date"] == pd.Timestamp("2021-08-09").date()]
    peaks_daytime = f[f["event"] & f["band"].isin(["06-10", "10-12", "12-14", "14-18"])]
    peaks_evening = f[f["event"] & (f["band"] == "18-22")]
    facts = {
        "h4_rows": len(f), "h4_peaks": int(f["event"].sum()), "h4_tp": int(f["tp"].sum()),
        "h4_fn": int(f["fn"].sum()), "h4_fp": int(f["fp"].sum()), "alert_share": float(f["alert"].mean()),
        "restart_day_2021_08_09": {"peaks": int(restart["event"].sum()), "fn": int(restart["fn"].sum()),
                                   "mean_peak_error": float(restart.loc[restart["event"], "err"].mean())
                                   if restart["event"].any() else None},
        "daytime_peak_mean_underprediction": float(peaks_daytime["err"].mean()),
        "evening_peak_mean_underprediction": float(peaks_evening["err"].mean()) if len(peaks_evening) else None,
        "evening_peaks": int(len(peaks_evening)),
        "fn_share_14_18": float(f.loc[f["band"] == "14-18", "fn"].sum() / f["fn"].sum()),
        "fp_share_weekday_06_18": float(f.loc[(f["offday"] == "평일") & f["band"].isin(["06-10", "10-12", "12-14", "14-18"]), "fp"].sum() / f["fp"].sum()),
        "peaks_in_top_production_band": float(f.loc[f["event"] & (f["production_band"] == table.loc[table["factor"] == "생산량 구간", "value"].iloc[-1]), "event"].sum() / f["event"].sum()),
    }
    return table, facts, f


def dev_peak_conditions(df: pd.DataFrame, test_start: pd.Timestamp, tau: float, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = df.loc[(df.index < test_start) & ~df["time_repaired"] & df["power"].notna()].copy()
    start = s.index - pd.Timedelta(minutes=15)
    s["date"] = s.index.date
    s["hour"] = s.index.hour          # 목표 시각(구간 종료) 기준
    s["start_hour"] = start.hour      # 생산량이 기록된 물리적 시간
    s["band"] = band_of(s.index.hour)
    s["offday"] = np.where(calendar_flags(pd.DatetimeIndex(s.index))["is_offday"].to_numpy() == 1, "휴일·주말", "평일")
    s["month"] = start.month
    s["peak"] = s["power"] > tau
    s["n"] = 1
    q_prod = np.unique(s["production_target"].quantile([0, .5, .75, .9, 1]).to_numpy())
    q_prod[0] -= 1e-9
    q_prod[-1] = np.inf
    s["production_band"] = pd.cut(s["production_target"], q_prod, include_lowest=True).astype(str)
    q_temp = np.unique(s["temperature"].quantile([0, .25, .5, .75, 1]).to_numpy())
    q_temp[0] -= 1e-9
    q_temp[-1] = np.inf
    s["temperature_band"] = pd.cut(s["temperature"], q_temp, include_lowest=True).astype(str)
    overall = s["peak"].mean()
    rows = []
    for label, col in (("월", "month"), ("시각대", "band"), ("요일 구분", "offday"),
                       ("생산량 구간", "production_band"), ("기온 구간", "temperature_band")):
        for value, g in s.groupby(col, sort=True):
            lo, hi = day_ci(g.assign(pk=g["peak"].astype(int)), "pk", "n", rng)
            rows.append({"factor": label, "value": str(value), "positions": len(g), "peaks": int(g["peak"].sum()),
                         "peak_rate": g["peak"].mean(), "ci_low": lo, "ci_high": hi,
                         "lift": g["peak"].mean() / overall, "share_of_peaks": g["peak"].sum() / s["peak"].sum()})
    # 상호작용: 평일 주간(06-18) 안에서 생산량 구간 × 기온 구간 피크율
    work = s[(s["offday"] == "평일") & s["band"].isin(["06-10", "10-12", "12-14", "14-18"])]
    inter = work.pivot_table(index="production_band", columns="temperature_band", values="peak", aggfunc="mean")
    heat = s[s["offday"] == "평일"].pivot_table(index="month", columns="hour", values="peak", aggfunc="mean")
    return pd.DataFrame(rows), inter, heat, s


def shift_scenario(df: pd.DataFrame, f: pd.DataFrame, s_dev: pd.DataFrame, tau: float, rng) -> tuple[pd.DataFrame, dict]:
    """가정 기반: 학습 기간 시간 단위 OLS(생산량 + 시각·요일 고정효과) 기울기로 이동 효과를 근사한다."""
    hourly = s_dev.groupby((s_dev.index - pd.Timedelta(minutes=15)).floor("h")).agg(power=("power", "mean"),
                                                                           production=("production_target", "first"),
                                                                           n=("power", "size"), date=("date", "first"),
                                                                           hour=("start_hour", "first"))
    hourly = hourly[hourly["n"] == 4]
    hourly["weekday"] = pd.to_datetime(hourly["date"]).dt.dayofweek

    def design(g):
        hour = pd.get_dummies(g["hour"].astype(pd.CategoricalDtype(range(24))), drop_first=True).to_numpy(float)
        wd = pd.get_dummies(g["weekday"].astype(pd.CategoricalDtype(range(7))), drop_first=True).to_numpy(float)
        return np.column_stack([np.ones(len(g)), g["production"].to_numpy(float), hour, wd])

    X, y = design(hourly), hourly["power"].to_numpy(float)
    slope = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    days = np.array(sorted(set(hourly["date"])))
    idx = {d: np.where(hourly["date"].to_numpy() == d)[0] for d in days}
    draws = []
    for _ in range(BOOT // 5):
        take = np.concatenate([idx[d] for d in rng.choice(days, len(days))])
        draws.append(np.linalg.lstsq(X[take], y[take], rcond=None)[0][1])
    slope_ci = tuple(np.quantile(draws, [.025, .975]))

    test = df.loc[(df.index >= f["target_time"].min()) & (df.index <= f["target_time"].max())
                  & ~df["time_repaired"] & df["power"].notna(), ["power", "production_target"]].copy()
    test["hour_start"] = (test.index - pd.Timedelta(minutes=15)).floor("h")
    test["date"] = test["hour_start"].dt.date
    prod = test.groupby("hour_start")["production_target"].first()
    plan_pred = pd.Series(f["pred"].to_numpy(), index=f["target_time"]).reindex(test.index)
    plan_q95 = pd.Series(f["q95"].to_numpy(), index=f["target_time"]).reindex(test.index)
    plans = {"pred": plan_pred.groupby(test["hour_start"].to_numpy()).max(),
             "q95": plan_q95.groupby(test["hour_start"].to_numpy()).max(),
             "actual": test["power"].groupby(test["hour_start"].to_numpy()).max()}
    ff = f.assign(hour_start=(f["target_time"] - pd.Timedelta(minutes=15)).dt.floor("h"))
    dev_rate = s_dev[s_dev["offday"] == "평일"].groupby("start_hour")["peak"].mean()
    static_hours = set(int(h) for h in dev_rate.index[dev_rate >= 2 * s_dev["peak"].mean()])
    triggers = {
        "예측 경보(FG-R11 h4)": (set(ff.loc[ff["alert"].astype(bool), "hour_start"]), "q95"),
        "예측 경보 + 점예측 여유": (set(ff.loc[ff["alert"].astype(bool), "hour_start"]), "pred"),
        "고정 시간대": (set(h for h in prod.index if h.hour in static_hours and h.dayofweek < 5), "pred"),
        "사후 정보(상한)": (set(test.loc[test["power"] > tau, "hour_start"]), "actual"),
    }

    def run(source_hours, plan, frac, b):
        adjusted = test["power"].to_numpy(float).copy()
        want_total = moved_total = 0.0
        for date, day in prod.groupby(prod.index.date):
            srcs = [h for h in day.index if h in source_hours and day[h] > 0]
            if not srcs:
                continue
            sinks = [h for h in day.index if h not in source_hours and day[h] > 0]
            want = sum(frac * day[h] for h in srcs)
            cap = {h: max(0.0, (0.95 * tau - plans[plan].get(h, np.inf)) / b) for h in sinks}
            cap = {h: c for h, c in cap.items() if np.isfinite(c)}
            placed = min(want, sum(cap.values()))
            want_total += want
            if placed <= 0:
                continue
            moved_total += placed
            delta = {h: -frac * day[h] * placed / want for h in srcs}
            remaining = placed
            for h in sorted(cap, key=cap.get, reverse=True):
                take = min(cap[h], remaining)
                if take <= 0:
                    break
                delta[h] = delta.get(h, 0.0) + take
                remaining -= take
            for h, d in delta.items():
                adjusted[(test["hour_start"] == h).to_numpy()] += b * d
        before = test["power"].to_numpy(float)
        daily_b = pd.Series(before).groupby(test["date"].to_numpy()).max()
        daily_a = pd.Series(adjusted).groupby(test["date"].to_numpy()).max()
        return {"exceed_before": int((before > tau).sum()), "exceed_after": int((adjusted > tau).sum()),
                "max_before": float(before.max()), "max_after": float(adjusted.max()),
                "daily_max_mean_before": float(daily_b.mean()), "daily_max_mean_after": float(daily_a.mean()),
                "days_daily_max_up": int((daily_a > daily_b + 1e-9).sum()),
                "placed_share": moved_total / want_total if want_total else np.nan}

    rows = []
    for name, (hours, plan) in triggers.items():
        for frac in (.1, .2, .3):
            for case, b in (("점추정", slope), ("CI 하한", slope_ci[0]), ("CI 상한", slope_ci[1])):
                r = run(hours, plan, frac, b)
                r.update(trigger=name, fraction=frac, slope_case=case, slope=b,
                         exceed_reduction=1 - r["exceed_after"] / r["exceed_before"])
                rows.append(r)
    info = {"slope": slope, "slope_ci": slope_ci, "hours": int(len(hourly)), "static_hours": sorted(static_hours)}
    return pd.DataFrame(rows), info


def relative_value(f: pd.DataFrame, s_dev: pd.DataFrame, tau: float, rng) -> pd.DataFrame:
    """비용-손실 상대 경제가치. p > C/L이면 조치. 기후평균 = 개발 이력의 요일·15분 위치 피크 빈도(라플라스 평활)."""
    dev = s_dev.assign(slot=s_dev.index.dayofweek * 96 + s_dev.index.hour * 4 + s_dev.index.minute // 15)
    grouped = dev.groupby("slot")["peak"].agg(["sum", "count"])
    rate = (grouped["sum"] + 1) / (grouped["count"] + 2)
    t = pd.DatetimeIndex(f["target_time"])
    slot = t.dayofweek * 96 + t.hour * 4 + t.minute // 15
    clim = pd.Series(slot).map(rate).fillna(dev["peak"].mean()).to_numpy()
    y = f["event"].to_numpy()
    prob = f["p_exceed"].to_numpy()
    groups = [np.where(f["date"].to_numpy() == d)[0] for d in pd.unique(f["date"])]

    def value(idx, r):
        yy, pp, cc = y[idx], prob[idx], clim[idx]
        model = r * np.mean(pp > r) + np.mean(yy & (pp <= r))
        base = r * np.mean(cc > r) + np.mean(yy & (cc <= r))
        perfect = r * np.mean(yy)
        return (base - model) / (base - perfect) if base - perfect > 1e-12 else np.nan

    rows = []
    all_idx = np.arange(len(f))
    for r in (.05, .1, .15, .2, .25, .3, .35, .4, .45, .5):
        draws = []
        for _ in range(BOOT):
            take = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            draws.append(value(take, r))
        draws = np.asarray(draws, dtype=float)
        draws = draws[np.isfinite(draws)]
        rows.append({"cost_loss_ratio": r, "rev": value(all_idx, r),
                     "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975)})
    return pd.DataFrame(rows)


def figures(result: dict, cond: pd.DataFrame, heat: pd.DataFrame, f: pd.DataFrame, shift: pd.DataFrame, tau: float):
    names = {"persistence": "직전값 유지", "B1 weekly naive": "1주 전 같은 시각", "analog only": "유사일",
             "FG-R6": "유사일+LightGBM", "Chronos-2 alone": "Chronos-2 단독", "TH reconciled Chronos": "+ 시간 계층 조정",
             "TH Chronos + MOS": "+ 최근 오차 보정", "FG-R11 without peak shift": "+ 동일일 복사 규칙",
             "FG-R11": "제안 모델(+ 피크 보정)"}
    p = result["point"]
    # 그림 2-1 모델 비교
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [names[k] for k in names]
    for ax, key, title in ((axes[0], "AUC_MAE", "전체 MAE (1~4시간 앞 평균)"), (axes[1], "AUC_PeakMAE", "피크 구간 MAE (실제 > 178)")):
        vals = [p[k][key] for k in names]
        colors = ["#1f6fb2" if k == "FG-R11" else "#9db7cf" for k in names]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_1_model_comparison.png", dpi=170)
    plt.close(fig)
    # 그림 2-2 지평별 MAE
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k in ("persistence", "B1 weekly naive", "analog only", "FG-R6", "Chronos-2 alone", "TH reconciled Chronos", "FG-R11"):
        ph = p[k]["per_horizon_MAE"]
        xs = sorted(int(h) for h in ph)
        ax.plot([x * 15 for x in xs], [ph[str(x)] if str(x) in ph else ph[x] for x in xs], marker="o", ms=3,
                lw=2 if k == "FG-R11" else 1, label=names[k])
    ax.set_xlabel("예측거리 (분)")
    ax.set_ylabel("평균 절대오차 (원자료 단위)")
    ax.set_xticks([60, 120, 180, 240])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_2_horizon_mae.png", dpi=170)
    plt.close(fig)
    # 그림 3-1 시각대별 재현율·오경보율
    part = cond[cond["factor"] == "시각대"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    has = part["peaks"] > 0
    axes[0].errorbar(part.loc[has, "value"], part.loc[has, "recall"],
                     yerr=[part.loc[has, "recall"] - part.loc[has, "recall_ci_low"],
                           part.loc[has, "recall_ci_high"] - part.loc[has, "recall"]], fmt="o", capsize=4, color="#1f6fb2")
    for _, r in part[has].iterrows():
        axes[0].text(r["value"], -0.1, f"n={int(r['peaks'])}", ha="center", fontsize=8)
    axes[0].set_ylim(-0.15, 1.05)
    axes[0].set_title("시각대별 피크 재현율 (1시간 앞 경보)", fontsize=10)
    axes[1].errorbar(part["value"], part["fp_rate"],
                     yerr=[part["fp_rate"] - part["fp_rate_ci_low"], part["fp_rate_ci_high"] - part["fp_rate"]],
                     fmt="o", capsize=4, color="#d97a1e")
    axes[1].set_title("시각대별 비피크 오경보율", fontsize=10)
    for ax in axes:
        ax.grid(alpha=.3)
        ax.set_xlabel("예측 대상 시각대")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_1_conditions_by_band.png", dpi=170)
    plt.close(fig)
    # 그림 3-2 개발 이력 평일 월 × 시각 피크율
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Reds", vmin=0)
    ax.set_yticks(range(len(heat.index)), [f"{m}월" for m in heat.index])
    ax.set_xticks(range(24), range(24))
    ax.set_xlabel("시각 (15분 구간이 끝나는 시각)")
    ax.set_title("개발 구간 평일의 월 × 시각 피크 비율", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_2_peak_heatmap.png", dpi=170)
    plt.close(fig)
    # 그림 2-3 테스트 첫 주 예측·구간·경보
    w = f[f["target_time"] < f["target_time"].min() + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(w["target_time"], w["q05"], w["q95"], color="#1f6fb2", alpha=.15, label="90% 예측구간")
    ax.plot(w["target_time"], w["actual"], color="#222", lw=1, label="실제")
    ax.plot(w["target_time"], w["pred"], color="#1f6fb2", lw=1, label="제안 모델 1시간 앞 예측")
    ax.axhline(tau, color="#b22222", ls="--", lw=.8, label=f"피크 기준값 {tau:.0f}")
    al = w[w["alert"]]
    ax.scatter(al["target_time"], np.full(len(al), w["actual"].max() * 1.04), marker="|", color="#b22222", s=40, label="경보")
    ax.legend(fontsize=7, ncol=5, loc="lower left")
    ax.set_ylabel("전력(원자료 단위)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_3_test_week.png", dpi=170)
    plt.close(fig)
    # 그림 4-2 이동 시나리오
    fig, ax = plt.subplots(figsize=(7, 4))
    shown = {"예측 경보(FG-R11 h4)": "예측 경보 시간", "고정 시간대": "피크가 잦은 시각에 고정 (예측 없음)",
             "사후 정보(상한)": "실제 피크 시간 (사후에 안 경우)"}
    for trig, g in shift[shift["slope_case"] == "점추정"].groupby("trigger", sort=False):
        if trig in shown:
            ax.plot(g["fraction"] * 100, g["exceed_reduction"] * 100, marker="o", label=shown[trig])
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("이동 비율 (%)")
    ax.set_ylabel("피크 구간 감소율 (%)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_4_shift_scenario.png", dpi=170)
    plt.close(fig)


def transfer_figure():
    """개발 개선이 테스트로 전이되었는가 (같은 개발·테스트 정의의 라운드·구성요소, 각 결과 문서 수치)."""
    rows = [("최근 오차 보정", .025, .385, .002, .348), ("시간 계층 조정", .038, .041, .147, .407),
            ("보정량 재조정", .020, .162, .029, -.219), ("시간 계층 통합", .099, .262, .114, .273),
            ("세부 파라미터 조절", .022, .041, -.003, -.018)]
    table = pd.DataFrame(rows, columns=["change", "dev_mae_gain", "dev_peak_gain", "test_mae_gain", "test_peak_gain"])
    table.to_csv(TAB / "dev_to_test_transfer.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, t, title in ((axes[0], "dev_mae_gain", "test_mae_gain", "전체 MAE 개선량"), (axes[1], "dev_peak_gain", "test_peak_gain", "피크 MAE 개선량")):
        ax.scatter(table[d], table[t], color="#1f6fb2")
        for _, r in table.iterrows():
            ax.annotate(r["change"], (r[d], r[t]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lim = max(abs(table[[d, t]].to_numpy()).max() * 1.2, .05)
        ax.plot([-lim, lim], [-lim, lim], color="grey", ls="--", lw=.8)
        ax.axhline(0, color="black", lw=.6)
        ax.axvline(0, color="black", lw=.6)
        ax.set_xlabel("개발 구간 개선량")
        ax.set_ylabel("평가 구간 개선량")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_4_dev_test_transfer.png", dpi=170)
    plt.close(fig)
    return table


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    setup_font()
    rng = np.random.default_rng(SEED)
    result, h4, df = load_inputs()
    tau = float(result["tau"])
    cond, facts, f = h4_conditions(h4, df, tau, rng)
    cond.to_csv(TAB / "h4_conditions.csv", index=False, encoding="utf-8-sig")
    dev_cond, inter, heat, s_dev = dev_peak_conditions(df, pd.Timestamp(result["test_first_origin"]), tau, rng)
    dev_cond.to_csv(TAB / "dev_peak_conditions.csv", index=False, encoding="utf-8-sig")
    inter.to_csv(TAB / "dev_peak_production_x_temperature.csv", encoding="utf-8-sig")
    shift, shift_info = shift_scenario(df, f, s_dev, tau, rng)
    shift.to_csv(TAB / "shift_scenario.csv", index=False, encoding="utf-8-sig")
    rev = relative_value(f, s_dev, tau, rng)
    rev.to_csv(TAB / "rev_fg_r11_h4.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(rev["cost_loss_ratio"], rev["rev"], marker="o", color="#1f6fb2", label="제안 모델 1시간 앞 피크 확률")
    ax.fill_between(rev["cost_loss_ratio"], rev["ci_low"], rev["ci_high"], color="#1f6fb2", alpha=.18, label="95% 구간")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("가정한 비용비 C/L (조치 비용 / 피크 손실)")
    ax.set_ylabel("상대 경제가치 REV")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_2_rev.png", dpi=170)
    plt.close(fig)
    transfer = transfer_figure()
    figures(result, cond, heat, f, shift, tau)
    summary = {"tau": tau, "h4": facts, "shift": shift_info,
               "dev_overall_peak_rate": float(s_dev["peak"].mean()), "dev_positions": int(len(s_dev))}
    (TAB / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    from reduction import main as reduction_main   # 최대 피크 사전 포착·목표 최대수요 운영
    reduction_main()


if __name__ == "__main__":
    main()
