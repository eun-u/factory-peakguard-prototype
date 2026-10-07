"""최대 피크 사전 포착과 피크 저감 시나리오 (보고서 2·4장).

    python scripts/submission_report/reduction.py

1) 일·월 최대 구간 사전 포착: 시험 구간 근무일의 일 최대 15분 구간과 월 최대 구간에 대해,
   1~4시간 앞 예측 시점 중 언제부터 경보가 켜졌는지(선행시간)를 센다. 경보 임계는 시험 전에 잠근 값이다.
2) 목표 최대수요 운영: 미리 정한 목표값 D를 넘는 15분 구간에서, 넘는 만큼(현재 부하의 c 이하)을 최대 1시간
   뒤로 미루고 D 아래 여유가 있는 구간에 다시 넣는다(1시간 안에 넣지 못하면 여유와 관계없이 넣는다).
   상한 제어는 구간 안의 실시간 계측으로 수행하고, 예측은 제어를 준비할 시간대(준비 시간)를 정하는 데 쓴다.
   준비 방식 세 가지를 같은 날짜·같은 규칙으로 비교한다.
   - 하루 내내 준비: 모든 시간 (효과의 상한, 준비 부담 최대)
   - 예측 경보로 준비: 1~4시간 앞 예측 중 하나라도 경보가 켜진 구간
   - 기동 시간대 고정 준비: 근무일 기동·재가동 직후 시간대(예측 미사용)
   미룰 수 있는 부하 비율 c는 가정이며 민감도로 보고한다.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import FIG, TAB, setup_font  # noqa: E402
from src.holidays import calendar_flags  # noqa: E402

MAX_DELAY = 4          # 미룬 부하는 최대 1시간 안에 다시 넣는다
# 개발 구간 평일 평균 전력에서 기동·재가동 직후 부하가 오르는 구간(구간 종료 시각 기준)
START_WINDOWS = [("07:45", "09:00"), ("10:30", "10:45"), ("13:00", "13:45"), ("15:30", "15:30"), ("17:45", "18:00")]


def working_mask(times: pd.DatetimeIndex) -> np.ndarray:
    return calendar_flags(times)["is_offday"].to_numpy() == 0


def peak_capture(full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """일 최대·월 최대 구간이 몇 분 전부터 경보되었는가. 선행시간 = 경보 시각부터 해당 15분 구간 시작까지."""
    truth = full.drop_duplicates("target_time").set_index("target_time")["actual"].sort_index()
    truth = truth[truth.index.normalize() < truth.index.max().normalize()]          # 마지막 부분 날짜 제외
    days = truth.groupby(truth.index.date)
    rows = []
    for date, s in days:
        when = s.idxmax()
        if not working_mask(pd.DatetimeIndex([when]))[0]:
            continue
        g = full[full["target_time"] == when]
        alerted = g.loc[g["alert"], "horizon"]
        lead = int((alerted.max() - 1) * 15) if len(alerted) else None
        rows.append({"date": str(date), "time": str(when), "max": float(s.max()),
                     "alert_h4": bool(g.loc[g["horizon"] == 4, "alert"].any()),
                     "alert_any": bool(len(alerted)), "earliest_lead_min": lead,
                     "alert_horizons": int(len(alerted)), "horizons": int(len(g))})
    table = pd.DataFrame(rows)
    month_rows = []
    for month, s in truth.groupby(truth.index.month):
        when = s.idxmax()
        g = full[full["target_time"] == when]
        alerted = g.loc[g["alert"], "horizon"]
        month_rows.append({"month": int(month), "time": str(when), "max": float(s.max()),
                           "alert_h4": bool(g.loc[g["horizon"] == 4, "alert"].any()),
                           "earliest_lead_min": int((alerted.max() - 1) * 15) if len(alerted) else None,
                           "alert_horizons": int(len(alerted)), "horizons": int(len(g))})
    top = truth.sort_values(ascending=False).head(20)
    top_alert = [bool(full[(full["target_time"] == t) & (full["horizon"] == 4)]["alert"].any()) for t in top.index]
    leads = table["earliest_lead_min"].dropna()
    facts = {"working_days": int(len(table)),
             "daily_max_alert_h4": float(table["alert_h4"].mean()),
             "daily_max_alert_any": float(table["alert_any"].mean()),
             "daily_max_lead_median": float(leads.median()) if len(leads) else None,
             "daily_max_lead_ge_120": float((table["earliest_lead_min"].fillna(-1) >= 120).mean()),
             "daily_max_lead_ge_180": float((table["earliest_lead_min"].fillna(-1) >= 180).mean()),
             "monthly": month_rows,
             "top20_alert_h4": float(np.mean(top_alert)),
             "daily_max_time_counts": table["time"].str[11:16].value_counts().to_dict()}
    curve = pd.DataFrame({"lead_min": list(range(45, 226, 15))})
    curve["captured_share"] = [(table["earliest_lead_min"].fillna(-1) >= m).mean() for m in curve["lead_min"]]
    return table, facts, curve


def demand_cap(actual: np.ndarray, armed: np.ndarray, D: float, c: float) -> dict:
    out = actual.astype(float).copy()
    queue: deque = deque()
    cut_total, cut_index, cut_amounts, forced = 0.0, [], [], 0
    for t in range(len(out)):
        for item in queue:
            item[1] += 1
        while queue and queue[0][1] > MAX_DELAY:      # 1시간이 지난 부하는 여유와 관계없이 다시 넣는다
            out[t] += queue.popleft()[0]
            forced += 1
        room = max(0.0, D - out[t])                    # 실시간 계측으로 본 여유
        while queue and room > 1e-9:
            take = min(room, queue[0][0])
            out[t] += take
            room -= take
            queue[0][0] -= take
            if queue[0][0] <= 1e-9:
                queue.popleft()
        if armed[t] and out[t] > D:                    # 준비된 구간에서만 상한 제어
            cut = min(out[t] - D, c * actual[t])
            out[t] -= cut
            queue.append([cut, 0])
            cut_total += cut
            cut_index.append(t)
            cut_amounts.append(cut)
    while queue:
        out[-1] += queue.popleft()[0]
    return {"out": out, "cut_total": cut_total, "cut_index": cut_index, "cut_amounts": np.array(cut_amounts), "forced": forced}


def run_scenarios(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = full.drop_duplicates("target_time").set_index("target_time")["actual"].sort_index()
    times = truth.index
    actual = truth.to_numpy(float)
    alert_any = full.groupby("target_time")["alert"].any().reindex(times).fillna(False).to_numpy()
    slot = times.strftime("%H:%M")
    work = working_mask(times)
    start = np.zeros(len(times), dtype=bool)
    for a, b in START_WINDOWS:
        start |= (slot >= a) & (slot <= b)
    masks = {"하루 내내 준비": np.ones(len(times), dtype=bool),
             "예측 경보로 준비": alert_any,
             "기동 시간대 고정 준비": start & work}
    per_day = pd.Series(1, index=times).groupby(times.date).sum()
    whole = np.isin(times.date, per_day.index[per_day == 96])     # 시작·끝의 일부만 있는 날은 준비 시간 계산에서 제외
    n_work_days = len(np.unique(times[work & whole].date))
    months = times.month
    base_month = {m: actual[months == m].max() for m in np.unique(months)}
    total_energy = actual.sum()
    rows, where = [], []
    for D in (210, 205, 200, 195):
        for c in (0.05, 0.10, 0.15):
            for name, armed in masks.items():
                r = demand_cap(actual, armed, D, c)
                out = r["out"]
                cut_idx = np.array(r["cut_index"], dtype=int)
                exceed = actual > D
                rows.append({
                    "target_D": D, "deferrable_share": c, "arming": name,
                    "max_before": float(actual.max()), "max_after": float(out.max()),
                    "max_reduction": float(1 - out.max() / actual.max()),
                    "aug_max_after": float(out[months == 8].max()), "sep_max_after": float(out[months == 9].max()),
                    "aug_reduction": float(1 - out[months == 8].max() / base_month[8]),
                    "sep_reduction": float(1 - out[months == 9].max() / base_month[9]),
                    "intervals_above_D_before": int(exceed.sum()), "intervals_above_D_after": int((out > D + 1e-9).sum()),
                    "exceed_armed_share": float(armed[exceed].mean()) if exceed.any() else np.nan,
                    "armed_hours_per_working_day": float(armed[work & whole].sum() / 4 / n_work_days),
                    "control_events": int(len(cut_idx)),
                    "control_days": int(len(np.unique(times[cut_idx].date))) if len(cut_idx) else 0,
                    "mean_deferred_per_event": float(r["cut_total"] / len(cut_idx)) if len(cut_idx) else 0.0,
                    "mean_deferred_share_of_load": float(np.mean(r["cut_amounts"] / actual[cut_idx])) if len(cut_idx) else 0.0,
                    "deferred_energy_share_pct": float(100 * r["cut_total"] / total_energy),
                    "forced_releases": r["forced"]})
                if name == "예측 경보로 준비" and c == 0.10:
                    for t in cut_idx:
                        where.append({"target_D": D, "time": str(times[t]), "slot": slot[t], "cut_start_window": bool(start[t])})
    return pd.DataFrame(rows), pd.DataFrame(where)


def figures(curve: pd.DataFrame, scen: pd.DataFrame) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(curve["lead_min"], curve["captured_share"] * 100, marker="o", color="#1f6fb2")
    ax.set_xlabel("경보 시각부터 최대 구간 시작까지 (분)")
    ax.set_ylabel("사전 경보된 일 최대 구간 (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_1_daily_max_lead.png", dpi=170)
    plt.close(fig)

    part = scen[scen["deferrable_share"] == 0.10]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for name, g in part.groupby("arming", sort=False):
        ax.plot(g["target_D"], g["max_after"], marker="o", label=name)
    ax.plot(part["target_D"].unique(), part["target_D"].unique(), color="grey", ls="--", lw=.8, label="목표 최대수요")
    ax.axhline(part["max_before"].iloc[0], color="black", lw=.8, label="운영 전 최대")
    ax.invert_xaxis()
    ax.set_xlabel("목표 최대수요 D (원자료 단위)")
    ax.set_ylabel("운영 후 평가 구간 최대 15분 전력")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_3_demand_cap.png", dpi=170)
    plt.close(fig)


def main() -> dict:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(ROOT / "outputs/predictions/final_test_fg_r11.csv", parse_dates=["origin", "target_time"])
    table, facts, curve = peak_capture(full)
    table.to_csv(TAB / "daily_max_capture.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(TAB / "daily_max_lead_curve.csv", index=False, encoding="utf-8-sig")
    scen, where = run_scenarios(full)
    scen.to_csv(TAB / "demand_cap_scenarios.csv", index=False, encoding="utf-8-sig")
    where.to_csv(TAB / "demand_cap_events.csv", index=False, encoding="utf-8-sig")
    figures(curve, scen)
    (TAB / "reduction_summary.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, indent=1, default=float))
    return facts


if __name__ == "__main__":
    main()
