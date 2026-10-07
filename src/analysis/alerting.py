"""경보 운영 규칙과 KPI. 경보관리 표준의 지속성(k/n) 규칙을 차용한다.

- 원 경보: 원점 t에서 1시간 후 예측값 ≥ 검증 F1 최대 임계값
- k/n 규칙: 최근 n개 원점 중 k개 이상 원 경보이면 경보 유지 (원점이 15분 간격으로 이어질 때만)
- 2단계: 주의 = 95% 분위수 예측 ≥ 피크 경계 / 조치 = k/n 확인된 점예측 경보
- KPI: 에피소드 재현율·정밀도, 하루당 오경보 에피소드, 경보 선행시간(실제 피크 시작 − 경보 발령 시각)
규칙은 검증 구간에서 고르고 테스트에는 그대로 적용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import daily_stats, runs


def k_of_n(raw_alarm: np.ndarray, origins: pd.DatetimeIndex, k: int, n: int) -> np.ndarray:
    out = np.zeros(len(raw_alarm), dtype=bool)
    block_start = 0
    for i in range(len(raw_alarm)):
        if i > 0 and origins[i] - origins[i - 1] != pd.Timedelta(minutes=15):
            block_start = i
        window = raw_alarm[max(block_start, i - n + 1):i + 1]
        out[i] = len(window) >= min(n, k) and window.sum() >= k
    return out


def lead_times(frame: pd.DataFrame, peak: float, alarm_column: str, horizon_minutes: int = 60) -> list[float]:
    """매칭된 실제 에피소드마다 (에피소드 시작 시각 − 첫 겹침 경보의 발령 시각), 분 단위."""
    leads = []
    for _, g in frame.groupby("date", sort=True):
        times = g["target_time"].to_numpy(dtype="datetime64[ns]")
        truth = runs(g["y"].to_numpy() > peak, times)
        alarms = runs(g[alarm_column].to_numpy(dtype=bool), times)
        for t0, t1 in truth:
            overlapping = [a0 for a0, a1 in alarms if max(t0, a0) <= min(t1, a1)]
            if overlapping:
                first = min(overlapping)
                issued = times[first] - np.timedelta64(horizon_minutes, "m")
                leads.append(float((times[t0] - np.timedelta64(15, "m") - issued) / np.timedelta64(1, "m")))
    return leads


def kpis(frame: pd.DataFrame, peak: float, alarm_column: str) -> dict:
    stats = daily_stats(frame, peak, None, "lgb", alarm_column=alarm_column)
    s = stats.drop(columns="date").sum(numeric_only=True)
    days = len(stats)
    leads = lead_times(frame, peak, alarm_column)
    recall = s["episode_tp"] / s["episode_n"] if s["episode_n"] else np.nan
    precision = s["episode_tp"] / s["alarm_episode_n"] if s["alarm_episode_n"] else np.nan
    return {"days": days, "actual_episodes": int(s["episode_n"]), "alarm_episodes": int(s["alarm_episode_n"]),
            "episode_tp": int(s["episode_tp"]), "episode_fp": int(s["episode_fp"]), "episode_fn": int(s["episode_fn"]),
            "episode_recall": float(recall), "episode_precision": float(precision),
            "episode_f1": float(2 * s["episode_tp"] / (2 * s["episode_tp"] + s["episode_fp"] + s["episode_fn"]))
            if (2 * s["episode_tp"] + s["episode_fp"] + s["episode_fn"]) else np.nan,
            "false_alarm_episodes_per_day": float(s["episode_fp"] / days) if days else np.nan,
            "alarm_positions_share": float(frame[alarm_column].mean()),
            "median_lead_minutes": float(np.median(leads)) if leads else np.nan,
            "share_warned_before_start": float(np.mean(np.asarray(leads) > 0)) if leads else np.nan}


def evaluate_rules(frames: dict[str, pd.DataFrame], peak: float, cutoff: float, rules) -> pd.DataFrame:
    rows = []
    for split, frame in frames.items():
        frame = frame.sort_values("target_time").copy()
        raw = (frame["lgb"] >= cutoff).to_numpy()
        frame["watch"] = (frame["q95"] >= peak).to_numpy()
        rows.append({"split": split, "stage": "주의", "rule": "q95 ≥ 경계", **kpis(frame, peak, "watch")})
        for k, n in rules:
            frame["act"] = k_of_n(raw, frame.index, k, n)
            rows.append({"split": split, "stage": "조치", "rule": f"{k}/{n}", **kpis(frame, peak, "act")})
    return pd.DataFrame(rows)


def select_rule(table: pd.DataFrame) -> str:
    """검증 구간 조치 단계에서 에피소드 F1이 가장 높은 규칙. 동률이면 오경보가 적은 규칙."""
    val = table[(table["split"] == "validation") & (table["stage"] == "조치")]
    best = val.sort_values(["episode_f1", "false_alarm_episodes_per_day"], ascending=[False, True]).iloc[0]
    return str(best["rule"])

