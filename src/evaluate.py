"""평가 지표, 피크 에피소드 매칭, 날짜 블록 부트스트랩.

- 피크: 실제값 > 학습 구간 상위 분위수(기본 95%)
- 에피소드: 같은 날짜 안에서 연속된 15분 초과 위치. 위치가 빠지면 새 에피소드
- 매칭: 경보 에피소드와 실제 에피소드를 시간 겹침으로 최대 1:1 매칭
- 신뢰구간: 테스트 날짜를 묶어 1,000회 복원추출한 백분위 95% 구간. 임계값·모델 선택은 고정
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_pinball_loss,
    precision_recall_curve,
)

from .analysis.decision import rev


def val_f1_cutoff(y: np.ndarray, score: np.ndarray) -> float:
    """검증 구간에서 위치 F1을 최대로 하는 점수 임계값."""
    if np.unique(y).size < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan")


def metric_bundle(frame: pd.DataFrame, peak: float, cutoffs: dict[str, float], ratios, quantiles) -> dict:
    """사전 검증 05_summary.json과 같은 지표 묶음."""
    y = frame["y"].to_numpy()
    event = y > peak
    out = {}
    for model in ("day", "week", "average", "naive", "lgb"):
        pred = frame[model].to_numpy()
        err = y - pred
        out[f"{model}_mae"] = float(np.mean(np.abs(err)))
        out[f"{model}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        nz = y != 0
        out[f"{model}_mape"] = float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else float("nan")
        out[f"{model}_peak_mae"] = float(np.mean(np.abs(err[event]))) if event.any() else float("nan")
    out["peak_mae_improvement_fraction"] = (float(1 - out["lgb_peak_mae"] / out["naive_peak_mae"])
                                            if out["naive_peak_mae"] > 0 else float("nan"))
    for q in quantiles:
        name = f"q{int(q * 100)}"
        pred = frame[name].to_numpy()
        out[f"{name}_pinball"] = float(mean_pinball_loss(y, pred, alpha=q))
        out[f"{name}_coverage"] = float(np.mean(y <= pred))
    out["mean_pinball"] = float(np.mean([out[f"q{int(q * 100)}_pinball"] for q in quantiles]))
    for model in ("classifier", "quantile"):
        prob = frame[f"{model}_prob"].to_numpy()
        out[f"{model}_f1"] = float(f1_score(event, prob >= cutoffs[model], zero_division=0))
        out[f"{model}_prauc"] = safe_ap(event, prob)
        out[f"{model}_brier"] = float(brier_score_loss(event, prob))
        action = prob >= cutoffs[model]
        out[f"{model}_precision"] = float(np.sum(action & event) / np.sum(action)) if action.any() else float("nan")
        out[f"{model}_recall"] = float(np.sum(action & event) / np.sum(event)) if event.any() else float("nan")
    for i, ratio in enumerate(ratios):
        out[f"rev_{i}"] = rev(event, frame["classifier_prob"].to_numpy(), frame["climate_prob"].to_numpy(), ratio)
    return out


def block_bootstrap(frame: pd.DataFrame, fn, n: int = 1000, seed: int = 42) -> tuple[dict, dict]:
    """날짜(frame['date']) 단위 복원추출. 표본 절반 이상이 유한할 때만 구간을 낸다."""
    point = fn(frame)
    groups = [np.where(frame["date"].to_numpy() == day)[0] for day in pd.unique(frame["date"])]
    rng = np.random.default_rng(seed)
    values = {k: [] for k in point}
    for _ in range(n):
        sampled = rng.integers(0, len(groups), len(groups))
        take = np.concatenate([groups[i] for i in sampled])
        result = fn(frame.iloc[take])
        for key, value in result.items():
            values[key].append(value)
    ci = {}
    for key, arr in values.items():
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        ci[key] = ([float(v) for v in np.quantile(finite, [0.025, 0.975])]
                   if len(finite) >= n // 2 else [None, None])
    return point, ci


# ---- 피크 위치·에피소드·날짜 단위 지표 (사전 검증 t5b_persistence와 동일) ----

STAT_KEYS = ("mae", "rmse", "peak_position_mae", "peak_episode_mae", "peak_day_mae",
             "position_f1", "episode_f1", "day_f1")


def runs(flags: np.ndarray, times: np.ndarray) -> list[tuple[int, int]]:
    """닫힌 구간 [시작, 끝]. 15분 간격이 끊기면 새 에피소드."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        contiguous = i > 0 and times[i] - times[i - 1] == np.timedelta64(15, "m")
        if start is not None and (not flag or not contiguous):
            out.append((start, i - 1))
            start = None
        if flag and start is None:
            start = i
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def overlap_match_count(truth: list[tuple[int, int]], alarms: list[tuple[int, int]]) -> int:
    """최대 1:1 겹침 매칭. 긴 경보 하나가 여러 실제 에피소드를 동시에 맞힌 것으로 세지 않는다."""
    edges = [[j for j, (a0, a1) in enumerate(alarms) if max(t0, a0) <= min(t1, a1)] for t0, t1 in truth]
    matched_alarm: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched_alarm or augment(matched_alarm[j], seen):
                matched_alarm[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(truth)))


def daily_stats(frame: pd.DataFrame, peak: float, cutoff: float | None, model: str,
                alarm_column: str | None = None) -> pd.DataFrame:
    """날짜별 합계. alarm_column이 있으면 그 불리언 열을 경보로 쓰고, 없으면 예측값 ≥ cutoff."""
    records = []
    for date, group in frame.groupby("date", sort=True):
        actual = group["y"].to_numpy(dtype=float)
        predicted = group[model].to_numpy(dtype=float)
        times = group["target_time"].to_numpy(dtype="datetime64[ns]")
        true_event = actual > peak
        alarm = group[alarm_column].to_numpy(dtype=bool) if alarm_column else predicted >= cutoff
        error = np.abs(actual - predicted)
        truth_runs = runs(true_event, times)
        alarm_runs = runs(alarm, times)
        matched = overlap_match_count(truth_runs, alarm_runs)
        records.append({
            "date": date, "n": len(group),
            "ae_sum": float(error.sum()), "se_sum": float(np.square(actual - predicted).sum()),
            "peak_n": int(true_event.sum()), "peak_ae_sum": float(error[true_event].sum()),
            "episode_n": len(truth_runs),
            "episode_ae_sum": float(sum(error[a:b + 1].mean() for a, b in truth_runs)),
            "peak_day_n": int(true_event.any()),
            "peak_day_ae_sum": float(error[true_event].mean()) if true_event.any() else 0.0,
            "position_tp": int(np.sum(true_event & alarm)),
            "position_fp": int(np.sum(~true_event & alarm)),
            "position_fn": int(np.sum(true_event & ~alarm)),
            "episode_tp": matched, "episode_fp": len(alarm_runs) - matched, "episode_fn": len(truth_runs) - matched,
            "alarm_episode_n": len(alarm_runs),
            "day_tp": int(true_event.any() and alarm.any()),
            "day_fp": int(not true_event.any() and alarm.any()),
            "day_fn": int(true_event.any() and not alarm.any()),
        })
    return pd.DataFrame.from_records(records)


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


def summarize_stats(stats: pd.DataFrame) -> dict[str, float]:
    s = stats.drop(columns="date").sum(numeric_only=True)

    def ratio(num: str, den: str) -> float:
        return float(s[num] / s[den]) if s[den] > 0 else float("nan")

    return {
        "mae": ratio("ae_sum", "n"),
        "rmse": float(np.sqrt(ratio("se_sum", "n"))),
        "peak_position_mae": ratio("peak_ae_sum", "peak_n"),
        "peak_episode_mae": ratio("episode_ae_sum", "episode_n"),
        "peak_day_mae": ratio("peak_day_ae_sum", "peak_day_n"),
        "position_f1": _f1(s["position_tp"], s["position_fp"], s["position_fn"]),
        "episode_f1": _f1(s["episode_tp"], s["episode_fp"], s["episode_fn"]),
        "day_f1": _f1(s["day_tp"], s["day_fp"], s["day_fn"]),
    }


def _ci(values, n: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) < n // 2:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])]


def paired_stats_bootstrap(stats: dict[str, pd.DataFrame], baseline: str = "persistence",
                           candidate: str = "lgb", n: int = 1000, seed: int = 42):
    """모델들을 같은 날짜 표본으로 함께 재표집한다. 개선률 = 1 − 후보 피크 MAE / 기준 피크 MAE."""
    names = tuple(stats)
    dates = stats[names[0]]["date"].tolist()
    if any(item["date"].tolist() != dates for item in stats.values()):
        raise AssertionError("Models have different evaluation dates")
    point = {name: summarize_stats(item) for name, item in stats.items()}
    rng = np.random.default_rng(seed)
    samples = {name: {metric: [] for metric in STAT_KEYS} for name in names}
    improvements = []
    for _ in range(n):
        take = rng.integers(0, len(dates), size=len(dates))
        draw = {name: summarize_stats(item.iloc[take]) for name, item in stats.items()}
        for name in names:
            for metric in STAT_KEYS:
                samples[name][metric].append(draw[name][metric])
        base = draw[baseline]["peak_position_mae"]
        improvements.append(1 - draw[candidate]["peak_position_mae"] / base if base > 0 else float("nan"))
    intervals = {name: {metric: _ci(samples[name][metric], n) for metric in STAT_KEYS} for name in names}
    comparison = {
        "baseline": baseline, "candidate": candidate,
        "peak_mae_improvement": 1 - point[candidate]["peak_position_mae"] / point[baseline]["peak_position_mae"],
        "ci95": _ci(improvements, n), "resampling_dates": len(dates), "bootstrap_draws": n,
    }
    return point, intervals, comparison
