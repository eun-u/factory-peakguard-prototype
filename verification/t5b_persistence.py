"""Compare persistence baselines with the unchanged task ⑤ one-hour setup.

Run from the repository root: python -X utf8 verification/t5b_persistence.py
Writes only 05b_ result files. The prior task ⑤ code and results are read only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from t5_power import (
    BOOT,
    OUT,
    SEED,
    feature_frame,
    load_series,
    naive_predict,
    regression_model,
    split_time,
    val_f1_cutoff,
)


METRIC_KEYS = (
    "mae", "rmse", "peak_position_mae", "peak_episode_mae", "peak_day_mae",
    "position_f1", "episode_f1", "day_f1",
)
MODEL_LABELS = {
    "seasonal": "계절 나이브",
    "persistence": "최선 persistence",
    "lgb": "LightGBM 회귀",
}
PERSISTENCE_LABELS = {
    "latest_15m": "P-a 원점의 최근 15분값",
    "previous_hour_slot": "P-b 원점 1시간 전 같은 위치값",
    "recent_hour_mean": "P-c 최근 네 15분값 평균",
}


def persistence_predict(x: pd.DataFrame, version: str) -> np.ndarray:
    columns = {
        "latest_15m": "current",
        "previous_hour_slot": "lag_4",
        "recent_hour_mean": "recent_hour_mean",
    }
    return x[columns[version]].to_numpy(dtype=float)


def runs(flags: np.ndarray, times: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive runs; a missing 15-minute timestamp starts a new episode."""
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


def overlap_match_count(
    truth: list[tuple[int, int]], alarms: list[tuple[int, int]]
) -> int:
    """Maximum one-to-one overlap matching, so one long alarm detects at most one event."""
    edges = [
        [j for j, (a0, a1) in enumerate(alarms) if max(t0, a0) <= min(t1, a1)]
        for t0, t1 in truth
    ]
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


def daily_stats(frame: pd.DataFrame, peak: float, cutoff: float, model: str) -> pd.DataFrame:
    records = []
    for date, group in frame.groupby("date", sort=True):
        actual = group["y"].to_numpy(dtype=float)
        predicted = group[model].to_numpy(dtype=float)
        times = group["target_time"].to_numpy(dtype="datetime64[ns]")
        true_event = actual > peak
        alarm = predicted >= cutoff
        error = np.abs(actual - predicted)
        truth_runs = runs(true_event, times)
        alarm_runs = runs(alarm, times)
        matched = overlap_match_count(truth_runs, alarm_runs)
        position_tp = int(np.sum(true_event & alarm))
        records.append({
            "date": date,
            "n": len(group),
            "ae_sum": float(error.sum()),
            "se_sum": float(np.square(actual - predicted).sum()),
            "peak_n": int(true_event.sum()),
            "peak_ae_sum": float(error[true_event].sum()),
            "episode_n": len(truth_runs),
            "episode_ae_sum": float(sum(error[a:b + 1].mean() for a, b in truth_runs)),
            "peak_day_n": int(true_event.any()),
            "peak_day_ae_sum": float(error[true_event].mean()) if true_event.any() else 0.0,
            "position_tp": position_tp,
            "position_fp": int(np.sum(~true_event & alarm)),
            "position_fn": int(np.sum(true_event & ~alarm)),
            "episode_tp": matched,
            "episode_fp": len(alarm_runs) - matched,
            "episode_fn": len(truth_runs) - matched,
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


def block_bootstrap(
    stats: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]], dict[str, object]]:
    names = tuple(stats)
    dates = stats[names[0]]["date"].tolist()
    if any(item["date"].tolist() != dates for item in stats.values()):
        raise AssertionError("Models have different evaluation dates")
    point = {name: summarize_stats(item) for name, item in stats.items()}
    rng = np.random.default_rng(SEED)
    samples: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in METRIC_KEYS} for name in names
    }
    improvements = []
    for _ in range(BOOT):
        take = rng.integers(0, len(dates), size=len(dates))
        draw = {name: summarize_stats(item.iloc[take]) for name, item in stats.items()}
        for name in names:
            for metric in METRIC_KEYS:
                samples[name][metric].append(draw[name][metric])
        baseline = draw["persistence"]["peak_position_mae"]
        candidate = draw["lgb"]["peak_position_mae"]
        improvements.append(1 - candidate / baseline if baseline > 0 else float("nan"))

    def ci(values: list[float]) -> list[float]:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if len(finite) < BOOT // 2:
            return [float("nan"), float("nan")]
        return [float(x) for x in np.quantile(finite, [0.025, 0.975])]

    intervals = {
        name: {metric: ci(samples[name][metric]) for metric in METRIC_KEYS}
        for name in names
    }
    baseline = point["persistence"]["peak_position_mae"]
    improvement = 1 - point["lgb"]["peak_position_mae"] / baseline
    comparison = {
        "peak_mae_improvement_vs_persistence": improvement,
        "ci95": ci(improvements),
        "resampling_dates": len(dates),
        "bootstrap_draws": BOOT,
    }
    return point, intervals, comparison


def _format(value: float, ci: list[float], digits: int = 2) -> str:
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def write_report(summary: dict) -> None:
    metrics = summary["test_metrics"]
    ci = summary["test_ci95"]
    validation = summary["validation"]
    support = summary["test_support"]
    compare = summary["comparison"]
    original = summary["original_reference"]

    lines = [
        "# ⑤ persistence 베이스라인 추가 검증",
        "",
        "## 증거",
        "",
        "### 고정된 비교 조건",
        "",
        f"- 예측 대상: 원점에서 1시간 후 15분값. 기존 학습/검증/시험 행 수 "
        f"{summary['split_rows']['train']:,}/{summary['split_rows']['validation']:,}/{summary['split_rows']['test']:,}. "
        f"학습값 상위 5% 피크 경계 {summary['peak_threshold']:.2f}.",
        "- 기존 t5_power.py의 원자료 정리, 특징, 시간순 분할, 4개 원점 경계 제거, LightGBM 회귀 설정을 그대로 호출했다. 기존 05_test_predictions.csv의 실제값·계절 나이브·LightGBM 예측값과 재계산값을 대조했다.",
        "- P-a는 원점의 마지막 관측값, P-b는 원점보다 정확히 1시간 앞선 같은 분 위치값, P-c는 원점을 포함한 최근 네 15분값 평균이다. 모두 목표값보다 앞선 관측이다.",
        "- 계절 후보와 persistence 후보는 각각 검증 전체 MAE 최소를 선택했다. 세 모델의 피크 사건 판정용 예측값 임계값은 각각 검증구간 위치 F1 최대로 고정했다.",
        "",
        "### 검증구간 선택",
        "",
        "| 후보 | 검증 전체 MAE |",
        "| :-- | --: |",
    ]
    for key, value in validation["seasonal_mae"].items():
        lines.append(f"| 계절 {key} | {value:.3f} |")
    for key, value in validation["persistence_mae"].items():
        lines.append(f"| {PERSISTENCE_LABELS[key]} | {value:.3f} |")
    lines += [
        "",
        f"계절 나이브는 {validation['seasonal_selected']}, persistence는 "
        f"{PERSISTENCE_LABELS[validation['persistence_selected']]}을 시험 전에 고정했다.",
        "",
        "### 테스트 점예측 오차",
        "",
        "값은 점추정 [날짜 블록 부트스트랩 95% CI]다. 전체 오차는 위치를, 피크 위치/에피소드/발생일 MAE는 각 단위를 동일 가중한다.",
        "",
        "| 모델 | 전체 MAE | 전체 RMSE | 피크 위치 MAE | 피크 에피소드 MAE | 피크 발생일 MAE |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for name in MODEL_LABELS:
        m, interval = metrics[name], ci[name]
        fields = [_format(m[key], interval[key]) for key in
                  ("mae", "rmse", "peak_position_mae", "peak_episode_mae", "peak_day_mae")]
        lines.append(f"| {MODEL_LABELS[name]} | " + " | ".join(fields) + " |")
    lines += [
        "",
        "### 테스트 피크 사건",
        "",
        f"시험의 {support['test_dates']}개 목표 날짜 중 실제 피크는 {support['positions']}개 15분 위치, "
        f"{support['episodes']}개 연속 에피소드, {support['days']}개 날짜에 발생했다. "
        "에피소드는 15분 간격이 끊기면 새 사건으로 세며, "
        "한 경보 에피소드는 겹치는 실제 에피소드 최대 1개와 짝짓는다.",
        "",
        "| 모델 | 검증 고정 임계값 | 위치 F1 | 에피소드 F1 | 피크 발생일 F1 |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for name in MODEL_LABELS:
        m, interval = metrics[name], ci[name]
        cutoff = validation["event_cutoff"][name]
        fields = [_format(m[key], interval[key], 3) for key in
                  ("position_f1", "episode_f1", "day_f1")]
        lines.append(f"| {MODEL_LABELS[name]} | {cutoff:.3f} | " + " | ".join(fields) + " |")
    lines += [
        "",
        "TP/FP/FN은 순서대로 다음과 같다. 날짜 F1은 해당 날짜에 한 번이라도 경보가 있었는지를 평가한다.",
        "",
        "| 모델 | 위치 TP/FP/FN | 에피소드 TP/FP/FN | 날짜 TP/FP/FN |",
        "| :-- | :-- | :-- | :-- |",
    ]
    for name in MODEL_LABELS:
        s = support["confusion"][name]
        groups = [f"{s[unit + '_tp']}/{s[unit + '_fp']}/{s[unit + '_fn']}"
                  for unit in ("position", "episode", "day")]
        lines.append(f"| {MODEL_LABELS[name]} | " + " | ".join(groups) + " |")
    lines += [
        "",
        f"최선 persistence 대비 LightGBM의 피크 위치 MAE 개선률은 "
        f"{_format(compare['peak_mae_improvement_vs_persistence'], compare['ci95'], 3)}. "
        f"사전 고정 K5-d: **{summary['criterion']['K5-d']}**.",
        f"기존 K5-b는 계절 나이브 대비 개선률 "
        f"{_format(original['peak_mae_improvement_fraction'], original['ci95'], 3)}에 근거해 "
        f"**{original['K5-b']}**이었다. 이번 모델링 여지 판단에는 K5-d가 K5-b를 대체한다. "
        f"기존 별도 LightGBM 분류기의 위치 F1은 "
        f"{_format(original['classifier_f1'], original['classifier_f1_ci95'], 3)}이며, "
        "위 표의 LightGBM 회귀 예측값을 임계화한 F1과는 다른 모델 결과다.",
        "",
        "재실행: python -X utf8 verification/t5b_persistence.py. "
        "재현 자료: 05b_summary.json, 05b_predictions.csv. 사전 기준: 05b_decision_criteria.md. "
        "③·⑤ 기존 보고서와 05_ 결과 파일은 수정하지 않았다.",
        "",
        "## 해석과 한계",
        "",
    ]
    if summary["criterion"]["K5-d"] == "해당":
        lines.append(
            "더 강한 persistence 기준에서 K5-d 경고가 성립하므로 1시간 후 피크 구간에 대한 "
            "추가 모델링 여지는 작다고 판정한다. 이는 기존 계절 나이브만 기준으로 한 "
            "K5-b 비해당과 결론이 달라진 것이다."
        )
    else:
        lines.append(
            "더 강한 persistence 기준에서도 K5-d 경고는 성립하지 않는다. 기존 K5-b 비해당의 "
            "결론은 유지되며, 개선폭은 이 시험기간과 경계값 정의에 한정된다."
        )
    lines += [
        "",
        "- 날짜 블록 재표집은 같은 날짜 안의 15분 위치 의존성을 보존한다. 계절 외삽, 새 공장, 미래 운영조건의 불확실성은 포함하지 않는다.",
        "- 피크는 학습값의 상위 5% 초과이며 실제 계약전력 초과가 아니다. 원자료의 전력 단위와 15분 구간 경계도 미확인이다.",
        "- 날짜 수준 F1은 하루 한 번만 경보해도 그날 피크를 잡은 것으로 볼 수 있으므로 현장 경보 품질의 충분한 지표가 아니다. 에피소드 F1은 시간 겹침만 요구하며 조기 경보 여부는 평가하지 않는다.",
        "- 검증 전체 MAE로 persistence를 선택했으므로 피크 MAE에 최적인 persistence를 고른 실험은 아니다. K5-d는 사전 선택 절차 그대로 평가했다.",
        "",
    ]
    (OUT / "05b_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    criterion = OUT / "05b_decision_criteria.md"
    if not criterion.exists():
        raise FileNotFoundError("Freeze 05b_decision_criteria.md before evaluation")
    series, _, _ = load_series()
    x, y, meta = feature_frame(series)
    split = split_time(x, y, meta)
    xt, yt, mt = split["train"]
    xv, yv, mv = split["validation"]
    xs, ys, ms = split["test"]
    if not (mt["target_time"].max() < xv.index.min()
            and mv["target_time"].max() < xs.index.min()):
        raise AssertionError("Split labels overlap a later forecast origin")
    peak = float(yt.quantile(0.95))

    seasonal_mae = {
        name: float(mean_absolute_error(yv, naive_predict(xv, name)))
        for name in ("day", "week", "average")
    }
    persistence_mae = {
        name: float(mean_absolute_error(yv, persistence_predict(xv, name)))
        for name in PERSISTENCE_LABELS
    }
    seasonal_selected = min(seasonal_mae, key=seasonal_mae.get)
    persistence_selected = min(persistence_mae, key=persistence_mae.get)
    model = regression_model()
    model.fit(xt, yt)
    validation_preds = {
        "seasonal": naive_predict(xv, seasonal_selected),
        "persistence": persistence_predict(xv, persistence_selected),
        "lgb": model.predict(xv),
    }
    test_preds = {
        "seasonal": naive_predict(xs, seasonal_selected),
        "persistence": persistence_predict(xs, persistence_selected),
        "lgb": model.predict(xs),
    }
    cutoffs = {
        name: val_f1_cutoff((yv > peak).to_numpy(), pred)
        for name, pred in validation_preds.items()
    }
    frame = pd.DataFrame({
        "y": ys.to_numpy(dtype=float),
        "seasonal": test_preds["seasonal"],
        "persistence": test_preds["persistence"],
        "lgb": test_preds["lgb"],
        "target_time": ms["target_time"].to_numpy(),
        "date": ms["target_time"].dt.date.to_numpy(),
    }, index=xs.index)
    frame.index.name = "forecast_origin"
    frame["actual_peak"] = frame["y"] > peak
    for name in MODEL_LABELS:
        frame[name + "_alarm"] = frame[name] >= cutoffs[name]

    # The reference is a read-only consistency check, never a source of fitted values.
    prior = json.loads((OUT / "05_summary.json").read_text(encoding="utf-8"))
    prior_frame = pd.read_csv(OUT / "05_test_predictions.csv", parse_dates=["forecast_origin"])
    expected_rows = {name: len(part[0]) for name, part in split.items()}
    if prior["one_hour"]["rows"] != expected_rows:
        raise AssertionError("Time split differs from existing task ⑤ results")
    if prior["one_hour"]["train_peak_95pct"] != peak:
        raise AssertionError("Training peak threshold differs from existing task ⑤")
    if prior["one_hour"]["naive_selected_on_validation"] != seasonal_selected:
        raise AssertionError("Seasonal naive selection differs from existing task ⑤")
    if not np.array_equal(prior_frame["forecast_origin"].to_numpy(), xs.index.to_numpy()):
        raise AssertionError("Test forecast origins differ from existing task ⑤")
    for current, old in (("y", "y"), ("seasonal", "naive"), ("lgb", "lgb")):
        if not np.allclose(frame[current].to_numpy(), prior_frame[old].to_numpy(), atol=1e-8):
            raise AssertionError(f"{current} differs from existing task ⑤ predictions")

    stats = {name: daily_stats(frame, peak, cutoffs[name], name) for name in MODEL_LABELS}
    point, ci, comparison = block_bootstrap(stats)
    k5d = (
        "해당" if comparison["peak_mae_improvement_vs_persistence"] < 0.10
        or comparison["ci95"][0] <= 0 else "비해당"
    )
    actual = stats["seasonal"].drop(columns="date").sum(numeric_only=True)
    confusion = {}
    for name, item in stats.items():
        total = item.drop(columns="date").sum(numeric_only=True)
        confusion[name] = {
            key: int(total[key]) for key in (
                "position_tp", "position_fp", "position_fn",
                "episode_tp", "episode_fp", "episode_fn",
                "day_tp", "day_fp", "day_fn",
            )
        }
    summary = {
        "seed": SEED,
        "split_rows": expected_rows,
        "split_ranges": {
            name: [str(part[0].index.min()), str(part[0].index.max())]
            for name, part in split.items()
        },
        "peak_threshold": peak,
        "validation": {
            "seasonal_mae": seasonal_mae,
            "persistence_mae": persistence_mae,
            "seasonal_selected": seasonal_selected,
            "persistence_selected": persistence_selected,
            "event_cutoff": cutoffs,
        },
        "test_support": {
            "positions": int(actual["peak_n"]),
            "episodes": int(actual["episode_n"]),
            "days": int(actual["peak_day_n"]),
            "test_dates": len(stats["seasonal"]),
            "confusion": confusion,
        },
        "test_metrics": point,
        "test_ci95": ci,
        "comparison": comparison,
        "criterion": {"K5-d": k5d},
        "original_reference": {
            "K5-b": prior["criteria"]["K5-b"],
            "peak_mae_improvement_fraction": prior["one_hour"]["metrics"]["peak_mae_improvement_fraction"],
            "ci95": prior["one_hour"]["ci95"]["peak_mae_improvement_fraction"],
            "classifier_f1": prior["one_hour"]["metrics"]["classifier_f1"],
            "classifier_f1_ci95": prior["one_hour"]["ci95"]["classifier_f1"],
        },
        "definitions": {
            "persistence_b": "power at origin minus four 15-minute positions",
            "peak": "actual value above training 95th percentile; alarms use validation-selected prediction cutoffs",
            "event_cutoff": "prediction-value threshold maximizing validation position F1",
            "episode": "consecutive 15-minute target positions within a calendar date; missing positions break runs",
            "episode_match": "maximum one-to-one time-overlap matching",
            "bootstrap": "1000 paired target-date block resamples; cutoffs and model selection held fixed",
        },
    }
    frame.to_csv(OUT / "05b_predictions.csv", encoding="utf-8-sig")
    (OUT / "05b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary)
    print(json.dumps({
        "persistence_selected": persistence_selected,
        "peak_mae_improvement": comparison["peak_mae_improvement_vs_persistence"],
        "ci95": comparison["ci95"],
        "K5-d": k5d,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
