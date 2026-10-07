"""과제 ⑤ 전체 재현: 전처리 → 학습 → 평가 → 분석 → 예측 파일 → 그림·표.

    python run_all.py                      # configs/default.yaml 사용
    python run_all.py --output-dir 경로    # 출력 위치만 바꿀 때

1시간 모델·임계값·기준선 선택은 사전 검증(verification/t5_power.py, t5b_persistence.py)과
같은 고정 설정이다. 이후 분석은 모델을 다시 고르지 않는다.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src import viz
from src.analysis import alerting, errors, importance, peak_conditions, simulate_shift
from src.analysis.decision import climatology
from src.analysis.horizon_curve import conformal_dev, horizon_curve, rolling_origin
from src.config import ROOT, load_config, output_dirs
from src.daily_max import daily_max_task
from src.data import calendar_profiles, load_series, variable_dictionary, zero_value_runs
from src.evaluate import (STAT_KEYS, block_bootstrap, daily_stats, metric_bundle, paired_stats_bootstrap,
                          val_f1_cutoff)
from src.features import feature_frame
from src.models.baselines import naive_predict, persistence_predict
from src.models.lgbm_point import regression_model
from src.models.lgbm_quantile import fit_quantiles, predict_quantiles, quantile_event_probability
from src.models.peak_classifier import fit_classifier
from src.split import split_time

# pandas 2.3 + numpy 2.5 조합에서 pandas 내부 시간 연산이 내는 경고. 결과에는 영향이 없다.
warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta")

MODEL_LABELS = {"seasonal": "계절 나이브", "persistence": "persistence", "lgb": "LightGBM 회귀"}


def _relative(path: str) -> str:
    """로그에 개인 계정 경로가 남지 않도록 저장소 기준 상대경로만 기록한다."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write(df: pd.DataFrame, path, index: bool = False) -> None:
    df.to_csv(path, index=index, encoding="utf-8-sig")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=None, help="점검용. 제출 결과는 설정값(1000) 사용")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.bootstrap_draws:
        cfg["bootstrap_draws"] = args.bootstrap_draws
    boot, seed = cfg["bootstrap_draws"], cfg["seed"]
    dirs = output_dirs(cfg)
    fig, tab, pred_dir = dirs["figures"], dirs["tables"], dirs["predictions"]
    started = time.time()
    font = viz.setup_font()

    # ---------------- 1. 데이터 이해·진단 ----------------
    log("1/9 원자료 로딩·시간 복원·15분 전개")
    series, raw, info = load_series(cfg)
    if not info["source_sha256_matches_expected"]:
        log("경고: 원자료 SHA-256이 data/README.md의 값과 다릅니다")
    write(variable_dictionary(raw), tab / "t1_variable_dictionary.csv")
    zero_runs = zero_value_runs(series)
    write(zero_runs, tab / "t1_zero_value_runs.csv")
    profiles = calendar_profiles(series, raw)
    for key, frame in profiles.items():
        write(frame, tab / f"t1_profile_{key}.csv")

    # ---------------- 2. 1시간 후 예측: 사전 검증과 같은 고정 설정 ----------------
    log("2/9 1시간 후 예측 학습 (점·분위수·분류기)")
    oh = cfg["one_hour"]
    horizon, quantiles = oh["horizon_steps"], tuple(oh["quantiles"])
    x, y, meta = feature_frame(series, horizon, oh["lags"])
    split = split_time(x, y, meta, tuple(oh["split_fractions"]), oh["purge_origins"])
    xt, yt, mt = split["train"]
    xv, yv, mv = split["validation"]
    xs, ys, ms = split["test"]
    peak = float(yt.quantile(oh["peak_quantile"]))
    model = regression_model(cfg).fit(xt, yt)
    qmodels = fit_quantiles(cfg, xt, yt, quantiles)
    classifier = fit_classifier(cfg, xt, yt > peak)

    val_q, test_q = predict_quantiles(qmodels, xv), predict_quantiles(qmodels, xs)
    val_qprob, val_cross = quantile_event_probability(val_q, peak, quantiles)
    test_qprob, test_cross = quantile_event_probability(test_q, peak, quantiles)
    val_cprob, test_cprob = classifier.predict_proba(xv)[:, 1], classifier.predict_proba(xs)[:, 1]
    prob_cutoffs = {"classifier": val_f1_cutoff((yv > peak).to_numpy(), val_cprob),
                    "quantile": val_f1_cutoff((yv > peak).to_numpy(), val_qprob)}

    # 기준선 대표: 검증 전체 MAE 최소 (시험 전에 고정)
    seasonal_mae = {v: float(mean_absolute_error(yv, naive_predict(xv, v))) for v in cfg["baselines"]["seasonal"]}
    persistence_mae = {v: float(mean_absolute_error(yv, persistence_predict(xv, v))) for v in cfg["baselines"]["persistence"]}
    seasonal_sel = min(seasonal_mae, key=seasonal_mae.get)
    persistence_sel = min(persistence_mae, key=persistence_mae.get)
    write(pd.DataFrame([{"family": "계절 나이브", "candidate": k, "validation_mae": v, "selected": k == seasonal_sel}
                        for k, v in seasonal_mae.items()] +
                       [{"family": "persistence", "candidate": k, "validation_mae": v, "selected": k == persistence_sel}
                        for k, v in persistence_mae.items()]), tab / "t2_validation_selection.csv")

    frame = pd.DataFrame(index=xs.index)
    frame["y"] = ys
    for version in ("day", "week", "average"):
        frame[version] = naive_predict(xs, version)
    frame["naive"] = frame[seasonal_sel]
    frame["lgb"] = model.predict(xs)
    for i, q in enumerate(quantiles):
        frame[f"q{int(q * 100)}"] = test_q[:, i]
    frame["classifier_prob"] = test_cprob
    frame["quantile_prob"] = test_qprob
    frame["climate_prob"] = climatology(mt, (yt > peak).to_numpy(), ms)
    frame["target_time"] = ms["target_time"]
    frame["date"] = ms["date"]
    frame["production_target"] = ms["production_target"]
    frame["persistence"] = persistence_predict(xs, persistence_sel)
    frame["seasonal"] = frame["naive"]

    log("3/9 테스트 평가 (날짜 블록 부트스트랩)")
    ratios = np.asarray(cfg["cost_loss_ratios"], dtype=float)
    point05, ci05 = block_bootstrap(frame, lambda f: metric_bundle(f, peak, prob_cutoffs, ratios, quantiles), boot, seed)

    val_preds = {"seasonal": naive_predict(xv, seasonal_sel), "persistence": persistence_predict(xv, persistence_sel),
                 "lgb": model.predict(xv)}
    event_cutoffs = {name: val_f1_cutoff((yv > peak).to_numpy(), p) for name, p in val_preds.items()}
    for name in MODEL_LABELS:
        frame[name + "_alarm"] = frame[name] >= event_cutoffs[name]
    frame["actual_peak"] = frame["y"] > peak
    stats = {name: daily_stats(frame, peak, event_cutoffs[name], name) for name in MODEL_LABELS}
    point, ci, comparison = paired_stats_bootstrap(stats, "persistence", "lgb", boot, seed)

    rows = []
    for name, label in MODEL_LABELS.items():
        total = stats[name].drop(columns="date").sum(numeric_only=True)
        row = {"model": name, "label": label, "event_cutoff": event_cutoffs[name]}
        for key in STAT_KEYS:
            row[key], row[key + "_ci_low"], row[key + "_ci_high"] = point[name][key], ci[name][key][0], ci[name][key][1]
        for unit in ("position", "episode", "day"):
            for kind in ("tp", "fp", "fn"):
                row[f"{unit}_{kind}"] = int(total[f"{unit}_{kind}"])
        rows.append(row)
    final_test = pd.DataFrame(rows)
    write(final_test, tab / "final_test.csv")
    write(pd.DataFrame([comparison]), tab / "t2_improvement_vs_persistence.csv")
    plot_rows = []
    for _, r in final_test.iterrows():
        for metric in ("peak_position_mae", "episode_f1"):
            plot_rows.append({"label": r["label"], "metric": metric, "value": r[metric],
                              "ci_low": r[metric + "_ci_low"], "ci_high": r[metric + "_ci_high"]})
    viz.model_comparison(pd.DataFrame(plot_rows), fig / "f2_model_comparison.png")

    prob_rows = [{"metric": k, "value": v, "ci_low": ci05[k][0], "ci_high": ci05[k][1]} for k, v in point05.items()
                 if not k.startswith("rev_")]
    write(pd.DataFrame(prob_rows), tab / "t2_point_probabilistic_metrics.csv")
    rev_table = pd.DataFrame([{"cost_loss_ratio": r, "rev": point05[f"rev_{i}"], "ci_low": ci05[f"rev_{i}"][0],
                               "ci_high": ci05[f"rev_{i}"][1]} for i, r in enumerate(ratios)])
    write(rev_table, tab / "t4_rev.csv")

    targets = cfg["reproduction_targets"]
    repro = {"persistence_peak_position_mae": point["persistence"]["peak_position_mae"],
             "lgb_peak_position_mae": point["lgb"]["peak_position_mae"]}
    repro["passed"] = all(abs(round(repro[k], 2) - targets[k]) <= targets["tolerance"]
                          for k in ("persistence_peak_position_mae", "lgb_peak_position_mae"))
    log(f"재현 점검: persistence {repro['persistence_peak_position_mae']:.2f}, "
        f"LightGBM {repro['lgb_peak_position_mae']:.2f} → {'통과' if repro['passed'] else '불일치'}")

    # 예측결과 파일 (가정 A7)
    out_cols = ["target_time", "y", "seasonal", "persistence", "lgb", "q10", "q50", "q90", "q95",
                "classifier_prob", "quantile_prob", "lgb_alarm", "actual_peak"]
    frame[out_cols].to_csv(pred_dir / "test_predictions_1h.csv", index_label="forecast_origin", encoding="utf-8-sig")
    viz.forecast_week(frame, peak, fig / "f2_test_forecast_week.png")
    viz.reliability(frame, peak, fig / "f2_reliability.png")
    viz.rev_curve(ratios, rev_table["rev"], rev_table["ci_low"], rev_table["ci_high"], fig / "f4_rev.png")

    # 피크 정의 민감도 (가정 A5). 모델은 그대로, 경계와 검증 임계값만 다시 계산
    sens_rows = []
    for q in cfg["sensitivity"]["peak_quantiles"]:
        pk = float(yt.quantile(q))
        cuts = {name: val_f1_cutoff((yv > pk).to_numpy(), p) for name, p in val_preds.items()}
        st = {name: daily_stats(frame, pk, cuts[name], name) for name in MODEL_LABELS}
        pt, ct, comp = paired_stats_bootstrap(st, "persistence", "lgb", boot, seed)
        for name in MODEL_LABELS:
            sens_rows.append({"peak_quantile": q, "threshold": pk, "model": name,
                              "test_peak_positions": int(st[name]["peak_n"].sum()),
                              **{k: pt[name][k] for k in ("peak_position_mae", "episode_f1", "position_f1")},
                              "peak_position_mae_ci_low": ct[name]["peak_position_mae"][0],
                              "peak_position_mae_ci_high": ct[name]["peak_position_mae"][1],
                              "episode_f1_ci_low": ct[name]["episode_f1"][0], "episode_f1_ci_high": ct[name]["episode_f1"][1],
                              "lgb_improvement_vs_persistence": comp["peak_mae_improvement"],
                              "improvement_ci_low": comp["ci95"][0], "improvement_ci_high": comp["ci95"][1]})
    write(pd.DataFrame(sens_rows), tab / "t2_peak_definition_sensitivity.csv")

    # ---------------- 4. 개발 구간 분석 (시험 미사용) ----------------
    log("4/9 rolling-origin·예측거리 곡선·conformal (개발 폴드)")
    write(pd.DataFrame(rolling_origin(cfg, x, y, meta)), tab / "t2_rolling_origin.csv")
    curve = horizon_curve(cfg, series)
    write(curve, tab / "t2_horizon_curve_dev.csv")
    viz.horizon(curve, fig / "f2_horizon_curve.png")
    conf_summary, conf_offsets = conformal_dev(cfg, x, y, meta)
    write(conf_summary, tab / "t2_conformal_dev.csv")
    write(conf_offsets, tab / "t2_conformal_offsets.csv")

    log("5/9 익일 일간 최대 (보조 과제)")
    daily, daily_test = daily_max_task(cfg, series, raw)
    write(pd.DataFrame([{"metric": k, "value": v, "ci_low": daily["ci95"][k][0], "ci_high": daily["ci95"][k][1]}
                        for k, v in daily["metrics"].items()]), tab / "t2_daily_max.csv")
    daily_test.drop(columns=["date"]).to_csv(pred_dir / "test_predictions_daily_max.csv",
                                             index_label="forecast_origin_day", encoding="utf-8-sig")

    # ---------------- 3장. 오류·영향요인·피크 조건 ----------------
    log("6/9 조건별 FN·FP, 영향변수")
    train_prod = mt["production_target"]
    cond_classifier = errors.error_conditions(frame, peak, (frame["classifier_prob"] >= prob_cutoffs["classifier"]).to_numpy(),
                                              train_prod, cfg, n=boot, seed=seed)
    write(cond_classifier, tab / "t3_error_conditions_classifier.csv")
    cond_lgb = errors.error_conditions(frame, peak, frame["lgb_alarm"].to_numpy(), train_prod, cfg, n=boot, seed=seed)
    write(cond_lgb, tab / "t3_error_conditions_lgb.csv")
    viz.conditions_by_hour(cond_lgb, fig / "f3_conditions_by_hour.png")
    cases = errors.evening_miss_cases(frame, peak, "lgb_alarm")
    write(pd.DataFrame(cases), tab / "t3_evening_miss_cases.csv")
    viz.evening_cases(frame, cases, peak, fig / "f3_evening_miss_cases.png")
    imp = importance.permutation_importance(model, xs, ys, peak, seed=seed)
    write(imp, tab / "t3_permutation_importance.csv")
    viz.importance(imp, fig / "f3_importance.png")
    write(importance.interaction_pdp(model, xs), tab / "t3_pdp_hour_production.csv", index=True)

    log("7/9 피크 발생 조건 (시험 이전 구간)")
    pre_test_end = mv["target_time"].max()
    positions = peak_conditions.position_table(series, raw, pre_test_end, peak, cfg, mt["target_time"].max())
    write(peak_conditions.rate_by_condition(positions, boot, seed), tab / "t3_peak_conditions.csv")
    write(peak_conditions.interaction_table(positions), tab / "t3_peak_weekday_shift.csv", index=True)
    write(peak_conditions.peak_vs_nonpeak(positions, boot, seed), tab / "t3_peak_vs_nonpeak.csv")
    write(peak_conditions.episode_table(positions), tab / "t3_peak_episodes_pretest.csv")
    viz.peak_heatmap(positions, fig / "f3_peak_heatmap.png")
    viz.calendar_profiles(profiles, positions, fig / "f1_calendar_profiles.png")
    viz.series_overview(series, peak, {"검증 시작": mv["target_time"].min(), "테스트 시작": ms["target_time"].min()},
                        fig / "f1_series_overview.png")

    # ---------------- 4장. 경보 운영·저감 시뮬레이션 ----------------
    log("8/9 경보 규칙 (검증에서 선택 → 테스트 적용)")
    val_frame = pd.DataFrame({"y": yv.to_numpy(), "lgb": val_preds["lgb"], "q95": val_q[:, quantiles.index(0.95)],
                              "target_time": mv["target_time"].to_numpy(), "date": mv["date"].to_numpy()}, index=xv.index)
    alert_table = alerting.evaluate_rules({"validation": val_frame, "test": frame}, peak, event_cutoffs["lgb"],
                                          cfg["alerting"]["rules"])
    selected_rule = alerting.select_rule(alert_table)
    alert_table["selected_on_validation"] = (alert_table["stage"] == "조치") & (alert_table["rule"] == selected_rule)
    write(alert_table, tab / "t4_alert_rules.csv")
    viz.alert_flow(fig / "f4_alert_flow.png")

    log("9/9 생산량 이동 시뮬레이션 (가정 기반)")
    hourly = simulate_shift.hourly_frame(series)
    train_hourly = hourly[hourly.index < mt["target_time"].max()]
    slope = simulate_shift.fit_slope(train_hourly, boot, seed)
    write(pd.DataFrame([slope]), tab / "t4_shift_slope.csv")
    train_positions = positions[positions.index <= mt["target_time"].max()]
    by_hour = train_positions.groupby("hour")["peak"].mean()
    static_hours = set(int(h) for h in by_hour.index[by_hour >= 2 * train_positions["peak"].mean()])
    test_series = series.loc[(series.index >= ms["target_time"].min()) & (series.index <= ms["target_time"].max())]
    test_series = test_series.loc[~test_series["recovered"] & test_series["power"].notna()]
    targets_by_trigger = simulate_shift.shift_targets(frame, peak, event_cutoffs["lgb"], static_hours)
    forecast_plan = pd.Series(frame["lgb"].to_numpy(), index=frame["target_time"].to_numpy())
    plans = {"forecast": forecast_plan, "static": forecast_plan, "oracle": test_series["power"]}
    sim_rows = []
    for trigger, shift_hours in targets_by_trigger.items():
        for case, b in (("point", slope["slope"]), ("ci_low", slope["ci_low"]), ("ci_high", slope["ci_high"])):
            for fraction in cfg["shift_simulation"]["fractions"]:
                r = simulate_shift.simulate(test_series, plans[trigger], shift_hours, b, fraction, peak,
                                            cfg["shift_simulation"]["cap_ratio"])
                monthly = r.pop("monthly_max")
                r.update(trigger=trigger, slope_case=case,
                         exceed_reduction_share=1 - r["exceed_after"] / r["exceed_before"] if r["exceed_before"] else np.nan,
                         max_reduction_share=1 - r["max_after"] / r["max_before"],
                         monthly_max=json.dumps(monthly, ensure_ascii=False))
                sim_rows.append(r)
    sim_table = pd.DataFrame(sim_rows)
    write(sim_table, tab / "t4_shift_simulation.csv")
    viz.shift_simulation(sim_table, fig / "f4_shift_simulation.png")

    summary = {
        "data": info, "font": font, "config": _relative(cfg["_config_path"]),
        "one_hour": {"rows": {k: len(v[0]) for k, v in split.items()},
                     "range": {k: [str(v[0].index[0]), str(v[0].index[-1])] for k, v in split.items()},
                     "train_peak_threshold": peak,
                     "events": {k: int((v[1] > peak).sum()) for k, v in split.items()},
                     "seasonal_selected": seasonal_sel, "persistence_selected": persistence_sel,
                     "event_cutoffs": event_cutoffs, "probability_cutoffs": prob_cutoffs,
                     "quantile_crossings": {"validation": val_cross, "test": test_cross},
                     "improvement_vs_persistence": comparison},
        "reproduction_check": repro,
        "alert_rule_selected_on_validation": selected_rule,
        "shift_static_hours": sorted(static_hours), "shift_slope": slope,
        "daily_max": {k: v for k, v in daily.items() if k not in ("metrics", "ci95")},
        "zero_value_runs": int(len(zero_runs)),
        "bootstrap_draws": boot, "seed": seed, "runtime_seconds": round(time.time() - started, 1),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: metadata.version(p) for p in ("pandas", "numpy", "scikit-learn", "lightgbm",
                                                            "matplotlib", "PyYAML")}},
    }
    (dirs["logs"] / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    log(f"완료: {summary['runtime_seconds']}초, 출력 {_relative(str(dirs['base']))}")
    return summary


if __name__ == "__main__":
    main()
