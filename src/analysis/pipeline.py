"""Run all approved development-only post-hoc analyses from one entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import configure, save
from ._common import canonical_history, canonical_predictions, selected_rows
from .alerting import run_alerting
from .applicability import run_applicability
from .decision import run_decision
from .energy_baseline import run_energy_baseline
from .errors import run_errors
from .horizon_curve import run_horizon_curve
from .importance import run_importance
from .simulate_shift import run_simulate_shift
from .supplemental import run_supplemental
from .tariff import classify_tariff, read_tariff, run_tariff


def _selected_map(predictions: pd.DataFrame, selection: dict | None) -> dict[int, str]:
    by_h = (selection or {}).get("by_horizon", {})
    result = {}
    for h in sorted(predictions.horizon.unique()):
        choice = by_h.get(str(int(h)), by_h.get(int(h), {}))
        candidate = choice.get("point_model") if isinstance(choice, dict) else None
        result[int(h)] = candidate or ("lgbm" if "lgbm" in set(predictions.loc[predictions.horizon.eq(h), "model"]) else
                                       str(predictions.loc[predictions.horizon.eq(h), "model"].iloc[0]))
    return result


def _verified_frozen_predictions(record_path: str | Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    record_path = Path(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError("Frozen analysis requires a completed freeze record")
    deadline = cfg.get("freeze", {}).get("not_before")
    if deadline and pd.Timestamp.now(tz="Asia/Seoul") < pd.Timestamp(deadline):
        raise ValueError("Frozen analysis is date-locked")
    base = record_path.parent.parent
    files = {
        "holdout_prediction_sha256": base/"predictions"/"final_test_predictions.csv",
        "final_metrics_sha256": base/"tables"/"final_test.csv",
        "model_sha256": base/"models"/"frozen_final.joblib",
    }
    for field, path in files.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get(field):
            raise ValueError(f"Frozen artifact missing or hash mismatch: {path}")
    frame = pd.read_csv(files["holdout_prediction_sha256"], parse_dates=["origin", "target_time"])
    boundary = pd.Timestamp(record["test_start_origin"])
    if frame.empty or pd.to_datetime(frame.origin).lt(boundary).any():
        raise ValueError("Frozen predictions include non-test origins")
    return frame, record


def _enrich_with_quantiles(point: pd.DataFrame, predictions: pd.DataFrame,
                           selection: dict | None, horizon: int) -> pd.DataFrame:
    """Keep selected point forecast and attach the selected calibrated risk output."""
    choice = (selection or {}).get("by_horizon", {}).get(str(horizon), {})
    method = choice.get("conformal", "a") if isinstance(choice, dict) else "a"
    risk_name = f"lgbm_quantile_{method}"
    risk = predictions.loc[predictions.horizon.eq(horizon) & predictions.model.eq(risk_name)]
    if risk.empty:
        return point.assign(risk_model=pd.NA)
    keys = ["origin", "target_time", "horizon", "fold"]
    columns = [c for c in ("q10", "q50", "q90", "q95", "q975", "q90_cal", "q95_cal",
                           "q975_cal", "p_exceed", "exp_exceed", "alert") if c in risk]
    if risk.duplicated(keys).any():
        raise ValueError("Selected quantile risk output has duplicate OOF keys")
    left = point.drop(columns=[c for c in columns if c in point and c != "alert"])
    if "alert" in left:
        left = left.rename(columns={"alert": "point_alert"})
    enriched = left.merge(risk[keys+columns], on=keys, how="left", validate="one_to_one")
    if enriched[columns].isna().all(axis=1).any():
        raise ValueError("Selected point/risk OOF rows do not align")
    enriched["risk_model"] = risk_name
    return enriched


def _plot_analysis(outdir: Path) -> dict:
    """Figures are generated only from saved, computed tables."""
    configure()
    tables, figures = outdir/"tables", outdir/"figures"
    paths = {}
    conditions = tables/"error_conditions.csv"
    if conditions.exists():
        x = pd.read_csv(conditions)
        x = x.loc[x.factor.isin(["time_block", "weekday", "month"])]
        if not x.empty:
            fig, axes = plt.subplots(3, 2, figsize=(11, 7), constrained_layout=True)
            factors = [("time_block", "시간대"), ("weekday", "요일"), ("month", "월")]
            weekdays = {"Monday":"월", "Tuesday":"화", "Wednesday":"수", "Thursday":"목", "Friday":"금", "Saturday":"토", "Sunday":"일"}
            for row_index, (factor, factor_label) in enumerate(factors):
                group = x.loc[x.factor.eq(factor)].copy()
                if factor == "weekday":
                    group = group.set_index("value").reindex(list(weekdays)).reset_index()
                labels = group.value.map(weekdays) if factor == "weekday" else group.value
                for col_index, (col, title) in enumerate([("fn", "미탐 위치"), ("fp", "오경보 위치")]):
                    ax = axes[row_index, col_index]
                    values = group[col].fillna(0).to_numpy(float)
                    maximum = max(float(x[col].max()), 1)
                    ax.imshow(values[None, :], aspect="auto", cmap="YlOrBr", vmin=0, vmax=maximum)
                    ax.set_xticks(range(len(labels)), labels, fontsize=9)
                    ax.set_yticks([0], [factor_label])
                    ax.set_title(title if row_index == 0 else "", fontsize=11)
                    for index, value in enumerate(values):
                        ax.text(index, 0, f"{int(value)}", ha="center", va="center",
                                color="white" if value > maximum*.6 else "#1C2A39")
            fig.suptitle("F3-1 조건별 미탐·오경보 위치 수 · 선택된 위험 경보")
            paths["condition_heatmap"] = str(save(fig, figures/"condition_fn_fp_heatmap.png"))
    rev = tables/"relative_economic_value.csv"
    if rev.exists():
        x = pd.read_csv(rev)
        x = x.loc[x.horizon.eq(4)]
        if not x.empty:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(x.cl_ratio, x.rev, marker="o", color="#3E6283")
            if x.ci_low.notna().any():
                ax.fill_between(x.cl_ratio, x.ci_low, x.ci_high, color="#DCE6EF")
            ax.axhline(0, color="#7D8894", linewidth=.8)
            ax.set(xlabel="조치 비용 / 피크 손실 가정", ylabel="상대 경제가치", title="개발구간 비용·손실 민감도")
            ax.grid(True)
            paths["rev_curve"] = str(save(fig, figures/"relative_economic_value.png"))
    peak_types = tables/"peak_type_summary.csv"
    if peak_types.exists():
        x = pd.read_csv(peak_types)
        if not x.empty:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            bars = ax.bar(x.type.replace({"production_explained": "생산 설명형", "excess_residual": "초과 잔차형"}),
                   x.episodes, color=["#3E6283", "#B8741A"][:len(x)])
            ax.bar_label(bars, padding=4)
            ax.set_ylim(0, max(x.episodes.max()*1.15, 1))
            ax.set(ylabel="피크 에피소드 수", title="사후 에너지 기준선에 따른 피크 유형")
            paths["peak_types"] = str(save(fig, figures/"peak_type_distribution.png"))
    alert_rules = tables/"alert_rules.csv"
    if alert_rules.exists():
        x = pd.read_csv(alert_rules)
        x = x.loc[x.prep_minutes.eq(30) & x.ratio.eq(.1)]
        if not x.empty:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            colors = {"1/1": "#3E6283", "2/2": "#B8741A", "2/3": "#2B7564"}
            for (rule, horizon), group in x.groupby(["rule", "horizon"]):
                ax.scatter(group.false_alarm_positions_per_operating_hour,
                           group.mean_actionable_lead_minutes, label=f"{rule} / {horizon*15}분", s=60,
                           marker="o" if horizon == 4 else "^", color=colors[rule])
            ax.set(xlabel="예측 평가시간당 오경보 위치", ylabel="적중 에피소드 평균 유효 선행시간(분)",
                   title="확인 규칙과 예측거리 · C/L=0.1, 준비 30분")
            ax.legend(title="규칙 / 예측거리", ncol=2, fontsize=8)
            ax.grid(True)
            paths["alert_rules"] = str(save(fig, figures/"alert_rule_tradeoff.png"))
    importance = tables/"permutation_importance_top10.csv"
    if importance.exists():
        x = pd.read_csv(importance)
        if not x.empty:
            horizons = sorted(x.horizon.unique())
            columns = min(2, len(horizons))
            fig, axes = plt.subplots((len(horizons)+columns-1)//columns, columns,
                                     figsize=(13, 4.7*((len(horizons)+columns-1)//columns)), squeeze=False)
            for ax, horizon in zip(axes.flat, horizons):
                group = x.loc[x.horizon.eq(horizon)].sort_values("mae_increase").tail(10)
                ax.barh(group.feature, group.mae_increase, color="#3E6283")
                role = "선택된 점예측 모델"
                if "attribution_role" in group and group.attribution_role.eq("development_comparator_not_selected").any():
                    role = "비선정 LightGBM 비교모델"
                ax.set_title(f"{horizon*15}분 전 · {role}", fontsize=11)
                ax.tick_params(axis="y", labelsize=9)
                ax.set_xlabel("순열 후 MAE 증가")
            for ax in list(axes.flat)[len(horizons):]:
                ax.set_visible(False)
            fig.suptitle("예측거리별 주요 변수 · 검증 폴드")
            fig.tight_layout()
            paths["importance"] = str(save(fig, figures/"horizon_importance.png"))
    tariff = tables/"tariff_counterfactual_summary.csv"
    if tariff.exists():
        x = pd.read_csv(tariff)
        if "weighted_missed_peak_share" in x and x.weighted_missed_peak_share.notna().any():
            pivot = x.pivot(index="band", columns="tariff_year", values="weighted_missed_peak_share")
            ax = pivot.plot.bar(figsize=(6, 3.6), color=["#7D8894", "#B8741A"])
            ax.set(ylabel="미탐 피크 가중 비중", title="2021 대 2026 시간대 가중 시나리오")
            ax.legend(title="분류 연도")
            paths["tariff_comparison"] = str(save(ax.figure, figures/"tariff_counterfactual.png"))
    shift = tables/"shift_summary.csv"
    if shift.exists():
        x = pd.read_csv(shift)
        if not x.empty:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            if x.applied_source_hours.sum() == 0:
                ax.axis("off")
                ax.text(.02, .83, "조건을 충족한 이동 0시간", fontsize=24, weight="bold", color="#3E6283", transform=ax.transAxes)
                ax.text(.02, .55, "경보 시점·준비시간·목적지 조건을 함께 만족하지 못했습니다.", fontsize=12, transform=ax.transAxes)
                ax.text(.02, .35, "10·20·30% 이동 모두 적용 사례 없음", fontsize=14, transform=ax.transAxes)
                ax.text(.02, .13, "월 최대 변화 0은 무조치 결과이며, 이동 효과가 없다는 추정이 아닙니다.", fontsize=11, color="#4F5F70", transform=ax.transAxes)
                ax.set_title("F4-5 시간 선후 제약을 적용한 부하 이동 시나리오", loc="left", pad=18)
            else:
                for fraction, group in x.groupby("fraction"):
                    ax.plot(group.month, group.max_change, marker="o", label=f"{fraction:.0%} 이동")
                ax.axhline(0, color="#7D8894", linewidth=.8)
                ax.set(ylabel="월 최대 15분 전력 변화 (원자료 단위)", title="가상 부하 이동 민감도")
                ax.legend()
                ax.grid(True)
            paths["shift_scenario"] = str(save(fig, figures/"shift_scenario.png"))
    return paths


def run_analysis(df: pd.DataFrame, predictions: pd.DataFrame, cfg: dict,
                 outdir: str | Path, *, train_df: pd.DataFrame | None = None,
                 tariff_2021: dict | str | Path | None = None,
                 tariff_2026: dict | str | Path | None = None,
                 selection: dict | None = None,
                 model_dir: str | Path | None = None,
                 scope: str = "development_oof",
                 frozen_record: str | Path | None = None) -> dict:
    """Analyse only rolling-origin development predictions, never held-out test rows.

    `train_df` must end before the first OOF target. All fitted explanatory
    quantities use this prefix; OOF targets are used for descriptive scoring.
    """
    if scope not in ("development_oof", "frozen_test"):
        raise ValueError(f"Unknown analysis scope: {scope}")
    verified_record = None
    if scope == "frozen_test":
        if frozen_record is None or selection is None:
            raise ValueError("Frozen analysis requires the completed freeze record and development selection")
        predictions, verified_record = _verified_frozen_predictions(frozen_record, cfg)
    outdir = Path(outdir)
    if scope == "frozen_test" and outdir.name != "final_analysis":
        outdir = outdir/"final_analysis"
    history = canonical_history(df)
    train = canonical_history(train_df) if train_df is not None else history.loc[history.index < pd.to_datetime(predictions.target_time).min()]
    oof = canonical_predictions(predictions)
    if not oof.target_time.isin(history.index).all():
        raise ValueError("OOF target outside supplied development history")
    if scope == "development_oof" and cfg.get("split", {}).get("test_start_origin"):
        boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
        if oof.origin.ge(boundary).any():
            raise ValueError("Test origins require explicit frozen_test scope and verified freeze record")
    if train.empty or train.index.max() >= oof.target_time.min():
        raise ValueError("Energy/applicability training prefix must precede every OOF target")
    if "split" in oof and oof["split"].astype(str).str.contains("test", case=False).any():
        if scope != "frozen_test":
            raise ValueError("Held-out test predictions cannot enter development analysis")
    selected = _selected_map(oof, selection)
    if scope == "frozen_test":
        by_h = selection.get("by_horizon", {})
        for h in selected:
            chosen_name = by_h.get(str(h), {}).get("point_model")
            if not chosen_name or chosen_name != selected[h]:
                raise ValueError(f"No frozen development model selection for h={h}")
    reps = ({int(h): {key: value.get(key) for key in ("persistence", "seasonal", "cbl")}
             for h, value in (selection or {}).get("by_horizon", {}).items()} if selection else None)
    chosen = pd.concat([_enrich_with_quantiles(selected_rows(oof, h, model), oof, selection, h)
                        for h, model in selected.items()], ignore_index=True)
    primary_h = int(cfg.get("primary_horizon", 4))
    primary = chosen.loc[chosen.horizon.eq(primary_h)].copy().sort_values("target_time")
    t21, t26 = read_tariff(tariff_2021), read_tariff(tariff_2026)
    results = {"scope": scope, "selected_models": selected,
               "horizon_curve": run_horizon_curve(oof, outdir, cfg, selected, reps)}
    if verified_record is not None:
        results["freeze_record"] = str(frozen_record)
        results["freeze_model_sha256"] = verified_record["model_sha256"]
    if t21:
        history["tariff_2021_band"] = classify_tariff(history.index, t21).to_numpy()
    results["errors"] = run_errors(primary, history, outdir)
    energy, baseline = run_energy_baseline(primary, history, train, outdir, cfg)
    results["energy_baseline"] = energy
    tariff, _, t26 = run_tariff(primary, outdir, t21, t26)
    results["tariff"] = tariff
    bands = None
    if t26:
        all_times = pd.DatetimeIndex(chosen.target_time.drop_duplicates())
        bands = classify_tariff(all_times, t26)
    results["alerting"] = run_alerting(chosen, outdir, cfg, bands, scope=scope)
    results["decision"] = run_decision(primary, outdir, cfg)
    excess_adopted = bool((selection or {}).get("adoption", {}).get(
        f"h{primary_h}_expected_exceedance", {}).get("adopted", False))
    results["supplemental"] = run_supplemental(
        primary, outdir, cfg, scope, expected_exceedance_adopted=excess_adopted)
    results["shift"] = run_simulate_shift(primary, history, train, baseline, t26, outdir, cfg)
    results["applicability"] = run_applicability(history, train, primary, outdir)
    results["importance"] = (run_importance(history, oof, cfg, outdir,
                           Path(model_dir) if model_dir else None, selected)
                           if scope == "development_oof" else
                           {"status": "development_only", "reason": "Frozen-test feature attribution is not fitted or reselected; use development importance."})
    results["figures"] = _plot_analysis(outdir)
    summary = outdir/"logs"/"analysis_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    results["summary_path"] = str(summary)
    return results
