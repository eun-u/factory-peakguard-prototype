"""Preregistered, development-only evening risk and production-oracle audit.

The oracle knows actual production at the future target interval. This is a
perfect observed-production availability scenario, not an operational input
or a guaranteed bound on forecast performance. Original model selection and
risk alerts remain unchanged; no frozen-test value is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from scripts.diagnose_h96_0924 import _assert_development_frame, _boundary, _checked_output
from scripts.peak_sensitivity_0924 import REFERENCE_QUANTILE, _at_tau, _base_fold
from src.analysis.errors import evening_misses, match_episode_table
from src.analysis.pipeline import _enrich_with_quantiles
from src.bootstrap import paired_mae_improvement
from src.evaluate import score_predictions
from src.models.lgbm_point import fit_point
from src.session_data import load_development_history, load_development_oof
from src.training import _cutoff, _row


HORIZON = 4
POINT_NAME = "lgbm_no_holiday_weight_2"
ORACLE_FEATURE = "target_production_observed"
HOLIDAY_COLUMNS = ("is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day")
N_BOOT = 1000
SCENARIO_LABEL = "사후 관측 생산량 기반 사후 가정 시나리오"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_a_parity(root: Path, oof_path: Path, manifest_path: Path) -> dict:
    """Use the completed A audit as independent original-cohort q95 evidence."""
    path = root / "outputs/logs/peak_sensitivity_0924.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (not audit.get("original_selection_unchanged") or
            audit.get("q95_refit_parity", {}).get("models_checked") != 39 or
            not audit["q95_refit_parity"].get("all_alerts_exact") or
            audit["q95_refit_parity"].get("max_abs_difference", float("inf")) > 1e-10 or
            audit.get("sources_sha256", {}).get("development_oof") != _digest(oof_path) or
            audit.get("sources_sha256", {}).get("development_selection") != _digest(manifest_path)):
        raise AssertionError("Original-cohort q95 parity evidence is absent or stale")
    return {"path": str(path.relative_to(root)), "sha256": _digest(path),
            "models_checked": audit["q95_refit_parity"]["models_checked"],
            "max_abs_difference": audit["q95_refit_parity"]["max_abs_difference"]}


def _target_production(df: pd.DataFrame, context: dict) -> pd.Series:
    """Attach deliberately future, contemporaneous *observed* production."""
    target_times = pd.DatetimeIndex(context["targets"].target_time)
    values = pd.to_numeric(df["production_target"].reindex(target_times), errors="coerce")
    return pd.Series(values.to_numpy(dtype=float), index=context["all_origins"], name=ORACLE_FEATURE)


def _common_cohort(context: dict, production: pd.Series) -> dict[str, pd.DatetimeIndex]:
    available = production.index[np.isfinite(production.to_numpy(dtype=float))]
    common = {name: context[name].intersection(available)
              for name in ("fit", "stop", "cal", "score")}
    if min(map(len, common.values())) < 30:
        raise AssertionError("The common observed-production cohort is too small")
    return common


def _excluded_score_peaks(context: dict, cohort: dict) -> int:
    missing = context["score"].difference(cohort["score"])
    target = context["targets"].loc[missing]
    return int(target.y.gt(context["tau"]).sum())


def _fit_point_pair(context: dict, production: pd.Series, cfg: dict,
                    cohort: dict[str, pd.DatetimeIndex]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same params/weights/rows; oracle adds exactly one post-hoc feature."""
    x, targets, tau, expected = (context[key] for key in ("x", "targets", "tau", "expected"))
    fit, stop, cal, score = (cohort[key] for key in ("fit", "stop", "cal", "score"))
    features = [column for column in x if column not in HOLIDAY_COLUMNS]
    x_point = x[features]
    x_oracle = x_point.assign(**{ORACLE_FEATURE: production.reindex(x.index).to_numpy(dtype=float)})
    if x_oracle.columns.tolist() != features + [ORACLE_FEATURE]:
        raise AssertionError("The oracle feature set differs by more than observed production")
    predictions = []
    for name, matrix in (("baseline_point_common", x_point), ("oracle_point_common", x_oracle)):
        model = fit_point(matrix.loc[fit], targets.loc[fit, "y"],
                          matrix.loc[stop], targets.loc[stop, "y"], cfg,
                          peak_threshold=tau, peak_weight=2.0, params=expected["chosen_params"])
        cal_pred = model.predict(matrix.loc[cal])
        cutoff = _cutoff(targets.loc[cal, "y"].to_numpy(dtype=float), cal_pred,
                         tau, targets.loc[cal, "target_time"])
        scored = _row(score, HORIZON, context["fold"], name, targets, tau,
                      model.predict(matrix.loc[score]), cutoff)
        scored["reference_only"] = True
        scored["selection_used"] = False
        scored["scenario_label"] = SCENARIO_LABEL
        predictions.append(scored)
    return predictions[0], predictions[1]


def _point_oof_parity(baseline: pd.DataFrame, oof: pd.DataFrame,
                      context: dict, cohort: dict) -> dict:
    """Assert original point parity whenever the common fit/stop rows match."""
    fold = context["fold"]
    old = oof.loc[(oof.horizon == HORIZON) & (oof.fold == fold) &
                  (oof.model == POINT_NAME)].sort_values("origin")
    fresh = baseline.sort_values("origin")
    old = old.set_index("origin").loc[pd.DatetimeIndex(fresh.origin)]
    difference = float(np.max(np.abs(fresh.pred.to_numpy(dtype=float) - old.pred.to_numpy(dtype=float))))
    alert_equal = bool(np.array_equal(fresh.alert.to_numpy(dtype=bool), old.alert.to_numpy(dtype=bool)))
    same_train = cohort["fit"].equals(context["fit"]) and cohort["stop"].equals(context["stop"])
    same_cal = cohort["cal"].equals(context["cal"])
    if same_train and same_cal and (difference > 1e-10 or not alert_equal):
        raise AssertionError(f"Fresh common-cohort point baseline did not reproduce original fold {fold}")
    return {"fold": fold, "same_fit_stop_as_original": same_train,
            "same_cal_as_original": same_cal, "max_abs_difference": difference,
            "alerts_exact": alert_equal}


def _risk_events(oof: pd.DataFrame, selection: dict, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    choice = selection["by_horizon"][str(HORIZON)]
    point = oof.loc[(oof.horizon == HORIZON) & (oof.model == choice["point_model"])].copy()
    primary = _enrich_with_quantiles(point, oof, selection, HORIZON).sort_values("target_time")
    risk_name = f"lgbm_quantile_{choice['conformal']}"
    if primary.empty or not primary.risk_model.eq(risk_name).all():
        raise AssertionError("Selected h4 risk layer was not reconstructed")
    primary["alert_flag"] = primary.alert.astype(bool)
    events = match_episode_table(primary)
    details, _ = evening_misses(primary, history, events)
    if details.empty:
        raise AssertionError("No evening peak episodes were reconstructed")
    if not details.status.isin(["FN", "TP"]).all():
        raise AssertionError("Retrospective evening cases must be actual peak episodes")
    return primary, events, details


def _enrich_evening_context(details: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = details.copy()
    temperature_means = []
    for start in pd.DatetimeIndex(cases.start):
        before = history.loc[(history.index >= start - pd.Timedelta(hours=4)) & (history.index < start)]
        if len(before) != 16:
            raise AssertionError(f"Evening case has an incomplete prior four-hour history: {start}")
        temperature_means.append(float(pd.to_numeric(before.temperature, errors="coerce").mean()))
    cases["temperature_mean_4h"] = temperature_means
    summary = cases.groupby("status", as_index=False).agg(
        episodes=("start", "size"),
        production_change_4h_median=("production_change_4h", "median"),
        production_change_4h_mean=("production_change_4h", "mean"),
        temperature_mean_4h_median=("temperature_mean_4h", "median"),
        temperature_mean_4h_mean=("temperature_mean_4h", "mean"),
        power_trend_4h_median=("power_trend_4h", "median"))
    weekdays = (cases.groupby(["status", "weekday"], dropna=False).size()
                .rename("episodes").reset_index())
    return cases, summary, weekdays


def _point_case_status(cases: pd.DataFrame, baseline: pd.DataFrame,
                       oracle: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference risk FN/TP starts; point alerts are a separate task."""
    result = cases.copy()
    for name, frame in (("baseline_point", baseline), ("oracle_point", oracle)):
        event_frame = frame.copy().sort_values("target_time")
        event_frame["alert_flag"] = event_frame.alert.astype(bool)
        actual = match_episode_table(event_frame)
        statuses = actual.loc[actual.status.isin(["TP", "FN"])].set_index("start").status
        result[f"{name}_episode_status"] = result.start.map(statuses).fillna("outside_common_cohort")
    result["reference_only"] = True
    result["selection_used"] = False
    result["scenario_label"] = SCENARIO_LABEL
    return result


def _gain_table(baseline: pd.DataFrame, oracle: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for scope, mask in (("all", np.ones(len(baseline), dtype=bool)),
                        ("evening_16_24", pd.DatetimeIndex(baseline.target_time).hour >= 16)):
        left = baseline.loc[mask]
        right = oracle.loc[np.ones(len(oracle), dtype=bool) if scope == "all" else
                           (pd.DatetimeIndex(oracle.target_time).hour >= 16)]
        for peak_only in (False, True):
            gain = paired_mae_improvement(left, right, peak_only=peak_only, n=N_BOOT, seed=seed)
            rows.append({"scope": scope, "peak_only": peak_only,
                         "baseline_minus_oracle_mae": gain["estimate"],
                         "ci95_low": gain["ci95"][0], "ci95_high": gain["ci95"][1],
                         "n_paired": gain["n"],
                         "improvement_established": bool(np.isfinite(gain["ci95"][0]) and gain["ci95"][0] > 0),
                         "reference_only": True, "selection_used": False,
                         "scenario_label": SCENARIO_LABEL})
    return pd.DataFrame(rows)


def _metric_table(baseline: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in (("baseline_point_common", baseline), ("oracle_point_common", oracle)):
        for scope, part in (("all", frame),
                            ("evening_16_24", frame.loc[pd.DatetimeIndex(frame.target_time).hour >= 16])):
            scored = score_predictions(part)
            rows.append({"model": name, "scope": scope,
                         **{key: scored[key] for key in ("n", "peak_n", "mae", "peak_mae",
                                                           "episode_f1", "false_alarms_positions")},
                         "reference_only": True, "selection_used": False,
                         "scenario_label": SCENARIO_LABEL})
    return pd.DataFrame(rows)


def run(root: Path = ROOT, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    output = _checked_output(root, output)
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    boundary = _boundary(cfg)
    history = load_development_history(root)
    _assert_development_frame(history, boundary)
    oof_path = root / "outputs/predictions/development_oof.csv"
    oof = load_development_oof(root)
    manifest_path = root / "outputs/logs/development_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    if selection["by_horizon"][str(HORIZON)]["point_model"] != POINT_NAME:
        raise AssertionError("The original selected h4 point model changed")
    a_evidence = _verify_a_parity(root, oof_path, manifest_path)
    primary, risk_events, details = _risk_events(oof, selection, history)
    cases, condition_summary, weekday_table = _enrich_evening_context(details, history)
    expected = {(int(row["horizon"]), int(row["fold"])): row for row in manifest["folds"]}
    point_rows, oracle_rows, cohort_rows, parity = [], [], [], []
    origins = history.index[history.index < boundary]
    for fold_id in range(int(cfg["split"]["n_folds"])):
        base = _base_fold(history, origins, HORIZON, fold_id, cfg, boundary)
        old = oof.loc[(oof.horizon == HORIZON) & (oof.fold == fold_id) &
                      (oof.model == POINT_NAME)].sort_values("origin")
        context = _at_tau(history, base, cfg, REFERENCE_QUANTILE,
                          expected[(HORIZON, fold_id)], pd.DatetimeIndex(old.origin))
        production = _target_production(history, context)
        cohort = _common_cohort(context, production)
        point, oracle = _fit_point_pair(context, production, cfg, cohort)
        parity.append(_point_oof_parity(point, oof, context, cohort))
        point_rows.append(point)
        oracle_rows.append(oracle)
        cohort_rows.append({"fold": fold_id, "tau": context["tau"],
                            **{f"{name}_original": len(context[name]) for name in cohort},
                            **{f"{name}_common": len(cohort[name]) for name in cohort},
                            **{f"{name}_production_missing": len(context[name]) - len(cohort[name]) for name in cohort},
                            "score_production_missing_peak_positions": _excluded_score_peaks(context, cohort),
                            "reference_only": True, "selection_used": False,
                            "scenario_label": SCENARIO_LABEL})
        print(f"[oracle] fold={fold_id} score={len(cohort['score'])} "
              f"excluded={len(context['score'])-len(cohort['score'])}", flush=True)
    point = pd.concat(point_rows, ignore_index=True)
    oracle = pd.concat(oracle_rows, ignore_index=True)
    if not point[["origin", "target_time", "fold"]].equals(oracle[["origin", "target_time", "fold"]]):
        raise AssertionError("Point and oracle forecasts do not share the same target cohort")
    if (point.origin >= boundary).any() or (point.target_time >= boundary).any():
        raise AssertionError("An oracle scoring row crossed the frozen boundary")
    cases = _point_case_status(cases, point, oracle)
    gains = _gain_table(point, oracle, int(cfg["seed"]))
    metrics = _metric_table(point, oracle)
    condition_summary["reference_only"] = True
    condition_summary["selection_used"] = False
    condition_summary["analysis_role"] = "retrospective_descriptive"
    weekday_table["reference_only"] = True
    weekday_table["selection_used"] = False
    weekday_table["analysis_role"] = "retrospective_descriptive"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    products = {"cases": output / "tables/evening_oracle_0924_cases.csv",
                "conditions": output / "tables/evening_oracle_0924_conditions.csv",
                "weekdays": output / "tables/evening_oracle_0924_weekdays.csv",
                "cohort": output / "tables/evening_oracle_0924_cohort.csv",
                "metrics": output / "tables/evening_oracle_0924_metrics.csv",
                "paired_gain": output / "tables/evening_oracle_0924_paired_gain.csv"}
    for name, frame in (("cases", cases), ("conditions", condition_summary),
                        ("weekdays", weekday_table), ("cohort", pd.DataFrame(cohort_rows)),
                        ("metrics", metrics), ("paired_gain", gains)):
        frame.to_csv(products[name], index=False)
    prediction_path = output / "predictions/evening_oracle_0924_point_oof.csv"
    pd.concat([point, oracle], ignore_index=True).to_csv(prediction_path, index=False)
    summary = {"scope": "development_oof_only", "test_boundary_exclusive": str(boundary),
               "history_max_timestamp": str(history.index.max()),
               "selected_point_model_unchanged": POINT_NAME,
               "selected_risk_model_unchanged": str(primary.risk_model.iloc[0]),
               "risk_evening_episode_counts": cases.status.value_counts().to_dict(),
               "risk_fn_case_count": int(cases.status.eq("FN").sum()),
               "risk_fn_with_oracle_point_tp": int((cases.status.eq("FN") &
                                                    cases.oracle_point_episode_status.eq("TP")).sum()),
               "case_common_cohort_audit": {
                   "risk_evening_cases": len(cases),
                   "baseline_point_outside_common_cohort": int(cases.baseline_point_episode_status.eq("outside_common_cohort").sum()),
                   "oracle_point_outside_common_cohort": int(cases.oracle_point_episode_status.eq("outside_common_cohort").sum())},
               "oracle_scenario": "perfect observed production availability at future target time; not a guaranteed performance bound",
               "scenario_label": SCENARIO_LABEL,
               "planned_production_availability_verified": False,
               "causal_or_operational_effect_claimed": False,
               "oracle_added_features": [ORACLE_FEATURE], "oracle_weather_features_added": [],
               "point_alerts_are_not_original_risk_alerts": True,
               "paired_gain": gains.to_dict("records"),
               "cohort": cohort_rows, "baseline_fresh_oof_parity": parity,
               "excluded_score_peak_positions": sum(row["score_production_missing_peak_positions"] for row in cohort_rows),
               "original_h4_peak_positions": int(oof.loc[(oof.horizon == HORIZON) &
                                                    (oof.model == POINT_NAME)].eval("y > tau").sum()),
               "common_h4_peak_positions": int(point.eval("y > tau").sum()),
               "cohort_interpretation": "Two boundary score positions lack future-hour production and are excluded from both point models; they are actual peaks. Common-cohort baseline peak MAE is therefore not the original full-cohort peak MAE.",
               "original_q95_parity_evidence": a_evidence,
               "reference_only": True, "selection_used": False,
               "original_selection_unchanged": True,
               "sources_sha256": {"development_oof": _digest(oof_path),
                                  "development_selection": _digest(manifest_path)},
               "products_sha256": {name: _digest(path) for name, path in products.items()},
               "ignored_predictions_sha256": _digest(prediction_path)}
    log = output / "logs/evening_oracle_0924.json"
    log.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Results directory under outputs/ or _validation/")
    args = parser.parse_args()
    result = run(ROOT, output=args.output)
    print(json.dumps({"status": "complete", "risk_evening_episode_counts": result["risk_evening_episode_counts"],
                      "paired_gain": result["paired_gain"]}, ensure_ascii=False, indent=2))
