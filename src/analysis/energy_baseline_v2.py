"""Fold-local explanatory Ridge and auditable peak-type decomposition (v2).

All measured production and weather here are retrospective inputs. They are
never passed to the operational peak forecast or its model-selection rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.holidays import calendar_flags
from ._common import block_ci, canonical_history, write_table
from .energy_baseline import _matrix
from .errors import episodes


TEST_ORIGIN = pd.Timestamp("2021-08-09 09:45")
RIDGE_ALPHA = 100.0
ATTRIBUTION_SCOPE = "full_ridge_baseline_including_weather_calendar_not_production_only"
TYPE_LABELS = {"production_explained": "생산 기인형(전체 기준선 설명형; 생산 단독 귀속 아님)",
               "mixed": "혼합형", "residual": "잔차형", "unclassified": "분류 불가"}


@dataclass
class FoldEnergyModelV2:
    fold: int
    model: Ridge
    temperature_median: float
    train_end: pd.Timestamp
    production_coefficient_per_hour: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_matrix(frame, self.temperature_median))


@dataclass
class EnergyV2Result:
    models: dict[int, FoldEnergyModelV2]
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    coefficient_ci: pd.DataFrame
    peak_types: pd.DataFrame
    type_summary: pd.DataFrame
    representatives: pd.DataFrame
    summary: dict
    paths: dict[str, str] = field(default_factory=dict)


def _safe_history(history: pd.DataFrame) -> pd.DataFrame:
    history = canonical_history(history)
    if history.empty or history.index.max() >= TEST_ORIGIN:
        raise ValueError("v2 requires development-only safe history before first test origin")
    if not history.index.is_unique or "power" not in history or "production_target" not in history:
        raise ValueError("v2 requires unique power and retrospective production rows")
    return history


def _fold_metadata(metadata: list[dict] | pd.DataFrame) -> dict[int, dict]:
    rows = metadata.to_dict("records") if isinstance(metadata, pd.DataFrame) else list(metadata)
    selected = [row for row in rows if int(row["horizon"]) == 4]
    out = {int(row["fold"]): row for row in selected}
    if len(out) != len(selected) or not out:
        raise ValueError("Unique h4 fold metadata required")
    for row in selected:
        if not all(key in row for key in ("fit_end", "score_start", "score_end")):
            raise ValueError("Fold metadata lacks fit/score boundaries")
        if pd.Timestamp(row["score_end"]) >= TEST_ORIGIN:
            raise ValueError("Fold score boundary reaches frozen test origin")
    return out


def _clean_train(history: pd.DataFrame, fit_end: pd.Timestamp) -> pd.DataFrame:
    train = history.loc[history.index <= fit_end].copy()
    power = pd.to_numeric(train.power, errors="coerce")
    production = pd.to_numeric(train.production_target, errors="coerce")
    valid = power.notna() & production.notna() & np.isfinite(power) & np.isfinite(production)
    if "time_repaired" in train:
        valid &= ~train.time_repaired.fillna(False).astype(bool)
    train = train.loc[valid]
    if len(train) < 100:
        raise ValueError("Fold training prefix has fewer than 100 clean retrospective rows")
    return train


def ridge_from_sufficient_stats(n: int, sum_x: np.ndarray, sum_y: float,
                                xx: np.ndarray, xy: np.ndarray,
                                alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, float]:
    """Exact centered normal equations for sklearn Ridge(fit_intercept=True)."""
    if n < 2 or alpha <= 0:
        raise ValueError("Ridge sufficient statistics require n>=2 and alpha>0")
    sx = np.asarray(sum_x, dtype=float)
    covariance = np.asarray(xx, dtype=float)-np.outer(sx, sx)/n
    target_covariance = np.asarray(xy, dtype=float)-sx*float(sum_y)/n
    coefficient = np.linalg.solve(covariance+alpha*np.eye(len(sx)), target_covariance)
    intercept = float(sum_y/n-np.dot(sx/n, coefficient))
    return coefficient, intercept


def coefficient_day_bootstrap(train: pd.DataFrame, model: FoldEnergyModelV2,
                              n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Refit the identical fixed-preprocessing Ridge via daily sufficient stats."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    x = _matrix(train, model.temperature_median)
    y = train.power.to_numpy(float)
    dates = pd.DatetimeIndex(train.index).normalize()
    days = dates.unique()
    if len(days) < 2:
        return np.nan, np.nan
    stats = []
    for day in days:
        xi, yi = x[dates == day], y[dates == day]
        stats.append((len(yi), xi.sum(axis=0), yi.sum(), xi.T@xi, xi.T@yi))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        chosen = [stats[i] for i in rng.integers(0, len(stats), len(stats))]
        n = sum(z[0] for z in chosen)
        coefficient, _ = ridge_from_sufficient_stats(
            n, sum((z[1] for z in chosen), np.zeros(x.shape[1])),
            sum(z[2] for z in chosen), sum((z[3] for z in chosen), np.zeros((x.shape[1], x.shape[1]))),
            sum((z[4] for z in chosen), np.zeros(x.shape[1])), alpha=float(model.model.alpha))
        estimates.append(float(coefficient[0]))
    return tuple(map(float, np.quantile(estimates, [.025, .975])))


def _cell_keys(index: pd.DatetimeIndex) -> pd.DataFrame:
    start = pd.DatetimeIndex(index)-pd.Timedelta(minutes=15)
    return pd.DataFrame({"offday": calendar_flags(start)["is_offday"].to_numpy(dtype=bool),
                         "slot": start.hour*4+start.minute//15}, index=index)


def _typicals(train: pd.DataFrame, model: FoldEnergyModelV2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cells = _cell_keys(pd.DatetimeIndex(train.index))
    cells["actual"] = train.power.to_numpy(float)
    cells["baseline"] = model.predict(train)
    by_cell = cells.groupby(["offday", "slot"])[["actual", "baseline"]].median()
    by_slot = cells.groupby("slot")[["actual", "baseline"]].median()
    overall = cells[["actual", "baseline"]].median()
    return by_cell, by_slot, overall


def _typical_value(by_cell: pd.DataFrame, by_slot: pd.DataFrame, overall: pd.Series,
                   offday: bool, slot: int, field: str) -> tuple[float, str]:
    key = (offday, slot)
    if key in by_cell.index and np.isfinite(by_cell.loc[key, field]):
        return float(by_cell.loc[key, field]), "same_daytype_slot"
    if slot in by_slot.index and np.isfinite(by_slot.loc[slot, field]):
        return float(by_slot.loc[slot, field]), "same_slot_fallback"
    return float(overall[field]), "global_fallback"


def classify_peak_types_v2(operational: pd.DataFrame, history: pd.DataFrame,
                           models: dict[int, FoldEnergyModelV2], trains: dict[int, pd.DataFrame],
                           n_boot: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold, group in operational.groupby("fold", sort=True):
        fold = int(fold)
        group = group.sort_values("target_time")
        by_cell, by_slot, overall = _typicals(trains[fold], models[fold])
        for start, end in episodes(group.target_time, group.y.gt(group.tau)):
            episode = group.loc[group.target_time.between(start, end)]
            peak = episode.loc[episode.y.idxmax()]
            stamp = pd.Timestamp(peak.target_time)
            key = _cell_keys(pd.DatetimeIndex([stamp])).iloc[0]
            slot = int(key.slot)
            offday = bool(key.offday)
            typical, source_actual = _typical_value(by_cell, by_slot, overall, offday, slot, "actual")
            baseline_typical, source_baseline = _typical_value(
                by_cell, by_slot, overall, offday, slot, "baseline")
            peak_production = pd.to_numeric(history.loc[stamp, "production_target"], errors="coerce")
            if pd.isna(peak_production) or not np.isfinite(float(peak_production)):
                baseline_at_peak = np.nan
                reason = "missing_production"
            elif "time_repaired" in history and bool(history.loc[stamp, "time_repaired"]):
                baseline_at_peak = np.nan
                reason = "repaired_interval"
            else:
                baseline_at_peak = float(models[fold].predict(history.loc[[stamp]])[0])
                reason = "classified"
            excess = float(peak.y-typical)
            rise = float(baseline_at_peak-baseline_typical)
            share = float(np.clip(rise/excess, 0, 1)) if np.isfinite(excess) and excess > 0 and np.isfinite(rise) else np.nan
            if reason == "classified" and not np.isfinite(share):
                reason = "nonpositive_or_nonfinite_excess_or_rise"
            kind = ("unclassified" if not np.isfinite(share) else
                    "production_explained" if share >= .5 else "mixed" if share >= .2 else "residual")
            rows.append({"fold": fold, "start": start, "end": end, "target_time": stamp,
                         "actual_max": float(peak.y), "tau": float(peak.tau),
                         "slot_typical": typical, "baseline_at_peak": baseline_at_peak,
                         "baseline_typical": baseline_typical, "excess": excess,
                         "rise": rise, "share": share, "type": kind, "classification_reason": reason,
                         "slot_typical_source": source_actual,
                         "baseline_typical_source": source_baseline,
                         "day_type": "offday" if offday else "workday", "slot": slot})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    for kind in ("production_explained", "mixed", "residual", "unclassified"):
        share, low, high = block_ci(detail, lambda x: float(x.type.eq(kind).mean()), n=n_boot, seed=seed)
        summary_rows.append({"type": kind, "episodes": int(detail.type.eq(kind).sum()),
                             "share": share, "ci_low": low, "ci_high": high,
                             "denominator_episodes": len(detail)})
    chosen = []
    used = set()
    for kind in ("production_explained", "mixed", "residual"):
        candidate = detail.loc[detail.type.eq(kind)].sort_values("excess", ascending=False).head(1)
        if not candidate.empty:
            row = candidate.iloc[0]
            chosen.append(row)
            used.add((row.fold, row.start))
    if len(chosen) < 3:
        remaining = detail.loc[~detail.apply(lambda r: (r.fold, r.start) in used, axis=1)]
        chosen += [r for _, r in remaining.sort_values("excess", ascending=False).head(3-len(chosen)).iterrows()]
    representatives = pd.DataFrame(chosen).reset_index(drop=True)
    if not representatives.empty:
        representatives.insert(0, "representative_rank", np.arange(1, len(representatives)+1))
    return detail, pd.DataFrame(summary_rows), representatives


def run_energy_baseline_v2(history: pd.DataFrame, operational: pd.DataFrame,
                           fold_metadata: list[dict] | pd.DataFrame,
                           outdir: str | Path | None = None, *, n_boot: int = 1000,
                           seed: int = 42, with_classification: bool = True) -> EnergyV2Result:
    """Fit one descriptive Ridge per forecast fit prefix and score only h4 OOF."""
    history = _safe_history(history)
    metadata = _fold_metadata(fold_metadata)
    if operational.empty or not operational.horizon.eq(4).all():
        raise ValueError("Only selected h4 development OOF operational rows are accepted")
    if operational.duplicated(["fold", "target_time"]).any():
        raise ValueError("Duplicate OOF target within a fold")
    if pd.to_datetime(operational.origin).ge(TEST_ORIGIN).any():
        raise ValueError("Frozen test origin entered v2 analysis")
    models, trains, forecasts, metrics, coefficient_rows = {}, {}, [], [], []
    for fold, group in operational.groupby("fold", sort=True):
        fold = int(fold)
        if fold not in metadata:
            raise ValueError(f"Missing fold metadata: {fold}")
        meta = metadata[fold]
        fit_end = pd.Timestamp(meta["fit_end"])
        if fit_end >= pd.to_datetime(group.origin).min():
            raise ValueError("Fold explanatory fit reaches OOF origins")
        score_start, score_end = pd.Timestamp(meta["score_start"]), pd.Timestamp(meta["score_end"])
        if pd.to_datetime(group.origin).lt(score_start).any() or pd.to_datetime(group.origin).gt(score_end).any():
            raise ValueError("Operational row outside its fold OOF score range")
        stamps = pd.DatetimeIndex(pd.to_datetime(group.target_time))
        if not stamps.isin(history.index).all() or stamps.max() >= TEST_ORIGIN:
            raise ValueError("OOF target absent from safe development history")
        observed = history.loc[stamps]
        if not np.allclose(observed.power.to_numpy(float), group.y.to_numpy(float), atol=1e-10, rtol=1e-10):
            raise ValueError("OOF labels differ from safe development history")
        train = _clean_train(history, fit_end)
        temperature = pd.to_numeric(train.get("temperature", pd.Series(dtype=float)), errors="coerce")
        temp_median = float(temperature.median()) if len(temperature) and temperature.notna().any() else 18.0
        x_train = _matrix(train, temp_median)
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(x_train, train.power.to_numpy(float))
        item = FoldEnergyModelV2(fold, ridge, temp_median, pd.Timestamp(train.index.max()), float(ridge.coef_[0]))
        models[fold], trains[fold] = item, train
        forecast = group[["fold", "origin", "target_time", "y", "tau"]].copy()
        valid_score = np.isfinite(pd.to_numeric(observed.production_target, errors="coerce").to_numpy(float)).copy()
        if "time_repaired" in observed:
            valid_score &= ~observed.time_repaired.fillna(False).astype(bool).to_numpy()
        forecast["ridge_pred"] = np.nan
        forecast["ridge_input_eligible"] = valid_score
        if valid_score.any():
            forecast.loc[valid_score, "ridge_pred"] = item.predict(observed.iloc[np.flatnonzero(valid_score)])
        forecasts.append(forecast)
        eligible = forecast.loc[forecast.ridge_input_eligible]
        fold_r2 = float(r2_score(eligible.y, eligible.ridge_pred)) if len(eligible) >= 2 and eligible.y.nunique() > 1 else np.nan
        metrics.append({"fold": fold, "n_train": len(train), "n_oof": len(forecast),
                        "n_oof_ridge_eligible": len(eligible),
                        "n_oof_ridge_excluded": len(forecast)-len(eligible),
                        "train_end": str(item.train_end), "score_start": str(score_start),
                        "score_end": str(score_end), "r2_oof": fold_r2,
                        "r2_below_0_2": bool(np.isfinite(fold_r2) and fold_r2 < .2),
                        "production_coefficient_per_hour": item.production_coefficient_per_hour,
                        "raw_hourly_production_unit_effect_per_15min": item.production_coefficient_per_hour/4,
                        "coefficient_positive": bool(item.production_coefficient_per_hour > 0)})
        low, high = coefficient_day_bootstrap(train, item, n_boot=n_boot, seed=seed+fold)
        coefficient_rows.append({"fold": fold, "coefficient": item.production_coefficient_per_hour,
                                 "ci_low": low, "ci_high": high, "n_boot": n_boot,
                                 "raw_hourly_production_unit_effect_per_15min": item.production_coefficient_per_hour/4,
                                 "raw_hourly_production_unit_effect_ci_low": low/4,
                                 "raw_hourly_production_unit_effect_ci_high": high/4,
                                 "bootstrap_unit": "training_date", "ridge_alpha": RIDGE_ALPHA,
                                 "fixed_preprocessing": "fold_training_temperature_median"})
    oof = pd.concat(forecasts, ignore_index=True)
    pooled = oof.loc[oof.ridge_input_eligible]
    pooled_r2 = float(r2_score(pooled.y, pooled.ridge_pred)) if len(pooled) >= 2 and pooled.y.nunique() > 1 else np.nan
    passed = bool(np.isfinite(pooled_r2) and pooled_r2 >= .2)
    if with_classification:
        peak_types, type_summary, representatives = classify_peak_types_v2(
            operational, history, models, trains, n_boot=n_boot, seed=seed)
    else:
        peak_types, type_summary, representatives = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = {"status": "ok", "pooled_oof_r2": pooled_r2, "r2_gate": .2,
               "r2_gate_passed": passed,
               "main_conclusion": ("not_computed" if not with_classification else
                                   "descriptive_peak_types" if passed else "production_does_not_explain_peaks"),
               "main_conclusion_ko": ("피크 유형 분석 미실행" if not with_classification else
                                      "설명모형 R² 문턱 통과, 피크 유형은 사후 분해" if passed else
                                      "생산량이 피크를 설명하지 못함"),
               "classification_role": ("not_computed" if not with_classification else
                                       "descriptive" if passed else "exploratory_audit_only"),
               "interpretation": "R2 evaluates the full explanatory model; production coefficients and types do not identify causal effects.",
               "production_coefficient_units": "Ridge feature is hourly production / 4. One raw hourly production unit corresponds to coefficient / 4 in each 15-minute prediction; this is an association, not a causal effect.",
               "folds": len(models), "peak_episodes": len(peak_types),
               "n_oof_ridge_eligible": len(pooled), "n_oof_ridge_excluded": len(oof)-len(pooled),
               "classification_computed": with_classification}
    summary["attribution_scope"] = ATTRIBUTION_SCOPE
    if with_classification:
        for frame in (peak_types, type_summary, representatives):
            frame["r2_gate_passed"] = passed
            frame["classification_role"] = summary["classification_role"]
            frame["attribution_scope"] = ATTRIBUTION_SCOPE
            frame["type_label"] = (frame["type"].map(TYPE_LABELS) if "type" in frame else
                                   pd.Series(index=frame.index, dtype="string"))
    result = EnergyV2Result(models, oof, pd.DataFrame(metrics), pd.DataFrame(coefficient_rows),
                            peak_types, type_summary, representatives, summary)
    if outdir is not None:
        tables = Path(outdir)/"tables"
        outputs = [("energy_baseline_oof_predictions_v2", result.oof_predictions),
                   ("energy_baseline_fold_metrics_v2", result.fold_metrics),
                   ("energy_baseline_coefficient_ci_v2", result.coefficient_ci)]
        if with_classification:
            outputs += [("peak_types_v2", result.peak_types),
                        ("peak_type_summary_v2", result.type_summary),
                        ("peak_type_representatives_v2", result.representatives)]
        for name, data in outputs:
            folder = Path(outdir)/"predictions" if name == "energy_baseline_oof_predictions_v2" else tables
            result.paths[name] = str(write_table(data, folder/f"{name}.csv"))
    return result
