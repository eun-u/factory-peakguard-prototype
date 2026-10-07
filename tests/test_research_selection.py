"""Synthetic Phase 2 ranking and holdout-lock regression tests."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.research_selection import (
    M1_NAME, M4_NAMES, _risk_gate, compare_point_candidate, extend_selection,
)
from src.research_pipeline import _assert_candidate_oof
from src.training import run_development


def _point(horizon: int, model: str, *, peak_prediction: float = 80.,
           last_peak_prediction: float | None = None) -> pd.DataFrame:
    rows = []
    for fold in range(3):
        for day in range(12):
            target = pd.Timestamp("2021-04-01") + pd.Timedelta(days=fold * 20 + day)
            target += pd.Timedelta(hours=10)
            actual = 100. if day < 4 else 80.
            peak_pred = last_peak_prediction if fold == 2 and last_peak_prediction is not None else peak_prediction
            pred = peak_pred if actual > 98. else 80.
            rows.append({"origin": target - horizon * pd.Timedelta(minutes=15),
                         "target_time": target, "horizon": horizon, "fold": fold,
                         "model": model, "y": actual, "tau": 98., "pred": pred,
                         "alert": pred > 98., "alert_cutoff": 98.})
    return pd.DataFrame(rows)


def _risk(horizon: int, name: str, *, q95: float) -> pd.DataFrame:
    rows = []
    for fold in range(3):
        for day in range(10):
            target = pd.Timestamp("2021-04-01") + pd.Timedelta(days=fold * 20 + day)
            target += pd.Timedelta(hours=10)
            actual = 100. if day < 3 else 90.
            rows.append({"origin": target - horizon * pd.Timedelta(minutes=15),
                         "target_time": target, "horizon": horizon, "fold": fold,
                         "model": name, "y": actual, "tau": 98., "pred": 90.,
                         "alert": False, "alert_cutoff": .5,
                         "p_exceed": .1, "q10": 80., "q50": 90., "q90": 95.,
                         "q95": 95., "q975": 105., "q90_cal": 95.,
                         "q95_cal": q95, "q975_cal": max(105., q95),
                         "q50_top_edge": 85.})
    return pd.DataFrame(rows)


def test_residual_challenger_wins_only_with_fixed_peak_ci_and_stable_last_fold():
    old = _point(16, "c3_holiday_hybrid", peak_prediction=80.)
    better = _point(16, M1_NAME, peak_prediction=98.)
    win = compare_point_candidate(old, better, n_boot=100)
    assert win["selected"] is True
    assert win["reason"] == "challenger_peak_mae_ci"
    assert win["n_positive_folds"] == 3
    degraded = _point(16, M1_NAME, peak_prediction=98., last_peak_prediction=60.)
    loss = compare_point_candidate(old, degraded, n_boot=100)
    assert loss["selected"] is False
    assert loss["reason"] == "last_fold_peak_mae_ge_1p10_incumbent"


def test_residual_ci_tie_keeps_original_simpler_incumbent_and_missing_fold_fails():
    old = _point(96, "c3_holiday_hybrid", peak_prediction=80.)
    tied = _point(96, M1_NAME, peak_prediction=80.)
    decision = compare_point_candidate(old, tied, n_boot=50)
    assert decision["selected"] is False
    assert decision["reason"] == "incumbent_simpler_after_ci_overlap"
    incomplete = tied.loc[tied.fold.ne(2)]
    missing = compare_point_candidate(old, incomplete, n_boot=50)
    assert missing["selected"] is False
    assert missing["all_original_folds_matched"] is False


def test_adaptive_gate_requires_all_three_fixed_conditions():
    baseline = _risk(16, "lgbm_quantile_b", q95=95.)
    candidate = _risk(16, "q95_rolling_672", q95=105.)
    point = _point(16, "c3_holiday_hybrid")
    accepted = _risk_gate(candidate, baseline, point)
    assert accepted["eligible"] is True
    assert accepted["gates"]["top_coverage_error_smaller"] is True
    failed = _risk_gate(_risk(16, "q95_rolling_672", q95=95.), baseline, point)
    assert failed["eligible"] is False
    assert failed["gates"]["last_fold_top_coverage_higher"] is False


def test_extension_retains_original_selection_and_h4_reference_role():
    original = {"by_horizon": {
        "4": {"point_model": "lgbm_no_holiday_weight_2", "conformal": "b"},
        "16": {"point_model": "c3_holiday_hybrid", "conformal": "b",
               "cbl": "c3_holiday_hybrid"},
        "96": {"point_model": "c3_holiday_hybrid", "conformal": "b",
               "cbl": "c3_holiday_hybrid"}}}
    original_snapshot = copy.deepcopy(original)
    original_oof = pd.concat([
        _point(4, "lgbm_no_holiday_weight_2"),
        _point(16, "c3_holiday_hybrid"), _point(96, "c3_holiday_hybrid"),
        _risk(16, "lgbm_quantile_b", q95=95.),
        _risk(96, "lgbm_quantile_b", q95=95.),
    ], ignore_index=True)
    candidate_oof = pd.concat([
        _point(4, M1_NAME, peak_prediction=98.),
        _point(16, M1_NAME, peak_prediction=98.),
        _point(96, M1_NAME, peak_prediction=80.),
        _risk(16, "q95_rolling_672", q95=105.),
    ], ignore_index=True)
    result, decisions = extend_selection(original, original_oof, candidate_oof,
                                         {"bootstrap": {"n": 50}, "seed": 42})
    assert original == original_snapshot
    assert result["by_horizon"]["4"]["point_model"] == "lgbm_no_holiday_weight_2"
    assert result["by_horizon"]["16"]["point_model"] == M1_NAME
    assert result["by_horizon"]["96"]["point_model"] == "c3_holiday_hybrid"
    assert result["by_horizon"]["16"]["conformal"] == "b"
    assert result["by_horizon"]["16"]["risk_model"] == "q95_rolling_672"
    assert decisions.loc[decisions.horizon.eq(4) & decisions.stage.eq("point"), "selected"].iloc[0] == False


def test_enabled_research_rejects_full_history_before_any_fold_fits(tmp_path):
    idx = pd.DatetimeIndex([pd.Timestamp("2021-08-09 09:30"),
                            pd.Timestamp("2021-08-09 09:45")])
    history = pd.DataFrame({"power": [1., 999.]}, index=idx)
    cfg = {"research_0925": {"enabled": True},
           "split": {"test_start_origin": "2021-08-09 09:45:00"}}
    with pytest.raises(ValueError, match="sealed-prefix input"):
        run_development(history, cfg, tmp_path)


def test_candidate_contract_preserves_original_grid_carrier_and_holdout_lock():
    original_parts, candidates = [], []
    for h in (4, 16, 96):
        point = _point(h, "c3_holiday_hybrid")
        original_parts.append(point)
        residual = point.copy()
        residual["model"] = M1_NAME
        residual["reference_only"] = h == 4
        residual["selection_used"] = False
        candidates.append(residual)
        if h == 4:
            continue
        base = point.copy()
        base["model"] = "lgbm_quantile_b"
        base["q50"] = base.pred
        base["q90_cal"] = 95.
        base["q95_cal"] = 105.
        base["q975_cal"] = 110.
        base["p_exceed"] = .25
        original_parts.append(base)
        for name in M4_NAMES:
            adaptive = base.copy()
            adaptive["model"] = name
            candidates.append(adaptive)
    original = pd.concat(original_parts, ignore_index=True, sort=False)
    candidate = pd.concat(candidates, ignore_index=True, sort=False)
    evidence = _assert_candidate_oof(original, candidate, pd.Timestamp("2021-08-09 09:45"))
    assert evidence["candidate_pairs"] == 11
    assert evidence["candidate_counts_by_horizon"] == {"4": 1, "16": 5, "96": 5}
    tampered = candidate.copy()
    tampered.loc[tampered.index[0], "target_time"] = pd.Timestamp("2021-08-09 09:45")
    with pytest.raises(AssertionError, match="locked holdout"):
        _assert_candidate_oof(original, tampered, pd.Timestamp("2021-08-09 09:45"))
