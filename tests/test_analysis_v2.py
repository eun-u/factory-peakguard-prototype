"""Synthetic-only checks for preregistered task 4/5 v2 analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.analysis.energy_baseline import _matrix
from src.analysis.energy_baseline_v2 import (
    EnergyV2Result, FoldEnergyModelV2, TEST_ORIGIN, classify_peak_types_v2,
    _typical_value, coefficient_day_bootstrap, ridge_from_sufficient_stats,
    run_energy_baseline_v2,
)
from src.analysis.shift_v2 import prepare_h4_operational_oof, run_shift_v2


def _tariff() -> dict:
    return {"bands": {"all": {"months": list(range(1, 13)), "intervals": [
        {"start": "00:00", "end": "09:00", "band": "off_peak"},
        {"start": "09:00", "end": "11:00", "band": "peak"},
        {"start": "11:00", "end": "24:00", "band": "off_peak"}]}},
        "scenario_weights": {"off_peak": 1, "mid_peak": 2, "peak": 3}}


class _ProductionBaseline:
    def predict(self, x):
        return 5+np.asarray(x)[:, 0]


class AnalysisV2Tests(unittest.TestCase):
    def test_sufficient_statistics_match_identical_ridge_preprocessing_and_penalty(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(120, 7))
        y = 4+2*x[:, 0]-x[:, 2]+rng.normal(scale=.2, size=len(x))
        fitted = Ridge(alpha=100.0).fit(x, y)
        coefficient, intercept = ridge_from_sufficient_stats(
            len(y), x.sum(axis=0), float(y.sum()), x.T@x, x.T@y)
        np.testing.assert_allclose(coefficient, fitted.coef_, atol=1e-10, rtol=1e-10)
        self.assertAlmostEqual(intercept, fitted.intercept_, places=10)

        dates = pd.date_range("2021-01-04 00:15", periods=120, freq="15min")
        train = pd.DataFrame({"power": y, "production_target": np.maximum(0, x[:, 0]*4+10),
                              "temperature": np.full(len(y), 18.)}, index=dates)
        fixed_x = _matrix(train, 18.0)
        model = FoldEnergyModelV2(0, Ridge(alpha=100).fit(fixed_x, y), 18.0,
                                  dates.max(), 0.)
        estimate, _ = coefficient_day_bootstrap(train, model, n_boot=1, seed=7)
        days = dates.normalize().unique()
        chosen = np.random.default_rng(7).integers(0, len(days), len(days))
        positions = np.concatenate([np.flatnonzero(dates.normalize() == days[i]) for i in chosen])
        direct = Ridge(alpha=100).fit(fixed_x[positions], y[positions]).coef_[0]
        self.assertAlmostEqual(estimate, direct, places=10)

    def test_prepare_h4_uses_operational_rule_and_retains_original_risk_alert(self):
        base = pd.DataFrame({"origin": pd.to_datetime(["2021-06-01 09:15"]),
                             "target_time": pd.to_datetime(["2021-06-01 10:15"]),
                             "horizon": [4], "fold": [0], "y": [80.], "tau": [90.]})
        point = base.assign(model="lgbm_no_holiday", pred=81.)
        risk = base.assign(model="lgbm_quantile_b", pred=81., p_exceed=.2,
                           q95_cal=85., alert=False)
        result = prepare_h4_operational_oof(pd.concat([point, risk], ignore_index=True),
            {"by_horizon": {"4": {"point_model": "lgbm_no_holiday", "conformal": "b"}}})
        self.assertTrue(result.alert.iloc[0])
        self.assertFalse(result.risk_alert_original.iloc[0])

    def test_fold_ridge_uses_fit_prefix_and_can_defer_type_outputs(self):
        idx = pd.date_range("2021-01-01 00:15", periods=96*6, freq="15min")
        production = 12+(idx.hour % 8)*2
        power = 50+production/4*3+3*np.sin(np.arange(len(idx))*2*np.pi/96)
        history = pd.DataFrame({"power": power, "production_target": production,
                                "temperature": 18.}, index=idx)
        targets = idx[(idx >= pd.Timestamp("2021-01-05 00:15")) &
                      (idx <= pd.Timestamp("2021-01-05 23:45"))]
        frame = pd.DataFrame({"fold": 0, "origin": targets-pd.Timedelta(hours=1),
                              "target_time": targets, "horizon": 4,
                              "y": history.loc[targets, "power"].to_numpy(),
                              "tau": 60.})
        meta = [{"horizon": 4, "fold": 0, "fit_end": "2021-01-04 12:00",
                 "score_start": str(frame.origin.min()), "score_end": str(frame.origin.max())}]
        with tempfile.TemporaryDirectory() as temp:
            result = run_energy_baseline_v2(history, frame, meta, temp,
                                             n_boot=8, with_classification=False)
            self.assertTrue(result.summary["r2_gate_passed"])
            self.assertEqual(result.summary["classification_role"], "not_computed")
            self.assertGreater(result.models[0].production_coefficient_per_hour, 0)
            self.assertAlmostEqual(result.coefficient_ci.raw_hourly_production_unit_effect_per_15min.iloc[0],
                                   result.models[0].production_coefficient_per_hour/4)
            self.assertIn("hourly production / 4", result.summary["production_coefficient_units"])
            self.assertLessEqual(result.models[0].train_end, pd.Timestamp(meta[0]["fit_end"]))
            self.assertTrue(np.isfinite(result.coefficient_ci.ci_low.iloc[0]))
            self.assertFalse(Path(temp, "tables", "peak_types_v2.csv").exists())
            self.assertTrue(Path(temp, "predictions", "energy_baseline_oof_predictions_v2.csv").exists())
            missing_history = history.copy()
            missing_history.loc[targets[10], "production_target"] = np.nan
            missing = run_energy_baseline_v2(missing_history, frame, meta,
                                             n_boot=4, with_classification=False)
            self.assertEqual(missing.summary["n_oof_ridge_excluded"], 1)
            self.assertTrue(np.isnan(missing.oof_predictions.loc[10, "ridge_pred"]))
            self.assertFalse(missing.oof_predictions.loc[10, "ridge_input_eligible"])
            weak_history = history.copy()
            weak_history.loc[targets, "power"] = np.linspace(1., 2., len(targets))
            weak_frame = frame.copy()
            weak_frame["y"] = weak_history.loc[targets, "power"].to_numpy()
            weak_frame["tau"] = 1.5
            weak = run_energy_baseline_v2(weak_history, weak_frame, meta,
                                          n_boot=4, with_classification=True)
            self.assertFalse(weak.summary["r2_gate_passed"])
            self.assertEqual(weak.summary["main_conclusion"], "production_does_not_explain_peaks")
            self.assertEqual(weak.summary["main_conclusion_ko"], "생산량이 피크를 설명하지 못함")
            self.assertEqual(weak.summary["classification_role"], "exploratory_audit_only")
            self.assertGreater(len(weak.peak_types), 0)
            self.assertTrue(weak.peak_types.r2_gate_passed.eq(False).all())
            self.assertTrue(weak.type_summary.classification_role.eq("exploratory_audit_only").all())
            self.assertEqual(set(weak.type_summary.attribution_scope),
                {"full_ridge_baseline_including_weather_calendar_not_production_only"})
            no_peak = frame.copy()
            no_peak["tau"] = 1e6
            empty_types = run_energy_baseline_v2(history, no_peak, meta,
                                                 n_boot=4, with_classification=True)
            self.assertTrue(empty_types.peak_types.empty)

    def test_peak_share_boundaries_and_training_cell_fallback(self):
        train_idx = pd.to_datetime(["2021-01-04 10:15", "2021-01-05 10:15"])
        target_idx = pd.to_datetime(["2021-01-06 10:15", "2021-01-07 10:15", "2021-01-08 10:15"])
        all_idx = train_idx.append(target_idx)
        history = pd.DataFrame({"power": [10., 10., 20., 20., 20.],
                                "production_target": [0., 0., 20., 8., 7.96],
                                "temperature": 18.}, index=all_idx)
        model = FoldEnergyModelV2(0, _ProductionBaseline(), 18., train_idx.max(), 1.)
        operational = pd.DataFrame({"fold": 0, "target_time": target_idx,
                                    "y": [20.]*3, "tau": [15.]*3})
        detail, summary, representatives = classify_peak_types_v2(
            operational, history, {0: model}, {0: history.loc[train_idx]}, n_boot=8)
        self.assertEqual(detail.type.tolist(), ["production_explained", "mixed", "residual"])
        np.testing.assert_allclose(detail.share, [.5, .2, .199])
        self.assertEqual(len(representatives), 3)
        self.assertEqual(int(summary.episodes.sum()), 3)
        self.assertEqual(set(detail.slot_typical_source), {"same_daytype_slot"})
        by_cell = pd.DataFrame({"actual": [10.], "baseline": [5.]},
                               index=pd.MultiIndex.from_tuples([(False, 40)], names=["offday", "slot"]))
        by_slot = pd.DataFrame({"actual": [11.], "baseline": [6.]}, index=[41])
        overall = pd.Series({"actual": 12., "baseline": 7.})
        self.assertEqual(_typical_value(by_cell, by_slot, overall, True, 41, "actual"),
                         (11., "same_slot_fallback"))
        self.assertEqual(_typical_value(by_cell, by_slot, overall, True, 42, "actual"),
                         (12., "global_fallback"))
        missing_history = history.copy()
        missing_history.loc[target_idx[0], "production_target"] = np.nan
        missing_detail, _, _ = classify_peak_types_v2(
            operational, missing_history, {0: model}, {0: history.loc[train_idx]}, n_boot=8)
        self.assertEqual(missing_detail.type.iloc[0], "unclassified")
        self.assertEqual(missing_detail.classification_reason.iloc[0], "missing_production")
        self.assertTrue(np.isnan(missing_detail.share.iloc[0]))

    def test_failed_first_alarm_may_retry_same_source_on_later_valid_alarm(self):
        idx = pd.date_range("2021-01-04 00:15", "2021-01-07 09:30", freq="15min")
        history = pd.DataFrame({"power": 70., "production_target": 20.,
                                "time_repaired": False}, index=idx)
        history.loc[pd.date_range("2021-01-06 11:15", "2021-01-07 08:30", freq="15min"), "power"] = 100.
        targets = pd.date_range("2021-01-06 00:15", "2021-01-07 09:30", freq="15min")
        p = pd.DataFrame({"origin": targets-pd.Timedelta(hours=1), "target_time": targets,
                          "horizon": 4, "fold": 0, "y": history.loc[targets, "power"].to_numpy(),
                          "tau": 90., "p_exceed": 0., "q95_cal": 80.})
        flagged = p.target_time.isin(pd.to_datetime(["2021-01-06 10:15", "2021-01-06 10:30"]))
        p.loc[flagged, "p_exceed"] = .9
        p["alert"] = flagged
        energy = EnergyV2Result({0: FoldEnergyModelV2(0, None, 18., pd.Timestamp("2021-01-04"), 2.)},
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
        meta = [{"horizon": 4, "fold": 0, "fit_end": "2021-01-04 12:00",
                 "score_start": str(p.origin.min()), "score_end": str(p.origin.max())}]
        result = run_shift_v2(history, p, meta, energy, _tariff(), fractions=(.2,), prep_minutes=(30,))
        d = result.actions.loc[result.actions.scenario.eq("D")].reset_index(drop=True)
        self.assertEqual(d.status.tolist(), ["destination_capacity_exceeded", "applied"])
        self.assertEqual(d.destination_start.iloc[1], pd.Timestamp("2021-01-07 08:30"))
        c = result.actions.loc[result.actions.scenario.eq("C")]
        self.assertEqual(c.status.tolist(), ["applied", "duplicate_after_applied"])

    def test_shift_15minute_destination_conservation_and_curtailment_upper_bound(self):
        idx = pd.date_range("2021-01-04 00:15", periods=96*3, freq="15min")
        history = pd.DataFrame({"power": 70., "production_target": 20.,
                                "temperature": 18., "time_repaired": False}, index=idx)
        history.loc[pd.Timestamp("2021-01-06 11:15"), "power"] = 95.
        targets = pd.date_range("2021-01-06 00:15", "2021-01-06 23:45", freq="15min")
        p = pd.DataFrame({"origin": targets-pd.Timedelta(hours=1),
                          "target_time": targets, "horizon": 4, "fold": 0,
                          "y": history.loc[targets, "power"].to_numpy(), "tau": 90.,
                          "p_exceed": 0., "q95_cal": 80.})
        alerted = p.target_time.isin(pd.to_datetime(["2021-01-06 10:15", "2021-01-06 10:30"]))
        p.loc[alerted, "p_exceed"] = .9
        p["alert"] = alerted
        energy = EnergyV2Result({0: FoldEnergyModelV2(0, None, 18., pd.Timestamp("2021-01-04"), 2.)},
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
        meta = [{"horizon": 4, "fold": 0, "fit_end": "2021-01-04 12:00",
                 "score_start": str(p.origin.min()), "score_end": str(p.origin.max())}]
        result = run_shift_v2(history, p, meta, energy, _tariff(), fractions=(.2,),
                              prep_minutes=(30, 60))
        d30 = result.actions.loc[result.actions.scenario.eq("D") & result.actions.prep_minutes.eq(30)]
        self.assertEqual(len(d30), 2)
        self.assertEqual(d30.status.iloc[0], "applied")
        self.assertEqual(d30.destination_start.iloc[0], pd.Timestamp("2021-01-06 11:15"))
        self.assertEqual(result.alerts.loc[result.alerts.prep_minutes.eq(30),
                                           "source_status"].tolist(),
                         ["source_eligible", "source_eligible"])
        self.assertEqual(result.actions.loc[result.actions.scenario.eq("D") &
                                             result.actions.prep_minutes.eq(30), "status"].tolist(),
                         ["applied", "duplicate_after_applied"])
        self.assertFalse(result.actions.loc[result.actions.prep_minutes.eq(60), "status"].eq("applied").any())
        monthly_d30 = result.monthly.loc[result.monthly.scenario.eq("D") &
                                         result.monthly.prep_minutes.eq(30)].iloc[0]
        self.assertAlmostEqual(monthly_d30.load_proxy_total_change, 0, places=10)
        self.assertAlmostEqual(monthly_d30.weighted_load_change, -16., places=10)
        self.assertEqual(monthly_d30.demand_scope, "partial_month_development_oof")
        self.assertEqual(monthly_d30.full_calendar_intervals, 31*96)
        self.assertAlmostEqual(monthly_d30.observed_fraction_of_full_month,
                               monthly_d30.evaluated_intervals/monthly_d30.full_calendar_intervals)
        self.assertEqual(set(result.summary.scenario_label), {"사후 관측 생산량 기반 사후 가정 시나리오"})
        self.assertEqual(set(result.actions.demand_scope), {"partial_month_development_oof"})
        self.assertEqual(set(result.alerts.demand_scope), {"partial_month_development_oof"})
        self.assertTrue(result.summary.actionable_fraction_denominator.str.contains("duplicate_sources").all())
        c30 = result.summary.loc[result.summary.scenario.eq("C") &
                                 result.summary.prep_minutes.eq(30)].iloc[0]
        self.assertEqual(c30.applied_unique_source_windows, 1)
        self.assertLess(c30.load_proxy_total_change, 0)

    def test_safe_history_rejects_test_boundary(self):
        history = pd.DataFrame({"power": [1.], "production_target": [1.]},
                               index=pd.DatetimeIndex([TEST_ORIGIN]))
        with self.assertRaisesRegex(ValueError, "development-only"):
            run_energy_baseline_v2(history, pd.DataFrame(), [], n_boot=1)


if __name__ == "__main__":
    unittest.main()
