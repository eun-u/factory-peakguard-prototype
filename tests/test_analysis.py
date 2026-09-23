"""Behavioral checks for development-only analysis and operational timing."""

from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.analysis import run_analysis
from src.analysis.alerting import compare_alert_rules, decision_feed
from src.analysis.decision import relative_economic_value
from src.analysis.errors import episodes, match_episode_table
from src.analysis.tariff import classify_tariff, discount_factor
from src.analysis.supplemental import expected_exceedance_episodes, reliability_table
from src.analysis.importance import run_importance
from src.analysis.horizon_curve import paired_horizon_advantage, plot_horizon_curve


def tariff_schedule() -> dict:
    return {"bands": {"all": {"months": list(range(1, 13)), "intervals": [
        {"start": "00:00", "end": "11:00", "band": "off_peak"},
        {"start": "11:00", "end": "15:00", "band": "mid_peak"},
        {"start": "15:00", "end": "24:00", "band": "peak"}]}},
        "rates_confirmed": False,
        "scenario_weights": {"off_peak": 1, "mid_peak": 2, "peak": 3}}


class AnalysisBehaviorTests(unittest.TestCase):
    def test_decision_feed_does_not_mark_nonoperational_horizons_normal(self):
        origins = pd.date_range("2021-06-01 10:00", periods=5, freq="15min")
        frame = pd.DataFrame({"origin": origins, "target_time": origins+pd.Timedelta(hours=1),
                              "horizon": [1, 4, 4, 16, 96], "fold": [1]*5,
                              "pred": [9.]*5, "tau": [10.]*5,
                              "p_exceed": [.9, .9, np.nan, .9, .01],
                              "q95_cal": [12., 12., np.nan, 12., 8.]})
        feed = decision_feed(frame, ratio=.1)
        nonoperational = feed.loc[feed.horizon.isin([1, 96])]
        self.assertTrue(nonoperational.alert_level.isna().all())
        self.assertFalse(nonoperational.action_applicable.any())
        self.assertEqual(nonoperational.risk_exceeded.tolist(), [True, False])
        actionable = feed.loc[feed.action_applicable].sort_values("origin")
        self.assertEqual(actionable.alert_level.tolist()[0], 2)
        self.assertTrue(pd.isna(actionable.alert_level.tolist()[1]))
        self.assertEqual(actionable.alert_level.tolist()[2], 1)
        self.assertTrue(pd.isna(actionable.risk_exceeded.tolist()[1]))
        self.assertEqual(str(feed.alert_level.dtype), "Int64")

    def test_horizon_curve_uses_no_holiday_comparator_when_baseline_wins(self):
        times = pd.date_range("2021-06-01 10:00", periods=8, freq="15min")
        rows = []
        for name, value in (("p1_latest", 2.), ("lgbm", 2.2), ("lgbm_no_holiday", 2.4)):
            rows.append(pd.DataFrame({"origin": times-pd.Timedelta(hours=1),
                                      "target_time": times, "fold": 1, "horizon": 4,
                                      "model": name, "y": [3.]*8,
                                      "pred": [value]*8, "tau": [2.]*8}))
        prediction = pd.concat(rows, ignore_index=True)
        advantages = paired_horizon_advantage(prediction, {4: "p1_latest"}, 10, 42,
                                               {4: {"persistence": "p1_latest"}})
        self.assertEqual(set(advantages.model), {"lgbm_no_holiday"})
        self.assertEqual(set(advantages.comparison_role), {"development_comparator_not_selected"})
        plot_table = pd.DataFrame({"horizon": [4, 4, 4], "minutes": [60]*3,
                                   "model": ["p1_latest", "lgbm", "lgbm_no_holiday"],
                                   "selected": [True, False, False],
                                   "peak_mae": [1., .8, .6], "episode_f1": [.7, .8, .9]})
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_horizon_curve(plot_table, Path(tmp)/"horizon.png",
                                      {4: {"persistence": "p1_latest"}})
            self.assertTrue(path.is_file())

    def test_importance_labels_saved_lgbm_when_baseline_wins(self):
        origins = pd.date_range("2021-06-01 00:00", periods=32, freq="15min")
        values = np.arange(len(origins), dtype=float)
        features = pd.DataFrame({"lag_power": values}, index=origins)
        target = values + 1
        model = DummyRegressor().fit(features, target)
        score = pd.DataFrame({"origin": origins, "target_time": origins+pd.Timedelta(hours=1),
                              "horizon": 4, "fold": 1, "model": "lgbm_no_holiday",
                              "y": target, "pred": model.predict(features), "tau": 20.})
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)/"models"
            model_dir.mkdir()
            joblib.dump({"model": model, "horizon": 4, "fold": 1,
                         "model_name": "lgbm_no_holiday", "tau": 20.,
                         "feature_names": ["lag_power"]},
                        model_dir/"development_h4_fold1_lgbm_no_holiday.joblib")
            with patch("src.features.build_features", return_value=(
                features, pd.Series(origins-pd.Timedelta(minutes=15), index=origins))):
                result = run_importance(features, score, {"seed": 7}, Path(tmp), model_dir,
                                        selected_models={4: "cbl_mid_6_10"})
            self.assertEqual(result["status"], "ok")
            top = pd.read_csv(Path(tmp)/"tables"/"permutation_importance_top10.csv")
            self.assertEqual(top.attribution_role.iloc[0], "development_comparator_not_selected")
            self.assertEqual(top.model.iloc[0], "lgbm_no_holiday")
            self.assertEqual(top.selected_model.iloc[0], "cbl_mid_6_10")

    def test_expected_exceedance_uses_actual_episode_max_and_signed_error(self):
        times = pd.date_range("2021-06-01 10:00", periods=6, freq="15min")
        frame = pd.DataFrame({"target_time": times, "fold": 1,
                              "y": [0, 3, 4, 0, 5, 0], "tau": [2]*6,
                              "exp_exceed": [.1, .4, .9, .1, 1.2, .1]})
        table, summary = expected_exceedance_episodes(frame, n_boot=10)
        self.assertEqual(summary["evaluated_episodes"], 2)
        self.assertEqual(table.actual_max_excess.tolist(), [2., 3.])
        self.assertEqual(table.predicted_max_pointwise_excess.tolist(), [.9, 1.2])
        self.assertAlmostEqual(summary["mean_signed_error"], -1.45)
        self.assertAlmostEqual(summary["mae"], 1.45)

    def test_reliability_bins_and_scores_use_same_valid_rows(self):
        from sklearn.metrics import average_precision_score, brier_score_loss

        frame = pd.DataFrame({
            "target_time": pd.to_datetime(["2021-01-01 10:00", "2021-01-01 10:15",
                                           "2021-01-02 10:00", "2021-01-02 10:15",
                                           "2021-01-03 10:00"]),
            "y": [0, 2, 2, 0, 2], "tau": [1]*5,
            "p_exceed": [0, 0.5, 1, 0.75, np.nan],
        })
        table, summary = reliability_table(frame, n_boot=25, seed=7, n_bins=2)
        valid_y = np.array([0, 1, 1, 0])
        valid_p = np.array([0, .5, 1, .75])
        self.assertEqual(table.n.sum(), summary["n"])
        self.assertEqual(table.events.sum(), summary["events"])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["events"], 2)
        self.assertAlmostEqual(summary["brier"], brier_score_loss(valid_y, valid_p))
        self.assertAlmostEqual(summary["pr_auc"], average_precision_score(valid_y, valid_p))
        self.assertEqual(int(table.loc[table.bin.eq(1), "n"].iloc[0]), 3)
        self.assertTrue(table.observed_rate_ci_low.dropna().between(0, 1).all())
        self.assertTrue(table.observed_rate_ci_high.dropna().between(0, 1).all())

    def test_reliability_rejects_invalid_probabilities(self):
        frame = pd.DataFrame({"target_time": pd.to_datetime(["2021-01-01"]),
                              "y": [2], "tau": [1], "p_exceed": [1.01]})
        with self.assertRaisesRegex(ValueError, "outside"):
            reliability_table(frame)

    def test_tariff_uses_interval_start_for_hour_boundary(self):
        bands = classify_tariff(pd.to_datetime(["2021-06-01 11:00", "2021-06-01 11:15"]), tariff_schedule())
        self.assertEqual(bands.tolist(), ["off_peak", "mid_peak"])

    def test_weekend_discount_applies_to_interval_start_only(self):
        tariff = tariff_schedule()
        tariff["weekend_day_discount"] = {"months": [5], "start": "11:00", "end": "14:00", "factor": .5}
        times = pd.to_datetime(["2021-05-01 11:00", "2021-05-01 11:15", "2021-05-03 11:15"])
        self.assertEqual(discount_factor(times, tariff).tolist(), [1.0, .5, 1.0])

    def test_episode_gap_creates_new_event(self):
        times = pd.to_datetime(["2021-01-01 01:00", "2021-01-01 01:15", "2021-01-01 01:45"])
        self.assertEqual(len(episodes(times, [True, True, True])), 2)

    def test_episode_matching_prioritizes_number_detected(self):
        times = pd.date_range("2021-01-01", periods=21, freq="15min")
        actual = np.zeros(21, dtype=bool)
        actual[:11] = True
        actual[12:] = True
        alert = np.zeros(21, dtype=bool)
        alert[:4] = True
        alert[6:16] = True
        frame = pd.DataFrame({"target_time": times, "y": actual.astype(float)*2,
                              "tau": np.ones(21), "alert_flag": alert})
        events = match_episode_table(frame)
        self.assertEqual(events.status.eq("TP").sum(), 2)

    def test_long_alarm_covering_two_peaks_marks_assignment_miss(self):
        times = pd.date_range("2021-06-01 18:00", periods=7, freq="15min")
        frame = pd.DataFrame({"target_time": times,
                              "y": [3, 3, 0, 4, 4, 0, 3], "tau": [2]*7,
                              "alert_flag": [True, True, True, True, True, False, False]})
        events = match_episode_table(frame)
        missed = events.loc[events.status.eq("FN")].sort_values("start")
        self.assertEqual(events.status.eq("TP").sum(), 1)
        self.assertEqual(missed.missed_reason.tolist(), ["one_to_one_assignment", "no_overlap"])
        self.assertEqual(missed.overlapping_alarm_count.tolist(), [1., 0.])
        # Four of five peak positions are alarmed despite only one matched episode.
        self.assertEqual(int((frame.y.gt(frame.tau) & frame.alert_flag).sum()), 4)

    def test_perfect_probabilities_have_unit_relative_value(self):
        frame = pd.DataFrame({"y": [0, 2, 0, 2], "tau": [1]*4,
                              "p_exceed": [0, 1, 0, 1]})
        self.assertAlmostEqual(relative_economic_value(frame, .2), 1.0)

    def test_confirmation_reduces_episode_lead_time(self):
        targets = pd.date_range("2021-06-01 10:00", periods=3, freq="15min")
        frame = pd.DataFrame({"origin": targets-pd.Timedelta(hours=1),
                              "target_time": targets, "horizon": 4, "fold": 0,
                              "y": [0., 2., 0.], "tau": [1.]*3,
                              "p_exceed": [.9, .9, 0.], "q95_cal": [2., 2., 0.]})
        rules = compare_alert_rules(frame, ratios=[.1], prep_minutes=[30])
        one = rules.loc[rules.rule.eq("1/1")].iloc[0]
        two = rules.loc[rules.rule.eq("2/2")].iloc[0]
        self.assertGreater(one.mean_actionable_lead_minutes, two.mean_actionable_lead_minutes)
        self.assertEqual(one.false_alarm_positions_per_operating_hour, 4/3)

    def test_pipeline_rejects_training_overlap_with_oof(self):
        idx = pd.date_range("2021-01-01 00:15", periods=120, freq="15min")
        history = pd.DataFrame({"power": np.arange(120)+1,
                                "production_target": np.ones(120),
                                "production_known": np.ones(120)}, index=idx)
        pred = pd.DataFrame({"origin": [idx[104]], "target_time": [idx[108]],
                             "horizon": [4], "fold": [1], "model": ["lgbm"],
                             "y": [109], "pred": [107], "tau": [100]})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "training prefix"):
                run_analysis(history, pred, {"primary_horizon": 4}, tmp,
                             train_df=history.loc[:idx[110]])

    def test_frozen_scope_requires_completed_record_and_development_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "completed freeze record"):
                run_analysis(pd.DataFrame(), pd.DataFrame(), {}, tmp,
                             scope="frozen_test", selection={})
            record = Path(tmp, "logs", "freeze_record.json")
            record.parent.mkdir(parents=True)
            record.write_text('{"status":"reserved"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed freeze record"):
                run_analysis(pd.DataFrame(), pd.DataFrame(), {}, tmp,
                             scope="frozen_test", selection={}, frozen_record=record)

    def test_frozen_scope_reads_only_hashed_predictions_into_separate_directory(self):
        idx = pd.date_range("2021-01-01 00:15", periods=96*8, freq="15min")
        prod = 8+np.arange(len(idx)) % 10
        power = 20+prod/4+7*np.sin(np.arange(len(idx))*2*np.pi/96)
        history = pd.DataFrame({"power": power, "production_target": prod,
                                "production_known": prod, "temperature": 18.}, index=idx)
        train = history.loc[history.index < "2021-01-05"]
        origins = idx[(idx >= pd.Timestamp("2021-01-05")) & (idx < pd.Timestamp("2021-01-07"))]
        targets = origins+pd.Timedelta(hours=1)
        y = history.loc[targets, "power"].to_numpy()
        point = pd.DataFrame({"origin": origins, "target_time": targets, "horizon": 4,
                              "fold": -1, "model": "lgbm", "y": y, "pred": y-1,
                              "tau": 25., "alert": y > 25})
        risk = point.assign(model="lgbm_quantile_a", q50=y-1, q95_cal=y+3,
                            p_exceed=np.where(y > 25, .8, .1))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for folder in ("logs", "predictions", "tables", "models"):
                (base/folder).mkdir()
            files = {
                "holdout_prediction_sha256": base/"predictions"/"final_test_predictions.csv",
                "final_metrics_sha256": base/"tables"/"final_test.csv",
                "model_sha256": base/"models"/"frozen_final.joblib",
            }
            pd.concat([point, risk]).to_csv(files["holdout_prediction_sha256"], index=False)
            files["final_metrics_sha256"].write_text("synthetic,1\n", encoding="utf-8")
            files["model_sha256"].write_bytes(b"synthetic")
            record = {"status": "completed", "test_start_origin": str(origins.min())}
            record.update({name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()})
            lock = base/"logs"/"freeze_record.json"
            lock.write_text(json.dumps(record), encoding="utf-8")
            cfg = {"primary_horizon": 4, "bootstrap": {"n": 10},
                   "freeze": {"not_before": "2020-01-01T00:00:00+09:00"}}
            result = run_analysis(history, None, cfg, base, train_df=train,
                                  selection={"by_horizon": {"4": {"point_model": "lgbm", "conformal": "a"}}},
                                  tariff_2021=tariff_schedule(), tariff_2026=tariff_schedule(),
                                  scope="frozen_test", frozen_record=lock)
            self.assertEqual(result["scope"], "frozen_test")
            self.assertTrue((base/"final_analysis"/"logs"/"analysis_summary.json").is_file())
            self.assertFalse((base/"tables"/"error_conditions.csv").exists())
            feed = pd.read_csv(base/"final_analysis"/"predictions"/"decision_feed.csv")
            self.assertEqual(set(feed.scenario), {"frozen_test"})
            files["holdout_prediction_sha256"].write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_analysis(history, None, cfg, base, train_df=train,
                             selection={"by_horizon": {"4": {"point_model": "lgbm"}}},
                             scope="frozen_test", frozen_record=lock)

    def test_evening_alert_cannot_shift_to_elapsed_midday(self):
        from src.analysis.energy_baseline import fit_energy_baseline
        from src.analysis.simulate_shift import simulate_shift

        idx = pd.date_range("2021-01-01 00:15", periods=96*8, freq="15min")
        production = 12.0 + np.arange(len(idx)) % 8
        power = 50 + production/4*2 + 5*np.sin(np.arange(len(idx))*2*np.pi/96)
        history = pd.DataFrame({"power": power, "production_target": production,
                                "temperature": np.full(len(idx), 18.)}, index=idx)
        train = history.loc[history.index < "2021-01-06"].copy()
        baseline = fit_energy_baseline(train)
        origin = pd.Timestamp("2021-01-07 17:00")
        target = origin+pd.Timedelta(hours=1)
        pred = pd.DataFrame({"origin": [origin], "target_time": [target],
                             "horizon": [4], "tau": [45.], "p_exceed": [.9],
                             "q95_cal": [70.]})
        actions, summary = simulate_shift(pred, history, train, baseline,
                                          tariff_schedule(), fractions=[.2])
        self.assertEqual(actions.status.iloc[0], "no_future_low_band_window")
        self.assertTrue(summary.max_change.eq(0).all())

    def test_development_pipeline_emits_computed_tables_without_test(self):
        idx = pd.date_range("2021-01-01 00:15", periods=96*9, freq="15min")
        production = 8.0 + np.arange(len(idx)) % 12
        power = 20 + production/4*1.5 + 7*np.sin(np.arange(len(idx))*2*np.pi/96)
        history = pd.DataFrame({"power": power, "production_target": production,
                                "production_known": production,
                                "temperature": np.full(len(idx), 18.)}, index=idx)
        train = history.loc[history.index < "2021-01-05"].copy()
        origins = idx[(idx >= pd.Timestamp("2021-01-05")) & (idx < pd.Timestamp("2021-01-08"))]
        target_times = origins+pd.Timedelta(hours=1)
        y = history.loc[target_times, "power"].to_numpy()
        pred = pd.DataFrame({"origin": origins, "target_time": target_times,
                             "horizon": 4, "fold": 1, "model": "lgbm",
                             "y": y, "pred": y-1, "tau": 30.,
                             "q50": y-1, "q95_cal": y+3,
                             "p_exceed": np.where(y > 30, .8, .1),
                             "alert": y > 30})
        risk = pred.assign(model="lgbm_quantile_a", pred=y-.5,
                           q50=y-.5, q95_cal=y+3, p_exceed=np.where(y > 30, .8, .1))
        pred = pd.concat([pred, risk], ignore_index=True)
        cfg = {"primary_horizon": 4, "bootstrap": {"n": 20},
               "decision": {"cl_ratios": [.1, .3]},
               "shift": {"fractions": [.2]}, "alert": {"prep_minutes": [30]}}
        with tempfile.TemporaryDirectory() as tmp:
            result = run_analysis(history, pred, cfg, tmp, train_df=train,
                                  tariff_2021=tariff_schedule(), tariff_2026=tariff_schedule())
            self.assertEqual(result["scope"], "development_oof")
            self.assertTrue(Path(tmp, "tables", "error_conditions.csv").is_file())
            self.assertTrue(Path(tmp, "tables", "relative_economic_value.csv").is_file())
            self.assertTrue(Path(tmp, "tables", "probability_reliability.csv").is_file())
            self.assertTrue(Path(tmp, "figures", "F2_probability_reliability.png").is_file())
            self.assertTrue(Path(tmp, "figures", "F4-1_forecast_alert_action.png").is_file())
            self.assertTrue(Path(tmp, "logs", "analysis_summary.json").is_file())
            feed = pd.read_csv(Path(tmp, "predictions", "decision_feed.csv"))
            self.assertTrue(feed.p_exceed.notna().all())


if __name__ == "__main__":
    unittest.main()
