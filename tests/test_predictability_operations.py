"""Synthetic safety and operating tests for P1 A4/A5."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analysis.predictability_operations import (
    _EnergyModels, _paired_original_models, _simulate_operational, run_a4,
)
from src.analysis.energy_baseline_v2 import FoldEnergyModelV2
from src.session_data import SEALED_BOUNDARY


def _tariff() -> dict:
    return {"bands": {"all": {"months": list(range(1, 13)), "intervals": [
        {"start": "00:00", "end": "11:00", "band": "peak"},
        {"start": "11:00", "end": "24:00", "band": "off_peak"}]}},
        "scenario_weights": {"off_peak": 1, "peak": 3}}


class PredictabilityOperationsTests(unittest.TestCase):
    def test_a4_joins_original_models_by_full_oof_key_and_detects_relative_restart_loss(self):
        dates = pd.to_datetime(["2021-05-04", "2021-05-10", "2021-05-17"])
        all_times = pd.DatetimeIndex([d+pd.Timedelta(hours=10, minutes=15*j)
                                      for d in dates for j in range(1, 9)])
        history = pd.DataFrame({"power": 100., "production_target": 20.,
                                "time_repaired": False}, index=all_times)
        oof_rows, meta = [], []
        for horizon in (16, 96):
            for fold, date in enumerate(dates):
                targets = all_times[all_times.normalize() == date]
                origins = targets-pd.Timedelta(minutes=15*horizon)
                meta.append({"horizon": horizon, "fold": fold,
                             "fit_end": str(origins.min()-pd.Timedelta(days=1)),
                             "score_start": str(origins.min()), "score_end": str(origins.max())})
                for model in ("c3_holiday_hybrid", "lgbm"):
                    for j, (origin, target) in enumerate(zip(origins, targets)):
                        # The first interval of each fold is a restart.
                        error = 1. if model.startswith("c3") else (3. if j == 0 else .5)
                        oof_rows.append({"origin": origin, "target_time": target,
                                         "horizon": horizon, "fold": fold, "model": model,
                                         "y": 100., "pred": 100.-error, "tau": 90.})
        manifest = {"folds": meta, "selection": {"by_horizon": {
            "16": {"cbl": "c3_holiday_hybrid"}, "96": {"cbl": "c3_holiday_hybrid"}}}}
        oof = pd.DataFrame(oof_rows).sample(frac=1, random_state=7)

        def thresholds(_history, fit_end):
            return {"fit_end": fit_end}

        def events(_history, _thresholds, *, start, end):
            return pd.DataFrame(index=pd.DatetimeIndex([start], name="ts_end"))

        result = run_a4(history, oof, manifest, (thresholds, events))
        self.assertEqual(len(result["hypotheses"]), 2)
        self.assertTrue(np.allclose(result["hypotheses"].estimate, 2.5))
        self.assertTrue(result["hypotheses"].eligible.all())
        self.assertTrue(result["hypotheses"].n_positive_folds.eq(3).all())
        self.assertEqual(len(result["paired"]), 48)
        self.assertEqual(set(result["hypotheses"].hypothesis_id), {"A4_h16", "A4_h96"})

        contaminated = history.copy()
        contaminated.loc[SEALED_BOUNDARY, "power"] = 9999.
        with self.assertRaisesRegex(ValueError, "development-only"):
            _paired_original_models(oof, manifest, 16, contaminated)

    def test_h16_action_uses_original_shift_constraints_and_keeps_day_episodes_separate(self):
        idx = pd.date_range("2021-01-04 00:15", "2021-01-06 23:45", freq="15min")
        history = pd.DataFrame({"power": 70., "production_target": 20.,
                                "temperature": 18., "time_repaired": False}, index=idx)
        peak = pd.Timestamp("2021-01-06 10:15")
        history.loc[peak, "power"] = 95.
        targets = pd.date_range("2021-01-06 00:15", "2021-01-06 23:45", freq="15min")
        p = pd.DataFrame({"origin": targets-pd.Timedelta(hours=4),
                          "target_time": targets, "horizon": 16, "fold": 0,
                          "y": history.loc[targets, "power"].to_numpy(),
                          "pred": 70., "tau": 90., "p_exceed": 0., "q95_cal": 80.})
        p.loc[p.target_time.eq(peak), "p_exceed"] = .9
        p["alert"] = p.p_exceed.gt(.1)
        p["alarm_source"] = "original_operational"
        meta = {0: {"horizon": 16, "fold": 0,
                    "fit_end": "2021-01-04 12:00",
                    "score_start": str(p.origin.min()), "score_end": str(p.origin.max())}}
        energy = _EnergyModels({0: FoldEnergyModelV2(
            0, None, 18., pd.Timestamp("2021-01-04 12:00"), 2.)})
        outputs = list(_simulate_operational(history, p, meta, energy,
                                              _tariff(), prep=30))
        self.assertEqual(len(outputs), 2)
        audit, d_actions, d_months, d = outputs[0]
        self.assertEqual(audit.source_status.tolist(), ["source_eligible"])
        self.assertEqual(d_actions.status.tolist(), ["applied"])
        self.assertEqual(d_actions.destination_start.iloc[0], pd.Timestamp("2021-01-06 11:00"))
        self.assertGreaterEqual(d_actions.destination_start.iloc[0],
                                d_actions.origin.iloc[0]+pd.Timedelta(minutes=30))
        self.assertEqual(d["actual_peak_unique_source_hours"], 1)
        self.assertEqual(d["applied_actual_peak_source_hours"], 1)
        self.assertEqual(d["new_peak_positions"], 0)
        self.assertAlmostEqual(d["load_proxy_total_change"], 0., places=10)
        self.assertEqual(d_months.observed_fraction_of_full_month.iloc[0],
                         len(targets)/(31*96))
        self.assertEqual(d["weight_basis"], "illustrative_ordinal_scenario")

        oracle = p.copy()
        oracle["alarm_source"] = "perfect_information_reference"
        oracle["alert"] = oracle.y.gt(oracle.tau)
        oracle_outputs = list(_simulate_operational(history, oracle, meta, energy,
                                                     _tariff(), prep=60))
        self.assertEqual(oracle_outputs[0][3]["alert_positions"], 1)
        too_late = p.copy()
        too_late["origin"] = too_late.target_time-pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "horizon"):
            list(_simulate_operational(history, too_late, meta, energy, _tariff(), prep=30))


if __name__ == "__main__":
    unittest.main()


def test_operating_analysis_uses_selected_research_risk(monkeypatch):
    import src.analysis.predictability_operations as operations
    target = pd.Timestamp("2021-05-03 10:15")
    common = {"origin": target-pd.Timedelta(hours=4), "target_time": target,
              "horizon": 16, "fold": 0, "y": 95., "pred": 70., "tau": 90., "alert": False}
    oof = pd.DataFrame([{**common, "model": "c3"},
                        {**common, "model": "lgbm_quantile_b", "p_exceed": 0., "q95_cal": 80.},
                        {**common, "model": "q95_rolling_672", "p_exceed": .2, "q95_cal": 96.}])
    manifest = {"selection": {"by_horizon": {"16": {
        "point_model": "c3", "conformal": "b", "risk_model": "q95_rolling_672"}}}}
    monkeypatch.setattr(operations, "_fold_meta", lambda *_: {})
    monkeypatch.setattr(operations, "_verify_oof", lambda frame, *_: frame)
    result = operations._prepare_operational(oof, manifest, 16, pd.DataFrame())
    assert result.risk_model.tolist() == ["q95_rolling_672"]
    assert result.alert.tolist() == [True]
