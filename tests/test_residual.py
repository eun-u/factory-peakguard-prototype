"""Origin availability and fixed-fold residual candidate regression tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.cbl import cbl_all_predictions
from src.models.residual import fit_residual_fold, residual_baseline
from src.session_data import SEALED_BOUNDARY


class ResidualTests(unittest.TestCase):
    def test_fixed_cbl_matches_existing_baseline_and_future_values_do_not_change_it(self):
        idx = pd.date_range("2021-01-01 00:15", "2021-03-16 00:00", freq="15min")
        signal = 85+10*np.sin(np.arange(len(idx))*2*np.pi/96)
        signal += np.where(idx.dayofweek < 5, 4, 0)
        history = pd.DataFrame({"power": signal, "time_repaired": False}, index=idx)
        origins = pd.DatetimeIndex([pd.Timestamp("2021-03-08 11:15"),
                                    pd.Timestamp("2021-03-09 11:30"),
                                    pd.Timestamp("2021-03-13 08:00")])
        for horizon, expected_col in ((4, "c2a_max_4_5_adjusted"),
                                      (16, "c3_holiday_mid_4_6"),
                                      (96, "c3_holiday_mid_4_6")):
            baseline = residual_baseline(history, origins, horizon)
            existing = cbl_all_predictions(history, origins, horizon)[expected_col]
            np.testing.assert_allclose(baseline, existing, equal_nan=True, atol=1e-10)
            self.assertTrue((pd.DatetimeIndex(baseline.attrs["latest_observation"]) <= origins).all())
            # Alter a value after the latest origin, including a value at a
            # target time. An origin-available baseline must remain unchanged.
            perturbed = history.copy()
            perturbed.loc[perturbed.index > origins.max(), "power"] = 99999.
            np.testing.assert_allclose(residual_baseline(perturbed, origins, horizon),
                                       baseline, equal_nan=True, atol=1e-10)

    def test_fit_uses_residual_labels_fixed_params_and_preserves_original_windows(self):
        origins = pd.date_range("2021-02-01 00:15", periods=160, freq="15min", name="origin")
        fit, stop, cal, score = (origins[i:i+40] for i in (0, 40, 80, 120))
        x = pd.DataFrame({"current": np.linspace(0, 1, len(origins)),
                          "is_offday": 0, "target_hour": origins.hour}, index=origins)
        x.attrs["latest_observation_by_feature"] = {
            "current": pd.Series(origins, index=origins),
            "is_offday": pd.Series(pd.NaT, index=origins),
            "target_hour": pd.Series(pd.NaT, index=origins)}
        target = pd.DataFrame({"target_time": origins+pd.Timedelta(hours=1),
                               "y": np.full(len(origins), 100.)}, index=origins)
        history = pd.DataFrame({"power": np.full(len(origins), 100.),
                                "time_repaired": False}, index=origins)
        cfg = {"seed": 42, "lgbm": {"grid": [{"num_leaves": 15,
                                               "min_child_samples": 40}]}}
        context = {"fit": fit, "stop": stop, "cal": cal, "score": score,
                   "x": x, "targets": target, "tau": 90.,
                   "expected": {"horizon": 4, "fold": 0, "tau": 90.,
                                "chosen_params": cfg["lgbm"]["grid"][0]}}
        observed = {}

        class _Model:
            best_iteration_ = 12

            def predict(self, features):
                self_features = features
                observed.setdefault("predicted_feature_columns", self_features.columns.tolist())
                return np.full(len(features), 20.)

        def fake_fit(x_fit, y_fit, x_stop, y_stop, used_cfg, **kwargs):
            observed["train_residual"] = y_fit.to_numpy(float)
            observed["stop_residual"] = y_stop.to_numpy(float)
            observed["fit_columns"] = x_fit.columns.tolist()
            observed["params"] = kwargs["params"]
            observed["peak_weight"] = kwargs["peak_weight"]
            return _Model()

        def known_baseline(_history, selected_origins, _horizon):
            values = pd.Series(80., index=pd.DatetimeIndex(selected_origins), name="residual_baseline")
            values.iloc[2] = np.nan
            values.attrs["latest_observation"] = pd.Series(values.index, index=values.index)
            return values

        with patch("src.models.residual.residual_baseline", side_effect=known_baseline), patch(
                "src.models.residual.fit_point", side_effect=fake_fit):
            result = fit_residual_fold(history, cfg, context, 4, 0)
        self.assertEqual(result.metadata["n_fit"], 39)
        self.assertEqual(result.metadata["original_n_fit"], 40)
        self.assertEqual(result.metadata["n_score"], 40)
        self.assertEqual(len(result.predictions), 40)
        np.testing.assert_allclose(observed["train_residual"], 20.)
        np.testing.assert_allclose(observed["stop_residual"], 20.)
        np.testing.assert_allclose(result.predictions.pred, 100.)
        self.assertEqual(observed["params"], cfg["lgbm"]["grid"][0])
        self.assertEqual(observed["peak_weight"], 1.)
        self.assertEqual(observed["fit_columns"], ["current", "target_hour", "residual_baseline"])
        self.assertTrue(result.predictions.reference_only.all())
        self.assertFalse(result.predictions.selection_used.any())

    def test_rejects_holdout_origin_before_parsing_baseline(self):
        index = pd.DatetimeIndex([pd.Timestamp("2021-08-09 09:30")])
        history = pd.DataFrame({"power": [1.], "time_repaired": [False]}, index=index)
        with self.assertRaisesRegex(ValueError, "sealed"):
            residual_baseline(history, pd.DatetimeIndex([SEALED_BOUNDARY]), 16)


if __name__ == "__main__":
    unittest.main()
