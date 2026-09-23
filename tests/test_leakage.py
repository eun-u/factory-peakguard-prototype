"""Behavioral checks that future observations cannot alter an origin forecast."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import build_features
from src.split import assert_purged, make_splits
from src.targets import next_day_max_targets, point_targets


def hourly_fixture(days: int = 35) -> pd.DataFrame:
    index = pd.date_range("2021-01-01 00:15", periods=days * 96, freq="15min", name="ts_end")
    df = pd.DataFrame(index=index)
    df["power"] = 90 + 10 * np.sin(np.arange(len(df)) / 8)
    df["time_repaired"] = False
    df["production_completed"] = np.where(index.minute == 0, 10 + index.hour, np.nan)
    df["production_completed_bad"] = False
    df["production_known"] = df["production_completed"].ffill()
    df["production_known_bad"] = df["production_known"].isna()
    df["production_target"] = 1000 + np.arange(len(df))  # analysis only
    df["temperature"] = 1000 + np.arange(len(df))
    return df


class LeakageTests(unittest.TestCase):
    def test_future_power_production_weather_and_target_never_enter_features(self):
        df = hourly_fixture()
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-25 10:45")])
        for horizon in (1, 4, 16, 96):
            before, max_used = build_features(df, origin, horizon, {"_tau": 95, "_include_cbl": True})
            self.assertTrue(before.attrs["valid_mask"].all())
            self.assertLessEqual(max_used.iloc[0], origin[0])
            self.assertNotIn("production_target", before.columns)
            self.assertNotIn("temperature", before.columns)
            changed = df.copy()
            changed.loc[changed.index > origin[0], "power"] = 999999
            changed.loc[changed.index > origin[0], "production_completed"] = 999999
            changed.loc[changed.index > origin[0], "production_known"] = 999999
            changed.loc[:, "production_target"] = 999999
            changed.loc[:, "temperature"] = 999999
            after, _ = build_features(changed, origin, horizon, {"_tau": 95, "_include_cbl": True})
            pd.testing.assert_frame_equal(before, after, check_like=True, check_flags=False)

    def test_repaired_dependencies_are_ineligible_without_removing_zero(self):
        df = hourly_fixture()
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-25 10:45")])
        df.loc[origin[0] - pd.Timedelta(days=7), "time_repaired"] = True
        x, _ = build_features(df, origin, 4, {"_tau": 95, "_include_cbl": False})
        self.assertFalse(x.attrs["valid_mask"].iloc[0])
        self.assertTrue(np.isnan(x.at[origin[0], "lag_672"]))
        df.loc[origin[0] - pd.Timedelta(days=7), "time_repaired"] = False
        df.loc[origin[0], "power"] = 0
        x2, _ = build_features(df, origin, 4, {"_tau": 95, "_include_cbl": False})
        self.assertTrue(x2.attrs["valid_mask"].iloc[0])
        self.assertEqual(x2.at[origin[0], "current"], 0)

    def test_purged_boundaries_and_fixed_external_test(self):
        origins = pd.date_range("2021-01-01 00:15", periods=96 * 50, freq="15min")
        for horizon in (1, 4, 16, 96):
            folds, test = make_splits(origins, horizon)
            self.assertEqual(test.min(), origins[int(len(origins) * .85)])
            assert_purged(folds, test, horizon)
            for fold in folds:
                self.assertLess(fold.train.max() + pd.Timedelta(minutes=15 * horizon), fold.validation.min())
            dev_folds, empty_test = make_splits(origins[:int(len(origins) * .85)], horizon, dev_frac=1.0)
            self.assertEqual(len(empty_test), 0)
            assert_purged(dev_folds, empty_test, horizon)
            self.assertLessEqual(dev_folds[-1].validation.max() + pd.Timedelta(minutes=15 * horizon),
                                 origins[int(len(origins) * .85) - 1])

    def test_targets_are_only_read_for_labels(self):
        df = hourly_fixture()
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-20 23:45")])
        label = point_targets(df, origin, 4)
        self.assertEqual(label.at[origin[0], "target_time"], pd.Timestamp("2021-01-21 00:45"))
        daily = next_day_max_targets(df, origin)
        self.assertTrue(daily.at[origin[0], "valid_target"])
        self.assertEqual(daily.at[origin[0], "target_time"], pd.Timestamp("2021-01-22 00:00"))


if __name__ == "__main__":
    unittest.main()
