"""Reference-day eligibility, trimming, and origin availability checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.cbl import cbl_all_predictions, cbl_predict


class CBLTests(unittest.TestCase):
    @staticmethod
    def fixture() -> pd.DataFrame:
        index = pd.date_range("2020-12-20 00:15", "2021-01-22 00:00", freq="15min", name="ts_end")
        df = pd.DataFrame(index=index)
        # Distinct daily values reveal which reference dates were selected.
        day = (index - pd.Timedelta(nanoseconds=1)).normalize()
        df["power"] = (day - pd.Timestamp("2020-12-20")).days.astype(float)
        df["time_repaired"] = False
        return df

    def test_mid_6_10_matches_last_ten_weekdays_and_trims_extremes(self):
        df = self.fixture()
        target = pd.DatetimeIndex([pd.Timestamp("2021-01-18 10:00")])
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-18 09:00")])
        result = cbl_predict(df, target, "mid_6_10", origin_times=origin)
        weekdays = pd.bdate_range("2021-01-04", "2021-01-15")
        values = [(day - pd.Timestamp("2020-12-20")).days for day in weekdays]
        self.assertAlmostEqual(result.iloc[0], np.mean(sorted(values)[2:-2]))

    def test_long_horizon_cannot_use_own_incomplete_origin_day(self):
        df = self.fixture()
        target = pd.DatetimeIndex([pd.Timestamp("2021-01-19 10:00")])
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-18 09:00")])
        before = cbl_predict(df, target, "mid_6_10", origin_times=origin).iloc[0]
        changed = df.copy()
        changed.loc[(changed.index >= pd.Timestamp("2021-01-18 00:15")) & (changed.index <= pd.Timestamp("2021-01-19 10:00")), "power"] = 9999
        after = cbl_predict(changed, target, "mid_6_10", origin_times=origin).iloc[0]
        self.assertEqual(before, after)

    def test_repaired_reference_is_skipped_and_holiday_pool_differs(self):
        df = self.fixture()
        target = pd.DatetimeIndex([pd.Timestamp("2021-01-18 10:00")])
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-18 09:00")])
        before = cbl_predict(df, target, "mid_6_10", origin_times=origin).iloc[0]
        changed = df.copy()
        changed.at[pd.Timestamp("2021-01-15 10:00"), "time_repaired"] = True
        after = cbl_predict(changed, target, "mid_6_10", origin_times=origin).iloc[0]
        self.assertNotEqual(before, after)
        holiday = cbl_predict(df, pd.DatetimeIndex([pd.Timestamp("2021-01-17 10:00")]),
                              "holiday_mid_4_6", origin_times=pd.DatetimeIndex([pd.Timestamp("2021-01-17 09:00")]))
        self.assertTrue(np.isfinite(holiday.iloc[0]))
        weekday = cbl_predict(df, target, "holiday_mid_4_6", origin_times=origin)
        self.assertTrue(np.isnan(weekday.iloc[0]))
        hybrid = cbl_all_predictions(df, origin, 4)
        self.assertEqual(hybrid.at[origin[0], "c3_holiday_mid_4_6"],
                         hybrid.at[origin[0], "c1_mid_6_10"])

    def test_same_day_adjustment_uses_only_pre_origin_window(self):
        df = self.fixture()
        target = pd.DatetimeIndex([pd.Timestamp("2021-01-18 10:00")])
        origin = pd.DatetimeIndex([pd.Timestamp("2021-01-18 09:00")])
        original = cbl_predict(df, target, "mid_6_10", True, origin).iloc[0]
        future = df.copy()
        future.loc[future.index > origin[0], "power"] = 9999
        self.assertEqual(original, cbl_predict(future, target, "mid_6_10", True, origin).iloc[0])
        recent = df.copy()
        recent.loc[origin[0] - pd.Timedelta(minutes=165):origin[0], "power"] += 12
        self.assertAlmostEqual(cbl_predict(recent, target, "mid_6_10", True, origin).iloc[0], original + 12)


if __name__ == "__main__":
    unittest.main()
