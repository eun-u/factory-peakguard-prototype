import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from prototype.forecast import HORIZONS, MODEL_NAMES, decide, load_series, make_features, run_snapshot


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "task05_power" / "source.zip"


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = load_series(DATA_PATH)

    def test_source_is_expanded_without_gaps(self):
        series = self.series
        self.assertEqual(len(series), 6168 * 4)
        self.assertEqual(series.index.min(), pd.Timestamp("2021-01-01 00:15"))
        self.assertEqual(series.index.max(), pd.Timestamp("2021-09-15 00:00"))
        self.assertTrue((series.index.to_series().diff().dropna() == timedelta(minutes=15)).all())
        self.assertEqual(int(series["hour_recovered"].sum()), 48 * 4)
        self.assertEqual(int(series["missing_power"].sum()), 0)

    def test_features_at_origin_ignore_later_power(self):
        cutoff = pd.Timestamp("2021-06-30 12:00")
        before = make_features(self.series, 4).loc[cutoff]
        changed = self.series.copy()
        changed.loc[changed.index > cutoff, "power"] = 999.0
        after = make_features(changed, 4).loc[cutoff]
        pd.testing.assert_series_equal(before, after)

    def test_historical_snapshot_ignores_all_later_observations(self):
        cutoff = pd.Timestamp("2021-06-30 12:00")
        original = run_snapshot(self.series, cutoff)
        changed = self.series.copy()
        changed.loc[changed.index > cutoff, "power"] = 999.0
        alternative = run_snapshot(changed, cutoff)
        self.assertEqual(original["train_end"], alternative["train_end"])
        self.assertEqual(original["threshold"], alternative["threshold"])
        pd.testing.assert_frame_equal(original["metrics"], alternative["metrics"])
        for model in MODEL_NAMES:
            for horizon in HORIZONS:
                for key in ("prediction", "lower", "upper", "radius"):
                    self.assertEqual(
                        original["results"][model][horizon][key],
                        alternative["results"][model][horizon][key],
                    )

    def test_four_decision_states(self):
        def sample(value, radius=5.0, quality=None):
            return {
                "current": 70.0,
                "threshold": 100.0,
                "quality_reasons": quality or [],
                "results": {
                    "Persistence": {
                        horizon: {
                            "prediction": value,
                            "radius": radius,
                            "weekly_fallback": False,
                        }
                        for horizon in HORIZONS
                    }
                },
            }

        self.assertEqual(decide(sample(80.0), "Persistence")["status"], "NORMAL")
        self.assertEqual(decide(sample(102.0), "Persistence")["status"], "PEAK_WATCH")
        self.assertEqual(decide(sample(110.0), "Persistence")["status"], "LOAD_SHIFT_REVIEW")
        self.assertEqual(decide(sample(110.0, quality=["결측"]), "Persistence")["status"], "REVIEW_REQUIRED")
        self.assertEqual(decide(sample(80.0, radius=16.0), "Persistence")["status"], "REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
