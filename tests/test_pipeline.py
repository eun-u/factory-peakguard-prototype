"""누수 방지·분할·사건 매칭 단위 점검과 합성 자료 전체 실행 점검.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.analysis.alerting import k_of_n  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import load_series  # noqa: E402
from src.evaluate import overlap_match_count, runs  # noqa: E402
from src.features import feature_frame  # noqa: E402
from src.models.lgbm_quantile import conformal_offset  # noqa: E402
from src.split import split_time  # noqa: E402
from synthetic import make_synthetic  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        csv = make_synthetic(base / "data" / "okm_augumented_2021.csv")
        cfg = load_config()
        cfg["data"]["path"] = str(csv)
        cfg["output_dir"] = str(base / "outputs")
        cls.cfg = cfg
        cls.series, cls.raw, cls.info = load_series(cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_expansion_and_hour_recovery(self):
        self.assertEqual(len(self.series), 257 * 96)
        self.assertEqual(self.info["invalid_hours"], 48)
        self.assertEqual(int(self.series["recovered"].sum()), 192)
        self.assertTrue((self.series.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all())

    def test_features_do_not_see_future(self):
        x, _, _ = feature_frame(self.series)
        origin = x.index[len(x) // 2]
        changed = self.series.copy()
        later = changed.index > origin
        changed.loc[later, "power"] = changed.loc[later, "power"] + 1000
        changed.loc[later, "production_known"] = -1
        x2, _, _ = feature_frame(changed)
        pd.testing.assert_series_equal(x.loc[origin], x2.loc[origin])

    def test_production_known_only_after_hour_ends(self):
        s = self.series
        t = s.index[(s.index.minute == 30) & ~s["recovered"]][100]
        hour_start = t.floor("h")
        raw = self.raw[(self.raw["date"] == hour_start.normalize()) & (self.raw["hour"] == hour_start.hour - 1)]
        self.assertEqual(s.loc[t, "production_known"], float(raw["생산량"].iloc[0]))

    def test_split_has_gaps_and_order(self):
        x, y, meta = feature_frame(self.series)
        parts = split_time(x, y, meta)
        self.assertLess(parts["train"][2]["target_time"].max(), parts["validation"][0].index.min())
        self.assertLess(parts["validation"][2]["target_time"].max(), parts["test"][0].index.min())

    def test_episode_matching_is_one_to_one(self):
        times = pd.date_range("2021-01-01", periods=8, freq="15min").to_numpy()
        truth = runs(np.array([1, 1, 0, 1, 1, 0, 0, 0], bool), times)
        alarm = runs(np.array([1, 1, 1, 1, 1, 0, 0, 0], bool), times)
        self.assertEqual(len(truth), 2)
        self.assertEqual(overlap_match_count(truth, alarm), 1)

    def test_k_of_n_rule(self):
        origins = pd.date_range("2021-01-01", periods=6, freq="15min")
        raw = np.array([1, 0, 1, 1, 0, 0], bool)
        self.assertEqual(k_of_n(raw, origins, 2, 3).tolist(), [False, False, True, True, True, False])
        self.assertEqual(k_of_n(raw, origins, 1, 1).tolist(), raw.tolist())

    def test_conformal_offset_reaches_target_on_calibration(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=500)
        q = np.zeros(500)
        delta = conformal_offset(y, q, 0.9)
        self.assertGreaterEqual(np.mean(y <= q + delta), 0.9)

    def test_full_run_on_synthetic(self):
        import run_all
        cfg_path = Path(self.tmp.name) / "config.yaml"
        cfg = {k: v for k, v in self.cfg.items() if not k.startswith("_")}
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        summary = run_all.main(["--config", str(cfg_path), "--bootstrap-draws", "50"])
        out = Path(self.cfg["output_dir"])
        for name in ("tables/final_test.csv", "predictions/test_predictions_1h.csv",
                     "predictions/test_predictions_daily_max.csv", "logs/run_summary.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(summary["one_hour"]["rows"]["test"], 3510)


if __name__ == "__main__":
    unittest.main()
