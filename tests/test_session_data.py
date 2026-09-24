"""Synthetic sealed-prefix tests; no real holdout values are loaded."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.session_data import load_development_history, load_development_oof, write_preflight_manifest


HEADER = ["날짜", "시간", "15분", "30분", "45분", "60분", "평균", "생산량",
          "기온", "풍속", "습도", "강수량", "전기요금(계절)", "day", "d", "m",
          "공장인원", "인건비"]


def fixture(root: Path, *, repaired: bool = False) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "configs/default.yaml").write_text(
        "source: data/raw/task05_power/source.csv\n"
        "split: {test_start_origin: '2021-08-09 09:45:00'}\n", encoding="utf-8")
    path = root / "data/raw/task05_power/source.csv"
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        stream.write((",".join(HEADER) + "\n").encode("utf-8"))
        for date, hours in (("20210808", range(24)), ("20210809", range(9))):
            for hour in hours:
                source_hour = hour + 24 if repaired and date == "20210808" else hour
                values = [date, source_hour, 100 + hour, 110 + hour, 120 + hour, 130 + hour,
                          115 + hour, 1000 + hour, 22, 2, 65, 0, 2, 0, 0, 0, 5, 10]
                stream.write((",".join(map(str, values)) + "\n").encode("utf-8"))
        # The byte sequence after the 30-minute field is deliberately invalid
        # UTF-8: decoding either holdout field would fail this test.
        stream.write(b"20210809,9,209,219,\xff\xfe,\xff\xfe,\xff\xfe\n")
        stream.write(b"20210809,10,\xff\xfe,\xff\xfe,\xff\xfe,\xff\xfe\n")
    return path


class DevelopmentPrefixTests(unittest.TestCase):
    def test_boundary_hour_reads_only_two_safe_power_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            history = load_development_history(root)
            self.assertIsInstance(history.index, pd.DatetimeIndex)
            self.assertEqual(history.index.name, "ts_end")
            self.assertEqual(history.index.max(), pd.Timestamp("2021-08-09 09:30"))
            self.assertEqual(len(history), 134)
            self.assertEqual(history.loc["2021-08-09 09:15", "power"], 209)
            self.assertEqual(history.loc["2021-08-09 09:30", "power"], 219)
            self.assertTrue(history.loc["2021-08-09 09:30", "missing_production_target"])
            self.assertEqual(history.loc["2021-08-09 09:30", "production_known"], 1008)
            self.assertTrue((history.index < pd.Timestamp("2021-08-09 09:45")).all())

    def test_complete_day_repair_matches_positional_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, repaired=True)
            history = load_development_history(root)
            self.assertTrue(history.loc["2021-08-08 00:15":"2021-08-09 00:00", "time_repaired"].all())
            self.assertFalse(history.loc["2021-08-09 00:15", "time_repaired"])
            self.assertTrue(history.loc["2021-08-09 00:00", "production_completed_bad"])

    def test_later_cutoff_is_rejected_before_source_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "configs/default.yaml").write_text(
                "source: data/raw/task05_power/source.csv\n"
                "split: {test_start_origin: '2021-08-09 10:45:00'}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed"):
                load_development_history(root)

    def test_preflight_hashes_protected_bytes_without_interpreting_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            for name in ("docs/roadmap.html", "eval_protocol.md",
                         "outputs/logs/adoption_criteria.md",
                         "outputs/logs/adoption_criteria_addendum_0924.md"):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\xff\xfe protected bytes")
            path = write_preflight_manifest(root)
            import hashlib
            import json
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["boundary"], "2021-08-09 09:45:00")
            self.assertEqual(result["protected_sha256"]["docs/roadmap.html"],
                             hashlib.sha256(b"\xff\xfe protected bytes").hexdigest())
            self.assertIn("data/raw/task05_power/source.csv", result["protected_sha256"])
            self.assertEqual(write_preflight_manifest(root), path)
            (root / "docs/roadmap.html").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "refusing to replace"):
                write_preflight_manifest(root)

    def test_oof_metadata_is_checked_before_holdout_body_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "outputs/predictions/development_oof.csv"
            path.parent.mkdir(parents=True)
            path.write_bytes(
                b"origin,target_time,y,pred\n"
                b"2021-08-09 08:30:00,2021-08-09 09:30:00,15,14\n")
            safe = load_development_oof(root)
            self.assertEqual(len(safe), 1)
            with path.open("ab") as stream:
                stream.write(b"2021-08-09 09:45:00,2021-08-09 10:45:00,\xff\xfe,\xff\xfe\n")
            with self.assertRaisesRegex(ValueError, "sealed development boundary"):
                load_development_oof(root)


if __name__ == "__main__":
    unittest.main()
