"""Synthetic, bounded checks for the September 24 cold-reproduction harness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.verify_session_0924 import (
    OPTIONAL_TASK_CSVS, REQUIRED_CSVS, TASK1_PRODUCTS,
    _protected_hashes, _session_source_hashes,
    _sha256, _verify_task1_products,
    check_holdout_absent, compare_csv,
    compare_original_cv, execute_tasks, restricted_output, verify_protected,
)
from src.session_data import SEALED_BOUNDARY


class SessionReproTests(unittest.TestCase):
    def test_original_task1_product_hashes_match_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in TASK1_PRODUCTS.values():
                file = root / "outputs" / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(relative, encoding="utf-8")
            manifest = {"products_sha256": {name: _sha256(root/"outputs"/relative)
                                            for name, relative in TASK1_PRODUCTS.items()}}
            _verify_task1_products(root, manifest)
            (root/"outputs"/next(iter(TASK1_PRODUCTS.values()))).write_text("modified", encoding="utf-8")
            with self.assertRaises(AssertionError):
                _verify_task1_products(root, manifest)

    def test_mutable_runner_inputs_are_hashed_and_changes_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {"tariff": {"old": "configs/old.yaml", "current": "configs/new.yaml"}}
            paths = ["configs/default.yaml", "configs/old.yaml", "configs/new.yaml",
                     "outputs/tables/T2-1_fva.csv"]
            paths += [f"outputs/models/development_h96_fold{fold}_{name}.joblib"
                      for fold in range(3) for name in ("lgbm", "lgbm_no_holiday")]
            paths += ["src/example.py", "src/models/point.py", "scripts/example.py"]
            for relative in paths:
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(relative, encoding="utf-8")
            original = _session_source_hashes(root, cfg)
            self.assertEqual(set(original), set(paths))
            (root / "configs/new.yaml").write_text("changed", encoding="utf-8")
            self.assertNotEqual(original, _session_source_hashes(root, cfg))
            a_log = root / "outputs/logs/peak_sensitivity_0924.json"
            a_log.parent.mkdir(parents=True, exist_ok=True)
            a_log.write_text("A complete", encoding="utf-8")
            self.assertIn("outputs/logs/peak_sensitivity_0924.json",
                          _session_source_hashes(root, cfg, extra_tasks=("C",)))
            self.assertNotIn("outputs/logs/peak_sensitivity_0924.json",
                             _session_source_hashes(root, cfg))

    def test_only_empty_validation_subdirectory_can_receive_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = root / "_validation" / "session_0924_repro"
            self.assertEqual(restricted_output(root, allowed), allowed)
            with self.assertRaises(ValueError):
                restricted_output(root, root / "outputs")
            with self.assertRaises(ValueError):
                restricted_output(root, root / "_validation")
            allowed.mkdir(parents=True)
            (allowed / "old.csv").write_text("cached", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                restricted_output(root, allowed)

    def test_protected_preflight_detects_change_and_unexpected_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protected = root / "report" / "original.md"
            protected.parent.mkdir()
            protected.write_text("original", encoding="utf-8")
            hashes = _protected_hashes(root)
            pinned = {"boundary": str(SEALED_BOUNDARY), "protected_file_count": 1,
                      "protected_sha256": hashes}
            self.assertTrue(verify_protected(root, pinned, expected_count=1)["unchanged"])
            protected.write_text("changed", encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_protected(root, pinned, expected_count=1)
            protected.write_text("original", encoding="utf-8")
            (protected.parent / "unexpected.md").write_text("added", encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_protected(root, pinned, expected_count=1)

    def test_comparison_tolerates_only_tiny_numeric_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, repeated = root / "original.csv", root / "repeated.csv"
            pd.DataFrame({"group": ["A", "B"], "metric": [1., 2.]}).to_csv(original, index=False)
            pd.DataFrame({"group": ["A", "B"], "metric": [1.+5e-11, 2.]}).to_csv(repeated, index=False)
            self.assertTrue(compare_csv(original, repeated)["passed"])
            pd.DataFrame({"group": ["A", "C"], "metric": [1., 2.]}).to_csv(repeated, index=False)
            with self.assertRaises(AssertionError):
                compare_csv(original, repeated)
            pd.DataFrame({"group": ["A"], "metric": [1.]}).to_csv(repeated, index=False)
            with self.assertRaises(AssertionError):
                compare_csv(original, repeated)

    def test_legacy_cv_ignores_only_three_new_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "outputs" / "tables" / "development_cv.csv"
            archive.parent.mkdir(parents=True)
            pd.DataFrame({"horizon": [4, 16], "mae": [10., 12.],
                          "peak_mae_ci95": ["[8.0, 12.0]", "[nan, nan]"]}).to_csv(archive, index=False)
            generated = pd.DataFrame({"horizon": [4, 16], "mae": [10., 12.],
                                      "peak_mae_ci95": [[np.float64(8.0+5e-12), np.float64(12.0-5e-12)],
                                                        [np.float64(np.nan), np.float64(np.nan)]],
                                      "peak_mae_union": [8., 9.], "peak_bias": [-1., -2.],
                                      "overpredict_rate": [.2, .3]})
            with patch("src.evaluate.evaluate_all", return_value=generated):
                self.assertTrue(compare_original_cv(root, pd.DataFrame(), {})["passed"])
            with patch("src.evaluate.evaluate_all", return_value=generated.assign(mae=11.)):
                with self.assertRaises(AssertionError):
                    compare_original_cv(root, pd.DataFrame(), {})
            broken_ci = generated.copy()
            broken_ci.at[0, "peak_mae_ci95"] = [8.0, 12.01]
            with patch("src.evaluate.evaluate_all", return_value=broken_ci):
                with self.assertRaises(AssertionError):
                    compare_original_cv(root, pd.DataFrame(), {})
            with patch("src.evaluate.evaluate_all", return_value=generated.drop(columns="peak_bias")):
                with self.assertRaises(AssertionError):
                    compare_original_cv(root, pd.DataFrame(), {})

    def test_fresh_diagnostic_then_two_before_three_and_optional_last(self):
        calls = []
        def diagnostic(root, *, output):
            calls.append("1")
            return {"fresh_refit_models": 6}
        def runner(name):
            def invoke(root, output):
                calls.append(name)
                return {"task": name}
            return invoke
        runners = {name: runner(name) for name in ("2", "3", "4", "5", "B")}
        timings = {}
        execute_tasks(Path("."), Path("_validation/synthetic"), extra_tasks=("B",),
                      diagnostic_runner=diagnostic, task_runners=runners,
                      stage_metrics=timings)
        self.assertEqual(calls, ["1", "2", "3", "4", "5", "B"])
        self.assertEqual(set(timings), set(calls))
        self.assertTrue(all(row["status"] == "completed" and row["elapsed_seconds"] >= 0
                            for row in timings.values()))
        before = calls.copy()
        with self.assertRaises(ValueError):
            execute_tasks(Path("."), Path("_validation/synthetic"), extra_tasks=("A",),
                          diagnostic_runner=diagnostic, task_runners=runners)
        self.assertEqual(calls, before, "Missing optional runner must fail before any refit")
        with self.assertRaises(AssertionError):
            execute_tasks(Path("."), Path("_validation/synthetic"),
                          diagnostic_runner=lambda root, output: {"fresh_refit_models": 5},
                          task_runners=runners)

    def test_final_test_csv_must_be_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            check_holdout_absent(root)
            target = root / "outputs" / "tables" / "final_test.csv"
            target.parent.mkdir(parents=True)
            target.write_text("sealed", encoding="utf-8")
            with self.assertRaises(AssertionError):
                check_holdout_absent(root)

    def test_early_stopping_monitor_table_is_mandatory(self):
        self.assertIn("tables/h96_diagnostic_early_stopping_metrics.csv", REQUIRED_CSVS)

    def test_optional_a_registered_files_and_auto_runner_name(self):
        from scripts import run_session_0924 as session

        self.assertEqual(OPTIONAL_TASK_CSVS["A"], {
            "tables/peak_sensitivity_0924_fold_grid.csv",
            "tables/peak_sensitivity_0924_metrics.csv",
            "tables/peak_sensitivity_0924_ranking.csv",
        })
        called = []
        def diagnostic(root, *, output):
            called.append("1")
            return {"fresh_refit_models": 6}
        def stub(name):
            def invoke(root, output):
                called.append(name)
                return {"task": name}
            return invoke
        with (patch.object(session, "task2", stub("2")),
              patch.object(session, "task3", stub("3")),
              patch.object(session, "task4", stub("4")),
              patch.object(session, "task5", stub("5")),
              patch.object(session, "task_a", stub("A"))):
            execute_tasks(Path("."), Path("_validation/synthetic"),
                          extra_tasks=("A",), diagnostic_runner=diagnostic)
        self.assertEqual(called, ["1", "2", "3", "4", "5", "A"])

    def test_optional_c_registered_files_and_auto_runner_name(self):
        from scripts import run_session_0924 as session

        self.assertEqual(OPTIONAL_TASK_CSVS["C"], {
            "tables/evening_oracle_0924_cases.csv",
            "tables/evening_oracle_0924_conditions.csv",
            "tables/evening_oracle_0924_weekdays.csv",
            "tables/evening_oracle_0924_cohort.csv",
            "tables/evening_oracle_0924_metrics.csv",
            "tables/evening_oracle_0924_paired_gain.csv",
            "predictions/evening_oracle_0924_point_oof.csv",
        })
        called = []
        def diagnostic(root, *, output):
            called.append("1")
            return {"fresh_refit_models": 6}
        def stub(name):
            def invoke(root, output):
                called.append(name)
                return {"task": name}
            return invoke
        with (patch.object(session, "task2", stub("2")),
              patch.object(session, "task3", stub("3")),
              patch.object(session, "task4", stub("4")),
              patch.object(session, "task5", stub("5")),
              patch.object(session, "task_c", stub("C"))):
            execute_tasks(Path("."), Path("_validation/synthetic"),
                          extra_tasks=("C",), diagnostic_runner=diagnostic)
        self.assertEqual(called, ["1", "2", "3", "4", "5", "C"])


if __name__ == "__main__":
    unittest.main()
