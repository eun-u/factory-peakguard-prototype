"""Cold, development-only reproduction of the 2026-09-24 analysis session.

This script calls only the bounded safe-history diagnostic and task runners.
It neither runs the forecasting pipeline nor opens the sealed holdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.session_data import PROTECTED_DIRECTORIES, PROTECTED_FILES, SEALED_BOUNDARY


REQUIRED_CSVS = {
    "tables/h96_diagnostic_fold_grid.csv",
    "tables/h96_diagnostic_feature_missing.csv",
    "tables/h96_diagnostic_audit_20.csv",
    "tables/h96_diagnostic_existing_candidates.csv",
    "tables/h96_diagnostic_model_compare.csv",
    "tables/h96_diagnostic_learning_curve.csv",
    "tables/h96_diagnostic_early_stopping_metrics.csv",
    "tables/T2-3_symmetric_peak_metrics.csv",
    "tables/T2-1a_fva_point.csv",
    "tables/T2-1b_fva_uncertainty.csv",
    "tables/cbl_alias_audit_0924.csv",
    "tables/shift_alert_audit_v2.csv",
    "tables/shift_actions_v2.csv",
    "tables/shift_monthly_v2.csv",
    "tables/shift_summary_v2.csv",
    "predictions/energy_baseline_oof_predictions_v2.csv",
    "tables/energy_baseline_fold_metrics_v2.csv",
    "tables/energy_baseline_coefficient_ci_v2.csv",
    "tables/peak_types_v2.csv",
    "tables/peak_type_summary_v2.csv",
    "tables/peak_type_representatives_v2.csv",
}
OPTIONAL_TASK_CSVS = {
    "A": {"tables/peak_sensitivity_0924_fold_grid.csv",
          "tables/peak_sensitivity_0924_metrics.csv",
          "tables/peak_sensitivity_0924_ranking.csv"},
    "B": {"tables/operational_target_sensitivity_0924.csv"},
    "C": {"tables/evening_oracle_0924_cases.csv",
          "tables/evening_oracle_0924_conditions.csv",
          "tables/evening_oracle_0924_weekdays.csv",
          "tables/evening_oracle_0924_cohort.csv",
          "tables/evening_oracle_0924_metrics.csv",
          "tables/evening_oracle_0924_paired_gain.csv",
          "predictions/evening_oracle_0924_point_oof.csv"},
}
TASK1_PRODUCTS = {name: f"tables/h96_diagnostic_{name}.csv" for name in (
    "fold_grid", "feature_missing", "audit_20", "existing_candidates",
    "model_compare", "learning_curve", "early_stopping_metrics")}
TOLERANCE = {"atol": 1e-10, "rtol": 1e-10}
ADDED_METRICS = {"peak_mae_union", "peak_bias", "overpredict_rate"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_source_hashes(root: Path, cfg: dict, *, extra_tasks: tuple[str, ...] = ()) -> dict[str, str]:
    """Pin mutable safe-task inputs beyond the protected raw data and OOF."""
    relative_paths = ["configs/default.yaml", cfg["tariff"]["old"],
                      cfg["tariff"]["current"],
                      "outputs/tables/T2-1_fva.csv"]
    relative_paths.extend(f"outputs/models/development_h96_fold{fold}_{model}.joblib"
                          for fold in range(3) for model in ("lgbm", "lgbm_no_holiday"))
    if "C" in extra_tasks:
        relative_paths.append("outputs/logs/peak_sensitivity_0924.json")
    for package in ("src", "scripts"):
        relative_paths.extend(path.relative_to(root).as_posix()
                              for path in sorted((root/package).rglob("*.py")))
    hashes = {}
    for relative in relative_paths:
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Task source escapes workspace: {relative}")
        hashes[path.relative_to(root).as_posix()] = _sha256(path)
    return hashes


def _verify_task1_products(root: Path, manifest: dict) -> None:
    """Authenticate the comparison tables against the original task-1 log."""
    expected = manifest.get("products_sha256")
    if not isinstance(expected, dict) or set(expected) != set(TASK1_PRODUCTS):
        raise AssertionError("Original task-1 product manifest is incomplete")
    actual = {name: _sha256(root/"outputs"/relative)
              for name, relative in TASK1_PRODUCTS.items()}
    if actual != expected:
        raise AssertionError("Original task-1 CSV differs from its recorded SHA-256")


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def restricted_output(root: Path, output: Path) -> Path:
    """Only a new isolated _validation child may receive reproduced files."""
    root = Path(root).resolve()
    output = Path(output)
    output = (root/output).resolve() if not output.is_absolute() else output.resolve()
    if output == (root/"_validation").resolve() or not output.is_relative_to((root/"_validation").resolve()):
        raise ValueError("Reproduction output must be below workspace/_validation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Cold reproduction output must be new or empty; no cache reuse")
    return output


def _protected_hashes(root: Path) -> dict[str, str]:
    files: set[Path] = set()
    for directory in PROTECTED_DIRECTORIES:
        files.update(path for path in (root/directory).rglob("*") if path.is_file())
    files.update(root/name for name in PROTECTED_FILES if (root/name).is_file())
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("Protected path escapes workspace")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(files)}


def verify_protected(root: Path, pinned: dict, *, expected_count: int = 75) -> dict:
    """Check the immutable preflight snapshot without rewriting that snapshot."""
    if pinned.get("boundary") != str(SEALED_BOUNDARY):
        raise AssertionError("Preflight boundary differs from the sealed development cutoff")
    expected = pinned.get("protected_sha256", {})
    if pinned.get("protected_file_count") != expected_count or len(expected) != expected_count:
        raise AssertionError("Pinned protected-file count differs from preregistration")
    current = _protected_hashes(Path(root).resolve())
    if current != expected:
        changed = sorted(set(current) ^ set(expected) |
                         {name for name in set(current) & set(expected) if current[name] != expected[name]})
        raise AssertionError(f"Protected file set or SHA-256 changed: {changed[:8]}")
    return {"count": len(current), "unchanged": True}


def check_holdout_absent(root: Path, output: Path | None = None) -> None:
    candidates = [root/"outputs/tables/final_test.csv"]
    if output is not None:
        candidates.append(output/"tables/final_test.csv")
    present = [str(path) for path in candidates if path.exists()]
    if present:
        raise AssertionError(f"Sealed final-test result must not exist during session reproduction: {present}")


def _numeric_max_abs_difference(left: pd.DataFrame, right: pd.DataFrame) -> float:
    if len(left) != len(right):
        return float("nan")
    numeric = [col for col in left if col in right and pd.api.types.is_numeric_dtype(left[col])
               and pd.api.types.is_numeric_dtype(right[col])]
    if not numeric:
        return 0.0
    a = left[numeric].to_numpy(float)
    b = right[numeric].to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0


def compare_csv(left_path: Path, right_path: Path) -> dict:
    """Exact categories/rows/columns; numeric values within preregistered tol."""
    left = pd.read_csv(left_path, low_memory=False)
    right = pd.read_csv(right_path, low_memory=False)
    result = {"file": right_path.name, "rows": len(right),
              "columns": len(right.columns), "max_abs_numeric_difference": _numeric_max_abs_difference(left, right),
              "atol": TOLERANCE["atol"], "rtol": TOLERANCE["rtol"], "passed": False}
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False,
                                  atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"])
    result["passed"] = True
    return result


def _ci_bounds(value) -> np.ndarray:
    """Parse a stored two-sided CI, accepting the historical Python list form."""
    if pd.isna(value):
        return np.array([np.nan, np.nan], dtype=float)
    def parse(node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return [parse(part) for part in node.elts]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id.lower() == "nan":
            return float("nan")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            scalar = parse(node.operand)
            return -scalar if isinstance(node.op, ast.USub) else scalar
        # NumPy 2 prints scalar entries as np.float64(1.0) in list repr.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
                and node.func.attr in {"float32", "float64"}
                and len(node.args) == 1 and not node.keywords):
            return parse(node.args[0])
        raise ValueError(f"Unsupported CI literal: {value!r}")

    parsed = parse(ast.parse(str(value), mode="eval").body)
    if isinstance(parsed, float) and np.isnan(parsed):
        return np.array([np.nan, np.nan], dtype=float)
    result = np.asarray(parsed, dtype=float)
    if result.shape != (2,):
        raise AssertionError(f"Peak MAE CI must contain exactly two bounds: {value!r}")
    return result


def compare_original_cv(root: Path, predictions: pd.DataFrame, cfg: dict) -> dict:
    """Re-score archived OOF; new symmetric metrics never revise old columns."""
    from src.evaluate import evaluate_all

    legacy = pd.read_csv(root/"outputs/tables/development_cv.csv", low_memory=False)
    recalculated = evaluate_all(predictions, cfg=cfg)
    extra = set(recalculated.columns)-set(legacy.columns)
    if extra != ADDED_METRICS:
        raise AssertionError(f"Unexpected changed evaluation schema: {sorted(extra)}")
    missing = set(legacy.columns)-set(recalculated.columns)
    if missing:
        raise AssertionError(f"Existing evaluation columns disappeared: {sorted(missing)}")
    # Round-trip through the same CSV parser so list-valued legacy CI cells
    # have the same representation, without writing a production artifact.
    equivalent = pd.read_csv(io.StringIO(recalculated.to_csv(index=False)), low_memory=False)
    equivalent = equivalent.loc[:, legacy.columns]
    if "peak_mae_ci95" not in legacy:
        raise AssertionError("Archived peak MAE CI column is missing")
    legacy_ci = np.vstack([_ci_bounds(value) for value in legacy.peak_mae_ci95])
    equivalent_ci = np.vstack([_ci_bounds(value) for value in equivalent.peak_mae_ci95])
    ci_finite = np.isfinite(legacy_ci) & np.isfinite(equivalent_ci)
    ci_max = float(np.max(np.abs(legacy_ci[ci_finite] - equivalent_ci[ci_finite]))) if ci_finite.any() else 0.0
    result = {"file": "tables/development_cv.csv", "rows": len(legacy),
              "excluded_new_columns": sorted(ADDED_METRICS),
              "max_abs_numeric_difference": max(_numeric_max_abs_difference(legacy, equivalent), ci_max),
              "peak_mae_ci95_max_abs_difference": ci_max, "passed": False}
    pd.testing.assert_frame_equal(legacy.drop(columns="peak_mae_ci95"),
                                  equivalent.drop(columns="peak_mae_ci95"),
                                  check_dtype=False, check_exact=False,
                                  atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"])
    np.testing.assert_allclose(legacy_ci, equivalent_ci, equal_nan=True,
                               atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"])
    result["passed"] = True
    return result


def execute_tasks(root: Path, output: Path, *, extra_tasks: tuple[str, ...] = (),
                  diagnostic_runner: Callable | None = None,
                  task_runners: Mapping[str, Callable] | None = None,
                  stage_metrics: dict | None = None) -> dict:
    """Task 1 fresh refit, then 2→3 alias merge, then 4→5; optional A/B/C hook."""
    if diagnostic_runner is None:
        from scripts.diagnose_h96_0924 import run as diagnostic_runner
    if task_runners is None:
        from scripts import run_session_0924 as session
        task_runners = {name: getattr(session, f"task{name}" if name.isdigit()
                                      else f"task_{name.lower()}", None)
                        for name in ("2", "3", "4", "5", "A", "B", "C")}
    if any(name not in ("A", "B", "C") for name in extra_tasks):
        raise ValueError("Only preregistered optional A/B/C tasks may be appended")
    if len(set(extra_tasks)) != len(extra_tasks):
        raise ValueError("Optional task names must be unique")
    unavailable = [name for name in ("2", "3", "4", "5", *extra_tasks)
                   if task_runners.get(name) is None]
    if unavailable:
        raise ValueError(f"Safe task runners not registered: {unavailable}")
    metrics = stage_metrics if stage_metrics is not None else {}

    def run_stage(name: str, runner: Callable) -> dict:
        started = time.perf_counter()
        finished = threading.Event()
        print(f"[repro] task {name} started", flush=True)

        def heartbeat() -> None:
            while not finished.wait(30):
                print(f"[repro] task {name} running {time.perf_counter()-started:.0f}s", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            result = runner()
        except Exception:
            metrics[name] = {"status": "failed", "elapsed_seconds": time.perf_counter()-started}
            print(f"[repro] task {name} failed after {metrics[name]['elapsed_seconds']:.1f}s", flush=True)
            raise
        else:
            metrics[name] = {"status": "completed", "elapsed_seconds": time.perf_counter()-started}
            print(f"[repro] task {name} completed in {metrics[name]['elapsed_seconds']:.1f}s", flush=True)
            return result
        finally:
            finished.set()
            thread.join(timeout=1)

    results = {}
    results["1"] = run_stage("1", lambda: diagnostic_runner(root, output=output))
    if results["1"].get("fresh_refit_models") != 6:
        raise AssertionError("Task 1 did not freshly refit all six preregistered h96 models")
    for name in ("2", "3", "4", "5", *extra_tasks):
        fn = task_runners.get(name)
        if fn is None:
            raise ValueError(f"No safe registered runner for task {name}")
        results[name] = run_stage(name, lambda fn=fn: fn(root, output))
    return results


def run_verification(root: Path = ROOT, output: Path | None = None, *,
                     extra_tasks: tuple[str, ...] = ()) -> dict:
    """Execute without prior cached outputs and audit every generated CSV."""
    verification_started = time.perf_counter()
    root = Path(root).resolve()
    output = restricted_output(root, output or root/"_validation/session_0924_repro")
    preflight_path = root/"outputs/logs/session_0924_preflight.json"
    pinned = json.loads(preflight_path.read_text(encoding="utf-8"))
    protected_before = verify_protected(root, pinned)
    check_holdout_absent(root)
    source_manifest = json.loads((root/"outputs/logs/h96_diagnostic_0924.json").read_text(encoding="utf-8"))
    source_manifest_path = root/"outputs/logs/h96_diagnostic_0924.json"
    source_manifest_sha = _sha256(source_manifest_path)
    _verify_task1_products(root, source_manifest)
    reference_hashes = source_manifest["sources_sha256"]
    reference_paths = {"development_oof": root/"outputs/predictions/development_oof.csv",
                       "development_selection": root/"outputs/logs/development_selection.json"}
    if {name: _sha256(path) for name, path in reference_paths.items()} != reference_hashes:
        raise AssertionError("Original OOF or selection SHA-256 differs from task 1 manifest")
    import yaml
    cfg = yaml.safe_load((root/"configs/default.yaml").read_text(encoding="utf-8"))
    source_input_hashes = _session_source_hashes(root, cfg, extra_tasks=extra_tasks)
    selection_before = _canonical_json(json.loads(
        reference_paths["development_selection"].read_text(encoding="utf-8")))
    comparison_csvs = REQUIRED_CSVS | set().union(
        *(OPTIONAL_TASK_CSVS.get(name, set()) for name in extra_tasks))
    missing_originals = sorted(relative for relative in comparison_csvs
                               if not (root/"outputs"/relative).is_file())
    if missing_originals:
        raise FileNotFoundError(f"Original session CSVs are absent: {missing_originals}")
    original_csv_hashes = {relative: _sha256(root/"outputs"/relative)
                           for relative in sorted(comparison_csvs)}
    legacy_cv_sha = _sha256(root/"outputs/tables/development_cv.csv")
    report = {"status": "running", "development_only": True, "test_opened": False,
              "output": output.relative_to(root).as_posix(), "protected_preflight": protected_before,
              "reference_sha256": reference_hashes,
              "task_input_sha256": source_input_hashes,
              "task1_manifest_sha256": source_manifest_sha,
              "numerical_tolerance": TOLERANCE, "compared_csvs": [],
              "task_results": {}}
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("tables", "logs", "predictions"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    try:
        tasks = execute_tasks(root, output, extra_tasks=extra_tasks,
                              stage_metrics=report["task_results"])
        if tasks["1"].get("sources_sha256") != reference_hashes:
            raise AssertionError("Fresh task 1 used different original OOF/selection sources")
        if not tasks["3"].get("saved_selection_reproduced") or tasks["3"].get("selection_changed"):
            raise AssertionError("Task 3 no longer reproduces the original selection")
        generated = {path.relative_to(output).as_posix() for path in output.rglob("*.csv")}
        missing = comparison_csvs-generated
        if missing:
            raise AssertionError(f"Session task outputs missing: {sorted(missing)}")
        for relative in sorted(generated):
            original = root/"outputs"/relative
            duplicate = output/relative
            if not original.is_file():
                raise FileNotFoundError(f"No original session CSV to compare: {relative}")
            comparison = compare_csv(original, duplicate)
            comparison["file"] = relative
            report["compared_csvs"].append(comparison)
        from src.session_data import load_development_oof
        report["legacy_cv"] = compare_original_cv(root, load_development_oof(root), cfg)
        if _session_source_hashes(root, cfg, extra_tasks=extra_tasks) != source_input_hashes:
            raise AssertionError("Safe task input or saved h96 model changed during reproduction")
        if _sha256(root/"outputs/tables/development_cv.csv") != legacy_cv_sha:
            raise AssertionError("Original development CV table changed during reproduction")
        if _sha256(source_manifest_path) != source_manifest_sha:
            raise AssertionError("Original task 1 manifest changed during reproduction")
        if {relative: _sha256(root/"outputs"/relative) for relative in comparison_csvs} != original_csv_hashes:
            raise AssertionError("Original session CSV changed during reproduction")
        if selection_before != \
           _canonical_json(json.loads((root/"outputs/logs/development_selection.json").read_text(encoding="utf-8"))):
            raise AssertionError("Original selection JSON changed")
        if {name: _sha256(path) for name, path in reference_paths.items()} != reference_hashes:
            raise AssertionError("Original OOF or selection changed during reproduction")
        report["protected_postflight"] = verify_protected(root, pinned)
        check_holdout_absent(root, output)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["total_elapsed_seconds"] = time.perf_counter()-verification_started
        target = root/"outputs/logs/session_0924_repro.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"_validation/session_0924_repro")
    parser.add_argument("--extra-task", action="append", choices=["A", "B", "C"], default=[],
                        help="Append an explicitly registered optional analysis after task 5")
    args = parser.parse_args()
    result = run_verification(ROOT, args.output, extra_tasks=tuple(args.extra_task))
    print(json.dumps({"status": result["status"], "compared_csvs": len(result["compared_csvs"]),
                      "output": result["output"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
