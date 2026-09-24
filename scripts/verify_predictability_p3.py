"""Recompute the fixed P1 A1/A2/A4 checks from sealed, preserved inputs.

This verification never reads the mutable P3 development OOF or fits a model.
All intermediate tables are written only under the isolated validation folder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.predictability import (  # noqa: E402
    analyze_a1,
    analyze_a2,
    detect_restart_events,
    fit_restart_thresholds,
)
from src.analysis.predictability_operations import run_a4  # noqa: E402
from src.session_data import (  # noqa: E402
    SEALED_BOUNDARY,
    load_development_history,
    read_development_oof,
)


PRESERVED = ROOT / "_validation/session_0925_original"
ISOLATED = ROOT / "_validation/session_0925_p1_repro"
REFERENCE = ROOT / "outputs/analysis_p1"
RESULT = ROOT / "outputs/logs/P3_predictability_repro.json"
ATOL = 1e-10


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_table(name: str, calculated: pd.DataFrame) -> dict:
    """Compare a deterministic P1 table, normalizing both through CSV."""
    ISOLATED.mkdir(parents=True, exist_ok=True)
    new_path = ISOLATED / f"{name}.csv"
    old_path = REFERENCE / f"{name}.csv"
    calculated.to_csv(new_path, index=False, encoding="utf-8-sig")
    old = pd.read_csv(old_path, low_memory=False)
    new = pd.read_csv(new_path, low_memory=False)
    pd.testing.assert_frame_equal(old, new, check_dtype=False,
                                  check_exact=False, atol=ATOL, rtol=0)
    columns = list(set(old.select_dtypes(include=np.number)) &
                   set(new.select_dtypes(include=np.number)))
    error = 0.0
    for column in columns:
        delta = (old[column].to_numpy(dtype=float) -
                 new[column].to_numpy(dtype=float))
        finite = np.isfinite(delta)
        if finite.any():
            error = max(error, float(np.max(np.abs(delta[finite]))))
    return {"rows": len(new), "columns": len(new.columns),
            "max_abs_numeric_difference": error,
            "source": old_path.relative_to(ROOT).as_posix()}


def _same_scalar(name: str, actual, expected) -> float:
    if isinstance(actual, (int, float, np.number)) and not isinstance(actual, bool):
        a, b = float(actual), float(expected)
        if np.isnan(a) and np.isnan(b):
            return 0.0
        if not np.isfinite(a) or not np.isfinite(b) or abs(a-b) > ATOL:
            raise AssertionError(f"{name}: recomputed {a} differs from preserved {b}")
        return abs(a-b)
    if actual != expected:
        raise AssertionError(f"{name}: recomputed {actual!r} differs from preserved {expected!r}")
    return 0.0


def _compare_record(label: str, actual: dict, expected: dict, fields: tuple[str, ...]) -> dict:
    differences = {key: _same_scalar(f"{label}.{key}", actual[key], expected[key])
                   for key in fields}
    return {"fields": list(fields), "max_abs_numeric_difference": max(differences.values(), default=0.0)}


def _original_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[int, pd.DataFrame], dict]:
    history = load_development_history(ROOT)
    original_oof = read_development_oof(PRESERVED / "predictions/development_oof.csv")
    manifest = json.loads((PRESERVED / "logs/development_selection.json").read_text(encoding="utf-8"))
    if history.index.max() >= SEALED_BOUNDARY or original_oof.target_time.ge(SEALED_BOUNDARY).any():
        raise AssertionError("A preserved P1 input crosses the sealed boundary")
    metadata = {(int(row["horizon"]), int(row["fold"])): row for row in manifest["folds"]}
    selected = {}
    for horizon in (4, 16, 96):
        name = manifest["selection"]["by_horizon"][str(horizon)]["point_model"]
        frame = original_oof.loc[original_oof.horizon.eq(horizon) & original_oof.model.eq(name)].copy()
        if frame.empty or frame.duplicated(["fold", "target_time"]).any() or frame.fold.nunique() != 3:
            raise AssertionError(f"Original h{horizon} score rows are incomplete or duplicated")
        for fid, block in frame.groupby("fold"):
            expected = metadata[horizon, int(fid)]
            if (len(block) != expected["n_score"] or
                    pd.Timestamp(block.origin.min()) != pd.Timestamp(expected["score_start"]) or
                    pd.Timestamp(block.origin.max()) != pd.Timestamp(expected["score_end"])):
                raise AssertionError(f"Original h{horizon}/fold{fid} score grid changed")
        selected[horizon] = frame
    return history, original_oof, manifest, selected, metadata


def main() -> None:
    started = time.perf_counter()
    original_files = [PRESERVED / "predictions/development_oof.csv",
                      PRESERVED / "logs/development_selection.json"]
    original_files += sorted(path for path in REFERENCE.iterdir() if path.is_file())
    before = {path.relative_to(ROOT).as_posix(): _digest(path) for path in original_files}
    report = {"status": "running", "scope": "sealed_original_development_oof_only",
              "sealed_boundary_exclusive": str(SEALED_BOUNDARY),
              "absolute_tolerance": ATOL, "original_input_sha256": before,
              "isolated_output_dir": ISOLATED.relative_to(ROOT).as_posix()}
    try:
        history, original_oof, manifest, selected, metadata = _original_inputs()
        report["safe_prefix"] = {"observations": len(history),
                                 "last_ts_end": str(history.index.max()),
                                 "original_oof_rows": len(original_oof)}

        a1_tables, a1 = analyze_a1(history, selected[4],
                                    {fid: metadata[4, fid] for fid in range(3)})
        preserved_a1 = json.loads((REFERENCE / "A1_A3_summary.json").read_text(encoding="utf-8"))["A1"]
        report["A1"] = {
            "summary": _compare_record("A1", a1, preserved_a1,
                                       ("n_peak_starts", "n_selected_slot_starts",
                                        "observed_proximity", "null_mean_proximity")),
            "hypothesis": _compare_record("A1.hypothesis", a1["hypothesis"],
                                          preserved_a1["hypothesis"],
                                          ("estimate", "ci_low", "ci_high", "p_raw",
                                           "n_positive_folds", "eligible")),
            "restart_slots": _compare_table("A1_restart_slots", a1_tables["A1_restart_slots"]),
            "peak_starts": _compare_table("A1_peak_starts", a1_tables["A1_peak_starts"]),
            "proximity": _compare_table("A1_proximity", a1_tables["A1_proximity"]),
        }

        a2_tables, a2, _ = analyze_a2(history, {16: selected[16], 96: selected[96]}, metadata)
        preserved_a2 = json.loads((REFERENCE / "A1_A3_summary.json").read_text(encoding="utf-8"))["A2"]
        current_cohort = a2["cohort_fold_counts"]
        prior_cohort = preserved_a2["cohort_fold_counts"]
        if len(current_cohort) != 6 or len(prior_cohort) != 6:
            raise AssertionError("A2 must retain exactly six preregistered fold cohorts")
        current_by_key = {(int(row["horizon"]), int(row["fold"])): row for row in current_cohort}
        prior_by_key = {(int(row["horizon"]), int(row["fold"])): row for row in prior_cohort}
        if current_by_key.keys() != prior_by_key.keys():
            raise AssertionError("A2 cohort horizon/fold identities changed")
        for key in current_by_key:
            _compare_record(f"A2.cohort.{key}", current_by_key[key], prior_by_key[key],
                            ("selected_slots", "train_n", "score_n", "all_score_peaks",
                             "selected_score_peaks"))
        if any(int(row["selected_score_peaks"]) != 0 for row in current_cohort):
            raise AssertionError("A2 selected score cohort unexpectedly contains a peak")
        hypotheses = {row["hypothesis_id"]: row for row in a2["hypotheses"]}
        previous_hypotheses = {row["hypothesis_id"]: row for row in preserved_a2["hypotheses"]}
        if hypotheses.keys() != previous_hypotheses.keys() or len(hypotheses) != 16:
            raise AssertionError("A2 hypothesis family changed")
        for key in hypotheses:
            _compare_record(f"A2.hypothesis.{key}", hypotheses[key], previous_hypotheses[key],
                            ("estimate", "ci_low", "ci_high", "p_raw", "eligible", "n_boot_valid"))
            if hypotheses[key]["eligible"] or np.isfinite(hypotheses[key]["estimate"]):
                raise AssertionError("A2 AUC must remain undefined for the fixed empty-positive cohort")
        report["A2"] = {
            "cohorts": _compare_table("A2_fold_cohorts", a2_tables["A2_fold_cohorts"]),
            "restart_slots": _compare_table("A2_restart_slots", a2_tables["A2_restart_slots"]),
            "score_cohort": _compare_table("A2_score_cohort", a2_tables["A2_score_cohort"]),
            "fixed_hypotheses": len(hypotheses), "eligible_hypotheses": 0,
            "selected_score_peaks": 0,
            "h16_score_cohort": sum(row["score_n"] for row in current_cohort if row["horizon"] == 16),
            "h96_score_cohort": sum(row["score_n"] for row in current_cohort if row["horizon"] == 96),
        }

        a4 = run_a4(history, original_oof, manifest,
                    (fit_restart_thresholds, detect_restart_events), ISOLATED)
        previous_a4 = pd.read_csv(REFERENCE / "A4_hypotheses.csv")
        report["A4"] = {
            "hypotheses": _compare_table("A4_hypotheses", a4["hypotheses"]),
            "folds": _compare_table("A4_folds", a4["folds"]),
            "breakdown": _compare_table("A4_breakdown", a4["breakdown"]),
            "paired_rows": _compare_table("A4_paired", a4["paired"]),
            "effects": a4["hypotheses"][["horizon", "estimate", "ci_low", "ci_high",
                                           "restart_rows", "nonrestart_rows"]].to_dict("records"),
            "preserved_effects": previous_a4[["horizon", "estimate"]].to_dict("records"),
        }
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after = {path.relative_to(ROOT).as_posix(): _digest(path) for path in original_files}
        report["original_inputs_unchanged"] = before == after
        if before != after:
            report["status"] = "fail"
            report["error"] = "Preserved original OOF or selection manifest changed during verification"
        report["elapsed_seconds"] = round(time.perf_counter()-started, 3)
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     default=str, allow_nan=True), encoding="utf-8")
    if report["status"] != "pass":
        raise RuntimeError(report["error"])
    print(json.dumps({"status": report["status"], "A1_peak_starts": a1["n_peak_starts"],
                      "A2_h16_score_cohort": report["A2"]["h16_score_cohort"],
                      "A2_h96_score_cohort": report["A2"]["h96_score_cohort"],
                      "A4_effects": report["A4"]["effects"],
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
