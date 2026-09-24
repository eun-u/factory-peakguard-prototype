"""Audit sealed P3 development CV and, optionally, two fixed partial refits.

This script never calls a freeze/final entrypoint or a full-data target loader.
All prediction CSVs are decoded only by the metadata-first sealed OOF reader.
The expensive ``--refit-partial`` path is reserved for after P3 preregistration
and the full development rebuild have completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adaptive_q95 import CANDIDATES, _baseline_b_quantiles, replay_adaptive_q95
from src.models.lgbm_quantile import fit_quantiles, predict_quantiles
from src.models.peak_prob import exceedance_from_quantiles
from src.models.residual import fit_residual_fold
from src.research_cv import build_contexts
from src.session_data import (PROTECTED_DIRECTORIES, PROTECTED_FILES, SEALED_BOUNDARY,
                              load_development_history, read_development_oof)
from src.training import _cutoff


TOL = 1e-10
PREREG = {"P1": "outputs/logs/preregistration_0925_P1.md",
          "P2": "outputs/logs/preregistration_0925_P2.md",
          "P3": "outputs/logs/preregistration_0925_P3.md"}
ORIGINAL = "_validation/session_0925_original/predictions/development_oof.csv"
P2_CANDIDATES = ("outputs/predictions/p2_residual_oof.csv",
                 "outputs/predictions/p2_adaptive_oof.csv")
P2_SNAPSHOT = "_validation/session_0925_P2"
KEYS = ["origin", "target_time", "horizon", "fold", "model"]


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def _protected_hashes(root: Path) -> dict[str, str]:
    files = set()
    for folder in PROTECTED_DIRECTORIES:
        files.update(path for path in (root / folder).rglob("*") if path.is_file())
    files.update(root / relative for relative in PROTECTED_FILES if (root / relative).is_file())
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise AssertionError("Protected path escapes workspace")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(files)}


def verify_protected(root: Path) -> dict:
    pinned_path = root / "outputs/logs/session_0925_preflight.json"
    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    expected = pinned.get("protected")
    if (not isinstance(expected, dict) or len(expected) != 75
            or pinned.get("boundary") != str(SEALED_BOUNDARY)
            or pinned.get("test_opened") is not False):
        raise AssertionError("P3 preflight did not pin 75 locked files and an unopened test")
    actual = _protected_hashes(root)
    if actual != expected:
        changed = sorted(expected.keys() ^ actual.keys()
                         | {name for name in expected.keys() & actual.keys()
                            if expected[name] != actual[name]})
        raise AssertionError(f"Protected preflight files changed: {changed}")
    original_artifacts = pinned.get("original_artifacts", {})
    if len(original_artifacts) != 4:
        raise AssertionError("P3 preflight did not pin four original CV artifacts")
    for relative, checksum in original_artifacts.items():
        archived = root / "_validation/session_0925_original" / relative
        if not archived.is_file() or _sha256(archived) != checksum:
            raise AssertionError(f"Original CV snapshot differs from preflight: {relative}")
    return {"count": len(actual), "preflight_sha256": _sha256(pinned_path),
            "protected_digest_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
            "original_snapshot_sha256": original_artifacts}


def verify_p2_snapshot(root: Path) -> dict[str, str]:
    folder = root / P2_SNAPSHOT
    files = {path.relative_to(folder).as_posix(): _sha256(path)
             for path in sorted(folder.rglob("*")) if path.is_file()}
    required = {"P2_candidate_selection.json", "predictions/p2_residual_oof.csv",
                "predictions/p2_adaptive_oof.csv", "analysis_p2/M1_comparisons.csv",
                "analysis_p2/M4_hypotheses.csv"}
    if not required <= set(files):
        raise AssertionError(f"P2 independent snapshot files missing: {sorted(required-set(files))}")
    return files


def compare_nested(reference, actual, label: str, path: str = "$", *, tol: float = TOL) -> float:
    """Match nested selection keys/text/bools exactly and numbers to tolerance."""
    if isinstance(reference, dict) and isinstance(actual, dict):
        if reference.keys() != actual.keys():
            raise AssertionError(f"{label} {path}: dictionary keys changed")
        return max((compare_nested(reference[key], actual[key], label, f"{path}.{key}", tol=tol)
                    for key in reference), default=0.)
    if isinstance(reference, list) and isinstance(actual, list):
        if len(reference) != len(actual):
            raise AssertionError(f"{label} {path}: list length changed")
        return max((compare_nested(a, b, label, f"{path}[{index}]", tol=tol)
                    for index, (a, b) in enumerate(zip(reference, actual))), default=0.)
    if isinstance(reference, bool) or isinstance(actual, bool):
        if type(reference) is not type(actual) or reference != actual:
            raise AssertionError(f"{label} {path}: boolean changed")
        return 0.
    if isinstance(reference, (int, float)) and isinstance(actual, (int, float)):
        a, b = float(reference), float(actual)
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                raise AssertionError(f"{label} {path}: NaN pattern changed")
            return 0.
        if math.isinf(a) or math.isinf(b):
            if a != b:
                raise AssertionError(f"{label} {path}: infinite value changed")
            return 0.
        difference = abs(a-b)
        if difference > tol:
            raise AssertionError(f"{label} {path}: {difference} exceeds {tol}")
        return difference
    if type(reference) is not type(actual) or reference != actual:
        raise AssertionError(f"{label} {path}: value or type changed")
    return 0.


def verify_preregistrations(root: Path) -> dict:
    evidence = {}
    for phase, relative in PREREG.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing committed {phase} preregistration: {relative}")
        # The first commit touching each file is its immutable preregistration.
        commits = _git(root, "rev-list", "--reverse", "HEAD", "--", relative).decode("ascii").splitlines()
        if not commits:
            raise AssertionError(f"{phase} preregistration was never committed")
        first = commits[0]
        subject = _git(root, "show", "-s", "--format=%s", first).decode("utf-8").strip()
        if f"[{phase}-REG]" not in subject:
            raise AssertionError(f"{phase} preregistration first commit has no REG marker")
        frozen = _git(root, "show", f"{first}:{relative}").replace(b"\r\n", b"\n")
        current = path.read_bytes().replace(b"\r\n", b"\n")
        if frozen != current:
            raise AssertionError(f"{phase} preregistration changed after first commit")
        evidence[phase] = {"commit": first, "sha256": hashlib.sha256(current).hexdigest(),
                           "subject": subject}
    return evidence


def _safe_oof(path: Path) -> pd.DataFrame:
    frame = read_development_oof(path)
    if frame.duplicated(KEYS).any():
        raise AssertionError(f"Duplicate sealed OOF keys in {path.name}")
    return frame


def compare_prediction_rows(reference: pd.DataFrame, actual: pd.DataFrame,
                            *, exact_reference_set: bool, label: str) -> dict:
    """Require reference schema, matching keys, and values within 1e-10."""
    if reference.empty or not set(KEYS) <= set(reference) or not set(KEYS) <= set(actual):
        raise ValueError(f"{label}: prediction keys missing")
    missing_columns = sorted(set(reference.columns) - set(actual.columns))
    if missing_columns:
        raise AssertionError(f"{label}: prediction columns missing: {missing_columns}")
    if reference.duplicated(KEYS).any() or actual.duplicated(KEYS).any():
        raise AssertionError(f"{label}: duplicate OOF row keys")
    left = reference.set_index(KEYS).sort_index()
    right = actual.set_index(KEYS).sort_index()
    missing = left.index.difference(right.index)
    unexpected = right.index.difference(left.index) if exact_reference_set else []
    if len(missing) or len(unexpected):
        raise AssertionError(f"{label}: key mismatch, missing={len(missing)}, extra={len(unexpected)}")
    right = right.loc[left.index]
    numeric = sorted(name for name in left.columns
                     if pd.api.types.is_numeric_dtype(left[name])
                     and not pd.api.types.is_bool_dtype(left[name]))
    if not numeric or "pred" not in numeric or "y" not in numeric or "tau" not in numeric:
        raise AssertionError(f"{label}: core numeric prediction columns missing")
    worst = 0.
    for name in numeric:
        if not pd.api.types.is_numeric_dtype(right[name]):
            raise AssertionError(f"{label}: {name} numeric dtype changed")
        a = left[name].to_numpy(dtype=float)
        b = right[name].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(f"{label}: {name} NaN pattern changed")
        finite = np.isfinite(a) & np.isfinite(b)
        inf_a, inf_b = np.isinf(a), np.isinf(b)
        if (not np.array_equal(inf_a, inf_b)
                or not np.array_equal(np.signbit(a[inf_a]), np.signbit(b[inf_b]))
                or not np.allclose(a[finite], b[finite], rtol=0, atol=TOL)):
            raise AssertionError(f"{label}: {name} exceeds {TOL:g}")
        if finite.any():
            worst = max(worst, float(np.max(np.abs(a[finite]-b[finite]))))
    text_columns = sorted(name for name in left.columns if name not in numeric)
    for name in text_columns:
        a = left[name].fillna("<NA>").astype(str).to_numpy()
        b = right[name].fillna("<NA>").astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise AssertionError(f"{label}: {name} provenance values changed")
    return {"rows": len(left), "numeric_columns": numeric,
            "text_columns": text_columns, "max_abs_difference": worst}


def _assert_no_final(root: Path) -> dict:
    out = root / "outputs"
    forbidden = []
    for folder in ("tables", "predictions"):
        forbidden.extend((out / folder).glob("final_test*"))
        forbidden.extend((out / folder).glob("t2_final_test*"))
    for relative in ("logs/freeze_record.json", "logs/human_freeze_approval.json"):
        if (out / relative).exists():
            forbidden.append(out / relative)
    if forbidden:
        raise AssertionError(f"A sealed final artifact or approval exists: {[str(p) for p in forbidden]}")
    return {"final_test_files": 0, "freeze_record": False, "human_approval": False,
            "test_opened": False}


def _compare_original_selection(root: Path) -> dict:
    previous = json.loads((root / "_validation/session_0925_original/logs/development_selection.json").read_text(encoding="utf-8"))
    p2_path = root / P2_SNAPSHOT / "P2_candidate_selection.json"
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    current = json.loads((root / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    original_difference = compare_nested(previous.get("selection"), current.get("original_selection"),
                                         "P3 original selection")
    p2_original_difference = compare_nested(previous.get("selection"), p2.get("original_selection"),
                                            "P2 original selection")
    p2_difference = compare_nested(p2.get("selection"), current.get("selection"),
                                   "P2/P3 selection and rejection reasons")
    unchanged = {}
    for horizon in (1, 4, 96):
        before = previous["selection"]["by_horizon"][str(horizon)]
        after = current["selection"]["by_horizon"][str(horizon)]
        for field in ("point_model", "conformal"):
            if before.get(field) != after.get(field):
                raise AssertionError(f"Unchanged h{horizon} {field} selection moved")
        unchanged[str(horizon)] = {"point_model": after.get("point_model"),
                                   "conformal": after.get("conformal")}
    return {"unchanged_horizons": unchanged, "p2_selection_matches_p3": True,
            "max_abs_original_selection": original_difference,
            "max_abs_p2_original_selection": p2_original_difference,
            "max_abs_p2_p3_selection": p2_difference,
            "p2_snapshot_sha256": _sha256(p2_path),
            "before_sha256": _sha256(root / "_validation/session_0925_original/logs/development_selection.json"),
            "after_sha256": _sha256(root / "outputs/logs/development_selection.json")}


def verify_oof_parity(root: Path) -> dict:
    original_path = root / ORIGINAL
    integrated_path = root / "outputs/predictions/development_oof.csv"
    original, integrated = _safe_oof(original_path), _safe_oof(integrated_path)
    all_original = compare_prediction_rows(original, integrated,
                                           exact_reference_set=False, label="original candidate OOF")
    selected = json.loads((root / "_validation/session_0925_original/logs/development_selection.json").read_text(encoding="utf-8"))["selection"]
    unchanged = {}
    for horizon in (1, 4, 96):
        model = selected["by_horizon"][str(horizon)]["point_model"]
        frame = original.loc[original.horizon.eq(horizon) & original.model.eq(model)]
        unchanged[str(horizon)] = compare_prediction_rows(frame, integrated,
                            exact_reference_set=False, label=f"unchanged selected h{horizon}")
        unchanged[str(horizon)]["model"] = model
    new = {}
    for relative in P2_CANDIDATES:
        path = root / P2_SNAPSHOT / "predictions" / Path(relative).name
        candidate = _safe_oof(path)
        new[path.name] = {**compare_prediction_rows(candidate, integrated,
                            exact_reference_set=False, label=path.name),
                          "sha256": _sha256(path)}
    expected_count = len(original) + sum(item["rows"] for item in new.values())
    if len(integrated) != expected_count:
        raise AssertionError("Integrated OOF contains missing or unregistered candidate rows")
    return {"original": {**all_original, "sha256": _sha256(original_path)},
            "unchanged_selected": unchanged, "new_candidates": new,
            "integrated_rows": len(integrated), "integrated_sha256": _sha256(integrated_path)}


def _partial_refit(root: Path, cfg: dict) -> dict:
    """Fresh h16/f2 M1 and h96/f2 quantile fits, with no final/test path."""
    history = load_development_history(root)
    original_manifest = json.loads((root / "_validation/session_0925_original/logs/development_selection.json").read_text(encoding="utf-8"))
    contexts = build_contexts(history, cfg, original_manifest, horizons=(16, 96))
    original = _safe_oof(root / ORIGINAL)
    m1_reference = _safe_oof(root / P2_SNAPSHOT / "predictions" / Path(P2_CANDIDATES[0]).name)
    m4_reference = _safe_oof(root / P2_SNAPSHOT / "predictions" / Path(P2_CANDIDATES[1]).name)
    integrated = _safe_oof(root / "outputs/predictions/development_oof.csv")
    h16 = fit_residual_fold(history, cfg, contexts[(16, 2)], 16, 2)
    m1 = compare_prediction_rows(m1_reference.loc[m1_reference.fold.eq(2) & m1_reference.horizon.eq(16)],
                                 h16.predictions, exact_reference_set=True,
                                 label="fresh h16/f2 residual")
    m1_integrated = compare_prediction_rows(h16.predictions, integrated,
                                            exact_reference_set=False,
                                            label="fresh/integrated h16/f2 residual")
    context = contexts[(96, 2)]
    x, targets = context["x"], context["targets"]
    fit, stop, cal, score = (context[name] for name in ("fit", "stop", "cal", "score"))
    cols = [col for col in x.columns if col not in
            {"is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day"}]
    models = fit_quantiles(x.loc[fit, cols], targets.loc[fit, "y"],
                           x.loc[stop, cols], targets.loc[stop, "y"], cfg,
                           params=context["expected"]["chosen_params"])
    q_cal = predict_quantiles(models, x.loc[cal, cols])
    q_score = predict_quantiles(models, x.loc[score, cols])
    raw = original.loc[original.horizon.eq(96) & original.fold.eq(2)
                       & original.model.eq("lgbm_quantile_raw")].sort_values("origin")
    if not pd.DatetimeIndex(raw.origin).equals(score):
        raise AssertionError("Fresh h96/f2 raw score origin grid changed")
    raw_diffs = {}
    for alpha in (.1, .5, .9, .95, .975):
        name = "q975" if alpha == .975 else f"q{int(alpha*100)}"
        diff = float(np.max(np.abs(q_score[alpha]-raw[name].to_numpy(float))))
        if diff > TOL:
            raise AssertionError(f"Fresh h96/f2 raw {name} differs")
        raw_diffs[name] = diff
    y_cal = targets.loc[cal, "y"].to_numpy(float)
    b_cal, b_score, edge = _baseline_b_quantiles(q_cal, q_score, y_cal, cfg)
    half, horizon = len(cal)//2, 96
    cutoff_start = min(len(cal), half+horizon+1)
    initial = pd.DataFrame({"target_time": targets.loc[cal[:half], "target_time"].to_numpy(),
                            "y": y_cal[:half], "q95": q_cal[.95][:half]})
    later_cal = cal[cutoff_start:]
    stream_origin = later_cal.append(score)
    stream = pd.DataFrame({"origin": stream_origin,
                           "target_time": targets.loc[stream_origin, "target_time"].to_numpy(),
                           "y": targets.loc[stream_origin, "y"].to_numpy(float),
                           "q95": np.concatenate([q_cal[.95][cutoff_start:], q_score[.95]]),
                           "q90_cal": np.concatenate([b_cal[.9][cutoff_start:], b_score[.9]]),
                           "q95_cal": np.concatenate([b_cal[.95][cutoff_start:], b_score[.95]]),
                           "q975_cal": np.concatenate([b_cal[.975][cutoff_start:], b_score[.975]]),
                           "is_score": [False]*len(later_cal)+[True]*len(score)})
    score_mask = stream.is_score.to_numpy(bool)
    fresh_frames, comparisons, integrated_comparisons = [h16.predictions], {}, {}
    for name, gamma in CANDIDATES.items():
        online = replay_adaptive_q95(initial, stream, gamma=gamma)
        q95 = online.q95_cal_online.to_numpy(float)
        q975 = online.q975_cal_online.to_numpy(float)
        knots = {.1: np.concatenate([q_cal[.1][cutoff_start:], q_score[.1]]),
                 .5: np.concatenate([q_cal[.5][cutoff_start:], q_score[.5]]),
                 .9: stream.q90_cal.to_numpy(float), .95: q95, .975: q975}
        prob = exceedance_from_quantiles(knots, float(context["tau"]))
        cutoff = _cutoff(stream.loc[~stream.is_score, "y"].to_numpy(float),
                         prob[~score_mask], float(context["tau"]),
                         stream.loc[~stream.is_score, "target_time"])
        from src.training import _row
        frame = _row(score, 96, 2, name, targets, float(context["tau"]), q_score[.5],
                     float("inf"), q10=q_score[.1], q50=q_score[.5], q90=q_score[.9],
                     q95=q_score[.95], q975=q_score[.975],
                     q90_cal=knots[.9][score_mask], q95_cal=q95[score_mask],
                     q975_cal=q975[score_mask], p_exceed=prob[score_mask],
                     q50_top_edge=edge, conformal_method=name,
                     latest_observation=online.latest_observation.loc[score_mask].to_numpy(),
                     fallback=online.fallback.loc[score_mask].to_numpy(),
                     beta_at_issue=online.beta_at_issue.loc[score_mask].to_numpy())
        frame["alert"] = prob[score_mask] > cutoff
        frame["alert_cutoff"] = cutoff
        reference = m4_reference.loc[m4_reference.horizon.eq(96)
                                     & m4_reference.fold.eq(2)
                                     & m4_reference.model.eq(name)]
        comparisons[name] = compare_prediction_rows(reference, frame,
                                       exact_reference_set=True, label=f"fresh h96/f2 {name}")
        integrated_comparisons[name] = compare_prediction_rows(
            frame, integrated, exact_reference_set=False, label=f"fresh/integrated h96/f2 {name}")
        fresh_frames.append(frame)
    archive = root / "_validation/session_0925_partial_refit.csv"
    archive.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(fresh_frames, ignore_index=True).to_csv(archive, index=False, encoding="utf-8-sig")
    # Verify our own evidence file through the same metadata-first reader.
    _safe_oof(archive)
    return {"h16_fold2_residual": m1, "h16_fold2_integrated": m1_integrated,
            "h96_fold2_adaptive": comparisons,
            "h96_fold2_integrated": integrated_comparisons,
            "h96_fold2_raw_quantile_max_abs": raw_diffs,
            "model_fit_count": 6, "archive": archive.relative_to(root).as_posix(),
            "archive_sha256": _sha256(archive)}


def verify_p1_reproduction(root: Path, pinned: dict) -> dict:
    """Attach the separately run P1 fresh replay without running it twice."""
    path = root / "outputs/logs/P3_predictability_repro.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "pass" or report.get("original_inputs_unchanged") is not True
            or report.get("scope") != "sealed_original_development_oof_only"
            or report.get("sealed_boundary_exclusive") != str(SEALED_BOUNDARY)
            or report.get("absolute_tolerance") != TOL):
        raise AssertionError("Independent P1 reproduction has no passing sealed result")
    original_files = {
        "_validation/session_0925_original/predictions/development_oof.csv":
            pinned["original_artifacts"]["predictions/development_oof.csv"],
        "_validation/session_0925_original/logs/development_selection.json":
            pinned["original_artifacts"]["logs/development_selection.json"],
    }
    if report.get("original_input_sha256") != original_files:
        raise AssertionError("Independent P1 replay did not use preflight-pinned original inputs")
    if (report.get("A1", {}).get("peak_starts", {}).get("rows") != 148
            or report.get("A2", {}).get("fixed_hypotheses") != 16
            or report["A2"].get("eligible_hypotheses") != 0
            or report["A2"].get("selected_score_peaks") != 0
            or report["A2"].get("h16_score_cohort") != 14
            or report["A2"].get("h96_score_cohort") != 8):
        raise AssertionError("Independent P1 replay changed its fixed cohort")
    checked_tables = {}
    for phase, entries in (("A1", ("restart_slots", "peak_starts", "proximity")),
                           ("A2", ("cohorts", "restart_slots", "score_cohort")),
                           ("A4", ("hypotheses", "folds", "breakdown", "paired_rows"))):
        for name in entries:
            item = report.get(phase, {}).get(name, {})
            relative = item.get("source", "")
            source = (root / relative).resolve()
            if (not source.is_relative_to((root / "outputs/analysis_p1").resolve())
                    or not source.is_file()
                    or float(item.get("max_abs_numeric_difference", float("inf"))) > TOL):
                raise AssertionError(f"Independent P1 {phase}.{name} table proof is missing")
            table = pd.read_csv(source)
            if table.shape != (item.get("rows"), item.get("columns")):
                raise AssertionError(f"Independent P1 {phase}.{name} table shape changed")
            checked_tables[f"{phase}.{name}"] = {"rows": len(table), "sha256": _sha256(source),
                                                    "max_abs_difference": item["max_abs_numeric_difference"]}
    for section in (report["A1"].get("summary", {}), report["A1"].get("hypothesis", {})):
        if float(section.get("max_abs_numeric_difference", float("inf"))) > TOL:
            raise AssertionError("Independent P1 A1 statistic differs")
    observed = pd.read_csv(root / "outputs/analysis_p1/A4_hypotheses.csv").set_index("horizon")
    if len(report["A4"].get("effects", [])) != 2:
        raise AssertionError("Independent P1 A4 horizon count changed")
    for effect in report["A4"]["effects"]:
        row = observed.loc[int(effect["horizon"])]
        for key in ("estimate", "ci_low", "ci_high"):
            if abs(float(effect[key]) - float(row[key])) > TOL:
                raise AssertionError(f"Independent P1 A4 {key} changed")
    return {"status": "pass", "report_sha256": _sha256(path),
            "tables": checked_tables, "a1_peak_starts": 148,
            "a2_eligible_hypotheses": 0,
            "a4_effects": report["A4"]["effects"]}


def verify_session(root: str | Path, *, refit_partial: bool = False) -> dict:
    root = Path(root).resolve()
    if not (root / "configs/default.yaml").is_file():
        raise FileNotFoundError("Verification root has no pinned configuration")
    no_final = _assert_no_final(root)
    protected = verify_protected(root)
    pinned = json.loads((root / "outputs/logs/session_0925_preflight.json").read_text(encoding="utf-8"))
    prereg = verify_preregistrations(root)
    p2_snapshot = verify_p2_snapshot(root)
    parity = verify_oof_parity(root)
    selection = _compare_original_selection(root)
    p1_reproduction = verify_p1_reproduction(root, pinned)
    report = {"scope": "sealed_development_only", "boundary_exclusive": str(SEALED_BOUNDARY),
              "protected": protected, "preregistrations": prereg,
              "p2_snapshot_sha256": p2_snapshot,
              "oof_parity": parity, "selection": selection, "no_final": no_final,
              "p1_reproduction": p1_reproduction,
              "git_head": _git(root, "rev-parse", "HEAD").decode("ascii").strip(),
              "partial_refit_executed": bool(refit_partial)}
    if refit_partial:
        cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
        report["partial_refit"] = _partial_refit(root, cfg)
    # Recheck after any expensive refit and before publishing the report.
    if (verify_protected(root) != protected or verify_p2_snapshot(root) != p2_snapshot
            or _assert_no_final(root) != no_final):
        raise AssertionError("Protected, snapshot or holdout state changed during P3 verification")
    output = root / "outputs/logs/session_0925_repro.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refit-partial", action="store_true")
    args = parser.parse_args()
    result = verify_session(args.root, refit_partial=args.refit_partial)
    print(json.dumps({"status": "verified", "test_opened": False,
                      "partial_refit_executed": result["partial_refit_executed"],
                      "original_rows": result["oof_parity"]["original"]["rows"],
                      "integrated_rows": result["oof_parity"]["integrated_rows"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
