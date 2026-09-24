"""Synthetic checks for the Phase 3 sealed reproduction verifier."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from scripts.verify_session_0925 import (_assert_no_final, _safe_oof,
                                         _sha256, compare_nested, compare_prediction_rows,
                                         verify_preregistrations, verify_protected)
from src.session_data import SEALED_BOUNDARY


def _frame(model="original"):
    return pd.DataFrame({
        "origin": pd.to_datetime(["2021-04-01 09:00", "2021-04-01 09:15"]),
        "target_time": pd.to_datetime(["2021-04-01 10:00", "2021-04-01 10:15"]),
        "horizon": [4, 4], "fold": [0, 0], "model": [model, model],
        "y": [100., 102.], "pred": [99., 103.], "tau": [110., 110.],
        "q95_cal": [112., 114.], "alert": [False, True],
        "latest_observation": ["2021-04-01 08:30", "2021-04-01 08:45"],
    })


def test_paired_oof_numeric_and_provenance_parity():
    original = _frame()
    augmented = pd.concat([original, _frame("candidate")], ignore_index=True)
    assert compare_prediction_rows(original, augmented, exact_reference_set=False,
                                   label="synthetic")["rows"] == 2
    tiny = original.copy()
    tiny.loc[0, "pred"] += 1e-11
    compare_prediction_rows(original, tiny, exact_reference_set=True, label="tiny")
    too_large = original.copy()
    too_large.loc[0, "pred"] += 1e-7
    with pytest.raises(AssertionError, match="pred exceeds"):
        compare_prediction_rows(original, too_large, exact_reference_set=True, label="large")
    changed = original.copy()
    changed.loc[0, "latest_observation"] = "2021-04-01 09:30"
    with pytest.raises(AssertionError, match="provenance"):
        compare_prediction_rows(original, changed, exact_reference_set=True, label="changed")
    missing = original.drop(columns=["q95_cal"])
    with pytest.raises(AssertionError, match="prediction columns missing.*q95_cal"):
        compare_prediction_rows(original, missing, exact_reference_set=True, label="missing q95")
    missing_alert = original.drop(columns=["alert"])
    with pytest.raises(AssertionError, match="prediction columns missing.*alert"):
        compare_prediction_rows(original, missing_alert, exact_reference_set=True, label="missing alert")


def test_nested_selection_compares_numbers_to_absolute_tolerance_and_text_exactly():
    before = {"selection": {"model": "old", "reason": "FP CI", "adopted": True,
                            "effects": [0.1, float("nan")]}}
    tiny = {"selection": {"model": "old", "reason": "FP CI", "adopted": True,
                          "effects": [0.1 + 1e-15, float("nan")]}}
    assert compare_nested(before, tiny, "selection") < 1e-10
    bad_reason = {"selection": {**tiny["selection"], "reason": "new rule"}}
    with pytest.raises(AssertionError, match="reason"):
        compare_nested(before, bad_reason, "selection")
    bad_bool = {"selection": {**tiny["selection"], "adopted": 1}}
    with pytest.raises(AssertionError, match="boolean"):
        compare_nested(before, bad_bool, "selection")


def test_rejects_final_artifact_and_oof_target_before_parsing_value(tmp_path: Path):
    assert _assert_no_final(tmp_path)["test_opened"] is False
    final = tmp_path / "outputs/tables/final_test.csv"
    final.parent.mkdir(parents=True)
    final.write_text("not read", encoding="utf-8")
    with pytest.raises(AssertionError, match="sealed final"):
        _assert_no_final(tmp_path)
    bad = tmp_path / "unsafe_oof.csv"
    bad.write_text("origin,target_time,horizon,fold,model,y,pred,tau\n"
                   "2021-08-09 09:30:00,2021-08-09 09:45:00,1,0,a,NOT_A_NUMBER,1,2\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="sealed development boundary"):
        _safe_oof(bad)


def test_protected_75_hashes_detect_content_change(tmp_path: Path):
    folder = tmp_path / "verification"
    folder.mkdir()
    for number in range(75):
        (folder / f"file_{number:02}.txt").write_text(f"evidence {number}", encoding="utf-8")
    protected = {f"verification/{path.name}": _sha256(path) for path in sorted(folder.iterdir())}
    original = tmp_path / "_validation/session_0925_original"
    artifacts = {}
    for relative in ("predictions/development_oof.csv", "tables/development_cv.csv",
                     "logs/development_selection.json", "logs/development_cache.json"):
        artifact = original / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(relative, encoding="utf-8")
        artifacts[relative] = _sha256(artifact)
    preflight = tmp_path / "outputs/logs/session_0925_preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"protected": protected,
                                     "original_artifacts": artifacts,
                                     "boundary": str(SEALED_BOUNDARY),
                                     "test_opened": False}), encoding="utf-8")
    assert verify_protected(tmp_path)["count"] == 75
    (folder / "file_07.txt").write_text("modified", encoding="utf-8")
    with pytest.raises(AssertionError, match="Protected preflight"):
        verify_protected(tmp_path)


def test_preregistration_first_commit_is_immutable(tmp_path: Path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    git("init", "-q")
    git("config", "user.name", "Verifier Test")
    git("config", "user.email", "verifier@example.invalid")
    folder = tmp_path / "outputs/logs"
    folder.mkdir(parents=True)
    for phase in ("P1", "P2", "P3"):
        path = folder / f"preregistration_0925_{phase}.md"
        path.write_text(f"{phase} fixed rule\n", encoding="utf-8")
        git("add", str(path.relative_to(tmp_path)))
        git("commit", "-qm", f"[{phase}-REG] fixed")
    assert len(verify_preregistrations(tmp_path)) == 3
    (folder / "preregistration_0925_P2.md").write_text("changed after seeing results\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="changed after first commit"):
        verify_preregistrations(tmp_path)
