"""Synthetic tests for the standalone Phase C artifact contract."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import verify_phase_c_artifacts as audit


def _synthetic():
    cfg = {"boundary": "2021-08-09 09:45:00", "peak_weight": 2.0,
           "cbl_candidates": ["cbl"], "ridge": {"alpha": 100.0, "scaler_ddof": 1},
           "mstl": {"periods": [96, 672]}, "kalman": {"phi_bounds": [-.99, .99]},
           "chronos": {"reference_only": True},
           "lgbm": {"candidates": [{"id": "leaf", "num_leaves": 15, "min_child_samples": 40}]},
           "tcn": {"candidates": [{"id": "conv", "channels": 32, "dropout": .1}]}}
    lock = {"baseline_status": "prepared",
            "cbl_by_horizon": {str(h): "cbl" for h in audit.HORIZONS},
            "lgbm": {"source": "stop_only", "selected_id": "leaf",
                     "selected_config": {"num_leaves": 15, "min_child_samples": 40}},
            "tcn": {"source": "stop_only", "selected_id": "conv",
                    "selected_config": {"channels": 32, "dropout": .1}}}
    selected = audit._locked_parts(lock, cfg)
    contexts, rows = {}, []
    for h in audit.HORIZONS:
        for f in audit.FOLDS:
            score = pd.date_range("2021-01-04", periods=2, freq="15min", name="origin")
            target = score + pd.Timedelta(minutes=15 * h)
            y = pd.Series([10.0, 11.0], index=score)
            d2 = pd.Series([False, (h + f) % 2 == 0], index=score)
            contexts[(h, f)] = {"score": score, "fit": score - pd.Timedelta(days=5),
                                "stop": score - pd.Timedelta(days=4),
                                "cal": score - pd.Timedelta(days=3),
                                "target_time": pd.Series(target, index=score),
                                "y": y, "tau": 12.0, "d2": d2}
            for model in audit.MAIN:
                for origin, t, truth, novel in zip(score, target, y, d2):
                    rows.append({"model": model, "horizon": h, "fold": f,
                                 "origin": origin, "target_time": t, "y": truth,
                                 "pred": truth + 0.2, "tau": 12.0, "d2": bool(novel),
                                 "train_seconds": 1.0, "inference_seconds": .1,
                                 "selected_config": json.dumps(audit._expected_config(model, h, selected, cfg)),
                                 "development_only": True})
    return pd.DataFrame(rows), contexts, lock, cfg


def test_main10_exact_keys_and_zero_d2_coverage():
    frame, contexts, lock, cfg = _synthetic()
    result = audit.validate_predictions(frame, contexts, lock, cfg)
    assert result["paired_cohort_shrinkage"] == 0
    assert len(result["D1_D2_coverage_by_model_horizon_fold"]) == 390
    assert any(item["D2_expected"] == 0 for item in result["D1_D2_coverage_by_model_horizon_fold"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "c1_config", "wrong_d2"])
def test_rejects_missing_or_corrupt_main_rows(mutation):
    frame, contexts, lock, cfg = _synthetic()
    if mutation == "missing":
        frame = frame.iloc[1:].copy()
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "nonfinite":
        frame.loc[0, "pred"] = np.nan
    elif mutation == "c1_config":
        index = frame.index[frame.model.eq("C1")][0]
        frame.loc[index, "selected_config"] = json.dumps({"target_transform": "direct"})
    else:
        frame.loc[0, "d2"] = True
    with pytest.raises(AssertionError):
        audit.validate_predictions(frame, contexts, lock, cfg)


def test_existing_manifest_hash_change_is_rejected(monkeypatch):
    class ExistingPath:
        def is_file(self):
            return True

    prior = {"artifacts": [{"path": "models/fitted.joblib", "sha256": "old"}],
             "sealed": {"context_cache": {"sha256": "context"},
                        "implementation_hashes": {"phase_c/training.py": "code"},
                        "nonraw_bound_inputs": {"configs/phase_c.json": "config"},
                        "raw_source_locked_without_reread": {"raw": "digest"},
                        "verifier_script_sha256": "verifier"},
             "preregistration_lock": {"sha256": "prereg"},
             "model_config_lock": {"sha256": "lock"}}
    current = copy.deepcopy(prior)
    monkeypatch.setattr(audit, "_read_json", lambda path: prior)
    assert audit._assert_existing_manifest_unchanged(ExistingPath(), current) is True
    current["artifacts"][0]["sha256"] = "changed"
    with pytest.raises(AssertionError, match="Existing output_cache_manifest"):
        audit._assert_existing_manifest_unchanged(ExistingPath(), current)
    current = copy.deepcopy(prior)
    current["model_config_lock"]["sha256"] = "rewritten"
    with pytest.raises(AssertionError, match="model lock hash changed"):
        audit._assert_existing_manifest_unchanged(ExistingPath(), current)


def test_optional_reference_reports_explicit_zero_coverage():
    _, contexts, _, cfg = _synthetic()
    result = audit._reference_coverage(audit.Path("absent_synthetic_phase_c_output"), contexts, cfg)
    assert result["status"] == "not_available"
    assert result["reference_only"] is True
    assert result["actual_total"] == result["finite_total"] == 0
    assert len(result["by_horizon_fold"]) == 39
    assert all(item["actual"] == item["finite"] == 0 for item in result["by_horizon_fold"])


def _fake_locked_cbl(monkeypatch):
    from src.models import cbl

    def predict(history, origins, horizon):
        expected = np.array([10.0 + .2, 11.0 + .2])
        if horizon == 4:
            expected[0] = np.nan
        return pd.DataFrame({"cbl": expected}, index=origins)

    monkeypatch.setattr(cbl, "cbl_all_predictions", predict)


def test_exact_source_certified_b2_missingness_preserves_raw_keys(monkeypatch):
    frame, contexts, lock, cfg = _synthetic()
    _fake_locked_cbl(monkeypatch)
    b2_first = frame.model.eq("B2") & frame.horizon.eq(4) & frame.origin.eq(pd.Timestamp("2021-01-04"))
    frame.loc[b2_first, "pred"] = np.nan
    certified = audit.certify_b2_missingness(frame, pd.DataFrame(), contexts, lock, cfg)
    result = audit.validate_predictions(frame, contexts, lock, cfg, certified)
    assert result["certified_B2_structural_missing_keys"] == 3
    assert result["paired_cohort_shrinkage"] == 3
    assert result["main_prediction_rows"] == result["score_origin_keys_per_model"] * 10
    assert all(item["removed"] == 3 for item in result["per_model_removal_counts"].values())
    assert any(item["removed_for_pairing"] == 1 for item in result["D1_D2_coverage_by_model_horizon_fold"])
    from phase_c.evaluation import _prepare
    _, evaluator_counts = _prepare(frame)
    assert result["per_model_removal_counts"] == evaluator_counts


def test_source_certification_rejects_extra_b2_nan(monkeypatch):
    frame, contexts, lock, cfg = _synthetic()
    _fake_locked_cbl(monkeypatch)
    b2_first = frame.model.eq("B2") & frame.horizon.eq(4) & frame.origin.eq(pd.Timestamp("2021-01-04"))
    frame.loc[b2_first, "pred"] = np.nan
    unexplained = frame.model.eq("B2") & frame.horizon.eq(5) & frame.fold.eq(0)
    frame.loc[unexplained.idxmax(), "pred"] = np.nan
    with pytest.raises(AssertionError, match="differs from sealed CBL source"):
        audit.certify_b2_missingness(frame, pd.DataFrame(), contexts, lock, cfg)


def test_evaluator_removal_must_match_certified_cohort(monkeypatch):
    counts = {model: {"input": 10, "common_finite": 8, "removed": 2} for model in audit.MAIN}
    monkeypatch.setattr(audit, "_read_json", lambda path: {"removed_counts": copy.deepcopy(counts)})
    assert audit._compare_evaluator_removal(audit.Path("unused"), counts)["MAIN10_removed_counts_match"]
    changed = copy.deepcopy(counts)
    changed["B2"]["removed"] = 3
    with pytest.raises(AssertionError, match="Evaluator paired-cohort removal differs"):
        audit._compare_evaluator_removal(audit.Path("unused"), changed)
