"""Focused evidence and hourly-normalization checks for final WF reporting."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f.registry import config_hash, sha256
from phase_f import wf_reporting as reporting
from phase_f.wf_evaluation import BASELINES


def _frame(arm: str, *, partial: bool = False) -> pd.DataFrame:
    times = pd.date_range("2021-07-05 00:00", periods=4 if not partial else 3, freq="15min")
    return pd.DataFrame({"role": "score", "arm": arm, "fold": 0, "horizon": 4,
                         "target_time": times, "y": np.array([10., 20., 30., 40.])[:len(times)],
                         "pred": np.array([11., 21., 31., 41.])[:len(times)]})


def _sealed(value: dict) -> dict:
    return {**value, "lock_sha256": config_hash(value)}


def test_hourly_uses_four_actual_slots_and_actual_mean():
    result = reporting.hourly_metrics(_frame("EXPLORE"), "EXPLORE")
    assert result["n_hours"] == 1
    assert result["actual_hourly_mean"] == 25
    assert result["CVRMSE"] == pytest.approx(1 / 25)
    assert result["NMBE"] == pytest.approx(-1 / 25)
    assert reporting.hourly_metrics(_frame("EXPLORE", partial=True), "EXPLORE")["n_hours"] == 0


def test_hourly_does_not_mix_folds_or_duplicate_targets():
    frame = _frame("EXPLORE")
    duplicated = pd.concat([frame, frame.assign(fold=1)], ignore_index=True)
    assert reporting.hourly_metrics(duplicated, "EXPLORE")["n_hours"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        reporting.hourly_metrics(pd.concat([frame, frame], ignore_index=True), "EXPLORE")


def test_final_report_requires_matching_sealed_completion(tmp_path):
    prepared = SimpleNamespace(root=tmp_path, out=tmp_path / "outputs/phase_f/walkforward_v2")
    lock = _sealed({"candidates": [], "specs": [], "prediction_sha256": {}})
    done = _sealed({"selection_lock_sha256": lock["lock_sha256"], "holdout_read": False,
                    "results": {"WF": {}}, "artifacts": {}})
    with pytest.raises(ValueError, match="absent"):
        reporting.final_report(prepared, lock, done)
    assert not (tmp_path / "outputs/phase_f/PHASE_F_REPORT.md").exists()


def test_final_report_uses_confirm_fields_without_mislabeling(tmp_path, monkeypatch):
    out = tmp_path / "outputs/phase_f/walkforward_v2"
    (out / "logs/experiments").mkdir(parents=True)
    (out / "logs/tuning").mkdir(parents=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "finalists_v2/predictions/EXPLORE").mkdir(parents=True)
    candidate = "F1-example"
    keys = [*BASELINES.values(), candidate]
    row = {"exp_id": candidate, "family": "F1", "status": "completed",
           "wf_explore_AUC_MAE": 4., "wf_explore_AUC_PeakMAE": 5.,
           "wf_explore_h16_Peak": 6., "n_seeds": 1, "config_json": "{}"}
    (out / "logs/experiments" / f"{candidate}.json").write_text(json.dumps(row), encoding="utf-8")
    predictions = {}
    for key in keys:
        path = out / "finalists_v2/predictions/EXPLORE" / f"{key}.parquet"
        path.write_bytes(key.encode())
        predictions[key] = sha256(path)
    artifact = out / "finalists_v2/tables/evidence.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("sealed", encoding="utf-8")
    lock = _sealed({"candidates": [candidate], "specs": [{"id": k} for k in keys],
                    "prediction_sha256": predictions})
    fields = {"wf_explore_AUC_MAE": 3.5, "wf_explore_AUC_PeakMAE": 4.5,
              "wf_explore_h16_Peak": 5.5, "confirm_gates_met": True,
              "E_c10_22_recall": 1.0}
    done = _sealed({"selection_lock_sha256": lock["lock_sha256"], "holdout_read": False,
                    "results": {"WF": {key: fields for key in keys},
                                "pc3": {key: fields for key in keys}},
                    "representative_candidate": candidate,
                    "verdict": "candidate_proposal",
                    "artifacts": {str(artifact.relative_to(out)).replace("\\", "/"): sha256(artifact)},
                    "overlapping_alternatives": []})
    (out / "logs/confirm_once.json").write_text(json.dumps(done), encoding="utf-8")
    def table(_view, _key, _arm, name):
        if name == "pairwise_ci":
            return pd.DataFrame({"horizon": [np.nan], "baseline": ["B5"], "dataset": ["D1"],
                                 "metric": ["AUC_MAE_improvement"], "estimate": [1.],
                                 "ci_low": [.1], "ci_high": [1.9], "ci_status": ["available"],
                                 "n_blocks": [8], "bootstrap_n": [1000]})
        if name == "pooled":
            return pd.DataFrame({"dataset": ["D1"], "nMAE": [.1], "CVRMSE": [.2], "NMBE": [.01]})
        if name == "daily_summary":
            return pd.DataFrame({"dataset": ["D1"], "scope": ["pooled"], "subset": ["full_96"],
                                 "horizon": [16], "daily_max_MAE": [2.],
                                 "timing_MAE_minutes": [15.], "n_days": [4]})
        raise AssertionError(name)
    monkeypatch.setattr(reporting, "_table", table)
    monkeypatch.setattr(reporting, "load_predictions", lambda _view, _key, arm="EXPLORE": _frame(arm))
    monkeypatch.setattr(reporting, "_figures", lambda *_: None)
    monkeypatch.setattr(reporting, "_diagnostics", lambda *_: {"EXPLORE": {"score_n": 4},
                                                                 "CONFIRM": {"score_n": 4}})
    path = reporting.final_report(SimpleNamespace(root=tmp_path, out=out), lock, done)
    text = path.read_text(encoding="utf-8")
    assert "CONFIRM" in text and "3.5000" in text
    assert "Field names retain `wf_explore_`" in text
    assert "hourly CV(RMSE) 30% reference" in text
    assert "no combined-episode claim" in text
    assert (out / "report/top20_explore.csv").is_file()
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        reporting.final_report(SimpleNamespace(root=tmp_path, out=out), lock, done)


def test_all_requested_figures_render_from_locked_table_shapes(tmp_path, monkeypatch):
    candidate = "F1-core-lightgbm"
    def table(_view, _key, _arm, name):
        if name == "pooled":
            return pd.DataFrame({"dataset": ["D1", "D1"], "horizon": [4, 16],
                                 "MAE": [1., 2.], "Peak_MAE": [2., 3.]})
        if name == "fold":
            return pd.DataFrame({"dataset": ["D1", "D1"], "fold": [0, 1],
                                 "MAE": [1., 2.]})
        raise AssertionError(name)
    monkeypatch.setattr(reporting, "_table", table)
    monkeypatch.setattr(reporting, "load_predictions", lambda *_: _frame("EXPLORE").assign(
        horizon=16, tau=15., fold=0))
    rows = [{"exp_id": candidate, "status": "completed", "wf_explore_AUC_MAE": 1.,
             "config_json": "{}"},
            {"exp_id": "F6-1-c1024-native_mean", "status": "completed",
             "wf_explore_AUC_MAE": 2., "config_json": '{"context_length":1024}'}]
    reporting._figures(SimpleNamespace(out=tmp_path), rows, tmp_path / "figures", candidate)
    expected = {"horizon_mae_peak", "context_length", "feature_ablation",
                "peak_signed_errors", "representative_week", "weekly_b5_delta"}
    assert {p.stem for p in (tmp_path / "figures").glob("*.png")} == expected


def test_preflight_claim_requires_lock_digest(tmp_path):
    folder = tmp_path / "outputs/phase_f/logs/revision_20261006"
    folder.mkdir(parents=True)
    path = folder / "preflight_audit.json"
    path.write_text(json.dumps({"full_pc3_baselines": {"R1": {"AUC_MAE": 6.4,
                            "AUC_PeakMAE": 15.3}}, "R1_prediction_max_abs_difference": 0,
                            "R1_parity_passed": True,
                            "candidate_confirm_metric_opened_evidence": False,
                            "audit_limit": "known logs only"}), encoding="utf-8")
    prepared = SimpleNamespace(root=tmp_path)
    assert "UNKNOWN" in reporting._preflight_evidence(prepared, {})[0]
    assert "6.4000" in reporting._preflight_evidence(
        prepared, {"preflight_sha256": sha256(path)})[0]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        reporting._preflight_evidence(prepared, {"preflight_sha256": "0" * 64})
