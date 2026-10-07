"""Small CPU-only checks for Phase F reporting boundaries and completion."""
from __future__ import annotations

import json

import pandas as pd

from phase_f import reporting
from phase_f.registry import config_hash, sha256


def _out(tmp_path):
    out = tmp_path / "outputs" / "phase_f"
    (out / "logs").mkdir(parents=True)
    return out


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_progress_remains_explore_only_without_candidate(tmp_path, monkeypatch):
    out = _out(tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Progress without candidate must not read predictions")))
    result = reporting.generate_progress(tmp_path)
    assert result["full_phase_complete"] is False
    assert result["confirm_read"] is False
    text = (out / "progress_report.md").read_text(encoding="utf-8")
    assert "진행 중" in text and "CONFIRM" in text
    assert all((out / "figures" / name).is_file() for name in reporting.FIGURES)
    assert not (out / "PHASE_F_REPORT.md").exists()


def test_final_rejects_missing_or_unresolved_completion_without_confirm_read(tmp_path, monkeypatch):
    out = _out(tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("CONFIRM prediction must stay closed")))
    assert reporting.generate_final(tmp_path)["full_phase_complete"] is False
    split = {"lock_sha256": "split-sha"}
    _write(out / "logs" / "split_lock.json", split)
    lock = {"primary_candidate": None, "candidates": [], "selection_arm": "EXPLORE",
            "config_hashes": {}, "prediction_hashes": {}, "split_lock_sha256": "split-sha",
            "holdout_read": False, "historical_final_artifact_read": False}
    lock["lock_sha256"] = config_hash(lock)
    _write(out / "logs" / "confirm_lock.json", lock)
    tables = {}
    for model in ("B5", "M1", "R1"):
        path = out / "tables" / "confirm" / model / "confirm_auc.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"dataset": d, "n_horizons": 13, "n_available_horizons": 13,
                       "AUC_MAE": 1.0, "AUC_PeakMAE": 2.0} for d in ("D1", "D2")]).to_csv(path, index=False)
        tables[path.relative_to(out).as_posix()] = sha256(path)
    done = {"complete": True, "lock_sha256": lock["lock_sha256"],
            "primary_candidate": None, "candidates": [], "tables": tables,
            "primary_confirmed": False, "holdout_read": False,
            "historical_final_artifact_read": False}
    done["confirm_once_sha256"] = config_hash(done)
    _write(out / "logs" / "confirm_complete.json", done)
    assert reporting.generate_final(tmp_path)["reason"] == "completion_gate_absent"
    _write(out / "logs" / "completion_gate.json",
           {"full_phase_complete": True, "required_family_coverage": {f: True for f in reporting.FAMILIES},
            "downstream_resolved": False, "expansion_converged": True,
            "holdout_read": False, "historical_final_artifact_read": False})
    assert reporting.generate_final(tmp_path)["full_phase_complete"] is False
    assert not (out / "PHASE_F_REPORT.md").exists()


def test_no_eligible_primary_can_be_reported_as_completed_failure(tmp_path, monkeypatch):
    out = _out(tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("No candidate means no candidate parquet")))
    _write(out / "logs" / "split_lock.json", {"lock_sha256": "split-sha"})
    lock = {"primary_candidate": None, "candidates": [], "selection_arm": "EXPLORE",
            "config_hashes": {}, "prediction_hashes": {}, "split_lock_sha256": "split-sha",
            "holdout_read": False, "historical_final_artifact_read": False}
    lock["lock_sha256"] = config_hash(lock)
    _write(out / "logs" / "confirm_lock.json", lock)
    tables = {}
    for model in ("B5", "M1", "R1"):
        path = out / "tables" / "confirm" / model / "confirm_auc.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"dataset": d, "n_horizons": 13, "n_available_horizons": 13,
                       "AUC_MAE": 1.0, "AUC_PeakMAE": 2.0} for d in ("D1", "D2")]).to_csv(path, index=False)
        tables[path.relative_to(out).as_posix()] = sha256(path)
    done = {"complete": True, "lock_sha256": lock["lock_sha256"],
            "primary_candidate": None, "candidates": [], "tables": tables,
            "primary_confirmed": False, "holdout_read": False,
            "historical_final_artifact_read": False}
    done["confirm_once_sha256"] = config_hash(done)
    _write(out / "logs" / "confirm_complete.json", done)
    weekly = out / "logs" / "weekly_diagnostic_status.json"
    _write(weekly, {"status": "unavailable", "reason": "No supported locked tabular candidate",
                    "selection_lock_sha256": lock["lock_sha256"],
                    "confirm_once_sha256": done["confirm_once_sha256"],
                    "holdout_read": False, "historical_final_artifact_read": False})
    gate = {"full_phase_complete": False, "ready_for_report": True,
            "required_family_coverage": {f: True for f in reporting.FAMILIES},
            "downstream_resolved": True, "expansion_converged": True,
            "holdout_read": False, "historical_final_artifact_read": False,
            "confirm_once_sha256": done["confirm_once_sha256"],
            "weekly_diagnostic_status_sha256": sha256(weekly)}
    _write(out / "logs" / "completion_gate.json", gate)
    pending = reporting.generate_final(tmp_path, provisional=True)
    assert pending["report_ready"] is True and pending["full_phase_complete"] is False
    assert (out / "PHASE_F_REPORT.pending.md").is_file()
    assert "full_phase_complete=false" in (out / "PHASE_F_REPORT.pending.md").read_text(encoding="utf-8")
    assert not (out / "PHASE_F_REPORT.md").exists()
    _write(out / "logs" / "completion_gate.json",
           {**gate, "full_phase_complete": True})
    result = reporting.generate_final(tmp_path)
    assert result["full_phase_complete"] is True
    assert result["primary_candidate"] is None
    text = (out / "PHASE_F_REPORT.md").read_text(encoding="utf-8")
    assert "적격 후보 부재" in text and "성능 개선 성공" in text


def test_period_respects_explicit_target_date_window():
    frame = pd.DataFrame({"target_time": pd.to_datetime(["2021-07-25", "2021-07-26", "2021-08-01", "2021-08-02"]),
                          "y": [1., 2., 5., 100.], "pred": [1., 3., 4., -500.],
                          "tau": [4., 4., 4., 4.]})
    row = reporting._period(frame, "2021-07-26", "2021-08-02")
    assert row["n"] == 2
    assert row["peak_n"] == 1
    assert row["signed_peak_error"] == -1.0


def test_f11_report_requires_hashed_tables_and_shows_observed_day_basis(tmp_path):
    out = _out(tmp_path)
    artifacts = {}
    risk, uncertainty, episodes = [], [], []
    for model in ("B5", "F1-core-ridge"):
        for subset in ("D1", "D2"):
            risk.append({"model": model, "subset": subset, "fold": "pooled",
                         "BS_cal": .12, "BSS": .04, "BSS_ci_lower": -.02,
                         "BSS_ci_upper": .08})
            uncertainty.append({"model": model, "subset": subset, "fold": "pooled",
                                "population": "all", "method": "conformal",
                                "coverage": .94, "coverage_ci_lower": .91,
                                "coverage_ci_upper": .97, "mean_width": 22.})
            for policy in ("1/1", "2/2"):
                episodes.append({"model": model, "subset": subset, "fold": "pooled",
                                 "threshold": .10, "policy": policy,
                                 "episode_recall": .7, "episode_recall_ci_low": .6,
                                 "episode_recall_ci_high": .8,
                                 "false_alert_episodes_per_operating_day": 1.2,
                                 "false_alert_episodes_per_operating_day_ci_low": .9,
                                 "false_alert_episodes_per_operating_day_ci_high": 1.5})
    for name, rows in (("risk_metrics", risk), ("uncertainty_metrics", uncertainty),
                       ("alert_episode_metrics", episodes),
                       ("comparison_vs_B5", [{"candidate": "F1-core-ridge"}]),
                       ("decision_value_curve", [{"model": "B5"}])):
        path = out / "tables" / "downstream" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        artifacts[path.relative_to(out).as_posix()] = sha256(path)
    _write(out / "logs" / "downstream_status.json",
           {"status": "completed", "artifacts": artifacts,
            "selection_lock_sha256": "selection", "confirm_once_sha256": "confirm"})
    lock = {"lock_sha256": "selection"}
    done = {"confirm_once_sha256": "confirm"}
    table, links, accepted = reporting._downstream_evidence(out, lock, done,
                                                            "F1-core-ridge")
    assert accepted, table
    assert "평가 관측일" in table and "BSS 95% CI" in table
    assert "1.2000" in table and links.count("](tables/downstream/") == 5
    path = out / "tables" / "downstream" / "risk_metrics.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _, _, accepted = reporting._downstream_evidence(out, lock, done,
                                                    "F1-core-ridge")
    assert not accepted


def test_f11_unavailable_requires_an_explicit_reason(tmp_path):
    out = _out(tmp_path)
    lock = {"lock_sha256": "selection"}
    done = {"confirm_once_sha256": "confirm"}
    path = out / "logs" / "downstream_status.json"
    base = {"status": "unavailable", "selection_lock_sha256": "selection",
            "confirm_once_sha256": "confirm"}
    _write(path, {**base, "reason": ""})
    assert reporting._downstream_evidence(out, lock, done, "F1-core-ridge")[2] is False
    _write(path, {**base, "reason": "No eligible calibrator"})
    assert reporting._downstream_evidence(out, lock, done, "F1-core-ridge")[2] is True


def test_weekly_diagnostic_requires_locked_config_and_all_artifact_digests(tmp_path):
    out = _out(tmp_path)
    candidate = "F1-core-ridge"
    config = {"id": candidate, "kind": "ridge", "family": "F1"}
    registry = pd.DataFrame([{"exp_id":candidate,"config_json":json.dumps(config),
                              "config_hash":config_hash(config)}])
    lock = {"lock_sha256":"selection", "weekly_diagnostic_candidate":candidate,
            "weekly_diagnostic_config_hash":config_hash(config)}
    done = {"confirm_once_sha256":"confirm"}
    weekly_lock = out / "logs" / "weekly_diagnostic_lock.json"
    _write(weekly_lock, {"candidate":candidate,"selection_lock_sha256":"selection",
                         "confirm_once_sha256":"confirm","frozen_config_hash":config_hash(
                             {**config,"horizon":16}),"horizon":16,
                         "score_origin_hashes":{str(i):f"fold{i}" for i in range(3)},
                         "holdout_read":False,"historical_final_artifact_read":False})
    artifacts = {}
    for fold in range(3):
        for relative in (f"predictions/F9-4-fold{fold}-predictions.parquet",
                         f"tables/F9-4-fold{fold}-updates.csv"):
            path = out / relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b"synthetic evidence")
            artifacts[relative] = sha256(path)
    status_path = out / "logs" / "weekly_diagnostic_status.json"
    _write(status_path, {"status":"completed","candidate":candidate,
                         "selection_lock_sha256":"selection","confirm_once_sha256":"confirm",
                         "artifacts":artifacts,"holdout_read":False,
                         "historical_final_artifact_read":False})
    _write(out / "logs" / "completion_gate.json",
           {"weekly_diagnostic_status_sha256":sha256(status_path),
            "weekly_diagnostic_lock_sha256":sha256(weekly_lock)})
    description, valid = reporting._weekly_evidence(out, lock, done, registry)
    assert valid, description
    assert "3-fold" in description
    path = out / "tables" / "F9-4-fold2-updates.csv"
    path.write_bytes(b"changed")
    description, valid = reporting._weekly_evidence(out, lock, done, registry)
    assert not valid and "해시 불일치" in description


def test_week_block_ci_is_sensitivity_only_and_shows_effective_weeks():
    rows = []
    for baseline in ("B5", "M1", "R1"):
        for dataset in ("D1", "D2"):
            for metric in ("AUC_MAE_improvement", "AUC_PeakMAE_degradation"):
                rows.append({"candidate":"F1-core-ridge", "baseline":baseline,
                             "dataset":dataset,"horizon":float("nan"),"metric":metric,
                             "block":"week","bootstrap_n":1000,"seed":42,"n_blocks":5,
                             "estimate":.2,"ci_low":float("nan"),"ci_high":float("nan"),
                             "ci_status":"unavailable","ci_reason":"degenerate_draws"})
    weekly = pd.DataFrame(rows)
    assert reporting._week_ci_valid(weekly,"F1-core-ridge")
    evidence = reporting._week_ci_evidence(weekly,None,"F1-core-ridge")
    assert "유효 ISO 주 수" in evidence and "degenerate_draws" in evidence
    assert evidence.count("| 5 |") == 12
    assert not reporting._week_ci_valid(weekly.assign(block="day"),"F1-core-ridge")


def test_final_rejects_unsealed_confirm_week_ci_before_numeric_read(tmp_path, monkeypatch):
    out = _out(tmp_path)
    monkeypatch.setattr(reporting,"_confirmed_lock",lambda out,registry: (
        {"primary_candidate":"F1-core-ridge"},
        {"tables":{},"confirm_once_sha256":"confirmed"}))
    monkeypatch.setattr(reporting,"_coverage_gate",lambda out,done,**kwargs:
                        (True,"verified"))
    monkeypatch.setattr(reporting,"_weekly_evidence",lambda out,lock,done,registry:
                        ("verified",True))
    monkeypatch.setattr(pd,"read_parquet",lambda *args,**kwargs: (_ for _ in ()).throw(
        AssertionError("No CONFIRM prediction or numeric parquet read")))
    result = reporting.generate_final(tmp_path)
    assert result["reason"] == "sealed_confirm_week_ci_absent"
    assert result["confirm_read"] is False
    assert not (out / "PHASE_F_REPORT.md").exists()
