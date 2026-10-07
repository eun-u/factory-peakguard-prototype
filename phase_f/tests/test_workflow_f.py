"""CPU-only workflow boundary and restart tests; no plant model is fitted."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import pandas as pd

from phase_f import workflow
from phase_f.registry import config_hash, sha256, write_json


def _prepared(tmp_path):
    out = tmp_path / "outputs" / "phase_f"
    (out / "logs").mkdir(parents=True)
    return SimpleNamespace(root=tmp_path, out=out)


def test_saved_wave_is_authoritative_before_adaptive_factory(tmp_path):
    prepared = _prepared(tmp_path)
    calls = []

    def factory():
        calls.append(1)
        return [{"id": "F1-example", "family": "F1", "adapter": "regression"}]

    first = workflow._locked_specs(prepared, "example", factory, "frozen hypothesis")
    second = workflow._locked_specs(prepared, "example", lambda: pytest.fail("parent recomputed"),
                                    "changed hypothesis")
    assert first == second and calls == [1]
    path = prepared.out / "logs" / "waves" / "example.json"
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["specs"][0]["family"] = "F2"
    write_json(path, altered)
    with pytest.raises(ValueError, match="invalid wave lock"):
        workflow._locked_specs(prepared, "example", factory, "frozen hypothesis")


def test_three_consecutive_failed_experiments_stop_wave(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    rows = {}

    class FakeRegistry:
        def __init__(self, root):
            pass

        def register(self, spec):
            return rows.setdefault(spec["id"], {"status": "planned"})

        def read(self, exp_id):
            return rows[exp_id]

    import phase_f.run as run
    monkeypatch.setattr(workflow, "Registry", FakeRegistry)

    def fail(prepared, spec, registry, retry=False):
        rows[spec["id"]] = {"status": "failed"}
        return rows[spec["id"]]

    monkeypatch.setattr(run, "execute", fail)
    specs = [{"id": f"F3-{n}", "family": "F3"} for n in range(4)]
    with pytest.raises(RuntimeError, match="Three consecutive"):
        workflow._execute_wave(prepared, "failures", lambda: specs, "failure audit")
    assert rows["F3-3"]["status"] == "planned"
    checkpoint = json.loads((prepared.out / "logs" / "workflow" / "failures.json").read_text())
    assert checkpoint["resolved"] == 3 and checkpoint["status"] == "running"


def test_stage_exception_keeps_full_completion_false(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(workflow, "_stage0", lambda *args: (_ for _ in ()).throw(RuntimeError("fit failed")))
    with pytest.raises(RuntimeError, match="fit failed"):
        workflow.run_stage(prepared, 0)
    checkpoint = json.loads((prepared.out / "logs" / "workflow" / "stage_0.json").read_text())
    driver = json.loads((prepared.out / "logs" / "driver_status.json").read_text())
    assert checkpoint["status"] == "failed" and checkpoint["full_phase_complete"] is False
    assert driver["full_phase_complete"] is False and driver["holdout_read"] is False


def test_completed_stage_is_not_reexecuted_on_resume(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    from phase_f import harness
    write_json(prepared.out / "logs" / "workflow" / "stage_0.json",
               {"stage": 0, "status": "completed", "full_phase_complete": False})
    monkeypatch.setattr(harness, "seal_parent", lambda root: {"verified": True})
    monkeypatch.setattr(workflow, "_stage0", lambda *args: pytest.fail("stage reran after checkpoint"))
    assert workflow.run_stage(prepared, 0)["status"] == "completed"


def test_search_manifest_requires_resolved_family_evidence(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    from phase_f import selection
    monkeypatch.setattr(selection, "complete_metric_manifests", lambda p: None)
    # Isolate F0-F10 wave resolution; real fine-budget proof has its own test.
    monkeypatch.setattr(selection, "foundation_training_coverage", lambda p: {
        "full": {"experiments": []}, "lora": {"experiments": []}})
    registry_dir = prepared.out / "logs" / "experiments"
    registry_dir.mkdir(parents=True)
    for n in range(11):
        spec = {"id": f"F{n}-evidence", "family": f"F{n}"}
        write_json(registry_dir / f"F{n}-evidence.json",
                   {"exp_id": f"F{n}-evidence", "family": f"F{n}",
                    "config_hash": config_hash(spec),
                    "status": "diagnostic_passed" if n in (0, 8, 10) else "completed"})
    wave = prepared.out / "logs" / "waves" / "required.json"
    tuning = prepared.out / "logs" / "tuning" / "gbdt_lightgbm.json"
    specs = [{"id": "F0-evidence", "family": "F0"}]
    write_json(wave, {"name": "required", "specs": specs, "config_sha256": config_hash(specs)})
    write_json(tuning, {"completed_trials": 500})
    write_json(prepared.out / "logs" / "workflow" / "expansion_rounds.json",
               {"rounds": [{"meaningful_improvement": False},
                           {"meaningful_improvement": False}]})
    record = workflow._search_manifest(prepared)
    assert set(record["required_family_coverage"]) == {f"F{n}" for n in range(11)}
    assert set(record["experiment_ids"]) == {f"F{n}-evidence" for n in range(11)}
    assert record["waves"]["logs/waves/required.json"] == sha256(wave)
    assert record["tuningfiles"]["logs/tuning/gbdt_lightgbm.json"] == sha256(tuning)
    assert set(record["foundation_training_coverage"]) == {"full", "lora"}
    assert record["holdout_read"] is False


def test_expansion_specs_includes_full_and_lora_training_budgets(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    from phase_f import experiment_plan as plan
    calls = []
    monkeypatch.setattr(plan, "completed", lambda root, *, adapter=None: [])
    monkeypatch.setattr(plan, "foundation_training_expansion",
                        lambda root, number: calls.append(number) or [
                            {"id": "F6-full-budget", "finetune": "full"},
                            {"id": "F6-lora-budget", "finetune": "lora"}])
    specs = workflow._expansion_specs(prepared, 2)
    assert calls == [2]
    assert [spec["id"] for spec in specs] == ["F6-full-budget", "F6-lora-budget"]


def test_two_finite_nonimproving_extension_rounds(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    import phase_f.tuning as tuning
    seen = []
    monkeypatch.setattr(tuning, "eligible_best", lambda root: 10.0)
    monkeypatch.setattr(workflow, "_execute_wave", lambda p, name, factory, hypothesis, retry=False: seen.append(name))
    monkeypatch.setattr(tuning, "run_search", lambda p, kind, **kw: seen.append(kw["label"]))
    rounds = workflow._expansion(prepared, retry=False)
    assert len(rounds) == 2 and [r["meaningful_improvement"] for r in rounds] == [False, False]
    assert seen == ["expansion_0_ablations", "expansion_0_lightgbm",
                    "expansion_0_xgboost", "expansion_0_catboost",
                    "expansion_1_ablations", "expansion_1_lightgbm",
                    "expansion_1_xgboost", "expansion_1_catboost"]


def test_confirm_is_after_local_lock_commit(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    from phase_f import selection, reporting
    events = []
    lock = {"lock_sha256": "selection", "primary_candidate": None, "candidates": []}
    done = {"confirm_once_sha256": "confirm"}
    monkeypatch.setattr(workflow, "_search_manifest", lambda p: events.append("search") or
                        {"required_family_coverage": {f"F{n}": {"status": "completed"} for n in range(11)}})
    monkeypatch.setattr(selection, "lock_selection", lambda p, **kw: events.append("lock") or lock)
    monkeypatch.setattr(workflow, "_git_commit", lambda p, message: events.append("commit"))
    monkeypatch.setattr(workflow, "_verify_committed_selection", lambda p, l: events.append("verify"))
    monkeypatch.setattr(selection, "confirm_once", lambda p: events.append("confirm") or done)
    monkeypatch.setattr(workflow, "_downstream", lambda p, l, d: {"status": "unavailable", "reason": "no primary"})
    def unavailable_weekly(p, l, d):
        write_json(p.out / "logs" / "weekly_diagnostic_status.json", {"status": "unavailable"})
        return {"status": "unavailable"}
    monkeypatch.setattr(workflow, "_weekly", unavailable_weekly)
    monkeypatch.setattr(reporting, "generate_progress", lambda *args, **kw: None)
    monkeypatch.setattr(reporting, "generate_final", lambda p, provisional=False:
                        {"report_ready": True, "full_phase_complete": False})
    write_json(prepared.out / "logs" / "search_complete.json",
               {"required_family_coverage": {f"F{n}": {"status": "completed"} for n in range(11)}})
    workflow._stage4(prepared)
    assert events == ["search", "lock", "commit", "verify", "confirm"]
    gate = json.loads((prepared.out / "logs" / "completion_gate.json").read_text())
    assert gate["required_family_coverage"]["F11"]["status"] == "unsupported_explicit"
    assert gate["ready_for_report"] is True and gate["full_phase_complete"] is False
    assert gate["holdout_read"] is False


def test_reserved_confirm_resume_never_recommits_partial_confirm(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    from phase_f import selection, reporting
    events = []
    lock = {"lock_sha256": "frozen", "primary_candidate": None, "candidates": []}
    done = {"confirm_once_sha256": "done"}
    write_json(prepared.out / "logs" / "confirm_reserved.json", {"lock_sha256": "frozen"})
    write_json(prepared.out / "logs" / "search_complete.json",
               {"required_family_coverage": {f"F{i}": {"status": "completed"} for i in range(11)}})
    monkeypatch.setattr(workflow, "_search_manifest", lambda p: events.append("search"))
    monkeypatch.setattr(selection, "lock_selection", lambda p, **kw: events.append("lock") or lock)
    monkeypatch.setattr(workflow, "_git_commit", lambda *a: pytest.fail("partial CONFIRM was recommitted"))
    monkeypatch.setattr(workflow, "_verify_committed_selection", lambda p, l: events.append("verify"))
    monkeypatch.setattr(selection, "confirm_once", lambda p: events.append("confirm") or done)
    monkeypatch.setattr(workflow, "_downstream", lambda *a: {"status": "unavailable", "reason": "no primary"})
    def unavailable_weekly(p, l, d):
        write_json(p.out / "logs" / "weekly_diagnostic_status.json", {"status": "unavailable"})
        return {"status": "unavailable"}
    monkeypatch.setattr(workflow, "_weekly", unavailable_weekly)
    monkeypatch.setattr(reporting, "generate_progress", lambda *a, **kw: None)
    monkeypatch.setattr(reporting, "generate_final", lambda p, provisional=False:
                        {"report_ready": True, "full_phase_complete": False})
    workflow._stage4(prepared)
    assert events == ["search", "lock", "verify", "confirm"]


def test_git_commit_does_not_traverse_optional_environment(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    (prepared.root / "phase_f").mkdir()
    (prepared.root / "phase_f" / "workflow.py").write_text("pass", encoding="utf-8")
    package = prepared.out / "optional_envs" / "numba" / "site" / "package.json"
    package.parent.mkdir(parents=True)
    package.write_text("{}", encoding="utf-8")
    good = prepared.out / "logs" / "stage.json"
    write_json(good, {"ok": True})
    commands = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda args, **kw: commands.append(args))
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda args, **kw: b"")
    workflow._git_commit(prepared, "synthetic")
    staged = [str(item) for command in commands for item in command]
    assert any("stage.json" in item for item in staged)
    assert not any("optional_envs" in item or "package.json" in item for item in staged)


def test_weekly_resume_verifies_confirm_lock_frozen_config_and_artifacts(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    candidate = "F1-core-ridge"
    spec = {"id": candidate, "kind": "ridge", "target": "direct"}
    row = {"status": "completed", "config_hash": config_hash(spec),
           "config_json": json.dumps(spec)}

    class FakeRegistry:
        def __init__(self, root): pass
        def read(self, exp_id): return row

    prepared.origins = lambda h, f, role: pd.DatetimeIndex([pd.Timestamp("2021-07-01") + pd.Timedelta(days=f)])
    monkeypatch.setattr(workflow, "Registry", FakeRegistry)
    selection = {"lock_sha256": "selection", "weekly_diagnostic_candidate": candidate,
                 "weekly_diagnostic_config_hash": config_hash(spec)}
    confirm = {"confirm_once_sha256": "confirm"}
    _, _, _, fixed = workflow._weekly_fixed(prepared, selection, confirm, candidate)
    write_json(prepared.out / "logs" / "weekly_diagnostic_lock.json", fixed)
    artifacts = {}
    for fold in range(3):
        for name, suffix, folder in (("predictions", "parquet", "predictions"),
                                      ("updates", "csv", "tables")):
            path = prepared.out / folder / f"F9-4-fold{fold}-{name}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"immutable synthetic artifact")
            artifacts[path.relative_to(prepared.out).as_posix()] = sha256(path)
    status_path = prepared.out / "logs" / "weekly_diagnostic_status.json"
    write_json(status_path, {"status": "completed", "candidate": candidate, "artifacts": artifacts,
                             "selection_lock_sha256": "selection", "confirm_once_sha256": "confirm"})
    assert workflow._weekly(prepared, selection, confirm)["status"] == "completed"
    with pytest.raises(ValueError, match="CONFIRM lock"):
        workflow._weekly(prepared, selection, {"confirm_once_sha256": "changed"})
    row["config_hash"] = "changed"
    with pytest.raises(ValueError, match="frozen registry"):
        workflow._weekly(prepared, selection, confirm)
    row["config_hash"] = config_hash(spec)
    target = prepared.out / next(iter(artifacts))
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        workflow._weekly(prepared, selection, confirm)


def test_f11_resume_rejects_artifact_path_escape_before_read(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    required = [f"tables/downstream/{name}.csv" for name in
                ("risk_metrics", "uncertainty_metrics", "alert_episode_metrics",
                 "comparison_vs_B5", "decision_value_curve")]
    required.append("predictions/F11_score_probabilities.parquet")
    record = {"status": "completed", "selection_lock_sha256": "selection",
              "confirm_once_sha256": "confirm",
              "artifacts": {"tables/downstream/..\\..\\..\\final.csv": "bad",
                            **{name: "hash" for name in required}}}
    write_json(prepared.out / "logs" / "downstream_status.json", record)
    monkeypatch.setattr(workflow, "sha256", lambda path: pytest.fail("artifact escaped before validation"))
    with pytest.raises(ValueError, match="outside expected output"):
        workflow._downstream(prepared, {"lock_sha256": "selection", "candidates": ["candidate"]},
                             {"confirm_once_sha256": "confirm"})


def test_stage4_final_report_failure_rolls_back_completion_claim(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    from phase_f import harness, reporting, selection
    for stage in range(4):
        write_json(prepared.out / "logs" / "workflow" / f"stage_{stage}.json",
                   {"stage": stage, "status": "completed"})
    events = []

    def provisional(p):
        write_json(p.out / "logs" / "completion_gate.json",
                   {"ready_for_report": True, "full_phase_complete": False,
                    "holdout_read": False})
        return {"report_ready": True, "full_phase_complete": False}

    monkeypatch.setattr(workflow, "_stage4", provisional)
    monkeypatch.setattr(selection, "complete_metric_manifests", lambda p: events.append("metrics"))
    monkeypatch.setattr(harness, "seal_parent", lambda root: events.append("seal"))
    monkeypatch.setattr(workflow, "_git_commit", lambda p, message: events.append("commit"))
    monkeypatch.setattr(reporting, "generate_final", lambda p: (_ for _ in ()).throw(RuntimeError("report crashed")))
    with pytest.raises(RuntimeError, match="report crashed"):
        workflow.run_stage(prepared, 4)
    gate = json.loads((prepared.out / "logs" / "completion_gate.json").read_text())
    driver = json.loads((prepared.out / "logs" / "driver_status.json").read_text())
    assert events == ["metrics", "seal", "commit"]
    assert gate["full_phase_complete"] is False and driver["full_phase_complete"] is False
    assert json.loads((prepared.out / "logs" / "workflow" / "stage_4.json").read_text())["status"] == "failed"
