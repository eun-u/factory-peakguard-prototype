"""Small contract tests for the week-fold search driver; no plant fits."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from phase_f.registry import config_hash, write_json
from phase_f import wf_workflow as workflow


def prepared(tmp_path):
    root = tmp_path / "repo"
    out = root / "outputs/phase_f/walkforward_v2"
    out.mkdir(parents=True)
    return SimpleNamespace(root=root, out=out)


def test_original_inventory_excludes_only_diagnostics_and_duplicates(tmp_path):
    p = prepared(tmp_path)
    source = p.root / "outputs/phase_f/logs/experiments"
    source.mkdir(parents=True)
    entries = [("F0-1-B5", "completed", None, "baseline"),
               ("F0-1-M1", "completed", None, "baseline"),
               ("F0-1-R1", "completed", None, "baseline"),
               ("F2-5", "failed", None, "regression"),
               ("F2-5-native_mean", "completed", "F2-5", "regression"),
               ("D1", "diagnostic_passed", None, "diagnostic")]
    for exp_id, status, duplicate, adapter in entries:
        spec = {"id": exp_id, "adapter": adapter, "tier": 1}
        write_json(source / f"{exp_id}.json", {
            "exp_id": exp_id, "config_json": json.dumps(spec, sort_keys=True),
            "config_hash": config_hash(spec), "status": status,
            "duplicate_of": duplicate})
    specs, audit = workflow._source_original_specs(p)
    assert {s["id"] for s in specs} == {"F0-1-B5", "F0-1-M1", "F0-1-R1", "F2-5"}
    assert audit["previous_failures"] == ["F2-5"]
    assert audit["unique_model_configs"] == 4
    assert {item["reason"] for item in audit["excluded"]} == {"diagnostic", "duplicate_of"}


def test_locked_wave_reuses_exact_settings_after_parent_changes(tmp_path, monkeypatch):
    p = prepared(tmp_path)
    from phase_f import wf_run
    monkeypatch.setattr(wf_run, "normalize_spec", lambda prepared, spec: spec)
    first = workflow._locked_wave(p, "one", lambda: [{"id": "A", "value": 1}], "fixed")
    second = workflow._locked_wave(p, "one", lambda: pytest.fail("factory reran"), "fixed")
    assert first == second
    path = p.out / "logs/waves/one.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["specs"][0]["value"] = 2
    write_json(path, record)
    with pytest.raises(ValueError, match="Changed weekly wave lock"):
        workflow._locked_wave(p, "one", lambda: [], "fixed")


def test_direction_gate_requires_relative_gain_CI_and_seed_sd():
    old = {"exp_id": "old", "value": 10., "seed_sd": .1, "n_seeds": 5}
    new = {"exp_id": "new", "value": 9.7, "seed_sd": .1, "n_seeds": 5}
    good = workflow.direction_gain(old, new, ci_low=.05, ci_high=.55, ci_status="available")
    assert good["significant_improvement"] is True
    assert workflow.direction_gain(old, new, ci_low=-.01, ci_high=.55,
                                   ci_status="available")["significant_improvement"] is False
    assert workflow.direction_gain(old, {**new, "seed_sd": None}, ci_low=.05,
                                   ci_high=.55, ci_status="available")["reason"] == "missing_seed_standard_deviation"
    assert workflow.direction_gain(old, {**new, "value": 9.85}, ci_low=.05,
                                   ci_high=.55, ci_status="available")["significant_improvement"] is False


@pytest.mark.parametrize("direction,metric", [
    ("AUC_PeakMAE", "AUC_PeakMAE_degradation"),
    ("h16_PeakMAE", "Peak_MAE_degradation")])
def test_peak_degradation_ci_is_reversed_for_improvement(monkeypatch, direction, metric):
    from phase_f import wf_metrics, wf_evaluation
    monkeypatch.setattr(wf_evaluation, "load_predictions", lambda *args: pd.DataFrame())
    monkeypatch.setattr(wf_metrics, "paired_ci", lambda *args, **kwargs: pd.DataFrame([{
        "dataset": "D1", "horizon": 16 if direction == "h16_PeakMAE" else None,
        "metric": metric, "estimate": -.4, "ci_low": -.6, "ci_high": -.2,
        "ci_status": "available", "ci_reason": "", "bootstrap_n": 1000}]))
    old = {"exp_id": "old", "value": 10., "seed_sd": .01, "n_seeds": 5}
    new = {"exp_id": "new", "value": 9.6, "seed_sd": .01, "n_seeds": 5}
    result = workflow._direction_evidence(SimpleNamespace(), direction, old, new)
    assert result["ci_low"] == pytest.approx(.2)
    assert result["ci_high"] == pytest.approx(.6)
    assert result["significant_improvement"] is True


def test_gbdt_extension_uses_locked_parent(tmp_path, monkeypatch):
    p = prepared(tmp_path)
    from phase_f import wf_plan, wf_tuning
    path = p.out / "logs/tuning/stage2_lightgbm.json"
    write_json(path, {"parent_spec": {"id": "fixed", "adapter": "regression",
                                      "kind": "lightgbm"}})
    monkeypatch.setattr(wf_plan, "best", lambda *args, **kwargs: pytest.fail("parent reselected"))
    captured = {}

    def fake_search(prepared, kind, callback, *, parent_spec, target_completed, label):
        captured.update(parent=parent_spec, target=target_completed, label=label)
        return {"completed_trials": target_completed, "best_exp_id": "winner"}

    monkeypatch.setattr(wf_tuning, "run_search", fake_search)
    monkeypatch.setattr(workflow, "_score_tuned_winner", lambda *args, **kwargs: None)
    workflow._tune_gbdt(p, "lightgbm", 600, label="stage2_lightgbm", retry=False)
    assert captured == {"parent": {"id": "fixed", "adapter": "regression",
                                     "kind": "lightgbm"}, "target": 600,
                        "label": "stage2_lightgbm"}


def test_expansion_resume_uses_locked_before_snapshot(tmp_path, monkeypatch):
    p = prepared(tmp_path)
    from phase_f import wf_contract
    monkeypatch.setattr(wf_contract, "two_rounds_converged", lambda rounds: bool(rounds))
    before = {name: {"exp_id": "old", "value": 10., "seed_sd": 0., "n_seeds": 1}
              for name in workflow._DIRECTIONS}
    after = {name: {"exp_id": "new", "value": 9., "seed_sd": 0., "n_seeds": 1}
             for name in workflow._DIRECTIONS}
    lock = p.out / "logs/workflow/expansion_round_0.json"
    write_json(lock, {"round": 0, "before": before, "arm": "EXPLORE"})
    monkeypatch.setattr(workflow, "_direction_snapshot", lambda prepared: after)
    monkeypatch.setattr(workflow, "_execute_wave", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "_tune_gbdt", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "_tune_neural_families", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "_direction_evidence", lambda prepared, direction, old, new: {
        "significant_improvement": False, "before_id": old["exp_id"]})
    rounds = workflow._expansion(p, retry=False)
    assert rounds[0]["before"] == before
    assert {v["before_id"] for v in rounds[0]["direction_evidence"].values()} == {"old"}


def test_stage0_predicts_all_three_before_any_score(tmp_path, monkeypatch):
    p = prepared(tmp_path)
    specs = [{"id": name} for name in workflow.BASELINE_IDS]
    audit = {"config_sha256": {name: name for name in workflow.BASELINE_IDS},
             "unique_model_configs": 128}
    monkeypatch.setattr(workflow, "_original_spec_waves", lambda prepared: (specs, [], audit))
    monkeypatch.setattr(workflow, "_locked_wave", lambda prepared, name, factory,
                        hypothesis, **kwargs: factory())
    class Registry:
        def __init__(self, root):
            self.root = root

        def register(self, spec):
            return spec

    monkeypatch.setattr(workflow, "WFRegistry", Registry)
    from phase_f import wf_run, wf_evaluation
    observed = []
    def fake_execute(prepared, spec, *, registry, retry, score):
        observed.append((spec["id"], score))
        return {"status": "completed" if score else "prediction_ready"}
    monkeypatch.setattr(wf_run, "execute", fake_execute)
    monkeypatch.setattr(wf_evaluation, "refresh_downstream", lambda *args, **kwargs: None)
    workflow._stage0(p, retry=False)
    assert observed == [(name, False) for name in workflow.BASELINE_IDS] + [
        (name, True) for name in workflow.BASELINE_IDS]


def test_original_failed_setting_gets_one_new_namespace_retry(tmp_path, monkeypatch):
    p = prepared(tmp_path)
    spec = {"id": "F2-5", "adapter": "regression"}
    monkeypatch.setattr(workflow, "_locked_wave", lambda *args, **kwargs: [spec])
    monkeypatch.setattr(workflow, "_top20", lambda prepared: ())
    class Registry:
        row = {"status": "planned"}
        def __init__(self, root):
            self.root = root
        def register(self, spec):
            return self.row
        def read(self, exp_id):
            return self.row
    monkeypatch.setattr(workflow, "WFRegistry", Registry)
    from phase_f import wf_run
    observed = []
    def fake_execute(prepared, spec, *, registry, retry, score):
        observed.append(retry)
        registry.row = {"status": "completed" if retry else "failed"}
        return registry.row
    monkeypatch.setattr(wf_run, "execute", fake_execute)
    result = workflow._execute_wave(p, "replay", lambda: [spec], "test",
                                    retry_original_failures={"F2-5"})
    assert observed == [False, True]
    assert result["attempts"] == {"F2-5": 2}
    assert result["status"] == "completed"
