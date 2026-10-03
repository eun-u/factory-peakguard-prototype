"""Metadata-only checks of the locked Chronos fine-tuning search plan."""
import json

import pytest

from phase_f.experiment_plan import foundation_fine, foundation_training_expansion
from phase_f.models.foundation import configurations


def _record(root, config, metric=None, *, status="completed"):
    path = root / "outputs" / "phase_f" / "logs" / "experiments" / f"{config['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"exp_id": config["id"], "config_json": json.dumps(config), "status": status}
    if metric is not None:
        row["explore_AUC_MAE"] = metric
    path.write_text(json.dumps(row), encoding="utf-8")


def _training_key(spec):
    return (spec["finetune"], spec["context_length"], spec["learning_rate"], spec["num_steps"])


def test_stage2_adds_budget_and_lr_grid_without_changing_original_twelve():
    old = [{**spec, "adapter": "foundation"} for spec in configurations() if spec["tier"] == 2]
    assert len(old) == 12
    plan = foundation_fine(2)
    assert plan[:12] == old
    assert len(plan) == len({spec["id"] for spec in plan}) == 72
    expected = {(mode, context, lr, steps)
                for mode in ("full", "lora") for context in (512, 2048, 8192)
                for lr in (1e-6, 1e-5, 3e-6)
                for steps in (100, 250, 500, 1000)}
    assert {_training_key(spec) for spec in plan} == expected
    assert all("-s" in spec["id"] and spec["parent"] in {old_spec["id"] for old_spec in old}
               for spec in plan[12:])


def test_stage3_calendar_grid_does_not_multiply_by_new_training_budgets():
    stage3 = foundation_fine(3)
    old_ids = {spec["id"] for spec in foundation_fine(2)[:12]}
    assert len(stage3) == 36
    assert {spec["parent"] for spec in stage3} == old_ids
    assert all(spec["covariates"] is True for spec in stage3)


def test_expansion_uses_only_completed_explore_and_advances_to_new_radius(tmp_path):
    old = foundation_fine(2)[:12]
    full = next(spec for spec in old if spec["finetune"] == "full" and spec["context_length"] == 512)
    lora = next(spec for spec in old if spec["finetune"] == "lora" and spec["context_length"] == 2048)
    _record(tmp_path, full, 10.0)
    _record(tmp_path, lora, 12.0)
    misleading = {**full, "id": "F6-confirm-only", "context_length": 2048,
                  "learning_rate": 9e-8}
    _record(tmp_path, misleading)  # A non-EXPLORE result cannot select a parent.

    first = foundation_training_expansion(tmp_path, 0)
    assert first and {spec["finetune"] for spec in first} == {"full", "lora"}
    assert {spec["parent"] for spec in first} == {full["id"], lora["id"]}
    assert len({_training_key(spec) for spec in first}) == len(first)
    assert ("full", 1024, full["learning_rate"], full["num_steps"]) in {
        _training_key(spec) for spec in first}
    assert {spec["context_length"] for spec in first if spec["finetune"] == "lora"
            and "-context-" in spec["id"]} == {1024, 4096}

    for spec in first:
        _record(tmp_path, spec, 100.0)
    second = foundation_training_expansion(tmp_path, 1)
    assert second and {spec["finetune"] for spec in second} == {"full", "lora"}
    assert not {_training_key(spec) for spec in first} & {_training_key(spec) for spec in second}
    assert {spec["parent"] for spec in second} == {full["id"], lora["id"]}


def test_expansion_requires_completed_explore_parent_for_each_finetune_mode(tmp_path):
    full = next(spec for spec in foundation_fine(2) if spec["finetune"] == "full")
    _record(tmp_path, full, 10.0)
    with pytest.raises(RuntimeError, match="No completed EXPLORE lora"):
        foundation_training_expansion(tmp_path, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        foundation_training_expansion(tmp_path, -1)
