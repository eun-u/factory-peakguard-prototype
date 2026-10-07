"""Revised finalist gates use only locked EXPLORE registry evidence."""

from __future__ import annotations

import copy

import pytest

from phase_f.wf_selection import eligibility, select_finalists


def _row(exp_id="candidate", family="regression", mae=7.0):
    return {
        "exp_id": exp_id, "family": family, "status": "completed",
        "leakage_test": "passed", "ranking_basis": "metric of seed-mean predictions",
        "wf_explore_AUC_MAE": mae, "wf_explore_AUC_PeakMAE": 14.0,
        "D2_wf_explore_AUC_MAE": 8.0,
        "ciLow_vs_B5": .2,
        "wf_peak_degradation_ci_high_vs_B5": 0.0,
        "wf_D2_mae_improvement_vs_B5": 0.0,
        "wf_weeks_won_vs_B5": 6, "wf_explore_weeks": 8,
        "E_status": "completed", "E_c10_22_recall": .5,
        "E_B5_c10_22_recall": .5, "E_c10_22_F1": .4,
    }


@pytest.mark.parametrize(("field", "value", "failed_check"), [
    ("ciLow_vs_B5", 0.0, "b5_mae_ci_low_positive"),
    ("wf_peak_degradation_ci_high_vs_B5", 1e-8, "b5_peak_ci_high_nonpositive"),
    ("wf_D2_mae_improvement_vs_B5", -1e-8, "d2_mae_not_reversed"),
    ("wf_weeks_won_vs_B5", 5, "two_thirds_weeks_won"),
    ("E_c10_22_recall", .49, "e_episode_recall_nonworse"),
    ("E_status", "unavailable", "e_episode_recall_nonworse"),
    ("leakage_test", "failed", "leakage_passed"),
    ("ranking_basis", "best single seed", "seed_mean_ranked"),
])
def test_revised_gate_failures_are_not_met(field, value, failed_check):
    row = _row()
    row[field] = value
    result = eligibility(row, expected_weeks=8)
    assert not result["eligible"]
    assert result["checks"][failed_check]["status"] == "NOT_MET"
    assert failed_check in result["not_met"]


@pytest.mark.parametrize(("field", "failed_check"), [
    ("ciLow_vs_B5", "b5_mae_ci_low_positive"),
    ("wf_peak_degradation_ci_high_vs_B5", "b5_peak_ci_high_nonpositive"),
    ("wf_D2_mae_improvement_vs_B5", "d2_mae_not_reversed"),
    ("E_B5_c10_22_recall", "e_episode_recall_nonworse"),
])
def test_missing_or_undefined_evidence_cannot_pass(field, failed_check):
    row = _row()
    row[field] = None
    result = eligibility(row, expected_weeks=8)
    assert not result["eligible"]
    assert failed_check in result["missing_evidence"]
    row[field] = float("nan")
    assert failed_check in eligibility(row, expected_weeks=8)["missing_evidence"]


def test_boundary_values_and_locked_week_coverage():
    row = _row()
    assert eligibility(row, expected_weeks=8)["eligible"]
    assert eligibility(row, expected_weeks=8)["minimum_weeks_won"] == 6
    assert not eligibility(row, expected_weeks=9)["eligible"]
    row["wf_explore_weeks"] = 0
    assert not eligibility(row)["eligible"]


def test_mae_family_diversity_and_two_specialists_max_seven():
    rows = [
        _row("A", "f1", 1.0), _row("B", "f1", 1.1),
        _row("C", "f2", 1.2), _row("D", "f3", 1.3),
        _row("E", "f4", 1.4), _row("F", "f5", 1.5),
        _row("G", "f6", 1.6), _row("H", "f7", 1.7),
    ]
    rows[-2]["wf_explore_AUC_PeakMAE"] = 10.0
    rows[-1]["E_c10_22_F1"] = .99
    result = select_finalists(rows, expected_weeks=8)
    assert result["primary_mae_finalists"] == ["A", "C", "D", "E", "F"]
    assert result["specialists"] == {"AUC_PeakMAE": "G", "E_c10_22_F1": "H"}
    assert result["candidates"] == ["A", "C", "D", "E", "F", "G", "H"]
    assert result["primary_candidate"] == "A"
    assert result["confirm_metric_read"] is False


def test_exclude_baseline_duplicate_oracle_and_no_eligible_claim():
    base = _row("F0-1-B5", "baseline", 0.0)
    duplicate = _row("duplicate", "f", 1.0)
    duplicate["duplicate_of"] = "origin"
    oracle = _row("F0-4-oracle", "oracle", 0.0)
    failed = _row("failed", "f", 2.0)
    failed["E_status"] = "unavailable"
    result = select_finalists([base, duplicate, oracle, failed], expected_weeks=8)
    assert result["candidates"] == []
    assert result["primary_candidate"] is None
    assert result["eligible_count"] == 0
    assert set(result["assessments"]) == {"F0-1-B5", "duplicate", "F0-4-oracle", "failed"}


def test_confirm_fields_cannot_change_preconfirm_ranking():
    rows = [_row("A", "a", 1.0), _row("B", "b", 2.0)]
    original = select_finalists(rows, expected_weeks=8)
    changed = copy.deepcopy(rows)
    changed[0]["confirm_AUC_MAE"] = 1000.0
    changed[1]["confirm_AUC_MAE"] = -1000.0
    later = select_finalists(changed, expected_weeks=8)
    assert later["candidates"] == original["candidates"]
    assert later["primary_candidate"] == original["primary_candidate"]


def test_reject_duplicate_or_empty_registry_ids():
    with pytest.raises(ValueError, match="unique nonempty"):
        select_finalists([_row("A"), _row("A")], expected_weeks=8)
    with pytest.raises(ValueError, match="unique nonempty"):
        select_finalists([_row("")], expected_weeks=8)
