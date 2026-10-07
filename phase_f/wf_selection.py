"""Pure Stage-4 finalist selection from sealed weekly EXPLORE registry rows.

All evidence is supplied by the caller. This module never reads score files,
CONFIRM results, or a model checkpoint. A missing or undefined gate is NOT_MET.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


BASELINE_IDS = frozenset({"F0-1-B5", "F0-1-M1", "F0-1-R1", "B5", "M1", "R1"})
MAE_CI_LOW = "ciLow_vs_B5"  # B5 minus candidate; strictly positive.
PEAK_CI_HIGH = "wf_peak_degradation_ci_high_vs_B5"  # candidate minus B5; <= 0.
D2_MAE_GAIN = "wf_D2_mae_improvement_vs_B5"  # B5 minus candidate; >= 0.
E_RECALL = "E_c10_22_recall"
E_B5_RECALL = "E_B5_c10_22_recall"
E_F1 = "E_c10_22_F1"


def _finite(row: Mapping, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check(value: bool, *, evidence: str, reason: str, missing: bool = False) -> dict:
    return {"status": "MET" if value else "NOT_MET", "evidence": evidence,
            "reason": "" if value else reason, "missing_evidence": bool(not value and missing)}


def eligibility(row: Mapping, *, expected_weeks: int | None = None) -> dict:
    """Evaluate every preregistered EXPLORE gate without optimistic defaults.

    `expected_weeks` should be the number of even ISO score weeks in the
    immutable walk-forward lock. The weekly evaluator emits the CI and D2
    fields using the signed conventions documented in ``phase_f.wf_metrics``.
    """
    exp_id = str(row.get("exp_id", ""))
    family = row.get("family")
    mae = _finite(row, "wf_explore_AUC_MAE")
    mae_low = _finite(row, MAE_CI_LOW)
    peak_high = _finite(row, PEAK_CI_HIGH)
    d2_gain = _finite(row, D2_MAE_GAIN)
    wins = _finite(row, "wf_weeks_won_vs_B5")
    weeks = _finite(row, "wf_explore_weeks")
    recall = _finite(row, E_RECALL)
    b5_recall = _finite(row, E_B5_RECALL)
    minimum = math.ceil(2 * weeks / 3) if weeks is not None and weeks > 0 else None

    checks = {
        "candidate_identity": _check(
            bool(exp_id and isinstance(family, str) and family.strip()
                 and exp_id not in BASELINE_IDS and not row.get("duplicate_of")
                 and not exp_id.startswith("F0-4") and family.lower() != "oracle"),
            evidence="exp_id/family/duplicate_of",
            reason="Missing family, baseline, oracle, or duplicate configuration",
            missing=not exp_id or not family),
        "completed": _check(row.get("status") == "completed", evidence="status",
                            reason="Weekly experiment is not completed",
                            missing="status" not in row),
        "leakage_passed": _check(row.get("leakage_test") in {"passed", "parent_sealed_contract"},
                                 evidence="leakage_test", reason="Causal leakage check is absent or failed",
                                 missing="leakage_test" not in row),
        "seed_mean_ranked": _check(row.get("ranking_basis") == "metric of seed-mean predictions"
                                   and mae is not None,
                                   evidence="ranking_basis/wf_explore_AUC_MAE",
                                   reason="Finite EXPLORE MAE of seed-mean predictions is absent",
                                   missing="ranking_basis" not in row or mae is None),
        "b5_mae_ci_low_positive": _check(mae_low is not None and mae_low > 0,
                                          evidence=MAE_CI_LOW,
                                          reason="B5-minus-candidate AUC-MAE 95% CI lower bound is not positive",
                                          missing=mae_low is None),
        "b5_peak_ci_high_nonpositive": _check(peak_high is not None and peak_high <= 0,
                                               evidence=PEAK_CI_HIGH,
                                               reason="Candidate-minus-B5 AUC-PeakMAE 95% CI upper bound exceeds zero",
                                               missing=peak_high is None),
        "d2_mae_not_reversed": _check(d2_gain is not None and d2_gain >= 0,
                                      evidence=D2_MAE_GAIN,
                                      reason="D2 B5-minus-candidate AUC-MAE improvement is negative",
                                      missing=d2_gain is None),
        "two_thirds_weeks_won": _check(
            wins is not None and weeks is not None and weeks > 0
            and wins.is_integer() and weeks.is_integer() and 0 <= wins <= weeks
            and wins >= minimum and (expected_weeks is None or weeks == expected_weeks),
            evidence="wf_weeks_won_vs_B5/wf_explore_weeks",
            reason="Fewer than two-thirds of locked EXPLORE weeks improve B5, or weeks are incomplete",
            missing=wins is None or weeks is None),
        "e_episode_recall_nonworse": _check(
            row.get("E_status") == "completed" and recall is not None
            and b5_recall is not None and recall >= b5_recall,
            evidence="E_status/E_c10_22_recall/E_B5_c10_22_recall",
            reason="Same-cohort Phase E c=.10, 2/2 episode recall is unavailable or below B5",
            missing=(row.get("E_status") != "completed" or recall is None or b5_recall is None)),
    }
    failures = [name for name, result in checks.items() if result["status"] == "NOT_MET"]
    missing = [name for name, result in checks.items() if result["missing_evidence"]]
    return {"exp_id": exp_id, "family": family, "arm": "EXPLORE",
            "eligible": not failures, "checks": checks,
            "not_met": failures, "missing_evidence": missing,
            "minimum_weeks_won": minimum,
            "mae_ci_low": mae_low, "peak_degradation_ci_high": peak_high,
            "d2_mae_improvement": d2_gain,
            "episode_recall": recall, "same_cohort_b5_episode_recall": b5_recall}


def select_finalists(rows: Iterable[Mapping], *, expected_weeks: int,
                     primary_limit: int = 5, family_limit: int = 2) -> dict:
    """Freeze five MAE finalists plus one peak and one alert specialist.

    Ties use experiment ID, making the same sealed registry snapshot select
    the same candidates. Family diversity takes precedence within the top five.
    A specialist is the single best eligible configuration for its metric; if
    it is already selected it is not replaced by a runner-up.
    """
    if expected_weeks < 1 or primary_limit < 1 or family_limit < 1:
        raise ValueError("Selection limits and locked week count must be positive")
    source = [dict(row) for row in rows]
    identifiers = [str(row.get("exp_id", "")) for row in source]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Registry rows require unique nonempty experiment IDs")
    assessments = {row["exp_id"]: eligibility(row, expected_weeks=expected_weeks)
                   for row in source}
    eligible = [row for row in source if assessments[row["exp_id"]]["eligible"]]
    ordered = sorted(eligible, key=lambda row: (_finite(row, "wf_explore_AUC_MAE"), row["exp_id"]))
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    for first_of_family in (True, False):
        for row in ordered:
            key = row["exp_id"]
            family = row["family"]
            count = family_counts.get(family, 0)
            if key in selected or count >= family_limit or (first_of_family and count):
                continue
            selected.append(key)
            family_counts[family] = count + 1
            if len(selected) == primary_limit:
                break
        if len(selected) == primary_limit:
            break
    primary = selected.copy()
    specialists = {}
    if eligible:
        peak_candidates = [row for row in eligible if _finite(row, "wf_explore_AUC_PeakMAE") is not None]
        if peak_candidates:
            peak = min(peak_candidates, key=lambda row: (
                _finite(row, "wf_explore_AUC_PeakMAE"),
                _finite(row, "wf_explore_AUC_MAE"), row["exp_id"]))["exp_id"]
            specialists["AUC_PeakMAE"] = peak
            if peak not in selected:
                selected.append(peak)
        episode_candidates = [row for row in eligible if _finite(row, E_F1) is not None]
        if episode_candidates:
            episode = min(episode_candidates, key=lambda row: (
                -_finite(row, E_F1), _finite(row, "wf_explore_AUC_MAE"), row["exp_id"]))["exp_id"]
            specialists["E_c10_22_F1"] = episode
            if episode not in selected:
                selected.append(episode)
    if len(selected) > primary_limit + 2:
        raise AssertionError("Finalist cap exceeded")
    return {"version": 2, "selection_arm": "EXPLORE", "expected_explore_weeks": expected_weeks,
            "primary_limit": primary_limit, "family_limit": family_limit,
            "primary_mae_finalists": primary, "specialists": specialists,
            "candidates": selected, "primary_candidate": primary[0] if primary else None,
            "eligible_count": len(eligible), "attempted_registry_rows": len(source),
            "assessments": assessments,
            "no_postconfirm_reselection": True,
            "confirm_metric_read": False, "holdout_read": False}
