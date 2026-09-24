"""Extend, without changing, the preregistered development selection order.

The original ``training._selection`` runs first and remains the authoritative
decision for its original candidate set.  Phase 2 candidates challenge only
their relevant point or risk choice on matched development score rows.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .bootstrap import paired_mae_improvement
from .evaluate import match_episodes, score_predictions


M1_NAME = "lgbm_residual_cbl"
M4_NAMES = ("q95_rolling_672", "q95_aci_005", "q95_aci_010", "q95_aci_050")
KEYS = ["target_time", "horizon", "fold"]


def _matched(incumbent: pd.DataFrame, challenger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match identical score positions before computing any candidate contrast."""
    if incumbent.duplicated(KEYS).any() or challenger.duplicated(KEYS).any():
        raise ValueError("Candidate comparison has duplicate score keys")
    left = incumbent.set_index(KEYS)
    right = challenger.set_index(KEYS)
    common = left.index.intersection(right.index)
    if common.empty:
        return incumbent.iloc[:0].copy(), challenger.iloc[:0].copy()
    a = left.loc[common].reset_index()
    b = right.loc[common].reset_index()
    if not np.allclose(pd.to_numeric(a.y, errors="coerce"), pd.to_numeric(b.y, errors="coerce"),
                       equal_nan=True, rtol=0, atol=1e-10):
        raise AssertionError("Candidate target differs from original development OOF")
    if not np.allclose(pd.to_numeric(a.tau, errors="coerce"), pd.to_numeric(b.tau, errors="coerce"),
                       equal_nan=True, rtol=0, atol=1e-10):
        raise AssertionError("Candidate peak threshold differs from original fold")
    valid = np.isfinite(pd.to_numeric(a.y, errors="coerce")) & np.isfinite(pd.to_numeric(a.pred, errors="coerce")) & np.isfinite(pd.to_numeric(b.pred, errors="coerce"))
    return a.loc[valid].reset_index(drop=True), b.loc[valid].reset_index(drop=True)


def _daily_alert_ci(frame: pd.DataFrame, *, n_boot: int, seed: int) -> dict:
    """Reproduce original per-fold/per-date episode and false-alarm CI rule."""
    daily = []
    part = frame.sort_values(["fold", "target_time"]).copy()
    part["day"] = pd.to_datetime(part.target_time).dt.normalize()
    for _, block in part.groupby(["fold", "day"], sort=True):
        actual = block.y.to_numpy(dtype=float) > block.tau.to_numpy(dtype=float)
        alert = block.alert.to_numpy(dtype=bool)
        matched = match_episodes(actual, alert, block.target_time)
        daily.append((matched["tp"], matched["fp"], matched["fn"], int((~actual & alert).sum())))
    if not daily:
        return {"episode_f1_ci95": [np.nan, np.nan],
                "false_alarm_positions_ci95": [np.nan, np.nan],
                "n_days": 0}
    values = np.asarray(daily, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(n_boot, len(values)))].sum(axis=1)
    denominator = 2 * sampled[:, 0] + sampled[:, 1] + sampled[:, 2]
    f1 = np.divide(2 * sampled[:, 0], denominator,
                   out=np.zeros(len(sampled)), where=denominator > 0)
    return {"episode_f1_ci95": list(map(float, np.quantile(f1, [.025, .975]))),
            "false_alarm_positions_ci95": list(map(float, np.quantile(sampled[:, 3], [.025, .975]))),
            "n_days": len(values)}


def compare_point_candidate(incumbent: pd.DataFrame, challenger: pd.DataFrame,
                            *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Apply original point ranking to a residual challenger on paired rows."""
    a, b = _matched(incumbent, challenger)
    if a.empty or b.empty:
        return {"selected": False, "eligible": False, "reason": "no_common_valid_score_rows",
                "n_common": 0}
    gain = paired_mae_improvement(a, b, peak_only=True, n=n_boot, seed=seed)
    fold_effects = []
    for fid in sorted(a.fold.unique()):
        left = a.loc[a.fold.eq(fid)]
        right = b.loc[b.fold.eq(fid)]
        peak = left.y.to_numpy(dtype=float) > left.tau.to_numpy(dtype=float)
        mae_a = float(np.mean(abs(left.y.to_numpy(dtype=float)[peak] - left.pred.to_numpy(dtype=float)[peak]))) if peak.any() else np.nan
        mae_b = float(np.mean(abs(left.y.to_numpy(dtype=float)[peak] - right.pred.to_numpy(dtype=float)[peak]))) if peak.any() else np.nan
        fold_effects.append({"fold": int(fid), "peak_mae_incumbent": mae_a,
                             "peak_mae_challenger": mae_b,
                             "gain": mae_a - mae_b if np.isfinite(mae_a) and np.isfinite(mae_b) else np.nan})
    original_folds = {int(fid) for fid in incumbent.fold.unique()}
    matched_folds = {int(fid) for fid in a.fold.unique()}
    last = max(original_folds)
    stability = next((row for row in fold_effects if row["fold"] == last), None)
    stable = bool(stability is not None and matched_folds == original_folds
                  and np.isfinite(stability["peak_mae_incumbent"])
                  and np.isfinite(stability["peak_mae_challenger"])
                  and stability["peak_mae_challenger"] < 1.10 * stability["peak_mae_incumbent"])
    base_ci = _daily_alert_ci(a, n_boot=n_boot, seed=seed)
    challenger_ci = _daily_alert_ci(b, n_boot=n_boot, seed=seed)
    evidence = {"eligible": bool(stable and np.isfinite(gain["estimate"])),
                "n_common": len(a), "n_peak_common": gain["n"],
                "paired_peak_mae_gain": gain,
                "incumbent_alert_ci": base_ci, "challenger_alert_ci": challenger_ci,
                "fold_effects": fold_effects, "n_positive_folds": sum(row["gain"] > 0 for row in fold_effects),
                "last_fold_stable": bool(stable),
                "all_original_folds_matched": matched_folds == original_folds,
                "complexity": {"incumbent_models": 1, "challenger_models": 2,
                               "challenger_extra_feature": "residual_baseline"}}
    if not stable:
        decision = "last_fold_peak_mae_ge_1p10_incumbent"
    elif not np.isfinite(gain["estimate"]):
        decision = "no_common_peak_rows"
    elif np.isfinite(gain["ci95"][0]) and gain["ci95"][0] > 0:
        decision = "challenger_peak_mae_ci"
    elif np.isfinite(gain["ci95"][1]) and gain["ci95"][1] < 0:
        decision = "incumbent_peak_mae_ci"
    elif challenger_ci["episode_f1_ci95"][0] > base_ci["episode_f1_ci95"][1]:
        decision = "challenger_episode_f1_ci"
    elif base_ci["episode_f1_ci95"][0] > challenger_ci["episode_f1_ci95"][1]:
        decision = "incumbent_episode_f1_ci"
    elif challenger_ci["false_alarm_positions_ci95"][1] < base_ci["false_alarm_positions_ci95"][0]:
        decision = "challenger_false_alarm_ci"
    elif base_ci["false_alarm_positions_ci95"][1] < challenger_ci["false_alarm_positions_ci95"][0]:
        decision = "incumbent_false_alarm_ci"
    else:
        # Neither point model has a comparable upper quantile; the original
        # incumbent is simpler and keeps priority when preceding CIs overlap.
        decision = "incumbent_simpler_after_ci_overlap"
    evidence["selected"] = decision.startswith("challenger")
    evidence["reason"] = decision
    return evidence


def _risk_gate(candidate: pd.DataFrame, original_b: pd.DataFrame,
               original_point: pd.DataFrame) -> dict:
    b, c = _matched(original_b, candidate)
    if b.empty or c.empty or not len(original_point):
        return {"eligible": False, "reason": "no_common_risk_or_point_rows"}
    b_score, c_score = score_predictions(b), score_predictions(c)
    last = int(max(c.fold))
    last_b = b.loc[b.fold.eq(last)]
    last_c = c.loc[c.fold.eq(last)]
    last_b_score = score_predictions(last_b)
    last_c_score = score_predictions(last_c)
    point_last = original_point.loc[original_point.fold.eq(last)]
    point_stable = bool(len(point_last) and np.isfinite(score_predictions(point_last).get("peak_mae", np.nan)))
    b_cov = float(b_score.get("top_coverage_0.95", np.nan))
    c_cov = float(c_score.get("top_coverage_0.95", np.nan))
    b_pin = float(b_score.get("pinball_mean", np.nan))
    c_pin = float(c_score.get("pinball_mean", np.nan))
    last_b_cov = float(last_b_score.get("top_coverage_0.95", np.nan))
    last_c_cov = float(last_c_score.get("top_coverage_0.95", np.nan))
    cover_error = abs(b_cov - .95) - abs(c_cov - .95)
    pinball_margin = 1.05 * b_pin - c_pin
    last_gain = last_c_cov - last_b_cov
    gates = {"top_coverage_error_smaller": np.isfinite(cover_error) and cover_error > 0,
             "pinball_within_5pct": np.isfinite(pinball_margin) and pinball_margin >= 0,
             "last_fold_top_coverage_higher": np.isfinite(last_gain) and last_gain > 0,
             "point_last_fold_stable": point_stable}
    passed = all(gates.values())
    return {"eligible": bool(passed), "reason": "passed" if passed else ";".join(
        name for name, ok in gates.items() if not ok),
            "gates": gates, "n_common": len(c), "n_last_fold": len(last_c),
            "top_coverage_b": b_cov, "top_coverage_candidate": c_cov,
            "top_coverage_error_gain": cover_error,
            "pinball_b": b_pin, "pinball_candidate": c_pin,
            "pinball_allowance_margin": pinball_margin,
            "last_fold_top_coverage_b": last_b_cov,
            "last_fold_top_coverage_candidate": last_c_cov,
            "last_fold_top_coverage_gain": last_gain}


def _risk_complexity(name: str) -> tuple[int, int]:
    if name == "lgbm_quantile_b":
        return (0, 0)
    if name == "q95_rolling_672":
        return (1, 0)
    return (2, M4_NAMES.index(name))


def _risk_challenge(incumbent: pd.DataFrame, challenger: pd.DataFrame,
                    *, n_boot: int, seed: int) -> dict:
    a, b = _matched(incumbent, challenger)
    if a.empty:
        return {"selected": False, "reason": "no_common_risk_rows"}
    left_ci = _daily_alert_ci(a, n_boot=n_boot, seed=seed)
    right_ci = _daily_alert_ci(b, n_boot=n_boot, seed=seed)
    left_cov = float(score_predictions(a).get("top_coverage_0.95", np.nan))
    right_cov = float(score_predictions(b).get("top_coverage_0.95", np.nan))
    left_error, right_error = abs(left_cov - .95), abs(right_cov - .95)
    if right_ci["episode_f1_ci95"][0] > left_ci["episode_f1_ci95"][1]:
        reason = "challenger_episode_f1_ci"
    elif left_ci["episode_f1_ci95"][0] > right_ci["episode_f1_ci95"][1]:
        reason = "incumbent_episode_f1_ci"
    elif right_ci["false_alarm_positions_ci95"][1] < left_ci["false_alarm_positions_ci95"][0]:
        reason = "challenger_false_alarm_ci"
    elif left_ci["false_alarm_positions_ci95"][1] < right_ci["false_alarm_positions_ci95"][0]:
        reason = "incumbent_false_alarm_ci"
    elif np.isfinite(right_error) and right_error < left_error - 1e-12:
        reason = "challenger_top_coverage_error"
    elif np.isfinite(left_error) and left_error < right_error - 1e-12:
        reason = "incumbent_top_coverage_error"
    else:
        left_name = str(a.model.iloc[0])
        right_name = str(b.model.iloc[0])
        reason = ("challenger_simpler_after_tie" if _risk_complexity(right_name) < _risk_complexity(left_name)
                  else "incumbent_simpler_after_tie")
    return {"selected": reason.startswith("challenger"), "reason": reason,
            "n_common": len(a), "incumbent_alert_ci": left_ci,
            "challenger_alert_ci": right_ci,
            "incumbent_top_coverage_error": left_error,
            "challenger_top_coverage_error": right_error}


def extend_selection(original_selection: dict, original_oof: pd.DataFrame,
                     candidate_oof: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Run h16/h96 M1/M4 challengers after the unchanged original selector."""
    selection = deepcopy(original_selection)
    n_boot = int(cfg.get("bootstrap", {}).get("n", 1000))
    seed = int(cfg.get("seed", 42))
    if n_boot < 1:
        raise ValueError("Date bootstrap must have positive draws")
    rows = []
    for horizon in (4, 16, 96):
        choice = selection["by_horizon"][str(horizon)]
        original_point = original_oof.loc[original_oof.horizon.eq(horizon)
                                          & original_oof.model.eq(choice["point_model"])]
        residual = candidate_oof.loc[candidate_oof.horizon.eq(horizon)
                                     & candidate_oof.model.eq(M1_NAME)]
        if residual.empty:
            evidence = {"selected": False, "eligible": False, "reason": "candidate_missing"}
        else:
            evidence = compare_point_candidate(original_point, residual,
                                               n_boot=n_boot, seed=seed)
        if horizon == 4:
            evidence["selected"] = False
            evidence["reason"] = "h4_reference_only"
        else:
            if evidence["selected"]:
                choice["point_model"] = M1_NAME
                for family, field in (("cbl", "vs_cbl_peak_mae"),
                                      ("persistence", "vs_persistence_peak_mae")):
                    source_name = choice.get(family)
                    if source_name:
                        source = original_oof.loc[original_oof.horizon.eq(horizon)
                                                  & original_oof.model.eq(source_name)]
                        choice[field] = paired_mae_improvement(source, residual, peak_only=True,
                                                               n=n_boot, seed=seed)
        choice["research_point_challenge"] = evidence
        rows.append({"horizon": horizon, "candidate": M1_NAME,
                     "stage": "point", "reference_only": horizon == 4,
                     "eligible": evidence.get("eligible", False),
                     "selected": evidence["selected"], "reason": evidence["reason"],
                     "incumbent_before": str(original_selection["by_horizon"][str(horizon)]["point_model"]),
                     "final_model": choice["point_model"],
                     "evidence": evidence})
        if horizon == 4:
            continue
        original_b = original_oof.loc[original_oof.horizon.eq(horizon)
                                      & original_oof.model.eq("lgbm_quantile_b")]
        if original_b.empty:
            raise ValueError("M4 comparison requires original B risk OOF")
        if choice.get("conformal") != "b":
            raise AssertionError("Phase 2 fixed risk comparison expected original conformal B")
        risk_incumbent = f"lgbm_quantile_{choice.get('conformal', 'b')}"
        original_risk = original_oof.loc[original_oof.horizon.eq(horizon)
                                         & original_oof.model.eq(risk_incumbent)]
        if original_risk.empty:
            raise ValueError("Original selected risk OOF is missing")
        risk_frames = {risk_incumbent: original_risk}
        selected_risk = risk_incumbent
        for name in M4_NAMES:
            candidate = candidate_oof.loc[candidate_oof.horizon.eq(horizon)
                                          & candidate_oof.model.eq(name)]
            if candidate.empty:
                gate = {"eligible": False, "reason": "candidate_missing"}
                challenge = {"selected": False, "reason": "gate_failed"}
            else:
                gate = _risk_gate(candidate, original_b, original_point)
                risk_frames[name] = candidate
                challenge = (_risk_challenge(risk_frames[selected_risk], candidate,
                                             n_boot=n_boot, seed=seed)
                             if gate["eligible"] else {"selected": False, "reason": "gate_failed"})
            selected_now = bool(gate["eligible"] and challenge["selected"])
            if selected_now:
                selected_risk = name
            evidence = {"gate": gate, "ranking": challenge,
                        "selected_at_challenge": selected_now}
            rows.append({"horizon": horizon, "candidate": name,
                         "stage": "risk_q95", "reference_only": False,
                         "eligible": gate["eligible"], "selected": selected_now,
                         "reason": (challenge["reason"] if gate["eligible"] else gate["reason"]),
                         "incumbent_before": risk_incumbent,
                         "final_model": selected_risk,
                         "evidence": evidence})
        choice["risk_model"] = selected_risk
        choice["adaptive_q95"] = selected_risk if selected_risk in M4_NAMES else None
        choice["research_risk_challenges"] = [row["evidence"] for row in rows
                                              if row["horizon"] == horizon and row["stage"] == "risk_q95"]
        for row in rows:
            if row["horizon"] == horizon and row["stage"] == "risk_q95":
                row["selected"] = row["candidate"] == selected_risk
                row["final_model"] = selected_risk
    selection["research_0925"] = {"selection_scope": "development_oof_only",
                                  "original_selection_retained_separately": True,
                                  "candidate_horizons": [4, 16, 96],
                                  "point_h4_reference_only": True,
                                  "m2_m3_gated_out": True}
    return selection, pd.DataFrame(rows)
