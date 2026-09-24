"""Causal, development-only candidates for adapting the upper power quantile.

The online replay keeps labels in a pending queue until their target interval
has ended.  This matters particularly for h96: the newest 96 forecasts cannot
contribute residuals merely because their origins precede the current origin.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .conformal import (apply_conformal, assign_mondrian_bins,
                        fit_conformal, fit_mondrian_edges, _finite_correction)
from .lgbm_quantile import fit_quantiles, predict_quantiles
from .peak_prob import exceedance_from_quantiles
from ..evaluate import score_predictions
from ..research_cv import build_contexts
from ..session_data import SEALED_BOUNDARY
from ..training import _cutoff, _row


CANDIDATES = {"q95_rolling_672": None, "q95_aci_005": .005,
              "q95_aci_010": .01, "q95_aci_050": .05}
LEVELS = (.1, .5, .9, .95, .975)
HOLIDAY_COLUMNS = {"is_offday", "pre_holiday", "post_holiday", "bridge_day", "labor_day"}


def replay_adaptive_q95(initial: pd.DataFrame, issued: pd.DataFrame, *,
                        gamma: float | None, max_residuals: int = 672,
                        min_residuals: int = 30,
                        sealed_only: bool = True) -> pd.DataFrame:
    """Issue chronological q95s using completed labels only.

    ``initial`` has first-half calibration residuals. ``issued`` contains the
    independent later calibration cutoff rows followed by score rows. Its y
    values stay unread in the pending queue until target_time <= next origin.
    For ACI, only *issued* forecasts update beta, using their issued q95.
    ``sealed_only=False`` is only for an externally approval-gated final replay;
    this pure function never opens data or authorizes a freeze itself.
    """
    if gamma is not None and gamma not in (.005, .01, .05):
        raise ValueError("ACI gamma must be one of the preregistered values")
    if max_residuals != 672 or min_residuals != 30:
        raise ValueError("Adaptive q95 window and fallback are preregistered")
    needed = {"origin", "target_time", "y", "q95", "q90_cal", "q95_cal", "q975_cal"}
    if not needed <= set(issued) or not {"target_time", "y", "q95"} <= set(initial):
        raise ValueError("Adaptive replay needs timestamps, labels, and raw/B quantiles")
    data = issued.copy().reset_index(drop=True)
    data["origin"] = pd.to_datetime(data.origin)
    data["target_time"] = pd.to_datetime(data.target_time)
    if data.origin.isna().any() or data.target_time.isna().any() or data.origin.ge(data.target_time).any():
        raise ValueError("Forecast origins must precede target intervals")
    if (not data.origin.is_monotonic_increasing or data.origin.duplicated().any()
            or not data.target_time.is_monotonic_increasing
            or data.target_time.duplicated().any()):
        raise ValueError("Adaptive forecasts need strictly chronological origins and targets")
    horizons = data.target_time - data.origin
    if horizons.nunique() != 1:
        raise ValueError("Adaptive replay requires one fixed forecast horizon")
    seed = initial.copy().reset_index(drop=True)
    seed["target_time"] = pd.to_datetime(seed.target_time)
    if seed.target_time.isna().any() or (len(seed) and seed.target_time.max() > data.origin.min()):
        raise ValueError("Initial calibration labels are not all known before replay")
    if sealed_only and (data.target_time.max() >= SEALED_BOUNDARY
                        or (len(seed) and seed.target_time.max() >= SEALED_BOUNDARY)):
        raise ValueError("Adaptive replay may not enter sealed test targets")
    if not np.isfinite(seed[["y", "q95"]].to_numpy(dtype=float)).all():
        raise ValueError("Initial calibration residuals must be finite")
    if not np.isfinite(data[["y", "q95", "q90_cal", "q95_cal", "q975_cal"]].to_numpy(dtype=float)).all():
        raise ValueError("Adaptive replay quantiles and labels must be finite")
    # All four preregistered candidates use the same maximum 672 completed
    # residuals; ACI changes beta while rolling holds it fixed at .95.
    residuals = deque(maxlen=max_residuals)
    latest = pd.NaT
    for row in seed.sort_values("target_time").itertuples(index=False):
        residuals.append(float(row.y - row.q95))
        latest = max(latest, row.target_time) if pd.notna(latest) else row.target_time
    beta = .95
    pending: deque[tuple[pd.Timestamp, float, float, float]] = deque()
    records = []
    for row in data.itertuples(index=False):
        # Forecast target times rise with origins for each direct horizon.
        while pending and pending[0][0] <= row.origin:
            target, observed, raw95, issued95 = pending.popleft()
            residuals.append(observed - raw95)
            if gamma is not None:
                hit = float(observed <= issued95)
                beta = float(np.clip(beta + gamma * (.95 - hit), .50, .999))
            latest = max(latest, target) if pd.notna(latest) else target
        fallback = len(residuals) < min_residuals
        correction = _finite_correction(residuals, beta) if not fallback else np.nan
        q95 = float(row.q95_cal if fallback else max(row.q90_cal, row.q95 + correction))
        q975 = float(max(row.q975_cal, q95))
        if q95 < row.q90_cal or q975 < q95 or (pd.notna(latest) and latest > row.origin):
            raise AssertionError("Adaptive q95 violated ordering or observation availability")
        records.append({"q95_cal": q95, "q975_cal": q975,
                        "latest_observation": latest, "n_completed_residuals": len(residuals),
                        "fallback": bool(fallback), "beta_at_issue": beta,
                        "correction_at_issue": correction})
        pending.append((row.target_time, float(row.y), float(row.q95), q95))
    online = pd.DataFrame(records)
    data["q95_cal_online"] = online.q95_cal.to_numpy(dtype=float)
    data["q975_cal_online"] = online.q975_cal.to_numpy(dtype=float)
    for col in ("latest_observation", "n_completed_residuals", "fallback",
                "beta_at_issue", "correction_at_issue"):
        data[col] = online[col].to_numpy()
    return data


def _baseline_b_quantiles(q_cal: dict, q_score: dict, y_cal: np.ndarray,
                          cfg: dict) -> tuple[dict, dict, float]:
    half = len(y_cal) // 2
    edges = fit_mondrian_edges(q_cal[.5], cfg.get("conformal", {}).get("mondrian_bins", [.5, .9]))
    cal_bins = assign_mondrian_bins(q_cal[.5], edges)
    score_bins = assign_mondrian_bins(q_score[.5], edges)
    cal_b, score_b = dict(q_cal), dict(q_score)
    for alpha in (.9, .95, .975):
        fitted = fit_conformal(y_cal[:half], q_cal[alpha][:half], alpha, cal_bins[:half])
        cal_b[alpha] = apply_conformal(q_cal[alpha], fitted, cal_bins)
        score_b[alpha] = apply_conformal(q_score[alpha], fitted, score_bins)
    from ..training import _ordered_quantiles
    return _ordered_quantiles(cal_b), _ordered_quantiles(score_b), float(edges[-1])


def _pinball_rows(frame: pd.DataFrame) -> np.ndarray:
    y = frame.y.to_numpy(dtype=float)
    columns = {"q10": .1, "q50": .5, "q90_cal": .9, "q95_cal": .95, "q975_cal": .975}
    losses = []
    for col, alpha in columns.items():
        delta = y - frame[col].to_numpy(dtype=float)
        losses.append(np.maximum(alpha * delta, (alpha - 1) * delta))
    return np.mean(np.column_stack(losses), axis=1)


def _paired_effects(reference: pd.DataFrame, candidate: pd.DataFrame,
                    n_boot: int = 1000, seed: int = 42) -> list[dict]:
    keys = ["fold", "target_time", "horizon"]
    b = reference[keys + ["y", "q50", "q50_top_edge", "q10", "q90_cal", "q95_cal", "q975_cal"]].copy()
    c = candidate[keys + ["q10", "q90_cal", "q95_cal", "q975_cal"]].copy()
    b = b.rename(columns={name: name + "_b" for name in ("q10", "q90_cal", "q95_cal", "q975_cal")})
    paired = b.merge(c, on=keys, validate="one_to_one")
    if len(paired) != len(reference) or len(paired) != len(candidate):
        raise AssertionError("Adaptive/B comparison changed the original score grid")
    top = paired.q50.to_numpy(dtype=float) >= paired.q50_top_edge.to_numpy(dtype=float)
    y = paired.y.to_numpy(dtype=float)
    b_hit = (y <= paired.q95_cal_b.to_numpy(dtype=float)).astype(float)
    c_hit = (y <= paired.q95_cal.to_numpy(dtype=float)).astype(float)
    # Each bootstrap block is a target date within its original fold. Never
    # create episodes by concatenating sampled dates.
    groups = [np.asarray(ids, dtype=int) for ids in paired.groupby(
        ["fold", pd.to_datetime(paired.target_time).dt.normalize()], sort=True).indices.values()]
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(n_boot, len(groups)))
    sizes = np.array([len(ids) for ids in groups], dtype=float)
    top_n = np.array([top[ids].sum() for ids in groups], dtype=float)
    bh = np.array([(b_hit[ids] * top[ids]).sum() for ids in groups])
    ch = np.array([(c_hit[ids] * top[ids]).sum() for ids in groups])
    b_pin = _pinball_rows(pd.DataFrame({"y": y, "q10": paired.q10_b,
                                         "q50": paired.q50, "q90_cal": paired.q90_cal_b,
                                         "q95_cal": paired.q95_cal_b,
                                         "q975_cal": paired.q975_cal_b}))
    c_pin = _pinball_rows(paired[["y", "q10", "q50", "q90_cal", "q95_cal", "q975_cal"]])
    bp = np.array([b_pin[ids].sum() for ids in groups])
    cp = np.array([c_pin[ids].sum() for ids in groups])
    fold_ids = [int(paired.fold.iloc[ids[0]]) for ids in groups]
    last = max(fold_ids)
    last_mask = np.asarray([fold == last for fold in fold_ids], dtype=bool)
    valid_top = top_n.sum() > 0 and top_n[last_mask].sum() > 0
    if not valid_top:
        raise ValueError("Top q50 coverage cohort is empty")
    b_cov = bh.sum() / top_n.sum(); c_cov = ch.sum() / top_n.sum()
    observed = [abs(b_cov-.95)-abs(c_cov-.95),
                (1.05*bp.sum()-cp.sum())/sizes.sum(),
                ch[last_mask].sum()/top_n[last_mask].sum() - bh[last_mask].sum()/top_n[last_mask].sum()]
    fold_effects = {}
    for fold in sorted(set(fold_ids)):
        part = np.asarray(fold_ids) == fold
        fold_b = bh[part].sum()/top_n[part].sum() if top_n[part].sum() else np.nan
        fold_c = ch[part].sum()/top_n[part].sum() if top_n[part].sum() else np.nan
        fold_effects[int(fold)] = [abs(fold_b-.95)-abs(fold_c-.95),
                                   (1.05*bp[part].sum()-cp[part].sum())/sizes[part].sum(),
                                   fold_c-fold_b]
    # Fold-specific resampling keeps the number of sampled dates per fold.
    indices_by_fold = [np.flatnonzero(np.asarray(fold_ids) == f) for f in sorted(set(fold_ids))]
    draw_blocks = np.empty_like(draws)
    for ids in indices_by_fold:
        draw_blocks[:, ids] = ids[rng.integers(len(ids), size=(n_boot, len(ids)))]
    d_top = top_n[draw_blocks].sum(axis=1)
    d_bh = bh[draw_blocks].sum(axis=1); d_ch = ch[draw_blocks].sum(axis=1)
    d_size = sizes[draw_blocks].sum(axis=1)
    d_bp = bp[draw_blocks].sum(axis=1); d_cp = cp[draw_blocks].sum(axis=1)
    d_last = draw_blocks[:, last_mask]
    d_last_top = top_n[d_last].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.column_stack([
            np.abs(d_bh/d_top-.95)-np.abs(d_ch/d_top-.95),
            (1.05*d_bp-d_cp)/d_size,
            ch[d_last].sum(axis=1)/d_last_top - bh[d_last].sum(axis=1)/d_last_top,
        ])
    rows = []
    for effect_index, (name, value, values) in enumerate(zip(
            ("top_coverage_error_improvement", "pinball_allowance",
             "last_fold_top_coverage_improvement"), observed, boot.T)):
        finite = values[np.isfinite(values)]
        ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
        # Centered one-sided bootstrap tests H0: effect <= 0.
        p = (1 + np.sum(finite - value >= value)) / (len(finite)+1) if len(finite) else 1.
        rows.append({"effect": name, "estimate": float(value), "ci_low": float(ci[0]),
                     "ci_high": float(ci[1]), "p_raw": float(p),
                     "fold_effects": json.dumps({str(f): float(v[effect_index]) if np.isfinite(v[effect_index]) else None
                                                  for f, v in fold_effects.items()}, sort_keys=True),
                     "n_positive_folds": int(sum(np.isfinite(v[effect_index]) and v[effect_index] > 0
                                                 for v in fold_effects.values())),
                     "bootstrap_valid_draws": len(finite), "n_top": int(top.sum()),
                     "n_top_last_fold": int(top[paired.fold.eq(last).to_numpy()].sum()),
                     "b_top_coverage": float(b_cov), "candidate_top_coverage": float(c_cov),
                     "b_pinball": float(bp.sum()/sizes.sum()), "candidate_pinball": float(cp.sum()/sizes.sum())})
    return rows


def fit_adaptive_candidates(history: pd.DataFrame, cfg: dict, manifest: dict,
                            reference_oof: pd.DataFrame, output_dir="outputs") -> dict:
    """Refit the *unchanged* quantile models on sealed h16/h96 folds and replay M4.

    This function does not select a candidate or read the final holdout.
    """
    import joblib
    output = Path(output_dir)
    if reference_oof.empty or (pd.to_datetime(reference_oof.target_time) >= SEALED_BOUNDARY).any():
        raise ValueError("M4 reference OOF must contain sealed development targets only")
    if int(cfg.get("bootstrap", {}).get("n", 1000)) != 1000 or int(cfg.get("seed", 42)) != 42:
        raise ValueError("M4 requires preregistered 1000 date blocks and seed 42")
    contexts = build_contexts(history, cfg, manifest, horizons=(16, 96))
    if len(contexts) != 6:
        raise AssertionError("M4 needs the original h16/h96 three folds")
    all_rows, audit_rows = [], []
    fit_count = 0
    for (horizon, fold), context in sorted(contexts.items()):
        x = context["x"]
        fit, stop, cal, score = (context[key] for key in ("fit", "stop", "cal", "score"))
        targets, tau = context["targets"], float(context["tau"])
        cols = [col for col in x.columns if col not in HOLIDAY_COLUMNS]
        params = context["expected"].get("chosen_params")
        if not isinstance(params, dict):
            raise ValueError("Original chosen_params missing")
        models = fit_quantiles(x.loc[fit, cols], targets.loc[fit, "y"],
                               x.loc[stop, cols], targets.loc[stop, "y"], cfg, params=params)
        fit_count += len(models)
        q_cal = predict_quantiles(models, x.loc[cal, cols])
        q_score = predict_quantiles(models, x.loc[score, cols])
        original_raw = reference_oof.loc[(reference_oof.horizon == horizon)
                                         & (reference_oof.fold == fold)
                                         & (reference_oof.model == "lgbm_quantile_raw")].sort_values("origin")
        if len(original_raw) != len(score) or not pd.DatetimeIndex(original_raw.origin).equals(score):
            raise AssertionError(f"Original raw OOF grid mismatch h{horizon}/f{fold}")
        for alpha in LEVELS:
            key = "q975" if alpha == .975 else f"q{int(alpha*100)}"
            if not np.allclose(q_score[alpha], original_raw[key].to_numpy(dtype=float), rtol=0, atol=1e-10):
                raise AssertionError(f"Original raw {key} OOF parity failed h{horizon}/f{fold}")
        y_cal = targets.loc[cal, "y"].to_numpy(dtype=float)
        b_cal, b_score, top_edge = _baseline_b_quantiles(q_cal, q_score, y_cal, cfg)
        original_b = reference_oof.loc[(reference_oof.horizon == horizon)
                                       & (reference_oof.fold == fold)
                                       & (reference_oof.model == "lgbm_quantile_b")].sort_values("origin")
        if len(original_b) != len(score) or not pd.DatetimeIndex(original_b.origin).equals(score):
            raise AssertionError("Original B OOF grid mismatch")
        for alpha, key in ((.9, "q90_cal"), (.95, "q95_cal"), (.975, "q975_cal")):
            if not np.allclose(b_score[alpha], original_b[key].to_numpy(dtype=float), rtol=0, atol=1e-10):
                raise AssertionError(f"Original B {key} OOF parity failed h{horizon}/f{fold}")
        half = len(cal)//2
        cutoff_start = min(len(cal), half+horizon+1)
        if cutoff_start >= len(cal) or pd.Timestamp(targets.loc[cal[half-1], "target_time"]) > cal[cutoff_start]:
            raise AssertionError("First-half calibration labels are not known at cutoff start")
        initial = pd.DataFrame({"target_time": targets.loc[cal[:half], "target_time"].to_numpy(),
                                "y": y_cal[:half], "q95": q_cal[.95][:half]})
        later_cal = cal[cutoff_start:]
        stream_origins = later_cal.append(score)
        stream = pd.DataFrame({"origin": stream_origins,
                               "target_time": targets.loc[stream_origins, "target_time"].to_numpy(),
                               "y": targets.loc[stream_origins, "y"].to_numpy(dtype=float),
                               "q95": np.concatenate([q_cal[.95][cutoff_start:], q_score[.95]]),
                               "q90_cal": np.concatenate([b_cal[.9][cutoff_start:], b_score[.9]]),
                               "q95_cal": np.concatenate([b_cal[.95][cutoff_start:], b_score[.95]]),
                               "q975_cal": np.concatenate([b_cal[.975][cutoff_start:], b_score[.975]]),
                               "is_score": [False]*len(later_cal)+[True]*len(score)})
        for name, gamma in CANDIDATES.items():
            issued = replay_adaptive_q95(initial, stream, gamma=gamma)
            q95 = issued.q95_cal_online.to_numpy(dtype=float)
            q975 = issued.q975_cal_online.to_numpy(dtype=float)
            q10 = np.concatenate([q_cal[.1][cutoff_start:], q_score[.1]])
            q50 = np.concatenate([q_cal[.5][cutoff_start:], q_score[.5]])
            q90 = stream.q90_cal.to_numpy(dtype=float)
            knots = {.1: q10, .5: q50, .9: q90, .95: q95, .975: q975}
            probabilities = exceedance_from_quantiles(knots, tau)
            cutoff = _cutoff(stream.loc[~stream.is_score, "y"].to_numpy(dtype=float),
                             probabilities[~stream.is_score.to_numpy()], tau,
                             stream.loc[~stream.is_score, "target_time"])
            score_mask = stream.is_score.to_numpy(dtype=bool)
            extras = {"q10": q_score[.1], "q50": q_score[.5], "q90": q_score[.9],
                      "q95": q_score[.95], "q975": q_score[.975],
                      "q90_cal": q90[score_mask], "q95_cal": q95[score_mask],
                      "q975_cal": q975[score_mask], "p_exceed": probabilities[score_mask],
                      "q50_top_edge": top_edge, "conformal_method": name,
                      "latest_observation": issued.latest_observation.loc[score_mask].to_numpy(),
                      "fallback": issued.fallback.loc[score_mask].to_numpy(),
                      "beta_at_issue": issued.beta_at_issue.loc[score_mask].to_numpy()}
            frame = _row(score, horizon, fold, name, targets, tau, q_score[.5], float("inf"), **extras)
            frame["alert"] = probabilities[score_mask] > cutoff
            frame["alert_cutoff"] = cutoff
            if (pd.to_datetime(frame.latest_observation) > pd.to_datetime(frame.origin)).any():
                raise AssertionError("Future score labels changed an earlier M4 forecast")
            all_rows.append(frame)
            audit_rows.append({"horizon": horizon, "fold": fold, "model": name,
                               "n_initial": len(initial), "n_cutoff_cal": len(later_cal),
                               "n_score": len(score), "fallback_cal": int(issued.fallback.loc[~score_mask].sum()),
                               "fallback_score": int(issued.fallback.loc[score_mask].sum()),
                               "cutoff": cutoff, "first_score_latest_observation": str(frame.latest_observation.iloc[0]),
                               "last_score_latest_observation": str(frame.latest_observation.iloc[-1]),
                               "quantile_fit_count": len(models)})
        bundle_dir = output / "models" / "p2_adaptive"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"quantile_models": models, "columns": cols, "tau": tau,
                     "initial_calibration": initial, "cutoff_calibration": stream.loc[~stream.is_score].copy(),
                     "top_edge": top_edge, "horizon": horizon, "fold": fold},
                    bundle_dir / f"h{horizon}_fold{fold}.joblib")
    predictions = pd.concat(all_rows, ignore_index=True).sort_values(
        ["horizon", "model", "fold", "target_time"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    metric_rows, hypothesis_rows = [], []
    for (horizon, name), candidate in predictions.groupby(["horizon", "model"], sort=True):
        baseline = reference_oof.loc[(reference_oof.horizon == horizon)
                                     & (reference_oof.model == "lgbm_quantile_b")].copy()
        candidate = candidate.sort_values(["fold", "target_time"]).reset_index(drop=True)
        baseline = baseline.sort_values(["fold", "target_time"]).reset_index(drop=True)
        for effect in _paired_effects(baseline, candidate, n_boot=int(cfg.get("bootstrap", {}).get("n", 1000)),
                                      seed=int(cfg.get("seed", 42))):
            hypothesis_rows.append({"analysis_id": "M4",
                                    "hypothesis_id": f"M4_h{horizon}_{name}_{effect['effect']}",
                                    "horizon": horizon, "model": name, **effect,
                                    "eligible": bool(np.isfinite(effect["estimate"])
                                                     and np.isfinite(effect["ci_low"])),
                                    "null_value": 0., "selection_used": False})
        scored = score_predictions(candidate)
        fold_scores = {int(fid): score_predictions(part) for fid, part in candidate.groupby("fold")}
        effects = hypothesis_rows[-3:]
        eligible = (effects[0]["estimate"] > 0 and effects[1]["estimate"] >= 0
                    and effects[2]["estimate"] > 0)
        metric_rows.append({"horizon": horizon, "model": name, "eligible_m4_nominal": bool(eligible),
                            "top_coverage_095": effects[0]["candidate_top_coverage"],
                            "top_coverage_error_improvement": effects[0]["estimate"],
                            "pinball_mean": scored.get("pinball_mean"),
                            "pinball_allowance": effects[1]["estimate"],
                            "last_fold_top_coverage_improvement": effects[2]["estimate"],
                            "episode_f1": scored.get("episode_f1"),
                            "false_alarms_positions": scored.get("false_alarms_positions"),
                            "fold_scores": str(fold_scores),
                            "point_stability_guard": "unchanged_original_point_model"})
    metrics = pd.DataFrame(metric_rows)
    hypotheses = pd.DataFrame(hypothesis_rows)
    (output / "analysis_p2").mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions" / "p2_adaptive_oof.csv", index=False, encoding="utf-8-sig")
    hypotheses.to_csv(output / "analysis_p2" / "M4_hypotheses.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "analysis_p2" / "M4_audit.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "analysis_p2" / "M4_metrics.csv", index=False, encoding="utf-8-sig")
    return {"predictions": predictions, "hypotheses": hypotheses, "audit": audit,
            "metrics": metrics, "model_fit_count": fit_count}
