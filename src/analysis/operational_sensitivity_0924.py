"""Preregistered management-target sensitivity; selection is unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import score_predictions
from ..targets import training_peak_threshold
from .alerting import _confirm
from .errors import match_episode_table
from .pipeline import _enrich_with_quantiles


def management_target_sensitivity(history, predictions, manifest, cfg):
    """Change management T alone; actual peaks remain the original fold tau.

    The operational lead convention is the existing target-end minus origin.
    The additional interval-start column exposes the shorter physical lead.
    Neither is a validation of the feasibility of moving production.
    """
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    if history.index.max() >= boundary or pd.to_datetime(predictions.target_time).ge(boundary).any():
        raise ValueError("Management sensitivity is development-only")
    fold_meta = {(int(f["horizon"]), int(f["fold"])): f for f in manifest["folds"]}
    rows, thresholds = [], []
    for horizon in (4, 16):
        choice = manifest["selection"]["by_horizon"][str(horizon)]
        point = predictions.loc[predictions.horizon.eq(horizon) & predictions.model.eq(choice["point_model"])]
        group = _enrich_with_quantiles(point, predictions, manifest["selection"], horizon).sort_values(["fold", "origin"])
        group["management_T"] = np.nan
        for fold, block in group.groupby("fold"):
            fit_end = pd.Timestamp(fold_meta[(horizon, int(fold))]["fit_end"])
            target_T = training_peak_threshold(history, pd.DatetimeIndex([fit_end]), .975)
            group.loc[block.index, "management_T"] = target_T
            thresholds.append({"horizon": horizon, "fold": int(fold),
                               "fit_end": str(fit_end), "management_T": target_T,
                               "management_target_quantile": .975,
                               "event_tau": float(block.tau.iloc[0])})
        group["candidate"] = group.p_exceed.gt(.1) | group.q95_cal.gt(group.management_T)
        for label, required, window in (("1/1", 1, 1), ("2/2", 2, 2), ("2/3", 2, 3)):
            if label not in cfg["alert"]["rules"]:
                continue
            group["confirmed"] = False
            for _, block in group.groupby("fold"):
                group.loc[block.index, "confirmed"] = _confirm(block, required, window)
            for prep in cfg["alert"]["prep_minutes"]:
                lead = (group.target_time - group.origin).dt.total_seconds() / 60
                action = group.confirmed & lead.ge(prep)
                physical_action = group.confirmed & (lead - 15).ge(prep)
                scored = group.assign(alert=action, alert_flag=action)
                metric = score_predictions(scored)
                events = match_episode_table(scored)
                hits = events.loc[events.status.eq("TP")]
                episode_lead = ((hits.start - hits.alarm_start).dt.total_seconds() / 60 + 15*horizon - prep)
                n_alert = int(group.confirmed.sum())
                rows.append({"horizon": horizon, "rule": label, "cl_ratio": .1,
                             "management_target_quantile": .975, "event_tau_quantile": .95,
                             "management_T_min": float(group.management_T.min()),
                             "management_T_max": float(group.management_T.max()),
                             "prep_minutes": prep, "alert_positions": n_alert,
                             "actionable_positions_target_end_convention": int(action.sum()),
                             "actionable_fraction": float(action.sum()/n_alert) if n_alert else np.nan,
                             "actionable_fraction_interval_start": float(physical_action.sum()/n_alert) if n_alert else np.nan,
                             "episode_f1": metric["episode_f1"],
                             "position_false_alarms": metric["position_fp"],
                             "episode_false_alarms": metric["episode_fp"],
                             "actionable_tp_episodes_target_end_convention": int(episode_lead.ge(0).sum()),
                             "actionable_tp_episodes_interval_start": int(episode_lead.ge(15).sum()),
                             "tp_episodes": len(hits), "selection_used": False,
                             "scenario": "사후 가정 시나리오",
                             "lead_convention": "existing target end; interval start feasibility separately reported"})
    result = pd.DataFrame(rows)
    result.attrs["fold_thresholds"] = thresholds
    return result
