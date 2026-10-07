"""Lead-time-aware warning policies on development out-of-fold forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import write_table
from .errors import match_episode_table


def decision_feed(predictions: pd.DataFrame, tariff_bands: pd.Series | None = None,
                  ratio: float = 0.1, target_T: float | None = None,
                  scope: str = "development_oof") -> pd.DataFrame:
    """Create provisional decision records; target_T is a specified management target."""
    x = predictions.copy().sort_values(["horizon", "origin"])
    q50 = pd.to_numeric(x["q50"], errors="coerce") if "q50" in x else x.pred
    q95 = pd.to_numeric(x["q95_cal"], errors="coerce") if "q95_cal" in x else (
        pd.to_numeric(x["q95"], errors="coerce") if "q95" in x else pd.Series(np.nan, index=x.index))
    prob = pd.to_numeric(x["p_exceed"], errors="coerce") if "p_exceed" in x else pd.Series(np.nan, index=x.index)
    targets = pd.Series(target_T, index=x.index) if target_T is not None else x.tau
    margin = targets-q95
    high = prob.gt(ratio) | margin.lt(0)
    known_risk = prob.notna() | q95.notna()
    action_applicable = x.horizon.isin([4, 16])
    risk_exceeded = high.astype("boolean").where(known_risk, pd.NA)
    level = pd.Series(pd.NA, index=x.index, dtype="Int64")
    level.loc[action_applicable & known_risk] = 0
    level.loc[x.horizon.eq(4) & high] = 2
    level.loc[x.horizon.eq(16) & high] = 1
    if tariff_bands is None:
        bands = pd.Series(pd.NA, index=x.index)
    else:
        bands = pd.Series(pd.DatetimeIndex(x.target_time).map(tariff_bands), index=x.index)
    return pd.DataFrame({"origin": x.origin, "target_time": x.target_time,
                         "horizon": x.horizon, "fold": x.fold, "q50": q50,
                         "q95_cal": q95, "p_exceed": prob,
                         "exp_exceed": x["exp_exceed"] if "exp_exceed" in x else np.nan,
                         "margin": margin, "tariff_band_2026": bands,
                         "action_applicable": action_applicable,
                         "risk_exceeded": risk_exceeded, "alert_level": level,
                         "point_model": x["model"] if "model" in x else pd.NA,
                         "risk_model": x["risk_model"] if "risk_model" in x else pd.NA,
                         "scenario": scope}).reset_index(drop=True)


def _confirm(group: pd.DataFrame, required: int, window: int) -> np.ndarray:
    """Consecutive 15-minute origins only; gaps reset the confirmation history."""
    group = group.sort_values("origin")
    raw = group.candidate.to_numpy(bool)
    origins = group.origin.to_numpy(dtype="datetime64[ns]")
    result = np.zeros(len(group), dtype=bool)
    for i in range(len(group)):
        first = max(0, i-window+1)
        recent = raw[first:i+1]
        contiguous = len(recent) == window and np.all(np.diff(origins[first:i+1]) == np.timedelta64(15, "m"))
        result[i] = bool(contiguous and recent.sum() >= required)
    return result


def compare_alert_rules(pred: pd.DataFrame, ratios=(0.1,), prep_minutes=(15, 30, 60)) -> pd.DataFrame:
    if "p_exceed" not in pred and "q95_cal" not in pred:
        return pd.DataFrame()
    results = []
    for horizon, h_group in pred.groupby("horizon"):
        if horizon not in (4, 16):
            continue
        for ratio in ratios:
            g = h_group.copy().sort_values(["fold", "origin"])
            prob = pd.to_numeric(g.get("p_exceed", pd.Series(np.nan, index=g.index)), errors="coerce")
            q95 = pd.to_numeric(g.get("q95_cal", pd.Series(np.nan, index=g.index)), errors="coerce")
            g["candidate"] = prob.gt(ratio) | q95.gt(g.tau)
            for label, required, window in (("1/1", 1, 1), ("2/2", 2, 2), ("2/3", 2, 3)):
                g["confirmed"] = False
                for _, fg in g.groupby("fold", sort=False):
                    g.loc[fg.index, "confirmed"] = _confirm(fg, required, window)
                events = match_episode_table(g.rename(columns={"confirmed": "alert_flag"}))
                tp = int(events.status.eq("TP").sum())
                fn = int(events.status.eq("FN").sum())
                fp = int(events.status.eq("FP").sum())
                for prep in prep_minutes:
                    lead = pd.to_timedelta(g.target_time-g.origin).dt.total_seconds()/60-prep
                    eligible = lead.ge(0)
                    alarm_positions = g.confirmed & eligible
                    hours = max(len(g)/4, 0.25)
                    hits = events.loc[events.status.eq("TP")].copy()
                    if not hits.empty:
                        hits["episode_lead_minutes"] = ((hits.start-hits.alarm_start).dt.total_seconds()/60
                                                        +horizon*15-prep)
                    actionable_hits = hits.loc[hits.episode_lead_minutes.ge(0)] if not hits.empty else hits
                    results.append({"horizon": int(horizon), "ratio": float(ratio), "rule": label,
                                    "prep_minutes": int(prep), "true_episodes": tp,
                                    "missed_episodes": fn, "false_episodes": fp,
                                    "episode_hit_rate": tp/(tp+fn) if tp+fn else np.nan,
                                    "false_alarm_positions_per_operating_hour": int((g.confirmed & ~g.y.gt(g.tau) & eligible).sum())/hours,
                                    "actionable_alert_positions": int(alarm_positions.sum()),
                                    "nonactionable_alert_positions": int((g.confirmed & ~eligible).sum()),
                                    "actionable_tp_episodes": len(actionable_hits),
                                    "actionable_episode_hit_rate": len(actionable_hits)/(tp+fn) if tp+fn else np.nan,
                                    "mean_actionable_lead_minutes": float(actionable_hits.episode_lead_minutes.mean()) if len(actionable_hits) else np.nan})
    return pd.DataFrame(results)


def run_alerting(pred: pd.DataFrame, outdir: Path, cfg: dict,
                 tariff_bands: pd.Series | None = None, scope: str = "development_oof") -> dict:
    ratios = cfg.get("decision", {}).get("cl_ratios", [0.1])
    prep = cfg.get("alert", {}).get("prep_minutes", [15, 30, 60])
    feed = decision_feed(pred, tariff_bands, ratio=0.1, scope=scope)
    rules = compare_alert_rules(pred, ratios=ratios, prep_minutes=prep)
    paths = {"decision_feed": str(write_table(feed, outdir/"predictions"/"decision_feed.csv")),
             "alert_rules": str(write_table(rules, outdir/"tables"/"alert_rules.csv"))}
    return {"status": "ok" if not rules.empty else "partial", "paths": paths,
            "warning": ("OOF 개발구간의 경보 정책 비교입니다." if scope == "development_oof" else
                        "동결된 테스트 예측에 개발구간에서 정한 경보 규칙을 적용한 사후 평가입니다.")}
