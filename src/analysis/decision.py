"""Cost-loss relative economic value for hypothetical threshold decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import write_table


def relative_economic_value(frame: pd.DataFrame, ratio: float) -> float:
    """REV against the optimal climate-frequency policy, with L normalised to one."""
    y = frame.y.to_numpy(float) > frame.tau.to_numpy(float)
    p = frame.p_exceed.to_numpy(float)
    valid = np.isfinite(p)
    if not valid.any():
        return float("nan")
    y, p = y[valid], p[valid]
    event_rate = float(y.mean())
    climate = min(ratio, event_rate)
    perfect = ratio*event_rate
    cost = float(np.mean(ratio*(p > ratio) + y*(p <= ratio)))
    denominator = climate-perfect
    return (climate-cost)/denominator if denominator > 0 else float("nan")


def run_decision(pred: pd.DataFrame, outdir: Path, cfg: dict) -> dict:
    if "p_exceed" not in pred or pd.to_numeric(pred.p_exceed, errors="coerce").notna().sum() < 2:
        return {"status": "unsupported", "reason": "Calibrated exceedance probabilities are unavailable"}
    ratios = cfg.get("decision", {}).get("cl_ratios", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    n_boot = int(cfg.get("bootstrap", {}).get("n", 1000))
    seed = int(cfg.get("seed", 42))
    rows = []
    for horizon, h_frame in pred.groupby("horizon"):
        for ratio in ratios:
            ratio = float(ratio)
            point = relative_economic_value(h_frame, ratio)
            x = h_frame.loc[pd.to_numeric(h_frame.p_exceed, errors="coerce").notna()].copy()
            event = x.y.gt(x.tau).astype(float)
            act = x.p_exceed.gt(ratio)
            x["cost"] = ratio*act + event*(~act)
            x["event"] = event
            daily = x.groupby(x.target_time.dt.normalize()).agg(cost=("cost", "sum"), event=("event", "sum"), n=("event", "size"))
            if len(daily) >= 2:
                rng = np.random.default_rng(seed)
                ids = rng.integers(0, len(daily), size=(n_boot, len(daily)))
                sample_n = daily.n.to_numpy()[ids].sum(axis=1)
                sample_event = daily.event.to_numpy()[ids].sum(axis=1)/sample_n
                sample_cost = daily.cost.to_numpy()[ids].sum(axis=1)/sample_n
                climate = np.minimum(ratio, sample_event)
                perfect = ratio*sample_event
                denominator = climate-perfect
                values = np.divide(climate-sample_cost, denominator, out=np.full(n_boot, np.nan), where=denominator > 0)
                finite = values[np.isfinite(values)]
                low, high = map(float, np.quantile(finite, [.025, .975])) if len(finite) else (np.nan, np.nan)
            else:
                low = high = np.nan
            rows.append({"horizon": int(horizon), "cl_ratio": ratio, "rev": point,
                         "ci_low": low, "ci_high": high, "n": len(h_frame),
                         "event_rate": float((h_frame.y > h_frame.tau).mean())})
    result = pd.DataFrame(rows)
    path = write_table(result, outdir/"tables"/"relative_economic_value.csv")
    return {"status": "ok", "path": str(path),
            "warning": "C/L은 가정한 무차원 비용비이며 실제 요금이나 절감액이 아닙니다."}
