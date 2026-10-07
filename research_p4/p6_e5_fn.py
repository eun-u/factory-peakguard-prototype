"""E5: layered decomposition of missed peaks for the selected h1/h4 point and risk outputs (dev OOF)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import _episodes, match_episodes
from src.session_data import read_development_oof
from .common import ROOT
from .p6_common import out_dir, write_json

SELECTED = {1: ("p4_mstl_daily", "p4_quantile_dense"), 4: ("p4_blend3", "lgbm_quantile_b")}


def block(ts):
    return "00-08" if ts.hour < 8 else ("08-16" if ts.hour < 16 else "16-24")


def run():
    oof = read_development_oof(ROOT / "outputs/predictions/development_oof.csv")
    oof["target_time"] = pd.to_datetime(oof.target_time)
    out = out_dir("e5_fn")
    episodes, summary = [], {}
    for h, (point_name, risk_name) in SELECTED.items():
        for f in range(3):
            risk = oof.loc[oof.horizon.eq(h) & oof.fold.eq(f) & oof.model.eq(risk_name)].sort_values("target_time")
            point = oof.loc[oof.horizon.eq(h) & oof.fold.eq(f) & oof.model.eq(point_name)].set_index("target_time")
            times = pd.DatetimeIndex(risk.target_time)
            actual = risk.y.to_numpy() > risk.tau.to_numpy()
            alert = risk.alert.astype(bool).to_numpy()
            match = match_episodes(actual, alert, times)
            matched = dict(match["matches"])
            q95 = risk.q95_cal.to_numpy(float)
            point_alert = point.reindex(times).alert.astype("boolean").fillna(False).to_numpy(bool)
            alerts = match["alert_episodes"]
            for ai, (a0, a1) in enumerate(match["actual_episodes"]):
                if ai in matched:
                    p0, _ = alerts[matched[ai]]
                    category = "late_hit" if p0 > a0 else "hit"
                elif np.nanmax(q95[a0:a1 + 1]) >= risk.tau.iloc[a0]:
                    category = "decision_miss"
                else:
                    category = "forecast_miss"
                episodes.append({"horizon": h, "fold": f, "start": times[a0], "end": times[a1],
                                 "length_intervals": a1 - a0 + 1, "block": block(times[a0]),
                                 "weekday": times[a0].day_name(), "peak_value": float(np.nanmax(risk.y.to_numpy()[a0:a1 + 1])),
                                 "excess_over_tau": float(np.nanmax(risk.y.to_numpy()[a0:a1 + 1]) - risk.tau.iloc[a0]),
                                 "category": category, "point_alert_any": bool(point_alert[a0:a1 + 1].any())})
            fp_eps = [pi for pi in range(len(alerts)) if pi not in set(matched.values())]
            summary.setdefault(str(h), {}).setdefault("fp_alert_episodes", 0)
            summary[str(h)]["fp_alert_episodes"] += len(fp_eps)
    table = pd.DataFrame(episodes)
    table.to_csv(out / "fn_episodes.csv", index=False)
    for h in SELECTED:
        t = table.loc[table.horizon.eq(h)]
        summary[str(h)].update({
            "point_model": SELECTED[h][0], "risk_model": SELECTED[h][1], "actual_episodes": len(t),
            "by_category": t.category.value_counts().to_dict(),
            "by_block_category": t.groupby(["block", "category"]).size().unstack(fill_value=0).to_dict(orient="index"),
            "excess_by_category_mean": t.groupby("category").excess_over_tau.mean().round(2).to_dict(),
            "point_model_catches_risk_misses": int(t.loc[t.category.isin(["decision_miss", "forecast_miss"])].point_alert_any.sum())})
    write_json(out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    run()
