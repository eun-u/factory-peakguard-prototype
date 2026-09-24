import pandas as pd

from src.analysis.operational_sensitivity_0924 import management_target_sensitivity


def test_management_target_does_not_redefine_actual_peaks():
    history = pd.DataFrame({"power": [2, 4, 8, 10], "time_repaired": False},
                           index=pd.date_range("2021-01-01", periods=4, freq="15min"))
    frames, folds, choices = [], [], {}
    for horizon in (4, 16):
        times = pd.date_range("2021-02-01", periods=4, freq="15min")
        for model in ("selected", "lgbm_quantile_b"):
            frames.append(pd.DataFrame({"origin": times-pd.Timedelta(minutes=15*horizon),
                "target_time": times, "horizon": horizon, "fold": 0, "model": model,
                "y": [9, 9, 1, 1], "pred": [9, 9, 1, 1], "tau": 8.,
                "alert": [True, True, False, False], "p_exceed": [.5, .5, 0., 0.],
                "q50": [9, 9, 1, 1], "q95_cal": [9, 9, 1, 1]}))
        choices[str(horizon)] = {"point_model": "selected", "conformal": "b"}
        folds.append({"horizon": horizon, "fold": 0, "fit_end": str(history.index.max())})
    cfg = {"split": {"test_start_origin": "2021-08-09 09:45"},
           "alert": {"rules": ["1/1"], "prep_minutes": [30, 60]}}
    result = management_target_sensitivity(history, pd.concat(frames),
                   {"selection": {"by_horizon": choices}, "folds": folds}, cfg)
    assert result.episode_f1.eq(1).all()
    assert result.event_tau_quantile.eq(.95).all()
    assert result.management_T_min.gt(9).all()
    assert len(result.attrs["fold_thresholds"]) == 2
    assert all(t["fit_end"] == str(history.index.max()) for t in result.attrs["fold_thresholds"])
    row = result.loc[result.horizon.eq(4) & result.prep_minutes.eq(60)].iloc[0]
    assert row.actionable_fraction == 1
    assert row.actionable_fraction_interval_start == 0
    assert row.actionable_tp_episodes_target_end_convention == 1
    assert row.actionable_tp_episodes_interval_start == 0
