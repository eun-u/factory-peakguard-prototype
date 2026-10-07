import numpy as np
import pandas as pd

from src.evaluate import symmetric_peak_metrics
from src.analysis.symmetric_metrics import metric_daily_totals, symmetric_peak_table


def test_auxiliary_metrics_use_tau_not_calibrated_alert():
    result = symmetric_peak_metrics([12, 8, 6, 15], [10, 13, 5, 18], 10)
    assert np.isclose(result["peak_mae_union"], 10 / 3)
    assert result["peak_bias"] == .5
    assert result["overpredict_rate"] == .5


def test_empty_populations_and_missing_values_remain_nan():
    result = symmetric_peak_metrics([0, np.nan], [0, 100], 10)
    assert np.isnan(result["peak_mae_union"])
    assert np.isnan(result["peak_bias"])
    assert result["overpredict_rate"] == 0
    assert np.isnan(symmetric_peak_metrics([12], [13], 10)["overpredict_rate"])


def test_cross_midnight_episode_is_not_split_or_joined_by_bootstrap():
    times = pd.date_range("2021-01-02 23:30", periods=5, freq="15min")
    frame = pd.DataFrame({"origin": times - pd.Timedelta(hours=1), "target_time": times,
                          "horizon": 4, "fold": 0, "model": "example",
                          "y": [8, 12, 14, 11, 8], "pred": [8, 11, 12, 12, 8],
                          "tau": 10, "alert": [False, True, True, True, False]})
    daily = metric_daily_totals(frame)
    assert daily.episode_tp.sum() == 1
    assert daily.episode_fp.sum() == daily.episode_fn.sum() == 0
    cfg = {"split": {"test_start_origin": "2021-08-09 09:45"}, "bootstrap": {"n": 1000}, "seed": 42}
    table = symmetric_peak_table(frame, cfg)
    assert table.iloc[0].episode_f1 == 1
    assert table.iloc[0].episode_f1_ci_low == 1
    assert not table.iloc[0].selection_used


def test_quantile_q50_reference_and_holdout_guard():
    import pytest
    times = pd.date_range("2021-02-02", periods=4, freq="15min")
    frame = pd.DataFrame({"origin": times-pd.Timedelta(hours=1), "target_time": times,
                          "horizon": 4, "fold": 0, "model": "lgbm_quantile_a",
                          "y": [12, 8, 12, 8], "pred": 999, "q50": [11, 9, 11, 9],
                          "tau": 10, "alert": [True, False, True, False]})
    cfg = {"split": {"test_start_origin": "2021-08-09 09:45"}, "bootstrap": {"n": 1000}, "seed": 42}
    table = symmetric_peak_table(frame, cfg)
    assert table.iloc[0].reference_only
    assert table.iloc[0].peak_mae == 1
    frame.loc[0, "target_time"] = pd.Timestamp("2021-08-09 09:45")
    with pytest.raises(ValueError, match="development"):
        symmetric_peak_table(frame, cfg)


def test_nonpurged_fold_boundary_is_rejected():
    import pytest
    times = pd.date_range("2021-02-02", periods=4, freq="15min")
    frame = pd.DataFrame({"target_time": times, "fold": [0, 0, 1, 1],
                          "y": 12, "pred": 13, "tau": 10, "alert": True})
    with pytest.raises(ValueError, match="purged"):
        metric_daily_totals(frame)
