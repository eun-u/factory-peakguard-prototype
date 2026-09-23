import pandas as pd
import pytest

from src.training import _partition, freeze_and_evaluate


def test_fit_stop_calibration_score_target_windows_are_disjoint():
    times = pd.date_range("2021-01-01", periods=1200, freq="15min")
    fit, stop, cal, score = _partition(times[:900], times[1000:], horizon=96)
    horizon = pd.Timedelta(days=1)
    assert fit.max() + horizon < stop.min()
    assert stop.max() + horizon < cal.min()
    assert cal.max() + horizon < score.min()


def test_final_holdout_cannot_run_before_freeze_even_with_data():
    cfg = {"freeze": {"not_before": "2100-01-01T18:00:00+09:00"}}
    with pytest.raises(RuntimeError, match="sealed"):
        freeze_and_evaluate(pd.DataFrame(), cfg)
