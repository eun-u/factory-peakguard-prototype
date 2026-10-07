"""Synthetic checks only: these never invoke the final evaluation path."""
import numpy as np
import pandas as pd
import pytest

import src.finalize as final


class ConstantResidual:
    def predict(self, frame):
        assert "residual_baseline" in frame
        assert "is_offday" not in frame
        return np.full(len(frame), 7.)


def test_final_point_preparation_fits_residuals_and_calibrates_restored_power(monkeypatch):
    origins = pd.date_range("2021-04-01", periods=120, freq="15min")
    x = pd.DataFrame({"lag_0": 75., "is_offday": 0}, index=origins)
    baseline = pd.DataFrame({"c3_holiday_hybrid": 100.}, index=origins)
    y = pd.Series(107., index=origins)
    target = pd.DataFrame({"target_time": origins+pd.Timedelta(hours=4)}, index=origins)
    fitted = {}

    def fit(train_x, train_y, stop_x, stop_y, cfg, **kwargs):
        assert np.array_equal(train_y, np.full(40, 7.))
        assert np.array_equal(stop_y, np.full(40, 7.))
        assert kwargs["peak_weight"] == 1.
        assert kwargs["peak_threshold"] is None
        fitted["columns"] = list(train_x)
        return ConstantResidual()

    def cutoff(actual, predicted, tau, timestamps):
        assert np.array_equal(actual, predicted)
        assert np.all(predicted == 107.)
        return 105.

    monkeypatch.setattr(final, "fit_point", fit)
    monkeypatch.setattr(final, "_cutoff", cutoff)
    point = final._fit_selected_point(x, y, origins[:40], origins[40:80], origins[80:],
        target, 105., {}, {}, "lgbm_residual_cbl", baseline, "c3_holiday_hybrid")
    assert fitted["columns"] == ["lag_0", "residual_baseline"]
    assert point["residual_baseline_model"] == "c3_holiday_hybrid"
    assert np.array_equal(final._predict_selected_point(point, x, baseline), np.full(120, 107.))


def test_missing_final_baseline_is_not_silently_filled():
    index = pd.date_range("2021-04-01", periods=3, freq="15min")
    x = pd.DataFrame({"lag_0": 75.}, index=index)
    b = pd.DataFrame({"c3": [100., np.nan, 200.]}, index=index)
    point = {"model": ConstantResidual(), "features": ["lag_0", "residual_baseline"],
             "residual_baseline_model": "c3"}
    assert np.allclose(final._predict_selected_point(point, x, b), [107., np.nan, 207.], equal_nan=True)
