"""P4 seasonal models and blend weights: causality and determinism."""
import numpy as np
import pandas as pd
import pytest

from src.models.seasonal import (DSHW, mstl_day_paths, mstl_forecast, origin_days, safe_grid,
                                 simplex_weights)


def _series(days=30, seed=0):
    index = pd.date_range("2021-01-01 00:15", periods=days * 96, freq="15min")
    t = np.arange(len(index))
    rng = np.random.default_rng(seed)
    values = 100 + 20 * np.sin(2 * np.pi * t / 96) + 8 * np.sin(2 * np.pi * t / 672) + rng.normal(0, 2, len(t))
    return pd.DataFrame({"power": values, "time_repaired": False}, index=index)


def test_safe_grid_masks_repaired_readings():
    df = _series(15)
    df.iloc[500, df.columns.get_loc("time_repaired")] = True
    grid = safe_grid(df)
    assert np.isnan(grid.iloc[500]) and grid.notna().sum() == len(df) - 1


def test_dshw_forecast_ignores_future_observations():
    df = _series(20)
    series = safe_grid(df)
    model = DSHW().fit(series, series.index[1500])
    origins = series.index[1500:1700]
    before = model.predict(series, origins, 4)
    changed = series.copy()
    changed[changed.index > origins[-1]] = 999.0
    assert np.array_equal(before, model.predict(changed, origins, 4))
    changed_fit = series.copy()
    changed_fit[changed_fit.index > series.index[1500]] = 999.0
    assert DSHW().fit(changed_fit, series.index[1500]).params_ == model.params_


def test_dshw_rejects_origins_inside_initialisation():
    series = safe_grid(_series(20))
    model = DSHW().fit(series, series.index[1500])
    with pytest.raises(ValueError):
        model.predict(series, series.index[:10], 1)


def test_mstl_day_path_uses_only_data_up_to_that_midnight():
    df = _series(30)
    series = safe_grid(df)
    day = pd.Timestamp("2021-01-30 00:00")
    base = mstl_day_paths(series, [day])
    changed = series.copy()
    changed[changed.index > day] = 999.0
    again = mstl_day_paths(changed, [day])
    pd.testing.assert_series_equal(base[day], again[day])
    origins = pd.date_range("2021-01-30 00:15", periods=8, freq="15min")
    assert (origin_days(origins) == day).all()
    now = mstl_forecast(series, base, origins, 1, .9)
    assert np.isfinite(now).all()


def test_simplex_weights_recovers_exact_component():
    y = np.arange(50, dtype=float)
    components = [y + 5, y.copy(), y - 3]
    assert simplex_weights(components, y) == (0.0, 1.0, 0.0)
    assert abs(sum(simplex_weights([y, y + 1], y)) - 1) < 1e-12
