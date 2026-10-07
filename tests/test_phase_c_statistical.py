"""Synthetic contracts for Phase C's fixed statistical benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_c.statistical import fit_kalman, fit_mstl, predict_kalman, predict_mstl


def _power(n: int = 1250) -> pd.Series:
    grid = pd.date_range("2025-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(2026)
    daily = 4.0 * np.sin(2 * np.pi * np.arange(n) / 96)
    return pd.Series(30.0 + daily + rng.normal(0, 0.8, n), index=grid)


def test_kalman_fit_ignores_post_fit_and_forecast_ignores_post_origin():
    power = _power()
    fit = power.index[:1050]
    bundle = fit_kalman(power, fit)
    changed = power.copy()
    changed.loc[changed.index > fit.max()] = 9999.0
    second = fit_kalman(changed, fit)
    for key in ("phi", "q", "r", "fit_nll", "n_fit_updates"):
        assert second[key] == pytest.approx(bundle[key])
    assert bundle["development_only"] is True
    assert bundle["fit_parameters_only"] is True
    origin = power.index[1100]
    base = predict_kalman(bundle, power, pd.DatetimeIndex([origin]), 16)
    changed = power.copy()
    changed.loc[changed.index > origin] = -5000.0
    assert predict_kalman(bundle, changed, pd.DatetimeIndex([origin]), 16) == pytest.approx(base)


def test_kalman_weekly_anchor_and_missing_gap_are_causal():
    power = _power()
    power.iloc[750:756] = np.nan
    bundle = {"phi": 0.65, "q": 2.0, "r": 0.5}
    origin = power.index[840]
    horizon = 4
    prediction = predict_kalman(bundle, power, pd.DatetimeIndex([origin]), horizon)[0]
    changed = power.copy()
    changed.iloc[origin == changed.index] = power.loc[origin] + 5.0
    changed_prediction = predict_kalman(bundle, changed, pd.DatetimeIndex([origin]), horizon)[0]
    assert np.isfinite(prediction)
    assert changed_prediction != prediction
    zero_phi = {"phi": 0.0, "q": 2.0, "r": 0.5}
    expected_anchor = power.loc[origin + pd.Timedelta(minutes=15 * horizon) - pd.Timedelta(days=7)]
    assert predict_kalman(zero_phi, power, pd.DatetimeIndex([origin]), horizon)[0] == pytest.approx(expected_anchor)
    # An origin in the first week has no observed seasonal target anchor.
    assert np.isnan(predict_kalman(bundle, power, pd.DatetimeIndex([power.index[20]]), horizon)[0])
    # The target's prior-week observation is required and cannot be imputed.
    missing_anchor = power.copy()
    missing_anchor.loc[origin + pd.Timedelta(minutes=15 * horizon) - pd.Timedelta(days=7)] = np.nan
    assert np.isnan(predict_kalman(bundle, missing_anchor, pd.DatetimeIndex([origin]), horizon)[0])
    with pytest.raises(ValueError, match="weekly target anchor"):
        predict_kalman(bundle, power, pd.DatetimeIndex([origin]), 673)


class _FakeMSTL:
    fits: list[np.ndarray] = []

    def __init__(self, season_length, trend_forecaster):
        assert season_length == [96, 672]
        assert trend_forecaster.model == "ZZN"

    def fit(self, y):
        self.fits.append(y.copy())
        self.model_ = pd.DataFrame({"trend": np.full(len(y), 25.0), "seasonal_96": np.full(len(y), 2.0)})
        return self

    def predict(self, h):
        return {"mean": np.full(h, 27.0)}


class _FakeETS:
    def __init__(self, model):
        self.model = model


def test_mstl_one_fit_only_forward_fill_and_origin_formula(monkeypatch):
    import statsforecast.models

    monkeypatch.setattr(statsforecast.models, "MSTL", _FakeMSTL)
    monkeypatch.setattr(statsforecast.models, "AutoETS", _FakeETS)
    _FakeMSTL.fits.clear()
    power = _power(2900)
    fit = power.index[:2800]
    power.iloc[1000] = np.nan
    power.iloc[2000] = np.nan
    forecast_end = power.index[-1] + pd.Timedelta(days=1)
    bundle = fit_mstl(power, fit, forecast_end)
    assert len(_FakeMSTL.fits) == 1
    position = power.index.get_loc(power.index[2000]) - power.index.get_loc(bundle["fit_window_start"])
    assert _FakeMSTL.fits[0][position] == pytest.approx(power.iloc[1999])
    assert bundle["fit_forward_filled"] == 2
    assert bundle["development_only"] is True
    assert bundle["fit_parameters_only"] is True
    origins = pd.DatetimeIndex([power.index[2810], power.index[2820]])
    horizon = 16
    prediction = predict_mstl(bundle, power, origins, horizon)
    expected = 27.0 + bundle["phi"] ** horizon * (power.reindex(origins).to_numpy() - 27.0)
    assert prediction == pytest.approx(expected)
    altered = power.copy()
    altered.loc[altered.index > origins.max()] = 9000.0
    assert predict_mstl(bundle, altered, origins, horizon) == pytest.approx(prediction)
    second = fit_mstl(altered, fit, forecast_end)
    assert second["phi"] == pytest.approx(bundle["phi"])
    assert second["base_path"].equals(bundle["base_path"])


def test_mstl_does_not_backfill_leading_fit_window(monkeypatch):
    import statsforecast.models

    monkeypatch.setattr(statsforecast.models, "MSTL", _FakeMSTL)
    monkeypatch.setattr(statsforecast.models, "AutoETS", _FakeETS)
    power = _power(2688)
    power.iloc[:100] = np.nan
    fit = power.index
    with pytest.raises(ValueError, match="28 complete days"):
        fit_mstl(power, fit, power.index[-1] + pd.Timedelta(hours=1))
