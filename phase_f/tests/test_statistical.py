"""Synthetic Phase F statistical contracts; no plant data are loaded."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_f.models import statistical as stat


def _history(n: int = 3100) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=n, freq="15min")
    slot = np.arange(n)
    value = 80 + 5 * np.sin(2 * np.pi * slot / 96) + 2 * np.cos(2 * np.pi * slot / 672)
    return pd.DataFrame({"power": value}, index=index)


def test_kweek_uses_only_complete_causal_anchors():
    history = _history()
    fit = history.index[:2700]
    bundle = stat.fit_model("seasonal_kweek", history, {"fit": fit}, {"weeks": 3, "statistic": "median"})
    origin = history.index[2750]
    h = 16
    target = origin + h * stat.STEP
    expected = np.median([history.loc[target - k * pd.Timedelta(days=7), "power"] for k in (1, 2, 3)])
    before = stat.predict_model(bundle, history, pd.DatetimeIndex([origin]), h)
    assert before[0] == pytest.approx(expected)
    future_changed = history.copy()
    future_changed.loc[future_changed.index > origin, "power"] = -1e6
    assert stat.predict_model(bundle, future_changed, pd.DatetimeIndex([origin]), h) == pytest.approx(before)
    missing = history.copy()
    missing.loc[target - 2 * pd.Timedelta(days=7), "power"] = np.nan
    assert np.isnan(stat.predict_model(bundle, missing, pd.DatetimeIndex([origin]), h)[0])


def test_daytype_profile_fit_masks_nonfit_values_and_forecast_is_causal():
    history = _history()
    fit = history.index[:2600:2]
    context = {"fit": fit}
    a = stat.fit_model("daytype_profile", history, context, {"level_span": 96})
    altered = history.copy()
    altered.loc[altered.index.difference(fit), "power"] = 10000
    b = stat.fit_model("daytype_profile", altered, context, {"level_span": 96})
    np.testing.assert_array_equal(a["profile"], b["profile"])
    origin = history.index[2800]
    p = stat.predict_model(a, history, pd.DatetimeIndex([origin]), 4)
    changed = history.copy()
    changed.loc[changed.index > origin, "power"] = -10000
    np.testing.assert_allclose(stat.predict_model(a, changed, pd.DatetimeIndex([origin]), 4), p)


class _MSTL:
    observed: list[np.ndarray] = []

    def __init__(self, season_length, trend_forecaster):
        assert season_length == [96, 672]

    def fit(self, y):
        self.observed.append(y.copy())
        n = len(y)
        self.model_ = pd.DataFrame({"trend": np.linspace(50, 100, n),
                                    "seasonal_96": 4 * np.sin(2 * np.pi * np.arange(n) / 96),
                                    "seasonal_672": 2 * np.cos(2 * np.pi * np.arange(n) / 672)})
        return self

    def predict(self, h):
        raise AssertionError("corrected MSTL must not extrapolate the fitted trend")


class _ETS:
    def __init__(self, model):
        assert model == "ZZN"


def test_stationary_mstl_freezes_seasonality_then_updates_causal_level(monkeypatch):
    import statsforecast.models

    monkeypatch.setattr(statsforecast.models, "MSTL", _MSTL)
    monkeypatch.setattr(statsforecast.models, "AutoETS", _ETS)
    _MSTL.observed.clear()
    history = _history(3100)
    fit = history.index[:2800]
    bundle = stat.fit_model("mstl_stationary", history, {"fit": fit})
    assert len(_MSTL.observed) == 1
    assert len(_MSTL.observed[0]) == 2688
    assert bundle["fit_end"] == fit.max()
    assert np.isfinite(bundle["weekly"]).all()
    origin = history.index[2900]
    initial = stat.predict_model(bundle, history, pd.DatetimeIndex([origin]), 16)
    changed = history.copy()
    changed.loc[changed.index > origin, "power"] = 9999
    np.testing.assert_allclose(stat.predict_model(bundle, changed, pd.DatetimeIndex([origin]), 16), initial)
    changed.loc[origin, "power"] += 30
    assert stat.predict_model(bundle, changed, pd.DatetimeIndex([origin]), 16)[0] > initial[0]
    changed_fit = history.copy()
    changed_fit.loc[changed_fit.index > fit.max(), "power"] = 9999
    again = stat.fit_model("mstl_stationary", changed_fit, {"fit": fit})
    np.testing.assert_allclose(again["weekly"], bundle["weekly"])
    assert again["alpha"] == bundle["alpha"]


@pytest.mark.parametrize("kind", ["kalman_profile", "kalman_hour_noise", "kalman_daytype_phi", "kalman_two_state", "kalman_robust"])
def test_kalman_variant_uses_only_origin_history(monkeypatch, kind):
    # Parameter optimizer has its own Phase C tests; this isolates each
    # Phase F variant's state update and origin forecast contracts.
    monkeypatch.setattr(stat, "_fit_ar1", lambda deviations, selected:
                        {"phi": .8, "q": 1.0, "r": 2.0, "n_fit_updates": int(selected.sum())})
    history = _history(2200)
    fit = history.index[:2050]
    bundle = stat.fit_model(kind, history, {"fit": fit})
    origin = history.index[2100]
    before = stat.predict_model(bundle, history, pd.DatetimeIndex([origin]), 8)
    changed = history.copy()
    changed.loc[changed.index > origin, "power"] = 100000
    np.testing.assert_allclose(stat.predict_model(bundle, changed, pd.DatetimeIndex([origin]), 8), before)


def test_configurations_have_distinct_ids_and_expose_all_requested_families():
    configs = stat.configurations()
    ids = [item["id"] for item in configs]
    assert len(ids) == len(set(ids))
    assert {"F0-2", "F0-3", "F4-1", "F4-2", "F4-4"} <= {row["family"] for row in configs}


def test_dshw_wrapper_masks_unselected_fit_and_trims_future(monkeypatch):
    import sys
    from types import SimpleNamespace

    class _FakeDSHW:
        fit_input = None
        predict_input = None

        def fit(self, series, fit_end):
            self.fit_input = series.copy()
            self.fit_end_ = fit_end
            self.fit_mse_ = 1.0
            self.params_ = (.1, .2, .2, .5)
            return self

        def predict(self, series, origins, horizon):
            self.predict_input = series.copy()
            return np.asarray(series.reindex(origins), dtype=float)

    monkeypatch.setitem(sys.modules, "src.models.seasonal", SimpleNamespace(DSHW=_FakeDSHW))
    history = _history(3100)
    fit = history.index[:2750:2]
    bundle = stat.fit_model("dshw", history, {"fit": fit})
    assert bundle["model"].fit_input.index.max() == fit.max()
    assert bundle["model"].fit_input.reindex(history.index[:2750].difference(fit)).isna().all()
    origin = history.index[2900]
    stat.predict_model(bundle, history, pd.DatetimeIndex([origin]), 4)
    assert bundle["model"].predict_input.index.max() == origin


def test_native_ets_fixed_parameters_forward_only_to_each_origin():
    history = _history(1700)
    fit = history.index[:1450]
    bundle = stat.fit_model("autoets", history, {"fit": fit}, {"season_length": 96})
    assert bundle["state_update"] == "native_forward"
    origins = pd.DatetimeIndex([history.index[1500], history.index[1540]])
    before = stat.predict_model(bundle, history, origins, 4)
    changed = history.copy()
    changed.loc[changed.index > origins[0], "power"] = 5000
    after = stat.predict_model(bundle, changed, origins, 4)
    assert after[0] == pytest.approx(before[0])
    assert after[1] != pytest.approx(before[1])
    fit_changed = history.copy()
    fit_changed.loc[fit_changed.index > fit.max(), "power"] = -5000
    again = stat.fit_model("autoets", fit_changed, {"fit": fit}, {"season_length": 96})
    np.testing.assert_allclose(again["model"].predict(h=4)["mean"], bundle["model"].predict(h=4)["mean"])


def test_b5_state_export_has_positive_variance_and_no_future_values():
    history = _history(1500)
    origin = history.index[1200]
    bundle = {"kind": "kalman_b5", "phi": .8, "q": 1.0, "r": 2.0, "fit_end": history.index[1100]}
    before = stat.predict_state(bundle, history, pd.DatetimeIndex([origin]))
    changed = history.copy()
    changed.loc[changed.index > origin, "power"] = 5000
    after = stat.predict_state(bundle, changed, pd.DatetimeIndex([origin]))
    pd.testing.assert_frame_equal(before, after)
    assert before["state_variance"].iloc[0] > 0
    with pytest.raises(ValueError, match="precedes"):
        stat.predict_state(bundle, history, pd.DatetimeIndex([history.index[1050]]))
