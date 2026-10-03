"""Synthetic fit/forecast checks; never touch the plant CSV or score artifacts."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from phase_f.models import regression as reg
from phase_c.statistical import predict_kalman as sealed_predict_kalman


def _data(horizon: int = 4) -> tuple[pd.DataFrame, dict]:
    index = pd.date_range("2021-01-01", periods=3000, freq="15min")
    slot = np.arange(len(index))
    power = 80 + 8 * np.sin(2 * np.pi * slot / 96) + 3 * np.cos(2 * np.pi * slot / 672)
    history = pd.DataFrame({"power": power}, index=index)
    targets = pd.Series(index + timedelta(minutes=15 * horizon), index=index)
    y = history.power.shift(-horizon)
    return history, {"horizon": horizon, "fit": index[1000:2200],
                     "stop": index[2250:2350], "cal": index[2400:2500],
                     "score": index[2550:2650], "y": y, "target_time": targets,
                     "tau": 80.0}


def test_ridge_fit_reads_only_fit_targets_and_prediction_is_future_invariant():
    history, context = _data()
    cfg = {"groups": ("slot_1_7d",), "target": "direct", "model_params": {"alpha": 10.}}
    fitted = reg.fit_model("ridge", history, context, cfg)
    changed = dict(context)
    y_changed = context["y"].copy()
    y_changed.loc[context["cal"]] = 1e9
    y_changed.loc[context["score"]] = -1e9
    changed["y"] = y_changed
    other = reg.fit_model("ridge", history, changed, cfg)
    assert fitted["n_fit_labels"] == len(context["fit"])
    assert fitted["stop_used"] is False
    stop_pred = reg.predict_model(fitted, history, context["stop"], 4)
    assert fitted["stop_mae"] == pytest.approx(np.mean(np.abs(stop_pred - context["y"].loc[context["stop"]])))
    assert fitted["stop_n"] == len(context["stop"])
    np.testing.assert_allclose(fitted["model"].coef_, other["model"].coef_)
    origin = pd.DatetimeIndex([history.index[2600]])
    forecast = reg.predict_model(fitted, history, origin, 4)
    altered = history.copy()
    altered.loc[altered.index > origin[0], "power"] = 1e6
    np.testing.assert_allclose(reg.predict_model(fitted, altered, origin, 4), forecast)


def test_missing_feature_is_retained_with_fit_median_and_mask():
    history, context = _data()
    model = reg.fit_model("ridge", history, context,
                          {"groups": ("same_slot_4w",), "model_params": {"alpha": 100.}})
    pre = model["preprocessor"]
    assert "same_slot_4w_mean" in pre["all_missing"]
    assert len(pre["medians"]) == len(pre["columns"])
    assert len(pre["mean"]) == 2 * len(pre["columns"])
    assert np.isfinite(reg.predict_model(model, history, pd.DatetimeIndex([history.index[2600]]), 4)).all()


@pytest.mark.parametrize("target", ["delta", "weekly", "profile", "log1p"])
def test_target_transforms_produce_finite_direct_forecasts(target: str):
    history, context = _data()
    fitted = reg.fit_model("ridge", history, context,
                           {"groups": (), "target": target, "model_params": {"alpha": 20.}})
    prediction = reg.predict_model(fitted, history, pd.DatetimeIndex([history.index[2600]]), 4)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    if target == "log1p":
        assert np.isfinite(fitted["bias"])


def test_lightgbm_uses_stop_only_for_early_stopping():
    pytest.importorskip("lightgbm")
    history, context = _data()
    cfg = {"groups": (), "target": "direct", "model_params": {"n_estimators": 30,
            "learning_rate": .1, "num_leaves": 8, "min_child_samples": 10},
           "early_stopping_rounds": 5}
    fitted = reg.fit_model("lightgbm", history, context, cfg)
    assert fitted["stop_used"]
    assert np.isfinite(reg.predict_model(fitted, history, pd.DatetimeIndex([history.index[2600]]), 4)).all()


@pytest.mark.parametrize("kind,package,params", [
    ("xgboost", "xgboost", {"n_estimators": 25, "max_depth": 3, "learning_rate": .1}),
    ("catboost", "catboost", {"iterations": 25, "depth": 3, "learning_rate": .1}),
])
def test_optional_boosters_fit_on_synthetic_history(kind, package, params):
    pytest.importorskip(package)
    history, context = _data()
    fitted = reg.fit_model(kind, history, context,
                           {"target": "direct", "model_params": params, "early_stopping_rounds": 5})
    assert fitted["stop_used"]
    assert np.isfinite(reg.predict_model(fitted, history, pd.DatetimeIndex([history.index[2600]]), 4)).all()


def test_daytype_and_two_stage_regressors_return_finite_predictions():
    history, context = _data()
    by_day = reg.fit_model("ridge", history, context,
                           {"target": "direct", "daytype": True, "feature_top_k": 7,
                            "model_params": {"alpha": 30.}})
    assert set(by_day["daytype_trained"]) >= {0, 1}
    origins = pd.DatetimeIndex([history.index[2600], history.index[2700]])
    assert np.isfinite(reg.predict_model(by_day, history, origins, 4)).all()
    stage = reg.fit_model("two_stage", history, context,
                          {"base_kind": "ridge", "target": "direct",
                           "model_params": {"alpha": 30.}})
    detail = reg.predict_model(stage, history, origins, 4, return_details=True)
    assert np.isfinite(detail["pred"]).all()
    assert np.all((detail["p_peak"] >= 0) & (detail["p_peak"] <= 1))


def test_window_and_exact_completed_profile_dedup():
    history, context = _data()
    window = reg.fit_model("ridge", history, context,
                           {"window_weeks": 1, "half_life_weeks": 1, "target": "direct"})
    assert 30 < window["n_fit_labels"] < len(context["fit"])
    h = 4
    targets = pd.DatetimeIndex([pd.Timestamp("2021-07-01") + timedelta(minutes=15 * k)
                                for k in range(1, 97)] +
                               [pd.Timestamp("2021-07-02") + timedelta(minutes=15 * k)
                                for k in range(1, 97)])
    origins = targets - timedelta(minutes=15 * h)
    y = np.tile(np.arange(96, dtype=float), 2)
    duplicate_context = {"target_time": pd.Series(targets, index=origins)}
    dropped = reg._duplicate_weights(duplicate_context, origins, y, h, "drop")
    weighted = reg._duplicate_weights(duplicate_context, origins, y, h, "weight")
    assert dropped[:96].sum() == 96
    assert dropped[96:].sum() == 0
    assert weighted.sum() == pytest.approx(96)


def test_continuous_peak_weights_match_q80_formula_and_disabled_path():
    y = np.arange(101, dtype=float)
    origins = pd.date_range('2021-01-01', periods=len(y), freq='15min')
    # q80=80 and tau=95: the weight at tau must be exactly 1+alpha.
    weights = reg._weights({}, origins, y, 95., 4, {'continuous_peak_alpha': 2.}, 'keep')
    np.testing.assert_allclose(weights, 1 + 2 * np.maximum(y - 80, 0) / 15)
    assert weights[95] == 3.
    np.testing.assert_array_equal(reg._weights({}, origins, y, 0., 4, {}, 'keep'), np.ones(101))
    from phase_f.support import UnsupportedConfiguration
    with pytest.raises(UnsupportedConfiguration, match='tau > fit q80'):
        reg._weights({}, origins, y, 80., 4, {'continuous_peak_alpha': 1.}, 'keep')
    for alpha in (-1., np.nan, np.inf):
        with pytest.raises(ValueError, match='continuous_peak_alpha'):
            reg._weights({}, origins, y, 95., 4, {'continuous_peak_alpha': alpha}, 'keep')


def test_continuous_weight_fit_ignores_nonfit_labels_on_transformed_target():
    history, context = _data()
    context['tau'] = float(context['y'].loc[context['fit']].quantile(.95))
    cfg = {'target': 'delta', 'continuous_peak_alpha': 1., 'model_params': {'alpha': 10.}}
    fitted = reg.fit_model('ridge', history, context, cfg)
    changed = dict(context)
    changed['y'] = context['y'].copy()
    for role in ('stop', 'cal', 'score'):
        changed['y'].loc[context[role]] = 1e9
    other = reg.fit_model('ridge', history, changed, cfg)
    np.testing.assert_array_equal(fitted['model'].coef_, other['model'].coef_)


def test_b5_residual_labels_are_forward_block_out_of_fit(monkeypatch):
    history, context = _data()
    train_ends = []

    def fake_fit(power, selected):
        train_ends.append(selected.max())
        return {"fit_end": selected.max()}

    def fake_predict(bundle, power, origins, horizon):
        assert bundle["fit_end"] < origins.min()
        return np.full(len(origins), 75.)

    monkeypatch.setattr(reg, "fit_kalman", fake_fit)
    monkeypatch.setattr(reg, "predict_kalman", fake_predict)
    oof, valid, metadata = reg._prequential_b5(history, context["fit"], 4, blocks=3)
    assert not valid[:metadata["warmup_excluded"]].any()
    assert valid.sum() >= 30
    assert np.all(oof[valid] == 75.)
    assert len(metadata["cutoffs"]) == 3
    assert train_ends[-1] == context["fit"].max()


def test_b5_residual_end_to_end_excludes_warmup_fit_rows(monkeypatch):
    history, context = _data()

    def fake_fit(power, selected):
        return {"fit_end": selected.max()}

    def fake_predict(bundle, power, origins, horizon):
        assert bundle["fit_end"] < origins.min()
        return np.full(len(origins), 75.)

    monkeypatch.setattr(reg, "fit_kalman", fake_fit)
    monkeypatch.setattr(reg, "predict_kalman", fake_predict)
    fitted = reg.fit_model("ridge", history, context,
                           {"target": "b5_residual", "b5_oof_blocks": 3,
                            "model_params": {"alpha": 30.}})
    assert 30 < fitted["n_fit_labels"] < len(context["fit"])
    assert fitted["b5"]["warmup_excluded"] > 0
    assert np.isfinite(reg.predict_model(fitted, history, pd.DatetimeIndex([history.index[2600]]), 4)).all()


@pytest.mark.parametrize("target", ["weekly", "delta"])
def test_kalman_state_features_use_forward_oof_fit_and_frozen_causal_score(monkeypatch, target):
    history, context = _data()
    fit_calls = []

    def fake_fit(power, selected):
        fit_calls.append(selected.max())
        return {"fit_end": selected.max(), "phi": .7, "q": .2, "r": .5}

    def fake_predict(bundle, power, origins, horizon):
        assert bundle["fit_end"] < origins.min()
        return 75 + .1 * power.reindex(origins).to_numpy(float)

    monkeypatch.setattr(reg, "fit_kalman", fake_fit)
    monkeypatch.setattr(reg, "predict_kalman", fake_predict)
    config = {"target": target, "kalman_features": True, "b5_oof_blocks": 3,
              "model_params": {"alpha": 30.}}
    fitted = reg.fit_model("ridge", history, context, config)
    assert fitted["kalman_features"]
    assert {"b5_forecast", "state_deviation", "state_variance"} <= set(fitted["preprocessor"]["columns"])
    assert fitted["n_fit_labels"] < len(context["fit"])
    assert fitted["target"] == target
    assert fitted["stop_mae"] >= 0
    calls_after_first = len(fit_calls)
    again = reg.fit_model("ridge", history, context, config)
    assert len(fit_calls) == calls_after_first  # fit-only B5 cache reused
    assert again["stop_mae"] == pytest.approx(fitted["stop_mae"])
    origin = pd.DatetimeIndex([history.index[2600]])
    expected = reg.predict_model(fitted, history, origin, 4)
    changed = history.copy()
    changed.loc[changed.index > origin[0], "power"] = -1e6
    np.testing.assert_allclose(reg.predict_model(fitted, changed, origin, 4), expected, rtol=0, atol=0)


def test_kalman_state_deviation_reconstructs_sealed_b5_forecast():
    history, _ = _data(horizon=16)
    bundle = {"phi": .7, "q": .2, "r": .5}
    origins = pd.DatetimeIndex([history.index[1200], history.index[1400], history.index[1800]])
    state, variance = reg._kalman_state(bundle, history.power, origins)
    forecast = sealed_predict_kalman(bundle, history.power, origins, 16)
    anchor = history.power.reindex(origins + timedelta(hours=4) - timedelta(days=7)).to_numpy(float)
    np.testing.assert_allclose(forecast, anchor + .7**16 * state, rtol=0, atol=1e-10)
    assert np.all(variance > 0)


def test_configurations_and_search_space_are_finite():
    rows = reg.configurations()
    assert len(rows) == len({row["id"] for row in rows})
    assert {"ridge", "lightgbm", "xgboost", "catboost", "two_stage"} <= {row["kind"] for row in rows}
    assert {"delta", "weekly", "profile", "b5_residual", "log1p"} <= {row["target"] for row in rows}

    class Trial:
        def suggest_int(self, name, lo, hi, **kwargs):
            return lo

        def suggest_float(self, name, lo, hi, **kwargs):
            return lo

        def suggest_categorical(self, name, options):
            return options[0]

    lgb = reg.suggest_parameters(Trial(), "lightgbm")
    assert lgb["n_estimators"] >= 300
    assert lgb["num_leaves"] == 15
    assert lgb["min_child_samples"] == 10
    assert lgb["learning_rate"] == .01
    assert {"max_bin", "reg_alpha", "reg_lambda"} <= lgb.keys()
