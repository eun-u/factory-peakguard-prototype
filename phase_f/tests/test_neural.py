"""Small CPU checks of genuine native neural adapters and split boundaries."""

import numpy as np
import pandas as pd
import pytest
import torch

from phase_f.models.neural import configurations, fit_model, predict_model


@pytest.fixture(scope="module")
def case():
    torch.set_num_threads(1)
    index = pd.date_range("2021-04-01 00:15", periods=16 * 96, freq="15min", name="ts_end")
    i = np.arange(len(index), dtype=float)
    power = 50 + 5 * np.sin(2 * np.pi * i / 96) + 0.002 * i
    history = pd.DataFrame({"power": power, "time_repaired": False}, index=index)
    fit = pd.DatetimeIndex(index[950:1010:2], name="origin")
    stop = pd.DatetimeIndex(index[1030:1050:2], name="origin")
    origins = pd.DatetimeIndex(index[1200:1203], name="origin")
    horizon = 4
    target_time = pd.Series(index, index=index) + pd.Timedelta(hours=1)
    y = pd.Series(power, index=index).reindex(index + pd.Timedelta(hours=1)).to_numpy()
    y = pd.Series(y, index=index)
    context = {"fit": fit, "stop": stop, "y": y, "target_time": target_time,
               "tau": 58.0, "summary": {"horizon_quarters": horizon}}
    return history, context, origins


@pytest.mark.parametrize("kind", ["dlinear", "nlinear", "tcn", "lstm", "gru"])
def test_native_architectures_fit_and_predict_without_future_access(case, kind):
    history, context, origins = case
    cfg = {"context_length": 16, "seeds": [42], "max_epochs": 2, "patience": 2,
           "batch_size": 16, "channels": 4, "depth": 2, "hidden_size": 8,
           "exog_columns": ["hour_sin"], "device": "cpu"}
    bundle = fit_model(kind, history, context, cfg)
    assert bundle["fit_count"] == len(context["fit"])
    assert bundle["stop_count"] == len(context["stop"])
    assert len(bundle["seeds"]) == 1
    assert bundle["seeds"][0]["best_epoch"] in (1, 2)
    before = predict_model(bundle, history, origins, 4)
    changed = history.copy()
    changed.loc[changed.index > origins.max(), "power"] += 50_000
    after = predict_model(bundle, changed, history.index.intersection(origins), 4)
    np.testing.assert_array_equal(before, after)
    assert before.shape == (len(origins),)
    assert np.isfinite(before).all()


def test_stop_labels_select_epoch_but_do_not_update_one_epoch_weights(case):
    history, context, _ = case
    cfg = {"context_length": 16, "seeds": [42], "max_epochs": 1, "patience": 1,
           "batch_size": 16, "device": "cpu"}
    original = fit_model("dlinear", history, context, cfg)
    poisoned = dict(context)
    poisoned["y"] = context["y"].copy()
    poisoned["y"].loc[context["stop"]] += 1_000
    modified = fit_model("dlinear", history, poisoned, cfg)
    state_a = original["seeds"][0]["state_dict"]
    state_b = modified["seeds"][0]["state_dict"]
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)
    assert original["seeds"][0]["stop_mae"] != modified["seeds"][0]["stop_mae"]


def test_weekly_residual_and_quantile_outputs_are_explicit(case):
    history, context, origins = case
    cfg = {"context_length": 16, "seeds": [42], "max_epochs": 1, "patience": 1,
           "target_baseline": "weekly", "loss": "quantile", "device": "cpu"}
    bundle = fit_model("dlinear", history, context, cfg)
    point, quantiles = predict_model(bundle, history, origins, 4, return_quantiles=True)
    assert set(quantiles) == {"0.1", "0.5", "0.9", "0.95"}
    np.testing.assert_array_equal(point, quantiles["0.5"])
    assert np.isfinite(point).all()
    assert (quantiles["0.1"] <= quantiles["0.5"]).all()
    assert (quantiles["0.5"] <= quantiles["0.9"]).all()
    with pytest.raises(ValueError, match="NLinear"):
        fit_model("nlinear", history, context, cfg)


def test_optional_models_are_named_honestly_and_split_violation_fails(case):
    history, context, _ = case
    plans = configurations()
    assert {row["kind"] for row in plans if row["status"] == "supported_native"} == {
        "dlinear", "nlinear", "tcn", "lstm", "gru"
    }
    assert any(row["kind"] == "nhits" and row["status"] == "supported_optional_neuralforecast" for row in plans)
    with pytest.raises(ValueError, match="forward does not consume"):
        fit_model("patchtst", history, context, {"use_calendar_exog": True})
    invalid = dict(context)
    invalid["stop"] = context["fit"][-3:]
    with pytest.raises(ValueError, match="chronological"):
        fit_model("dlinear", history, invalid, {"context_length": 16, "seeds": [42],
                                                 "max_epochs": 1, "patience": 1})


@pytest.mark.parametrize("kind,architecture", [
    ("nhits", {"n_blocks": [1, 1, 1], "mlp_units": [[16, 16]] * 3}),
    ("nbeats", {"n_blocks": [1, 1, 1], "mlp_units": [[16, 16]] * 3}),
    ("patchtst", {"encoder_layers": 1, "n_heads": 2, "hidden_size": 16,
                  "linear_hidden_size": 32, "patch_len": 4, "stride": 2}),
    ("tide", {"hidden_size": 32, "decoder_output_dim": 8,
              "temporal_decoder_dim": 16, "temporal_width": 4}),
    ("tsmixer", {"n_block": 1, "ff_dim": 16, "dropout": 0.1}),
    ("timesnet", {"hidden_size": 8, "conv_hidden_size": 8,
                  "top_k": 2, "num_kernels": 1, "encoder_layers": 1}),
    ("itransformer", {"hidden_size": 32, "n_heads": 2,
                      "e_layers": 1, "d_layers": 1, "d_ff": 64}),
])
def test_nixtla_native_architecture_trains_one_cpu_epoch_and_is_causal(case, kind, architecture):
    pytest.importorskip("neuralforecast")
    history, context, origins = case
    cfg = {"context_length": 16, "horizon": 4, "seeds": [42],
           "max_epochs": 1, "patience": 1, "batch_size": 8,
           "architecture_kwargs": architecture, "device": "cpu"}
    bundle = fit_model(kind, history, context, cfg)
    predicted = predict_model(bundle, history, origins, 4)
    changed = history.copy()
    changed.loc[changed.index > origins.max(), "power"] *= 100
    np.testing.assert_array_equal(predicted, predict_model(bundle, changed, origins, 4))
    assert predicted.shape == (len(origins),)
    assert np.isfinite(predicted).all()


def test_nixtla_quantile_head_is_native_and_ordered(case):
    pytest.importorskip("neuralforecast")
    history, context, origins = case
    cfg = {"context_length": 16, "horizon": 4, "seeds": [42],
           "max_epochs": 1, "patience": 1, "batch_size": 8, "loss": "quantile",
           "architecture_kwargs": {"n_blocks": [1, 1, 1], "mlp_units": [[16, 16]] * 3},
           "device": "cpu"}
    bundle = fit_model("nhits", history, context, cfg)
    point, quantiles = predict_model(bundle, history, origins, 4, return_quantiles=True)
    assert np.isfinite(point).all()
    np.testing.assert_array_equal(point, quantiles["0.5"])
    assert (quantiles["0.1"] <= quantiles["0.5"]).all()
