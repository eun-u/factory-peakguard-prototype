"""Synthetic routing checks for the sealed Phase C training integration."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from phase_c import training


class _DummyPointModel:
    best_iteration_ = 7

    def predict(self, x):
        return np.full(len(x), 3.0)


def test_c1_uses_locked_lightgbm_and_weekly_residual(monkeypatch):
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    x = pd.DataFrame({"current": np.arange(8, dtype=float),
                      "slot7d": np.arange(10, 18, dtype=float)}, index=index)
    y = pd.Series(np.arange(20, 28, dtype=float), index=index)
    context = {"x": x, "y": y, "fit": index[:3], "stop": index[3:5], "score": index[5:]}
    candidate = {"id": "selected", "num_leaves": 15, "min_child_samples": 40}
    cfg = {"seed": 42, "lgbm": {"n_estimators": 600, "early_stopping": 60}}
    seen = []

    def fake_fit_point(x_fit, y_fit, x_stop, y_stop, passed_cfg, *, peak_threshold,
                       peak_weight, params):
        seen.append({"fit_y": y_fit.to_numpy(), "stop_y": y_stop.to_numpy(),
                     "cfg": passed_cfg, "threshold": peak_threshold,
                     "weight": peak_weight, "params": params})
        return _DummyPointModel()

    monkeypatch.setattr(training, "fit_point", fake_fit_point)
    saved = {}

    def fake_save(path, family, model, seconds, effective):
        saved.update({"path": path, "family": family,
                      "artifact": {"model": model, "train_seconds": seconds,
                                   "effective": effective}})

    monkeypatch.setattr(training, "_save_family", fake_save)
    monkeypatch.setattr(training, "_load_family", lambda path, family: saved["artifact"])
    prediction, record = training._score_family(training.Path("synthetic_models"), "lgbm", "C1", 4, 0, context,
                                                cfg, candidate, residual=True)
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0]["fit_y"], [10, 10, 10])
    np.testing.assert_allclose(seen[0]["stop_y"], [10, 10])
    assert seen[0]["cfg"] is cfg
    assert seen[0]["threshold"] is None
    assert seen[0]["weight"] == 1.0
    assert seen[0]["params"] == {"num_leaves": 15, "min_child_samples": 40}
    np.testing.assert_allclose(prediction, [18, 19, 20])
    assert record["effective"]["best_iteration"] == 7
    assert saved["family"] == "lgbm"
    assert saved["path"].name == "score_lgbm_C1_h04_f0.joblib"

    lock = {"lgbm": {"selected_id": "selected",
                     "selected_config": {"num_leaves": 15, "min_child_samples": 40}}}
    selected_cfg = {"peak_weight": 2.0}
    m1 = json.loads(training._config_for_model("M1", 4, lock, selected_cfg))
    weighted = json.loads(training._config_for_model("M1-W", 4, lock, selected_cfg))
    c1 = json.loads(training._config_for_model("C1", 4, lock, selected_cfg))
    assert m1["model_params"] == weighted["model_params"] == c1["model_params"]
    assert [m1["peak_weight"], weighted["peak_weight"], c1["peak_weight"]] == [1.0, 2.0, 1.0]
    assert c1["target_transform"] == "weekly_residual"


def test_peak_weight_only_variant_excludes_cal_and_does_not_reuse_anchor(monkeypatch):
    index = pd.date_range("2021-02-01", periods=10, freq="15min")
    x = pd.DataFrame({"current":np.arange(10,dtype=float),"slot7d":np.ones(10)},index=index)
    y = pd.Series([10,20,30,40,50,1e12,1e12,60,70,80],index=index,dtype=float)
    c={"x":x,"y":y,"fit":index[:3],"stop":index[3:5],"cal":index[5:7],"score":index[7:],"tau":25.0}
    candidate={"id":"default","num_leaves":31,"min_child_samples":40}
    seen={}
    def fake_fit(x_fit,y_fit,x_stop,y_stop,cfg,**kwargs):
        seen.update(kwargs,fit_index=x_fit.index,stop_index=x_stop.index,fit_y=y_fit.to_numpy())
        return _DummyPointModel()
    monkeypatch.setattr(training,"fit_point",fake_fit)
    saved={}
    def fake_save(path,family,model,seconds,effective):
        saved.update(path=path,artifact={"model":model,"train_seconds":seconds,"effective":effective})
    monkeypatch.setattr(training,"_save_family",fake_save)
    monkeypatch.setattr(training,"_load_family",lambda *args:saved["artifact"])
    training._score_family(training.Path("synthetic_models"),"lgbm","M1-W",4,0,c,{},candidate,peak_weight=2.0)
    assert seen["peak_threshold"]==25.0 and seen["peak_weight"]==2.0
    assert seen["params"]=={"num_leaves":31,"min_child_samples":40}
    assert not seen["fit_index"].append(seen["stop_index"]).isin(c["cal"]).any()
    assert seen["fit_y"].max()==30.0
    assert saved["path"].name=="score_lgbm_M1-W_h04_f0.joblib"
