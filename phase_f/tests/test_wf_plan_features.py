"""Conditional H1 feature propagation and single production comparison."""

import json

import pandas as pd
import pytest

from phase_f import wf_plan


def row(exp_id, groups, score, *, status="completed", production_check=False):
    cfg={"id":exp_id,"adapter":"regression","kind":"lightgbm","family":"F1",
         "groups":groups,"production_single_check":production_check}
    return {"exp_id":exp_id,"status":status,"config_json":json.dumps(cfg),
            "explore_AUC_MAE":score,"leakage_test":"passed"}


def test_feature_finish_never_repeats_production_or_uses_it_as_parent(monkeypatch):
    core=row("F1-core-lightgbm",[],10.)
    expanded=row("F1-expanded",["lags","rolling"],9.)
    production=row("F1-10-production-once",["lags","production"],8.,status="failed",
                   production_check=True)
    monkeypatch.setattr(wf_plan,"rows",lambda root:[core,expanded,production])
    monkeypatch.setattr(wf_plan,"completed",lambda root,**kwargs:[production,expanded,core])
    generated=wf_plan.feature_finish("unused")
    assert len(generated)==3
    assert all("production" not in cfg["groups"] for cfg in generated)
    assert all(cfg["parent"]=="F1-expanded" for cfg in generated)
    assert {cfg["feature_top_k"] for cfg in generated}=={10,20,40}


def test_feature_finish_emits_single_production_check_when_not_attempted(monkeypatch):
    core=row("F1-core-lightgbm",[],10.)
    expanded=row("F1-expanded",["lags"],9.)
    monkeypatch.setattr(wf_plan,"rows",lambda root:[core,expanded])
    monkeypatch.setattr(wf_plan,"completed",lambda root,**kwargs:[expanded,core])
    generated=wf_plan.feature_finish("unused")
    checks=[cfg for cfg in generated if cfg.get("production_single_check")]
    assert len(checks)==1
    assert checks[0]["groups"]==["lags","production"]
    assert all("production" not in cfg["groups"] for cfg in generated if cfg not in checks)


@pytest.mark.parametrize("ci_low,expected",[(.1,["lags"]),(-.1,None)])
def test_expanded_default_requires_paired_positive_ci(monkeypatch,ci_low,expected):
    core=row("F1-core-lightgbm",[],10.)
    expanded=row("F1-expanded",["lags"],9.)
    monkeypatch.setattr(wf_plan,"completed",lambda root,**kwargs:[expanded,core])
    from phase_f import wf_evaluation,wf_metrics
    loaded=[]
    def load(prepared,exp_id,arm="EXPLORE"):
        loaded.append((exp_id,arm))
        return pd.DataFrame({"exp_id":[exp_id]})
    monkeypatch.setattr(wf_evaluation,"load_predictions",load)
    monkeypatch.setattr(wf_metrics,"paired_ci",lambda *args,**kwargs:pd.DataFrame([{
        "dataset":"D1","metric":"AUC_MAE_improvement","horizon":None,
        "ci_low":ci_low,"ci_status":"available"}]))
    assert wf_plan._helpful_expanded_groups("unused")==expected
    assert loaded==[("F1-expanded","EXPLORE"),("F1-core-lightgbm","EXPLORE")]


def test_neural_grid_keeps_univariate_and_adds_paired_features(monkeypatch):
    from phase_f.models import neural
    monkeypatch.setattr(neural,"configurations",lambda:[
        {"exp_id":"F5-1","kind":"tcn","context_length":96},
        {"exp_id":"F5-3","kind":"nhits","context_length":192}])
    monkeypatch.setattr(wf_plan,"_helpful_expanded_groups",lambda root:["lags","rolling"])
    monkeypatch.setattr(wf_plan,"_origin_exog_columns",lambda root,groups:["lag_1","rolling_4"])
    settings=wf_plan.neural_grid("unused",2)
    assert len(settings)==4
    for kind in ("tcn","nhits"):
        base=next(s for s in settings if s["kind"]==kind and not s["id"].endswith("-features"))
        exog=next(s for s in settings if s["kind"]==kind and s["id"].endswith("-features"))
        assert "exog_columns" not in base
        assert exog["parent"]==base["id"]
        assert exog["exog_groups"]==["lags","rolling"]
        assert exog["exog_columns"]==["lag_1","rolling_4"]
        assert exog["expanded_feature_default"] is True


def test_neural_grid_stays_univariate_without_h1_evidence(monkeypatch):
    from phase_f.models import neural
    monkeypatch.setattr(neural,"configurations",lambda:[
        {"exp_id":"F5-1","kind":"tcn","context_length":96}])
    monkeypatch.setattr(wf_plan,"_helpful_expanded_groups",lambda root:None)
    monkeypatch.setattr(wf_plan,"_origin_exog_columns",lambda *args:pytest.fail("no columns needed"))
    settings=wf_plan.neural_grid("unused",2)
    assert len(settings)==1
    assert settings[0]["id"]=="F5-1-tcn-c96"
