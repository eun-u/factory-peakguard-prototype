import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from phase_f.tests.test_metrics import predictions
from phase_f import wf_metrics,wf_evaluation
from phase_f.registry import write_json,sha256


def test_chronos_status_requires_actual_zero_perturbation_for_every_seed():
    spec={'adapter':'foundation'}
    audit={'leakage_test':'requires_family_audit','n_seeds':2,'seeds':[
        {'seed':42,'audit':{'fit_cells':[{'future_perturbation_max_abs_difference':0}]}},
        {'seed':43,'audit':{'fit_cells':[{'future_perturbation_max_abs_difference':0}]}}]}
    assert wf_evaluation.verified_leakage_status(spec,audit)=='passed'
    audit['seeds'][1]['audit']['fit_cells'][0]['future_perturbation_max_abs_difference']=1
    assert wf_evaluation.verified_leakage_status(spec,audit)=='requires_family_audit'
    audit['seeds'][1]['audit']['fit_cells'][0]['future_perturbation_max_abs_difference']=None
    assert wf_evaluation.verified_leakage_status(spec,audit)=='requires_family_audit'
    assert wf_evaluation.verified_leakage_status({'adapter':'neural'},audit)=='requires_family_audit'
    audit['leakage_test']='passed'
    for invalid in (1,None,False,float('nan')):
        audit['seeds'][1]['audit']['fit_cells'][0]['future_perturbation_max_abs_difference']=invalid
        assert wf_evaluation.verified_leakage_status(spec,audit)=='requires_family_audit'
    audit['seeds'][1]['audit']['fit_cells']=[]
    assert wf_evaluation.verified_leakage_status(spec,audit)=='requires_family_audit'


def test_variable_locked_folds_and_contaminated_confirm_are_separate():
    f=predictions('candidate'); f['fold']=f.fold.map({0:1,1:3,2:5})
    extra=f.loc[f.fold.eq(5)].copy();extra['fold']=7
    extra['origin']+=pd.Timedelta(days=7);extra['target_time']+=pd.Timedelta(days=7)
    f=pd.concat([f,extra],ignore_index=True)
    contamination=f.iloc[:1].copy();contamination['arm']='CONFIRM'
    contamination['y']=np.nan;contamination['pred']=np.nan
    got=wf_metrics.evaluate(pd.concat([f,contamination],ignore_index=True))
    assert set(got['fold'].fold)=={1,3,5,7}
    b=f.copy();b['model']='B5';b['pred']+=2
    ci=wf_metrics.paired_ci(f,b,n=1000,seed=42)
    assert ci.query("dataset == 'D1' and metric == 'AUC_MAE_improvement'").estimate.iloc[0]==pytest.approx(2)
    with pytest.raises(ValueError):wf_metrics.paired_ci(f,b.iloc[:-1])


def test_mean_prediction_requires_arm_and_byte_identity(tmp_path):
    p=SimpleNamespace(out=tmp_path)
    path=wf_evaluation.prediction_path(p,'F1-example');path.parent.mkdir(parents=True)
    pd.DataFrame({'x':[1]}).to_parquet(path)
    write_json(path.with_suffix('.json'),{'arm':'CONFIRM','sha256':sha256(path)})
    with pytest.raises(ValueError,match='identity'):wf_evaluation.load_predictions(p,'F1-example')
    write_json(path.with_suffix('.json'),{'arm':'EXPLORE','sha256':sha256(path)})
    assert wf_evaluation.load_predictions(p,'F1-example').x.iloc[0]==1
    pd.DataFrame({'x':[2]}).to_parquet(path)
    with pytest.raises(ValueError):wf_evaluation.load_predictions(p,'F1-example')
