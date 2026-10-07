"""Fail-closed development selection tests; no plant data or model fits."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from phase_f import selection
from phase_f.registry import sha256,write_json
from phase_f.tuning import meaningful


def test_boolean_claims_cannot_unlock_confirmation(tmp_path):
    p=SimpleNamespace(root=tmp_path,out=tmp_path/'outputs/phase_f')
    with pytest.raises(RuntimeError,match='Search completion evidence'):
        selection.lock_selection(p,all_waves_complete=True,expansion_converged=True)
    assert not (p.out/'logs/confirm_lock.json').exists()


def test_zero_peak_margin_and_missing_ci_cannot_pass(monkeypatch,tmp_path):
    pairs=pd.DataFrame([
        ['D1','B5',np.nan,'AUC_MAE_improvement',3.,1.,4.,'available'],
        ['D1','B5',np.nan,'AUC_PeakMAE_degradation',-2.,-3.,-.01,'available'],
        ['D2','B5',np.nan,'AUC_MAE_improvement',2.,.1,4.,'available'],
        ['D2','B5',np.nan,'AUC_PeakMAE_degradation',-1.,-3.,2.,'available']],
        columns=['dataset','baseline','horizon','metric','estimate','ci_low','ci_high','ci_status'])
    baseline={f'fold{f}_MAE':10. for f in (0,1,2)}
    monkeypatch.setattr(selection,'Registry',lambda root:SimpleNamespace(read=lambda key:baseline))
    monkeypatch.setattr(selection,'verify_explore_metrics',lambda *args:'digest')
    monkeypatch.setattr(pd,'read_csv',lambda path:pairs.copy())
    row={'exp_id':'candidate','leakage_test':'passed',**{f'fold{f}_MAE':8. for f in (0,1,2)}}
    assert selection.eligibility(tmp_path,row)['eligible']
    pairs.loc[1,'ci_high']=.000001
    assert not selection.eligibility(tmp_path,row)['eligible']
    pairs.loc[1,'ci_high']=-1
    pairs.loc[1,'ci_status']='unavailable'
    assert not selection.eligibility(tmp_path,row)['eligible']


def test_changed_metric_bytes_and_registry_are_rejected(tmp_path):
    out=tmp_path/'outputs/phase_f';dest=out/'tables/E'
    dest.mkdir(parents=True);(out/'predictions').mkdir();(out/'logs').mkdir();(tmp_path/'phase_f').mkdir()
    pred=out/'predictions/E.parquet';pred.write_bytes(b'fixed predictions')
    metric=dest/'explore_pairwise_ci.csv';metric.write_text('fixed metrics')
    source=tmp_path/'phase_f/metrics.py';source.write_text('fixed source')
    baseline=tmp_path/'baseline';baseline.write_bytes(b'fixed baseline')
    write_json(out/'logs/split_lock.json',{'lock_sha256':'split'})
    row={'exp_id':'E','explore_AUC_MAE':2.,'explore_PredPeakMAE':float('nan')}
    record={'prediction_sha256':sha256(pred),'tables':{metric.name:sha256(metric)},
        'registry_metrics':{k:v for k,v in row.items() if k!='exp_id'},
        'metrics_source_sha256':sha256(source),'split_sha256':'split',
        'baseline_prediction_sha256':{'baseline':sha256(baseline)}}
    write_json(dest/'explore_manifest.json',record)
    assert selection.verify_explore_metrics(tmp_path,row)
    with pytest.raises(ValueError,match='Registry metrics'):
        selection.verify_explore_metrics(tmp_path,{**row,'explore_AUC_MAE':1.})
    metric.write_text('changed metrics')
    with pytest.raises(ValueError,match='metric table changed'):
        selection.verify_explore_metrics(tmp_path,row)


def test_expansion_requires_eligible_half_percent_gain():
    assert not meaningful(None,None)
    assert meaningful(None,12.)
    assert not meaningful(10.,9.96)
    assert meaningful(10.,9.94)


def test_primary_is_best_explore_and_family_cap_is_two(monkeypatch,tmp_path):
    out=tmp_path/'outputs/phase_f';(out/'predictions').mkdir(parents=True)
    rows=[]
    for i,family in enumerate(('F6','F6','F6','F3','F5','F8')):
        key=f'{family}-{i}';(out/'predictions'/f'{key}.parquet').write_bytes(key.encode())
        rows.append({'exp_id':key,'family':family,'config_hash':key,'explore_AUC_MAE':i+1,
                     'config_json':json.dumps({'id':key,'adapter':'foundation' if family=='F6' else 'regression','kind':'lightgbm','target':'direct'})})
    monkeypatch.setattr(selection,'verify_search_complete',lambda p:'search')
    monkeypatch.setattr(selection,'completed',lambda root:rows)
    monkeypatch.setattr(selection,'explore_eligibility',lambda root:[{'exp_id':r['exp_id'],'eligible':True} for r in rows])
    monkeypatch.setattr(selection,'verify_explore_metrics',lambda *args:'metrics')
    monkeypatch.setattr(selection,'Registry',lambda root:SimpleNamespace(read=lambda key:next(r for r in rows if r['exp_id']==key)))
    p=SimpleNamespace(root=tmp_path,out=out,split_lock={'lock_sha256':'split'})
    lock=selection.lock_selection(p,all_waves_complete=True,expansion_converged=True)
    assert lock['primary_candidate']=='F6-0'
    assert sum(key.startswith('F6-') for key in lock['candidates'])==2
    assert len(lock['candidates'])==5
    # Resume consumes the existing lock, not a freshly ranked primary.
    rows.reverse()
    assert selection.lock_selection(p,all_waves_complete=True,expansion_converged=True)==lock
