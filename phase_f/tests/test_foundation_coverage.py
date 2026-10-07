"""Synthetic evidence checks for the pre-CONFIRM F6 training-budget gate."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from phase_f import selection
from phase_f.experiment_plan import foundation_fine
from phase_f.registry import config_hash, sha256, write_json


def _fixture(tmp_path):
    out=tmp_path/'outputs/phase_f'
    (out/'logs').mkdir(parents=True)
    (tmp_path/'phase_f').mkdir()
    (tmp_path/'phase_f/metrics.py').write_text('synthetic metric source',encoding='utf-8')
    write_json(out/'logs/split_lock.json',{'lock_sha256':'synthetic_split'})
    keys=[]
    for h in range(4,17):
        for fold in range(3):
            origin=pd.Timestamp('2021-06-01')+pd.Timedelta(days=fold,hours=h)
            keys.append({'horizon':h,'fold':fold,'origin':origin,
                         'target_time':origin+pd.Timedelta(minutes=15*h)})
    prepared=SimpleNamespace(root=tmp_path,out=out,keys=pd.DataFrame(keys))
    grid=foundation_fine(2)
    specs=[]
    for mode in ('full','lora'):
        for steps in (100,500):
            spec=next(s for s in grid if s['finetune']==mode and s['context_length']==512
                      and s['learning_rate']==1e-6 and s['num_steps']==steps)
            specs.append(spec)
            _write_fine(prepared,spec)
    return prepared,specs


def _write_fine(prepared,spec):
    out=prepared.out;exp_id=spec['id']
    row={'exp_id':exp_id,'family':'F6','status':'completed',
         'config_json':json.dumps(spec),'config_hash':config_hash(spec),
         'explore_AUC_MAE':5.0,'explore_queries':1,
         'historical_final_artifact_read':False,'holdout_read':False}
    write_json(out/'logs/experiments'/f'{exp_id}.json',row)
    frame=prepared.keys.copy()
    frame['model']=exp_id;frame['role']='score'
    pred=out/'predictions'/f'{exp_id}.parquet'
    pred.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(pred,index=False)
    digest=sha256(pred)
    audit={'config_hash':row['config_hash'],'prediction_sha256':digest,
           'leakage_test':'passed','historical_final_artifact_read':False,
           'holdout_read':False,'fit_cells':[
               {'fold':fold,'checkpoint_files':{'model.safetensors':'synthetic_hash'}}
               for fold in range(3)]}
    write_json(out/'logs'/f'{exp_id}_audit.json',audit)
    write_json(out/'tables'/exp_id/'explore_manifest.json',{
        'prediction_sha256':digest,'metrics_source_sha256':sha256(prepared.root/'phase_f/metrics.py'),
        'split_sha256':'synthetic_split','tables':{},'baseline_prediction_sha256':{},
        'registry_metrics':{'explore_AUC_MAE':5.0}})


def test_f6_full_and_lora_require_two_verified_steps(tmp_path):
    prepared,specs=_fixture(tmp_path)
    proof=selection.foundation_training_coverage(prepared)
    assert set(proof)=={'full','lora'}
    assert all(proof[mode]['distinct_num_steps']==[100,500] for mode in proof)
    assert all(item['score_cells']==39 and item['score_key_count']==39
               for mode in proof for item in proof[mode]['experiments'])
    removed=next(s for s in specs if s['finetune']=='lora' and s['num_steps']==500)
    row_path=prepared.out/'logs/experiments'/f"{removed['id']}.json"
    row=json.loads(row_path.read_text(encoding='utf-8'));row['status']='failed'
    write_json(row_path,row)
    with pytest.raises(ValueError,match='lora needs two distinct'):
        selection.foundation_training_coverage(prepared)


def test_f6_claims_do_not_replace_config_and_score_evidence(tmp_path):
    prepared,specs=_fixture(tmp_path)
    target=next(s for s in specs if s['finetune']=='full' and s['num_steps']==100)
    exp_id=target['id']
    row_path=prepared.out/'logs/experiments'/f'{exp_id}.json'
    original=json.loads(row_path.read_text(encoding='utf-8'))
    write_json(row_path,{**original,'config_hash':'fabricated'})
    with pytest.raises(ValueError,match='registry identity'):
        selection.foundation_training_coverage(prepared)
    write_json(row_path,original)

    pred=prepared.out/'predictions'/f'{exp_id}.parquet'
    frame=pd.read_parquet(pred).iloc[:-1].copy()
    frame.to_parquet(pred,index=False)
    digest=sha256(pred)
    audit_path=prepared.out/'logs'/f'{exp_id}_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    audit['prediction_sha256']=digest;write_json(audit_path,audit)
    metric_path=prepared.out/'tables'/exp_id/'explore_manifest.json'
    metric=json.loads(metric_path.read_text(encoding='utf-8'))
    metric['prediction_sha256']=digest;write_json(metric_path,metric)
    with pytest.raises(ValueError,match='complete 39-cell'):
        selection.foundation_training_coverage(prepared)


def test_search_complete_rechecks_f6_proof_before_confirmation(tmp_path,monkeypatch):
    prepared,specs=_fixture(tmp_path)
    fine=selection.foundation_training_coverage(prepared)
    wave=prepared.out/'logs/waves/f6.json'
    write_json(wave,{'name':'f6','specs':specs,'config_sha256':config_hash(specs)})
    tuning=prepared.out/'logs/tuning/gbdt.json'
    write_json(tuning,{'complete':True})
    manifest=prepared.out/'logs/search_complete.json'
    record={'waves':{'logs/waves/f6.json':sha256(wave)},
            'tuningfiles':{'logs/tuning/gbdt.json':sha256(tuning)},
            'required_family_coverage':{f'F{i}':{'status':'completed'} for i in range(11)},
            'expansion_rounds':[{'meaningful_improvement':False}]*2,
            'experiment_ids':[s['id'] for s in specs],
            'foundation_training_coverage':fine}
    write_json(manifest,record)
    original_completed=selection.completed
    trials=[{'exp_id':f'{prefix}{i:05d}','stop_cells':39}
            for prefix in ('F3-1-gbdt_lightgbm-t','F3-4-gbdt_xgboost-t',
                           'F3-4-gbdt_catboost-t') for i in range(500)]
    def completed_with_trials(root,**filters):
        actual=original_completed(root,**filters)
        return actual if filters else [*actual,*trials]
    monkeypatch.setattr(selection,'completed',completed_with_trials)
    assert selection.verify_search_complete(prepared)==sha256(manifest)
    record['foundation_training_coverage']['full']['distinct_num_steps']=[100,250]
    write_json(manifest,record)
    with pytest.raises(ValueError,match='training-budget completion evidence changed'):
        selection.verify_search_complete(prepared)
