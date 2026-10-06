"""Weekly EXPLORE summaries, seed uncertainty, and fixed Phase E evaluation."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from phase_f import wf_metrics as metrics
from phase_f.registry import write_json,sha256,config_hash,now
from phase_f.wf_registry import WFRegistry

BASELINES={'B5':'F0-1-B5','M1':'F0-1-M1','R1':'F0-1-R1'}


def prediction_path(prepared,key,arm='EXPLORE'):
    return prepared.out/'predictions'/arm/f'{key}.parquet'


def load_predictions(prepared,key,arm='EXPLORE'):
    path=prediction_path(prepared,key,arm)
    record=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if record.get('sha256')!=sha256(path) or record.get('arm')!=arm:
        raise ValueError(f'Changed weekly prediction identity: {key}')
    return pd.read_parquet(path)


def baselines_ready(prepared,arm='EXPLORE'):
    return all(prediction_path(prepared,key,arm).exists() for key in BASELINES.values())


def summarize(prepared,spec,frame,audit,*,arm='EXPLORE',destination=None,update_registry=True):
    dest=Path(destination) if destination else prepared.out/'tables'/spec['id']/arm
    dest.mkdir(parents=True,exist_ok=True)
    tables=metrics.evaluate(frame,arm)
    pairs=[]
    for name,key in BASELINES.items():
        base=load_predictions(prepared,key,arm)
        pair=metrics.paired_ci(frame,base,arm,n=1000,seed=42)
        pair['baseline']=name;pairs.append(pair)
    pair=pd.concat(pairs,ignore_index=True)
    for name,value in tables.items():value.to_csv(dest/f'{name}.csv',index=False)
    pair.to_csv(dest/'pairwise_ci.csv',index=False)
    d1=tables['auc'].query("dataset == 'D1'").iloc[0]
    d2=tables['auc'].query("dataset == 'D2'").iloc[0]
    h16=tables['pooled'].query("dataset == 'D1' and horizon == 16").iloc[0]
    fields={'wf_explore_AUC_MAE':float(d1.AUC_MAE),'wf_explore_AUC_PeakMAE':float(d1.AUC_PeakMAE),
        'D2_wf_explore_AUC_MAE':float(d2.AUC_MAE),'D2_wf_explore_AUC_PeakMAE':float(d2.AUC_PeakMAE),
        'wf_explore_h16_MAE':float(h16.MAE),'wf_explore_h16_Peak':float(h16.Peak_MAE),
        'wf_explore_PredPeakMAE':float(d1.AUC_PredPeakMAE),
        'wf_explore_AUC_MAE_seed_sd':audit.get('seed_auc_mae_sd'),
        'n_seeds':audit['n_seeds'],'seed_auc_mae_mean':audit.get('seed_auc_mae_mean'),
        'leakage_test':audit.get('leakage_test'),'ranking_basis':'metric of seed-mean predictions'}
    seedrows=[]
    parent=spec.get('parent_spec',{}).get('id',spec['id']) if spec.get('adapter')=='seed_ensemble' else spec['id']
    for entry in audit.get('seeds',[]):
        seed=entry['seed'];path=prepared.out/'predictions/seeds'/arm/parent/f'seed_{seed}.parquet'
        if not path.exists():continue
        if sha256(path)!=entry['prediction_sha256']:raise ValueError('Seed prediction bytes changed')
        seedtables=metrics.evaluate(pd.read_parquet(path),arm)
        for data in ('D1','D2'):
            a=seedtables['auc'].loc[seedtables['auc'].dataset.eq(data)].iloc[0]
            p=seedtables['pooled'].query('dataset == @data and horizon == 16').iloc[0]
            seedrows.append({'seed':seed,'dataset':data,'AUC_MAE':a.AUC_MAE,
                            'AUC_PeakMAE':a.AUC_PeakMAE,'h16_PeakMAE':p.Peak_MAE})
    if seedrows:
        seeds=pd.DataFrame(seedrows);seeds.to_csv(dest/'seed_metrics.csv',index=False)
        s=seeds.loc[seeds.dataset.eq('D1')]
        fields['wf_explore_AUC_PeakMAE_seed_sd']=float(s.AUC_PeakMAE.std(ddof=1)) if len(s)>1 else 0.
        fields['wf_explore_h16_Peak_seed_sd']=float(s.h16_PeakMAE.std(ddof=1)) if len(s)>1 else 0.
    base=load_predictions(prepared,BASELINES['B5'],arm)
    basefold=metrics.evaluate(base,arm)['fold'].query("dataset == 'D1'").groupby('fold').MAE.mean()
    candfold=tables['fold'].query("dataset == 'D1'").groupby('fold').MAE.mean()
    weeks=pd.DataFrame({'candidate_MAE':candfold,'B5_MAE':basefold})
    weeks['MAE_improvement']=weeks.B5_MAE-weeks.candidate_MAE
    weeks.to_csv(dest/'weekly_improvement.csv')
    fields['wf_weeks_won_vs_B5']=int(weeks.MAE_improvement.gt(0).sum())
    fields['wf_explore_weeks']=len(weeks)
    b5ci=pair.query("baseline == 'B5' and dataset == 'D1' and horizon.isna()")
    m=b5ci.loc[b5ci.metric.eq('AUC_MAE_improvement')].iloc[0]
    fields['ciLow_vs_B5']=float(m.ci_low)
    p=b5ci.loc[b5ci.metric.eq('AUC_PeakMAE_degradation')].iloc[0]
    fields['wf_peak_degradation_ci_high_vs_B5']=float(p.ci_high)
    d=pair.query("baseline == 'B5' and dataset == 'D2' and horizon.isna()")
    fields['wf_D2_mae_improvement_vs_B5']=float(d.loc[d.metric.eq('AUC_MAE_improvement'),'estimate'].iloc[0])
    manifest={'arm':arm,'candidate':spec['id'],'at':now(),
        'prediction_sha256':sha256(prediction_path(prepared,spec['id'],arm)),
        'metrics_source_sha256':sha256(Path(metrics.__file__)),
        'split_sha256':prepared.split_lock['lock_sha256'],
        'tables':{p.name:sha256(p) for p in sorted(dest.glob('*.csv'))},
        'baseline_prediction_sha256':{name:sha256(prediction_path(prepared,key,arm)) for name,key in BASELINES.items()},
        'registry_metrics':fields,'candidate_confirm_metric_read':arm=='CONFIRM',
        'historical_final_artifact_read':False,'holdout_read':False}
    write_json(dest/'manifest.json',manifest)
    if arm=='EXPLORE' and update_registry:
        r=WFRegistry(prepared.root);old=r.read(spec['id'])
        r.update(spec['id'],**fields,explore_AUC_MAE=fields['wf_explore_AUC_MAE'],
            explore_AUC_PeakMAE=fields['wf_explore_AUC_PeakMAE'],
            explore_queries=old.get('explore_queries',0)+1,
            metric_manifest_sha256=sha256(dest/'manifest.json'))
    return fields


def verify_metrics(prepared,row,arm='EXPLORE',destination=None):
    dest=Path(destination) if destination else prepared.out/'tables'/row['exp_id']/arm
    path=dest/'manifest.json';record=json.loads(path.read_text(encoding='utf-8'))
    if (record['arm']!=arm or record['split_sha256']!=prepared.split_lock['lock_sha256'] or
        record['metrics_source_sha256']!=sha256(Path(metrics.__file__)) or
        record['prediction_sha256']!=sha256(prediction_path(prepared,row['exp_id'],arm))):
        raise ValueError('Weekly metric source, cohort or prediction changed')
    for name,digest in record['tables'].items():
        if sha256(dest/name)!=digest:raise ValueError('Weekly metric table changed')
    for name,digest in record['baseline_prediction_sha256'].items():
        if sha256(prediction_path(prepared,BASELINES[name],arm))!=digest:raise ValueError('Weekly metric baseline changed')
    if arm=='EXPLORE':
        for name,value in record['registry_metrics'].items():
            current=row.get(name)
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if current is None or not np.isclose(float(current),float(value),rtol=0,atol=1e-12,equal_nan=True):
                    raise ValueError(f'Weekly registry metric changed: {name}')
            elif current!=value:raise ValueError(f'Weekly registry metric changed: {name}')
    return record


def refresh_downstream(prepared,*,arm='EXPLORE',ids=None):
    from phase_f.wf_downstream import evaluate_candidate
    registry=WFRegistry(prepared.root)
    if ids is None:
        from phase_f.wf_plan import completed,spec
        rows=[r for r in completed(prepared.root) if spec(r).get('adapter')!='baseline'
              and r['exp_id'] not in BASELINES.values() and not r.get('duplicate_of')]
        ids=list(dict.fromkeys([*BASELINES.values(),*[r['exp_id'] for r in rows[:20]]]))
    base=load_predictions(prepared,BASELINES['B5'],arm).copy();base['model']='B5'
    for key in ids:
        frame=load_predictions(prepared,key,arm)
        digest=sha256(prediction_path(prepared,key,arm))
        dest=prepared.out/'tables/phase_e'/arm/key/digest[:16]
        try:
            manifest_path=dest/'manifest.json'
            if manifest_path.exists():
                record=json.loads(manifest_path.read_text(encoding='utf-8'))
                if record.get('manifest_sha256')!=config_hash({k:v for k,v in record.items() if k!='manifest_sha256'}):
                    raise ValueError('Phase E manifest changed')
                for relative,expected in record['artifacts'].items():
                    if sha256(dest/relative)!=expected:raise ValueError('Phase E artifact changed')
                for relative,expected in record['source_sha256'].items():
                    if sha256(prepared.root/relative)!=expected:raise ValueError('Phase E source changed')
                fields=record['registry_fields']
            else:
                result=evaluate_candidate(prepared,frame,base,arm=arm,destination=dest)
                fields=result['registry_fields']
            if arm=='EXPLORE':registry.update(key,**fields,E_status='completed',
                E_manifest_sha256=sha256(manifest_path),E_manifest_path=str(dest.relative_to(prepared.out)))
            if arm=='EXPLORE':
                audit=json.loads(prediction_path(prepared,key,arm).with_suffix('.json').read_text(encoding='utf-8'))['audit']
                values=[]
                cfg=json.loads(registry.read(key)['config_json'])
                parent=cfg.get('parent_spec',{}).get('id',key) if cfg.get('adapter')=='seed_ensemble' else key
                for entry in audit.get('seeds',[]):
                    seedpath=prepared.out/'predictions/seeds'/arm/parent/f"seed_{entry['seed']}.parquet"
                    if not seedpath.is_file():continue
                    if sha256(seedpath)!=entry['prediction_sha256']:raise ValueError('Phase E seed input changed')
                    seedframe=pd.read_parquet(seedpath);seedframe['model']=key
                    # Individual seed variability, not another tuning or score selection.
                    try:
                        seedresult=evaluate_candidate(prepared,seedframe,base,arm=arm,n_boot=1000)
                        values.append(seedresult['registry_fields'].get('E_c10_22_recall'))
                    except ValueError as exc:
                        values.append(None)
                        write_json(prepared.out/'logs/phase_e/seed_unavailable'/f"{key}_{entry['seed']}.json",
                            {'reason':str(exc),'seed_prediction_sha256':sha256(seedpath)})
                sd=float(np.std(values,ddof=1)) if len(values)>1 and all(v is not None and np.isfinite(v) for v in values) else 0. if audit['n_seeds']==1 else None
                registry.update(key,E_c10_22_recall_seed_sd=sd,E_recall_seed_metrics=values)
        except Exception as exc:
            if manifest_path.exists():raise
            if arm=='EXPLORE':registry.update(key,E_status='unavailable',E_reason=f'{type(exc).__name__}: {exc}')
            write_json(prepared.out/'logs/phase_e'/arm/f'{key}_unavailable.json',
                {'candidate':key,'reason':f'{type(exc).__name__}: {exc}','prediction_sha256':digest,
                 'holdout_read':False,'historical_final_artifact_read':False})
    return ids
