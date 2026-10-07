"""F11 fixed-policy aggregation after the one-time confirmation has opened."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase_f.registry import sha256,config_hash
from phase_f.wf_downstream import _registry,_write_artifacts,_risk_and_coverage,paired_episode_ci
from outputs.phase_e.code.alerts import evaluate_alerts,THRESHOLDS


def merge_scores(explore,confirm,candidate):
    source=[]
    for arm,frame in (('EXPLORE',explore),('CONFIRM',confirm)):
        selected=frame.loc[frame.model.isin((candidate,'B5'))].copy()
        if set(selected.model)!={candidate,'B5'} or not selected.arm.eq(arm).all():
            raise ValueError('Combined F11 requires two sealed, separate weekly arms')
        source.append(selected)
    score=pd.concat(source,ignore_index=True)
    key=['fold','origin','target_time']
    if score.duplicated(['model',*key]).any():raise ValueError('Combined F11 repeats a forecast')
    for col in ('origin','target_time'):score[col]=pd.to_datetime(score[col])
    a=score.loc[score.model.eq(candidate)].sort_values(key).reset_index(drop=True)
    b=score.loc[score.model.eq('B5')].sort_values(key).reset_index(drop=True)
    if not a[key+['y','tau','d2']].equals(b[key+['y','tau','d2']]):
        raise ValueError('Combined F11 candidate/B5 cohort changed')
    return score


def evaluate_combined(view,candidate,*,n_boot=1000):
    sources=[];manifests={}
    for arm in ('EXPLORE','CONFIRM'):
        directory=view.out/'tables/phase_e'/arm/candidate
        path=directory/'manifest.json'
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('manifest_sha256')!=config_hash({k:v for k,v in record.items() if k!='manifest_sha256'}):
            raise ValueError('Combined F11 source manifest changed')
        for name,digest in record['source_sha256'].items():
            if sha256(view.root/name)!=digest:raise ValueError('Combined F11 source code changed')
        for name,digest in record['artifacts'].items():
            if sha256(directory/name)!=digest:raise ValueError('Combined F11 source table changed')
        sources.append(pd.read_csv(directory/'score_probabilities.csv'))
        manifests[arm]=sha256(path)
    score=merge_scores(*sources,candidate)
    tables=_risk_and_coverage(score,n_boot,42)
    alerts=[]
    for model,part in score.groupby('model',sort=True):
        alerts.append({name:frame.assign(model=model) for name,frame in evaluate_alerts(part,n_boot=n_boot,seed=42).items()})
    for name in alerts[0]:tables[name]=pd.concat([a[name] for a in alerts],ignore_index=True)
    fields,comparison,missing=_registry(tables,candidate)
    tables['comparison_vs_B5']=comparison
    tables['score_probabilities']=score
    ci=paired_episode_ci(tables['alert_episode_events'],candidate,'B5',n=1000,seed=42,
        represented_days=score.loc[score.model.eq(candidate),['fold','target_time']])
    tables['paired_episode_ci']=pd.DataFrame([{k:v for k,v in ci.items() if not isinstance(v,dict)}])
    clean={k:float(v) if isinstance(v,(float,np.floating)) and np.isfinite(v) else
        None if isinstance(v,(float,np.floating)) else v for k,v in fields.items()}
    manifest=_write_artifacts(view.out/'tables/f11_combined'/candidate,tables,{
        'candidate':candidate,'arm':'EXPLORE+CONFIRM','horizon':16,'at_stage':'after locked CONFIRM only',
        'source_manifest_sha256':manifests,'registry_fields':clean,'missing_metrics':missing,
        'fixed_thresholds':list(THRESHOLDS),'policies':['1/1','2/2'],
        'calibrator_refit':False,'source':'same per-week CAL-only calibrators, combined frozen score rows',
        'holdout_read':False,'selection_or_tuning_from_combined':False})
    return {'registry_fields':clean,'manifest_sha256':manifest['manifest_sha256'],
        'output':str((view.out/'tables/f11_combined'/candidate).relative_to(view.out))}
