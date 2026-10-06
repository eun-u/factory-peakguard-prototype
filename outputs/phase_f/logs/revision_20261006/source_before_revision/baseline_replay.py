"""Read-only parent-model replay; separate Phase F calibration caches."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from phase_c import training
from phase_c.statistical import predict_kalman, fit_kalman
from phase_f.harness import make_frame, KEY
from phase_f.registry import write_json, sha256


def _artifact(prepared,model,h,f,lock):
    models=prepared.root/'outputs/phase_c/models'
    if model=='B5':return joblib.load(models/f'B5_h{h:02d}_f{f}.joblib')
    family='tcn' if model=='M2' else 'lgbm'
    candidate_id=lock[family]['selected_id'] if model in ('M1','M2') and h in training.ANCHORS else model
    path=training._family_path(models,family,candidate_id,h,f,tuning=model in ('M1','M2') and h in training.ANCHORS)
    if not path.is_file():raise FileNotFoundError(path.name)
    return training._load_family(path,family)


def replay(prepared,model,*,roles=('cal','score')):
    cfg=json.loads((prepared.root/'configs/phase_c.json').read_text(encoding='utf-8'))
    lock=json.loads((prepared.root/'outputs/phase_c/logs/model_config_lock.json').read_text(encoding='utf-8'))
    frames=[]
    audit=[]
    for (h,f),context in prepared.contexts.items():
        artifact=_artifact(prepared,model,h,f,lock) if model!='B1' else None
        for role in roles:
            origins=prepared.origins(h,f,role)
            tick=perf_counter()
            if model=='B1':pred=context['x'].loc[origins,'slot7d'].to_numpy(float)
            elif model=='B5':pred=predict_kalman(artifact['bundle'],prepared.history.power,origins,h)
            else:
                copied={**context,role:origins}
                pred=training._predict_family('tcn' if model=='M2' else 'lgbm',artifact['model'],copied,role,cfg)
            frames.append(make_frame(context,origins,h,f,pred,model,role,
                train_seconds=0 if artifact is None else artifact['train_seconds'],inference_seconds=perf_counter()-tick))
        part=frames[-1]
        if 'score' in roles:
            original=prepared.baselines[prepared.baselines.model.eq(model)&prepared.baselines.horizon.eq(h)&prepared.baselines.fold.eq(f)].sort_values('origin')
            diff=np.abs(part.sort_values('origin').pred.to_numpy()-original.pred.to_numpy())
            audit.append({'model':model,'horizon':h,'fold':f,'max_abs_prediction_difference':float(diff.max()),'n':len(diff)})
    return pd.concat(frames,ignore_index=True),audit


def run(prepared,*,fresh_fit=False):
    tables=[]
    out=prepared.out
    for model in ('B1','B5','M1','M1-W','M2'):
        frame,audit=replay(prepared,model)
        frame.to_parquet(out/'cache'/f'parent_{model}_cal_score.parquet',index=False)
        tables.extend(audit)
    # R1's immutable saved path has no calibration origins; its new F6 run supplies them.
    state=joblib.load(prepared.root/'outputs/phase_c/models/chronos_paths.joblib')
    r1=prepared.baselines[prepared.baselines.model.eq('R1')]
    for (h,f),part in r1.groupby(['horizon','fold']):
        values=np.array([state['paths'][o]['pred'][h-1] for o in part.origin])
        tables.append({'model':'R1','horizon':h,'fold':f,'max_abs_prediction_difference':float(np.max(np.abs(values-part.pred.to_numpy()))),'n':len(part)})
    pd.DataFrame(tables).to_csv(out/'tables/parent_model_replay.csv',index=False)
    record={'saved_model_replay':tables,'fresh_refit_scope':'not_requested',
            'historical_final_artifact_read':False,'holdout_read':False}
    if fresh_fit:
        c=prepared.contexts[(4,0)]
        origins=prepared.origins(4,0,'score')
        cfg=json.loads((prepared.root/'configs/phase_c.json').read_text(encoding='utf-8'))
        lock=json.loads((prepared.root/'outputs/phase_c/logs/model_config_lock.json').read_text(encoding='utf-8'))
        comparisons=[]
        for model in ('B5','M1','M1-W','M2'):
            tick=perf_counter()
            if model=='B5':
                fit=fit_kalman(prepared.history.power,c['fit'])
                pred=predict_kalman(fit,prepared.history.power,origins,4)
            else:
                family='tcn' if model=='M2' else 'lgbm'
                candidate={'id':lock[family]['selected_id'],**lock[family]['selected_config']}
                fit,_,_=training._fit_family(family,c,cfg,candidate,peak_weight=2.0 if model=='M1-W' else 1.0)
                pred=training._predict_family(family,fit,{**c,'score':origins},'score',cfg)
            joblib.dump(fit,out/'models'/f'F0-fresh-{model}-h04-f0.joblib')
            original=prepared.baselines[prepared.baselines.model.eq(model)&prepared.baselines.horizon.eq(4)&prepared.baselines.fold.eq(0)].sort_values('origin')
            comparisons.append({'model':model,'horizon':4,'fold':0,
                'max_abs_prediction_difference':float(np.max(np.abs(np.asarray(pred)-original.pred.to_numpy()))),
                'seconds':perf_counter()-tick})
        record.update(fresh_refit_scope='h4 fold0 representative cell per fitted family',fresh_refit=comparisons)
    write_json(out/'logs/parent_reproduction.json',record)
    return record
