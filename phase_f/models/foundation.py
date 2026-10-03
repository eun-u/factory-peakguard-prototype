"""Chronos inference/fit on causal windows; calibrated labels stay outside fit."""
from __future__ import annotations
import json
import importlib.util
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from src.holidays import HOLIDAYS_2021
from phase_f.harness import make_frame
from phase_f.registry import config_hash, sha256, write_json

QUANTILES = [.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]


def require_finetune_mode(mode):
    if mode not in (None,'full','lora'):
        raise ValueError('Unknown finetuning mode')
    if mode=='lora':
        if importlib.util.find_spec('peft') is None:
            raise RuntimeError('LoRA requires PEFT; full-finetune fallback is forbidden')
        from peft import LoraConfig  # fail before fitting if installed package cannot import
        if LoraConfig is None:
            raise RuntimeError('Invalid PEFT installation')


def calendar(index):
    idx = pd.DatetimeIndex(index)
    hour = idx.hour.to_numpy()+idx.minute.to_numpy()/60
    day = idx.dayofweek.to_numpy()
    return {'hour_sin':np.sin(2*np.pi*hour/24).astype('float32'),
            'hour_cos':np.cos(2*np.pi*hour/24).astype('float32'),
            'dow_sin':np.sin(2*np.pi*day/7).astype('float32'),
            'dow_cos':np.cos(2*np.pi*day/7).astype('float32'),
            'holiday':idx.strftime('%Y-%m-%d').isin(HOLIDAYS_2021).astype('float32')}


def causal_input(history, origin, length, prediction_length=96, covariates=False):
    origin=pd.Timestamp(origin)
    power=history.power.loc[:origin].iloc[-int(length):].astype('float32')
    if power.empty or power.index[-1]!=origin:
        raise ValueError('Origin absent from observed history')
    values=power.to_numpy(copy=True)
    if not covariates:
        return values
    future=pd.date_range(origin+pd.Timedelta(minutes=15),periods=prediction_length,freq='15min')
    return {'target':values,'past_covariates':calendar(power.index),'future_covariates':calendar(future)}


def training_windows(history, context, role, length, prediction_length, covariates=False):
    """Fixed-length series: min_past=length leaves exactly one training cut.

    All future labels lie within the supplied fit/stop target set. Windows
    missing the required full context are excluded from TRAINING only.
    """
    index=history.index
    values=history.power.to_numpy(dtype='float32')
    allowed=set(pd.DatetimeIndex(context['target_time'].loc[context[role]]))
    result=[]
    for origin in context[role]:
        pos=index.get_loc(origin)
        if pos+1 < length or pos+prediction_length>=len(index):
            continue
        targets=index[pos+1:pos+1+prediction_length]
        if not all(t in allowed for t in targets):
            continue
        array=values[pos+1-length:pos+1+prediction_length].copy()
        if not np.isfinite(array[-prediction_length:]).all():
            continue
        if covariates:
            times=index[pos+1-length:pos+1+prediction_length]
            # Chronos fit treats named future covariates as available tasks.
            result.append({'target':array,'past_covariates':calendar(times),
                           'future_covariates':{name:None for name in calendar(times)}})
        else:
            result.append(array)
    if not result:
        raise ValueError(f'No complete {length}-context {role} window within fixed fold')
    return result


def point_from_quantiles(array, point):
    if point in ('median','native_mean'):
        # In chronos-forecasting 2.3.2 predict_quantiles mean is exactly q0.5.
        return array[...,QUANTILES.index(.5)]
    if point=='integrated_mean':
        q=np.array([0,*QUANTILES,1])
        extended=np.concatenate([array[...,:1],array,array[...,-1:]],axis=-1)
        return np.trapezoid(extended,q,axis=-1)
    return array[...,QUANTILES.index(float(point))]


def configurations():
    rows=[]
    for length in (512,1024,2048,4096,8192):
        for point in ('median','native_mean','integrated_mean'):
            rows.append({'id':f'F6-1-c{length}-{point}','family':'F6','tier':1,
                         'kind':'chronos2','context_length':length,'point':point})
    for length in (512,2048,4096):
        rows.append({'id':f'F6-2-c{length}-calendar','family':'F6','tier':1,
                     'kind':'chronos2','context_length':length,'point':'median','covariates':True})
    for q in (.55,.6,.65):
        rows.append({'id':f'F6-3-c2048-q{q}','family':'F6','tier':1,
                     'kind':'chronos2','context_length':2048,'point':q})
    for mode in ('full','lora'):
        for length in (512,2048,8192):
            for lr in (1e-6,1e-5):
                rows.append({'id':f'F6-4-{mode}-c{length}-lr{lr:g}','family':'F6','tier':2,
                    'kind':'chronos2','context_length':length,'point':'median','finetune':mode,
                    'learning_rate':lr,'num_steps':500})
    return rows


def run(prepared,spec):
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(2)
    torch.manual_seed(int(spec.get('seed',42)))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    root,out,history=prepared.root,prepared.out,prepared.history
    info=json.loads((root/'outputs/phase_c/logs/chronos_model_download.json').read_text(encoding='utf-8'))
    local_model=root/'outputs/phase_c/models/hf_cache/models--amazon--chronos-2/snapshots'/info['revision']
    if not local_model.is_dir():
        raise FileNotFoundError('Pinned local Chronos snapshot missing')
    length=int(spec.get('context_length',2048))
    pred_length=int(spec.get('prediction_length',96))
    cov=bool(spec.get('covariates',False))
    fit_mode=spec.get('finetune')
    require_finetune_mode(fit_mode)
    cache_spec={k:v for k,v in spec.items() if k not in ('id','parent','family','tier','point','note')}
    cache_spec.update(model_revision=info['revision'],quantiles=QUANTILES,
                      source_sha256=sha256(Path(__file__)),split_sha256=prepared.split_lock['lock_sha256'])
    cache_id=config_hash(cache_spec)[:20]
    work=out/'models'/f'chronos_{cache_id}'
    work.mkdir(parents=True,exist_ok=True)
    folds=(0,1,2) if fit_mode else (-1,)
    frames=[]
    audit={'model_revision':info['revision'],'cache_id':cache_id,'input_spec':cache_spec,
           'native_mean_is_median':True,'integrated_mean_tail_policy':'flat tails outside q.01..q.99',
           'fit_cells':[],'historical_final_artifact_read':False,'holdout_read':False}
    for fold_group in folds:
        contexts={k:c for k,c in prepared.contexts.items() if fold_group==-1 or k[1]==fold_group}
        origins=pd.DatetimeIndex(sorted(set().union(*[
            set(prepared.origins(h,f,role)) for h,f in contexts for role in ('cal','score')])) )
        cache=work/f'paths_f{fold_group}.joblib'
        state=joblib.load(cache) if cache.exists() else {'cache_id':cache_id,'paths':{},'origin_seconds':{},'seconds':0.0,'train_seconds':0.0}
        if state['cache_id']!=cache_id:
            raise RuntimeError('Inference cache identity mismatch')
        todo=[o for o in origins if o not in state['paths']]
        if todo:
            pipe=Chronos2Pipeline.from_pretrained(str(local_model),device_map=device,torch_dtype=torch.float32)
            if length>pipe.model_context_length:
                raise ValueError('Requested context exceeds model supported context; no silent truncation')
            if fit_mode:
                c=contexts[(16,fold_group)]
                fit_dir=work/f'finetune_f{fold_group}'
                checkpoint=fit_dir/'finetuned-ckpt'
                if checkpoint.is_dir():
                    pipe=Chronos2Pipeline.from_pretrained(str(checkpoint),device_map=device,torch_dtype=torch.float32)
                else:
                    train=training_windows(history,c,'fit',length,16,cov)
                    valid=training_windows(history,c,'stop',length,16,cov)
                    from transformers import EarlyStoppingCallback
                    tick=perf_counter()
                    pipe=pipe.fit(inputs=train,prediction_length=16,validation_inputs=valid,
                        finetune_mode=fit_mode,context_length=length,min_past=length,
                        learning_rate=float(spec.get('learning_rate',1e-6)),num_steps=int(spec.get('num_steps',500)),
                        batch_size=int(spec.get('fit_batch_size',16)),output_dir=fit_dir,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                        eval_strategy='steps',eval_steps=50,save_strategy='steps',save_steps=50,
                        load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
                        save_total_limit=2,report_to='none',dataloader_num_workers=0,
                        seed=int(spec.get('seed',42)),disable_tqdm=True)
                    state['train_seconds']=perf_counter()-tick
                    del train,valid
            batch_size=int(spec.get('batch_size',16 if cov or length>2048 else 32))
            # Mandatory family-level future perturbation using the identical pipeline.
            check_origin=todo[0]
            altered=history.copy()
            future=altered.index>check_origin
            altered.loc[future,'power']=np.random.default_rng(42).normal(10000,1000,int(future.sum()))
            a=causal_input(history,check_origin,length,pred_length,cov)
            b=causal_input(altered,check_origin,length,pred_length,cov)
            with torch.inference_mode():
                qa,_=pipe.predict_quantiles([a],prediction_length=pred_length,quantile_levels=QUANTILES,context_length=length)
                qb,_=pipe.predict_quantiles([b],prediction_length=pred_length,quantile_levels=QUANTILES,context_length=length)
            difference=float(torch.max(torch.abs(qa[0]-qb[0])).cpu())
            if difference!=0:
                raise AssertionError(f'Future perturbation changed predictions: {difference}')
            state['future_perturbation_max_abs_difference']=difference
            for start in range(0,len(todo),batch_size):
                batch=todo[start:start+batch_size]
                inputs=[causal_input(history,o,length,pred_length,cov) for o in batch]
                if device=='cuda':torch.cuda.synchronize()
                tick=perf_counter()
                with torch.inference_mode():
                    quantiles,_=pipe.predict_quantiles(inputs,prediction_length=pred_length,
                        quantile_levels=QUANTILES,context_length=length,batch_size=max(batch_size,256 if cov else batch_size))
                if device=='cuda':torch.cuda.synchronize()
                seconds=perf_counter()-tick
                state['seconds']+=seconds
                for origin,q in zip(batch,quantiles):
                    array=q.detach().cpu().numpy()
                    if array.shape!=(1,pred_length,len(QUANTILES)) or not np.isfinite(array).all():
                        raise ValueError('Invalid quantile forecast shape/values')
                    state['paths'][origin]=array[0]
                    state['origin_seconds'][origin]=seconds/len(batch)
                temp=cache.with_suffix('.tmp')
                joblib.dump(state,temp);temp.replace(cache)
                if start%(batch_size*10)==0:
                    print(f"{spec['id']} fold {fold_group}: {start+len(batch)}/{len(todo)} origins",flush=True)
            del pipe
            if device=='cuda':torch.cuda.empty_cache()
        audit['fit_cells'].append({'fold':fold_group,'train_seconds':state['train_seconds'],
            'inference_seconds':state['seconds'],'origins':len(origins),
            'future_perturbation_max_abs_difference':state.get('future_perturbation_max_abs_difference')})
        for (h,f),c in contexts.items():
            for role in ('cal','score'):
                idx=prepared.origins(h,f,role)
                q=np.stack([state['paths'][o][h-1] for o in idx])
                prediction=point_from_quantiles(q,spec.get('point','median'))
                monotone=np.maximum.accumulate(q,axis=1)
                prob=np.array([1-np.interp(c['tau'],row,QUANTILES,left=.01,right=.99) for row in monotone])
                frame=make_frame(c,idx,h,f,prediction,spec['id'],role,
                    train_seconds=state['train_seconds'],
                    train_seconds_run_id=f'{cache_id}_fit_f{fold_group}',
                    inference_seconds=np.asarray([state['origin_seconds'][o] for o in idx]),
                    inference_seconds_run_id=[f'{cache_id}_f{fold_group}_{o.isoformat()}' for o in idx],
                    q10=q[:,QUANTILES.index(.1)],q50=q[:,QUANTILES.index(.5)],
                    q90=q[:,QUANTILES.index(.9)],q95=q[:,QUANTILES.index(.95)],p_peak=prob)
                frames.append(frame)
    write_json(out/'logs'/f"{spec['id']}_foundation.json",audit)
    return pd.concat(frames,ignore_index=True),audit
