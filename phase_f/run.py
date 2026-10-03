"""Resumable Phase F driver. EXPLORE is the only default evaluation arm."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from phase_f.harness import Prepared, make_frame, KEY, BASELINES
from phase_f.registry import Registry, write_json, config_hash, sha256, now
from phase_f.metrics import evaluate, paired_ci


def write_predictions(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    frame.to_parquet(temp,index=False)
    temp.replace(path)


def complete_prediction_cache(path,audit_path):
    """Quarantine a partial transaction; never strand resume on a missing audit."""
    if not path.exists():return False
    if audit_path.exists():return True
    suffix=sha256(path)[:12]
    orphan=path.with_name(path.stem+f'.orphan-{suffix}'+path.suffix)
    counter=0
    while orphan.exists():
        counter+=1
        orphan=path.with_name(path.stem+f'.orphan-{suffix}-{counter}'+path.suffix)
    path.rename(orphan)
    return False


def _fallback(pred,context,origins):
    values=np.asarray(pred,dtype=float).copy()
    missing=~np.isfinite(values)
    # Predeclared causal fallback preserves the complete comparison cohort.
    values[missing]=context['x'].loc[origins,'slot7d'].to_numpy(float)[missing]
    return values,int(missing.sum())


def run_adapter(prepared,spec):
    adapter=spec['adapter']
    module=importlib.import_module('phase_f.models.'+adapter)
    identity=config_hash({'spec':spec,'source':sha256(module.__file__),
        'features':sha256(prepared.root/'phase_f/features_ext.py'),
        'split':prepared.split_lock['lock_sha256']})
    output=prepared.out/'models'/spec['id']
    output.mkdir(parents=True,exist_ok=True)
    frames=[]
    audits=[]
    for (h,f),context in prepared.contexts.items():
        cache=output/f'h{h:02d}_f{f}.joblib'
        chunk=prepared.out/'predictions/chunks'/spec['id']/f'h{h:02d}_f{f}.parquet'
        if cache.exists():
            saved=joblib.load(cache)
            if saved['identity']!=identity:
                raise RuntimeError('Changed code/config for cached experiment: use new experiment ID')
            bundle=saved['bundle']
            seconds=saved['train_seconds']
        else:
            cfg={**spec,**spec.get('params',{}),'horizon':h,
                 'checkpoint_dir':str(output/f'h{h:02d}_f{f}_seeds')}
            tick=perf_counter()
            bundle=module.fit_model(spec['kind'],prepared.history,context,cfg)
            seconds=perf_counter()-tick
            saved={'identity':identity,'bundle':bundle,'train_seconds':seconds}
            tmp=cache.with_suffix('.tmp');joblib.dump(saved,tmp);tmp.replace(cache)
        if chunk.exists():
            frame=pd.read_parquet(chunk)
            frames.append(frame)
            audits.append({'horizon':h,'fold':f,'train_seconds':seconds,'cache_reused':True})
            continue
        # Test a real fitted model for each fold/horizon, using identical origins.
        probe=prepared.origins(h,f,'score')[:1]
        origin=probe.max()
        altered=prepared.history.copy()
        future=altered.index>origin
        altered.loc[future,'power']=np.random.default_rng(42).normal(10000,500,int(future.sum()))
        a=module.predict_model(bundle,prepared.history,probe,h)
        b=module.predict_model(bundle,altered,probe,h)
        if not np.array_equal(a,b,equal_nan=True):
            raise AssertionError('Future perturbation changed a prediction')
        cell=[]
        fallback_count=0
        infer=0.0
        for role in ('cal','score'):
            origins=prepared.origins(h,f,role)
            tick=perf_counter()
            extra={}
            if adapter=='neural' and bundle.get('config',{}).get('loss')=='quantile':
                prediction,quantiles=module.predict_model(bundle,prepared.history,origins,h,return_quantiles=True)
                extra={f'q{int(round(float(q)*100)):02d}':v for q,v in quantiles.items()}
            elif adapter=='regression' and spec['kind']=='two_stage':
                detail=module.predict_model(bundle,prepared.history,origins,h,return_details=True)
                prediction=detail['pred'];extra={'p_peak':detail['p_peak']}
            else:
                prediction=module.predict_model(bundle,prepared.history,origins,h)
            elapsed=perf_counter()-tick;infer+=elapsed
            prediction,nfallback=_fallback(prediction,context,origins);fallback_count+=nfallback
            cell.append(make_frame(context,origins,h,f,prediction,spec['id'],role,
                train_seconds=seconds,inference_seconds=elapsed,**extra))
        frame=pd.concat(cell,ignore_index=True)
        write_predictions(chunk,frame)
        frames.append(frame)
        audits.append({'horizon':h,'fold':f,'train_seconds':seconds,'inference_seconds':infer,
                       'future_perturbation_max_abs_difference':0,'weekly_fallback_rows':fallback_count})
        write_json(output/'cell_audit.json',audits)
        if f==2:
            print(f"{spec['id']} h{h}: 3 folds saved; last fit {seconds:.2f}s, infer {infer:.2f}s",flush=True)
    return pd.concat(frames,ignore_index=True),{'cells':audits,'leakage_test':'passed',
           'nonfinite_policy':'causal weekly target anchor; count recorded per cell'}


def summarize_experiment(prepared,spec,frame,registry):
    prepared.validate_predictions(frame)
    tables=evaluate(frame,arm='EXPLORE')
    dest=prepared.out/'tables'/spec['id'];dest.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(dest/f'explore_{name}.csv',index=False)
    cis=[]
    for baseline in ('B5','M1','R1'):
        cis.append(paired_ci(frame,prepared.baselines[prepared.baselines.model.eq(baseline)],arm='EXPLORE'))
    ci=pd.concat(cis,ignore_index=True)
    ci.to_csv(dest/'explore_pairwise_ci.csv',index=False)
    auc=tables['auc'].set_index('dataset')
    pooled=tables['pooled'];h16=pooled[pooled.dataset.eq('D1')&pooled.horizon.eq(16)].iloc[0]
    fields={'explore_AUC_MAE':float(auc.loc['D1','AUC_MAE']),
            'explore_AUC_PeakMAE':float(auc.loc['D1','AUC_PeakMAE']),
            'D2_explore_AUC_MAE':float(auc.loc['D2','AUC_MAE']),
            'D2_explore_AUC_PeakMAE':float(auc.loc['D2','AUC_PeakMAE']),
            'explore_h16_MAE':float(h16.MAE),'explore_h16_Peak':float(h16.Peak_MAE),
            'explore_PredPeakMAE':float(auc.loc['D1','AUC_PredPeakMAE'])}
    fold_table=tables['fold']
    for fold in (0,1,2):
        fields[f'fold{fold}_MAE']=float(fold_table[fold_table.dataset.eq('D1')&fold_table.fold.eq(fold)].MAE.mean())
    previous=registry.read(spec['id'])
    registry.update(spec['id'],**fields,status='completed',explore_queries=previous['explore_queries']+1)
    return fields


def execute(prepared,spec,registry,*,retry=False):
    row=registry.register(spec)
    if row['status']=='completed' or (row['status'] in ('failed','rejected','unsupported') and not retry):
        return row
    registry.update(spec['id'],status='run',started_at=now(),pid=os.getpid())
    path=prepared.out/'predictions'/f"{spec['id']}.parquet"
    audit_path=prepared.out/'logs'/f"{spec['id']}_audit.json"
    started=perf_counter()
    try:
        if complete_prediction_cache(path,audit_path):
            audit=json.loads(audit_path.read_text(encoding='utf-8'))
            if audit.get('config_hash')!=config_hash(spec) or audit.get('prediction_sha256')!=sha256(path):
                raise RuntimeError('Prediction/audit identity mismatch')
            frame=pd.read_parquet(path)
        elif spec['adapter']=='baseline':
            source=prepared.baselines if spec['baseline'] in BASELINES else prepared.legacy_aux
            frame=source[source.model.eq(spec['baseline'])].copy()
            frame['model']=spec['id']
            audit={'source':'sealed Phase C predictions','baseline':spec['baseline'],
                   'retrained':False,'leakage_test':'parent_sealed_contract'}
        elif spec['adapter']=='foundation':
            from phase_f.models.foundation import run
            frame,audit=run(prepared,spec)
            audit['leakage_test']='passed'
        elif spec['adapter']=='multihorizon':
            from phase_f.models.multihorizon import run
            frame,audit=run(prepared,spec)
            audit['leakage_test']='passed'
        else:
            frame,audit=run_adapter(prepared,spec)
        prepared.validate_predictions(frame)
        write_predictions(path,frame)
        write_json(audit_path,{**audit,
            'config_hash':config_hash(spec),'prediction_sha256':sha256(path),
            'historical_final_artifact_read':False,'holdout_read':False})
        result=summarize_experiment(prepared,spec,frame,registry)
        registry.update(spec['id'],wall_sec=perf_counter()-started,leakage_test=audit.get('leakage_test','passed'))
        print(spec['id'],json.dumps(result),flush=True)
        return registry.read(spec['id'])
    except Exception as exc:
        # Raw traceback is local and ignored; public record excludes personal paths.
        trace=traceback.format_exc().replace(str(prepared.root),'<repo>')
        (prepared.out/'logs'/f"{spec['id']}_failure.log").write_text(trace,encoding='utf-8')
        reason=str(exc).replace(str(prepared.root),'<repo>')
        registry.update(spec['id'],status='failed',wall_sec=perf_counter()-started,
                        error_type=type(exc).__name__,error=reason)
        print(f"FAILED {spec['id']}: {type(exc).__name__}: {reason}",flush=True)
        return registry.read(spec['id'])


def catalog(stage):
    from phase_f.models.statistical import configurations as statistical_configs
    from phase_f.models.foundation import configurations as foundation_configs
    if stage==0:
        return ([{'id':f'F0-1-{b}','family':'F0','tier':0,'adapter':'baseline','baseline':b} for b in BASELINES]
            +[{'id':'F0-3-CBL','family':'F0','tier':0,'adapter':'baseline','baseline':'B2'}]
            +[{**s,'adapter':'statistical','tier':0,'nonfinite_policy':'weekly_anchor'}
                for s in statistical_configs() if s['id'].startswith('F0')])
    if stage==1:
        from phase_f.features_ext import available_group_configs
        specs=[]
        for group in available_group_configs():
            if group['id'] in ('F1-10','F1-11'):continue
            for kind in ('ridge','lightgbm'):
                specs.append({'id':group['id']+'-'+kind,'family':'F1','tier':1,'adapter':'regression',
                    'kind':kind,'groups':list(group['groups']),'target':'direct','nonfinite_policy':'weekly_anchor'})
        specs.extend([{'id':'F1-core-'+kind,'family':'F1','tier':1,'adapter':'regression','kind':kind,
                       'groups':[],'target':'direct','nonfinite_policy':'weekly_anchor'} for kind in ('ridge','lightgbm')])
        specs.extend({**s,'adapter':'foundation'} for s in foundation_configs() if s['tier']==1)
        for length in (96,192,336,672,1344,2016,2688):
            for kind in ('dlinear','nlinear'):
                specs.append({'id':f'F5-2-{kind}-c{length}','family':'F5','tier':1,
                    'adapter':'neural','kind':kind,'context_length':length,'seeds':[42,43,44,45,46],
                    'nonfinite_policy':'weekly_anchor'})
        return specs
    if stage==2:
        return [{**s,'adapter':'statistical','tier':2,'nonfinite_policy':'weekly_anchor'}
                for s in statistical_configs() if s['id'].startswith('F4-1')]
    if stage==3:
        return [{**s,'adapter':'statistical','tier':3,'nonfinite_policy':'weekly_anchor'}
                for s in statistical_configs() if s['id'].startswith(('F4-2','F4-4'))]
    raise ValueError('Unsupported stage')


def stage_summary(prepared,stage,registry):
    frame=pd.read_csv(prepared.out/'registry.csv')
    done=frame[frame.status.eq('completed')].sort_values('explore_AUC_MAE')
    columns=['exp_id','explore_AUC_MAE','explore_AUC_PeakMAE','D2_explore_AUC_MAE','explore_h16_MAE']
    top=done[columns].head(10)
    markdown='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    markdown+='\n'.join('| '+' | '.join(str(v) if isinstance(v,str) else f'{v:.6f}' for v in row)+' |' for row in top.itertuples(index=False,name=None))
    text=(f'# Phase F Stage {stage} 실행 요약\n\nEXPLORE 개발 탐색 결과. CONFIRM 미평가, 최종 holdout 미사용.\n\n'
          +markdown
          +f'\n\n등록 {len(frame)}개, 완료 {len(done)}개, 실패 {int(frame.status.eq("failed").sum())}개.\n'
          +'\n다음 단계는 승인된 계획의 후속 계열과 제거 비교이며, Phase C/D 선정은 보존한다.\n')
    (prepared.out/f'STAGE_{stage}_summary.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='0')
    parser.add_argument('--exp')
    parser.add_argument('--tier',type=int)
    parser.add_argument('--retry',action='store_true')
    args=parser.parse_args()
    root=Path.cwd()
    if not (root/'outputs/phase_f/logs/user_approval.json').is_file():
        raise RuntimeError('Approved Phase F contract is required')
    prepared=Prepared(root);registry=Registry(root)
    if args.stage in ('0','all'):
        from phase_f.leakage_tests import run_oracle_negative_control
        run_oracle_negative_control(root)
    stages=range(4) if args.stage=='all' else [int(args.stage)]
    for stage in stages:
        specs=catalog(stage)
        if args.exp:specs=[s for s in specs if s['id']==args.exp]
        for spec in specs:registry.register(spec)
        for spec in specs:execute(prepared,spec,registry,retry=args.retry)
        if specs:stage_summary(prepared,stage,registry)
    write_json(prepared.out/'logs/driver_status.json',{'status':'requested_stage_completed',
        'stage':args.stage,'exp':args.exp,'at':now(),'historical_final_artifact_read':False,'holdout_read':False,
        'full_phase_complete':False})


if __name__=='__main__':
    main()
