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
from phase_f.support import UnsupportedConfiguration, validate_support


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
    from phase_f.runtime import activate_optional_dependencies
    activate_optional_dependencies(prepared.root)
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
            if adapter=='neural':cfg.setdefault('device','cuda')
            tick=perf_counter()
            bundle=module.fit_model(spec['kind'],prepared.history,context,cfg)
            seconds=perf_counter()-tick
            if bundle.get('stop_MAE',bundle.get('stop_mae')) is None:
                stop_prediction=module.predict_model(bundle,prepared.history,context['stop'],h)
                stop_prediction,_=_fallback(stop_prediction,context,context['stop'])
                bundle['stop_MAE']=float(np.mean(np.abs(stop_prediction-context['y'].loc[context['stop']].to_numpy(float))))
            saved={'identity':identity,'bundle':bundle,'train_seconds':seconds}
            tmp=cache.with_suffix('.tmp');joblib.dump(saved,tmp);tmp.replace(cache)
        if complete_prediction_cache(chunk,chunk.with_suffix('.json')):
            manifest=chunk.with_suffix('.json')
            entry=json.loads(manifest.read_text(encoding='utf-8'))
            if entry['identity']!=identity or entry['sha256']!=sha256(chunk):
                raise RuntimeError('Cached chunk identity/hash mismatch')
            frame=pd.read_parquet(chunk)
            frames.append(frame)
            audits.append({**entry['audit'],'cache_reused':True})
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
        cell_audit={'horizon':h,'fold':f,'train_seconds':seconds,'inference_seconds':infer,
                       'stop_MAE':bundle.get('stop_MAE',bundle.get('stop_mae')),
                       'future_perturbation_max_abs_difference':0,'weekly_fallback_rows':fallback_count}
        if adapter=='neural':
            cell_audit.update(determinism_mode=bundle['determinism_mode'],
                              nondeterministic_operations=bundle['nondeterministic_operations'])
        audits.append(cell_audit)
        write_json(chunk.with_suffix('.json'),{'identity':identity,'sha256':sha256(chunk),'audit':cell_audit})
        write_json(output/'cell_audit.json',audits)
        if f==2:
            print(f"{spec['id']} h{h}: 3 folds saved; last fit {seconds:.2f}s, infer {infer:.2f}s",flush=True)
    return pd.concat(frames,ignore_index=True),{'cells':audits,'leakage_test':'passed',
           'nonfinite_policy':'causal weekly target anchor; count recorded per cell'}


def base_frame(prepared,exp_id):
    row=Registry(prepared.root).read(exp_id)
    if row is None or row['status']!='completed':raise ValueError('Incomplete ensemble parent')
    cfg=json.loads(row['config_json'])
    if cfg.get('adapter')=='baseline':
        model=cfg['baseline']
        path=prepared.out/'cache'/f'parent_{model}_cal_score.parquet'
        if not path.exists():raise ValueError('Parent cal replay unavailable')
        frame=pd.read_parquet(path);frame['model']=exp_id
    else:
        path=prepared.out/'predictions'/f'{exp_id}.parquet'
        audit=json.loads((prepared.out/'logs'/f'{exp_id}_audit.json').read_text(encoding='utf-8'))
        if sha256(path)!=audit['prediction_sha256']:raise ValueError('Parent prediction changed')
        frame=pd.read_parquet(path)
    if 'cal_provenance' not in frame:frame['cal_provenance']=np.where(frame.role.eq('cal'),'fit_only','not_cal')
    return frame


def run_ensemble(prepared,spec):
    from phase_f.ensemble import combine_predictions
    inputs={exp_id:base_frame(prepared,exp_id) for exp_id in spec['bases']}
    tick=perf_counter()
    frame,audit=combine_predictions(inputs,spec['method'],spec['id'],params=spec.get('params',{}))
    seconds=perf_counter()-tick
    # Perturb every score label: fitted weights and every prediction must be identical.
    changed={key:part.copy() for key,part in inputs.items()}
    for part in changed.values():part.loc[part.role.eq('score'),'y']=1e8
    probe,_=combine_predictions(changed,spec['method'],spec['id'],params=spec.get('params',{}))
    if not np.array_equal(frame.pred.to_numpy(),probe.pred.to_numpy()):
        raise AssertionError('Ensemble learned from score labels')
    frame['train_seconds']=seconds;frame['train_seconds_run_id']=spec['id']+'_cal_fit_and_combine'
    frame['inference_seconds']=0.;frame['inference_seconds_run_id']=spec['id']+'_included_above'
    audit.update(leakage_test='passed',score_label_perturbation_max_abs_difference=0,
                 base_prediction_hashes={key:sha256(prepared.out/'predictions'/f'{key}.parquet') for key in inputs},
                 timing_scope='combination only; base model costs are listed in their own experiments')
    return frame,audit


def run_seed_ensemble(prepared,spec,registry):
    bases=[]
    for seed in spec['seeds']:
        cfg={**spec,'id':f"{spec['parent']}-seed{seed}",'adapter':'regression','seed':seed}
        cfg.pop('seeds',None)
        # Reused across 5/10/20-seed ensembles with an identical configuration.
        cfg['parent']=spec['parent']
        execute(prepared,cfg,registry)
        bases.append(cfg['id'])
    return run_ensemble(prepared,{**spec,'bases':bases,'method':'mean','params':{}})


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
    week_ci=pd.concat([paired_ci(frame,prepared.baselines[prepared.baselines.model.eq(b)],
                                arm='EXPLORE',block='week') for b in ('B5','M1','R1')],ignore_index=True)
    week_ci.to_csv(dest/'explore_pairwise_week_ci.csv',index=False)
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
    write_json(dest/'explore_manifest.json',{
        'prediction_sha256':sha256(prepared.out/'predictions'/f"{spec['id']}.parquet"),
        'tables':{p.name:sha256(p) for p in sorted(dest.glob('explore_*.csv'))},
        'registry_metrics':fields,'metrics_source_sha256':sha256(prepared.root/'phase_f/metrics.py'),
        'split_sha256':prepared.split_lock['lock_sha256'],
        'baseline_prediction_sha256':{f'outputs/phase_c/predictions/{name}.parquet':sha256(prepared.root/'outputs/phase_c/predictions'/f'{name}.parquet') for name in ('main_predictions','reference_predictions')},
        'historical_final_artifact_read':False,'holdout_read':False})
    registry.update(spec['id'],**fields,explore_queries=previous['explore_queries']+1)
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
        validate_support(prepared,spec)
        import psutil
        rss_before=psutil.Process().memory_info().rss
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
        elif spec['adapter']=='other_foundation':
            from phase_f.models.other_foundation import run
            frame,audit=run(prepared,spec)
        elif spec['adapter']=='ensemble':
            frame,audit=run_ensemble(prepared,spec)
        elif spec['adapter']=='seed_ensemble':
            frame,audit=run_seed_ensemble(prepared,spec,registry)
        else:
            frame,audit=run_adapter(prepared,spec)
        if 'cal_provenance' not in frame:
            frame['cal_provenance']=np.where(frame.role.eq('cal'),'fit_only','not_cal')
        prepared.validate_predictions(frame)
        write_predictions(path,frame)
        write_json(audit_path,{**audit,
            'config_hash':config_hash(spec),'prediction_sha256':sha256(path),
            'historical_final_artifact_read':False,'holdout_read':False})
        result=summarize_experiment(prepared,spec,frame,registry)
        registry.update(spec['id'],wall_sec=perf_counter()-started,leakage_test=audit.get('leakage_test','passed'))
        from phase_f.metrics import _cost
        score=frame[frame.role.eq('score')]
        memory=psutil.Process().memory_info()
        registry.update(spec['id'],train_sec=_cost(score,'train_seconds'),
                        infer_sec=_cost(score,'inference_seconds'),
                        rss_before_bytes=rss_before,rss_after_bytes=memory.rss,
                        process_lifetime_peak_rss_bytes=getattr(memory,'peak_wset',None),
                        memory_measurement='process RSS snapshots and Windows lifetime peak; not isolated model peak')
        cells=audit.get('cells',[])
        values=[c['stop_MAE'] for c in cells if c.get('stop_MAE') is not None]
        if values:
            registry.update(spec['id'],stop_MAE=float(np.mean(values)),stop_cells=len(values))
        registry.update(spec['id'],status='completed')
        print(spec['id'],json.dumps(result),flush=True)
        return registry.read(spec['id'])
    except Exception as exc:
        # Raw traceback is local and ignored; public record excludes personal paths.
        trace=traceback.format_exc().replace(str(prepared.root),'<repo>')
        (prepared.out/'logs'/f"{spec['id']}_failure.log").write_text(trace,encoding='utf-8')
        reason=str(exc).replace(str(prepared.root),'<repo>')
        status='unsupported' if isinstance(exc,UnsupportedConfiguration) else 'failed'
        fields={'unsupported_evidence':exc.evidence} if status=='unsupported' else {}
        registry.update(spec['id'],status=status,wall_sec=perf_counter()-started,
                        error_type=type(exc).__name__,error=reason,**fields)
        print(f"{status.upper()} {spec['id']}: {type(exc).__name__}: {reason}",flush=True)
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


def dispatch_stage(prepared, stage, *, retry=False):
    from phase_f.workflow import run_all,run_stage
    requested = 'all' if stage == 'all' else int(stage or 0)
    if requested != 'all' and requested not in range(5):
        raise ValueError('Phase F stage must be 0..4 or all')
    stages = range(5) if requested == 'all' else (requested,)
    def done(number):
        path = prepared.out/'logs/workflow'/f'stage_{number}.json'
        return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') == 'completed'
    pending = any(not done(number) for number in stages)
    status_path = prepared.out/'logs/driver_status.json'
    if pending:
        write_json(status_path,{'status':'running','stage':requested,'pid':os.getpid(),
            'started_at':now(),'full_phase_complete':False,
            'stage_details':'logs/workflow/stage_*.json',
            'historical_final_artifact_read':False,'holdout_read':False})
    try:
        result = run_all(prepared,retry=retry) if requested == 'all' else run_stage(prepared,requested,retry=retry)
    except Exception as exc:
        write_json(status_path,{'status':'failed','stage':requested,'at':now(),
            'error_type':type(exc).__name__,'full_phase_complete':False,
            'historical_final_artifact_read':False,'holdout_read':False})
        raise
    if pending and requested not in ('all',4):
        write_json(status_path,{'status':'completed','stage':requested,'at':now(),
            'full_phase_complete':False,'historical_final_artifact_read':False,'holdout_read':False})
    elif not pending and requested in ('all',4):
        prior = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        if prior.get('status') != 'completed' or prior.get('full_phase_complete') is not True:
            # Dispatch has verified the finished stage and final report. Recover
            # an interruption between the stage checkpoint and driver marker.
            write_json(status_path,{'status':'completed','stage':'all','at':now(),
                'full_phase_complete':True,'recovered_from_completed_checkpoints':True,
                'historical_final_artifact_read':False,'holdout_read':False})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage')
    parser.add_argument('--exp')
    parser.add_argument('--tier',type=int)
    parser.add_argument('--retry',action='store_true')
    args=parser.parse_args()
    root=Path.cwd()
    from phase_f.runtime import activate_optional_dependencies
    activate_optional_dependencies(root)
    if not (root/'outputs/phase_f/logs/user_approval.json').is_file():
        raise RuntimeError('Approved Phase F contract is required')
    prepared=Prepared(root);registry=Registry(root)
    if not args.exp:
        return dispatch_stage(prepared,args.stage,retry=args.retry)
    prior=registry.read(args.exp)
    if prior is not None:
        cfg=json.loads(prior['config_json'])
    else:
        possibilities=[s for stage in range(4) for s in catalog(stage)]
        found=[s for s in possibilities if s['id']==args.exp]
        if len(found)!=1:raise ValueError('Unknown experiment ID; adaptive experiments must first be planned in their stage')
        cfg=found[0]
    if args.tier is not None and cfg.get('tier')!=args.tier:
        raise ValueError('Requested tier differs from registered configuration')
    execute(prepared,cfg,registry,retry=args.retry)
    return


if __name__=='__main__':
    main()
