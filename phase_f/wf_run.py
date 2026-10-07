"""Transactional runner for the revised weekly protocol; CONFIRM is gated separately."""
from __future__ import annotations
import argparse
import json
import os
import traceback
from pathlib import Path
from time import perf_counter
import pandas as pd
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_registry import WFRegistry
from phase_f.wf_evaluation import prediction_path, load_predictions, baselines_ready, summarize, verify_metrics


def normalize_spec(prepared, spec):
    spec=dict(spec)
    # R1 is the pinned zero-shot Chronos recipe, refitted/replayed on each week.
    if spec.get('adapter')=='baseline' and spec.get('baseline')=='R1':
        spec={k:v for k,v in spec.items() if k!='baseline'}
        spec.update(adapter='foundation',context_length=2048,point='median',finetune=False)
    if spec.get('adapter')=='foundation' and not spec.get('finetune'):
        spec['verify_determinism']=True
    if spec.get('adapter')=='seed_ensemble' and 'parent_spec' not in spec:
        row=WFRegistry(prepared.root).read(spec['parent'])
        if not row:raise ValueError('Seed ensemble parent has not been frozen')
        spec['parent_spec']=json.loads(row['config_json'])
    prior=WFRegistry(prepared.root).read(spec['id'])
    if prior and prior['config_hash']!=config_hash(spec):
        spec['id']=spec['id']+'-wf'+config_hash(spec)[:10]
    return spec


def cache_identity(prepared,spec):
    adapter=spec.get('adapter')
    source=['wf_models.py','wf_harness.py','wf_run.py','harness.py','features_ext.py','run.py']
    model={'regression':'regression.py','statistical':'statistical.py','neural':'neural.py',
        'foundation':'foundation.py','multihorizon':'multihorizon.py','other_foundation':'other_foundation.py'}.get(adapter)
    if model:source.append('models/'+model)
    if adapter in ('ensemble','seed_ensemble'):source.append('ensemble.py')
    return config_hash({'spec':spec,'sources':{p:sha256(prepared.root/'phase_f'/p) for p in source},
        'split':prepared.split_lock['lock_sha256'],'raw':prepared.seal['raw_sha256']})


def save_mean(prepared,spec,frame,audit,arm='EXPLORE'):
    path=prediction_path(prepared,spec['id'],arm)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(path)
    entry={'identity':cache_identity(prepared,spec),'config_hash':config_hash(spec),
        'sha256':sha256(path),'prediction_sha256':sha256(path),'arm':arm,'audit':audit,
        'holdout_read':False,'historical_final_artifact_read':False}
    write_json(path.with_suffix('.json'),entry)
    write_json(prepared.out/'logs/model_audits'/arm/f"{spec['id']}.json",entry)
    return entry


def execute(prepared,spec,registry=None,*,retry=False,score=True):
    from phase_f.wf_models import run
    from phase_f.run import complete_prediction_cache
    from phase_f.support import validate_support,UnsupportedConfiguration
    registry=registry or WFRegistry(prepared.root)
    row=registry.register(spec)
    if (prepared.out/'logs/confirm_reservation.json').exists():
        raise RuntimeError('Search is closed after the CONFIRM reservation')
    if row['status']=='completed':
        path=prediction_path(prepared,spec['id'])
        saved=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if saved.get('identity')!=cache_identity(prepared,spec):
            raise RuntimeError('Completed model source changed; use a new experiment ID')
        verify_metrics(prepared,row)
        return row
    if row['status'] in ('failed','unsupported','rejected') and not retry:return row
    path=prediction_path(prepared,spec['id'])
    started=perf_counter();registry.update(spec['id'],status='running',started_at=now())
    try:
        from phase_f.wf_models import _arm_view
        validate_support(_arm_view(prepared,'EXPLORE'),spec)
        if complete_prediction_cache(path,path.with_suffix('.json')):
            saved=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if saved['identity']!=cache_identity(prepared,spec):
                raise RuntimeError('Changed prediction identity; use a new experiment ID')
            frame=load_predictions(prepared,spec['id']);audit=saved['audit']
        else:
            frame,audit=run(prepared,spec,arm='EXPLORE')
            save_mean(prepared,spec,frame,audit)
        registry.update(spec['id'],status='prediction_ready',n_seeds=audit['n_seeds'],
            stop_MAE=audit.get('stop_MAE'),stop_cells=audit.get('stop_cells'),
            leakage_test=audit.get('leakage_test'),wall_sec=perf_counter()-started,
            prediction_sha256=sha256(path))
        if score and baselines_ready(prepared):
            fields=summarize(prepared,spec,frame,audit)
            row=registry.update(spec['id'],status='completed')
            print(spec['id'],json.dumps(fields),flush=True)
            return row
        print(f"PREDICTION_READY {spec['id']} seeds={audit['n_seeds']}",flush=True)
        return registry.read(spec['id'])
    except Exception as exc:
        dest=prepared.out/'logs/failures';dest.mkdir(parents=True,exist_ok=True)
        (dest/f"{spec['id']}.log").write_text(traceback.format_exc(),encoding='utf-8')
        status='unsupported' if isinstance(exc,UnsupportedConfiguration) else 'failed'
        extra={'unsupported_evidence':exc.evidence} if status=='unsupported' else {}
        row=registry.update(spec['id'],status=status,error_type=type(exc).__name__,
            error=str(exc).replace(str(prepared.root),'<repo>'),**extra)
        print(f"{status.upper()} {spec['id']}: {exc}",flush=True)
        return row


def dispatch_stage(prepared,stage='all',*,retry=False):
    from phase_f.wf_workflow import run_stage
    stages=range(5) if stage=='all' else (int(stage or 0),)
    if any(s not in range(5) for s in stages):raise ValueError('Stage must be 0..4 or all')
    status=prepared.out/'logs/driver_status.json'
    write_json(status,{'status':'running','stage':stage,'pid':os.getpid(),'started_at':now(),
        'full_phase_complete':False,'holdout_read':False,'historical_final_artifact_read':False})
    try:
        for number in stages:
            if number==4:
                from phase_f.wf_final import run_final
                run_final(prepared)
            else:run_stage(prepared,number,retry=retry)
    except Exception as exc:
        write_json(status,{'status':'failed','stage':number,'at':now(),'error_type':type(exc).__name__,
            'error':str(exc),'full_phase_complete':False,'holdout_read':False})
        raise
    write_json(status,{'status':'completed','stage':stage,'at':now(),
        'full_phase_complete':stage in ('all','4',4),'holdout_read':False})


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all')
    parser.add_argument('--exp');parser.add_argument('--retry',action='store_true')
    args=parser.parse_args(argv);root=Path.cwd()
    from phase_f.wf_harness import WeeklyPrepared
    from phase_f.runtime import activate_optional_dependencies
    activate_optional_dependencies(root)
    approval=root/'outputs/phase_f/logs/active_revision.json'
    if not approval.is_file():raise RuntimeError('Revised user instruction evidence required')
    record=json.loads(approval.read_text(encoding='utf-8'))
    if record.get('protocol')!='walkforward_v2' or record.get('authorized') is not True:
        raise RuntimeError('Wrong active evaluation protocol')
    source=root/record['instruction_file']
    if sha256(source)!=record['instruction_sha256']:raise ValueError('User instruction evidence changed')
    prepared=WeeklyPrepared(root)
    if args.exp:
        row=WFRegistry(root).read(args.exp)
        if not row:raise ValueError('Experiment must first be locked in a wave')
        return execute(prepared,json.loads(row['config_json']),retry=args.retry)
    return dispatch_stage(prepared,args.stage,retry=args.retry)


if __name__=='__main__':main()
