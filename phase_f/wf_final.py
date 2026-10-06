"""Frozen finalist replay and one resumable CONFIRM transaction, never holdout."""
from __future__ import annotations
import copy
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from phase_f.registry import code_identity,config_hash,now,sha256,write_json
from phase_f.wf_contract import CONTRACT,two_rounds_converged
from phase_f.wf_evaluation import BASELINES,load_predictions,prediction_path,summarize,refresh_downstream,verify_metrics
from phase_f.wf_registry import WFRegistry
from phase_f.wf_selection import select_finalists,eligibility
from phase_f.wf_run import save_mean,cache_identity


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _sealed(path,value):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    value=clean(value)
    record={**value,'lock_sha256':config_hash(value)}
    write_json(path,record,exclusive=True)
    return record


def verify_lock(record):
    value={k:v for k,v in record.items() if k!='lock_sha256'}
    if record.get('lock_sha256')!=config_hash(value):raise ValueError('Changed final selection lock')
    return record


def _source_hashes(root):
    """Freeze Phase F and its model/Phase E source dependencies before CONFIRM."""
    root=Path(root)
    hashes=code_identity(root)['source_hashes']
    for folder in ('phase_c','src/models','outputs/phase_e/code'):
        for path in sorted((root/folder).rglob('*.py')):
            hashes[path.relative_to(root).as_posix()]=sha256(path)
    errors=root/'src/analysis/errors.py'
    if errors.is_file():hashes[errors.relative_to(root).as_posix()]=sha256(errors)
    return hashes


def _verify_sources(root,lock):
    if lock.get('source_hashes')!=_source_hashes(root):
        raise ValueError('Finalist source changed after the selection lock')


_PREFLIGHT_AUDITS={
    'preflight_sha256':'outputs/phase_f/logs/revision_20261006/preflight_audit.json',
    'alias_calendar_sha256':'outputs/phase_f/logs/revision_20261006/alias_calendar_audit.json',
}


def _verify_preflight_audits(root,lock):
    for field,name in _PREFLIGHT_AUDITS.items():
        path=Path(root)/name
        if not path.is_file() or sha256(path)!=lock.get(field):
            raise ValueError(f'Pre-CONFIRM audit changed: {name}')


def _phase_e_source_hashes(root):
    names=('phase_f/wf_downstream.py','phase_f/downstream.py','outputs/phase_e/code/probability.py',
           'outputs/phase_e/code/alerts.py','src/analysis/errors.py')
    paths={name:Path(root)/name for name in names}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError('CONFIRM Phase E source evidence is missing')
    return {name:sha256(path) for name,path in paths.items()}


def _snapshot_files(root,*folders):
    root=Path(root)
    return {p.relative_to(root).as_posix():sha256(p) for folder in folders
        for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.suffix in ('.json','.csv','.parquet')}


def _verify_snapshot(root,files):
    root=Path(root).resolve()
    for name,digest in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Frozen finalist artifact changed: {name}')


def search_manifest(prepared):
    """Do not permit a partial search to be advertised as a completed Phase F."""
    existing=prepared.out/'logs/search_complete.json'
    if existing.exists():
        record=verify_lock(_read(existing))
        for name,digest in record['files'].items():
            if sha256(prepared.out/name)!=digest:raise ValueError('Search completion evidence changed')
        return record
    r=WFRegistry(prepared.root);rows=[_read(p) for p in sorted(r.records.glob('*.json'))]
    if any(row['status'] not in ('completed','failed','unsupported','rejected','diagnostic_passed')
            and not (row['status']=='prediction_ready' and row.get('tpe_stop_completed') is True) for row in rows):
        raise RuntimeError('Unresolved weekly experiments before finalist selection')
    for stage in range(4):
        p=prepared.out/'logs/workflow'/f'stage_{stage}.json'
        if not p.is_file() or _read(p).get('status')!='completed':raise RuntimeError(f'Stage {stage} not completed')
    for family in ('F0','F1','F2','F3','F4','F5','F6','F7','F8','F9'):
        if not any(row['family']==family for row in rows):raise RuntimeError(f'Required family {family} absent')
    tuning=[_read(p) for p in (prepared.out/'logs/tuning').glob('*.json')]
    for kind in ('lightgbm','xgboost','catboost'):
        if not any(v.get('kind')==kind and (v.get('complete') is True or v.get('status')=='complete') and v.get('completed_trials',0)>=500 for v in tuning):
            raise RuntimeError(f'{kind} needs 500 completed weekly TPE trials')
    for row in rows:
        if row.get('tpe_stop_completed'):
            p=prediction_path(prepared,row['exp_id'])
            if not p.is_file() or sha256(p)!=row.get('prediction_sha256'):
                raise ValueError('A completed stop-only TPE forecast changed')
    rounds=_read(prepared.out/'logs/workflow/expansion_rounds.json')['rounds']
    if not two_rounds_converged(rounds):raise RuntimeError('Revised stopping criterion not satisfied')
    for mode in ('full','lora'):
        completed=[row for row in rows if row['status']=='completed' and
            json.loads(row['config_json']).get('finetune')==mode]
        budgets={json.loads(row['config_json']).get('num_steps') for row in completed}
        if len(budgets-{None})<2:raise RuntimeError(f'{mode} requires two completed training budgets')
    for row in rows:
        if row['status']=='completed':verify_metrics(prepared,row)
    files={p.relative_to(prepared.out).as_posix():sha256(p) for folder in ('logs/waves','logs/tuning','logs/workflow')
        for p in sorted((prepared.out/folder).glob('*.json'))}
    return _sealed(prepared.out/'logs/search_complete.json',{
        'completed_at':now(),'contract_sha256':CONTRACT['contract_sha256'],
        'registry_snapshot_sha256':sha256(prepared.out/'registry.csv'),'files':files,
        'trials':{kind:sum(v.get('completed_trials',0) for v in tuning if v.get('kind')==kind)
            for kind in ('lightgbm','xgboost','catboost')},
        'experiment_count':len(rows),'holdout_read':False})


def _make_junction(link,target):
    link=Path(link);target=Path(target)
    if not target.is_dir():raise ValueError('Junction target is not a directory')
    if os.path.lexists(link):
        if not link.exists() or link.resolve()!=target.resolve():
            raise ValueError('Existing junction points elsewhere or is broken')
        return
    link.parent.mkdir(parents=True,exist_ok=True)
    quote=lambda p: "'"+str(p).replace("'","''")+"'"
    command=f'New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} | Out-Null'
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],check=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),capture_output=True)
    if link.resolve()!=target.resolve():raise RuntimeError('Junction did not resolve to the expected directory')


def _junction(link,target):
    target=Path(target);link=Path(link)
    permitted=Path(r'D:\PeakGuard_PhaseF_20261003\models').resolve()
    if not target.resolve().is_relative_to(permitted):raise ValueError('Model output outside the dedicated Phase F disk')
    target.mkdir(parents=True,exist_ok=True)
    _make_junction(link,target)


def _optional_envs_junction(link,root):
    target=Path(root)/'outputs/phase_f/optional_envs'
    if not target.is_dir():raise ValueError('Original Phase F optional environments are absent')
    _make_junction(link,target)


def _final_view(prepared):
    view=copy.copy(prepared);view.out=prepared.out/'finalists_v2';view.out.mkdir(parents=True,exist_ok=True)
    _junction(view.out/'models',Path(r'D:\PeakGuard_PhaseF_20261003\models\walkforward_v2\finalists_v2'))
    _optional_envs_junction(view.out/'optional_envs',prepared.root)
    return view


def _dependencies(spec,catalog):
    if spec.get('adapter')=='ensemble':return [catalog[key] for key in spec['bases']]
    return []


def _fit_frozen(view,spec,catalog,*,arm='EXPLORE',pc3=False):
    from phase_f.wf_models import run,run_pc3_confirm,_is_stochastic
    path=prediction_path(view,spec['id'],arm)
    metadata=path.with_suffix('.json')
    if path.exists() and not metadata.exists():
        from phase_f.run import complete_prediction_cache
        complete_prediction_cache(path,metadata)
    elif metadata.exists() and not path.exists():
        digest=sha256(metadata)[:12];orphan=metadata.with_name(f'{metadata.stem}.orphan-{digest}{metadata.suffix}')
        counter=0
        while orphan.exists():
            counter+=1;orphan=metadata.with_name(f'{metadata.stem}.orphan-{digest}-{counter}{metadata.suffix}')
        metadata.rename(orphan)
    if path.exists():
        record=_read(metadata)
        if record['identity']!=cache_identity(view,spec):raise ValueError('Frozen finalist cache identity changed')
        return load_predictions(view,spec['id'],arm),record['audit']
    for parent in _dependencies(spec,catalog):_fit_frozen(view,parent,catalog,arm=arm,pc3=pc3)
    # A 5-seed ensemble finalist receives ten parent seed paths; an originally
    # larger declared 20-seed experiment retains all its paths.
    current=copy.deepcopy(spec)
    current['arm']=arm
    current.pop('n_seeds',None)
    if current.get('adapter')=='seed_ensemble' and len(current['seeds'])<10:
        from phase_f.wf_models import SEEDS
        current['seeds']+= [s for s in SEEDS if s not in current['seeds']][:10-len(current['seeds'])]
    count=10 if _is_stochastic(current) else 1
    if pc3:
        frame,audit=run_pc3_confirm(view,current,n_seeds=10)
    else:frame,audit=run(view,current,arm=arm,n_seeds=count)
    save_mean(view,spec,frame,audit,arm)
    return frame,audit


def _confirm_e(view,key):
    from phase_f.wf_downstream import evaluate_candidate
    frame=load_predictions(view,key,'CONFIRM')
    base=load_predictions(view,BASELINES['B5'],'CONFIRM').copy();base['model']='B5'
    dest=view.out/'tables/phase_e/CONFIRM'/key
    inputs={'candidate':sha256(prediction_path(view,key,'CONFIRM')),
        'B5':sha256(prediction_path(view,BASELINES['B5'],'CONFIRM'))}
    input_path=dest/'input_identity.json'
    manifest_path=dest/'manifest.json';unavailable_path=dest/'unavailable.json'
    if (manifest_path.exists() or unavailable_path.exists()) and not input_path.exists():
        raise ValueError('CONFIRM Phase E input evidence is missing')
    write_json(input_path,inputs,exclusive=True)
    if manifest_path.exists() and unavailable_path.exists():
        raise ValueError('CONFIRM Phase E has conflicting completed and unavailable evidence')
    if manifest_path.exists():
        manifest=_read(manifest_path)
        if manifest.get('manifest_sha256')!=config_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
            raise ValueError('CONFIRM Phase E manifest changed')
        for name,digest in manifest['artifacts'].items():
            if sha256(dest/name)!=digest:raise ValueError('CONFIRM Phase E artifact changed')
        for name,digest in manifest['source_sha256'].items():
            if sha256(view.root/name)!=digest:raise ValueError('CONFIRM Phase E source changed')
        if _read(input_path)!=inputs:raise ValueError('CONFIRM Phase E input changed')
        return {**manifest['registry_fields'],'E_status':'completed'}
    if unavailable_path.exists():
        record=verify_lock(_read(unavailable_path))
        if (record.get('status')!='unavailable' or record.get('input_sha256')!=inputs or
            record.get('input_identity_sha256')!=sha256(input_path) or
            record.get('source_sha256')!=_phase_e_source_hashes(view.root)):
            raise ValueError('CONFIRM Phase E unavailable evidence changed')
        return {'E_status':'unavailable','E_reason':record['reason']}
    try:
        result=evaluate_candidate(view,frame,base,arm='CONFIRM',destination=dest)
        return {**result['registry_fields'],'E_status':'completed'}
    except ValueError as exc:
        _sealed(unavailable_path,{'reason':str(exc),'status':'unavailable',
            'input_sha256':inputs,'input_identity_sha256':sha256(input_path),
            'source_sha256':_phase_e_source_hashes(view.root),'holdout_read':False})
        return {'E_status':'unavailable','E_reason':str(exc)}


def confirm_gate(fields):
    def finite(key):
        try:return float(fields[key]) if np.isfinite(float(fields[key])) else None
        except (KeyError,ValueError,TypeError):return None
    lo=finite('ciLow_vs_B5');hi=finite('wf_peak_degradation_ci_high_vs_B5')
    d2=finite('wf_D2_mae_improvement_vs_B5');recall=finite('E_c10_22_recall');base=finite('E_B5_c10_22_recall')
    return bool(lo is not None and lo>0 and hi is not None and hi<=0 and d2 is not None and d2>=0
        and fields.get('E_status')=='completed' and recall is not None and base is not None and recall>=base)


def _committed_selection(prepared,lockpath,lock):
    relative=lockpath.relative_to(prepared.root).as_posix()
    committed=json.loads(subprocess.check_output(['git','show',f'HEAD:{relative}'],cwd=prepared.root))
    if committed!=lock:raise ValueError('Finalist selection lock is not committed in HEAD')


def _pc3_view(prepared):
    from phase_f.harness import Prepared
    from phase_f.wf_models import _pc3_confirm_view
    pc3=Prepared(prepared.root);pc3.out=prepared.out/'pc3_finalists';pc3.out.mkdir(parents=True,exist_ok=True)
    _junction(pc3.out/'models',Path(r'D:\PeakGuard_PhaseF_20261003\models\walkforward_v2\pc3_finalists'))
    _optional_envs_junction(pc3.out/'optional_envs',prepared.root)
    pc3.contexts=_pc3_confirm_view(pc3).contexts
    return pc3


def _completion_artifacts(prepared,view,pc3):
    """Seal every forecast input consumed by the single CONFIRM transaction."""
    return _snapshot_files(prepared.out,
        view.out/'tables',pc3.out/'tables',
        view.out/'predictions/CONFIRM',view.out/'predictions/seeds/CONFIRM',
        pc3.out/'predictions/CONFIRM',pc3.out/'predictions/seeds/CONFIRM',
        view.out/'predictions/EXPLORE',view.out/'predictions/seeds/EXPLORE')


def run_final(prepared):
    from phase_f.workflow import _git_commit
    registry=WFRegistry(prepared.root)
    lockpath=prepared.out/'logs/confirm_lock.json';reservation=prepared.out/'logs/confirm_reservation.json'
    finalpath=prepared.out/'logs/confirm_once.json'
    if finalpath.exists():
        done=verify_lock(_read(finalpath));lock=verify_lock(_read(lockpath))
        if done['selection_lock_sha256']!=lock['lock_sha256']:raise ValueError('CONFIRM completion belongs to a different lock')
        _verify_sources(prepared.root,lock)
        _verify_preflight_audits(prepared.root,lock)
        _committed_selection(prepared,lockpath,lock)
        if not reservation.is_file():raise ValueError('CONFIRM reservation evidence is missing')
        reserved=verify_lock(_read(reservation))
        if (reserved.get('selection_lock_sha256')!=lock['lock_sha256'] or
            done.get('reservation_sha256')!=sha256(reservation)):
            raise ValueError('CONFIRM reservation evidence changed')
        _verify_snapshot(prepared.out,lock['explore_artifacts'])
        _verify_snapshot(prepared.out,done['artifacts'])
        from phase_f.wf_reporting import final_report
        final_report(prepared,lock,done)
        write_json(prepared.out/'logs/workflow/stage_4.json',{'status':'completed','at':now(),
            'confirm_once_sha256':done['lock_sha256'],'holdout_read':False})
        _git_commit(prepared,'Phase F: complete locked weekly development confirmation')
        return done
    if not lockpath.exists():
        search=search_manifest(prepared)
        from phase_f.wf_plan import completed
        rows=completed(prepared.root)
        # All completed finalists can be assessed; the automatic top-20 runs
        # during search remain preserved. This also covers global specialists.
        refresh_downstream(prepared,ids=[r['exp_id'] for r in rows])
        rows=completed(prepared.root);weeks=sum(w['arm']=='EXPLORE' for w in prepared.split_lock['weeks'])
        chosen=select_finalists(rows,expected_weeks=weeks)
        provisional=prepared.out/'logs/finalist_seed_expansion_lock.json'
        catalog={row['exp_id']:json.loads(row['config_json']) for row in rows}
        specs=[catalog[key] for key in [*BASELINES.values(),*chosen['candidates']]]
        if provisional.exists():
            p=verify_lock(_read(provisional));chosen=p['selection'];specs=p['specs'];catalog=p['catalog']
        else:_sealed(provisional,{'at':now(),'selection':chosen,'specs':specs,'catalog':catalog,
            'seed_policy':'10 stochastic finalist seeds, one deterministic replay','candidate_confirm_metric_read':False})
        view=_final_view(prepared);ten={}
        for spec in specs:_fit_frozen(view,spec,catalog)
        for spec in specs:
            frame=load_predictions(view,spec['id']);audit=_read(prediction_path(view,spec['id']).with_suffix('.json'))['audit']
            ten[spec['id']]=summarize(view,spec,frame,audit,destination=view.out/'tables'/spec['id']/'EXPLORE',update_registry=False)
        for spec in specs:
            if spec['id'] in chosen['candidates']:
                from phase_f.wf_downstream import evaluate_candidate
                b=load_predictions(view,BASELINES['B5']).copy();b['model']='B5'
                result=evaluate_candidate(view,load_predictions(view,spec['id']),b,
                    destination=view.out/'tables/phase_e/EXPLORE'/spec['id'])
                ten[spec['id']].update(**result['registry_fields'],E_status='completed')
        lock=_sealed(lockpath,{'at':now(),'contract_sha256':CONTRACT['contract_sha256'],
            'split_sha256':prepared.split_lock['lock_sha256'],'search_sha256':search['lock_sha256'],
            'selection':chosen,'candidates':chosen['candidates'],'specs':specs,'catalog':catalog,
            'ten_seed_explore':ten,'candidate_confirm_metric_read':False,'holdout_read':False,
            'source_hashes':_source_hashes(prepared.root),
            'preflight_sha256':sha256(prepared.root/'outputs/phase_f/logs/revision_20261006/preflight_audit.json'),
            'alias_calendar_sha256':sha256(prepared.root/'outputs/phase_f/logs/revision_20261006/alias_calendar_audit.json'),
            'explore_artifacts':_snapshot_files(prepared.out,view.out/'tables',
                view.out/'predictions/EXPLORE',view.out/'predictions/seeds/EXPLORE'),
            'prediction_sha256':{s['id']:sha256(prediction_path(view,s['id'])) for s in specs}})
        _git_commit(prepared,'Phase F: lock revised weekly finalists before CONFIRM')
    lock=verify_lock(_read(lockpath));view=_final_view(prepared)
    _verify_sources(prepared.root,lock)
    _verify_preflight_audits(prepared.root,lock)
    _committed_selection(prepared,lockpath,lock)
    _verify_snapshot(prepared.out,lock['explore_artifacts'])
    for key,digest in lock['prediction_sha256'].items():
        if sha256(prediction_path(view,key))!=digest:raise ValueError('Seed-expanded EXPLORE forecast changed')
    if not reservation.exists():
        _sealed(reservation,{'at':now(),'selection_lock_sha256':lock['lock_sha256'],
            'protocol':'one transaction: WF CONFIRM + pc3 CONFIRM + fixed Phase E','holdout_read':False})
    else:
        if verify_lock(_read(reservation))['selection_lock_sha256']!=lock['lock_sha256']:raise ValueError('Reserved CONFIRM candidate mismatch')
    # Reservation is durable before the first candidate CONFIRM numeric read.
    # An interrupted transaction resumes only these IDs and cached cells.
    pc3=_pc3_view(prepared)
    results={'WF':{},'pc3':{}}
    for name,current in (('WF',view),('pc3',pc3)):
        for spec in lock['specs']:_fit_frozen(current,spec,lock['catalog'],arm='CONFIRM',pc3=name=='pc3')
        for spec in lock['specs']:
            frame=load_predictions(current,spec['id'],'CONFIRM')
            audit=_read(prediction_path(current,spec['id'],'CONFIRM').with_suffix('.json'))['audit']
            fields=summarize(current,spec,frame,audit,arm='CONFIRM')
            fields.update(_confirm_e(current,spec['id']))
            fields['evaluation_arm']='CONFIRM'
            fields['confirm_gates_met']=confirm_gate(fields)
            results[name][spec['id']]=fields
    keep=[key for key in lock['candidates'] if results['WF'][key]['confirm_gates_met']]
    primary=min(keep,key=lambda k:(results['WF'][k]['wf_explore_AUC_MAE'],k)) if keep else None
    overlapping=[]
    if primary:
        from phase_f.wf_metrics import paired_ci
        baseline=load_predictions(view,primary,'CONFIRM')
        for key in keep:
            if key==primary:continue
            ci=paired_ci(load_predictions(view,key,'CONFIRM'),baseline,arm='CONFIRM',n=1000,seed=42)
            row=ci.query("dataset == 'D1' and metric == 'AUC_MAE_improvement'").iloc[0]
            ci.to_csv(view.out/'tables'/key/'CONFIRM'/'paired_vs_representative.csv',index=False)
            if row.ci_status=='available' and row.ci_low<=0<=row.ci_high:
                cfg=lock['catalog'][key];value=results['WF'][key]
                overlapping.append({'candidate':key,'paired_mae_ci':[float(row.ci_low),float(row.ci_high)],
                    'peak':value['wf_explore_AUC_PeakMAE'],'alert':value.get('E_c10_22_recall'),
                    'complexity':{'adapter':cfg['adapter'],'kind':cfg.get('kind'),
                        'context_length':cfg.get('context_length'),'n_seeds':value.get('n_seeds')},
                    'decision':'human choice; no automatic replacement'})
    combined={}
    if primary:
        from phase_f.wf_combined import evaluate_combined
        ids=[primary]
        if overlapping:ids.append(min(overlapping,key=lambda r:(r['peak'],r['candidate']))['candidate'])
        for key in ids:combined[key]=evaluate_combined(view,key)
    artifacts=_completion_artifacts(prepared,view,pc3)
    done=_sealed(finalpath,{'at':now(),'selection_lock_sha256':lock['lock_sha256'],
        'reservation_sha256':sha256(reservation),
        'representative_candidate':primary,'results':results,'artifacts':artifacts,
        'overlapping_alternatives':overlapping,'combined_phase_e':combined,
        'verdict':'candidate_proposal' if primary else 'exploratory_improvement_not_confirmed',
        'candidate_confirm_metric_read':True,'holdout_read':False,'new_candidates_after_confirm':False})
    from phase_f.wf_reporting import final_report
    final_report(prepared,lock,done)
    write_json(prepared.out/'logs/workflow/stage_4.json',{'status':'completed','at':now(),
        'confirm_once_sha256':done['lock_sha256'],'holdout_read':False})
    _git_commit(prepared,'Phase F: complete locked weekly development confirmation')
    return done
