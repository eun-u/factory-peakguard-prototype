"""EXPLORE eligibility and one-time, precommitted development CONFIRM.

No holdout workflow is imported here. Once the selection lock exists, candidate
identities and their prediction bytes are immutable; confirm cannot reselect.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from phase_f.registry import Registry,config_hash,sha256,write_json,now
from phase_f.experiment_plan import completed,spec
from phase_f.metrics import evaluate,paired_ci


def foundation_training_coverage(prepared):
    """Prove two distinct completed Chronos fit budgets per full/LoRA mode.

    Only forecast key metadata is read from the prediction parquet. Neither
    SCORE labels/predictions nor CONFIRM metrics enter this prelock check.
    """
    out=prepared.out
    expected_cells={(h,f) for h in range(4,17) for f in range(3)}
    candidates={'full':{},'lora':{}}
    for row in completed(prepared.root,adapter='foundation'):
        cfg=spec(row);mode=cfg.get('finetune')
        if mode not in candidates:continue
        exp_id=row['exp_id']
        if (cfg.get('id')!=exp_id or cfg.get('kind')!='chronos2' or
                cfg.get('family')!='F6' or row.get('config_hash')!=config_hash(cfg) or
                row.get('holdout_read') is not False or
                row.get('historical_final_artifact_read') is not False):
            raise ValueError(f'Invalid completed fine-tune registry identity: {exp_id}')
        if not np.isfinite(float(row['explore_AUC_MAE'])) or int(row.get('explore_queries',0))<1:
            raise ValueError(f'Fine-tune lacks completed EXPLORE evaluation: {exp_id}')
        steps=cfg.get('num_steps')
        if isinstance(steps,bool) or not isinstance(steps,int) or steps<1:
            raise ValueError(f'Fine-tune has invalid training steps: {exp_id}')
        candidates[mode].setdefault(steps,[]).append(row)

    proof={}
    for mode,by_steps in candidates.items():
        if len(by_steps)<2:
            raise ValueError(f'F6 {mode} needs two distinct completed num_steps values')
        experiments=[]
        # A single deterministic representative for each of the first two
        # budgets keeps repeated pre/post-CONFIRM verification bounded.
        for steps in sorted(by_steps)[:2]:
            row=min(by_steps[steps],key=lambda item:item['exp_id'])
            exp_id=row['exp_id']
            if not exp_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.+'
                                 for c in exp_id):
                raise ValueError('Unsafe fine-tune experiment ID')
            pred=out/'predictions'/f'{exp_id}.parquet'
            audit_path=out/'logs'/f'{exp_id}_audit.json'
            if not pred.is_file() or not audit_path.is_file():
                raise ValueError(f'Fine-tune prediction/audit is missing: {exp_id}')
            pred_hash=sha256(pred)
            audit=json.loads(audit_path.read_text(encoding='utf-8'))
            fit_cells=audit.get('fit_cells')
            if (audit.get('config_hash')!=row['config_hash'] or
                    audit.get('prediction_sha256')!=pred_hash or
                    audit.get('leakage_test')!='passed' or
                    audit.get('holdout_read') is not False or
                    audit.get('historical_final_artifact_read') is not False or
                    not isinstance(fit_cells,list) or
                    {cell.get('fold') for cell in fit_cells} != {0,1,2} or
                    len(fit_cells)!=3 or
                    any(not isinstance(cell.get('checkpoint_files'),dict) or
                        not cell['checkpoint_files'] for cell in fit_cells)):
                raise ValueError(f'Fine-tune fit/audit identity is incomplete: {exp_id}')
            metadata=pd.read_parquet(pred,columns=['model','role','horizon','fold','origin','target_time'])
            score=metadata.loc[metadata.role.eq('score')]
            keys=['horizon','fold','origin','target_time']
            cells=set(map(tuple,score[['horizon','fold']].drop_duplicates().to_numpy()))
            if (not metadata.model.eq(exp_id).all() or cells!=expected_cells or
                    score.duplicated(keys).any() or
                    not pd.MultiIndex.from_frame(score[keys].sort_values(keys)).equals(
                        pd.MultiIndex.from_frame(prepared.keys[keys].sort_values(keys)))):
                raise ValueError(f'Fine-tune lacks the complete 39-cell score cohort: {exp_id}')
            metric_hash=verify_explore_metrics(prepared.root,row)
            experiments.append({'exp_id':exp_id,'num_steps':steps,'config_hash':row['config_hash'],
                'score_cells':39,'score_key_count':len(score),'prediction_sha256':pred_hash,
                'audit_sha256':sha256(audit_path),'explore_manifest_sha256':metric_hash})
        proof[mode]={'status':'completed','distinct_num_steps':sorted(by_steps)[:2],
                     'experiments':experiments}
    return proof


def verify_search_complete(prepared):
    path=prepared.out/'logs/search_complete.json'
    if not path.exists():raise RuntimeError('Search completion evidence is required before CONFIRM')
    record=json.loads(path.read_text(encoding='utf-8'))
    for mapping in ('waves','tuningfiles'):
        if not record.get(mapping):raise ValueError(f'Missing {mapping} evidence')
        for name,digest in record[mapping].items():
            target=(prepared.out/name).resolve()
            if not target.is_relative_to(prepared.out.resolve()):raise ValueError('Search evidence outside experiment')
            if sha256(target)!=digest:raise ValueError(f'Changed search evidence: {name}')
    coverage=record.get('required_family_coverage',{})
    if set(coverage)!=set(f'F{i}' for i in range(11)):
        raise ValueError('F0-F10 coverage is incomplete')
    for key,value in coverage.items():
        status=value.get('status') if isinstance(value,dict) else value
        if status not in ('completed','done','failed_with_evidence'):
            raise ValueError(f'Unresolved family: {key}')
    expansion=record.get('expansion_rounds',[])
    if len(expansion)<2 or any(r.get('meaningful_improvement') is not False for r in expansion[-2:]):
        raise ValueError('Two finite nonimproving extension rounds are required')
    registry=Registry(prepared.root)
    if not record.get('experiment_ids'):raise ValueError('No executed experiment evidence')
    for key in record['experiment_ids']:
        row=registry.read(key)
        if not row or row['status'] not in ('completed','failed','unsupported','rejected','diagnostic_passed'):
            raise ValueError(f'Unresolved experiment: {key}')
    fine_proof=foundation_training_coverage(prepared)
    if record.get('foundation_training_coverage')!=fine_proof:
        raise ValueError('F6 full/LoRA training-budget completion evidence changed')
    locked_ids=set()
    for relative in record['waves']:
        wave=json.loads((prepared.out/relative).read_text(encoding='utf-8'))
        if wave.get('config_sha256')!=config_hash(wave.get('specs')):
            raise ValueError(f'Changed wave specifications: {relative}')
        locked_ids.update(item['id'] for item in wave['specs'])
    for evidence in fine_proof.values():
        for item in evidence['experiments']:
            if item['exp_id'] not in record['experiment_ids'] or item['exp_id'] not in locked_ids:
                raise ValueError('F6 training evidence is outside the locked search')
    done=completed(prepared.root)
    for prefix in ('F3-1-gbdt_lightgbm-t','F3-4-gbdt_xgboost-t','F3-4-gbdt_catboost-t'):
        trials=[row for row in done if row['exp_id'].startswith(prefix) and row.get('stop_cells')==39]
        if len(trials)<500:raise ValueError(f'Fewer than 500 completed 39-cell trials: {prefix}')
    return sha256(path)


def verify_explore_metrics(root,row):
    out=Path(root)/'outputs/phase_f';dest=out/'tables'/row['exp_id']
    path=dest/'explore_manifest.json'
    if not path.exists():raise ValueError('EXPLORE metric manifest missing; recompute from sealed predictions before selection')
    record=json.loads(path.read_text(encoding='utf-8'))
    if record['metrics_source_sha256']!=sha256(Path(root)/'phase_f/metrics.py'):
        raise ValueError('Metrics implementation changed after evaluation')
    if record['split_sha256']!=json.loads((out/'logs/split_lock.json').read_text(encoding='utf-8'))['lock_sha256']:
        raise ValueError('Metric split differs from locked cohort')
    if sha256(out/'predictions'/f"{row['exp_id']}.parquet")!=record['prediction_sha256']:
        raise ValueError('EXPLORE predictions differ from metric manifest')
    for name,digest in record['tables'].items():
        if sha256(dest/name)!=digest:raise ValueError('EXPLORE metric table changed')
    for name,digest in record['baseline_prediction_sha256'].items():
        if sha256(Path(root)/name)!=digest:raise ValueError('Metric baseline predictions changed')
    for name,value in record['registry_metrics'].items():
        actual=row.get(name)
        if isinstance(value,(int,float)) and isinstance(actual,(int,float)) and np.isnan(value) and np.isnan(actual):continue
        if actual!=value:raise ValueError('Registry metrics differ from sealed table computation')
    return sha256(path)


def complete_metric_manifests(prepared):
    from phase_f.run import summarize_experiment
    registry=Registry(prepared.root)
    for row in completed(prepared.root):
        path=prepared.out/'tables'/row['exp_id']/'explore_manifest.json'
        needs_manifest=not path.exists()
        if not needs_manifest:
            verify_explore_metrics(prepared.root,row)
            needs_manifest='explore_pairwise_week_ci.csv' not in json.loads(path.read_text(encoding='utf-8'))['tables']
        if needs_manifest:
            predictions=prepared.out/'predictions'/f"{row['exp_id']}.parquet"
            audit=json.loads((prepared.out/'logs'/f"{row['exp_id']}_audit.json").read_text(encoding='utf-8'))
            if sha256(predictions)!=audit['prediction_sha256']:raise ValueError('Cannot reseal modified prediction')
            summarize_experiment(prepared,spec(row),pd.read_parquet(predictions),registry)


def eligibility(root,row,arm='EXPLORE',directory=None):
    if arm not in ('EXPLORE','CONFIRM'):raise ValueError('Unknown development arm')
    root=Path(root);out=root/'outputs/phase_f'
    if arm=='EXPLORE':verify_explore_metrics(root,row)
    if directory is None:directory=out/'tables'/row['exp_id']
    directory=Path(directory)
    ci=pd.read_csv(directory/f'{arm.lower()}_pairwise_ci.csv')
    selected=ci[ci.baseline.eq('B5') & ci.horizon.isna()]
    def metric(dataset,name):
        found=selected[selected.dataset.eq(dataset)&selected.metric.eq(name)]
        if len(found)!=1:raise ValueError('Ambiguous eligibility metric')
        return found.iloc[0]
    mae=metric('D1','AUC_MAE_improvement');peak=metric('D1','AUC_PeakMAE_degradation')
    d2mae=metric('D2','AUC_MAE_improvement');d2peak=metric('D2','AUC_PeakMAE_degradation')
    if arm=='EXPLORE':
        b5=Registry(root).read('F0-1-B5')
        verify_explore_metrics(root,b5)
        folds=sum(row[f'fold{f}_MAE']<b5[f'fold{f}_MAE'] for f in (0,1,2))
    else:
        # The approved CONFIRM gate retains MAE/peak/D2 protection. Fold evidence
        # is shown separately; it does not create a new selection opportunity.
        folds=None
    checks={'leakage_passed':row.get('leakage_test') in ('passed','parent_sealed_contract'),
        'mae_ci_available':mae.ci_status=='available',
        'mae_ci_lower_positive':bool(np.isfinite(mae.ci_low) and mae.ci_low>0),
        'peak_ci_available':peak.ci_status=='available',
        'peak_ci_upper_nonpositive':bool(np.isfinite(peak.ci_high) and peak.ci_high<=0),
        'd2_mae_improves':bool(np.isfinite(d2mae.estimate) and d2mae.estimate>0),
        'd2_peak_nonworse':bool(np.isfinite(d2peak.estimate) and d2peak.estimate<=0)}
    if folds is not None:checks['two_folds_improve']=folds>=2
    return {'exp_id':row['exp_id'],'eligible':bool(all(checks.values())),
            'checks':checks,'folds_improved':folds,'mae_improvement':float(mae.estimate),
            'mae_ci_low':float(mae.ci_low),'peak_degradation':float(peak.estimate),
            'peak_ci_high':float(peak.ci_high),'arm':arm}


def explore_eligibility(root):
    return [eligibility(root,row) for row in completed(root)
            if spec(row).get('adapter')!='baseline']


def lock_selection(prepared,*,all_waves_complete=False,expansion_converged=False):
    out=prepared.out;path=out/'logs/confirm_lock.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    if not all_waves_complete or not expansion_converged:
        raise RuntimeError('Required experiment coverage and convergence must precede selection')
    search_hash=verify_search_complete(prepared)
    found=completed(prepared.root)
    eligible={item['exp_id']:item for item in explore_eligibility(prepared.root)}
    choices=[row for row in found if eligible.get(row['exp_id'],{}).get('eligible')]
    selected=[];counts={}
    # Preserve the best eligible EXPLORE primary; prefer new families thereafter.
    for distinct in (True,False):
        for row in choices:
            family=row['family']
            if row['exp_id'] in selected or counts.get(family,0)>=2:continue
            if distinct and counts.get(family,0):continue
            selected.append(row['exp_id']);counts[family]=counts.get(family,0)+1
            if len(selected)==5:break
        if len(selected)==5:break
    registry=Registry(prepared.root)
    # A fixed secondary tabular policy enables F9-4 even when the primary is a
    # zero-shot model with no supervised weekly update definition.
    tabular=[row for row in found if spec(row).get('adapter')=='regression'
             and spec(row).get('kind') not in ('two_stage',)
             and spec(row).get('target')!='b5_residual'
             and 'production' not in spec(row).get('groups',[])]
    weekly=tabular[0]['exp_id'] if tabular else None
    record={'version':1,'created_at':now(),'primary_candidate':selected[0] if selected else None,
        'search_complete_sha256':search_hash,
        'candidates':selected,'weekly_diagnostic_candidate':weekly,
        'weekly_diagnostic_config_hash':registry.read(weekly)['config_hash'] if weekly else None,
        'selection_arm':'EXPLORE','confirm_is_reused_development':True,
        'config_hashes':{key:registry.read(key)['config_hash'] for key in selected},
        'prediction_hashes':{key:sha256(out/'predictions'/f'{key}.parquet') for key in selected},
        'explore_metric_manifest_hashes':{key:verify_explore_metrics(prepared.root,registry.read(key)) for key in selected},
        'split_lock_sha256':prepared.split_lock['lock_sha256'],
        'explore_eligibility':eligible,'no_postconfirm_reselection':True,
        'historical_final_artifact_read':False,'holdout_read':False}
    record['lock_sha256']=config_hash(record)
    write_json(path,record,exclusive=True)
    return record


def verify_lock(prepared,record):
    checksum=record['lock_sha256']
    content={k:v for k,v in record.items() if k!='lock_sha256'}
    if config_hash(content)!=checksum:raise ValueError('Selection lock changed')
    if record['split_lock_sha256']!=prepared.split_lock['lock_sha256']:raise ValueError('Split changed')
    registry=Registry(prepared.root)
    if verify_search_complete(prepared)!=record['search_complete_sha256']:
        raise ValueError('Search completion evidence changed after selection')
    for key in record['candidates']:
        if registry.read(key)['config_hash']!=record['config_hashes'][key]:raise ValueError('Candidate config changed')
        if sha256(prepared.out/'predictions'/f'{key}.parquet')!=record['prediction_hashes'][key]:
            raise ValueError('Candidate predictions changed after selection')
        if verify_explore_metrics(prepared.root,registry.read(key))!=record['explore_metric_manifest_hashes'][key]:
            raise ValueError('Locked EXPLORE evidence changed')


def confirm_once(prepared):
    out=prepared.out;record=json.loads((out/'logs/confirm_lock.json').read_text(encoding='utf-8'))
    verify_lock(prepared,record)
    rel='outputs/phase_f/logs/confirm_lock.json'
    committed=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=prepared.root)
    if config_hash(json.loads(committed))!=config_hash(record):
        raise ValueError('Commit the unchanged selection lock before opening CONFIRM')
    complete=out/'logs/confirm_complete.json';reservation=out/'logs/confirm_reserved.json'
    if complete.exists():
        done=json.loads(complete.read_text(encoding='utf-8'))
        if done['lock_sha256']!=record['lock_sha256']:raise ValueError('Confirm lock mismatch')
        if config_hash({k:v for k,v in done.items() if k!='confirm_once_sha256'})!=done.get('confirm_once_sha256'):
            raise ValueError('Completed CONFIRM record changed')
        if done.get('primary_candidate')!=record['primary_candidate'] or done.get('candidates')!=record['candidates']:
            raise ValueError('Completed CONFIRM identities differ from locked candidates')
        for name,digest in done['tables'].items():
            if sha256(out/name)!=digest:raise ValueError('Completed CONFIRM table changed')
        registry=Registry(prepared.root)
        checked=[eligibility(prepared.root,registry.read(key),'CONFIRM',out/'tables/confirm'/key)
                 for key in record['candidates']]
        if config_hash(checked)!=config_hash(done['eligibility']):
            raise ValueError('Completed eligibility differs from sealed CONFIRM tables')
        primary=next((r for r in checked if r['exp_id']==record['primary_candidate']),None)
        if done.get('primary_confirmed')!=bool(primary and primary['eligible']):
            raise ValueError('Completed primary claim differs from sealed eligibility')
        return done
    if reservation.exists():
        previous=json.loads(reservation.read_text(encoding='utf-8'))
        if previous['lock_sha256']!=record['lock_sha256']:raise ValueError('Reserved CONFIRM lock mismatch')
        # An interrupted calculation can deterministically finish the same
        # immutable cohort/candidates; it cannot change their configuration.
    else:write_json(reservation,{'lock_sha256':record['lock_sha256'],'reserved_at':now()},exclusive=True)
    tables={};rows=[];registry=Registry(prepared.root)
    for key in [*record['candidates'],'B5','M1','R1']:
        frame=(pd.read_parquet(out/'predictions'/f'{key}.parquet') if key in record['candidates']
               else prepared.baselines[prepared.baselines.model.eq(key)].copy())
        dest=out/'tables/confirm'/key;dest.mkdir(parents=True,exist_ok=True)
        for name,table in evaluate(frame,arm='CONFIRM').items():
            path=dest/f'confirm_{name}.csv';table.to_csv(path,index=False)
            tables[path.relative_to(out).as_posix()]=sha256(path)
        cis=pd.concat([paired_ci(frame,prepared.baselines[prepared.baselines.model.eq(b)],arm='CONFIRM')
                       for b in ('B5','M1','R1')],ignore_index=True)
        path=dest/'confirm_pairwise_ci.csv';cis.to_csv(path,index=False)
        tables[path.relative_to(out).as_posix()]=sha256(path)
        weekly=pd.concat([paired_ci(frame,prepared.baselines[prepared.baselines.model.eq(b)],
                                    arm='CONFIRM',block='week') for b in ('B5','M1','R1')],ignore_index=True)
        path=dest/'confirm_pairwise_week_ci.csv';weekly.to_csv(path,index=False)
        tables[path.relative_to(out).as_posix()]=sha256(path)
        if key in record['candidates']:rows.append(eligibility(prepared.root,registry.read(key),'CONFIRM',dest))
    primary=next((row for row in rows if row['exp_id']==record['primary_candidate']),None)
    done={'version':1,'completed_at':now(),'complete':True,'lock_sha256':record['lock_sha256'],
        'primary_candidate':record['primary_candidate'],'candidates':record['candidates'],
        'primary_confirmed':bool(primary and primary['eligible']),'eligibility':rows,'tables':tables,
        'primary_never_reselected':True,'historical_final_artifact_read':False,'holdout_read':False}
    done['confirm_once_sha256']=config_hash(done)
    write_json(complete,done,exclusive=True)
    return done
