"""Finite, immutable waves for the approved Phase F experiment families.

Adaptive choices use completed EXPLORE records only. Every returned list is
locked before execution by the orchestrator; no CONFIRM result is an input.
"""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from phase_f.registry import config_hash,write_json


def rows(root):
    return [json.loads(p.read_text(encoding='utf-8')) for p in
            sorted((Path(root)/'outputs/phase_f/walkforward_v2/logs/experiments').glob('*.json'))]


def spec(row):
    return json.loads(row['config_json'])


def completed(root, *, adapter=None,kind=None,family=None):
    result=[]
    for row in rows(root):
        cfg=spec(row)
        if row['status']!='completed' or 'explore_AUC_MAE' not in row:continue
        if adapter and cfg.get('adapter')!=adapter:continue
        if kind and cfg.get('kind')!=kind:continue
        if family and cfg.get('family')!=family:continue
        result.append(row)
    return sorted(result,key=lambda r:(r['explore_AUC_MAE'],r['exp_id']))


def best(root,**filters):
    found=completed(root,**filters)
    if not found:raise RuntimeError(f'No completed parent for {filters}')
    return spec(found[0])


def child(parent,exp_id,**changes):
    cfg={**parent,'id':exp_id,'parent':parent['id'],**changes}
    cfg.pop('note',None)
    cfg['family']=exp_id.split('-')[0]
    return cfg


def lock_wave(root,name,specs,hypothesis):
    path=Path(root)/'outputs/phase_f/walkforward_v2/logs/waves'/f'{name}.json'
    # Existing wave is authoritative. Re-running later must not select new parents.
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))['specs']
    ids=[s['id'] for s in specs]
    if len(ids)!=len(set(ids)):raise ValueError('Wave has duplicate configuration IDs')
    record={'name':name,'hypothesis':hypothesis,'specs':specs,
            'config_sha256':config_hash(specs),'selection_arm':'EXPLORE',
            'historical_final_artifact_read':False,'holdout_read':False}
    write_json(path,record,exclusive=True)
    return specs


def feature_union(root):
    baseline={k:next(r for r in completed(root,kind=k) if r['exp_id']==f'F1-core-{k}')
              for k in ('ridge','lightgbm')}
    union=[]
    for row in completed(root,family='F1'):
        cfg=spec(row);kind=cfg.get('kind')
        if kind not in baseline or not cfg.get('groups'):continue
        if row['explore_AUC_MAE']<baseline[kind]['explore_AUC_MAE']:
            union.extend(cfg['groups'])
    union=sorted(set(union)-{'production'})
    result=[]
    for kind in ('ridge','lightgbm'):
        parent=spec(baseline[kind])
        result.append(child(parent,f'F1-union-{kind}',groups=union))
        for group in union:
            result.append(child(parent,f'F1-union-without-{group}-{kind}',groups=[g for g in union if g!=group]))
    return result


def feature_finish(root):
    candidates=[r for r in completed(root,kind='lightgbm',family='F1')
                if 'production' not in spec(r).get('groups',[])
                and not spec(r).get('production_single_check')]
    if not candidates:raise RuntimeError('No non-production F1 LightGBM parent')
    parent=spec(candidates[0])
    # The original F1-10 is replayed before this adaptive wave. A failed or
    # unsupported replay still consumes the single production comparison.
    production_attempted=any(spec(r).get('production_single_check') or
                             'production' in spec(r).get('groups',[]) for r in rows(root))
    result=[]
    if not production_attempted:
        result.append(child(parent,'F1-10-production-once',
                            groups=list(dict.fromkeys([*parent['groups'],'production'])),
                            production_single_check=True))
    for k in (10,20,40):result.append(child(parent,f'F1-11-top{k}',feature_top_k=k))
    return result


def targets(root):
    # The production exception is not propagated or retuned.
    choices=[r for r in completed(root,kind='lightgbm',family='F1')
             if 'production' not in spec(r).get('groups',[]) and not spec(r).get('feature_top_k')]
    parent=spec(choices[0])
    result=[]
    for number,target in enumerate(('direct','delta','weekly','profile','b5_residual','log1p'),1):
        result.append(child(parent,f'F2-{number}',tier=1 if number in (3,5) else 3,target=target))
    for k in (2,3):result.append(child(parent,f'F2-4-profile{k}',tier=3,target='profile',profile_weeks=k))
    result.append(child(parent,'F2-7-daytype',tier=3,daytype=True))
    return result


def base_tabular(root):
    candidates=[r for r in completed(root,adapter='regression',kind='lightgbm')
                if 'production' not in spec(r).get('groups',[])]
    return spec(candidates[0])


def loss_specs():
    return [('l2',{'objective':'regression'}),('l1',{'objective':'regression_l1'}),
            *[(f'huber{a}',{'objective':'huber','alpha':a}) for a in (.5,.8,.95)],
            ('fair',{'objective':'fair'}),
            *[(f'q{q}',{'objective':'quantile','alpha':q}) for q in (.5,.55,.6)],
            ('tweedie',{'objective':'tweedie','tweedie_variance_power':1.5})]


def weight_specs():
    return [(f'w{w}',{'peak_weight':w}) for w in (1,1.5,2,3,4)]+[
        (f'a{a}',{'continuous_peak_alpha':a}) for a in (.5,1,2)]


def tabular_variants(root):
    parent=base_tabular(root);result=[]
    for name,loss in loss_specs():
        result.append(child(parent,f'F3-2-{name}',tier=2,model_params={**parent.get('model_params',{}),**loss}))
    for name,weight in weight_specs():result.append(child(parent,f'F3-3-{name}',tier=2,**weight))
    for adapter in ('xgboost','catboost'):
        result.append(child(parent,f'F3-4-{adapter}-base',tier=2,kind=adapter,model_params={}))
    from phase_f.models.multihorizon import configurations
    for cfg in configurations():result.append({**cfg,'adapter':'multihorizon','tier':2,'groups':parent['groups']})
    result.append(child(parent,'F4-3-kalman-features',tier=2,kalman_features=True))
    return result


def data_strategies(root):
    parent=base_tabular(root);result=[]
    for weeks in (4,6,8):result.append(child(parent,f'F9-1-{weeks}w',tier=3,window_weeks=weeks))
    for weeks in (1,2,4):result.append(child(parent,f'F9-2-half{weeks}w',tier=3,half_life_weeks=weeks))
    for policy in ('keep','drop','weight'):
        result.append(child(parent,f'F9-3-{policy}',tier=2,dedup=policy))
    result.append(child(parent,'F8-1-two-stage',tier=3,kind='two_stage',target='direct',base_kind='lightgbm'))
    return result


def _helpful_expanded_groups(root):
    """Only a paired weekly D1 gain over the F1 core activates default exog."""
    all_rows=completed(root,adapter='regression',kind='lightgbm',family='F1')
    core=next((row for row in all_rows if row['exp_id']=='F1-core-lightgbm'),None)
    if core is None:
        raise RuntimeError('Weekly F1 core LightGBM must finish before neural grid')
    candidates=[row for row in all_rows if spec(row).get('groups') and
                'production' not in spec(row)['groups'] and
                not spec(row).get('feature_top_k') and
                row.get('leakage_test')=='passed']
    if not candidates:return None
    candidate=candidates[0]  # EXPLORE AUC MAE rank; never choose from CONFIRM.
    if float(candidate['explore_AUC_MAE'])>=float(core['explore_AUC_MAE']):return None
    from phase_f.wf_evaluation import load_predictions
    from phase_f.wf_metrics import paired_ci
    view=SimpleNamespace(out=Path(root)/'outputs/phase_f/walkforward_v2')
    paired=paired_ci(load_predictions(view,candidate['exp_id']),
                     load_predictions(view,core['exp_id']),'EXPLORE',n=1000,seed=42)
    metric=paired.loc[paired.dataset.eq('D1') &
                      paired.metric.eq('AUC_MAE_improvement') & paired.horizon.isna()]
    if len(metric)!=1:raise ValueError('Missing paired D1 AUC MAE feature comparison')
    result=metric.iloc[0]
    if result.ci_status!='available' or not math.isfinite(float(result.ci_low)) or result.ci_low<=0:
        return None
    return list(spec(candidate)['groups'])


def _origin_exog_columns(root,groups):
    from phase_f.features_ext import build_features
    from phase_f.wf_harness import WeeklyPrepared
    prepared=WeeklyPrepared(root)
    context=prepared.contexts[(4,0)]
    frame,_=build_features(prepared.history,context['fit'][:1],4,
                           context['tau'],groups)
    if frame.empty or not len(frame.columns):
        raise RuntimeError('Expanded origin feature metadata is empty')
    return list(frame.columns)


def neural_grid(root,stage):
    from phase_f.models.neural import configurations
    result=[]
    groups=_helpful_expanded_groups(root) if stage in (2,3) else None
    columns=_origin_exog_columns(root,groups) if groups else None
    for cfg in configurations():
        number=cfg['exp_id'];kind=cfg['kind'];length=cfg['context_length']
        tier=1 if number=='F5-2' else (2 if number in ('F5-1','F5-3') else 3)
        if tier!=stage:continue
        original={'id':f'{number}-{kind}-c{length}','family':'F5','tier':tier,
            'adapter':'neural','kind':kind,'context_length':length,'seeds':[42,43,44,45,46],
            'determinism_mode':'warn_only' if kind=='nhits' else 'strict',
            'use_calendar_exog':False,'nonfinite_policy':'weekly_anchor'}
        result.append(original)
        if columns:
            result.append(child(original,f"{original['id']}-features",
                                exog_groups=groups,exog_columns=columns,
                                expanded_feature_default=True))
    return result


def neural_extensions(root):
    parents={}
    for row in completed(root,adapter='neural'):
        cfg=spec(row);parents.setdefault(cfg['kind'],cfg)
    result=[]
    groups=base_tabular(root)['groups']
    # Deterministic source column names come from feature definitions, not labels.
    from phase_f.features_ext import build_features
    from phase_f.wf_harness import WeeklyPrepared as Prepared
    prepared=Prepared(root);c=prepared.contexts[(4,0)]
    x,_=build_features(prepared.history,c['fit'][:1],4,c['tau'],groups)
    for kind,parent in parents.items():
        for loss in ('mae','huber','peak_weighted_mae','quantile'):
            result.append(child(parent,f'F5-6-{kind}-{loss}',tier=3,loss=loss))
        result.append(child(parent,f'F5-8-{kind}-features',tier=3,exog_groups=groups,exog_columns=list(x.columns)))
        if kind in ('patchtst','tide'):
            result.append(child(parent,f'F5-4-{kind}-calendar',tier=3,exog_groups=[],exog_columns=[],use_calendar_exog=True))
    return result


def foundation_fine(stage):
    from phase_f.models.foundation import configurations
    # Retain the original 12 full/LoRA, context and learning-rate records
    # byte-for-byte as experiment configurations. New budgets have new IDs.
    base=[{**s,'adapter':'foundation'} for s in configurations() if s['tier']==2]
    if stage==2:
        result=list(base)
        by_key={(s['finetune'],s['context_length'],s['learning_rate']):s for s in base}
        for mode in ('full','lora'):
            for length in (512,2048,8192):
                for lr in (1e-6,1e-5,3e-6):
                    source=by_key[(mode,length,lr if lr!=3e-6 else 1e-6)]
                    budgets=(100,250,1000) if lr!=3e-6 else (100,250,500,1000)
                    for steps in budgets:
                        exp_id=f'F6-4-{mode}-c{length}-lr{lr:g}-s{steps}'
                        result.append(child(source,exp_id,tier=2,learning_rate=lr,num_steps=steps))
        return result
    result=[]
    # The established stage-three calendar/point followups use only the
    # original parents, not the enlarged 72-cell training grid.
    for s in base:
        for q in (.5,.6,.65):
            result.append(child(s,s['id'].replace('F6-4','F6-6')+f'-calendar-q{q}',tier=3,covariates=True,point=q))
    return result


def foundation_training_expansion(root, number):
    """Finite new training settings near each mode's best EXPLORE fine-tune.

    The round number increases the radius if the same parent remains best.
    This avoids replaying an already tried physical configuration while
    allowing the orchestrator's two-non-improving-wave rule to decide when
    exploration ends. Structural context-window support is checked by
    ``validate_support`` at execution against the locked fold cohort.
    """
    if not isinstance(number,int) or number<0:
        raise ValueError('Expansion round must be a nonnegative integer')
    old=rows(root)
    seen=set()
    for row in old:
        cfg=spec(row)
        if cfg.get('adapter')=='foundation' and cfg.get('finetune') in ('full','lora'):
            seen.add(_foundation_training_key(cfg))
    result=[]
    radius=number+1
    contexts=(512,1024,2048,4096,8192)
    for mode in ('full','lora'):
        candidates=[row for row in completed(root,adapter='foundation')
                    if spec(row).get('finetune')==mode and
                    math.isfinite(float(row['explore_AUC_MAE']))]
        if not candidates:
            raise RuntimeError(f'No completed EXPLORE {mode} fine-tune parent for training expansion')
        parent=spec(candidates[0])
        length=int(parent['context_length'])
        if length not in contexts:
            raise ValueError(f'Unsupported fine-tune parent context: {length}')
        lr=float(parent['learning_rate']);steps=int(parent['num_steps'])
        if not math.isfinite(lr) or lr<=0 or steps<1:
            raise ValueError('Invalid fine-tune parent learning rate or steps')
        factor=2**radius
        proposals=[('lrhalf',length,lr/factor,steps),
                   ('lrdouble',length,lr*factor,steps),
                   ('stepshalf',length,lr,max(1,steps//factor)),
                   ('stepsdouble',length,lr,steps*factor)]
        pos=contexts.index(length)
        for neighbor in (pos-radius,pos+radius):
            if 0<=neighbor<len(contexts) and contexts[neighbor] in (512,1024,2048,4096):
                proposals.append(('context',contexts[neighbor],lr,steps))
        for axis,context,new_lr,new_steps in proposals:
            if new_lr<=0 or not math.isfinite(new_lr) or new_steps<1:
                continue
            key=(mode,context,float(new_lr),int(new_steps),
                 bool(parent.get('covariates',False)))
            if key in seen:
                continue
            seen.add(key)
            exp_id=(f'F6-expansion-{number}-{mode}-{axis}-c{context}'
                    f'-lr{new_lr:g}-s{new_steps}')
            result.append(child(parent,exp_id,tier=3,context_length=context,
                                learning_rate=new_lr,num_steps=new_steps))
    return result


def _foundation_training_key(cfg):
    return (cfg.get('finetune'),int(cfg.get('context_length',2048)),
            float(cfg.get('learning_rate',1e-6)),int(cfg.get('num_steps',500)),
            bool(cfg.get('covariates',False)))


def tabular_cross(root):
    parent=base_tabular(root)
    bundles=[[],parent.get('groups',[])]
    bundles.extend(spec(r).get('groups',[]) for r in completed(root,kind='lightgbm',family='F1')
                   if 'production' not in spec(r).get('groups',[]) and 'without' not in r['exp_id'])
    unique={tuple(v):v for v in bundles};result=[]
    for groups,(loss_name,loss),(weight_name,weight) in itertools.product(unique.values(),loss_specs(),weight_specs()):
        cfg=child(parent,f'F3-8-{config_hash(groups)[:7]}-{loss_name}-{weight_name}',tier=2,
                  groups=groups,model_params={**parent.get('model_params',{}),**loss},
                  peak_weight=1,continuous_peak_alpha=0)
        cfg.update(weight);result.append(cfg)
    return result


def ensemble_specs(root,round_name='initial'):
    from phase_f.ensemble import configurations
    # Base models must produce fit/stop-only cal predictions. Learned ensembles
    # cannot be recursively trained on their own in-sample calibration outputs.
    by_family={}
    for row in completed(root):
        cfg=spec(row)
        if cfg.get('adapter') not in ('regression','statistical','neural','foundation','other_foundation','multihorizon'):continue
        group=cfg.get('kind',cfg['family'])
        by_family.setdefault(group,[])
        if len(by_family[group])<3:by_family[group].append(cfg['id'])
    pool=list(dict.fromkeys(['F0-1-B5','F6-1-c2048-median',*[v for values in by_family.values() for v in values]]))
    sets=list(itertools.combinations(pool,2))
    lgb=best(root,kind='lightgbm')['id'];dl=best(root,adapter='neural')['id']
    sets.extend([('F0-1-B5',lgb),('F0-1-B5','F6-1-c2048-median'),('F6-1-c2048-median',lgb),
                 ('F0-1-B5','F6-1-c2048-median',lgb,dl)])
    result=[];seen=set()
    for models in sets:
        models=tuple(dict.fromkeys(models))
        if len(models)<2 or models in seen:continue
        seen.add(models)
        for op in configurations():
            if not op['id'].startswith('F7'):continue
            params={**op['params'],'cal_fit_fraction':.5}
            if op['method']=='peak_gate':
                risks=[m for m in models if m.startswith('F6') or 'quantile' in m or 'two-stage' in m]
                if not risks:continue
                params['risk_source']=risks[0]
            result.append({'id':f"{op['id']}-{round_name}-{config_hash(models)[:10]}",
                'family':'F7','tier':2,'adapter':'ensemble','method':op['method'],'bases':list(models),'params':params})
    for kind in ('lightgbm','xgboost','catboost'):
        parent=best(root,kind=kind)
        for n in (5,10,20):
            result.append(child(parent,f'F3-7-{kind}-{n}seeds-{round_name}',tier=2,adapter='seed_ensemble',seeds=list(range(42,42+n))))
    explicit={'b5-lgbm':['F0-1-B5',lgb],
              'b5-chronos':['F0-1-B5','F6-1-c2048-median'],
              'chronos-lgbm':['F6-1-c2048-median',lgb],
              'b5-chronos-lgbm-dl':['F0-1-B5','F6-1-c2048-median',lgb,dl]}
    for name,bases in explicit.items():
        result.append({'id':f'F7-6-{name}-{round_name}','family':'F7','tier':2,
                       'adapter':'ensemble','method':'nnls','bases':bases,
                       'params':{'cal_fit_fraction':.5}})
    return result


def peak_postprocessors(root):
    pool=completed(root,adapter='foundation')[:3]+completed(root,adapter='regression')[:3]
    result=[]
    for row in pool:
        base=row['exp_id'];common={'family':'F8','tier':3,'adapter':'ensemble','bases':[base]}
        result.append({**common,'id':f'F8-2-{base}','method':'bias_hour_daytype',
                       'params':{'cal_fit_fraction':.5,'min_count':8}})
        if spec(row)['adapter']=='foundation':
            for shift in (.25,.5,.75):
                result.append({**common,'id':f'F8-3-{base}-s{shift}','method':'quantile_risk_shift',
                               'params':{'shift':shift,'risk_threshold':.5}})
    return result
