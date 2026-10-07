"""Checkpointed finite Optuna batches, with no wall-clock termination shortcut."""
from __future__ import annotations
import json
from pathlib import Path
import math
import joblib
import optuna
from phase_f.registry import Registry,write_json,config_hash
from phase_f.experiment_plan import best,base_tabular,child,lock_wave
from phase_f.selection import explore_eligibility


def eligible_best(root):
    registry=Registry(root)
    qualified=[r for r in explore_eligibility(root) if r['eligible']]
    return min((registry.read(r['exp_id'])['explore_AUC_MAE'] for r in qualified),default=None)


def meaningful(before,after):
    return after is not None and (before is None or after<=before*.995)


def neural_parameters(trial,kind):
    from phase_f.models.neural import suggest_parameters
    return suggest_parameters(trial,kind)


def run_search(prepared,kind,*,group='gbdt',minimum=500,batch_size=100,max_batches=None,label=None):
    """Trials fit all 13 horizons / 3 folds; objective is original-scale stop MAE.

    EXPLORE metrics are recorded for every completed trial, never used to fit
    weights. Every batch has a fixed size and is recorded before it starts.
    `max_batches` is only used by predeclared expansion waves, not a time budget.
    """
    from phase_f.run import execute
    from phase_f.models.regression import suggest_parameters
    name=label or f'{group}_{kind}'
    state_path=prepared.out/'logs/tuning'/f'{name}.json'
    if state_path.exists():state=json.loads(state_path.read_text(encoding='utf-8'))
    else:
        parent=base_tabular(prepared.root) if group=='gbdt' else best(prepared.root,adapter='neural',kind=kind)
        if group=='gbdt':parent={**parent,'kind':kind,'model_params':{}}
        state={'name':name,'kind':kind,'group':group,'parent':parent,'minimum':minimum,
               'batch_size':batch_size,'completed_batches':0,'nonimproving_batches':0,'batches':[],
               'objective':'mean original-scale stop MAE over 39 horizon/fold cells',
               'best_eligible_before':eligible_best(prepared.root),'complete':False,
               'historical_final_artifact_read':False,'holdout_read':False}
        write_json(state_path,state,exclusive=True)
    if state['complete']:return state
    sampler_path=prepared.out/'cache'/f'{name}_sampler.joblib'
    sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=42,n_startup_trials=25)
    storage='sqlite:///'+str((prepared.out/'logs'/f'{name}.sqlite').resolve()).replace('\\','/')
    study=optuna.create_study(study_name=name,storage=storage,sampler=sampler,direction='minimize',load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    registry=Registry(prepared.root)
    while True:
        successes=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        if successes>=minimum and (state['nonimproving_batches']>=2 or
                (max_batches is not None and state['completed_batches']>=max_batches)):
            state['complete']=True;state['completed_trials']=successes;write_json(state_path,state);return state
        batch=state['completed_batches']
        # Freeze exact trial-number bounds before the adaptive suggestions within
        # that finite batch. TPE may use stop objectives of earlier trials only.
        batch_path=prepared.out/'logs/tuning'/f'{name}_batch{batch:03d}.json'
        if batch_path.exists():locked=json.loads(batch_path.read_text(encoding='utf-8'))
        else:
            locked={'start':len(study.trials),'end':len(study.trials)+batch_size,
                    'before':eligible_best(prepared.root),'objective':state['objective'],
                    'hypothesis':'Optimize structure on stop; check eligible EXPLORE improvement after the finite batch'}
            write_json(batch_path,locked,exclusive=True)
        failures=0
        while len([t for t in study.trials if t.state.is_finished()])<locked['end']:
            running=[t for t in study.trials if t.state==optuna.trial.TrialState.RUNNING]
            trial=optuna.trial.Trial(study,running[0]._trial_id) if running else study.ask()
            identifier=('F3-1' if kind=='lightgbm' else 'F3-4') if group=='gbdt' else 'F5-tune'
            cfg=child(state['parent'],f'{identifier}-{name}-t{trial.number:05d}',tier=2 if group=='gbdt' else 3)
            if group=='gbdt':cfg['model_params']=suggest_parameters(trial,kind)
            else:cfg.update(neural_parameters(trial,kind))
            cfg['note']=f'Stop-only TPE trial {trial.number}; finite batch {batch}; parent locked in {name}'
            row=execute(prepared,cfg,registry)
            value=row.get('stop_MAE')
            if row['status']=='completed' and row.get('stop_cells')==39 and value is not None and math.isfinite(value):
                study.tell(trial,float(value));failures=0
            else:
                study.tell(trial,state=optuna.trial.TrialState.FAIL);failures+=1
            temp=sampler_path.with_suffix('.tmp');joblib.dump(study.sampler,temp);temp.replace(sampler_path)
            if failures>=3:
                raise RuntimeError(f'Three consecutive {name} fit/objective failures: inspect registry and repair before resuming')
        after=eligible_best(prepared.root)
        improved=meaningful(locked['before'],after)
        state['nonimproving_batches']=0 if improved else state['nonimproving_batches']+1
        state['completed_batches']+=1
        state['batches'].append({'batch':batch,'before':locked['before'],'after':after,'meaningful_improvement':improved})
        write_json(state_path,state)
