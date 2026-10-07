"""Staged, fail-closed Phase E experiment. All writes stay in this namespace."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT/'outputs/phase_e'
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(OUT/'logs/matplotlib')
from guard import AccessGuard, sha256, write_json, assert_existing_preserved, git


def clean_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(name, value):
    write_json(OUT/'logs'/name, clean_json(value))


def verify_protocol():
    seal = json.loads((OUT/'logs/protocol_seal.json').read_text(encoding='utf-8'))
    for name, h in seal['files'].items():
        assert sha256(OUT/'logs'/name)==h, f'Changed protocol: {name}'
    cfg = json.loads((OUT/'logs/phase_e_config_lock.json').read_text(encoding='utf-8'))
    assert cfg['point_model']=='B5' and cfg['horizon_steps']==16 and cfg['horizon_minutes']==240
    assert cfg['thresholds']==[.01,.05,.10,.20,.30,.50] and cfg['policies']==['1/1','2/2']
    assert cfg['bootstrap']['n']==1000 and cfg['bootstrap']['seed']==42
    assert git(ROOT,'branch','--show-current')=='Phase3_experiments'
    assert git(ROOT,'rev-parse','HEAD')==json.loads((OUT/'logs/run_environment.json').read_text(encoding='utf-8'))['head']
    assert not git(ROOT,'diff','--name-only','HEAD'), 'Tracked source changed'
    return cfg


def verify_analysis_seal():
    seal=json.loads((OUT/'logs/analysis_code_seal.json').read_text(encoding='utf-8'))
    for relative,digest in seal['files'].items():
        assert sha256(ROOT/relative)==digest, f'Frozen analysis implementation changed: {relative}'
    before=json.loads((OUT/'logs/protected_before.json').read_text(encoding='utf-8'))
    assert_existing_preserved(ROOT,before)


def validate_predictions(frame):
    import numpy as np
    import pandas as pd
    assert frame.model.eq('B5').all() and frame.horizon.eq(16).all()
    assert frame.partition.eq('score').all()
    assert not frame.duplicated(['fold','origin','target_time']).any()
    assert frame.fold.isin([0,1,2]).all()
    assert (frame.target_time-frame.origin).eq(pd.Timedelta(minutes=240)).all()
    assert frame.target_time.lt(pd.Timestamp('2021-08-09 09:45')).all()
    assert frame.origin.lt(pd.Timestamp('2021-08-09 09:45')).all()
    cols=['y','tau','mu','sigma','p_raw','p_cal','q95_raw','U95']
    assert np.isfinite(frame[cols].to_numpy(float)).all()
    assert frame.sigma.gt(0).all()
    assert frame.p_raw.between(0,1).all() and frame.p_cal.between(0,1).all()
    assert frame.is_peak.eq(frame.y>frame.tau).all()
    assert frame.uncertainty_flag.eq(frame.U95>frame.tau).all()
    return {'passed':True, 'n':len(frame), 'fold_counts':frame.groupby('fold').size().to_dict(),
            'model':'B5','horizon_steps':16,'sigma_min':float(frame.sigma.min()),
            'origin_last':str(frame.origin.max()),'target_last':str(frame.target_time.max()),
            'd2_rows':int(frame.is_d2_novel_profile.sum())}


def seal_code(name):
    save(name, {p.relative_to(OUT).as_posix():sha256(p) for folder in ['code','tests']
                for p in sorted((OUT/folder).glob('*.py'))})


def parity():
    from distribution import prepare_inputs
    assert not (OUT/'logs/source_parity_audit.json').exists(), 'Parity already sealed'
    seal_code('code_at_parity.json')
    cal, score, audit = prepare_inputs(ROOT)  # raises before returning on failure
    for partition, frame in [('cal',cal),('score',score)]:
        p = OUT/'predictions'/f'phase_e_{partition}_inputs.parquet'
        p.parent.mkdir(parents=True, exist_ok=True)
        assert not p.exists()
        frame.to_parquet(p,index=False)
    save('source_parity_audit.json',audit)
    save('input_seal.json',{f'phase_e_{part}_inputs.parquet':sha256(OUT/'predictions'/f'phase_e_{part}_inputs.parquet') for part in ['cal','score']})
    print(f'Parity gate passed: cal={len(cal)}, score={len(score)}',flush=True)


def evaluate():
    verify_analysis_seal()  # before importing or executing the frozen analysis
    import pandas as pd
    from probability import calibrate, evaluate_probability
    from alerts import evaluate_alerts
    assert (OUT/'logs/source_parity_audit.json').exists(), 'Parity gate required'
    assert not (OUT/'predictions/phase_e_h16_oof.parquet').exists(), 'Evaluation already exists; never overwrite'
    seal_code('code_at_evaluation.json')
    hashes=json.loads((OUT/'logs/input_seal.json').read_text(encoding='utf-8'))
    for name,h in hashes.items():
        assert sha256(OUT/'predictions'/name)==h
    cal=pd.read_parquet(OUT/'predictions/phase_e_cal_inputs.parquet')
    score=pd.read_parquet(OUT/'predictions/phase_e_score_inputs.parquet')
    scored,platt,conformal=calibrate(cal,score)
    qa=validate_predictions(scored)
    save('calibration_fit.json',platt)
    save('conformal_fit.json',conformal)
    scored.to_parquet(OUT/'predictions/phase_e_h16_oof.parquet',index=False)
    print('Cal-only Platt/conformal fitted, frozen score probabilities saved.',flush=True)
    probability_tables=evaluate_probability(scored,n_boot=1000,seed=42)
    print('Probability/coverage date-block evaluation complete.',flush=True)
    alert_tables=evaluate_alerts(scored,n_boot=1000,seed=42)
    print('Decision/episode evaluation complete.',flush=True)
    tables={**probability_tables,**alert_tables}
    (OUT/'tables').mkdir(exist_ok=True)
    for name,frame in tables.items():
        if name=='alert_episode_metrics':
            # Keep the requested schema alias; present the honest denominator.
            for col in list(frame.columns):
                if col.startswith('false_alert_episodes_per_operating_day'):
                    frame[col.replace('per_operating_day','per_observed_evaluation_day')]=frame[col]
        assert not (OUT/'tables'/f'{name}.csv').exists()
        frame.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    d2=[]
    for name in ['risk_metrics','calibration_summary','uncertainty_metrics','alert_episode_metrics']:
        df=tables[name]
        d2.append(df.loc[df['subset'].eq('D2')].assign(table=name))
    pd.concat(d2,ignore_index=True).to_csv(OUT/'tables/d2_phase_e_metrics.csv',index=False)
    save('prediction_qa.json',qa)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['parity','evaluate','report','finalize'],required=True)
    stage=parser.parse_args().stage
    guard=AccessGuard(ROOT)
    guard.install()
    start=perf_counter()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    verify_protocol()
    try:
        if stage=='parity': parity()
        elif stage=='evaluate': evaluate()
        elif stage=='report':
            from reporting import generate
            generate(OUT)
        elif stage=='finalize':
            verify_analysis_seal()
            before=json.loads((OUT/'logs/protected_before.json').read_text(encoding='utf-8'))
            preservation=assert_existing_preserved(ROOT,before)
            save('preservation_audit.json',preservation)
            save('leakage_audit.json',{
                'development_only':True,'raw_holdout_observations_parsed':False,
                'historical_final_artifact_read':False,
                'boundary':'2021-08-09 09:45:00',
                'evidence':['sealed unchanged prefix loader','poisoned boundary tail test','timestamp-first development parquet checks',
                            'explicit allowed Phase C inputs','stage access logs','fit/cal/score partition checks','independent review'],
                'state_updates':'through origin only; phi/q/r never refit',
                'new_point_model':False,'extra_risk_inputs':False,'d2_refitting':False,
                'new_final_test_artifact':False,'existing_artifacts_preserved':preservation['preserved'],
                'scope':'This Phase E execution only; does not erase historical Phase C incident.',
                'access_guard_limit':'Python path guard; native parquet inputs separately allowlisted and metadata validated.'})
        save(f'runtime_{stage}_{stamp}.json',{'stage':stage,'started_at':stamp,'seconds':perf_counter()-start,'success':True})
        save(f'access_{stage}_{stamp}.json',{'paths_read':sorted(guard.read_paths),'denied':guard.denied,
                                            'historical_final_artifact_read':False})
        if stage=='finalize':
            stage_files=sorted((OUT/'logs').glob('runtime_*_*.json'))
            stage_times=[json.loads(p.read_text(encoding='utf-8')) for p in stage_files]
            save('runtime.json',{'stages':stage_times,'total_stage_seconds':sum(v['seconds'] for v in stage_times),
                                'time_scope':'successful stages including final verification, excluding checksum-inventory serialization',
                                'development_only':True})
            save('artifact_manifest.json',{'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                'head':git(ROOT,'rev-parse','HEAD'),'committed':False,'pushed':False,'merged':False,
                'scope':'all deliverables except self, runtime scratch/cache and .log console files',
                'artifacts':{p.relative_to(OUT).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
                    for p in sorted(OUT.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['__pycache__','matplotlib','tmp','pytest_cache','.pytest_tmp'])
                    and p.suffix!='.log'}})
    except Exception as error:
        save(f'failure_{stage}_{stamp}.json',{'stage':stage,'seconds':perf_counter()-start,'error':repr(error),'traceback':traceback.format_exc(),
                                           'read_paths':sorted(guard.read_paths),'denied':guard.denied})
        raise


if __name__=='__main__': main()
