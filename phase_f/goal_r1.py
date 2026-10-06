"""Focused, resumable EXPLORE experiments toward the user's absolute targets.

This runner never opens candidate CONFIRM or holdout. The existing full Phase F
search remains a separate process/evidence namespace. Attaining an EXPLORE
target is a candidate result, not achievement of the verified thread goal.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from phase_f.goal_protocol import CONTRACT, SEEDS, initial_specs, meets_targets
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path
from phase_f import wf_metrics


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sources(root):
    names = ('phase_f/goal_r1.py', 'phase_f/goal_protocol.py', 'phase_f/goal_r1_paths.py',
        'phase_f/models/r1_residual.py', 'phase_f/models/regression.py',
        'phase_f/models/foundation.py', 'phase_f/features_ext.py',
        'phase_f/wf_models.py', 'phase_f/wf_metrics.py', 'phase_f/harness.py',
        'phase_f/wf_harness.py', 'phase_c/data.py', 'src/targets.py',
        'src/holidays.py', 'src/models/cbl.py')
    return {name: sha256(Path(root)/name) for name in names}


def prepare(root, namespace='goal_r1_v1'):
    from phase_f.wf_harness import WeeklyPrepared
    from phase_f.wf_models import _arm_view
    from phase_f.wf_final import _junction
    if not namespace.startswith('goal_r1_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in namespace):
        raise ValueError('Unsafe goal experiment namespace')
    original = WeeklyPrepared(Path(root))
    local = copy.copy(original)
    local.out = original.root/'outputs/phase_f'/namespace
    local.out.mkdir(parents=True, exist_ok=True)
    _junction(local.out/'models', Path(r'D:\PeakGuard_PhaseF_20261003\models')/namespace)
    return original, _arm_view(local, 'EXPLORE')


def copy_baselines(original, prepared):
    from phase_f.wf_run import cache_identity
    from phase_f.wf_registry import WFRegistry
    evidence = {}
    for key in BASELINES.values():
        src = prediction_path(original, key)
        record = _read(src.with_suffix('.json'))
        spec = json.loads(WFRegistry(original.root).read(key)['config_json'])
        if record['identity'] != cache_identity(original, spec):
            raise ValueError('Baseline source identity changed before target experiment')
        load_predictions(original, key)  # Verify mean prediction bytes before copying.
        dest = prediction_path(prepared, key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(src)
        if dest.exists() and sha256(dest) != digest:
            raise ValueError('Goal baseline changed; preserve namespace and create a new one')
        if not dest.exists():
            temporary = dest.with_suffix('.tmp')
            shutil.copyfile(src, temporary)
            temporary.replace(dest)
        write_json(dest.with_suffix('.json'), {'arm': 'EXPLORE', 'sha256': digest,
            'source': src.relative_to(original.root).as_posix(),
            'source_metadata_sha256': sha256(src.with_suffix('.json'))}, exclusive=True)
        evidence[key] = digest
    return evidence


def baseline_diagnostics(prepared):
    """Observed future outcomes are used here only to describe score errors."""
    from src.holidays import calendar_flags
    dest = prepared.out/'diagnostics'
    dest.mkdir(parents=True, exist_ok=True)
    f = load_predictions(prepared, BASELINES['R1'])
    f = f.loc[f.role.eq('score') & f.arm.eq('EXPLORE')].copy()
    f['error'] = f.pred-f.y
    f['absolute_error'] = f.error.abs()
    f['offday'] = calendar_flags(pd.DatetimeIndex(f.target_time)).is_offday.to_numpy(bool)
    f['hour'] = pd.DatetimeIndex(f.target_time).hour
    f['peak_outcome'] = f.y.gt(f.tau)
    rows = []
    conditions = {'all': np.ones(len(f), bool), 'peak': f.peak_outcome,
        'non_peak': ~f.peak_outcome, 'workday': ~f.offday, 'offday': f.offday,
        'morning_06_10': f.hour.ge(6) & f.hour.lt(10)}
    for name, mask in conditions.items():
        part = f.loc[mask]
        for h, data in part.groupby('horizon'):
            rows.append({'condition': name, 'horizon': int(h), 'rows': len(data),
                'MAE': float(data.absolute_error.mean()), 'bias_pred_minus_actual': float(data.error.mean()),
                'p95_absolute_error': float(data.absolute_error.quantile(.95)),
                'evaluation_only_outcome_group': name in ('peak', 'non_peak')})
    pd.DataFrame(rows).to_csv(dest/'R1_error_regimes.csv', index=False)
    tables = wf_metrics.evaluate(f)
    tables['pooled'].to_csv(dest/'R1_horizon_metrics.csv', index=False)
    d1 = tables['pooled'].loc[tables['pooled'].dataset.eq('D1')]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(d1.horizon/4, d1.MAE, marker='o', label='R1 MAE')
        ax.plot(d1.horizon/4, d1.Peak_MAE, marker='o', label='R1 peak MAE')
        ax.axhline(4.5, color='C0', linestyle='--', label='Overall MAE target (4.5)')
        ax.axhline(9., color='C1', linestyle='--', label='Overall peak MAE target (9)')
        ax.set(xlabel='Forecast hours', ylabel='Measured units (unverified)')
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(dest/'R1_error_by_horizon.png', dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    write_json(dest/'manifest.json', {'at': now(), 'prediction_sha256': sha256(prediction_path(prepared, BASELINES['R1'])),
        'artifacts': {p.name:sha256(p) for p in dest.glob('*') if p.is_file() and p.name!='manifest.json'},
        'analysis_arm': 'EXPLORE', 'holdout_read': False, 'outcomes_are_not_model_features': True})


def _publish(path, frame, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    frame.to_parquet(temporary, index=False); temporary.replace(path)
    write_json(path.with_suffix('.json'), {**metadata, 'sha256': sha256(path), 'arm': 'EXPLORE'})


def _cached(path, identity):
    from phase_f.run import complete_prediction_cache
    metadata = path.with_suffix('.json')
    if metadata.exists() and not path.exists():
        backup = metadata.with_name(metadata.stem+'.orphan-'+sha256(metadata)[:12]+'.json')
        if backup.exists(): raise ValueError('Duplicate orphan metadata; preserve and investigate')
        metadata.rename(backup)
    if not complete_prediction_cache(path, metadata): return None
    record = _read(metadata)
    if record.get('identity') != identity or record['sha256'] != sha256(path):
        raise ValueError('Goal forecast identity changed; create a new namespace/configuration')
    return pd.read_parquet(path), record['audit']


def _target_fields(tables):
    d1 = tables['auc'].loc[tables['auc'].dataset.eq('D1')].iloc[0]
    pooled = tables['pooled'].loc[tables['pooled'].dataset.eq('D1')]
    if sorted(pooled.horizon.tolist()) != list(range(4, 17)):
        raise ValueError('Target evaluation requires every horizon exactly once')
    return {'AUC_MAE':float(d1.AUC_MAE), 'AUC_PeakMAE':float(d1.AUC_PeakMAE),
        'AUC_nMAE':float(pooled.nMAE.to_numpy(float).mean()),
        'h4_MAE':float(pooled.loc[pooled.horizon.eq(4),'MAE'].iloc[0]),
        'h16_MAE':float(pooled.loc[pooled.horizon.eq(16),'MAE'].iloc[0]),
        'h4_PeakMAE':float(pooled.loc[pooled.horizon.eq(4),'Peak_MAE'].iloc[0]),
        'h16_PeakMAE':float(pooled.loc[pooled.horizon.eq(16),'Peak_MAE'].iloc[0])}


def score(prepared, spec, frame, audit):
    dest = prepared.out/'tables'/spec['id']/'EXPLORE'; dest.mkdir(parents=True, exist_ok=True)
    tables = wf_metrics.evaluate(frame)
    fields = _target_fields(tables)
    for name, table in tables.items(): table.to_csv(dest/(name+'.csv'), index=False)
    pairs = []
    for name in ('B5', 'R1'):
        paired = wf_metrics.paired_ci(frame, load_predictions(prepared, BASELINES[name]), n=1000, seed=42)
        paired['baseline'] = name; pairs.append(paired)
    pd.concat(pairs, ignore_index=True).to_csv(dest/'paired_ci.csv', index=False)
    seed_rows = []
    for entry in audit['seeds']:
        path = prepared.out/'predictions/seeds/EXPLORE'/spec['id']/f"seed_{entry['seed']}.parquet"
        if sha256(path) != entry['prediction_sha256']: raise ValueError('Goal seed forecast changed')
        seed_rows.append({'seed':entry['seed'], **_target_fields(wf_metrics.evaluate(pd.read_parquet(path)))})
    pd.DataFrame(seed_rows).to_csv(dest/'seed_metrics.csv', index=False)
    uncertainty = {name: {'individual_mean':float(np.mean([row[name] for row in seed_rows])),
        'individual_sd':float(np.std([row[name] for row in seed_rows], ddof=1))} for name in fields}
    record = {'at':now(), 'candidate':spec['id'], 'arm':'EXPLORE', 'fields':fields,
        **meets_targets(fields), 'seed_uncertainty':uncertainty, 'n_seeds':len(seed_rows),
        'ranking_basis':'metrics of mean forecasts', 'independent_confirmation':'pending',
        'goal_achieved':False, 'holdout_read':False, 'leakage_test':audit['leakage_test'],
        'prediction_sha256':sha256(prediction_path(prepared,spec['id'])),
        'artifacts':{p.name:sha256(p) for p in dest.glob('*.csv')}}
    write_json(dest/'manifest.json', record)
    return record


def run_search(prepared, paths, rolling_audit, baseline_hashes, *, only=None):
    from phase_f.models.r1_residual import run
    from phase_f.wf_models import _mean_frames, _validate_frame
    from phase_f.goal_r1_paths import _hash_frame, _sort_paths
    manifest = _read(rolling_audit['manifest'])
    checksum = manifest.pop('manifest_sha256', None)
    paths_sha256 = _hash_frame(_sort_paths(paths))
    if (checksum != config_hash(manifest) or not manifest.get('complete')
        or manifest.get('identity_hash') != rolling_audit.get('identity_hash')
        or manifest.get('paths_sha256') != paths_sha256):
        raise ValueError('Rolling path bytes or provenance changed before search')
    source = _sources(prepared.root)
    if only and only not in {s['id'] for s in initial_specs()}:
        raise ValueError('Unknown goal configuration: '+only)
    planpath = prepared.out/'logs/execution_plan.json'
    plan = {'contract': CONTRACT, 'specs':initial_specs(), 'seeds':list(SEEDS),
        'sources':source, 'baseline_prediction_sha256':baseline_hashes,
        'split_sha256':prepared.split_lock['lock_sha256'],
        'rolling_paths_sha256':paths_sha256}
    write_json(planpath, plan, exclusive=True)
    paths.attrs['causal_provenance'] = rolling_audit
    rolling_digest = config_hash(rolling_audit)
    for spec in plan['specs']:
        if only and spec['id'] != only: continue
        if (prepared.out/'logs/stop_requested.json').exists(): raise RuntimeError('Requested checkpoint stop')
        identity = config_hash({'spec':spec, 'rolling':rolling_digest, 'plan':config_hash(plan)})
        recordpath = prepared.out/'logs/experiments'/f"{spec['id']}.json"
        write_json(recordpath, {'spec':spec, 'status':'running', 'at':now(), 'holdout_read':False})
        meanpath = prediction_path(prepared, spec['id'])
        cached = _cached(meanpath, identity)
        if cached is None:
            frames, seeds = [], []
            for seed in SEEDS:
                child = copy.deepcopy(spec)
                child.update(seed=seed, rolling_audit_sha256=rolling_digest)
                child['id'] = spec['id']+'__seed'+str(seed)
                path = prepared.out/'predictions/seeds/EXPLORE'/spec['id']/f'seed_{seed}.parquet'
                seed_identity = config_hash({'mean_identity':identity, 'child':child})
                value = _cached(path, seed_identity)
                if value is None:
                    frame, seed_audit = run(prepared, child, paths)
                    _validate_frame(prepared, frame, child['id'], 'EXPLORE')
                    _publish(path, frame, {'identity':seed_identity, 'audit':seed_audit})
                else: frame, seed_audit = value
                if seed_audit.get('leakage_test') != 'passed': raise ValueError('Unverified residual model leakage audit')
                frames.append(frame)
                seeds.append({'seed':seed, 'prediction_sha256':sha256(path), 'audit':seed_audit})
                print('SEED_READY', spec['id'], seed, flush=True)
            frame = _mean_frames(frames, spec['id'])
            if 'correction' in frame: frame.drop(columns='correction', inplace=True)
            if 'r1' in frame: frame['applied_correction'] = frame.pred-frame.r1
            _validate_frame(prepared, frame, spec['id'], 'EXPLORE')
            audit = {'n_seeds':len(SEEDS), 'seeds':seeds, 'leakage_test':'passed',
                'rolling_audit_sha256':rolling_digest, 'holdout_read':False}
            _publish(meanpath, frame, {'identity':identity, 'audit':audit})
        else: frame, audit = cached
        result = score(prepared, spec, frame, audit)
        write_json(recordpath, {'spec':spec, 'status':'completed', 'result':result, 'holdout_read':False})
        print('CANDIDATE_RESULT', json.dumps({'id':spec['id'], **result['fields'],
            'explore_targets_met':result['targets_met']}), flush=True)
        update_progress(prepared)


def update_progress(prepared):
    rows = [_read(p) for p in (prepared.out/'logs/experiments').glob('*.json')]
    completed = sorted([r for r in rows if r['status']=='completed'],
        key=lambda r:(r['result']['fields']['AUC_MAE'], r['spec']['id']))
    write_json(prepared.out/'logs/goal_progress.json', {'at':now(), 'contract':CONTRACT,
        'completed_candidates':len(completed), 'best':completed[0] if completed else None,
        'explore_targets_met':any(r['result']['targets_met'] for r in completed),
        'independent_confirmation':'pending', 'goal_achieved':False, 'holdout_read':False})
    lines = ['# Absolute performance target experiment', '',
        'EXPLORE candidate results; independent confirmation is pending. No holdout read.', '',
        '| Candidate | MAE | Peak MAE | h4 MAE | h16 MAE | nMAE % | Targets met |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in completed:
        f = row['result']['fields']
        lines.append(f"| {row['spec']['id']} | {f['AUC_MAE']:.4f} | {f['AUC_PeakMAE']:.4f} | {f['h4_MAE']:.4f} | {f['h16_MAE']:.4f} | {f['AUC_nMAE']*100:.2f} | {row['result']['targets_met']} |")
    (prepared.out/'GOAL_PROGRESS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('analyze','paths','search','all'), default='all')
    parser.add_argument('--namespace', default='goal_r1_v1')
    parser.add_argument('--only')
    parser.add_argument('--device')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    original, prepared = prepare(root, args.namespace)
    statuspath = prepared.out/'logs/driver_status.json'
    write_json(statuspath, {'status':'running', 'pid':os.getpid(), 'stage':args.stage,
        'started_at':now(), 'goal_achieved':False, 'holdout_read':False})
    try:
        baselines = copy_baselines(original, prepared)
        write_json(prepared.out/'logs/target_contract.json', CONTRACT, exclusive=True)
        baseline_diagnostics(prepared)
        if args.stage != 'analyze':
            from phase_f.goal_r1_paths import load_or_build
            anchor = load_predictions(original, BASELINES['R1'])
            paths, audit = load_or_build(prepared, prepared.out, anchor=anchor, device=args.device)
            if args.stage in ('search','all'): run_search(prepared, paths, audit, baselines, only=args.only)
        update_progress(prepared)
        write_json(statuspath, {'status':'completed_explore_stage', 'stage':args.stage, 'at':now(),
            'configuration_subset':args.only,
            'full_explore_search_complete':args.stage in ('search','all') and not args.only,
            'independent_confirmation':'pending', 'goal_achieved':False, 'holdout_read':False})
    except BaseException as exc:
        write_json(statuspath, {'status':'failed', 'at':now(), 'error_type':type(exc).__name__,
            'error':str(exc), 'goal_achieved':False, 'holdout_read':False})
        raise


if __name__ == '__main__':
    main()
