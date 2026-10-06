"""Reproducible post-observation EXPLORE motivation; never a model input."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from phase_f.features_ext import _power
from phase_f.models.r1_residual import _common_roles
from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path
from phase_f.wf_harness import WeeklyPrepared
from phase_f.wf_models import _arm_view, _validate_frame
from phase_f.wf_registry import WFRegistry
from phase_f.wf_run import cache_identity


def analyze(root):
    root = Path(root)
    original = WeeklyPrepared(root)
    view = _arm_view(original, 'EXPLORE')
    path = prediction_path(original, BASELINES['R1'])
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    spec = json.loads(WFRegistry(root).read(BASELINES['R1'])['config_json'])
    if meta['identity'] != cache_identity(original, spec):
        raise ValueError('R1 source identity changed')
    frame = load_predictions(original, BASELINES['R1'])
    _validate_frame(view, frame, BASELINES['R1'], 'EXPLORE')
    frame = frame.loc[frame.role.eq('score')].copy()
    power = _power(view.history)
    origins = pd.DatetimeIndex(frame.origin)
    frame['known_power'] = power.reindex(origins).to_numpy(float)
    frame['past_ramp_1h'] = (frame.known_power.to_numpy(float)
        - power.reindex(origins - pd.Timedelta(hours=1)).to_numpy(float))
    if not np.isfinite(frame.known_power).all():
        raise ValueError('Locked cohort has unavailable origin power')
    frame['future_change_outcome_only'] = frame.y - frame.known_power
    frame['absolute_error'] = (frame.pred - frame.y).abs()
    frame['class_outcome_only'] = np.select(
        [frame.future_change_outcome_only.lt(-30), frame.future_change_outcome_only.gt(30)],
        ['fall', 'rise'], default='neutral')
    rows = []
    novel = frame.loc[frame.d2.astype(bool)]
    for scope, subset in [('all13', frame), ('all13_D2', novel),
                          *[(f'h{h}', g) for h, g in frame.groupby('horizon')]]:
        if subset.empty:
            continue
        for label, part in subset.groupby('class_outcome_only'):
            rows.append({'scope': scope, 'outcome_class': label, 'rows': len(part),
                'row_share': len(part) / len(subset),
                'R1_MAE': float(part.absolute_error.mean()),
                'absolute_error_share': float(part.absolute_error.sum() / subset.absolute_error.sum())})
    conditional = []
    for fold, subset in [('all8', frame), ('all8_D2', novel),
                         *[(int(f), g) for f, g in frame.groupby('fold')]]:
        if subset.empty:
            continue
        for name, mask, outcome in [
            ('past_fall_to_future_rise', subset.past_ramp_1h.lt(-30), 'rise'),
            ('past_rise_to_future_fall', subset.past_ramp_1h.gt(30), 'fall')]:
            part = subset.loc[mask]
            base_rate = float(subset.class_outcome_only.eq(outcome).mean())
            rate = float(part.class_outcome_only.eq(outcome).mean()) if len(part) else None
            conditional.append({'fold': fold, 'condition': name, 'rows': len(part),
                'conditional_rate': rate, 'base_rate': base_rate,
                'lift': rate / base_rate if rate is not None and base_rate > 0 else None})
    support = []
    for fold in sorted({f for _, f in view.contexts}):
        fit, stop, embargo = _common_roles(view.contexts, fold)
        for role, times in [('fit', fit), ('stop', stop)]:
            parts = []
            for h in range(4, 17):
                context = view.contexts[(h, fold)]
                y = context['y'].loc[times].to_numpy(float)
                delta = y - power.reindex(times).to_numpy(float)
                parts.append(pd.DataFrame({'class': np.select([delta < -30, delta > 30],
                    [0, 2], default=1), 'peak': y > context['tau']}))
            part = pd.concat(parts, ignore_index=True)
            support.append({'fold': fold, 'role': role, 'rows': len(part),
                'fall_rows': int(part['class'].eq(0).sum()),
                'neutral_rows': int(part['class'].eq(1).sum()),
                'rise_rows': int(part['class'].eq(2).sum()),
                'peak_rows': int(part.peak.sum()), 'embargo': config_hash(embargo)})
    out = root / 'outputs/phase_f/goal_r1_transition_v1/diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    for name, records in [('error_cohorts', rows), ('past_future_ramp', conditional),
                          ('fit_stop_class_support', support)]:
        pd.DataFrame(records).to_csv(out / (name + '.csv'), index=False)
    result = {'at': now(), 'analysis_only': True, 'arm': 'EXPLORE', 'score_rows': len(frame),
        'threshold_data_units': 30, 'threshold_origin': 'Post-observation EXPLORE error diagnosis',
        'aggregation': 'Diagnostic rows pooled; official target is equal mean of 13 horizons',
        'future_outcomes_used_as_inputs': False, 'independent_validation': 'pending',
        'prediction_sha256': sha256(path), 'metadata_sha256': sha256(path.with_suffix('.json')),
        'source_sha256': sha256(Path(__file__)), 'split_sha256': view.split_lock['lock_sha256'],
        'raw_sha256': view.seal['raw_sha256'], 'holdout_read': False,
        'historical_final_artifact_read': False,
        'artifacts': {p.name: sha256(p) for p in out.glob('*.csv')}}
    write_json(out / 'manifest.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(Path(__file__).resolve().parents[1]), ensure_ascii=False))
