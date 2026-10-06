"""Post-observation EXPLORE error analysis, never model inputs or selection."""
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

from phase_f.features_ext import _power
from phase_f.registry import now, sha256, write_json
from phase_f.wf_harness import WeeklyPrepared


def analyze(root, candidate, namespace='goal_r1_v2'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', candidate):
        raise ValueError('Unsafe candidate ID')
    if not re.fullmatch(r'goal_r1_[a-z0-9_]+', namespace):
        raise ValueError('Unsafe namespace')
    root = Path(root)
    prepared = WeeklyPrepared(root)
    base = root/'outputs/phase_f/walkforward_v2/predictions/EXPLORE/F0-1-R1.parquet'
    cand = root/'outputs/phase_f'/namespace/'predictions/EXPLORE'/f'{candidate}.parquet'
    frames = [pd.read_parquet(p) for p in (base, cand)]
    b, c = [f.loc[f.role.eq('score') & f.arm.eq('EXPLORE')].copy() for f in frames]
    keys = ['origin', 'horizon', 'fold', 'role', 'arm']
    x = b.merge(c[keys+['pred']], on=keys, validate='one_to_one', suffixes=('_r1','_candidate'))
    if len(x) != len(b) or len(x) != len(c):
        raise ValueError('Diagnostics require the same complete evaluation cohort')
    known = _power(prepared.history)
    x['known_power'] = known.reindex(pd.DatetimeIndex(x.origin)).to_numpy(float)
    x['known_ramp_1h'] = (x.known_power.to_numpy(float)
        - known.reindex(pd.DatetimeIndex(x.origin)-pd.Timedelta(hours=1)).to_numpy(float))
    peak = x.y.gt(x.tau)
    observed = x.known_power.notna()
    origin_high = x.known_power.gt(x.tau)
    x['known_ramp_direction'] = np.select(
        [x.known_ramp_1h.isna(), x.known_ramp_1h.gt(.1*x.fit_mean),
         x.known_ramp_1h.lt(-.1*x.fit_mean)],
        ['missing', 'rising_gt_10pct_fit_mean', 'falling_gt_10pct_fit_mean'], default='stable')
    rows = []

    def summarize(label, mask):
        for horizon in (None, 4, 16):
            z = x.loc[mask & (True if horizon is None else x.horizon.eq(horizon))]
            if z.empty:
                continue
            rows.append({'group':label, 'horizon':horizon or 'all13', 'rows':len(z),
                'R1_MAE':float((z.pred_r1-z.y).abs().mean()),
                'candidate_MAE':float((z.pred_candidate-z.y).abs().mean()),
                'R1_bias':float((z.pred_r1-z.y).mean()),
                'candidate_bias':float((z.pred_candidate-z.y).mean()),
                'mean_applied_correction':float((z.pred_candidate-z.pred_r1).mean()),
                'mean_future_change_outcome_only':float((z.y-z.known_power).mean())})

    for label, mask in [('all',np.ones(len(x),bool)), ('peak_at_target',peak),
        ('peak_origin_already_high',peak & observed & origin_high),
        ('peak_origin_not_high',peak & observed & ~origin_high),
        ('peak_origin_missing',peak & ~observed)]:
        summarize(label, mask)
    for direction in sorted(x.known_ramp_direction.unique()):
        summarize('peak_known_ramp_'+direction, peak & x.known_ramp_direction.eq(direction))
    out = root/'outputs/phase_f'/namespace/'diagnostics'
    out.mkdir(exist_ok=True)
    table = out/f'{candidate}_vs_R1_origin_regimes.csv'
    pd.DataFrame(rows).to_csv(table, index=False)
    write_json(table.with_suffix('.json'), {'at':now(), 'candidate':candidate,
        'sources':{str(p.relative_to(root)):sha256(p) for p in (base,cand)},
        'artifact_sha256':sha256(table), 'source_sha256':sha256(Path(__file__)),
        'analysis_only':True, 'target_outcomes_not_features':True,
        'aggregation':'all13 rows pooled; official target uses equal mean of 13 horizon metrics',
        'known_ramp_threshold':'.1 times FIT-only mean; posthoc diagnostic only',
        'arm':'EXPLORE', 'holdout_read':False})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--namespace', default='goal_r1_v2')
    args = parser.parse_args()
    print(analyze(Path(__file__).resolve().parents[1], args.candidate, args.namespace).to_string(index=False))
