"""Evidence-based rejection of structurally unavailable fixed-cohort settings."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


class UnsupportedConfiguration(ValueError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = evidence


def validate_support(prepared, spec):
    if spec.get('adapter') == 'foundation' and spec.get('finetune'):
        length = int(spec.get('context_length', 2048))
        counts = {}
        for (_, fold), context in sorted(prepared.contexts.items()):
            if context['summary']['horizon_quarters'] != 16:
                continue
            for role in ('fit', 'stop'):
                allowed = set(pd.DatetimeIndex(context['target_time'].loc[context[role]]))
                count = 0
                for origin in context[role]:
                    pos = prepared.history.index.get_loc(origin)
                    if pos + 1 < length or pos + 16 >= len(prepared.history):
                        continue
                    targets = prepared.history.index[pos + 1:pos + 17]
                    if all(t in allowed for t in targets) and np.isfinite(
                            prepared.history.power.iloc[pos + 1:pos + 17]).all():
                        count += 1
                counts[f'fold{fold}_{role}'] = count
        if counts and not all(counts.values()):
            raise UnsupportedConfiguration(
                'Complete requested-context training windows are unavailable in the fixed three-fold cohort',
                {'context_length': length, 'prediction_length': 16, 'complete_window_counts': counts,
                 'policy': 'No fold removal, silent context truncation, or future training labels'})
    if spec.get('adapter') == 'other_foundation' and spec.get('kind') == 'chronos_bolt':
        from phase_f.models.other_foundation import MODELS
        info = MODELS['chronos_bolt']
        snapshot = prepared.out / 'models/hf_optional/models--amazon--chronos-bolt-small/snapshots' / info['revision']
        path = snapshot / 'config.json'
        if path.is_file():
            config = json.loads(path.read_text(encoding='utf-8'))
            limit = config['chronos_config']['context_length']
            if int(spec['context_length']) > int(limit):
                from phase_f.registry import sha256
                raise UnsupportedConfiguration('Requested context exceeds pinned Chronos-Bolt model support',
                    {'requested_context_length': int(spec['context_length']), 'supported_context_length': int(limit),
                     'model_revision': info['revision'], 'model_config_sha256': sha256(path)})
