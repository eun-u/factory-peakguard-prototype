import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f.support import UnsupportedConfiguration, validate_support


def test_full_context_rejection_preserves_all_folds():
    index = pd.date_range('2021-01-01', periods=130, freq='15min')
    context = {'fit': index[20:50], 'stop': index[70:100],
               'target_time': pd.Series(index + pd.Timedelta(hours=4), index=index),
               'summary': {'horizon_quarters': 16}}
    prepared = SimpleNamespace(history=pd.DataFrame({'power': np.arange(130.)}, index=index),
                               contexts={(16, f): context for f in (0, 1, 2)})
    spec = {'adapter': 'foundation', 'finetune': 'full', 'context_length': 8}
    validate_support(prepared, spec)
    with pytest.raises(UnsupportedConfiguration) as caught:
        validate_support(prepared, {**spec, 'context_length': 80})
    counts = caught.value.evidence['complete_window_counts']
    assert all(counts[f'fold{f}_fit'] == 0 for f in (0, 1, 2))
    assert all(counts[f'fold{f}_stop'] > 0 for f in (0, 1, 2))
    # A zero-shot input has a separately documented variable-context policy.
    validate_support(prepared, {**spec, 'finetune': None, 'context_length': 80})


def test_bolt_support_uses_pinned_config_not_assumed_512(tmp_path):
    from phase_f.models.other_foundation import MODELS
    revision = MODELS['chronos_bolt']['revision']
    path = tmp_path / 'models/hf_optional/models--amazon--chronos-bolt-small/snapshots' / revision / 'config.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'chronos_config': {'context_length': 2048}}))
    prepared = SimpleNamespace(out=tmp_path)
    spec = {'adapter': 'other_foundation', 'kind': 'chronos_bolt', 'context_length': 2048}
    validate_support(prepared, spec)
    with pytest.raises(UnsupportedConfiguration) as caught:
        validate_support(prepared, {**spec, 'context_length': 2049})
    assert caught.value.evidence['supported_context_length'] == 2048
    assert len(caught.value.evidence['model_config_sha256']) == 64


def test_peak_threshold_changes_neural_checkpoint_identity():
    from phase_f.models.neural import _fingerprint
    index = pd.date_range('2021-01-01', periods=4, freq='15min')
    arrays = (np.arange(8.).reshape(2, 4),)
    args = (index[:2], index[2:], arrays, arrays, {'kind': 'dlinear', 'loss': 'peak_weighted_mae'}, 4)
    assert _fingerprint(*args, 10.) != _fingerprint(*args, 11.)
    assert _fingerprint(*args, 10.) == _fingerprint(*args, 10.)


def test_tweedie_residual_domain_mismatch_is_recorded_before_fitting():
    from phase_f.models.regression import _fit_estimator
    with pytest.raises(UnsupportedConfiguration) as caught:
        _fit_estimator('lightgbm',np.ones((3,2)),np.array([-1.,0.,2.]),np.ones(3),
                       {'objective':'tweedie'},42,None,10)
    assert caught.value.evidence['fit_negative_labels']==1
