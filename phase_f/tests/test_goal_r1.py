import numpy as np
import pandas as pd
import pytest

from phase_f.goal_protocol import meets_targets
from phase_f.goal_r1 import _target_fields


def tables():
    return {'auc': pd.DataFrame([{'dataset':'D1','AUC_MAE':4.,'AUC_PeakMAE':8.}]),
            'pooled':pd.DataFrame({'dataset':['D1']*13,'horizon':range(4,17),
                                  'MAE':[3.]*13,'Peak_MAE':[8.]*13,'nMAE':[.04]*13})}


def test_one_missing_normalization_cannot_pass_goal():
    value = tables()
    value['pooled'].loc[5,'nMAE'] = np.nan
    assert not meets_targets(_target_fields(value))['targets_met']


def test_missing_or_duplicate_horizon_rejected():
    value = tables()
    value['pooled'].loc[5,'horizon'] = 4
    with pytest.raises(ValueError, match='every horizon'):
        _target_fields(value)
