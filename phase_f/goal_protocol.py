"""User-approved point-forecast targets; independent proof is still required."""
from phase_f.registry import config_hash


TARGETS = {
    'AUC_MAE': 4.5,
    'AUC_PeakMAE': 9.0,
    'h4_MAE': 3.5,
    'h16_MAE': 5.5,
    'AUC_nMAE': .05,
}
SEEDS = (42, 123, 2024, 3407, 777)
CONTRACT = {
    'protocol': 'goal_r1_v1',
    'targets': TARGETS,
    'target_authorization': 'User accepted competitive targets and requested execution',
    'primary_geometry': 'existing walkforward_v2; 13 horizons h4..h16',
    'ranking': 'metrics of five-seed mean predictions; individual mean and SD disclosed',
    'training': 'fit-only causal rolling R1 residuals; stop-only early stopping and attenuation',
    'selection_arm': 'EXPLORE',
    'confirmation': 'locked candidate set; independent WF CONFIRM and pc3 auxiliary before achievement',
    'holdout_read': False,
    'commercial_qualification': 'not implied by target attainment on this single-site development set',
    'past_result_used_for_design': 'R1 EXPLORE underprediction and morning error diagnostics',
}
CONTRACT['contract_sha256'] = config_hash(CONTRACT)


def meets_targets(fields):
    """Return false for missing/nonfinite evidence, even if other metrics pass."""
    import math
    result = {}
    for name, limit in TARGETS.items():
        value = fields.get(name)
        if isinstance(value, bool):
            result[name] = False
            continue
        try:
            numeric = float(value)
            result[name] = math.isfinite(numeric) and numeric <= limit
        except (TypeError, ValueError):
            result[name] = False
    return {'targets_met': all(result.values()), 'individual': result}


def initial_specs():
    """Fixed diverse residual hypotheses, frozen before their first scored fit."""
    base = {'adapter': 'r1_residual', 'groups': [], 'peak_weight': 1.,
            'model_params': {'n_estimators': 1200, 'learning_rate': .03,
                'num_leaves': 31, 'min_child_samples': 100, 'objective': 'regression_l1',
                'colsample_bytree': .85, 'subsample': .85, 'subsample_freq': 1,
                'reg_lambda': 5., 'n_jobs': 4}, 'early_stopping_rounds': 60}
    import copy
    result = []
    expanded = ['lag_1_16', 'slot_7_28d', 'profile', 'rolling', 'trend', 'calendar', 'peak']
    recipes = [
        ('core-l1', [], 'regression_l1', 1., None, 31),
        ('expanded-l1', expanded, 'regression_l1', 1., None, 31),
        ('expanded-l2', expanded, 'regression', 1., None, 31),
        ('expanded-peak2', expanded, 'regression_l1', 2., None, 31),
        ('expanded-peak4', expanded, 'regression_l1', 4., None, 31),
        ('expanded-recent30', expanded, 'regression_l1', 1., 30., 31),
        ('expanded-peak2-recent30', expanded, 'regression_l1', 2., 30., 31),
        ('expanded-small', expanded, 'regression_l1', 2., None, 15),
        ('expanded-q55', expanded, 'quantile', 1., None, 31),
        ('expanded-q60', expanded, 'quantile', 1., None, 31),
    ]
    for name, groups, objective, weight, recency, leaves in recipes:
        spec = copy.deepcopy(base)
        spec.update(id='FG-R1-'+name, family='F2', groups=groups, peak_weight=weight,
                    recency_halflife_days=recency)
        spec['model_params'].update(objective=objective, num_leaves=leaves)
        if objective == 'quantile': spec['model_params']['alpha'] = .55 if name.endswith('q55') else .60
        result.append(spec)
    return result
