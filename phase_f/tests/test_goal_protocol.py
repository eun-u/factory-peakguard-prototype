from phase_f.goal_protocol import TARGETS, initial_specs, meets_targets


def test_target_requires_every_metric_and_finite_evidence():
    assert meets_targets(TARGETS)['targets_met']
    assert not meets_targets({**TARGETS, 'AUC_PeakMAE': 9.01})['targets_met']
    assert not meets_targets({**TARGETS, 'AUC_MAE': float('nan')})['targets_met']
    assert not meets_targets({**TARGETS, 'AUC_MAE': False})['targets_met']
    assert not meets_targets({k:v for k,v in TARGETS.items() if k!='h16_MAE'})['targets_met']


def test_recipes_are_distinct_and_independent():
    specs = initial_specs()
    assert len({s['id'] for s in specs}) == len(specs) == 10
    specs[0]['model_params']['num_leaves'] = 999
    assert specs[1]['model_params']['num_leaves'] == 31
    assert all('production' not in s['groups'] for s in specs)
