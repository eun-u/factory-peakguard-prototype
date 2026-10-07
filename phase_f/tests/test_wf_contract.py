from phase_f.wf_contract import significant_gain,two_rounds_converged


def test_revised_stop_uses_relative_ci_and_seed_uncertainty():
    assert significant_gain(10,9.7,.1,.5,.01)
    assert not significant_gain(10,9.9,.05,.15,.01)
    assert not significant_gain(10,9.7,-.1,.5,.01)
    assert not significant_gain(10,9.7,.1,.5,.4)
    assert significant_gain(.4,.5,.01,.2,.01,higher_is_better=True)
    assert not significant_gain(10,9.7,None,None,0.)
    assert not two_rounds_converged([{'significant_improvement':False}])
    assert two_rounds_converged([{'significant_improvement':False}]*2)
    assert not two_rounds_converged([{'significant_improvement':False},{'significant_improvement':True}])
