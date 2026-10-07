"""The shared-horizon fitter must receive training origins, never score origins."""
from types import SimpleNamespace
import pandas as pd
import pytest
from phase_f.harness import Prepared


def test_prepared_origins_preserves_every_named_role():
    idx=pd.date_range('2021-01-01',periods=8,freq='15min')
    context={role:idx[k:k+1] for k,role in enumerate(('fit','stop','cal','score'))}
    prepared=SimpleNamespace(contexts={(4,0):context},
        keys=pd.DataFrame({'horizon':[4],'fold':[0],'origin':context['score']}))
    for role in context:
        assert Prepared.origins(prepared,4,0,role).equals(context[role])
    with pytest.raises(ValueError,match='Unknown'):
        Prepared.origins(prepared,4,0,'unknown')
