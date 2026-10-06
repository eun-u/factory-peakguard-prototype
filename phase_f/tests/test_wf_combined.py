import pandas as pd
import pytest
from phase_f.wf_combined import merge_scores


def frame(arm,fold):
    return pd.DataFrame([{'model':model,'fold':fold,'origin':'2021-04-19 00:00',
        'target_time':'2021-04-19 04:00','y':3.,'tau':2.,'d2':False,'arm':arm} for model in ('X','B5')])


def test_f11_combines_only_matching_frozen_arms_and_no_repeat_rows():
    a=frame('EXPLORE',1);b=frame('CONFIRM',0)
    assert len(merge_scores(a,b,'X'))==4
    wrong=b.copy();wrong.loc[wrong.model.eq('B5'),'y']=7
    with pytest.raises(ValueError,match='cohort'):merge_scores(a,wrong,'X')
    wrong=b.copy();wrong['fold']=1
    with pytest.raises(ValueError,match='repeats'):merge_scores(a,wrong,'X')
    with pytest.raises(ValueError,match='arms'):merge_scores(a,a,'X')
