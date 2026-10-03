import numpy as np
import pandas as pd
from phase_f.models.foundation import causal_input,point_from_quantiles,QUANTILES,training_windows


def test_causal_inputs_and_known_calendar():
    idx=pd.date_range('2021-01-01',periods=500,freq='15min')
    history=pd.DataFrame({'power':np.arange(500,dtype=float)},index=idx)
    origin=idx[250]
    altered=history.copy();altered.loc[altered.index>origin,'power']=-999
    a=causal_input(history,origin,192,16,True)
    b=causal_input(altered,origin,192,16,True)
    np.testing.assert_array_equal(a['target'],b['target'])
    for key in a['future_covariates']:
        np.testing.assert_array_equal(a['future_covariates'][key],b['future_covariates'][key])
    assert len(a['target'])==192


def test_native_mean_alias_and_integrated_mean():
    values=np.asarray(QUANTILES)**2
    assert point_from_quantiles(values,'native_mean')==point_from_quantiles(values,'median')
    assert point_from_quantiles(values,'integrated_mean')>point_from_quantiles(values,'median')


def test_fit_windows_fixed_cut_and_boundary():
    idx=pd.date_range('2021-01-01',periods=500,freq='15min')
    history=pd.DataFrame({'power':np.arange(500,dtype=float)},index=idx)
    origins=idx[120:200]
    context={'fit':origins,'target_time':pd.Series(idx+pd.Timedelta(hours=4),index=idx)}
    windows=training_windows(history,context,'fit',96,16)
    assert windows and all(len(w)==112 for w in windows)
    assert max(w[-1] for w in windows)<=215
