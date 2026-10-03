"""Synthetic negative control replacing the forbidden future-truth oracle."""
import numpy as np
import pandas as pd
from phase_f.registry import Registry,write_json


def run_oracle_negative_control(root):
    index=pd.date_range('2021-01-01',periods=400,freq='15min')
    values=pd.Series(100+np.sin(np.arange(400)/10),index=index)
    origin=index[250]
    altered=values.copy();altered.loc[altered.index>origin]+=1000
    causal=float(values.loc[:origin].iloc[-4:].mean())
    causal_changed=float(altered.loc[:origin].iloc[-4:].mean())
    illegal=float(values.loc[origin+pd.Timedelta(hours=1)])
    illegal_changed=float(altered.loc[origin+pd.Timedelta(hours=1)])
    assert causal==causal_changed and illegal!=illegal_changed
    registry=Registry(root)
    registry.register({'id':'F0-4','family':'F0','tier':0,'kind':'synthetic_leakage_negative_control',
                       'note':'Approved replacement; no real future-truth input or performance oracle'})
    record={'synthetic_only':True,'causal_max_abs_difference':abs(causal-causal_changed),
        'intentional_leak_detected':True,'oracle_ineligible':True,
        'historical_final_artifact_read':False,'holdout_read':False}
    write_json(registry.out/'logs/F0-4_negative_control.json',record)
    registry.update('F0-4',status='diagnostic_passed',leakage_test='passed',**record)
    return record
