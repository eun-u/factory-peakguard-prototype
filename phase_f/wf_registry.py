"""Weekly trial records and a combined export with immutable pc3 evidence."""
import json
import os
from pathlib import Path
import pandas as pd
from phase_f.registry import Registry


class WFRegistry(Registry):
    def __init__(self, root):
        self.root=Path(root)
        self.out=self.root/'outputs/phase_f/walkforward_v2'
        self.records=self.out/'logs/experiments'
        self.records.mkdir(parents=True,exist_ok=True)

    def _save(self,row,event):
        super()._save(row,event)
        old_dir=self.root/'outputs/phase_f/logs/experiments'
        legacy={}
        for path in sorted(old_dir.glob('*.json')):
            value=json.loads(path.read_text(encoding='utf-8'))
            value['pc3_status']=value['status']
            value['pc3_config_json']=value['config_json']
            for key in tuple(value):
                if key.startswith(('explore_','D2_explore_','fold0_','fold1_','fold2_')):
                    value['pc3_'+key]=value[key]
            value['wf_status']='pending'
            legacy[value['exp_id']]=value
        for path in sorted(self.records.glob('*.json')):
            weekly=json.loads(path.read_text(encoding='utf-8'))
            old=legacy.get(weekly['exp_id'],{})
            legacy[weekly['exp_id']]={**old,**weekly,'wf_status':weekly['status'],
                                     'evaluation_protocol':'walkforward_v2'}
        target=self.root/'outputs/phase_f/registry.csv'
        temp=target.with_suffix('.csv.tmp')
        pd.DataFrame(list(legacy.values())).to_csv(temp,index=False)
        os.replace(temp,target)
