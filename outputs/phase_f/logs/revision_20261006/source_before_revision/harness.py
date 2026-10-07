"""Sealed Phase C contexts, immutable comparison keys and target-week locks."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from phase_c.data import load_history, build_contexts, EXPECTED_SHA256
from phase_c.evaluation import _prepare as prepare_legacy_predictions
from phase_f.registry import sha256, config_hash, write_json, now

KEY = ['horizon','fold','origin','target_time']
BASELINES = ('B1','B5','M1','M1-W','M2','R1')
BOUNDARY = pd.Timestamp('2021-08-09 09:45:00')


def output_dir(root):
    out = Path(root)/'outputs/phase_f'
    for name in ('logs','tables','predictions','models','cache','figures'):
        (out/name).mkdir(parents=True,exist_ok=True)
    return out


def seal_parent(root):
    root = Path(root)
    out = output_dir(root)
    seal = out/'logs/parent_evidence_seal.json'
    if seal.exists():
        record = json.loads(seal.read_text(encoding='utf-8'))
        for name,digest in record['files'].items():
            if not (root/name).is_file() or sha256(root/name) != digest:
                raise RuntimeError(f'Protected parent evidence changed: {name}')
        return record
    tracked = subprocess.check_output(['git','ls-files','-z','phase_c','configs/phase_c.json',
            'outputs/phase_c','outputs/phase_e','scripts/phase_b_reference'],cwd=root).decode('utf-8').split('\0')
    paths = set(name for name in tracked if name)
    # Verify and protect the exact model/prediction caches before any deserialization.
    manifest = json.loads((root/'outputs/phase_c/logs/output_cache_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['artifacts']:
        name = item['path']
        if not name.startswith('outputs/phase_c/'):
            raise ValueError('Unexpected cache manifest path')
        if sha256(root/name) != item['sha256']:
            raise RuntimeError(f'Phase C cache hash mismatch: {name}')
        paths.add(name)
    record = {'created_at':now(),'parent_commit':'e76523c84587727ec2137f8d34c75dc371b7be62',
              'files':{name:sha256(root/name) for name in sorted(paths)},
              'raw_sha256':EXPECTED_SHA256,'historical_final_artifact_read':False,'holdout_read':False}
    write_json(seal,record,exclusive=True)
    return record


def arm_for_targets(targets):
    index = pd.DatetimeIndex(targets)
    return np.where(index.isocalendar().week.to_numpy(dtype=int)%2 == 0,'EXPLORE','CONFIRM')


def fit_scales(context):
    y = context['y'].loc[context['fit']].to_numpy(float)
    weekly = context['x'].loc[context['fit'],'slot7d'].to_numpy(float)
    return float(np.mean(y)),float(np.mean(np.abs(y-weekly)))


def make_frame(context, origins, horizon, fold, prediction, model, role, **extra):
    origins = pd.DatetimeIndex(origins)
    target = pd.DatetimeIndex(context['target_time'].loc[origins])
    values = np.asarray(prediction,float)
    if len(values)!=len(origins) or not np.isfinite(values).all():
        raise ValueError('Candidate lacks finite predictions for required origins')
    if len(origins) and (target.max()>=BOUNDARY or origins.max()>=BOUNDARY):
        raise ValueError('Sealed boundary violation')
    mean,scale = fit_scales(context)
    frame = pd.DataFrame({'model':model,'horizon':horizon,'fold':fold,'origin':origins,
        'target_time':target,'y':context['y'].loc[origins].to_numpy(float),'pred':values,
        'tau':context['tau'],'d2':context['d2'].loc[origins].to_numpy(bool),
        'role':role,'arm':arm_for_targets(target) if role=='score' else 'CAL',
        'fit_mean':mean,'mase_scale':scale,'development_only':True})
    for name,value in extra.items():
        frame[name] = value
    return frame


class Prepared:
    def __init__(self,root):
        self.root = Path(root)
        self.out = output_dir(root)
        self.seal = seal_parent(root)
        self.history,audit = load_history(root)
        self.contexts = build_contexts(self.history)
        if self.history.index.max()>=BOUNDARY:
            raise AssertionError('History crossed boundary')
        raw = pd.concat([pd.read_parquet(self.root/'outputs/phase_c/predictions/main_predictions.parquet'),
                         pd.read_parquet(self.root/'outputs/phase_c/predictions/reference_predictions.parquet')],ignore_index=True)
        paired,legacy_audit = prepare_legacy_predictions(raw)
        common = paired[paired.model.eq('B5')][KEY]
        r1 = paired[paired.model.eq('R1')][KEY]
        if not pd.MultiIndex.from_frame(common).equals(pd.MultiIndex.from_frame(r1)):
            if set(pd.MultiIndex.from_frame(common)) != set(pd.MultiIndex.from_frame(r1)):
                raise RuntimeError('R1 coverage differs from fixed MAIN10 cohort')
        self.keys = common.sort_values(KEY).reset_index(drop=True)
        self._key_index = pd.MultiIndex.from_frame(self.keys)
        self.baselines = paired[paired.model.isin(BASELINES)].copy()
        self.legacy_aux = paired[paired.model.isin(('B2','B3'))].copy()
        self.baselines['role']='score'
        self.baselines['arm']=arm_for_targets(self.baselines.target_time)
        self.legacy_aux['role']='score'
        self.legacy_aux['arm']=arm_for_targets(self.legacy_aux.target_time)
        for (h,f),c in self.contexts.items():
            mask=self.baselines.horizon.eq(h)&self.baselines.fold.eq(f)
            mean,scale=fit_scales(c)
            self.baselines.loc[mask,'fit_mean']=mean
            self.baselines.loc[mask,'mase_scale']=scale
            auxmask=self.legacy_aux.horizon.eq(h)&self.legacy_aux.fold.eq(f)
            self.legacy_aux.loc[auxmask,'fit_mean']=mean
            self.legacy_aux.loc[auxmask,'mase_scale']=scale
        self._lock_split()
        write_json(self.out/'logs/data_audit.json', {**audit,'parent_cohort':legacy_audit,
            'common_score_keys':len(self.keys),'historical_final_artifact_read':False,'holdout_read':False})

    def _lock_split(self):
        keys=self.keys.copy()
        keys['arm']=arm_for_targets(keys.target_time)
        keys['target_date']=keys.target_time.dt.strftime('%Y-%m-%d')
        if keys.groupby('target_time').arm.nunique().max()!=1:
            raise AssertionError('Target appears in both arms')
        dates=keys[['target_date','arm']].drop_duplicates().sort_values('target_date')
        counts=keys.groupby(['horizon','fold','arm']).size().rename('n').reset_index()
        record={'version':1,'boundary':str(BOUNDARY),'split_by':'target_calendar_ISO_week',
                'even':'EXPLORE','odd':'CONFIRM','independent_unseen_validation':False,
                'prior_score_use':'Phase C and Phase E development evaluation',
                'dates':dates.to_dict('records'),'counts':counts.to_dict('records'),
                'key_sha256':config_hash(keys.astype(str).to_dict('records')),
                'cohort':'Phase C MAIN10 finite common keys, with full R1 coverage required',
                'long_context_policy':'No score row loss. Causal masks/imputation fixed per candidate.',
                'baseline_models':list(BASELINES),'historical_final_artifact_read':False,'holdout_read':False}
        record['lock_sha256']=config_hash(record)
        write_json(self.out/'logs/split_lock.json',record,exclusive=True)
        self.split_lock=record
        keys.to_parquet(self.out/'cache/score_keys.parquet',index=False)

    def origins(self,horizon,fold,role):
        c=self.contexts[(horizon,fold)]
        if role=='cal':
            return c['cal']
        return pd.DatetimeIndex(self.keys.loc[self.keys.horizon.eq(horizon)&self.keys.fold.eq(fold),'origin'])

    def validate_predictions(self,frame):
        if frame.duplicated(['model','role',*KEY]).any():
            raise ValueError('Duplicate candidate keys')
        if frame.model.nunique()!=1:
            raise ValueError('One experiment must produce one model')
        score=frame.loc[frame.role.eq('score')].sort_values(KEY)
        if not pd.MultiIndex.from_frame(score[KEY]).equals(self._key_index):
            raise ValueError('Score cohort changed')
        truth=self.baselines[self.baselines.model.eq('B5')].sort_values(KEY)
        for name in ['y','tau','d2','arm']:
            if not np.array_equal(score[name].to_numpy(),truth[name].to_numpy()):
                raise ValueError(f'Candidate has different {name}')
        if not np.isfinite(frame.pred).all():
            raise ValueError('Nonfinite prediction')
        if (frame.target_time>=BOUNDARY).any():
            raise ValueError('Future target crossed boundary')
        return frame
