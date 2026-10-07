"""Atomic, append-audited experiment records; only EXPLORE metrics here."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)


def config_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != value:
            raise RuntimeError(f'Refusing to replace locked artifact: {path.name}')
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def code_identity(root):
    root = Path(root)
    hashes = {p.relative_to(root).as_posix(): sha256(p)
              for p in sorted((root/'phase_f').rglob('*.py')) if '__pycache__' not in p.parts}
    return {'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
            'source_hashes': hashes, 'source_digest': config_hash(hashes)}


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.out = self.root/'outputs/phase_f'
        self.records = self.out/'logs/experiments'
        self.records.mkdir(parents=True, exist_ok=True)

    def read(self, exp_id):
        path = self.records/f'{exp_id}.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def register(self, spec):
        exp_id = spec['id']
        if not exp_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.+' for c in exp_id):
            raise ValueError('Unsafe experiment ID')
        digest = config_hash(spec)
        prior = self.read(exp_id)
        if prior:
            if prior['config_hash'] != digest:
                raise RuntimeError(f'Configuration changed for existing {exp_id}; use a new ID')
            return prior
        identity = code_identity(self.root)
        row = {'exp_id': exp_id, 'family': spec.get('family', exp_id.split('-')[0]),
               'tier': spec.get('tier', 0), 'parent_exp': spec.get('parent', ''),
               'config_json': canonical(spec), 'config_hash': digest,
               'code_commit': identity['commit'], 'source_digest': identity['source_digest'],
               'status':'planned', 'created_at':now(), 'explore_queries':0,
               'historical_final_artifact_read':False, 'holdout_read':False,
               'note':spec.get('note','')}
        self._save(row, 'planned')
        return row

    def update(self, exp_id, **fields):
        if any('confirm' in key.lower() for key in fields):
            raise ValueError('CONFIRM results belong to the separate one-time evaluator')
        row = self.read(exp_id)
        if row is None:
            raise KeyError(exp_id)
        row.update(fields, updated_at=now())
        self._save(row, fields.get('status','update'))
        return row

    def _save(self, row, event):
        write_json(self.records/f"{row['exp_id']}.json", row)
        with (self.out/'logs/registry_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(canonical({'at':now(),'event':event,**row})+'\n')
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.records.glob('*.json'))]
        temp = self.out/'registry.csv.tmp'
        pd.DataFrame(rows).to_csv(temp,index=False)
        os.replace(temp,self.out/'registry.csv')
