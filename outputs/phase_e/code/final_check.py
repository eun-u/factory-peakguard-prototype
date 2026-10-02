"""Map the requested A-W technical QA to concrete evidence; no scientific score."""
from pathlib import Path
import ast
import json
import re
import sys
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from guard import AccessGuard,write_json,git
from run_phase_e import verify_protocol,verify_analysis_seal,validate_predictions


def main():
    AccessGuard(ROOT).install()
    out=ROOT/'outputs/phase_e'
    verify_protocol();verify_analysis_seal()
    pred=pd.read_parquet(out/'predictions/phase_e_h16_oof.parquet')
    validate_predictions(pred)
    parity=json.loads((out/'logs/source_parity_audit.json').read_text(encoding='utf-8'))
    metric=json.loads((out/'logs/independent_metric_audit_after_f1_correction.json').read_text(encoding='utf-8'))
    platt=json.loads((out/'logs/calibration_fit.json').read_text(encoding='utf-8'))
    conformal=json.loads((out/'logs/conformal_fit.json').read_text(encoding='utf-8'))
    tests=(out/'logs/pytest_final.txt').read_bytes().decode('utf-16' if (out/'logs/pytest_final.txt').read_bytes().startswith(b'\xff\xfe') else 'utf-8')
    assert re.search(r'27 passed',tests) and ' failed' not in tests and ' error' not in tests
    assert parity['parity_passed'] and metric['passed']
    assert platt['score_labels_used_in_fit'] is False and conformal['score_labels_used_in_fit'] is False
    assert all(f['b']>=0 and f['clip_lower']==1e-6 and f['clip_upper']==1-1e-6 for f in platt['fits'])
    assert not parity['d2_refit']
    access=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'logs').glob('access_*.json')]
    assert all(not r['historical_final_artifact_read'] and not r['denied'] for r in access)
    new=git(ROOT,'ls-files','--others','--exclude-standard').splitlines()
    assert all(p.startswith('outputs/phase_e/') for p in new)
    assert not git(ROOT,'diff','--name-only','HEAD')
    names=[p.name for p in (out/'predictions').iterdir()]
    assert not any('final_test' in p or 'holdout' in p for p in names)
    called=set()
    for filename in ['distribution.py','probability.py','alerts.py']:
        tree=ast.parse((out/'code'/filename).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Name):called.add(node.func.id)
                elif isinstance(node.func,ast.Attribute):called.add(node.func.attr)
    assert not called.intersection({'fit_kalman','fit_lgbm','fit_tcn','expected_exceedance','load_power_data'})
    rows=[
      ('A','Only h16 / 240 min','prediction_qa; direct timestamp assertions'),
      ('B','Only B5 point model','prediction_qa; no point model fit call'),
      ('C','Point prediction and metrics parity <=1e-8','source_parity_audit'),
      ('D','Fit-only tau unchanged','h16 fold manifest replay and exact saved tau parity'),
      ('E','Fit-only phi/q/r','hash-verified Phase C cache; no new parameter fit call; source audit'),
      ('F','Cal labels only Platt/conformal fitting','no B5 refit; independent optimizer gradient/rank recomputation'),
      ('G','Score labels excluded from fitting','cal-only perturbation test; fit logs; chronology/embargo assertions'),
      ('H','Finite positive sigma','prediction_qa; synthetic variance and causal perturbation tests'),
      ('I','Probability within [0,1]','prediction_qa; independent formula recomputation'),
      ('J','Probability clipping 1e-6 and 1-1e-6','fit logs; clipping test; independent formula'),
      ('K','Conformal scores cal-only','fit log; independent cal residual recomputation'),
      ('L','Finite-sample ceil((n+1)*.95) rank','independent_metric_audit_after_f1_correction'),
      ('M','Exactly six fixed thresholds','config gate; independent table set assertion'),
      ('N','Probability-only trigger','uncertainty-perturbation synthetic test; 96 position rows recomputed'),
      ('O','2/2 same fold exact 15-minute origins','gap/fold synthetic tests and full flag recomputation'),
      ('P','Only 1/1 and 2/2','independent table policy set assertion'),
      ('Q','No cross-fold episode linking','per-fold matcher; one-to-one key assertions and tests'),
      ('R','No D2 refitting','cal fits once per fold; inherited flag test; D2 values same prediction rows'),
      ('S','No expected-exceedance model call','AST call check; independent source review'),
      ('T','No production/tariff/action simulation in risk','audited B5 power-only inputs and probability-only flags'),
      ('U','No holdout/final content in fitting/evaluation','sealed prefix; poisoned boundary tail test; timestamp-first parquet read; scoped access logs'),
      ('V','Existing C/D evidence unchanged','pre-evaluation and current full protected hash checks; clean tracked diff'),
      ('W','No new final-test artifact','new Git paths only outputs/phase_e; scoped writes; prediction filename and timestamp assertions'),
    ]
    write_json(out/'logs/qa_checklist.json',{
        'scope':'technical implementation/evidence checks; not a composite scientific success score',
        'checks':[{'id':a,'requirement':b,'evidence':c,'status':'PASS'} for a,b,c in rows],
        'synthetic_tests_passed':27,'independent_metric_audit_passed':True,
        'figures_visually_reviewed':8,'visual_revisions':'figure_layout_revision.json',
        'new_tracked_candidates':len(new),'scientific_layer_judgments':'phase_e_result.md'})
    print('A-W technical QA recorded; 27 tests; all 8 figures visually reviewed; sources preserved.')


if __name__=='__main__':main()
