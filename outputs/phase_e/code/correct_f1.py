"""Explicit metric-only erratum; preserve original CSV bytes and all other cells.

Original harmonic-P/R code marked F1 unavailable whenever precision or recall
was unavailable. Count-based F1 is zero if TP=0 and FP+FN>0. Only 0/0 is NaN.
No model, calibration, predictions, thresholds, flags, matching or CI is rerun.
"""
from __future__ import annotations
import csv
import io
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from guard import AccessGuard,sha256,write_json


def f1_counts(tp,fp,fn):
    denominator=2*tp+fp+fn
    return 2*tp/denominator if denominator>0 else float('nan')


def correct(out):
    audit_path=out/'logs/f1_correction_audit.json'
    assert not audit_path.exists(), 'F1 correction already completed'
    original=out/'logs/metric_correction_v1'
    original.mkdir(exist_ok=False)
    pred_sha=sha256(out/'predictions/phase_e_h16_oof.parquet')
    corrections=[]
    for name in ['decision_value_curve','alert_position_metrics','alert_episode_metrics','d2_phase_e_metrics']:
        path=out/'tables'/f'{name}.csv'
        raw=path.read_bytes()
        backup=original/f'{name}.csv'
        backup.write_bytes(raw)
        text=raw.decode('utf-8')
        reader=csv.DictReader(io.StringIO(text,newline=''))
        fieldnames=reader.fieldnames
        rows=list(reader)
        original_rows=[dict(row) for row in rows]
        changed=[]
        episode=name in ['alert_episode_metrics','d2_phase_e_metrics']
        metric='episode_F1' if episode else 'F1'
        columns=['detected_peak_episodes','false_alert_episodes','missed_peak_episodes'] if episode else ['TP','FP','FN']
        for i,row in enumerate(rows):
            if name=='d2_phase_e_metrics' and row['table']!='alert_episode_metrics': continue
            tp,fp,fn=[int(float(row[c])) for c in columns]
            expected=f1_counts(tp,fp,fn)
            value=float(row[metric]) if row[metric] else float('nan')
            if math.isfinite(value):
                assert math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-12)
            elif math.isfinite(expected):
                assert expected==0
                row[metric]='0.0'
                changed.append(i)
        for i,(before,after) in enumerate(zip(original_rows,rows)):
            assert all(before[c]==after[c] for c in fieldnames if c!=metric)
            assert before[metric]==after[metric] or i in changed
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=fieldnames,lineterminator='\r\n' if '\r\n' in text else '\n')
        writer.writeheader();writer.writerows(rows)
        path.write_bytes(stream.getvalue().encode('utf-8'))
        reread=list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8'),newline='')))
        assert reread==rows
        corrections.append({'table':name,'metric':metric,'rows':len(rows),'changed_row_indices':changed,
                            'changed_cells':len(changed),'original_sha256':sha256(backup),'corrected_sha256':sha256(path),
                            'all_other_cell_strings_unchanged':True,'schema_and_row_order_unchanged':True})
    assert pred_sha==sha256(out/'predictions/phase_e_h16_oof.parquet')
    result={'reason':'count-defined F1=0 when TP=0 and FP+FN>0; original NaN overstated undefinedness',
            'formula':'2*TP/(2*TP+FP+FN); NaN only for zero denominator',
            'independent_review_confirmed':True,'model_or_policy_change':False,'refit_or_rerun':False,
            'predictions_sha256_unchanged':pred_sha,'corrections':corrections}
    write_json(audit_path,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    AccessGuard(ROOT).install()
    correct(ROOT/'outputs/phase_e')
