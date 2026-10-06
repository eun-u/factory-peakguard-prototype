from pathlib import Path
import json
import os
import pandas as pd
from phase_f import goal_day_analog as runner, goal_r1, wf_metrics
from phase_f.registry import now, sha256, write_json
root=Path.cwd()
original,prepared=goal_r1.prepare(root,runner.NAMESPACE)
plan=runner._read(prepared.out/'logs/execution_plan.json')
paths,recomputed=runner.preflight(original,prepared)
assert recomputed==plan
runner._verify_smoke(prepared,paths,plan)
replay=runner._replay_view(prepared)
assert not os.path.samefile(prepared.out/'models',replay.out/'models')
assert prepared.source is not replay.source
rows=[]
for spec in runner.SPECS:
 record=runner._read(prepared.out/'logs/experiments'/f"{spec['id']}.json")
 assert record['status']=='completed' and record['spec']==spec
 primary,primary_meta=runner._verify_prediction(prepared,spec,paths,plan,'primary')
 fresh,fresh_meta=runner._verify_prediction(replay,spec,paths,plan,'fresh_replay')
 equal=runner._compare_replay(primary,fresh)
 assert len(primary)==102427 and primary_meta['audit']['n_fold_models']==fresh_meta['audit']['n_fold_models']==8
 assert all(cell['cache_reused'] is False for cell in fresh_meta['audit']['cells'])
 fields=goal_r1._target_fields(wf_metrics.evaluate(primary))
 assert fields==record['result']['fields']
 manifest=runner._read(prepared.out/'tables'/spec['id']/'EXPLORE/manifest.json')
 assert manifest==record['result']
 assert manifest['goal_achieved'] is False and manifest['holdout_read'] is False
 rows.append({'id':spec['id'],'fields':fields,'targets_met':manifest['targets_met'],'rows':len(primary),'primary_sha256':sha256(runner.prediction_path(prepared,spec['id'])),'replay_sha256':sha256(runner.prediction_path(replay,spec['id'])),'completed_record_sha256':sha256(prepared.out/'logs/experiments'/f"{spec['id']}.json"),'score_manifest_sha256':sha256(prepared.out/'tables'/spec['id']/'EXPLORE/manifest.json'),'n_primary_models':primary_meta['audit']['n_fold_models'],'n_fresh_models':fresh_meta['audit']['n_fold_models'],'replay_exact_equality':equal,'primary_cells':primary_meta['audit']['cells'],'fresh_cells':fresh_meta['audit']['cells']})
runner._verify_frozen(prepared,plan,paths,deep_model_snapshot=True)
proof={'at':now(),'plan_sha256':plan['plan_sha256'],'source_runtime_raw_split_parent_verified':True,'actual_primary_and_fresh_checkpoints_verified':True,'completed_candidate_count':len(rows),'candidate_results':rows,'best':min(rows,key=lambda row:row['fields']['AUC_MAE']),'independent_confirmation':'pending','goal_achieved':False,'holdout_read':False}
write_json(root/'outputs/phase_f/logs/revision_20261006/day_analog_v2_full_verification.json',proof,exclusive=True)
print(json.dumps({'at':proof['at'],'completed_candidate_count':len(rows),'best':proof['best']['id'],'fields':proof['best']['fields']}),flush=True)
