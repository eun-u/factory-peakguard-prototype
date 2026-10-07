from pathlib import Path
import json
import os
import numpy as np
from phase_f import goal_recent_analog as runner, goal_r1
from phase_f.registry import now, sha256, write_json
root=Path.cwd()
original,prepared=goal_r1.prepare(root,runner.NAMESPACE)
plan=runner._read(prepared.out/'logs/execution_plan.json')
prelaunch=runner._read(root/'outputs/phase_f/logs/revision_20261006/recent_analog_v1_root_prelaunch_verification.json')
assert prelaunch['source_sha256']==runner._sources(root)==plan['sources']
paths,recomputed=runner.preflight(original,prepared)
assert recomputed==plan
runner._verify_frozen(prepared,plan,paths,deep_model_snapshot=True)
runner._verify_smoke(prepared,paths,plan)
cells=[]
for spec in runner.SPECS:
 frame,meta=runner._verify_prediction(prepared,spec,paths,plan,'smoke',smoke_first_fold=True)
 assert len(frame)==13442 and meta['audit']['n_fold_models']==1
 cell=meta['audit']['cells'][0]
 checkpoint=prepared.out/'models/recent_day_analog'/spec['id']/f"f{cell['fold']}.joblib"
 physical=runner._read(checkpoint.with_suffix('.json'))
 assert sha256(checkpoint)==cell['model_sha256']==physical['model_sha256']
 cells.append({'id':spec['id'],'rows':len(frame),'n_fold_models':meta['audit']['n_fold_models'],'horizons':sorted(map(int,frame.horizon.unique())), 'prediction_sha256':sha256(prepared.out/'smoke'/spec['id']/'first_fold.parquet'),'metadata_sha256':sha256(prepared.out/'smoke'/spec['id']/'first_fold.json'),'checkpoint_sha256':sha256(checkpoint),'checkpoint_metadata_sha256':sha256(checkpoint.with_suffix('.json')),'checkpoint_audit':cell})
primary_out=prepared.out
primary_source=prepared.source
replay=runner._replay_view(prepared)
assert prepared.out==primary_out and prepared.source is primary_source and prepared.source.out==primary_out
assert replay.source is not primary_source and set(replay.contexts)==set(prepared.contexts)
assert replay.history is prepared.history and replay.split_lock is prepared.split_lock and replay.seal is prepared.seal
assert not os.path.samefile(prepared.out/'models',replay.out/'models')
for key in prepared.contexts:
 assert replay.contexts[key] is prepared.contexts[key]
 for role in ('fit','stop','cal','score'):
  assert replay.origins(*key,role).equals(prepared.origins(*key,role))
runner._verify_frozen(prepared,plan,paths,deep_model_snapshot=True)
proof={'at':now(),'plan_sha256':plan['plan_sha256'],'source_runtime_raw_parent_verified':True,'root_tests_source_binding_verified':True,'physical_checkpoints_verified':True,'required_R1_path_keys':len(paths),'candidate_count':len(cells),'cells':cells,'actual_replay_wrapper_verified':True,'actual_primary_source_unchanged':True,'actual_replay_context_and_origins_equal':True,'physical_model_dirs_distinct':True,'primary_model_dir':str((prepared.out/'models').resolve()),'replay_model_dir':str((replay.out/'models').resolve()),'full_explore_score_computed':False,'goal_achieved':False,'holdout_read':False,'historical_final_artifact_read':False,'no_further_wave_after_current_R5':True}
write_json(root/'outputs/phase_f/logs/revision_20261006/recent_analog_v1_smoke_verification.json',proof,exclusive=True)
print(json.dumps({'at':proof['at'],'plan_sha256':proof['plan_sha256'],'four_smokes_verified':True,'actual_replay_wrapper_verified':True,'physical_model_dirs_distinct':True}),flush=True)
