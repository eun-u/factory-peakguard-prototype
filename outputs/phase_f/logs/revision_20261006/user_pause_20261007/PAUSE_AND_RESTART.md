# User pause checkpoint — 2026-10-07 KST

The direct instruction is: 검증 후 점수가 안나와도 멈춰.
Finish only the already frozen FG-R5 four-recipe validation/evaluation. Stop regardless of score or validation failure. No next variant, new sweep, download, fit, automatic retry or automatic resume is authorized after this checkpoint. Resume requires a new direct user instruction.

Repository: current phase_c_workspace; branch Phase4_performance. Do not reset the checkout, replace frozen sources, or overwrite prior logs, models, baselines, registry changes or failed evidence. No GitHub push was performed for this pause.

## Original Phase F

Verified owned worker 27432 was stopped. Its launcher 27880 and wrapper 19436 exited. Original/transition worker absence was verified via elevated CIM, not sandboxed Get-Process. The wrapper terminal exit0 does not establish completed PhaseF.

M2 checkpoint: 406/520 physical model+metadata pairs verified. seed42/123/2024 have104 each and complete prediction pairs. seed3407 has94/104; seed777 has0/104. No M2 mean. No temporary/orphan files. Stage1 replay resolved2/125; all later mandatory search budgets/final integration/confirmation remain incomplete. Original status/registry files still contain the last historical running entry; the independent pause record is authoritative for process state.

Evidence: original_checkpoint_after_stop.json and original_root_checkpoint_acceptance.json. Preserve these and the original caches. Current model junction: outputs/phase_f/walkforward_v2/models -> D:\PeakGuard_PhaseF_20261003\models\walkforward_v2.

## Restart only after explicit instruction

Read PROGRESS.md / DECISIONS.md and this folder first. Recheck branch, dirty checkout, exact source/runtime/raw/split/parent hashes, physical checkpoint pairs and absence of owned jobs. Read logs/goal_confirmation_policy.json without activating final confirmation. Never open numeric candidate CONFIRM/holdout or historical final metrics for selection.

Original approved resume path is the existing outputs/phase_f/env/Scripts/python.exe -u -m phase_f.run --stage all --retry, with UTF8, CPU4 and original approved CUDA setup. Do not reuse PIDs. It reuses completed model and seed caches; M2 and Stage1 remain incomplete. Do not execute it merely because this restart text exists.

FG-R5 sources and scientific plan are frozen under goal_r1_recent_analog_v1. Retain current byte-level source files: Git CRLF conversion can invalidate frozen source hashes. Use the new final result checkpoint written beside this file to identify actual completed/failure state. Never start another R5 wave automatically.

Accepted targets remain equal-horizon MAE<=4.5 / PeakMAE<=9 / h4MAE<=3.5 / h16MAE<=5.5 / FIT-normalized nMAE<=5%. Physical units UNKNOWN. No external commercial proof, cross-site or field validation. All current validation is complete and experiment processes are absent. Work is paused; the goal status transition is requested separately and the objective is not achieved. No final confirmation transaction has been consumed.

Final files: RESULTS.md, FINAL_RESULT_CHECKPOINT.json, FINAL_PROCESS_CHECK.json. FG-R5 logs/stop_requested.json prevents another wave. Preserve/archive that marker and remove it only after an explicit user resume request. No next trial was started.
