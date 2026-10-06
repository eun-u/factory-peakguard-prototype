# Independent confirmation integration (pending)

This is an implementation plan, not a candidate freeze, reservation, or confirmation result. The focused point forecast goal and original operational Phase F eligibility remain separate.

1. Complete the frozen 10-configuration, 5-seed EXPLORE wave. Preserve every failure. Further waves must have independent IDs and disclose observation-informed design. Choose a single representative with a declared EXPLORE rule before any candidate CONFIRM read.
2. Preserve original Stage 0–3 and their completed tuning budgets. Before its Stage 4 reservation, add the focused representative to the same final lock within the existing total cap of seven, tagged `point_target_diagnostic`. If no slot remains, fail before reservation and specify a priority rule; never silently displace an already frozen finalist.
3. Seal source, contract, settings, R1 rolling paths, baseline/seed/mean prediction hashes and metric artifact hashes. Expand actual finalist seeds to ten in EXPLORE. Frozen selection must precede numeric CONFIRM evaluation.
4. Add a dedicated final adapter through `wf_final._fit_frozen`; avoid changes to `wf_models.py` and `wf_run.py`, whose source hashes protect existing full-search caches. The current EXPLORE-only producer/residual cannot be used for CONFIRM without a separately tested final implementation. Preserve these original fitted source versions.
5. Within one durable reservation, produce baseline R1 CONFIRM CAL/SCORE forecasts first, then generate only causal FIT/STOP paths for the selected WF/pc3 cohorts and replay the frozen residual with ten actual seeds. Store standard prediction files and immutable causal path audits. Resume only the reserved IDs/settings.
6. Require all absolute targets on independent WF CONFIRM before a point-goal achievement claim; disclose pc3 auxiliary outcomes. Retain original Phase E gate without marking its unavailable CAL-class result as eligible. Point accuracy does not establish alert quality, commercial field performance, deployment, or savings.

No candidate CONFIRM or holdout was evaluated to produce this plan. The original current CAL limitation is that each EXPLORE week has no peak-positive rows in the later Platt CAL half; this remains a separate unresolved operational issue.

## Verified integration seams and entry barrier, 2026-10-06

- `wf_run.dispatch_stage --stage all` calls Stage4 immediately after0–3; it had no supported goal-ready marker. Stage4 writes `finalist_seed_expansion_lock.json` before CONFIRM reservation. The goal representative and cap priority must be ready before entering this path, not merely before reservation.
- The first call of `wf_final.run_final` is now a fail-closed goal campaign gate. An explicitly activated signed `integration_pending` policy blocks before any existing artifact read, registry/search, provisional lock or reservation. Status `ready` alone is unsupported; full single-transaction integration still needs implementation and review. An absent policy preserves legacy behavior.
- `wf_final.py` and the new gate are not in active EXPLORE forecast/model cache source identities. The final source lock does include all PhaseF Python files, so finish integration before locking and change none afterward.
- Add the single goal representative to `chosen/catalog/specs` before provisional lock. If all7 original slots are full, require an explicit frozen priority or fail. Keep `point_target_diagnostic` separate from original PhaseE eligibility.
- A selected Chronos LoRA configuration can use existing `foundation` adapter WF/pc3 actual10-seed execution. Residual/guard winners require separately tested causal final producers/adapters; current EXPLORE-only paths cannot be reused as CONFIRM.
- Goal sibling namespaces are outside `walkforward_v2`; freeze their plan/contract/path chunks/physical seeds/mean/metrics as a separate repository-root-relative `goal_evidence` hash map and verify before provisional selection, reservation and resume.
- Existing `summarize` omits h4 MAE and normalized AUC. Compute all five absolute fields from the same reserved CONFIRM evaluation tables; retain a separate goal verdict without changing `confirm_gate`.
- Stage4 EXPLORE PhaseE expansion currently calls `evaluate_candidate` directly and assumes completed, unlike CONFIRM's unavailable handling. Goal diagnostic integration must preserve unavailable evidence on degenerate CAL and must not synthesize recall or mark operational eligibility passed.
