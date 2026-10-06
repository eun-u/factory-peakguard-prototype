# Independent confirmation integration (pending)

This is an implementation plan, not a candidate freeze, reservation, or confirmation result. The focused point forecast goal and original operational Phase F eligibility remain separate.

1. Complete the frozen 10-configuration, 5-seed EXPLORE wave. Preserve every failure. Further waves must have independent IDs and disclose observation-informed design. Choose a single representative with a declared EXPLORE rule before any candidate CONFIRM read.
2. Preserve original Stage 0–3 and their completed tuning budgets. Before its Stage 4 reservation, add the focused representative to the same final lock within the existing total cap of seven, tagged `point_target_diagnostic`. If no slot remains, fail before reservation and specify a priority rule; never silently displace an already frozen finalist.
3. Seal source, contract, settings, R1 rolling paths, baseline/seed/mean prediction hashes and metric artifact hashes. Expand actual finalist seeds to ten in EXPLORE. Frozen selection must precede numeric CONFIRM evaluation.
4. Add a dedicated final adapter through `wf_final._fit_frozen`; avoid changes to `wf_models.py` and `wf_run.py`, whose source hashes protect existing full-search caches. The current EXPLORE-only producer/residual cannot be used for CONFIRM without a separately tested final implementation. Preserve these original fitted source versions.
5. Within one durable reservation, produce baseline R1 CONFIRM CAL/SCORE forecasts first, then generate only causal FIT/STOP paths for the selected WF/pc3 cohorts and replay the frozen residual with ten actual seeds. Store standard prediction files and immutable causal path audits. Resume only the reserved IDs/settings.
6. Require all absolute targets on independent WF CONFIRM before a point-goal achievement claim; disclose pc3 auxiliary outcomes. Retain original Phase E gate without marking its unavailable CAL-class result as eligible. Point accuracy does not establish alert quality, commercial field performance, deployment, or savings.

No candidate CONFIRM or holdout was evaluated to produce this plan. The original current CAL limitation is that each EXPLORE week has no peak-positive rows in the later Platt CAL half; this remains a separate unresolved operational issue.
