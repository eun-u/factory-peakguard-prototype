# FG-R4: fixed FIT-bank analog forecasting

Status: v2 interface repair, fixed before any v2 fitting or numerical SCORE evaluation. The v1 four first-fold technical smokes passed, but full search failed before full candidate fitting or scoring because the actual _ArmView output property has no setter. All v1 evidence and old sources are preserved under logs/revision_20261006/day_analog_v1_failed_replay_view. This v2 changes only the runner interface and output namespace to goal_r1_day_analog_v2; the four scientific recipes, targets, model implementation, and FIT support analysis are unchanged. Actual real-wrapper regression, root verification, independent review, new v2 technical smoke, and full primary/fresh replay remain required before reporting results. This post-observation design is not a preregistration made before prior EXPLORE results.

## Research question and distinction

RQ1/C01/C06: reduce absolute overall and peak point forecast errors. RQ3/C03: test whether observed operating-shape similarity helps on D2 novel profiles and later weeks. Evidence belongs in the method, experimental design, results, discussion, and threats sections as exploratory evidence only.

The existing same-slot averages, medians, fixed prior-week correlations, and profile anchors do not rank historical days by the currently observed power prefix. FG-R3 adds transition classification rather than retrieval. This wave tests a distinct structural hypothesis: a current observed prefix may identify a relevant previously completed suffix more accurately than unconditional same-slot aggregation. The hypothesis follows observed R1 errors and existing factory-profile repetition; no FG-R4 improvement is yet established.

## Four fixed configurations

- FG-R4-analog-w16-k3: prefix16, top3.
- FG-R4-analog-w16-k5: prefix16, top5.
- FG-R4-analog-w96-k3: prefix96, top3.
- FG-R4-analog-w96-k5: prefix96, top5.

Each prefix consists of 16 or96 consecutive 15-minute interval-end observations, ending at the query origin. Do not equate the timestamp span with a full4 or24hours. Require finite quality-masked power at all exact quarter timestamps. Use the existing `time_repaired` and `quality_bad` masks; no future production or actual weather.

For each EXPLORE week, intersect all13 horizon FIT origins and retain the existing maximum-h16 FIT-to-STOP embargo. Construct a frozen library from those FIT origins r at or after FIT.max()-56calendar days. Retrieve only origins with the same15-minute interval-end slot as the query. Prefixes end at r; every suffix is the corresponding already observed FIT label y[r+h], with target_time strictly before the first STOP origin. Preserve midnight interval-end conventions. The library does not grow or change during STOP/CAL/SCORE. If a FIT query is used in an extension or test, require r<query_origin and target_time<=query_origin to exclude self/future labels.

Center each prefix by its own mean. Rank by mean absolute difference between centered prefixes, with distance ties resolved by more recent reference origin first. Select exactly k records. Do not silently deduplicate repeated profiles in this wave; report the augmentation and profile repetition limitation.

For each chosen reference, estimate y[r+h]+power(query_origin)-power(r). Use weights1/max(distance,1e-6), and the smallest sorted suffix whose cumulative weight reaches half the total as the weighted median; clip the analog point at zero. The epsilon is fixed in the unknown data units. If the query prefix or fewer than k clean same-slot bank suffixes are available, preserve the locked causal R1 point. Retain every original evaluation key and report fallback/coverage by role, fold, horizon, and D2 evaluation cohort.

Choose one pooled per-week alpha from [0,.25,.5,.75,1] using only embargoed common STOP labels: objective=pooled MAE+.25*pooled PeakMAE, with FIT-only horizon tau and ties toward smaller alpha. Both selection and inference use nonnegative clipping and the same fallback. Zero STOP peak support is unavailable/failure, not perfect performance. CAL/SCORE labels and D2 classification never choose neighbors, bank, alpha, or features. Official evaluation remains equal-horizon aggregation, distinct from the pooled STOP objective.

## Determinism and physical evidence

This is a deterministic method: one distinct forecast per configuration, n_stochastic_seeds=0. The revised Phase F five/ten-seed requirement applies to stochastic models and is not satisfied by cloning deterministic outputs. Perform a separate full fresh replay in a physically separate checkpoint directory using the identical spec and frozen library recipe. Require exact output equality. A replay is reproducibility evidence, not a second independent seed or a second statistical observation. Stochastic standard deviation is not applicable.

Reuse the completed signed R1 producer and original B5/M1/R1 baseline files. Before any fit, bind all249246 required rolling path keys, physical/content chunk SHA, producer CSV hash separately from normalized model-table hash, causal proof, protected parent seal, raw source, weekly split locks, source/runtime identity, and this design file to an immutable execution plan. The plan also binds the completed best-overall guard comparison. No GPU re-inference is needed.

Checkpoint identity binds the actual masked FIT prefixes, reference keys, all FIT suffix targets, purged FIT/STOP roles, masked STOP query prefixes and labels/tau, fixed spec, normalized R1 paths, source/runtime, raw/split/seal. The identity is independent of output directory so a genuine fresh replay can prove the same scientific recipe. Verify physical checkpoint and metadata SHA before deserializing, and verify reconstructed bank/payload and recomputed STOP selection/support after load. Joblib files here are trusted private checkpoints generated locally by this execution; external checkpoint import is outside this adapter's interface. Preserve corrupt/orphan/interrupted files with byte hashes and fail; do not silently infer missing provenance or count an unverified cache as a completed fit.

Technical smoke covers all four configurations, first EXPLORE fold, all13 horizons, unchanged CAL/SCORE keys, and fitted prediction invariance after perturbing power and its quality flags strictly after a query origin. Smoke is technical validation and produces no whole-wave candidate score. Full search requires verified smoke, all8 EXPLORE weeks, primary and fresh replay physical evidence, and source/runtime/raw/parent verification before and after publication and scoring.

## Evaluation and decision boundary

Targets remain equal-horizon MAE<=4.5, PeakMAE<=9, h4 MAE<=3.5, h16 MAE<=5.5, and FIT-normalized mean nMAE<=5%. h4=60minutes and h16=240minutes; physical power units are UNKNOWN. The stricter3.5/7 competitiveness suggestion is not an amendment of the accepted execution contract.

Report D1/D2, each horizon/week, bank support, fallback shares, STOP alpha, actual fit/replay time, primary/replay hashes and exact equality. Compute paired day-block1000-draw intervals against B5, R1, and the verified best-overall guard. Preserve adverse results. A favorable EXPLORE score is subject to selection bias and does not prove independent accuracy, operational Phase E eligibility, or commercial performance.

Frozen FIT-only retrieval can lose recency as SCORE drifts. Same-slot matching can miss shifted schedules; repeated augmented profiles can overstate utility. D2 and fold-specific results must accompany any gain. In a new regime or insufficient bank, fallback remains R1; this is a declared limitation, not new evidence of analog effectiveness.

The original128 replay, each GBDT family's500 completed trials, remaining F0-F11 scope, top20+three-baseline Phase E, stochastic finalist10seeds, and one union of at most7 WF/pc3 CONFIRM candidates remain required. Any selected deterministic finalist requires a declared deterministic final adapter/replay rule under that same frozen union. The integration-pending Stage4 barrier stays active. No numerical candidate CONFIRM, holdout, historical final artifact, automatic control, deployment, or GitHub push is part of this wave.
