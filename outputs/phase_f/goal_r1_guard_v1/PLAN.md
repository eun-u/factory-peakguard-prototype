# Tail guard experiment plan

RQ1/C01/C06 forecast accuracy and RQ3/C03 error regimes; paper method, experiment design, results and threats. Existing first-wave EXPLORE results inform this design: in fold15 the fitted correction lowered many future peak forecasts even though R1 q90 indicated peak potential. The diagnostic q90 guard value 7.2478 MAE /13.8919 peak MAE is post-observation exploration and is not independent validation.

Freeze all 30 combinations (all 10 completed parent configurations × q90 suppress negative correction, q95 suppress negative correction, q90 halve negative correction) before any guard scoring. Parent initial10 must first fully complete. Each rule uses only frozen forecast quantiles, FIT-only tau and parent prediction; target truth does not trigger it. Transform each of five actual parent seed forecasts individually, then evaluate their mean. There are no new model fits; the five parent fits and their dispersion/provenance remain explicit. Strict shared cohort, original R1 anchor and source/prediction/chunk/metadata hashes must pass.

Preserve the parent wave and original full Phase F search. Absolute targets remain MAE4.5, peak MAE9, h4 MAE3.5, h16 MAE5.5, mean FIT-normalized nMAE5%. No claim of goal achievement without the one locked independent WF CONFIRM/pc3 transaction. No holdout, commercial field proof or automatic push.

Command after parent completion: python -m phase_f.goal_guard. Driver status and per-candidate tables are authoritative; a plan is not an executed result.
