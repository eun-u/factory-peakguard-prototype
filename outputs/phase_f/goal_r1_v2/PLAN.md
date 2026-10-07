# Absolute forecast performance experiment

User accepted targets on 2026-10-06: overall 13-horizon MAE <=4.5, peak MAE <=9.0, h4 MAE <=3.5, h16 MAE <=5.5, mean fit-normalized horizon MAE <=5%. These are project development targets, not a universal commercial certificate. RQ1/C01/C06 address forecast accuracy, RQ3/C03 addresses failure regimes; relevant paper sections are method, experimental design, results and threats.

Design is informed by already observed R1 EXPLORE errors. R1 has peak underprediction bias about -12.52, h16 peak bias -14.93 and morning 06–10 MAE about 11.25. We do not describe this design as preregistered before those observations.

Fixed zero-shot Chronos-2 c2048 median with original 96-step prediction recipe produces causal FIT/STOP residual training paths. Its EXPLORE CAL/SCORE forecasts are anchored to existing checksum-verified R1 predictions. A new global horizon-conditioned LightGBM per weekly fold fits residuals using FIT only. All 13 horizons share a conservative cross-horizon target embargo. STOP chooses early stopping and correction attenuation; CAL/score cannot train or choose those settings. Expanded inputs are observed power lags/profiles/rolling/trend and known calendar. Same-hour retrospective temperature/production and future observed values are excluded.

Ten fixed hypotheses compare core/expanded inputs, L1/L2/quantile losses, peak weights, recency and small-tree regularization. Each runs actual seeds 42,123,2024,3407,777. Rank and absolute targets use the mean forecast, not the mean of metrics or best seed. Retain each seed metric, dispersion, candidate/B5 and candidate/R1 1000-day-block paired CI. Preserve original Phase A–F and all failed evidence. The existing full Phase F search and its required tuning budgets remain separate and unchanged.

Execution is EXPLORE-only in this namespace. Crossing a target here creates a candidate, not verified goal achievement. Any final candidate must be locked before independent WF CONFIRM and pc3 auxiliary evaluation within the existing one-time confirmation boundary. No holdout, freeze approval, deployment or automatic GitHub push is authorized here. Phase E's current empty-positive CAL limitation is recorded independently; point-forecast gains must not be represented as established alert quality or field savings.

Command: `python -m phase_f.goal_r1 --stage all`. Check logs/driver_status.json and logs/goal_progress.json. Causal path producer, residual implementation and execution plan identities are immutable once actual work starts; changed code/settings need a new namespace or configuration ID.

Execution revision: goal_r1_v1 contains a preserved failed pre-chunk GPU attempt (Chronos list[tensor] API adaptation). goal_r1_v2 freezes the corrected source before any inferred chunk or fitted residual. All original baseline anchor bytes remain identical.
