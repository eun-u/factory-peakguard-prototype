# FG-R5: recent observed-history analog feature

Status: post-observation design fixed before any FG-R5 fit or candidate SCORE calculation. Four FG-R4 fixed-FIT recipes completed without meeting targets; their full root verification is being finalized. This is a separate hypothesis, not a change to FG-R4 or a preregistration before those observations.

## Scientific question and approved boundary

RQ1/C01/C06 and RQ3/C03: can recently observed operating-shape similarity reduce absolute point/peak errors, especially D2 and later weeks? FIT-only FG-R4 references are 23.7 to 48.2 days older than each SCORE week start because the weekly pre-score history is divided 80% FIT / 20% STOP. Recency differs substantially from current observed lags. Benefit is unproven.

The revised user prompt section 3 permits only observations through origin and section 4(c) permits additional past-power context/profile information. Section 5.2 prohibits training on SCORE dates; 5.2-B defines each week's training from preweek history. Here inference is a fixed stateless function of observed raw power, like a nonlinear seasonal-lag feature. No model weights or memory table are updated from evaluation frames. FIT defines tau/scales; STOP alone chooses alpha. Earlier elapsed SCORE/CONFIRM-period raw power may enter a later query as then-known observed input and can equal prior evaluation labels. We do not claim no evaluation-period raw values are ever read. No current/future target, candidate CONFIRM prediction/metric, holdout or historical final artifact is opened for selection.

## Four fixed configurations and inference

FG-R5-recent-w16-k3, w16-k5, w96-k3, w96-k5. Prefix16/96 exact quarter-hour observations ends at query or reference origin; k3/5; recent calendar days56. No additional window, temperature, normalization, or mixing grid is selected from SCORE.

For query t take reference origins r=t-d calendar days, integer d=1..56, same interval-end quarter slot. Source/query prefixes must be finite under both quality_bad/time_repaired masks. Each selected reference must have all thirteen observed suffix powers at r+h*15min for h4..16 finite and clean, with every latest source timestamp <=t. Do not use context['y'] or future production/weather in inference. Source power comes from the sealed raw history loader. Handle missing timestamps and midnight directly by timestamps, never positional shifting.

Center each prefix by its own mean, rank by mean absolute prefix difference, break ties by most recent reference. Use exactly k references, inverse weights1/max(distance,1e-6), smallest ordered weighted median at cumulative half, and per-reference level shift y_raw(r+h)+power(t)-power(r); clip the analog at zero. This keeps the FG-R4 point recipe and changes only prediction-time reference recency. If query or reference support is insufficient, preserve locked causal R1 and all keys. Never update fitted state while traversing CAL/SCORE.

Choose one pooled weekly alpha in [0,.25,.5,.75,1] by common purged STOP objective MAE+.25*PeakMAE with FIT-only horizon taus, ties lower alpha. Current STOP targets may select this frozen coefficient but each STOP analog feature sees only observations through its own origin. Zero STOP peak support fails instead of becoming zero error. Official equal-horizon aggregation is separate from pooled STOP selection.

## Identity, provenance and validation

New files only: phase_f/models/recent_day_analog.py, phase_f/goal_recent_analog.py and their two tests. Existing live original/R3/R4/core modules remain unchanged. New namespace goal_r1_recent_analog_v1 and approved D model junction. Bind the immutable scientific/input contract and plan, source/runtime/raw/protected seal/weekly locks, complete R1/guard evidence and both producer/normalized path digests before fit. Checkpoint identity binds FIT/STOP roles, labels, tau, all STOP query and actually used historical feature values, fixed spec, causal R1 path/producer proof and source/runtime. Serialize a frozen recipe/STOP coefficient and audits, not an online table learned from SCORE. Private locally generated joblib is trusted; verify bytes before loading and actual fitted recipe/STOP selection after loading. Preserve corrupt or partial artifacts and fail.

Prediction-time input audit records selected reference/source timestamps and support by role/fold/horizon/D2 and verifies latest_observed<=origin. Count earlier elapsed evaluation-period raw observations honestly without inspecting current/future evaluation labels. Avoid a dense all-rows x56x96 allocation; bounded chunks or per-origin arrays, CPU2.

Required tests: synthetic past perturbation changes the analog feature, future power and quality perturbations do not; all13 unblended analog values plus ranks/references and blended predictions invariant under actual fitted future probes (alpha0 must not hide leakage); exact quarter timestamps/missing/quality/fallback/negative clipping/midnight/ties; no context y passed to inference; source/model/spec/physical cache mutation fails with preserved bytes; actual _ArmView fresh replay interface and no primary mutation. One deterministic primary run plus a fresh physical full replay with exact keys/point/aux/context equality; no fake5 seeds and stochastic SD not applicable. Four actual first-fold smokes precede full8-week search.

## Decision and remaining work

Targets unchanged: equal-horizon MAE<=4.5, PeakMAE<=9, h4(60min)<=3.5, h16(240min)<=5.5, FIT-normalized nMAE<=5%. Physical units UNKNOWN. The competitive3.5/7 suggestion is not a contract amendment. Report D1/D2, weeks/horizons/coverage, actual timing and paired day-block1000-draw intervals against B5/R1/best-overall guard. Preserve adverse outcomes; past EXPLORE designs/selection and augmentation invalidate any external/commercial proof claim.

The original128 replay, each GBDT family500 completed trials, top20+3baseline PhaseE, stochastic finalist10 seeds and one<=7 WF/pc3 CONFIRM union remain required. Stage4 integration_pending barrier remains. No deployment/control or automatic GitHub push.
