# Phase C preregistration

Development-only. Registered on 2026-10-01 before the first real-data fit, tuning or score. Primary question: which single forecast family represents all 13 direct horizons, 60 through 240 minutes at 15-minute intervals, under the locked Phase A/B information set? No horizon is selected in Phase C.

## Authoritative scope and provenance

The current user request governs this experiment. Base is remote main commit `34c1e7b42f635053eea6358e569074997011dfa0`, isolated branch `Phase3_experiments`. No commit, push or merge. Prior outputs and verification evidence are immutable and cannot serve as new Phase C results. Source bytes are checked against SHA256 in data/README.md, `8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830`. Whole-file hashing/copying is opaque integrity work, not numerical holdout decoding.

The user-supplied temporary Phase B script is preserved unchanged at `scripts/phase_b_reference/run_phase_b_temp_probe.py`, SHA256 `9c0541ffafdd69cc403081b1cebdbeea5eae2cb0438b3024421c8e7777f1685f`. Its definitions provide exact feature formulas and Ridge alpha. Its full-data reader, full-data descriptive analyses, exogenous row filters, and old D2 date-removal rule are NOT executed or copied into the experiment. Prior Phase B output numbers are not claimed reproduced here.

## Data contract

- Numerical source access exclusively via `src.session_data.load_development_history`. No value at or after 2021-08-09 09:45 is parsed. Observed development ends at 09:30.
- Four hourly values reconstruct 15-minute interval END timestamps HH:15/30/45/(HH+1):00. Physical units and official interval boundaries remain unconfirmed. Average column is never input or target.
- Existing row-order time repair and bad-date/ERP exclusion remain. Repaired/missing/negative power and their required feature dependencies are invalid. No deletion due solely to statistically large power; no winsorization, clipping or target smoothing.
- Probable measurement gap rule: a fully completed hour has four zero power observations and missing headcount. Flag these observations, exclude gap targets and dependencies, and never backfill from future observations. The sealed development prefix has zero zero-power readings, so this rule does not alter its cohort. No holdout values are checked to confirm this rule.
- Pre-execution review resolved a quality-availability edge case: a completed-hour gap label cannot be applied retrospectively to an earlier forecast origin. The implementation therefore fails closed if any such gap is present, pending an origin-specific quality adapter, rather than claim causal support it does not have. This fixed-data development run has none; no hyperparameter or sample choice depends on this safeguard.
- Scaling/imputation/normalization are estimated only on fit. Core requires finite observed dependencies and does not impute them. MSTL's fit-only history allows forward fill within fit and never backward interpolation.

## Exact Phase B core, unchanged mathematical definitions

G0 = hour_sin/hour_cos of fractional TARGET hour; dow_sin/dow_cos of Sunday-based target weekday; weekend; holiday using the audited 2021 holiday list. G1 = current power, lag4 (one hour), target-slot previous day and previous week. G2 = r4_mean, r4_max, r4_std (sample std, ddof=1), r16_max, r96_max. Total 15 features. No slope, peak-history, production, workforce, measured weather, tariff, extra lag/window, or future observed inputs. Measured feature provenance must have latest observation <= origin; calendar features are preknown.

TCN observes the same 96 consecutive quarter-hour power readings ending at origin, six G0 values and two seasonal anchors. Sequence order is oldest to newest. No additional exogenous information. Every tabular and sequence sample requires the same finite 24h power history, including across excluded dates; dates are not compressed out of the time grid.

## Splits and peak

Reuse `make_splits` with all sealed development grid origins, dev_frac=1 and 3 expanding folds for each h. Apply locked core/target validity, then reuse `_partition`: fit=first 80% of train with horizon purge, stop=train tail; cal=first 35% of validation with embargo, score=remaining validation with embargo. Core feature differences mean historical sample counts can differ; never force compatibility through excluded-feature filters. Assert every fit/stop/cal/score transition has prior target strictly before next origin and every origin/target is below sealed boundary. No fit+stop refit. No cal labels are used for tuning, calibration, stopping, scoring or model selection. Past observations from cal may appear as genuinely observed lags/states at later origins; this is distinct from using cal labels as model-selection cases.

tau(h,fold) = linear-interpolated empirical Q95 of the actual y_fit labels. Peak means y>tau, a statistical high-load event, not contract-demand exceedance. Report all thresholds and sample counts.

## Candidates and fixed training

Exact machine-readable parameters and candidate simplicity order are in `configs/phase_c.json`; its hash is sealed before fitting.

- B0 current-value persistence; B1 observed target-slot seven days earlier. No fitted parameters.
- B2 existing five `cbl_all_predictions` columns only. Select one representative per horizon by mean of the three stop MAEs on the shared valid CBL cohort; exact numerical ties use listed order. No score-dependent formula change.
- B3 MSTL periods [96,672], last 28 fit days, AutoETS ZZN plus existing phi^h origin-level correction. Historical daily refits would estimate parameters on cal/score, and historical bidirectional interpolation conflicts with Phase C. Therefore fit decomposition and residual phi once on fit per h/fold, then project a fixed base path and apply observed-origin residual correction. This is an explicit implementation adaptation, not numerical parity with old MSTL outcomes. No period/specification search.
- B4 fixed Ridge alpha100; fit-only column means and sample std(ddof1), centered y, closed-form solve, no tuning.
- B5 one AR(1) Gaussian weekly-deviation state. phi,q,r fit only on fit with bounded L-BFGS-B innovations likelihood; positive q/r and phi[-.99,.99]. Missing gaps advance using exact AR step powers. Causal filtering may ingest observations through the current origin but never refits parameters. Forecast is known weekly target anchor + phi^h filtered deviation; no predicted observation is fed back.
- M1 audited LightGBM L2 regression objective, MAE early stopping. Three configs only: base31/40, grid15/40, grid31/80; lr.05,max600,patience60,seed42,threads4. No Optuna.
- M1-W exactly selected M1 config/training protocol and actual-y fit weight2 above tau; no weight tuning.
- C1 exactly selected M1 config/training protocol, target y-weekly anchor, prediction anchor+residual. No ensemble or other residual formula. Actual target remains the peak definition.
- M2 fixed causal scalar-output TCN per horizon/fold, two causal kernel3 convolutions per residual dilation block [1,2,4,8,16,32], uniform channels32/64, dropout.1/.2, final-time embedding concatenated with calendar/seasonal context followed scalar linear head. Four architectures only. L1 loss, AdamW lr.001,weight_decay.0001,batch256,max100epochs,patience10,seed42. Fit-only normalization. CUDA if available; deterministic operations. Epoch selection uses stop only. No mixed precision or automatic training-budget reduction on real data.
- R1 Chronos-2 is reference only, context2048, forecast96 steps with median read at h-1, batch32, original P4 univariate protocol. Existing artifacts are checked for all required origins/horizons and compatible input quality before reuse; absent equivalence requires fresh inference. No fine-tuning. Downloaded model revision is recorded. R1 never shrinks the MAIN10 comparison cohort or enters M* eligibility.
- Chronos past-context NaNs (repaired intervals) use the pretrained model's native observed-value mask; no interpolation or backward fill. Verify this on synthetic input and reject nonfinite predictions. Historical cache inspection is schema/provenance-only; no historical numerical prediction columns are opened.

## Tuning and score lock

M1 and M2 candidates are compared on anchors h=4,8,12,16 across all three stop folds. Criterion is the equal mean of the 12 cell MAEs. A joint calendar-target-day paired bootstrap1000/95% compares candidate-minus-lowest-mean errors, preserving equal cell means; if CI includes zero prefer the preregistered simpler config. For TCN same-channel dropout.2 precedes.1 (stronger fixed regularization); for LightGBM fewer leaves precede more, then larger minimum child size. No per-horizon configuration tuning; per-model stop early stopping is allowed by the fixed protocol. Anchor fitted models are reusable after configuration lock. CBL choices and all family configurations are written to `model_config_lock.json` before score prediction generation/evaluation. Failed fits raise and are logged, never replaced with undisclosed predictions. Implementation repairs may be made and separately logged without changing preregistered formulas/search spaces.

## D1, D2 and paired metrics

D1 is all valid development score cases on the global MAIN10 timestamp intersection per h/fold. Report exclusions rather than filling predictions. D2 only subsets score, never deletes dates/history or changes training. A raw interval day spans 00:15 through next 00:00. A complete clean 96-slot profile is serialized as little-endian float64 and hashed SHA256; incomplete/invalid profiles are unknown and excluded from D2, never assumed novel. Compare score-day hash against complete daily profiles ending no later than that fold's last fit-target timestamp. Stop/cal profiles are not training references. Save hash definition and profile status metadata.

Here training profile period explicitly means the observed fit-history prefix, including lag warmup days before the first complete feature row. This conservative definition treats profiles available in fit inputs as seen; it does not use stop/cal outcomes.

For every model,h,fold,D1/D2 record n,peak_n,MAE,Peak-MAE,RMSE,train/inference time,configuration,score bounds,tau. AUC-MAE/AUC-PeakMAE mean the equally spaced horizon-normalized mean error: the equally weighted mean of 13 pooled per-horizon errors, not an unnormalized trapezoidal integral. Pair differences are candidate error minus comparator error (negative favors candidate). Joint calendar TARGET-date block bootstrap1000 with seed42 gives95% percentile CI, retaining cross-horizon dependence. Undefined peak/D2 statistics remain undefined. Reference coverage is separately labeled. CIs are development-selection evidence and not independent confirmatory tests; no multiplicity guarantee is claimed.

## M* selection, no composite score

Strongest industrial/statistical baseline is lowest D1 AUC-MAE among B1/B2/B3. Exclude a candidate if paired AUC-PeakMAE minus this baseline has CI lower>0. Among remaining MAIN10, primary criterion is lowest AUC-MAE. Record paired strongest-baseline comparison, all horizon weekly skills and minimum, per-fold direction and D2 ranking. C1 additionally requires a clear MAE gain over M1 (CI upper<0), improvement in at least two of three fold AUC-MAEs, and available D2 with no C1/M1 rank reversal; otherwise reject C1 as unstable/unproven. This fixes the user's qualitative residual gate before results.

Models whose paired AUC-MAE CI against the lowest-error eligible candidate includes zero are statistically indistinguishable for this tie-break. Choose the simplest in order B0,B1,B2,B4,B5,B3,M1,M1-W,C1,M2. Fold/D2 reversals for other models are reported, not post-hoc converted into new numeric exclusion thresholds. An unavailable primary peak-safeguard CI prevents declaring an M*; preserve the insufficient-evidence status. D2 absence is separately flagged and disqualifies C1. R1 ineligible. Save all13 horizon predictions for Phase D; h* remains unselected.

## QA and preservation

Automatic A-K checks from the user request are mandatory, plus exact feature parity and source/config/preregistration hashes. Hash every tracked historical file before execution and compare afterward without interpreting contents; no final_test artifact creation. Protected raw source remains unchanged. Tests use synthetic perturbations for future invariance and boundary rejects. Independent code review precedes final score interpretation. Report runtime and failures honestly.

Boundary disclosure: a subagent mistakenly read historical `verification/results/05_summary.json`, which contained historical test metrics. No metric values were sent to the main owner or used in this registered design. See `boundary_incident.json`. Another source search matched verification Python code without displaying numerical historical results. This task cannot claim that all historical final artifacts were unread. Raw holdout observations remain sealed; distinguish these facts in final reporting.
