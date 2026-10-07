# Phase E preregistered development-only protocol

Locked before Phase E real-data prediction, calibration, or scoring on 2026-10-02 (Asia/Seoul). The attached user specification is authoritative. Source branch: `Phase3_experiments`, commit `5d15ff0b44c41b7b4978cd318dfc7e14bfe6ce19`.

## Fixed research questions and scope

E1 tests B5 native Gaussian exceedance probability and chronological Platt recalibration; E2 tests empirical one-sided upper coverage and sharpness; E3 tests normalized cost-loss decision value; E4 tests episode detection, false alert burden and conservative direct lead. These support the project's uncertainty, alert-validity and failure-analysis research questions, independently of point-forecast validity. The report is development evidence for the methods/results/limitations sections, not field or final-test evidence.

The user fixes M*=B5 Weekly-deviation Kalman and h*=16 quarters (240 minutes). No point-model/horizon selection, extra inputs, classifiers or expected-exceedance regression. This checkout contains Phase C evidence; no separate Phase D directory was present at preflight. h16 authority is the user's explicit locked instruction, not a newly inferred Phase D result.

All new code, tests, locks, logs, predictions, figures and reports live under `outputs/phase_e/`. Existing files, Phase C/D evidence and Git history remain unchanged. No commit, push, merge, final evaluation, approval generation or scheduling.

## Information boundary and provenance

Use the unchanged Phase C sealed loader, which stops numerical parsing before `2021-08-09 09:45:00`, including inside the boundary hour. Raw whole-file SHA256 is identity verification only. No historical final/test artifact is opened. Do not search historical output contents. The runtime file-access guard rejects historical output/verification access and any repository write outside Phase E. Source and Phase C development artifact hashes plus Git state verify preservation.

Reuse the original h16 eligibility, splits, fit/stop/cal/score embargo and novel-profile D2 logic. Fit-label Q0.95 defines tau; fit-label prevalence defines climatology. Reuse hash-verified existing B5 fit parameters, whose original fit code uses only fit deviations. Filtering can assimilate observations through each origin, including earlier observed calibration/score values; this is causal state updating, not parameter refitting. No weather/production/headcount is a risk input; inherited quality/eligibility handling remains unchanged.

## Parity gate and evaluation cohort

First compare every regenerated B5 h16 score mean with its original Phase C prediction, maximum absolute difference <=1e-8. Compare count, MAE, peak MAE and RMSE both on all B5 rows and on the Phase C MAIN10 finite common cohort. Phase C's published pooled metric uses a common cohort that excludes structural missing B2 rows. E D1 instead uses **all valid B5 h16 score rows**, as requested; the two cohort counts and metrics must be explicitly distinguished. Do not discard otherwise valid B5 rows merely to match a rounded published metric. Stop before calibration/evaluation if any parity check fails.

## Probability and uncertainty

Preserve Kalman filtered mean arithmetic and add posterior variance P. Predictive variance is phi^(2h)*P + q*(1-phi^(2h))/(1-phi^2) + r. Every sigma must be finite and positive. p_raw=Normal.sf((tau-mu)/sigma); q95_raw=mu+1.6448536269514722*sigma. Statistical high-load means y>tau, not a contractual or safety limit.

Fit only two Platt parameters on each fold's cal labels: sigmoid(a+b*logit(clip(p_raw,1e-6,1-1e-6))), b>=0, unpenalized binomial likelihood. Numerical failure or absent calibration class is a run failure, not permission to choose another method. Score diagnostic intercept/slope is evaluation-only and unconstrained; nonidentifiable diagnostics are marked unavailable, never used to modify p_cal.

Conformal cal scores are (y-mu)/sigma, all finite. k=min(n,ceil((n+1)*0.95)); q_star=sorted_scores[k-1]. U95=mu+q_star*sigma. Empirical chronological coverage only; no iid distribution-free coverage claim. uncertainty_flag=(U95>tau) is context only and never triggers an alert.

Probability metrics: Brier raw/cal/climatology, calibrated BSS, PR-AUC defined as **average precision** (stepwise precision-recall integral), event prevalence and evaluation-only calibration intercept/slope. Reliability uses exactly ten [0,.1),...,[.9,1] bins, retaining empty bins. Uncertainty metrics report overall/peak/nonpeak coverage and mean/median/quantile widths by fold, pooled, D1 and D2.

## Decision and alerts

Fixed C/L grid: .01,.05,.10,.20,.30,.50. 1/1 action=(p_cal>=c), with L=1,C=c and exactly the user-specified E_model/E_no/E_all/E_clim/E_perf/value formulas. E_clim is the ex-post best constant reference on the evaluation set, not a fitted policy. No threshold winner or monetary saving claim.

Only 1/1 Watch and 2/2 Confirmed Alert. Confirmation requires the current and immediately previous origin in the same fold, exactly 15 minutes apart, both positive. D2 inherits frozen D1 probabilities and policy flags; filtering down to D2 is evaluation-only and does not reset the original decision history. D2 episode construction still breaks at removed timestamps.

Actual and alert episodes use contiguous 15-minute target timestamps, separately within each fold. Match one-to-one, maximizing detected actual episodes first and total overlap second. Direct lead for each matched actual is its onset minus the earliest positive forecast origin in the intersection with its uniquely assigned alert. This prevents another alert assignment from inflating lead. No assumed preparation-time cutoff. Report matched-episode lead mean/median/p10/p25/p75/p90/min/max, including nonpositive leads.

Miss decomposition is locked as a transparent diagnostic: matched with direct lead>0 is `hit`; matched with lead<=0 is `late_hit`; unmatched with maximum p_cal over that actual episode below .01 (the minimum fixed grid value) is `forecast_miss`; every other unmatched actual is `decision_miss`. One-to-one assignment loss is separately flagged. This operational categorization does not establish a causal failure mechanism or model invalidity.

Burden denominator is the number of distinct represented target-calendar dates, called **observed evaluation days**, because actual factory operating days are unknown. Episode duration includes its positive 15-minute slots. Position metrics are secondary.

## Statistical uncertainty and D2

Use 1000 bootstrap draws, seed 42, target-calendar-day blocks stratified by fold. Resample each fold's represented dates with replacement, preserving all rows and shared weights for paired differences. No row bootstrap and no cross-fold concatenation. PR-AUC uses the same day weights. Report undefined/sparse draws rather than fabricate certainty.

For episode intervals, match complete original episodes once. Attribute actual TP/FN and lead to actual-onset day and unmatched FP to alert-onset day, then resample these day clusters within fold. This preserves event identity and avoids spurious events at concatenated-day seams, but conditions on the observed matching and is not a rematched streaming-path confidence interval. State this limitation. Width/coverage and reliability intervals use position day blocks. D2 uses identical fit/cal parameters and policy decisions, with no refitting.

## Reporting and QA

Generate every required user table/log/figure plus the complete score parquet. Case figures use the first chronological TP/FN/FP at c=.10, policy 2/2, purely as fixed illustrative cases; absent categories are disclosed. No example-driven model/policy selection. Test h16/B5, partitions/fit-only quantities, causal state variance, clipping/rank, fixed grid/policies, gaps/folds, one-to-one matching, D2 freezing, probability-only triggers, and forbidden historical access. Independent review precedes completion. Report all failures and limitations; do not reopen C/D or claim one composite Phase E PASS score. Final evaluation needs a separately authorized, fully frozen pipeline.
