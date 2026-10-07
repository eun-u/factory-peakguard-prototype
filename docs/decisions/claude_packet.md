# DECISION
Implement an autonomous competition forecasting pipeline while preserving an October 1 test lock and historical evidence.
# GOAL
Leakage-safe peak forecasting and useful, honest analysis before the user returns for report formatting October 5.
# OPTIONS
A: blindly execute supplied design. B: correct statistical/temporal inconsistencies, preserve declared assumptions and explicitly report unavailable personal inputs.
# CONSTRAINTS
CPU only, no deep learning, one final holdout evaluation after Oct1 18:00 KST, raw data and verification immutable, no fabricated survey/account activity, unknown electricity units and contract option.
# EVIDENCE
Verified local: 6168 hourly records, 4 quarter-hour values each, 48 malformed hour rows. Existing holdout first origin Aug9 09:45 2021. Existing LightGBM peak MAE17.77 vs persistence27.73 was already seen once. Design requests 3 rolling CV folds, last validation fold calibration, same fold optional-method adoption, end timestamp features y(t), evening alarm shifts to already-past midday. Official government April16 2026 reform verifies spring/summer/autumn band changes. Exact applicable contract rates unknown.
# CURRENT GPT POSITION
Choose B. Use origin just after interval end, observations <= origin. Fit/calibrate/score disjoint with horizon embargo within development. Preserve same timestamp holdout border. Fit CBL only on observations available by origin (also horizon96). Exclude all repaired dependencies. Register adoption criteria before execution. Keep final-test calendar lock. Distinguish retrospective load-shift scenario from feasible future-only operations. Compare tariff bands and explicit ordinal sensitivities, never actual savings. Auto-source official calendar and document tariff gaps; survey is unresolved personal input.
# ALTERNATIVES
Simplify optional classifier/foundation/PDP before cutting mandatory CBL, coverage, evening FN or reproducibility. Remove expected exceedance if unsupported. Use development-only report until final lock releases.
# UNCERTAINTIES
Are calibrated quantile distribution interpolation, CI selection and episode bootstrap sufficiently defensible? Last-fold conformal versus nested calibration costs. 30-minute runtime with multiple horizons. External human-dependent work cannot be guaranteed.
# SPECIAL REVIEW FOCUS
Challenge leakage and validation design, temporal impossibility in shift simulation, statistical guarantees under time dependence, and whether automatic freeze/date semantics match user no-intervention request. One review only; no tools or project inspection.
