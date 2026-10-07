# Phase C preparation audit

Initial preparation snapshot: this audit was written before the supplied Phase B file arrived. Its pending items below describe that earlier state. The file has since been preserved unchanged under scripts/phase_b_reference/, the exact 15-feature contract resolved, and preregistration_phase_c.md plus preregistration_lock.json sealed before real-data fitting. Current execution state is in runtime.json. The original audit details are retained below for provenance.

## Scope and provenance

- New isolated clone of remote `main` at `34c1e7b42f635053eea6358e569074997011dfa0`, branch `Phase3_experiments`.
- Existing `05_experiments` checkout has a modified `outputs/tables/T6-1_run_steps.csv`; it was left untouched.
- Raw SHA-256 matches `data/README.md`: `8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830`.
- Opaque whole-file hashing and an identical copy were used for integrity/provisioning. Raw numerical parsing used `src.session_data.load_development_history` only, ending at 2021-08-09 09:30.
- PPT is a reference for the locked Phase A/B design. The current pasted request takes precedence, including the revised D2 definition, model selection order and TCN authorization.
- The user will provide the exact Phase B implementation. Do not infer an exact feature manifest from the high-level PPT or freeze one before inspecting those artifacts.

## Reuse and required wrappers

| Existing code | Decision |
|---|---|
| `src/session_data.py` | Reuse field-by-field sealed prefix reader. Never call the full-data reader for Phase C. |
| `src/data.py` | Read code for data semantics only; its full-CSV reader is unsuitable. |
| `src/features.py` | Reuse permitted feature implementations only after exact Phase B manifest arrives. Recompute the validity mask on selected features, because the old all-column mask also depends on excluded information. |
| `src/holidays.py` | Reuse audited target calendar assumptions; retain the documented official-review limitation. |
| `src/split.py` | Reuse chronological/purged fold construction on the complete development origin grid. |
| `src/training.py::_partition` | Reuse fit/stop/cal/score boundary logic; no cal model selection or alert calibration. |
| `src/research_cv.py::build_contexts` | Cannot directly reuse: accepts only historical h=1,4,16,96 and requires historical fold metadata. Add Phase C wrapper for h=4..16. |
| `src/models/cbl.py` | Reuse formulas and audited holiday hybrid; always supply actual origins. Select representative on stop only. |
| `src/models/lgbm_point.py` | Reuse model semantics and fixed base/grid configurations; globally aggregate the four anchor horizons across three folds. |
| `research_p4/stat_models.py` | Preserve MSTL periods [96,672], 28-day context and AutoETS ZZN. Existing bidirectional interpolation and daily refits require a Phase C adapter for causal missing handling and fit-only parameter estimation. |
| `research_p4/chronos_ref.py` | Reuse Chronos-2 reference semantics only after cache origin/horizon/context/provenance equivalence check; no historical metric reuse. |
| Existing bootstrap/evaluation | Historical peak/alert adoption rules differ. A separate mean-horizon MAE and paired calendar-day bootstrap implementation is necessary. |

## Pending scientific specification

1. Exact Phase B G0/G1/G2 feature names, encodings and quality masks from the incoming artifacts.
2. Confirm fit-sample target definition for tau, as the new request specifies Q95(y_fit) while historical helper uses a historical observed prefix.
3. Record all model grids, stopping, complexity order, residual stability criteria, paired sample intersection and day-profile hashing before any model fits.
4. Freeze completed preregistration and candidate configuration hashes before execution. Do not retrofit plans after observing new performance.

## Boundary incident

See `boundary_incident.json`. A subagent opened historical `verification/results/05_summary.json`, unexpectedly displaying old test metrics. No numerical test values were transmitted to the main owner or used in its decisions. The incident was disclosed to the user before any real-data model training. The final leakage audit must distinguish this historical artifact read from sealed raw-data access and must not claim complete historical-artifact nonaccess.

## Resume

Inspect incoming Phase B artifacts as reference data, extract the actual feature and preprocessing contract, then finish preregistration and the Phase C orchestration. Model/evaluation adapters are being prepared in `phase_c/` with synthetic tests only. No commit, push, merge, final-test evaluation, freeze approval, or scheduled task activation is authorized by this run.
