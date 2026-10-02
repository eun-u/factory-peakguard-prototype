# Phase C independent review

Development-only. Review findings and validation additions do not change the sealed model candidates, feature formulas, fitting protocol or selection criteria.

- Fresh M1 / M1-W / C1 training routes use the same locked LightGBM candidate. M1-W supplies fit-label peak weight 2; C1 trains y minus weekly anchor and adds that anchor back. Four anchors by three folds select a single configuration using stop only.
- B3 and B5 parameter estimation is limited to fit timestamps. Their origin predictions use fixed fitted parameters and observations available through that origin. Already observed cal-period power can enter a later lag/state, but cal labels are not tuning or scoring cases.
- Evaluation implements the preregistered MAIN10 paired cohort, equal-horizon AUC, joint target-calendar-day bootstrap, peak safeguard, residual gate, complexity tie and R1 exclusion.
- Resume cache identity checks inside the sealed training module are incomplete. This execution started in a newly created Phase C namespace, without prior real fitted caches. The standalone scripts/verify_phase_c_artifacts.py adds strict final origin/truth/configuration/availability checks and a SHA256 manifest for the resulting artifacts. A completed manifest must be checked before future cache reuse; the training CLI alone is not a complete tamper-proof resume interface.
- The evaluator omits empty D2 groups and does not emit a separate reference coverage table. The standalone validation records explicit D1/D2 expected/actual counts, including zero, and separate R1 coverage. Its actual-count checks determine whether this limitation affects this run.
- The stability column d2_rank_reversal means a sign change relative to the strongest baseline. The d1_rank and d2_rank columns contain actual rankings; a rank change and a comparator sign change are distinct.

## Additional legacy regression check

The additional synthetic legacy-boundary suite completed with 15 passing tests and one dependency failure: test_training_guard.py::test_final_holdout_cannot_run_before_freeze_even_with_data imports a legacy seasonal module that requires numba, which is absent from the isolated Phase C environment. The fixture is empty and no holdout was accessed. This is not a real model-run failure, nor is that one test claimed passed. The dedicated Phase C pre-execution suite passed 20 tests. The actual trace is retained in legacy_boundary_pytest.txt. Package changes during the real run are avoided to preserve its environment.

## Scope of progress records

The user's isolated Phase C output requirement and sealed baseline hashes are preserved. Phase C progress, decisions and results are stored in this new namespace; existing project progress/decision documents and historical results remain unchanged.

## Structural CBL missingness and independent verifier correction

The first standalone verification rejected 1,862 B2 NaNs, with every other model finite. The already sealed preregistration explicitly defines the global MAIN10 finite intersection and says to report exclusions rather than fill predictions. Thus absolute finiteness of every raw B2 score row was an overly strict additional verifier assumption, not the registered evaluation policy. Its failed audit and traceback are preserved as artifact_integrity_attempt1_failed.json and artifact_integrity_attempt1.stderr.log.

The selected original adjusted CBL can return NaN when an origin-aligned three-hour window on a selected reference day includes repaired/missing power, even if that reference day's target-slot value is valid. The corrected standalone check must reproduce the locked CBL directly from the sealed development history and original CBL implementation, confirm every saved finite value and NaN location, and permit only those independently proven structural invalid keys. Other nonfinite predictions, extra NaNs, infinities and mismatched CBL values still fail. Every MAIN10 model then uses the identical registered intersection. The training source, model/configuration choices, prediction artifacts, evaluation source, metrics and selection result remain unchanged. This is an audit implementation correction; no search or selection rule is changed after results.

A second verifier attempt reproduced the CBL values and mask, then exposed an audit schema mismatch: the independent removal summary had three fields while the existing evaluator records seven. The verifier was corrected to derive every original field, including raw-missing, nonfinite and additional pairing removals. A synthetic test now compares the independent summary directly to the sealed evaluator's actual `_prepare` output. The second failed audit/trace are preserved separately. This did not change the paired cohort or any numerical experiment result.
