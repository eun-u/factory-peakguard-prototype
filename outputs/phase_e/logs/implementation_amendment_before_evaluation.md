# Implementation corrections before probability/alert evaluation

2026-10-02, after successful B5 input parity and before any Phase E Platt fitting, conformal fitting, Brier/coverage/decision/alert evaluation.

The original protocol/config files and original parity audit/code hashes are retained unchanged. Independent review found that the runner recorded parity-time code hashes but did not enforce them before analysis, and wrote the final artifact inventory too early. The runner now requires a separate immutable reviewed analysis seal, a clean tracked worktree, the original HEAD and unchanged C/D hashes **before importing/executing evaluation**; final stage logs and runtime are written before the artifact inventory.

The probability worker completed an additional supplied-p_raw-versus-Gaussian-formula input validation and its test near the parent parity run; probability.py and test_probability.py consequently differ from the original parity-time hash. No probability calibration or Phase E score evaluation had occurred. The reviewed final implementation keeps the exact preregistered Gaussian, monotone two-parameter logistic, conformal rank, threshold grid and policy definitions. This is explicitly recorded rather than silently replacing the original seal. The original B5 point parity remains applicable because its distribution code did not change. `analysis_code_seal.json` pins the final reviewed implementation and source dependencies before probability execution.

The user-required `false_alert_episodes_per_operating_day` schema name is retained as a compatibility alias. An explicit `false_alert_episodes_per_observed_evaluation_day` alias and matching CI fields are added, and all presentation uses observed evaluation days. The denominator and calculations are unchanged.

Synthetic testing initially hit Windows permission errors in pytest's default temporary directory (21 passed, 2 fixture setup errors). The path-only access guard tests now use a synthetic path without creating files. The full rerun passed 23 tests. A worker's inaccessible, empty `.pytest_tmp` scratch directory is excluded from the artifact inventory; it contains no experiment evidence.

No model, horizon, probability method, scientific metric, threshold, policy, or selection criterion was changed in response to Phase E outcomes. No final/holdout artifact was read, and no C/D file was modified.
