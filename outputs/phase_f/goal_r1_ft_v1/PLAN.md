# Goal-focused Chronos LoRA experiment plan

RQ1/C01/C06: test whether adapting the foundation forecaster itself improves factory forecasts after residual-only hypotheses failed to improve R1. RQ3/C03: inspect peak and startup regime failures. This is a new development experiment informed by prior EXPLORE observations, not a retrospectively preregistered claim.

Use the existing audited foundation adapter and current Chronos2.3.2/PEFT runtime. Two frozen LoRA configurations: context512 or2048, learning rate1e-5, maximum100 steps, training prediction length16, inference prediction length96, median point forecast. Batch16 for512 and batch8 for2048 are explicit resource choices. FIT-only supervised windows and STOP-only validation every50 steps and best-checkpoint selection, all13 horizons and8 EXPLORE weeks. The patience3 early-stopping callback is configured but cannot fire with only two evaluation events. Evaluate actual5 seeds42/123/2024/3407/777 by their mean forecast, never best seed. Match B5/M1/R1 baseline bytes and retain paired block CI and seed dispersion.

Before full search, one first-week seed42 smoke establishes actual LoRA execution, future perturbation and resource feasibility. It is not a complete cohort or ranking result. Its causal checkpoint may be reused by the exact same full seed recipe, without counting it as an additional independent fit.

Protect the running original Phase F models/source/cache and all required full-search budgets. Separate goal_r1_ft_v1 output/model namespace. Coordinate the owned GPU baseline and resume it after any deliberate allocation; never stop unrelated jobs. Freeze execution sources, installed model/runtime and snapshot bytes before actual fits. Different settings/source require new IDs/namespace.

Accepted absolute targets4.5/9, h4<=3.5, h16<=5.5, normalized MAE<=5% remain. Independent WF CONFIRM and pc3 via one locked final transaction remain pending. No holdout, deployment, commercial field proof or automatic push.
