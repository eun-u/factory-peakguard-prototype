"""Outcome-independent evaluation and revised stopping rules."""
import math
from phase_f.registry import config_hash

CONTRACT={
    'version':2,'primary':'pooled walk-forward EXPLORE metric of seed mean forecasts',
    'even_iso_week':'EXPLORE','odd_iso_week':'CONFIRM',
    'stochastic_seeds':5,'finalist_stochastic_seeds':10,
    'bootstrap_n':1000,'bootstrap_seed':42,'bootstrap_block':'target_date',
    'minimum_gbdt_trials_each':500,'maximum_wallclock_budget':None,
    'relative_improvement_minimum':.02,'nonimproving_rounds':2,
    'continuation_metrics':['AUC_MAE','AUC_PeakMAE','h16_PeakMAE','E_c10_22_recall'],
    'improvement_requires':['relative_change_at_least_2pct','paired_CI_excludes_zero','larger_than_seed_sd'],
    'eligibility':['leakage_pass','B5_MAE_improvement_CI_low_gt_0','B5_Peak_degradation_CI_high_le_0',
        'D2_MAE_not_reversed','two_thirds_explore_weeks_MAE_improve','E_c10_22_recall_ge_same_cohort_B5'],
    'finalists':'five MAE eligible with family cap two, plus peak and episode-F1 specialists up to seven',
    'confirm':'one reserved evaluation of locked finalists and B5/M1/R1 on WF and pc3; no tuning afterward',
    'representative':'lowest WF CONFIRM MAE among finalists maintaining required gates; alternatives proposed to human',
    'pc3_role':'auxiliary reused-development comparison',
    'real_oracle':'synthetic negative leakage control only',
    'holdout_authorized':False,'historical_final_artifact_read':False,
    'commercial_readiness':'requires new nonaugmented field observations and operating tolerances',
}
CONTRACT['contract_sha256']=config_hash(CONTRACT)


def significant_gain(before,after,ci_low,ci_high,seed_sd=0.,*,higher_is_better=False):
    values=(before,after,ci_low,ci_high,seed_sd)
    if any(v is None or not math.isfinite(float(v)) for v in values):return False
    gain=after-before if higher_is_better else before-after
    relative=gain/max(abs(before),1e-12)
    # CI represents the candidate's improvement, for either metric direction.
    return bool(gain>0 and relative>=.02 and ci_low>0 and gain>seed_sd)


def two_rounds_converged(rounds):
    return len(rounds)>=2 and all(r.get('significant_improvement') is False for r in rounds[-2:])
