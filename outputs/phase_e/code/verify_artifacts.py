"""Independent recomputation from Phase E's saved cal/score artifacts only."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.metrics import average_precision_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from guard import AccessGuard, write_json, sha256


def close(a,b):
    if not np.allclose(a,b,atol=1e-10,rtol=1e-9,equal_nan=True):
        raise AssertionError(f'Independent recomputation mismatch: {a} vs {b}')


def verify(out):
    score=pd.read_parquet(out/'predictions/phase_e_h16_oof.parquet')
    cal=pd.read_parquet(out/'predictions/phase_e_cal_inputs.parquet')
    source=pd.read_parquet(out/'predictions/phase_e_score_inputs.parquet')
    fits=json.loads((out/'logs/calibration_fit.json').read_text(encoding='utf-8'))['fits']
    conformal=json.loads((out/'logs/conformal_fit.json').read_text(encoding='utf-8'))['fits']
    cf={r['fold']:r for r in conformal}
    fit_checks=[]
    close(score.mu,source.mu)
    assert len(score)==len(source)
    assert score.model.eq('B5').all() and score.horizon.eq(16).all()
    assert score.partition.eq('score').all() and cal.partition.eq('cal').all()
    assert (score.target_time-score.origin).eq(pd.Timedelta(minutes=240)).all()
    assert score.target_time.max()<pd.Timestamp('2021-08-09 09:45')
    assert cal.target_time.max()<pd.Timestamp('2021-08-09 09:45')
    assert not score.duplicated(['fold','origin']).any()
    assert score.sigma.gt(0).all() and np.isfinite(score.sigma).all()
    assert score.is_peak.eq(score.y>score.tau).all()
    assert score.is_d2_novel_profile.equals(source.is_d2_novel_profile)
    close(score.p_raw,norm.sf((score.tau-score.mu)/score.sigma))
    close(score.q95_raw,score.mu+1.6448536269514722*score.sigma)
    for f in fits:
        c=cal[cal.fold.eq(f['fold'])]; s=score[score.fold.eq(f['fold'])]
        assert c.target_time.max()<s.origin.min()
        x=logit(np.clip(norm.sf((c.tau-c.mu)/c.sigma),1e-6,1-1e-6))
        eta=f['a']+f['b']*x
        residual=expit(eta)-c.is_peak.to_numpy(float)
        gradient=np.array([residual.sum(),np.dot(residual,x)])
        assert f['b']>=0 and f['optimizer_success']
        # Mean gradient is scale-invariant; a constrained b=0 uses its KKT sign.
        assert abs(gradient[0])/len(c)<1e-6
        assert (abs(gradient[1])/len(c)<1e-6 if f['b']>1e-8 else gradient[1]/len(c)>-1e-6)
        close(s.p_cal,expit(f['a']+f['b']*logit(np.clip(s.p_raw,1e-6,1-1e-6))))
        standardized=np.sort(((c.y-c.mu)/c.sigma).to_numpy())
        rank=min(len(c),math.ceil((len(c)+1)*.95))
        assert cf[f['fold']]['rank_1_based']==rank
        close(cf[f['fold']]['q_star'],standardized[rank-1])
        close(s.U95,s.mu+standardized[rank-1]*s.sigma)
        fit_checks.append({'fold':f['fold'],'cal_n':len(c),'mean_gradient':(gradient/len(c)).tolist(),'rank':rank})
    assert score.uncertainty_flag.eq(score.U95>score.tau).all()
    risk=pd.concat([pd.read_csv(out/'tables/risk_metrics.csv'),pd.read_csv(out/'tables/risk_metrics_by_fold.csv')])
    uncertainty=pd.read_csv(out/'tables/uncertainty_metrics.csv')
    decisions=pd.read_csv(out/'tables/decision_value_curve.csv')
    positions=pd.read_csv(out/'tables/alert_position_metrics.csv')
    episodes=pd.read_csv(out/'tables/alert_episode_metrics.csv')
    events=pd.read_csv(out/'tables/alert_episode_events.csv')
    misses=pd.read_csv(out/'tables/miss_decomposition.csv')
    assert set(positions.policy)=={'1/1','2/2'}
    assert set(decisions.threshold)=={.01,.05,.10,.20,.30,.50}
    full=score.sort_values(['fold','origin']).copy()
    counts={'risk_rows':0,'uncertainty_rows':0,'position_rows':0,'decision_rows':0,'episode_rows':0,'reliability_bins':0}
    def subset(name,fold='pooled'):
        p=full if name=='D1' else full[full.is_d2_novel_profile]
        return p if str(fold)=='pooled' else p[p.fold.eq(int(fold))]
    for r in risk.itertuples():
        p=subset(r.subset,r.fold); z=p.is_peak.to_numpy(float)
        bs=[np.mean((p[col]-z)**2) for col in ['p_raw','p_cal','p_climatology']]
        close(bs,[r.BS_raw,r.BS_cal,r.BS_climatology]);close(r.BSS,1-bs[1]/bs[2])
        close(r.PR_AUC_cal,average_precision_score(z,p.p_cal) if len(np.unique(z))==2 else np.nan)
        assert r.n==len(p)
        counts['risk_rows']+=1
    for r in uncertainty.itertuples():
        p=subset(r.subset,r.fold)
        if r.population!='all':p=p[p.is_peak.eq(r.population=='peak')]
        bound=p.q95_raw if r.method=='raw' else p.U95
        close(r.coverage,np.mean(p.y<=bound));close(r.mean_width,np.mean(bound-p.mu));close(r.median_width,np.median(bound-p.mu))
        counts['uncertainty_rows']+=1
    for threshold in [.01,.05,.10,.20,.30,.50]:
        watch=full.p_cal.ge(threshold)
        previous=watch.groupby(full.fold).shift(1,fill_value=False)
        contiguous=full.groupby('fold').origin.diff().eq(pd.Timedelta(minutes=15))
        all_flags={'1/1':watch,'2/2':watch&previous&contiguous}
        for r in positions[positions.threshold.eq(threshold)].itertuples():
            p=subset(r.subset,r.fold); a=all_flags[r.policy].loc[p.index]; z=p.is_peak
            tp=int((a&z).sum());fp=int((a&~z).sum());fn=int((~a&z).sum());tn=int((~a&~z).sum())
            assert [r.TP,r.FP,r.FN,r.TN]==[tp,fp,fn,tn]
            close(r.F1,2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan)
            counts['position_rows']+=1
            if r.policy=='1/1':
                row=decisions[(decisions.subset==r.subset)&(decisions.fold.astype(str)==str(r.fold))&(decisions.threshold==threshold)].iloc[0]
                em=threshold*(tp+fp)+fn; ec=min(tp+fn,threshold*len(p));ep=threshold*(tp+fn)
                close(row.E_model,em); close(row.relative_value,(ec-em)/(ec-ep) if ec>ep else np.nan)
                counts['decision_rows']+=1
    for r in episodes.itertuples():
        ev=events[(events.subset==r.subset)&(events.threshold==r.threshold)&(events.policy==r.policy)]
        mi=misses[(misses.subset==r.subset)&(misses.threshold==r.threshold)&(misses.policy==r.policy)&(misses.fold.astype(str)==str(r.fold))]
        if str(r.fold)!='pooled':ev=ev[ev.fold.astype(str).eq(str(r.fold))]
        tp=ev[ev.status.eq('TP')];fp=ev[ev.status.eq('FP')];fn=ev[ev.status.eq('FN')]
        assert [r.detected_peak_episodes,r.false_alert_episodes,r.missed_peak_episodes]==[len(tp),len(fp),len(fn)]
        denom=2*len(tp)+len(fp)+len(fn)
        close(r.episode_F1,2*len(tp)/denom if denom else np.nan)
        assert mi.episodes.sum()==len(tp)+len(fn)
        assert not tp.duplicated(['fold','actual_start']).any() and not tp.duplicated(['fold','alert_start']).any()
        if len(tp):
            lead=(pd.to_datetime(tp.actual_start)-pd.to_datetime(tp.first_direct_issue_time)).dt.total_seconds()/60
            close(lead,tp.direct_lead_minutes);close(np.median(lead),r.direct_lead_median_minutes)
            assert lead.max()<=240
        counts['episode_rows']+=1
    for prob,name in [('p_raw','reliability_bins_raw'),('p_cal','reliability_bins_calibrated')]:
        bins=pd.read_csv(out/'tables'/f'{name}.csv')
        for (ds,fold),g in bins.groupby(['subset','fold']):
            p=subset(ds,fold);assert set(g.bin)==set(range(10));assert g.n.sum()==len(p)
            assert g.event_n.sum()==p.is_peak.sum()
            for r in g.itertuples():
                sel=p[p[prob].ge(r.bin_lower)&(p[prob].le(r.bin_upper) if r.bin==9 else p[prob].lt(r.bin_upper))]
                assert len(sel)==r.n
            counts['reliability_bins']+=len(g)
    return {'passed':True,'fit_checks':fit_checks,'recomputed':counts,'score_rows':len(score),
            'predictions_sha256':sha256(out/'predictions/phase_e_h16_oof.parquet'),
            'independent_of_report_generator':True,'historical_final_artifact_read':False}


if __name__=='__main__':
    AccessGuard(ROOT).install()
    out=ROOT/'outputs/phase_e'
    result=verify(out)
    write_json(out/'logs/independent_metric_audit.json',result)
    print(json.dumps(result,ensure_ascii=False))
