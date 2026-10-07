"""Render the sealed Phase E development evidence without choosing a policy.

Only Phase E score predictions, tables, and run logs are consumed. The report
is descriptive: each C/L threshold and each alert policy stays visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import configure, save


INK = "#1C2A39"
BLUE = "#3E6283"
AMBER = "#B8741A"
GREEN = "#398577"
RED = "#A44C42"
THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
POLICIES = ("1/1", "2/2")
FIGURES = (
    "reliability_raw_vs_calibrated.png", "brier_by_fold.png",
    "upper_coverage_raw_vs_conformal.png", "uncertainty_width_distribution.png",
    "decision_value_curve.png", "episode_tradeoff.png",
    "lead_time_distribution.png", "risk_case_examples.png",
)
TABLES = (
    "risk_metrics", "risk_metrics_by_fold", "reliability_bins_raw",
    "reliability_bins_calibrated", "calibration_summary", "uncertainty_metrics",
    "decision_value_curve", "alert_episode_metrics", "alert_position_metrics",
    "alert_episode_events", "miss_decomposition", "d2_phase_e_metrics",
)


def _require_columns(frame: pd.DataFrame, name: str, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _row(frame: pd.DataFrame, **filters):
    selected = frame
    for key, value in filters.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def _fmt(value, digits=3):
    if value is None or pd.isna(value):
        return "판정불가"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _ci(row, stem, digits=3):
    def field(name, fallback=np.nan):
        return row.get(name, fallback) if isinstance(row, (dict, pd.Series)) else getattr(row, name, fallback)
    lo = field(f"{stem}_ci_lower", field(f"{stem}_ci_low"))
    hi = field(f"{stem}_ci_upper", field(f"{stem}_ci_high"))
    if pd.isna(lo) or pd.isna(hi):
        return "CI 판정불가"
    return f"95% CI [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _md_table(headers, rows):
    escaped = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(escaped, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(escaped, row)) + " |" for row in rows)])


def _pooled(frame: pd.DataFrame):
    return frame.loc[frame.fold.astype(str).eq("pooled")]


def _figure_reliability(out: Path, tables):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.plot([0, 1], [0, 1], color=INK, lw=1, ls="--", label="ideal")
    for table, label, color in (("reliability_bins_raw", "raw", AMBER),
                                ("reliability_bins_calibrated", "calibrated", BLUE)):
        data = tables[table]
        data = data.loc[data.subset.eq("D1") & data.fold.astype(str).eq("pooled")].sort_values("bin")
        valid = data.loc[data.n.gt(0)]
        ax.errorbar(valid.mean_predicted_probability, valid.observed_event_rate,
                    yerr=[(valid.observed_event_rate-valid.observed_rate_ci_lower).clip(lower=0).fillna(0),
                          (valid.observed_rate_ci_upper-valid.observed_event_rate).clip(lower=0).fillna(0)],
                    marker="o", markersize=5, capsize=2, color=color, label=f"{label} (n={int(valid.n.sum())})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate",
           title="D1 reliability, fixed 0.1 bins", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.grid(True)
    ax.legend(loc="upper left")
    save(fig, out/FIGURES[0])


def _figure_brier(out: Path, tables):
    data = tables["risk_metrics_by_fold"].loc[lambda f: f.subset.eq("D1")].sort_values("fold")
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.7, 5.5))
    for offset, name, label, color in ((-.27, "BS_raw", "raw", AMBER),
                                       (0, "BS_cal", "calibrated", BLUE),
                                       (.27, "BS_climatology", "fit climatology", GREEN)):
        ax.bar(x+offset, data[name], width=.25, color=color, label=label)
    ax.set(xticks=x, xticklabels=[f"fold {v}" for v in data.fold], ylabel="Brier score (lower is better)",
           title="D1 Brier score by fold")
    ax.grid(axis="y")
    ax.legend()
    save(fig, out/FIGURES[1])


def _figure_coverage(out: Path, tables):
    data = tables["uncertainty_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, constrained_layout=True)
    for ax, population, title in zip(axes, ['all','peak'], ['All score rows','Actual high-load rows']):
        labels, values, colors, errors = [], [], [], [[], []]
        for subset in ('D1','D2'):
            for method,color in [('raw',AMBER),('conformal',BLUE)]:
                r=_row(data,subset=subset,fold='pooled',population=population,method=method)
                labels.append(f'{subset}\n{method}')
                values.append(float(r.coverage)); colors.append(color)
                errors[0].append(max(0,float(r.coverage-r.coverage_ci_lower)))
                errors[1].append(max(0,float(r.coverage_ci_upper-r.coverage)))
        ax.bar(np.arange(4),values,color=colors,yerr=errors,capsize=3)
        ax.axhline(.95,color=INK,ls='--',lw=1,label='nominal 0.95')
        ax.set(xticks=np.arange(4),xticklabels=labels,ylim=(0,1.07),title=title)
        ax.legend(loc='lower left');ax.grid(axis='y')
    axes[0].set_ylabel('Empirical upper coverage (95% day-block CI)')
    save(fig, out/FIGURES[2])


def _figure_width(out: Path, score):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for values, label, color in ((score.q95_raw-score.mu, "Gaussian q95 - mean", AMBER),
                                 (score.U95-score.mu, "Conformal U95 - mean", BLUE)):
        ax.hist(values.to_numpy(float), bins=35, density=True, alpha=.5, color=color, label=label)
        ax.axvline(values.median(), color=color, ls="--", lw=1.5)
    ax.set(xlabel="Upper-bound width (power units)", ylabel="Density",
           title="D1 upper-bound width distribution")
    ax.legend()
    ax.grid(axis="y")
    save(fig, out/FIGURES[3])


def _figure_decision(out: Path, tables):
    data = _pooled(tables["decision_value_curve"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for subset, color in (("D1", BLUE), ("D2", AMBER)):
        part = data.loc[data.subset.eq(subset)].sort_values("threshold")
        ax.plot(part.threshold, part.relative_value, "o-", color=color, label=subset)
    ax.axhline(0, color=INK, ls="--", lw=1)
    ax.set(xticks=THRESHOLDS, xlabel="C / L", ylabel="Normalized relative decision value",
           title="Fixed cost/loss grid, 1/1 action candidate")
    ax.grid(True)
    ax.legend()
    save(fig, out/FIGURES[4])


def _figure_episode(out: Path, tables):
    from matplotlib.lines import Line2D
    data = _pooled(tables["alert_episode_metrics"])
    fig, axes = plt.subplots(1,2,figsize=(12,5.8),sharey=True)
    markers=['o','s','^','D','P','X']
    for ax,subset in zip(axes,['D1','D2']):
        for policy, color in (("1/1", AMBER), ("2/2", BLUE)):
            part = data.loc[data.subset.eq(subset) & data.policy.eq(policy)].sort_values("threshold")
            ax.plot(part.false_alert_episodes_per_operating_day, part.episode_recall,
                    lw=1.4,color=color,label=f'{policy} '+('Watch' if policy=='1/1' else 'Confirmed'))
            for r,marker in zip(part.itertuples(),markers):
                ax.scatter(r.false_alert_episodes_per_operating_day,r.episode_recall,marker=marker,s=62,color=color,zorder=3)
        ax.set(xlabel='False alert episodes / observed evaluation day',title=subset,xlim=(0,1.8),ylim=(0,.7))
        ax.grid(True);ax.legend(loc='upper left',fontsize=9)
    axes[0].set_ylabel('Peak episode recall')
    handles=[Line2D([0],[0],marker=m,color=INK,ls='',label=f'C/L={c:g}') for m,c in zip(markers,THRESHOLDS)]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False)
    fig.tight_layout(rect=(0,.09,1,1))
    save(fig, out/FIGURES[5])


def _figure_lead(out: Path, tables):
    events = tables["alert_episode_events"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    groups={}
    for policy in POLICIES:
        data = events.loc[events.subset.eq("D1") & events.fold.astype(str).ne("pooled") &
                          events.threshold.eq(.10) & events.policy.eq(policy) & events.status.eq("TP")]
        groups[policy]=pd.to_numeric(data.direct_lead_minutes,errors='coerce').dropna()
    bins=sorted(set().union(*(set(v) for v in groups.values())))
    x=np.arange(len(bins))
    for policy,color,offset in [('1/1',AMBER,-.18),('2/2',BLUE,.18)]:
        leads=groups[policy]
        counts=[int(leads.eq(v).sum()) for v in bins]
        ax.bar(x+offset,counts,width=.34,color=color,label=f'{policy}: n={len(leads)}, median={leads.median():.0f} min')
    ax.set_xticks(x,[f'{v:g}' for v in bins])
    ax.set(xlabel="Conservative direct lead (minutes)", ylabel="Matched episodes",
           title="D1 direct lead at fixed c=0.10")
    ax.grid(axis="y")
    ax.legend()
    save(fig, out/FIGURES[6])


def _figure_cases(out: Path, tables, score):
    events = tables["alert_episode_events"]
    sample = events.loc[events.subset.eq("D1") & events.fold.astype(str).ne("pooled") &
                        events.threshold.eq(.10) & events.policy.eq("2/2")].copy()
    sample["onset"] = pd.to_datetime(sample.actual_start.fillna(sample.alert_start), errors="coerce")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for ax, status in zip(axes, ("TP", "FN", "FP")):
        matches = sample.loc[sample.status.eq(status)].sort_values(["onset", "fold"])
        if matches.empty:
            ax.text(.5, .5, f"{status}: no case on D1 at c=0.10, 2/2",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        event = matches.iloc[0]
        onset = event.onset
        segment = score.loc[score.fold.astype(str).eq(str(event.fold)) &
                            score.target_time.between(onset-pd.Timedelta(hours=12),
                                                      onset+pd.Timedelta(hours=12))].sort_values("target_time")
        if segment.empty:
            ax.text(.5, .5, f"{status}: selected event has no plotted score rows", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.plot(segment.target_time, segment.y, color=INK, lw=1.4, label="observed y")
        ax.plot(segment.target_time, segment.mu, color=BLUE, lw=1.2, label="B5 mean")
        ax.plot(segment.target_time, segment.U95, color=GREEN, lw=1.2, label="U95")
        ax.plot(segment.target_time, segment.tau, color=AMBER, ls="--", lw=1, label="fit tau")
        ax.axvline(onset, color=RED, ls=":", lw=1)
        risk_axis = ax.twinx()
        risk_axis.plot(segment.target_time, segment.p_cal, color=RED, lw=1, alpha=.8, label="p_cal")
        risk_axis.axhline(.10, color=RED, ls="--", lw=.7, alpha=.5)
        risk_axis.set_ylim(0, 1.02)
        risk_axis.set_ylabel("p_cal")
        ax.set_title(f"{status} | fold {event.fold} | onset {onset} | first chronological case")
        ax.set_ylabel("Power")
        handles, labels = ax.get_legend_handles_labels()
        other_h, other_l = risk_axis.get_legend_handles_labels()
        ax.legend(handles+other_h, labels+other_l, loc="upper left", ncol=5, fontsize=8)
        ax.grid(True)
        ax.tick_params(axis="x", rotation=15)
    save(fig, out/FIGURES[7])


def _summary_rows(tables):
    risk = tables["risk_metrics"]
    cal = tables["calibration_summary"]
    unc = tables["uncertainty_metrics"]
    decision = tables["decision_value_curve"]
    alert = tables["alert_episode_metrics"]
    rows = []
    for subset in ("D1", "D2"):
        r = _row(risk, subset=subset, fold="pooled")
        c = _row(cal, subset=subset, fold="pooled", probability="p_cal")
        u_raw = _row(unc, subset=subset, fold="pooled", population="all", method="raw")
        u = _row(unc, subset=subset, fold="pooled", population="all", method="conformal")
        values = _pooled(decision).loc[lambda f: f.subset.eq(subset)]
        a1 = _row(alert, subset=subset, fold="pooled", threshold=.10, policy="1/1")
        a2 = _row(alert, subset=subset, fold="pooled", threshold=.10, policy="2/2")
        rows.extend([
            {"layer":"E1_probability","subset":subset,"n":r.n,"metric_1":"BS_cal","value_1":r.BS_cal,
             "metric_2":"BSS","value_2":r.BSS,"metric_3":"PR_AUC_cal","value_3":r.PR_AUC_cal,
             "evidence_note":f"BS_raw={r.BS_raw:.6g}; BS_climatology={r.BS_climatology:.6g}; score intercept={_fmt(c.intercept)} slope={_fmt(c.slope)}"},
            {"layer":"E2_uncertainty","subset":subset,"n":u.n,"metric_1":"coverage_u95","value_1":u.coverage,
             "metric_2":"mean_width_u95","value_2":u.mean_width,"metric_3":"coverage_raw","value_3":u_raw.coverage,
             "evidence_note":f"nominal=.95; raw mean width={_fmt(u_raw.mean_width)}; peak coverage separate in uncertainty_metrics"},
            {"layer":"E3_decision","subset":subset,"n":r.n,"metric_1":"positive_grid_count",
             "value_1":int(values.relative_value.gt(0).sum()),"metric_2":"grid_count","value_2":len(values),
             "metric_3":"c_0.10_relative_value","value_3":_row(values, threshold=.10).relative_value,
             "evidence_note":"All six locked C/L rows are in decision_value_curve.csv; no threshold selected"},
            {"layer":"E4_alert","subset":subset,"n":a2.actual_peak_episodes,"metric_1":"c_0.10_2of2_recall",
             "value_1":a2.episode_recall,"metric_2":"c_0.10_2of2_false_alert_episodes","value_2":a2.false_alert_episodes,
             "metric_3":"c_0.10_1of1_recall","value_3":a1.episode_recall,
             "evidence_note":"Both policies at all six thresholds remain in alert_episode_metrics.csv; no policy selected"},
        ])
    frame = pd.DataFrame(rows)
    interpretations = {
        'E1_probability': 'Brier skill and ranking assessed separately from residual calibration bias',
        'E2_uncertainty': 'Overall upper coverage and reduced width do not imply peak-conditional coverage',
        'E3_decision': 'Grid point estimates only; no monetary benefit or operating threshold selected',
        'E4_alert': 'Confirmation trades alert burden against detection; direct lead is conditional on matched events',
    }
    frame['interpretation'] = frame.layer.map(interpretations)
    return frame


def _report(tables, score, logs, summary):
    risk, cal, unc = tables["risk_metrics"], tables["calibration_summary"], tables["uncertainty_metrics"]
    decisions, alerts = tables["decision_value_curve"], tables["alert_episode_metrics"]
    parity = logs["source_parity_audit"]
    d1, d2 = (_row(risk, subset=s, fold="pooled") for s in ("D1", "D2"))
    cal_d1 = _row(cal, subset="D1", fold="pooled", probability="p_cal")
    raw_u = _row(unc, subset="D1", fold="pooled", population="all", method="raw")
    conf_u = _row(unc, subset="D1", fold="pooled", population="all", method="conformal")
    fixed1 = _row(alerts, subset="D1", fold="pooled", threshold=.10, policy="1/1")
    fixed2 = _row(alerts, subset="D1", fold="pooled", threshold=.10, policy="2/2")
    positives = [f"{v:g}" for v in THRESHOLDS if _row(decisions, subset="D1", fold="pooled", threshold=v).relative_value > 0]
    access = logs["access"]
    no_final = bool(access) and all(not item.get("historical_final_artifact_read", True) and not item.get("denied")
                                    for item in access) and not parity.get("final_holdout_parsed", True)
    lines = [
        "# Phase E — 4시간 고부하 위험·상한·경보 개발구간 결과",
        "",
        "## 먼저 확인할 12개 결과",
        "",
        "1. **실행:** 개발구간의 parity·calibration·score 평가·보고 단계 완료. 최종 보존 및 파일 봉인은 logs/preservation_audit.json과 logs/artifact_manifest.json에 기록한다.",
        f"2. **Phase C B5 parity:** score point prediction 최대 절대 차이 {_fmt(parity['point_prediction_max_abs_difference'], 10)} (허용 1e-8). E 전체 B5 {parity['all_b5_score_count']}행, Phase C MAIN10 공통행 {parity['phase_c_main10_common_count']}행을 구분한다.",
        f"3. **D1 Brier:** raw {_fmt(d1.BS_raw, 6)}, calibrated {_fmt(d1.BS_cal, 6)}, fit climatology {_fmt(d1.BS_climatology, 6)}.",
        f"4. **D1 Brier Skill:** {_fmt(d1.BSS, 4)} ({_ci(d1, 'BSS', 4)}).",
        f"5. **D1 PR-AUC:** calibrated {_fmt(d1.PR_AUC_cal, 4)} ({_ci(d1, 'PR_AUC_cal', 4)}); average precision 정의.",
        f"6. **D1 calibration 진단:** intercept {_fmt(cal_d1.intercept, 4)}, slope {_fmt(cal_d1.slope, 4)} (상태 {cal_d1.status}; 이상적 기준 0/1).",
        f"7. **D1 one-sided 상한:** raw q95 coverage {_fmt(raw_u.coverage, 4)}, U95 coverage {_fmt(conf_u.coverage, 4)}; 평균 폭 raw {_fmt(raw_u.mean_width, 3)}, U95 {_fmt(conf_u.mean_width, 3)}. 명목값 0.95.",
        f"8. **D1 양의 결정가치 C/L grid:** {', '.join(positives) if positives else '없음'}; 여섯 고정값 모두 아래 표에 있다.",
        f"9. **c=0.10 D1 episode:** 1/1 R/P/F1 {_fmt(fixed1.episode_recall)}/{_fmt(fixed1.episode_precision)}/{_fmt(fixed1.episode_F1)}, FP {int(fixed1.false_alert_episodes)}; 2/2 R/P/F1 {_fmt(fixed2.episode_recall)}/{_fmt(fixed2.episode_precision)}/{_fmt(fixed2.episode_F1)}, FP {int(fixed2.false_alert_episodes)}. FP 부담 분모는 평가 관측일이다.",
        f"10. **c=0.10 D1 2/2 direct lead:** 중앙값 {_fmt(fixed2.direct_lead_median_minutes, 1)}분, 범위 {_fmt(fixed2.direct_lead_min_minutes, 1)}–{_fmt(fixed2.direct_lead_max_minutes, 1)}분 (매칭 {int(fixed2.direct_lead_count)}건).",
        f"11. **D2:** BSS {_fmt(d2.BSS, 4)}, PR-AUC {_fmt(d2.PR_AUC_cal, 4)}, U95 coverage {_fmt(_row(unc, subset='D2', fold='pooled', population='all', method='conformal').coverage, 4)}. 세부 경보·가치는 D2 행을 함께 보아야 하므로 단일 '결론 유지'로 합치지 않는다.",
        f"12. **final holdout:** 이 보고 단계까지의 명시적 Phase E 접근 기록상 {'미열람·미파싱 확인' if no_final else '확인 미완료—접근 기록 점검 필요'}. 이전 Phase C의 historical artifact read incident를 소급해 지우는 진술은 아니다.",
        "",
        "## 목적과 고정 조건",
        "",
        "B5 Weekly-deviation Kalman의 16개 15분 간격(240분) 점예측을 statistical high-load event 확률, 상한, 정규화 결정가치, episode 경보로 변환한 개발구간 평가다. M*=B5와 h*=16은 사용자 고정 조건이다. 현재 checkout의 별도 Phase D 디렉터리는 preflight 당시 없었으며, h16을 새 Phase D 결과로 추정하지 않았다. τ는 fold별 fit power의 Q0.95로 계산한 통계적 상위 5% 사건 경계이며 계약전력·안전한계가 아니다. 기존 fit/stop/cal/score 분할과 D2 novel-profile flag를 그대로 사용한다. D2 재적합은 없다.",
        "",
        "## B5 분산·parity 및 E1 확률",
        "",
        "원래 점예측 산술을 유지하면서 origin posterior variance P를 계산했다. h=16의 관측 예측분산은 φ^(2h)P + q(1−φ^(2h))/(1−φ²) + r이다. σ는 모든 유효 행에서 유한하고 양수여야 한다. 위 parity의 전체 B5 집합에는 Phase C MAIN10 공통 코호트에서 구조적으로 제외된 행도 있다. 따라서 전체 행 MAE와 공통행 MAE를 직접 동일시하지 않는다.",
        f"전체 B5 D1 {parity['all_b5_score_metrics']['n']}행 MAE {_fmt(parity['all_b5_score_metrics']['MAE'], 5)}, Peak-MAE {_fmt(parity['all_b5_score_metrics']['Peak_MAE'], 5)}, RMSE {_fmt(parity['all_b5_score_metrics']['RMSE'], 5)}. Phase C MAIN10 공통 {parity['phase_c_main10_common_metrics']['D1']['pooled']['n']}행 MAE {_fmt(parity['phase_c_main10_common_metrics']['D1']['pooled']['MAE'], 5)}, Peak-MAE {_fmt(parity['phase_c_main10_common_metrics']['D1']['pooled']['Peak_MAE'], 5)}, RMSE {_fmt(parity['phase_c_main10_common_metrics']['D1']['pooled']['RMSE'], 5)}.",
        "",
        "raw 확률은 p_raw = 1−Φ((τ−μ)/σ)이다. fold별 cal label만으로 두 매개변수 Platt logistic p_cal = sigmoid(a+b·logit(clip(p_raw, 1e-6, 1−1e-6)))을 적합하며 b≥0이다. score label은 적합에 사용하지 않는다. score의 intercept/slope는 평가용 진단이며 p_cal을 다시 수정하지 않는다.",
        "",
        _md_table(["집합", "n / 사건", "BS raw", "BS cal", "BS climatology", "BSS", "PR-AUC raw / cal", "score intercept / slope"],
                  [[s, f"{int(r.n)} / {int(r.event_n)}", _fmt(r.BS_raw, 6), _fmt(r.BS_cal, 6),
                    _fmt(r.BS_climatology, 6), _fmt(r.BSS, 4), f"{_fmt(r.PR_AUC_raw, 4)} / {_fmt(r.PR_AUC_cal, 4)}",
                    f"{_fmt(_row(cal, subset=s, fold='pooled', probability='p_cal').intercept, 4)} / {_fmt(_row(cal, subset=s, fold='pooled', probability='p_cal').slope, 4)}"]
                   for s, r in (("D1", d1), ("D2", d2))]),
        "",
        f"D1 paired day-block 차이: BS_raw−BS_cal {_fmt(d1.BS_raw_minus_cal, 5)} ({_ci(d1, 'BS_raw_minus_cal', 5)}), BS_cal−BS_climatology {_fmt(d1.BS_cal_minus_climatology, 5)} ({_ci(d1, 'BS_cal_minus_climatology', 5)}). 1,000회, seed 42, fold별 target-calendar-day block bootstrap이다.",
        "",
        "### Reliability (고정 10개 bin)", "",
    ]
    for s in ("D1", "D2"):
        lines += [f"**{s} pooled.** 빈 bin도 0행으로 유지한다.", ""]
        raw_bins = tables["reliability_bins_raw"]
        cal_bins = tables["reliability_bins_calibrated"]
        pairs = []
        for b in range(10):
            rr = _row(raw_bins, subset=s, fold="pooled", bin=b)
            cr = _row(cal_bins, subset=s, fold="pooled", bin=b)
            pairs.append([f"[{b/10:.1f},{(b+1)/10:.1f}{']' if b==9 else ')'}",
                          int(rr.n), int(rr.distinct_target_dates), _fmt(rr.mean_predicted_probability),
                          int(rr.event_n), _fmt(rr.observed_event_rate),
                          _fmt(cr.mean_predicted_probability), int(cr.n), int(cr.distinct_target_dates),
                          int(cr.event_n), _fmt(cr.observed_event_rate)])
        lines += [_md_table(["bin", "raw n", "raw days", "raw mean p", "raw events", "raw rate",
                             "cal mean p", "cal n", "cal days", "cal events", "cal rate"], pairs), "",
                  "각 observed rate의 day-block CI는 해당 reliability_bins CSV에 보존된다.", ""]
    lines += [
        "## E2 상한 불확실성",
        "",
        "Gaussian 기준 q95_raw = μ + 1.6448536269514722σ. cal에서 표준화 잔차 (y−μ)/σ를 정렬하고 k=min(n,ceil((n+1)·0.95))번째 q*를 사용하여 U95=μ+q*σ를 만들었다. 이는 chronological calibration의 **경험적** upper coverage다. 시계열 의존성이 있어 iid distribution-free 보장은 주장하지 않는다. U95는 경보 trigger가 아닌 설명용 맥락이다.",
        "",
        _md_table(["집합", "대상", "방법", "n", "coverage (95% CI)", "mean width (95% CI)", "median", "p10 / p25 / p75 / p90"],
                  [[str(r.subset), str(r.population), str(r.method), int(r.n),
                    f"{_fmt(r.coverage, 4)} ({_ci(r, 'coverage', 4)})",
                    f"{_fmt(r.mean_width, 2)} ({_ci(r, 'mean_width', 2)})", _fmt(r.median_width, 2),
                    " / ".join(_fmt(getattr(r, f'width_p{q}'), 2) for q in (10,25,75,90))]
                   for r in _pooled(unc).itertuples(index=False)]),
        "",
        "Fold별 all/peak/nonpeak coverage와 width는 uncertainty_metrics.csv에 있다. Coverage가 0.95와 가깝더라도 폭과 peak-subset 결과를 함께 평가한다.",
        f"해석: D1 U95는 raw보다 평균 폭이 {100*(1-conf_u.mean_width/raw_u.mean_width):.1f}% 줄었지만 전체 coverage는 명목보다 {100*(conf_u.coverage-.95):.2f}%p 높다. 피크 조건부 coverage는 D1 {_fmt(_row(unc, subset='D1', fold='pooled', population='peak', method='conformal').coverage,4)}, D2 {_fmt(_row(unc, subset='D2', fold='pooled', population='peak', method='conformal').coverage,4)}로 낮다. 전체 coverage만으로 피크 위험에서 95% 보호를 제공한다고 해석할 수 없다.",
        "",
        "## E3 정규화 Cost/Loss",
        "",
        "C/L=c, L=1, C=c를 가정한 **상대 결정가치**다. 1/1 action candidate만 사용한다. E_model=c·actions+missed positions, E_no=events, E_all=c·N, E_clim=min(E_no,E_all), E_perf=c·events, V=(E_clim−E_model)/(E_clim−E_perf) (분모가 양수일 때)이다. 실제 action cost와 손실액은 없으므로 화폐 절감액이나 단일 최적 임계값을 제시하지 않는다.",
        "",
        _md_table(["집합", "C/L", "actions", "TP/FP/FN/TN", "P/R/F1", "E_model", "E_no", "E_all", "E_clim", "E_perf", "relative V"],
                  [[s, _fmt(c,2), int(r.actions), "/".join(str(int(r[k])) for k in ("TP","FP","FN","TN")),
                    "/".join(_fmt(r[k]) for k in ("precision","recall","F1")),
                    *[_fmt(r[k],2) for k in ("E_model","E_no","E_all","E_clim","E_perf")],
                    _fmt(r.relative_value,4)]
                   for s in ("D1","D2") for c in THRESHOLDS
                   for r in [_row(decisions, subset=s, fold="pooled", threshold=c)]]),
        "",
        "## E4 Episode alert와 lead",
        "",
        "1/1은 p_cal≥c일 때 Watch, 2/2는 같은 fold에서 정확히 15분 간격의 현재·직전 origin 모두 양성일 때 Confirmed Alert다. 고부하 사건과 경보는 target-time 15분 연속 구간으로 묶고 fold와 timestamp gap에서 끊는다. 아래 FP/day의 분모는 평가에 나타난 target-calendar day, 즉 **평가 관측일**이다. 실제 공장 운영일로 검증되지 않았다. Position 지표는 alert_position_metrics.csv에 보조로 둔다.",
        "",
        _md_table(["집합", "C/L", "정책", "peak episodes", "TP/FP/FN", "episode P/R/F1", "FP/평가 관측일", "lead median [min,max] min"],
                  [[s, _fmt(c,2), p, int(r.actual_peak_episodes),
                    "/".join(str(int(r[k])) for k in ("detected_peak_episodes","false_alert_episodes","missed_peak_episodes")),
                    "/".join(_fmt(r[k]) for k in ("episode_precision","episode_recall","episode_F1")),
                    _fmt(r.false_alert_episodes_per_operating_day,3),
                    f"{_fmt(r.direct_lead_median_minutes,1)} [{_fmt(r.direct_lead_min_minutes,1)}, {_fmt(r.direct_lead_max_minutes,1)}]"]
                   for s in ("D1","D2") for c in THRESHOLDS for p in POLICIES
                   for r in [_row(alerts, subset=s, fold="pooled", threshold=c, policy=p)]]),
        "",
        "Episode precision/recall·FP/day 및 direct lead의 95% CI는 alert_episode_metrics.csv에 있다. Bootstrap은 원래 매칭한 완전한 episode를 onset-day별로 resample하며 매칭 불확실성은 포함하지 않는다. Lead는 매칭된 actual onset에서 겹친 양성 forecast의 첫 발행시각을 뺀 conservative direct 값이다. 음수·0도 숨기지 않는다. 별도의 준비시간 기준은 없으므로 240분보다 큰 continuity를 직접 4시간 예측이라고 해석하지 않는다.",
        "",
        _md_table(['집합','C/L','정책','lead mean','median','p10','p25','p75','p90','경보 길이 mean/median (분)'],
                  [[r.subset,_fmt(r.threshold,2),r.policy,
                    *[_fmt(getattr(r,f'direct_lead_{q}_minutes'),1) for q in ['mean','median','p10','p25','p75','p90']],
                    f'{_fmt(15*r.alert_episode_duration_mean,1)} / {_fmt(15*r.alert_episode_duration_median,1)}']
                   for r in _pooled(alerts).itertuples(index=False)]),
        "",
        "### Miss decomposition: 고정 c=0.10, 2/2 예시", "",
        _md_table(["집합", "hit", "late_hit", "decision_miss", "forecast_miss"],
                  [[s, *[int(_row(tables['miss_decomposition'], subset=s, fold='pooled', threshold=.10,
                                   policy='2/2', miss_class=k).episodes)
                         for k in ("hit","late_hit","decision_miss","forecast_miss")]] for s in ("D1","D2")]),
        "",
        "hit은 양의 direct lead로 매칭, late_hit은 lead≤0으로 매칭, forecast_miss는 unmatched actual episode의 최대 p_cal이 고정 grid 최솟값 0.01 미만, 나머지 unmatched는 decision_miss다. 이는 원인 진단을 위한 조작적 분류이며 실제 실패 원인을 증명하지 않는다. 모든 임계값·정책의 분해와 event별 matching_reason은 CSV에 있다.",
        "",
        "## D2 견고성과 네 layer의 판단",
        "",
        f"D2는 원래 D1의 {len(score)}행 중 {int(score.is_d2_novel_profile.sum())}행이며 동일 fit/cal parameter와 확률·상한·정책을 평가했다. D1과 D2의 값·CI·episode burden 차이를 위 표에서 별도로 확인해야 한다. D2만 보고 재보정하거나 임계값을 재선택하지 않았다.",
        "",
        f"- **E1 Probability:** D1 BSS {_fmt(d1.BSS,4)} ({_ci(d1,'BSS',4)}), D2 BSS {_fmt(d2.BSS,4)}. Reliability와 intercept/slope의 편차를 함께 본다. 좋은 Brier가 곧 경보 타당성을 뜻하지 않는다.",
        f"  판단: 보정 확률은 climatology보다 Brier가 낮다. 다만 D1 intercept {_fmt(cal_d1.intercept,3)} 및 D2 intercept/slope {_fmt(_row(cal,subset='D2',fold='pooled',probability='p_cal').intercept,3)}/{_fmt(_row(cal,subset='D2',fold='pooled',probability='p_cal').slope,3)}의 편차가 남아 완전한 calibration이라고 판정하지 않는다.",
        f"- **E2 Uncertainty:** D1 upper coverage raw {_fmt(raw_u.coverage,4)}, conformal {_fmt(conf_u.coverage,4)}, 평균 폭 {_fmt(conf_u.mean_width,2)}. Peak subset 및 D2와 함께 명목 0.95와 폭의 tradeoff를 평가한다.",
        "  판단: raw 대비 상한 폭은 줄었으나 전체 과대 coverage와 피크 조건부 과소 coverage가 공존한다. 상한의 신뢰성은 조건부로 제한된다.",
        f"- **E3 Decision Value:** D1 양수 grid {', '.join(positives) if positives else '없음'}; D2 곡선과 차이를 병기한다. 실제 C/L은 알려져 있지 않아 운영값 선정은 판정불가다.",
        "  판단: 고정 grid에서 관측된 정규화 가치는 유용성 근거다. 이 양수 여부는 point estimate이며 현장 비용·실행 효과의 입증이 아니다.",
        f"- **E4 Alert:** c=0.10 D1 2/2 episode recall {_fmt(fixed2.episode_recall)}, FP/day {_fmt(fixed2.false_alert_episodes_per_operating_day)}, direct lead median {_fmt(fixed2.direct_lead_median_minutes,1)}분. 1/1 및 여섯 임계값의 경보부담과 D2를 함께 판단한다. 이 한 점은 대표 표시일 뿐 선택된 정책이 아니다.",
        f"  판단: 고정 c=0.10에서는 2/2가 동일 {int(fixed2.detected_peak_episodes)}개 사건을 탐지하면서 FP를 {int(fixed1.false_alert_episodes)}→{int(fixed2.false_alert_episodes)}로 줄였다. 그러나 전체 {int(fixed2.actual_peak_episodes)}개 중 탐지 비율이 낮다. 긴 lead는 탐지된 사건에 한정되므로 운영 가능한 피크 예방을 입증하지 않는다.",
        "",
        "D2의 Brier skill과 ranking 개선 방향은 유지되지만 calibration 편차 및 피크 조건부 coverage 한계도 유지된다. D2 fold 1에는 고부하 사건이 없어 해당 fold의 PR-AUC·calibration slope는 미정의로 남긴다. 이는 계절·상태별 일반화의 공백이며 전체 D2 지표로 덮지 않는다.",
        "",
        "## 제한, 출처, 다음 결정 경계",
        "",
        "Expected Exceedance 회귀는 과거 탐색에서 systematic underprediction 문제가 있어 이번 고정 protocol의 Main decision input에서 제외했다. 과거 p OR q95 / p OR U95 결합은 확률 비용규칙과 상한 표시를 섞어 경보 근거를 바꾸므로 사용하지 않았다. 현재 경보 입력은 p_cal≥c뿐이다. 실제 생산계획, 날씨, 인력, tariff, action execution은 평가 입력이 아니다.",
        "",
        "이 결과는 development 구간의 statistical high-load event 평가다. 실제 행동비용, 준비시간, 현장 경보절차, 안전·계약한계와 연결할 증거가 없다. Final holdout은 이번 실행에서 사용하지 않았고, 최종 평가로 넘어가려면 코드·파라미터·정책·보고 기준과 보존 감사가 모두 동결된 뒤 별도 승인 단계가 필요하다. 이 보고서 자체는 final 평가 승인 또는 수행 기록이 아니다.",
        "",
        f"검증 기록: 독립 재계산 결과는 {logs['independent_metric_audit']['recomputed']}이다. 독립 검토의 실행 전 코드 잠금/최종 manifest 기록 문제를 수정했고 원래 parity seal은 보존했다. F1은 TP=0, FP+FN>0인 행의 빈값을 count formula 2TP/(2TP+FP+FN)=0으로 정정했다. 원본 CSV와 수정 해시, 나머지 모든 셀 불변 검사는 logs/f1_correction_audit.json과 logs/metric_correction_v1/에 있다. 확률·예측·정책·매칭·비용·coverage는 바뀌지 않았다. D2 단일 클래스 PR-AUC 검증기의 잘못된 기대값도 기록하고 바로잡았다.",
        "",
        "## Figures and machine-readable evidence", "",
    ]
    for name in FIGURES:
        lines.append(f"![{name}](figures/{name})")
        lines.append("")
    lines += ["모든 원시 수치는 `tables/*.csv`, `predictions/phase_e_h16_oof.parquet`, "
              "`logs/source_parity_audit.json`, `logs/calibration_fit.json`, `logs/conformal_fit.json`에 있다. "
              "최종 `logs/leakage_audit.json`과 `logs/artifact_manifest.json`은 report 이후 finalize 단계에서 생성된다.", ""]
    return "\n".join(lines)


def generate(out: Path) -> None:
    """Fail closed on absent evaluation evidence, then write only new E outputs."""
    out = Path(out).resolve()
    if out.name != "phase_e" or out.parent.name != "outputs":
        raise ValueError("Report output must be the outputs/phase_e namespace")
    target = out/"phase_e_result.md"
    summary_target = out/"tables/phase_e_summary.csv"
    expected = [target, summary_target, *(out/"figures"/name for name in FIGURES)]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(f"Phase E report artifacts already exist: {existing}")
    configure()
    tables = {name: pd.read_csv(out/"tables"/f"{name}.csv") for name in TABLES}
    required = {
        "risk_metrics": ("subset","fold","n","event_n","BS_raw","BS_cal","BS_climatology","BSS","PR_AUC_cal"),
        "risk_metrics_by_fold": ("subset","fold","BS_raw","BS_cal","BS_climatology"),
        "uncertainty_metrics": ("subset","fold","population","method","coverage","mean_width"),
        "decision_value_curve": ("subset","fold","threshold","actions","relative_value"),
        "alert_episode_metrics": ("subset","fold","threshold","policy","episode_recall","episode_precision","false_alert_episodes"),
        "alert_episode_events": ("subset","fold","threshold","policy","status","actual_start","alert_start","direct_lead_minutes"),
    }
    for name, cols in required.items():
        _require_columns(tables[name], name, cols)
    score = pd.read_parquet(out/"predictions/phase_e_h16_oof.parquet")
    _require_columns(score, "score", ("fold","target_time","y","mu","tau","q95_raw","U95","p_cal","is_d2_novel_profile"))
    score["target_time"] = pd.to_datetime(score.target_time)
    if not score.model.eq("B5").all() or not score.horizon.eq(16).all() or \
            score.target_time.ge(pd.Timestamp("2021-08-09 09:45:00")).any():
        raise ValueError("Only pre-boundary B5 h16 score rows can enter Phase E reporting")
    parity = json.loads((out/"logs/source_parity_audit.json").read_text(encoding="utf-8"))
    if not parity.get("parity_passed") or float(parity["point_prediction_max_abs_difference"]) > 1e-8:
        raise ValueError("Source parity gate is not satisfied")
    access_files = sorted((out/"logs").glob("access_*.json"))
    access = [json.loads(path.read_text(encoding="utf-8")) for path in access_files]
    if not access or any(row.get("historical_final_artifact_read", True) or row.get("denied") for row in access):
        raise ValueError("Phase E access audit is absent or records a denied/historical access")
    audit = json.loads((out/'logs/independent_metric_audit_after_f1_correction.json').read_text(encoding='utf-8'))
    if audit.get('passed') is not True:
        raise ValueError('Independent metric audit has not passed')
    logs = {"source_parity_audit": parity, "access": access, 'independent_metric_audit':audit}
    summary = _summary_rows(tables)
    markdown = _report(tables, score, logs, summary)
    figures = out/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    _figure_reliability(figures, tables)
    _figure_brier(figures, tables)
    _figure_coverage(figures, tables)
    _figure_width(figures, score)
    _figure_decision(figures, tables)
    _figure_episode(figures, tables)
    _figure_lead(figures, tables)
    _figure_cases(figures, tables, score)
    summary.to_csv(summary_target, index=False)
    target.write_text(markdown, encoding="utf-8")
