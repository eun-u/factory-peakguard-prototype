"""FVA and uncertainty diagnostics computed from identical OOF scoring rows."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .evaluate import score_predictions, match_episodes
from .viz import configure, plt, save
from .bootstrap import day_mean_ci


def _report_score(frame):
    """Use raw quantiles when the pooled prediction schema has empty cal columns."""
    empty_cal = [col for col in ("q90_cal", "q95_cal", "q975_cal")
                 if col in frame and not frame[col].notna().any()]
    return score_predictions(frame.drop(columns=empty_cal))


def _coverage_count(frame, alpha, top=False):
    suffix = str(int(alpha * 100))
    calibrated = f"q{suffix}_cal"
    raw = f"q{suffix}"
    column = calibrated if calibrated in frame and frame[calibrated].notna().any() else raw
    if column not in frame:
        return 0
    eligible = frame.y.notna() & frame.pred.notna() & frame[column].notna()
    if top:
        if "q50_top_edge" in frame and frame.q50_top_edge.notna().any():
            eligible &= frame.q50.ge(frame.q50_top_edge)
        elif "q50" in frame and frame.q50.notna().any():
            eligible &= frame.q50.ge(frame.q50.quantile(.9))
        else:
            return 0
    return int(eligible.sum())


def _coverage_ci(frame, alpha, top, cfg):
    suffix = str(int(alpha*100))
    column = f"q{suffix}_cal" if f"q{suffix}_cal" in frame and frame[f"q{suffix}_cal"].notna().any() else f"q{suffix}"
    eligible = frame.y.notna() & frame[column].notna()
    if top:
        eligible &= frame.q50.ge(frame.q50_top_edge) if "q50_top_edge" in frame else frame.q50.ge(frame.q50.quantile(.9))
    part = frame.loc[eligible]
    low, high = day_mean_ci(part, part.y.le(part[column]).astype(float), n=cfg["bootstrap"]["n"], seed=cfg["seed"])
    return {"ci_low": low, "ci_high": high, "target_days": part.target_time.dt.normalize().nunique()}


def _gain(base, candidate, cfg):
    keys = ["origin", "target_time", "fold", "horizon"]
    merged = base[keys+["y", "pred", "tau"]].merge(candidate[keys+["pred"]], on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
    merged = merged.loc[merged.y.gt(merged.tau) & merged.pred_base.notna() & merged.pred_candidate.notna()].copy()
    if merged.empty:
        return {}
    merged["base_error"] = (merged.y-merged.pred_base).abs()
    merged["new_error"] = (merged.y-merged.pred_candidate).abs()
    grouped = merged.groupby(merged.target_time.dt.normalize()).agg(base=("base_error", "sum"), new=("new_error", "sum"), n=("y", "size"))
    rng = np.random.default_rng(cfg["seed"])
    ids = rng.integers(0, len(grouped), size=(cfg["bootstrap"]["n"], len(grouped)))
    b = grouped.base.to_numpy()[ids].sum(axis=1)
    c = grouped.new.to_numpy()[ids].sum(axis=1)
    n = grouped.n.to_numpy()[ids].sum(axis=1)
    relative = np.divide(b-c, b, out=np.full_like(b, np.nan), where=b>0)
    absolute = (b-c)/n
    lo, hi = np.nanquantile(relative, [.025, .975])
    alo, ahi = np.nanquantile(absolute, [.025, .975])
    return {"gain": float((grouped.base.sum()-grouped.new.sum())/grouped.n.sum()),
            "gain_ci_low": alo, "gain_ci_high": ahi,
            "improvement_fraction": float(1-grouped.new.sum()/grouped.base.sum()),
            "improvement_ci_low": lo, "improvement_ci_high": hi, "paired_peak_n": len(merged)}


def build_model_report(predictions, selection, cfg, outdir):
    out = Path(outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    configure()
    horizon = cfg["primary_horizon"]
    pred = predictions.loc[predictions.horizon.eq(horizon)].copy()
    choice = selection["by_horizon"][str(horizon)]
    adopted = selection.get("adoption", {})
    get = lambda name: pred.loc[pred.model.eq(name)].sort_values("target_time")
    holiday = adopted.get(f"h{horizon}_holiday", {}).get("adopted", False)
    stem = "lgbm" if holiday else "lgbm_no_holiday"
    # Each optional point candidate is compared to the model specified by its
    # adoption criterion. Rejected candidates never become the next reference.
    stages = [("CBL 대표", choice.get("cbl"), True, "point", None),
              ("Persistence 대표", choice.get("persistence"), True, "point", choice.get("cbl")),
              ("LightGBM 기본", "lgbm_no_holiday", True, "point", choice.get("persistence")),
              ("공휴일 특징 후보", "lgbm", holiday, "point", "lgbm_no_holiday")]
    last_adopted_point = stem
    for weight in cfg.get("peak_weight", [1, 2, 4]):
        if float(weight) == 1:
            continue
        label = int(weight) if float(weight).is_integer() else weight
        name = f"{stem}_weight_{label}"
        accepted = bool(adopted.get(f"h{horizon}_weight_{label}", {}).get("adopted", False))
        stages.append((f"피크 가중 {label} 후보", name, accepted, "point", stem))
        if accepted and not get(name).empty:
            last_adopted_point = name
    stages.extend([("분위수 원본", "lgbm_quantile_raw", True, "uncertainty", None),
                   ("전역 보정 A", "lgbm_quantile_a", True, "uncertainty", "lgbm_quantile_raw"),
                   ("구간별 보정 B", "lgbm_quantile_b", choice.get("conformal") == "b",
                    "uncertainty", "lgbm_quantile_a"),
                   ("최종 점예측", choice["point_model"], True, "point", last_adopted_point)])
    rows = []
    cbl = get(choice.get("cbl"))
    for stage, name, accept, domain, reference_name in stages:
        frame = get(name)
        if frame.empty:
            continue
        metrics = _report_score(frame)
        rec = {"stage": stage, "model": name, "adopted": bool(accept),
               "metric_domain": domain, "previous_model": reference_name,
               "selected_final": stage == "최종 점예측",
               "is_selected_point_model": name == choice["point_model"], **metrics}
        reference = get(reference_name)
        if not reference.empty and domain == "point":
            rec.update({"vs_previous_"+k: v for k,v in _gain(reference, frame, cfg).items()})
        if not reference.empty and domain == "uncertainty":
            prior = _report_score(reference)
            for alpha in (.9, .95):
                key = f"top_coverage_{alpha}"
                if np.isfinite(metrics.get(key, np.nan)) and np.isfinite(prior.get(key, np.nan)):
                    rec[f"vs_previous_top_coverage_error_reduction_{alpha}"] = (
                        abs(prior[key] - alpha) - abs(metrics[key] - alpha))
            old_pin, new_pin = prior.get("pinball_mean", np.nan), metrics.get("pinball_mean", np.nan)
            if np.isfinite(old_pin) and np.isfinite(new_pin) and old_pin > 0:
                rec["vs_previous_pinball_improvement_fraction"] = (old_pin - new_pin) / old_pin
        if not cbl.empty and domain == "point":
            rec.update({"vs_cbl_"+k: v for k,v in _gain(cbl, frame, cfg).items()})
        rows.append(rec)
    fva = pd.DataFrame(rows)
    fva.to_csv(out/"tables/T2-1_fva.csv", index=False)
    adoption = pd.DataFrame([{"criterion": key, **value} for key,value in adopted.items()])
    adoption.to_csv(out/"tables/T5-1_adoption.csv", index=False)
    pooled = pd.DataFrame([{"model": name, **_report_score(g)} for name,g in pred.groupby("model")])
    pooled.to_csv(out/"tables/T2-2_pooled_metrics.csv", index=False)
    coverage_rows = []
    for name in ["lgbm_quantile_raw", "lgbm_quantile_a", "lgbm_quantile_b"]:
        frame = get(name)
        if frame.empty:
            continue
        metrics = _report_score(frame)
        for alpha in (.9, .95):
            for segment,prefix in [("전체", "coverage"), ("상위 예측", "top_coverage")]:
                coverage_rows.append({"model": name, "alpha": alpha, "segment": segment,
                                      "coverage": metrics.get(f"{prefix}_{alpha}", np.nan),
                                      "coverage_error": abs(metrics.get(f"{prefix}_{alpha}", np.nan)-alpha),
                                      "scoring_n": _coverage_count(frame, alpha, top=segment == "상위 예측"),
                                      "total_n": len(frame), **_coverage_ci(frame, alpha, segment == "상위 예측", cfg)})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out/"tables/coverage_comparison.csv", index=False)
    by_fold = []
    for (name, fold), group in pred.loc[pred.model.str.startswith("lgbm_quantile_")].groupby(["model", "fold"]):
        metrics = _report_score(group)
        by_fold.append({"model": name, "fold": fold, "top_n": _coverage_count(group, .95, True),
                        "n": len(group), "top_q95_coverage": metrics.get("top_coverage_0.95"),
                        "tau": group.tau.iloc[0], "top_edge": group.q50_top_edge.iloc[0],
                        **_coverage_ci(group, .95, True, cfg)})
    pd.DataFrame(by_fold).to_csv(out/"tables/coverage_by_fold.csv", index=False)
    q95b = get("lgbm_quantile_b")
    top = q95b.loc[q95b.q50.ge(q95b.q50_top_edge)].copy()
    top["date"] = top.target_time.dt.normalize()
    top["missed_upper"] = top.y.gt(top.q95_cal)
    top["actual_peak"] = top.y.gt(top.tau)
    top.groupby(["fold", "date"]).agg(n=("y", "size"), missed_upper=("missed_upper", "sum"),
                                      actual_peak=("actual_peak", "sum")).reset_index().to_csv(
        out/"tables/coverage_top_by_day.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    subset = coverage.loc[coverage.alpha.eq(.95)]
    labels = {"lgbm_quantile_raw":"보정 전", "lgbm_quantile_a":"전역 A", "lgbm_quantile_b":"구간별 B"}
    for segment,g in subset.groupby("segment", sort=False):
        ax.plot(g.model.map(labels), g.coverage, marker="o", label=segment)
    ax.axhline(.95, color="#B8741A", linestyle="--", label="목표 0.95")
    ax.set(title="F2-3 보정 전후 커버리지 · 개발 검증", ylabel="실제 커버리지", ylim=(0,1.02))
    ax.legend()
    save(fig, out/"figures/F2-3_coverage.png")
    main_models = [choice.get("cbl"), choice.get("persistence"), choice["point_model"]]
    counts = pooled.set_index("model").reindex(main_models)[["position_tp", "position_fp"]]
    counts.index = ["CBL 대표", "Persistence 대표", "최종 점예측"]
    ax = counts.plot.bar(figsize=(7,4), color=["#3E6283", "#B8741A"], rot=0)
    ax.set(title="F2-2 위치 적중·오경보", ylabel="15분 위치 수")
    ax.legend(["적중", "오경보"])
    save(ax.figure, out/"figures/F2-2_hits_false_alarms.png")
    final = get(choice["point_model"])
    episodes = match_episodes(final.y.gt(final.tau), final.alert, final.target_time, final.y, final.pred)
    timing = episodes.get("timing_errors_minutes", [])
    magnitude = episodes.get("magnitude_errors", [])
    pd.DataFrame({"timing_error_minutes":timing, "magnitude_error":magnitude}).to_csv(out/"tables/peak_episode_errors.csv", index=False)
    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist(timing,bins=15,color="#3E6283");axes[0].set(xlabel="예측 최대 - 실제 최대 시각 (분)",ylabel="에피소드 수")
    axes[1].hist(magnitude,bins=15,color="#B8741A");axes[1].set(xlabel="예측 최대 - 실제 최대 (원자료 단위)")
    fig.suptitle("F2-4 매칭된 피크의 시점·크기 오차")
    save(fig,out/"figures/F2-4_episode_errors.png")
    return {"fva": str(out / "tables/T2-1_fva.csv"),
            "coverage": str(out / "tables/coverage_comparison.csv")}
