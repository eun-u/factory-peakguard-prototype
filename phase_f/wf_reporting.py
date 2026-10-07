"""Evidence-bound report for the locked walk-forward Phase F transaction.

This module only reads predictions and tables named by the completed lock. It
never selects a new candidate or reads the sealed final holdout.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_f.diagnostics import summarize_errors
from phase_f.registry import config_hash, sha256, write_json
from phase_f.wf_evaluation import BASELINES, load_predictions, prediction_path


ASHRAE_SOURCE = "https://handbook.ashrae.org/Handbooks/F25/IP/F25_Ch19/F25_Ch19_ip.aspx"
METRICS = ("wf_explore_AUC_MAE", "wf_explore_AUC_PeakMAE", "wf_explore_h16_Peak")


def _verified(record: dict, name: str) -> None:
    if not isinstance(record, dict) or record.get("lock_sha256") != config_hash(
        {key: value for key, value in record.items() if key != "lock_sha256"}
    ):
        raise ValueError(f"{name} is not a sealed record")


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "NA"
    except (TypeError, ValueError):
        return "NA" if value is None else str(value)


def _md(headers: list[str], rows: list[list[object]]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(clean, row)) + " |" for row in rows)])


def _registry(prepared) -> list[dict]:
    folder = prepared.out / "logs/experiments"
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(folder.glob("*.json"))]


def _preflight_evidence(prepared, lock: dict) -> list[str]:
    folder = prepared.root / "outputs/phase_f/logs/revision_20261006"
    claims = []
    for filename, field in (("preflight_audit.json", "preflight_sha256"),
                            ("alias_calendar_audit.json", "alias_calendar_sha256")):
        expected = lock.get(field)
        path = folder / filename
        if expected is None:
            claims.append(f"{filename}: selection lock has no SHA-256 reference; status UNKNOWN.")
            continue
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Preflight evidence changed: {filename}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if filename == "preflight_audit.json":
            r1 = record.get("full_pc3_baselines", {}).get("R1", {})
            claims.append("Preflight R1 authorized full Phase C development-score parity: "
                          f"AUC-MAE {_fmt(r1.get('AUC_MAE'))}, actual-peak AUC-MAE "
                          f"{_fmt(r1.get('AUC_PeakMAE'))}; maximum prediction difference "
                          f"{_fmt(record.get('R1_prediction_max_abs_difference'))}; "
                          f"parity_passed={record.get('R1_parity_passed')}. "
                          "This is baseline replay parity, not evidence of candidate CONFIRM performance.")
            claims.append("Known-output/log audit found candidate CONFIRM metric opened before revision: "
                          f"{record.get('candidate_confirm_metric_opened_evidence')}. "
                          f"Scope limitation: {record.get('audit_limit', 'UNKNOWN')}")
        else:
            aliases = record.get("aliases", [])
            claims.append(f"Alias audit identified {len(aliases)} native_mean→median exact EXPLORE forecast aliases "
                          "for de-duplication; the alias evidence alone does not prove every later registry row was excluded. "
                          f"Saved calendar future-perturbation differences: "
                          f"{record.get('saved_calendar_future_perturbation_differences', 'UNKNOWN')}.")
    return claims


def _table(view, key: str, arm: str, name: str) -> pd.DataFrame:
    path = view.out / "tables" / key / arm / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"Missing sealed finalist table: {path}")
    return pd.read_csv(path)


def _figure(path: Path, draw) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    try:
        draw(ax)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def hourly_metrics(frame: pd.DataFrame, arm: str) -> dict[str, float | int | None]:
    """Descriptive hourly means from four observed 15-minute target slots.

    Each fold and horizon is kept separate. The denominator is the mean of
    *actual hourly targets*, unlike existing fit-mean-normalized 15-min metrics.
    """
    score = frame.loc[frame.role.eq("score") & frame.arm.eq(arm)].copy()
    if score.empty:
        raise ValueError(f"No {arm} score rows for hourly aggregation")
    score["target_time"] = pd.to_datetime(score.target_time)
    if score.duplicated(["fold", "horizon", "target_time"]).any():
        raise ValueError("Duplicate 15-minute target in hourly aggregation")
    score["hour"] = score.target_time.dt.floor("h")
    score["minute"] = score.target_time.dt.minute
    score["y"] = pd.to_numeric(score.y)
    score["pred"] = pd.to_numeric(score.pred)
    if not np.isfinite(score[["y", "pred"]].to_numpy(float)).all():
        raise ValueError("Nonfinite hourly inputs")
    hourly = score.groupby(["fold", "horizon", "hour"], observed=True).agg(
        n=("target_time", "size"), slots=("minute", lambda v: frozenset(v)),
        actual=("y", "mean"), prediction=("pred", "mean"))
    complete_slots = hourly.slots.map(lambda value: value == frozenset((0, 15, 30, 45)))
    hourly = hourly.loc[hourly.n.eq(4) & complete_slots]
    if hourly.empty:
        return {"n_hours": 0, "CVRMSE": None, "NMBE": None, "actual_hourly_mean": None}
    actual = hourly.actual.to_numpy(float)
    pred = hourly.prediction.to_numpy(float)
    denominator = float(actual.mean())
    if denominator <= 0:
        return {"n_hours": len(hourly), "CVRMSE": None, "NMBE": None,
                "actual_hourly_mean": denominator}
    return {"n_hours": int(len(hourly)),
            "CVRMSE": float(np.sqrt(np.mean((pred - actual) ** 2)) / denominator),
            "NMBE": float(np.mean(actual - pred) / denominator),
            "actual_hourly_mean": denominator}


def _diagnostics(view, key: str, output: Path) -> dict[str, dict]:
    audits = {}
    for arm in ("EXPLORE", "CONFIRM"):
        result = summarize_errors(load_predictions(view, key, arm), arm=arm, selected=True)
        destination = output / "f10" / arm
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in result.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(destination / f"{name}.csv", index=False)
        write_json(destination / "audit.json", result["audit"])
        audits[arm] = result["audit"]
    return audits


def _paired_rows(view, keys: list[str], output: Path) -> pd.DataFrame:
    rows = []
    for key in keys:
        for arm in ("EXPLORE", "CONFIRM"):
            pair = _table(view, key, arm, "pairwise_ci")
            part = pair.loc[pair.horizon.isna() & pair.baseline.isin(BASELINES)
                            & pair.dataset.isin(("D1", "D2"))]
            if part.empty:
                raise ValueError(f"No paired AUC intervals for {key}/{arm}")
            for _, item in part.iterrows():
                rows.append({"candidate": key, "arm": arm, "baseline": item.baseline,
                             "dataset": item.dataset, "metric": item.metric,
                             "estimate": item.estimate, "ci_low": item.ci_low,
                             "ci_high": item.ci_high, "ci_status": item.ci_status,
                             "n_blocks": item.n_blocks, "bootstrap_n": item.bootstrap_n})
    table = pd.DataFrame(rows)
    table.to_csv(output / "finalist_paired_ci.csv", index=False)
    return table


def _figures(view, rows: list[dict], figure_dir: Path, key: str | None) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in rows if r.get("status") == "completed"
                  and r.get("wf_explore_AUC_MAE") is not None]
    candidate_map = {r["exp_id"]: r for r in candidates}
    plot_keys = [BASELINES["B5"]] + ([key] if key and key != BASELINES["B5"] else [])
    def horizons(ax):
        for model in plot_keys:
            part = _table(view, model, "EXPLORE", "pooled")
            part = part.loc[part.dataset.eq("D1")].sort_values("horizon")
            ax.plot(part.horizon, part.MAE, marker="o", label=f"{model} MAE")
            ax.plot(part.horizon, part.Peak_MAE, linestyle="--", label=f"{model} actual peak MAE")
        ax.set(xlabel="15-minute horizon", ylabel="absolute error", title="Weekly EXPLORE horizon errors")
        ax.legend(fontsize=7)
    _figure(figure_dir / "horizon_mae_peak.png", horizons)

    context = []
    for row in candidates:
        cfg = json.loads(row.get("config_json", "{}"))
        length = cfg.get("context_length")
        if length is not None and row["exp_id"].startswith("F6"):
            context.append((int(length), float(row["wf_explore_AUC_MAE"]), row["exp_id"]))
    def contexts(ax):
        for length, mae, _ in sorted(context): ax.scatter(length, mae, color="#2a6f97")
        ax.set(xlabel="context length (15-minute steps)", ylabel="EXPLORE AUC-MAE",
               title="F6 context experiments (observed settings only)")
        if not context: ax.text(.5, .5, "No eligible context comparisons", ha="center", transform=ax.transAxes)
    _figure(figure_dir / "context_length.png", contexts)

    ablations = []
    for model, row in candidate_map.items():
        if model.startswith(("F1-core-", "F1-union-", "F1-union-without-")):
            ablations.append((model, float(row["wf_explore_AUC_MAE"])))
    ablations = sorted(ablations, key=lambda item:
                       ("-without-" in item[0], item[0]))[:18]
    def features(ax):
        if ablations:
            labels, values = zip(*ablations)
            ax.barh(range(len(values)), values, color="#779b93")
            ax.set_yticks(range(len(values)), labels, fontsize=7)
            ax.invert_yaxis()
        else: ax.text(.5, .5, "No completed feature ablations", ha="center", transform=ax.transAxes)
        ax.set(xlabel="EXPLORE AUC-MAE", title="Information-budget ablations")
    _figure(figure_dir / "feature_ablation.png", features)

    def signed(ax):
        for model in plot_keys:
            frame = load_predictions(view, model, "EXPLORE")
            part = frame.loc[frame.role.eq("score") & frame.arm.eq("EXPLORE")
                             & frame.y.gt(frame.tau)]
            if len(part):
                ax.hist((part.pred - part.y).to_numpy(float), bins=32, histtype="step",
                        density=True, label=model)
        ax.axvline(0, color="black", lw=.8)
        ax.set(xlabel="prediction − actual on actual peak rows", ylabel="density",
               title="Signed peak errors (EXPLORE)")
        ax.legend(fontsize=7)
    _figure(figure_dir / "peak_signed_errors.png", signed)

    chosen = key or BASELINES["B5"]
    frame = load_predictions(view, chosen, "EXPLORE")
    part = frame.loc[frame.role.eq("score") & frame.arm.eq("EXPLORE") & frame.horizon.eq(16)].copy()
    part["target_time"] = pd.to_datetime(part.target_time)
    part = part.sort_values(["fold", "target_time"])
    fold = part.fold.iloc[len(part) // 2]
    part = part.loc[part.fold.eq(fold)].head(96)
    def representative(ax):
        ax.plot(part.target_time, part.y, label="actual", lw=1.2)
        ax.plot(part.target_time, part.pred, label=chosen, lw=1)
        ax.set(xlabel="target time", ylabel="measured units (unverified)",
               title=f"Frozen EXPLORE week, h16, fold {fold}")
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
    _figure(figure_dir / "representative_week.png", representative)

    if key and key != BASELINES["B5"]:
        weekly = _table(view, key, "EXPLORE", "fold")
        baseline = _table(view, BASELINES["B5"], "EXPLORE", "fold")
        a = weekly.loc[weekly.dataset.eq("D1")].groupby("fold").MAE.mean()
        b = baseline.loc[baseline.dataset.eq("D1")].groupby("fold").MAE.mean()
        difference = (b - a).dropna()
        def week_delta(ax):
            ax.bar(range(len(difference)), difference.values, color="#347c6c")
            ax.set_xticks(range(len(difference)), [str(v) for v in difference.index])
            ax.axhline(0, color="black", lw=.8)
            ax.set(xlabel="locked ISO week fold", ylabel="B5 MAE − candidate MAE",
                   title="Weekly paired MAE difference (EXPLORE)")
        _figure(figure_dir / "weekly_b5_delta.png", week_delta)


def final_report(prepared, lock: dict, done: dict) -> Path:
    """Publish only an authenticated, fully completed CONFIRM transaction."""
    _verified(lock, "selection lock")
    _verified(done, "CONFIRM completion")
    if done.get("selection_lock_sha256") != lock["lock_sha256"] or done.get("holdout_read") is not False:
        raise ValueError("CONFIRM completion does not match selection lock")
    complete = prepared.out / "logs/confirm_once.json"
    if not complete.is_file() or json.loads(complete.read_text(encoding="utf-8")) != done:
        raise ValueError("CONFIRM completion record is absent or changed")
    for relative, digest in done.get("artifacts", {}).items():
        target = (prepared.out / relative).resolve()
        if not target.is_relative_to(prepared.out.resolve()) or sha256(target) != digest:
            raise ValueError(f"Finalist artifact changed: {relative}")
    if not done.get("artifacts") or not done.get("results", {}).get("WF"):
        raise ValueError("Incomplete CONFIRM evidence")
    for key, digest in lock.get("prediction_sha256", {}).items():
        view = type("View", (), {"out": prepared.out / "finalists_v2"})()
        if sha256(prediction_path(view, key)) != digest:
            raise ValueError(f"Frozen EXPLORE prediction changed: {key}")
    view = type("View", (), {"out": prepared.out / "finalists_v2"})()
    if set(done["results"]["WF"]) != {spec["id"] for spec in lock["specs"]}:
        raise ValueError("CONFIRM result coverage differs from frozen finalists")
    if set(done["results"].get("pc3", {})) != set(done["results"]["WF"]):
        raise ValueError("Auxiliary 3-fold result coverage differs")
    representative = done.get("representative_candidate")
    if representative is not None and representative not in lock["candidates"]:
        raise ValueError("Representative was not locked before CONFIRM")

    output = prepared.out / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = _registry(prepared)
    scored = [r for r in rows if r.get("status") == "completed"
              and r.get("wf_explore_AUC_MAE") is not None]
    scored.sort(key=lambda r: (float(r["wf_explore_AUC_MAE"]), r["exp_id"]))
    if not scored:
        raise ValueError("No completed EXPLORE scoreboard")
    top20 = scored[:20]
    scored_by_id = {row["exp_id"]: row for row in scored}
    columns = ["exp_id", "family", "wf_explore_AUC_MAE", "wf_explore_AUC_PeakMAE",
               "wf_explore_h16_Peak", "wf_explore_AUC_MAE_seed_sd", "E_c10_22_recall",
               "E_B5_c10_22_recall", "E_c10_22_F1", "E_peak_coverage"]
    pd.DataFrame(top20).reindex(columns=columns).to_csv(output / "top20_explore.csv", index=False)
    ablations = []
    for kind in ("ridge", "lightgbm"):
        core = scored_by_id.get(f"F1-core-{kind}", {})
        union = scored_by_id.get(f"F1-union-{kind}", {})
        old = core.get("wf_explore_AUC_MAE")
        new = union.get("wf_explore_AUC_MAE")
        ablations.append({"family": kind, "core_AUC_MAE": old, "union_AUC_MAE": new,
                          "MAE_improvement_core_minus_union":
                          float(old) - float(new) if old is not None and new is not None else None,
                          "interpretation": "lower_MAE_observed" if old is not None and new is not None
                          and float(new) < float(old) else "not_supported_or_unavailable"})
    ablation_table = pd.DataFrame(ablations)
    ablation_table.to_csv(output / "feature_ablation.csv", index=False)
    context_table = pd.DataFrame([{"candidate": row["exp_id"],
                                   "context_length": json.loads(row.get("config_json", "{}"))["context_length"],
                                   "wf_explore_AUC_MAE": row["wf_explore_AUC_MAE"]}
                                  for row in scored if row["exp_id"].startswith("F6")
                                  and "context_length" in json.loads(row.get("config_json", "{}"))])
    context_table.to_csv(output / "context_length.csv", index=False)
    candidate_ids = list(lock["candidates"])
    result_rows = []
    for protocol in ("WF", "pc3"):
        for key, fields in done["results"][protocol].items():
            result_rows.append({"protocol": protocol, "candidate": key, **fields})
    pd.DataFrame(result_rows).to_csv(output / "finalists_confirm.csv", index=False)
    ci = _paired_rows(view, list(dict.fromkeys([*BASELINES.values(), *candidate_ids])), output)
    diagnose_key = representative or (candidate_ids[0] if candidate_ids else BASELINES['B5'])
    diagnostic_audits = _diagnostics(view, diagnose_key, output) if diagnose_key else {}
    figure_dir = prepared.root / "outputs/phase_f/figures/walkforward_v2"
    _figures(view, rows, figure_dir, diagnose_key)

    absolute = []
    for key in dict.fromkeys([*BASELINES.values(), *candidate_ids]):
        for arm in ("EXPLORE", "CONFIRM"):
            pooled = _table(view, key, arm, "pooled")
            d1 = pooled.loc[pooled.dataset.eq("D1")]
            frame = load_predictions(view, key, arm)
            hourly = hourly_metrics(frame, arm)
            absolute.append({"candidate": key, "arm": arm,
                             "15min_AUC_nMAE_fit_mean": d1.nMAE.mean(),
                             "15min_AUC_CVRMSE_fit_mean": d1.CVRMSE.mean(),
                             "15min_AUC_NMBE_fit_mean": d1.NMBE.mean(),
                             "hourly_complete_hours": hourly["n_hours"],
                             "hourly_CVRMSE_actual_mean": hourly["CVRMSE"],
                             "hourly_NMBE_actual_mean": hourly["NMBE"]})
    absolute_table = pd.DataFrame(absolute)
    absolute_table.to_csv(output / "absolute_performance.csv", index=False)
    counts = Counter(r.get("status", "unknown") for r in rows)
    seed_forecasts = 0
    audited_cells = 0
    for path in (prepared.out / "predictions/EXPLORE").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for seed in record.get("audit", {}).get("seeds", []):
            seed_forecasts += 1
            audited_cells += len(seed.get("audit", {}).get("cells", []))
    family = pd.DataFrame([{"family": name, "registered": len(part),
                            "scored_completed": sum(r.get("status") == "completed" for r in part),
                            "stop_only_completed": sum(r.get("tpe_stop_completed") is True for r in part),
                            "failed": sum(r.get("status") == "failed" for r in part)}
                           for name in sorted({r.get("family", "UNKNOWN") for r in rows})
                           for part in [[r for r in rows if r.get("family", "UNKNOWN") == name]]])
    family.to_csv(output / "family_counts.csv", index=False)
    tuning = [json.loads(p.read_text(encoding="utf-8")) for p in
              sorted((prepared.out / "logs/tuning").glob("*.json"))]
    study_trials = {kind: max((int(t.get("completed_trials", 0)) for t in tuning
                               if t.get("kind") == kind), default=0)
                    for kind in ("lightgbm", "xgboost", "catboost")}
    trial_states = Counter()
    for path in sorted((prepared.out / "logs/tuning").glob("*_trials.csv")):
        table = pd.read_csv(path)
        if "state" not in table:
            raise ValueError(f"TPE trial registry missing state: {path}")
        trial_states.update(table.state.astype(str))
    paired_auc = ci.loc[ci.metric.isin(("AUC_MAE_improvement", "AUC_PeakMAE_degradation"))]
    seed_table = pd.DataFrame([{"candidate": r["exp_id"], "n_seeds": r.get("n_seeds"),
                                "AUC_MAE_seed_sd": r.get("wf_explore_AUC_MAE_seed_sd")}
                               for r in top20])
    seed_table.to_csv(output / "top20_seed_sd.csv", index=False)

    preflight_claims = _preflight_evidence(prepared, lock)
    report = ["# Phase F — revised weekly walk-forward experiment", "",
              f"Verdict: **{done['verdict']}**. Representative candidate: **{representative or 'none'}**.", "",
              "This is a development-set candidate proposal. It does not authorize frozen-model replacement, holdout evaluation, or commercial deployment.", "",
              "## Execution and coverage", "",
              f"Registered distinct settings: {len(rows)}; settings started: "
              f"{sum(bool(r.get('started_at')) for r in rows)}; scored complete: {counts['completed']}; "
              f"stop-only TPE complete: {sum(r.get('tpe_stop_completed') is True for r in rows)}; "
              f"failed: {counts['failed']}; unsupported: {counts['unsupported']}; rejected: {counts['rejected']}.", "",
              f"Audited EXPLORE seed-forecast artifacts: {seed_forecasts}; per-seed fitted/evaluated cells exposed by adapters: "
              f"{audited_cells}. A forecast artifact can contain multiple fitted horizon/week cells; adapters without cell-level audit "
              "are not counted as zero training. Trial completion and scored completion are separate.", "",
              "TPE completed trials (highest persisted study count per family): " +
              ", ".join(f"{k}={v}" for k, v in study_trials.items()) +
              f". All persisted trial rows: COMPLETE={trial_states['COMPLETE']}, "
              f"FAIL={trial_states['FAIL']}, RUNNING={trial_states['RUNNING']}.", "",
              _md(list(family.columns), family.values.tolist()), "",
              "## EXPLORE top 20", "",
              _md(columns, [[_fmt(r.get(c)) if c not in ("exp_id", "family") else r.get(c, "NA")
                             for c in columns] for r in top20]), "",
              "The rank metric is the AUC-MAE of mean forecasts across seeds, on locked even ISO weeks only. "
              "[Full table](walkforward_v2/report/top20_explore.csv).", "",
              "## Locked finalists and CONFIRM", "",
              _md(["protocol", "candidate", "AUC-MAE", "actual-peak AUC-MAE", "predicted-peak AUC-MAE", "h16 peak MAE", "eligible", "weeks won vs B5", "E recall", "B5 recall", "E peak coverage", "B5 peak coverage"],
                  [[protocol, key, _fmt(fields.get("wf_explore_AUC_MAE")),
                    _fmt(fields.get("wf_explore_AUC_PeakMAE")), _fmt(fields.get("wf_explore_PredPeakMAE")),
                    _fmt(fields.get("wf_explore_h16_Peak")), str(fields.get("confirm_gates_met")),
                    _fmt(fields.get("wf_weeks_won_vs_B5")), _fmt(fields.get("E_c10_22_recall")),
                    _fmt(fields.get("E_B5_c10_22_recall")), _fmt(fields.get("E_peak_coverage")),
                    _fmt(fields.get("E_B5_peak_coverage"))]
                   for protocol, result in done["results"].items() for key, fields in result.items()]), "",
              "Field names retain `wf_explore_` for schema compatibility; every number in the table above is from CONFIRM. "
              "The pc3 rows reuse historical development folds and are auxiliary, not an independent external test.", "",
              "## Paired baseline intervals", "",
              _md(["candidate", "arm", "baseline", "dataset", "metric", "estimate", "CI low", "CI high", "status"],
                  [[r.candidate, r.arm, r.baseline, r.dataset, r.metric, _fmt(r.estimate),
                    _fmt(r.ci_low), _fmt(r.ci_high), r.ci_status] for r in paired_auc.itertuples()]), "",
              "Intervals use the frozen 1,000-replicate paired bootstrap tables; unavailable intervals remain NA. "
              "[Full interval table](walkforward_v2/report/finalist_paired_ci.csv).", "",
              "## Absolute error and the hourly reference", "",
              _md(["candidate", "arm", "15-min nMAE (%)", "15-min CV(RMSE) (%)", "15-min NMBE (%)", "complete hourly groups", "hourly CV(RMSE) (%)", "hourly NMBE (%)"],
                  [[r[0], r[1], *[_fmt(None if v is None else 100*v) for v in r[2:5]], r[5],
                    *[_fmt(None if v is None else 100*v) for v in r[6:8]]] for r in absolute_table.itertuples(index=False, name=None)]), "",
              f"[ASHRAE's hourly CV(RMSE) 30% reference]({ASHRAE_SOURCE}) concerns measurement-and-verification model calibration. "
              "The hourly figures above average four valid 15-minute actual and forecast slots per fold, horizon, and target hour, "
              "and divide by the observed hourly mean. They are descriptive proximity to that reference, not a commercial-readiness gate "
              "or proof of ASHRAE compliance. The 15-minute metrics use each fold's fit mean denominator and are not directly comparable.", "",
              "## Peak, daily maxima, and downstream alerts", ""]
    alternatives = done.get("overlapping_alternatives", [])
    if alternatives:
        report.extend(["CONFIRM paired intervals overlap the representative for these locked alternatives. "
                       "Peak, alert, and complexity evidence is shown for human choice; no automatic model replacement follows.", "",
                       _md(["alternative", "paired MAE CI", "peak", "alert", "complexity"],
                           [[item.get("candidate", item.get("id", "NA")),
                             str(item.get("paired_mae_ci", item.get("ci_status", "NA"))),
                             _fmt(item.get("peak", item.get("wf_explore_AUC_PeakMAE"))),
                             _fmt(item.get("alert", item.get("E_c10_22_recall"))),
                             str(item.get("complexity", "UNKNOWN"))] if isinstance(item, dict)
                            else [str(item), "NA", "NA", "NA", "UNKNOWN"] for item in alternatives]), ""])
    elif representative:
        report.extend(["No overlapping alternative was recorded in the sealed CONFIRM completion. "
                       "This does not establish that its superiority is statistically resolved.", ""])
    if diagnose_key:
        for arm in ("EXPLORE", "CONFIRM"):
            daily = _table(view, diagnose_key, arm, "daily_summary")
            daily = daily.loc[daily.dataset.eq("D1") & daily.scope.eq("pooled") &
                              daily.subset.eq("full_96") & daily.horizon.eq(16)]
            if len(daily):
                r = daily.iloc[0]
                report.append(f"{arm} h16 complete-day maximum MAE: {_fmt(r.daily_max_MAE)}; "
                              f"timing MAE: {_fmt(r.timing_MAE_minutes, 1)} minutes; "
                              f"complete days: {int(r.n_days)}.")
        combined = done.get("combined_phase_e")
        report.extend(["", "Phase E fixed-policy fields are retained in `top20_explore.csv` and `finalists_confirm.csv`; "
                       "a point-forecast improvement is not an alert improvement unless the paired recall and coverage fields support it.", ""])
        if combined:
            report.extend(["F11 combined 17-week fixed-policy episode evaluation (EXPLORE + CONFIRM, no new policy tuning):", "",
                           "```json", json.dumps(combined, ensure_ascii=False, indent=2, default=str), "```", ""])
        else:
            report.extend(["F11 combined 17-week evaluation: unavailable in sealed completion; no combined-episode claim.", ""])
        late_path = output / "f10/EXPLORE/late_july.csv"
        if late_path.is_file():
            late = pd.read_csv(late_path).set_index("group")
            if {"preceding_two_weeks", "late_july_two_weeks"} <= set(late.index):
                prior, collapse = late.loc["preceding_two_weeks"], late.loc["late_july_two_weeks"]
                report.extend(["Late-July descriptive comparison: preceding two-week MAE "
                               f"{_fmt(prior.mae)} vs late-July two-week MAE {_fmt(collapse.mae)}; "
                               f"actual mean {_fmt(prior.actual_mean)} vs {_fmt(collapse.actual_mean)}; "
                               f"prediction-minus-actual bias {_fmt(prior.bias_pred_minus_actual)} "
                               f"vs {_fmt(collapse.bias_pred_minus_actual)}. These values alone cannot identify "
                               "level shift, pattern change, missingness, or augmentation as a cause.", ""])
        report.extend(["## F10 post-selection error slices", "",
                       f"Diagnostic model: {diagnose_key}; selection was locked before CONFIRM. "
                       f"EXPLORE rows: {diagnostic_audits['EXPLORE']['score_n']}; "
                       f"CONFIRM rows: {diagnostic_audits['CONFIRM']['score_n']}. "
                       "Hour, weekday, holiday, actual-peak, fold, D1/D2, and late-July slices are in `walkforward_v2/report/f10/`. "
                       "Production-volume bins are unavailable unless present on the forecast frame; they are post hoc only.", ""])
    else:
        report.extend(["No frozen candidate was available for F10 model diagnostics.", ""])
    report.extend(["## Ablations, variability, and figures", "",
                   "Information-budget ablation by fixed model family (positive delta is lower MAE for the union):", "",
                   _md(["family", "core AUC-MAE", "union AUC-MAE", "core − union", "interpretation"],
                       [[row.family, _fmt(row.core_AUC_MAE), _fmt(row.union_AUC_MAE),
                         _fmt(row.MAE_improvement_core_minus_union), row.interpretation]
                        for row in ablation_table.itertuples()]), "",
                   "This is an observed comparison on EXPLORE, not a causal estimate. Context-length results are "
                   "in [the context table](walkforward_v2/report/context_length.csv); model/training differences may also matter.", "",
                   "The information-budget feature groups and context comparisons appear in "
                   "[feature ablation](figures/walkforward_v2/feature_ablation.png) and "
                   "[context length](figures/walkforward_v2/context_length.png), with only completed observed settings plotted. "
                   "The horizon, signed peak error, and representative-week figures use locked forecasts. " +
                   ("Weekly B5 differences are descriptive; see [weekly B5 delta](figures/walkforward_v2/weekly_b5_delta.png). "
                    if diagnose_key and diagnose_key != BASELINES["B5"] else
                    "Weekly B5 delta figure is unavailable because no locked nonbaseline diagnostic candidate exists. ") +
                   "Individual-seed standard deviations are in [the seed table](walkforward_v2/report/top20_seed_sd.csv).", "",
                   "## 5.0 checks and limitations", "",
                   "CONFIRM was opened only after the committed finalist selection lock and one durable reservation. "
                   "R1 uses the frozen Chronos-2 ctx2048 median baseline recipe.", "",
                   *[claim for item in preflight_claims for claim in (item, "")],
                   "Fold overlap and the duplicate-heavy first fold can repeat target values; D2 is a retrospective novelty slice. "
                   "The power unit is unverified. EXPLORE selection can bias apparent gain, the sealed final holdout was unused, "
                   "and no new nonaugmented field deployment or operational tolerance validation was performed.", "",
                   f"Selection lock SHA-256: `{lock['lock_sha256']}`; CONFIRM completion SHA-256: `{done['lock_sha256']}`.", ""])
    path = prepared.root / "outputs/phase_f/PHASE_F_REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    scoped = prepared.out / "PHASE_F_REPORT.md"
    scoped.write_text("\n".join(report).replace("(walkforward_v2/report/", "(report/")
                      .replace("(figures/walkforward_v2/", "(../../figures/walkforward_v2/"),
                      encoding="utf-8")
    return path
