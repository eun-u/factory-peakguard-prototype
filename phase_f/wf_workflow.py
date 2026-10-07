"""Resumable Stage 0-3 search for the revised weekly Phase F contract.

Every configuration wave is locked before its first fit. The 128 unique
original model configurations are replayed on the new week folds before any
adaptive search. Only EXPLORE results may create later configurations or stop
a search direction. CONFIRM and the final holdout are outside this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from phase_f.registry import config_hash, now, sha256, write_json
from phase_f.wf_registry import WFRegistry


BASELINE_IDS = ("F0-1-B5", "F0-1-M1", "F0-1-R1")
TERMINAL = frozenset({"completed", "unsupported", "rejected", "failed"})
_DIRECTIONS = {
    "AUC_MAE": ("wf_explore_AUC_MAE", "wf_explore_AUC_MAE_seed_sd", False),
    "AUC_PeakMAE": ("wf_explore_AUC_PeakMAE", "wf_explore_AUC_PeakMAE_seed_sd", False),
    "h16_PeakMAE": ("wf_explore_h16_Peak", "wf_explore_h16_Peak_seed_sd", False),
    "E_c10_22_recall": ("E_c10_22_recall", "E_c10_22_recall_seed_sd", True),
}


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _status(prepared: Any, stage: int, **fields: Any) -> dict:
    path = prepared.out / "logs/workflow" / f"stage_{stage}.json"
    record = _read(path) if path.exists() else {"stage": stage, "started_at": now()}
    record.update(fields, updated_at=now(), arm="EXPLORE", holdout_read=False,
                  historical_final_artifact_read=False)
    write_json(path, record)
    return record


def _source_original_specs(prepared: Any) -> tuple[list[dict], dict]:
    """Read immutable original settings, excluding only diagnostics/duplicates."""
    root = Path(prepared.root)
    source = root / "outputs/phase_f/logs/experiments"
    if not source.is_dir():
        raise FileNotFoundError("Original Phase F experiment records are missing")
    records = [_read(path) for path in sorted(source.glob("*.json"))]
    specs, hashes, excluded = [], {}, []
    for row in records:
        cfg = json.loads(row["config_json"])
        exp_id = row["exp_id"]
        if exp_id != cfg.get("id") or config_hash(cfg) != row["config_hash"]:
            raise ValueError(f"Original configuration changed: {exp_id}")
        if cfg.get("adapter") == "diagnostic" or row["status"] == "diagnostic_passed":
            excluded.append({"id": exp_id, "reason": "diagnostic"})
            continue
        if row.get("duplicate_of"):
            excluded.append({"id": exp_id, "reason": "duplicate_of",
                             "canonical_id": row["duplicate_of"]})
            continue
        specs.append(cfg)
        hashes[exp_id] = row["config_hash"]
    specs.sort(key=lambda spec: (int(spec.get("tier", 0)), spec["id"]))
    if not all(any(item["id"] == name for item in specs) for name in BASELINE_IDS):
        raise RuntimeError("B5/M1/R1 original baseline settings are incomplete")
    if len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Duplicate original experiment IDs")
    audit = {"original_record_count": len(records), "unique_model_configs": len(specs),
             "excluded": excluded, "config_sha256": hashes,
             "source_record_sha256": {path.name: sha256(path) for path in sorted(source.glob("*.json"))},
             "previous_failures": [row["exp_id"] for row in records if row["status"] == "failed"
                                   and row["exp_id"] in hashes],
             "previous_planned": [row["exp_id"] for row in records if row["status"] == "planned"
                                  and row["exp_id"] in hashes]}
    return specs, audit


def _original_spec_waves(prepared: Any) -> tuple[list[dict], list[dict], dict]:
    specs, audit = _source_original_specs(prepared)
    baselines = [next(item for item in specs if item["id"] == name) for name in BASELINE_IDS]
    replay = [item for item in specs if item["id"] not in BASELINE_IDS]
    if len(baselines) + len(replay) != audit["unique_model_configs"]:
        raise AssertionError("Original replay omitted a unique configuration")
    return baselines, replay, audit


def _locked_wave(prepared: Any, name: str, factory: Callable[[], list[dict]],
                 hypothesis: str, *, source: dict | None = None,
                 allow_empty: bool = False) -> list[dict]:
    from phase_f.wf_run import normalize_spec

    path = prepared.out / "logs/waves" / f"{name}.json"
    if path.exists():
        record = _read(path)
        if (record.get("name") != name or record.get("selection_arm") != "EXPLORE" or
                record.get("config_sha256") != config_hash(record.get("specs")) or
                record.get("holdout_read") is not False):
            raise ValueError(f"Changed weekly wave lock: {name}")
        if source is not None and record.get("original_source") != source:
            raise ValueError(f"Original source evidence changed for wave {name}")
        return record["specs"]
    raw = factory()
    if not isinstance(raw, list) or (not raw and not allow_empty):
        raise RuntimeError(f"No settings produced for required wave {name}")
    skipped = []
    specs = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise ValueError(f"Malformed weekly setting in {name}")
        if entry["id"].endswith("-native_mean"):
            skipped.append({"id": entry["id"], "reason": "same output as median"})
            continue
        specs.append(normalize_spec(prepared, dict(entry)))
    ids = [entry["id"] for entry in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Normalized weekly wave has colliding IDs: {name}")
    record = {"name": name, "hypothesis": hypothesis, "specs": specs,
              "config_sha256": config_hash(specs), "selection_arm": "EXPLORE",
              "original_source": source, "skipped_aliases": skipped,
              "locked_at": now(), "holdout_read": False,
              "historical_final_artifact_read": False}
    write_json(path, record, exclusive=True)
    return specs


def _top20(prepared: Any) -> tuple[str, ...]:
    from phase_f.wf_plan import completed

    rows = [row for row in completed(prepared.root)
            if row.get("duplicate_of") is None and row.get("exp_id") not in BASELINE_IDS]
    return tuple(row["exp_id"] for row in rows[:20])


def _refresh_when_rank_changes(prepared: Any, before: tuple[str, ...],
                               after: tuple[str, ...]) -> None:
    if before == after:
        return
    from phase_f.wf_evaluation import refresh_downstream

    refresh_downstream(prepared, arm="EXPLORE")


def _execute_wave(prepared: Any, name: str, factory: Callable[[], list[dict]],
                  hypothesis: str, *, retry: bool = False, source: dict | None = None,
                  retry_original_failures: set[str] | None = None,
                  allow_empty: bool = False) -> dict:
    from phase_f.wf_run import execute

    registry = WFRegistry(prepared.root)
    specs = _locked_wave(prepared, name, factory, hypothesis, source=source,
                         allow_empty=allow_empty)
    for spec in specs:
        registry.register(spec)
    checkpoint = prepared.out / "logs/workflow" / f"{name}.json"
    state = _read(checkpoint) if checkpoint.exists() else {
        "name": name, "spec_ids": [spec["id"] for spec in specs],
        "attempts": {}, "status": "running", "started_at": now()}
    if state.get("spec_ids") != [spec["id"] for spec in specs]:
        raise ValueError(f"Wave checkpoint differs from lock: {name}")
    if state.get("status") == "completed":
        if any((registry.read(spec["id"]) or {}).get("status") not in TERMINAL for spec in specs):
            raise ValueError(f"Completed wave has unresolved experiment: {name}")
        return state
    failures = 0
    for spec in specs:
        before = _top20(prepared)
        prior_attempts = state.setdefault("attempts", {}).get(spec["id"], 0)
        row = execute(prepared, spec, registry=registry,
                      retry=retry or bool(prior_attempts and retry_original_failures and
                                          spec["id"] in retry_original_failures and prior_attempts < 2),
                      score=True)
        attempts = prior_attempts + 1
        state["attempts"][spec["id"]] = attempts
        # The original failed settings get one additional new-protocol attempt.
        # Persist the first failure before retry so interruption is recoverable.
        if (row["status"] == "failed" and retry_original_failures and
                spec["id"] in retry_original_failures and attempts < 2):
            write_json(checkpoint, state)
            row = execute(prepared, spec, registry=registry, retry=True, score=True)
            state["attempts"][spec["id"]] = 2
        if row["status"] not in TERMINAL:
            raise RuntimeError(f"Unresolved weekly experiment {spec['id']}: {row['status']}")
        after = _top20(prepared)
        if row["status"] == "completed":
            _refresh_when_rank_changes(prepared, before, after)
        failures = failures + 1 if row["status"] in {"failed", "rejected"} else 0
        state.update(last_id=spec["id"], resolved=sum(
            (registry.read(item["id"]) or {}).get("status") in TERMINAL for item in specs),
            total=len(specs), status="running", updated_at=now(),
            holdout_read=False, historical_final_artifact_read=False)
        write_json(checkpoint, state)
        if failures >= 3:
            raise RuntimeError(f"Three consecutive weekly experiment failures in {name}; repair and resume")
    state.update(status="completed", completed_at=now())
    write_json(checkpoint, state)
    return state


def _stage0(prepared: Any, retry: bool) -> None:
    from phase_f.wf_run import execute
    from phase_f.wf_evaluation import refresh_downstream

    baselines, _, audit = _original_spec_waves(prepared)
    specs = _locked_wave(prepared, "stage0_baseline_three", lambda: baselines,
                         "Fit fixed B5, M1 and R1 on every weekly fold before scoring",
                         source={"config_sha256": {key: audit["config_sha256"][key]
                                                   for key in BASELINE_IDS}})
    registry = WFRegistry(prepared.root)
    for spec in specs:
        registry.register(spec)
    for spec in specs:
        row = execute(prepared, spec, registry=registry, retry=retry, score=False)
        if row["status"] not in {"prediction_ready", "completed"}:
            raise RuntimeError(f"Weekly baseline prediction incomplete: {spec['id']}")
    for spec in specs:
        row = execute(prepared, spec, registry=registry, retry=retry, score=True)
        if row["status"] != "completed":
            raise RuntimeError(f"Weekly baseline score incomplete: {spec['id']}")
    refresh_downstream(prepared, arm="EXPLORE", ids=list(BASELINE_IDS))
    write_json(prepared.out / "logs/workflow/original_replay_inventory.json", {
        **audit, "stage0_baselines": list(BASELINE_IDS),
        "stage1_original_replay_count": audit["unique_model_configs"]-len(BASELINE_IDS),
        "selection_arm": "EXPLORE", "holdout_read": False})


def _stage1(prepared: Any, retry: bool) -> None:
    from phase_f import wf_plan as plan
    from phase_f.run import catalog

    _, replay, audit = _original_spec_waves(prepared)
    source = {"config_sha256": {item["id"]: audit["config_sha256"][item["id"]]
                                for item in replay},
              "previous_failures": audit["previous_failures"],
              "previous_planned": audit["previous_planned"]}
    _execute_wave(prepared, "stage1_replay_original", lambda: replay,
                  "Re-evaluate all original nonduplicate model settings on weekly folds",
                  retry=retry, source=source,
                  retry_original_failures=set(audit["previous_failures"]))
    _execute_wave(prepared, "stage1_catalog", lambda: catalog(1),
                  "Complete original tier-one features and deterministic references", retry=retry)
    _execute_wave(prepared, "stage1_feature_union", lambda: plan.feature_union(prepared.root),
                  "Union only independently helpful EXPLORE feature groups", retry=retry)
    _execute_wave(prepared, "stage1_feature_finish", lambda: plan.feature_finish(prepared.root),
                  "One production check and fit-only feature reduction", retry=retry)
    target_plan = _locked_wave(prepared, "stage1_target_plan", lambda: plan.targets(prepared.root),
                               "Fix a common tabular parent for all target transformations")
    _execute_wave(prepared, "stage1_targets", lambda: [item for item in target_plan
                  if item.get("tier") == 1], "Weekly and B5 residual target tests", retry=retry)


def _score_tuned_winner(prepared: Any, state: dict, kind: str, *, retry: bool) -> dict:
    from phase_f.wf_run import execute

    registry = WFRegistry(prepared.root)
    winner = state.get("best_trial_id") or state.get("best_exp_id")
    if not winner:
        raise RuntimeError(f"{kind} search has no fixed best stop-trial ID")
    row = registry.read(winner)
    if row is None or row.get("status") not in {"prediction_ready", "completed"}:
        raise RuntimeError(f"{kind} best stop trial is not complete: {winner}")
    spec = json.loads(row["config_json"])
    before = _top20(prepared)
    scored = execute(prepared, spec, registry=registry, retry=retry, score=True)
    if scored["status"] != "completed":
        raise RuntimeError(f"{kind} tuned winner did not complete EXPLORE score")
    _refresh_when_rank_changes(prepared, before, _top20(prepared))
    return scored


def _tune_gbdt(prepared: Any, kind: str, target: int, *, label: str, retry: bool) -> dict:
    from phase_f import wf_plan as plan
    from phase_f.wf_run import execute, normalize_spec
    from phase_f.wf_tuning import run_search

    state_path = prepared.out / "logs/tuning" / f"{label}.json"
    # Study identity includes the original parent. Expanding a study must
    # reuse that locked parent even when its winner changes the ranking.
    parent = _read(state_path)["parent_spec"] if state_path.exists() else plan.best(
        prepared.root, kind=kind)
    registry = WFRegistry(prepared.root)

    def trial(child: dict, score: bool = False) -> dict:
        if score:
            raise ValueError("TPE objective must never inspect EXPLORE score")
        item = normalize_spec(prepared, child)
        return execute(prepared, item, registry=registry, retry=retry,
                       score=False)

    state = run_search(prepared, kind, trial, parent_spec=parent,
                       target_completed=target, label=label)
    if state.get("completed_trials", state.get("complete_trials", -1)) < target:
        raise RuntimeError(f"{kind} TPE stopped before {target} complete trials")
    _score_tuned_winner(prepared, state, kind, retry=retry)
    return state


def _stage2(prepared: Any, retry: bool) -> None:
    from phase_f import wf_plan as plan
    from phase_f.run import catalog

    _execute_wave(prepared, "stage2_tabular", lambda: plan.tabular_variants(prepared.root),
                  "GBDT losses, peak weights, multi-horizon and Kalman inputs", retry=retry)
    for kind in ("lightgbm", "xgboost", "catboost"):
        _tune_gbdt(prepared, kind, 500, label=f"stage2_{kind}", retry=retry)
    _execute_wave(prepared, "stage2_cross", lambda: plan.tabular_cross(prepared.root),
                  "Declared feature by loss by weight cross", retry=retry)
    _execute_wave(prepared, "stage2_statistics", lambda: catalog(2),
                  "Weekly refit state-space comparators", retry=retry)
    _execute_wave(prepared, "stage2_neural", lambda: plan.neural_grid(prepared.root, 2),
                  "Every tier-two neural architecture and context", retry=retry)
    _execute_wave(prepared, "stage2_foundation", lambda: plan.foundation_fine(2),
                  "Full and LoRA Chronos training-rate, context and step search", retry=retry)
    data_plan = _locked_wave(prepared, "stage2_data_plan", lambda: plan.data_strategies(prepared.root),
                             "Fix one tabular parent for data-window and peak interventions")
    _execute_wave(prepared, "stage2_data", lambda: [item for item in data_plan
                  if item.get("tier") == 2], "Complete-profile duplicate sensitivity", retry=retry)
    _execute_wave(prepared, "stage2_ensemble", lambda: plan.ensemble_specs(prepared.root, "stage2"),
                  "Declared cross-family pairs and four core F7-6 combinations", retry=retry)


def _stage3(prepared: Any, retry: bool) -> None:
    from phase_f import wf_plan as plan
    from phase_f.run import catalog
    from phase_f.models.other_foundation import configurations as other_foundation

    target_plan = _locked_wave(prepared, "stage1_target_plan", lambda: plan.targets(prepared.root),
                               "Fix a common tabular parent for all target transformations")
    _execute_wave(prepared, "stage3_targets", lambda: [item for item in target_plan
                  if item.get("tier") == 3], "Remaining target transforms and day-type models", retry=retry)
    _execute_wave(prepared, "stage3_statistics", lambda: catalog(3),
                  "ETS, TBATS, ARIMA and Theta alternatives", retry=retry)
    _execute_wave(prepared, "stage3_neural", lambda: plan.neural_grid(prepared.root, 3),
                  "Remaining neural contexts, including reference LSTM/GRU", retry=retry)
    _execute_wave(prepared, "stage3_neural_extensions", lambda: plan.neural_extensions(prepared.root),
                  "Per-family loss and causal input ablations", retry=retry)
    _tune_neural_families(prepared, retry=retry)
    _execute_wave(prepared, "stage3_finetune_extensions", lambda: plan.foundation_fine(3),
                  "Fine-tuned foundation calendar and point ablations", retry=retry)
    _execute_wave(prepared, "stage3_other_foundation", other_foundation,
                  "Pinned additional foundation families", retry=retry)
    data_plan = _locked_wave(prepared, "stage2_data_plan", lambda: plan.data_strategies(prepared.root),
                             "Fix one tabular parent for data-window and peak interventions")
    _execute_wave(prepared, "stage3_data", lambda: [item for item in data_plan
                  if item.get("tier") == 3], "Sliding, weighted and two-stage models", retry=retry)
    _execute_wave(prepared, "stage3_peak_postprocessing", lambda: plan.peak_postprocessors(prepared.root),
                  "Cal-only bias and quantile risk shifts", retry=retry)
    _execute_wave(prepared, "stage3_ensemble_refresh", lambda: plan.ensemble_specs(prepared.root, "stage3"),
                  "Cross-family ensembles after all families", retry=retry)
    _expansion(prepared, retry=retry)


def _tune_neural_families(prepared: Any, *, retry: bool,
                          target: int = 20) -> None:
    """Require a genuine WF stop-only search per neural family, if provided."""
    from phase_f import wf_plan as plan
    from phase_f.wf_run import execute, normalize_spec
    from phase_f.wf_tuning import NEURAL_KINDS, run_neural_search

    rows = plan.completed(prepared.root, adapter="neural")
    kinds = sorted({plan.spec(row)["kind"] for row in rows
                    if plan.spec(row)["kind"] not in {"lstm", "gru"}})
    missing = sorted(set(NEURAL_KINDS) - set(kinds))
    if missing:
        raise RuntimeError(f"No completed weekly parent for required neural searches: {missing}")
    registry = WFRegistry(prepared.root)
    for kind in kinds:
        label = f"stage3_{kind}"
        state_path = prepared.out / "logs/tuning" / f"{label}.json"
        parent = _read(state_path)["parent_spec"] if state_path.exists() else plan.best(
            prepared.root, adapter="neural", kind=kind)

        def trial(child: dict, score: bool = False) -> dict:
            if score:
                raise ValueError("Neural search objective must use stop labels only")
            item = normalize_spec(prepared, child)
            return execute(prepared, item, registry=registry, retry=retry, score=False)

        state = run_neural_search(prepared, kind, trial, parent_spec=parent,
                                  target_completed=target, label=label)
        if state.get("completed_trials", state.get("complete_trials", -1)) < target:
            raise RuntimeError(f"Neural {kind} search did not complete {target} stop-only trials")
        _score_tuned_winner(prepared, state, kind, retry=retry)


def _direction_snapshot(prepared: Any) -> dict:
    from phase_f.wf_plan import completed

    rows = completed(prepared.root)
    result = {}
    for name, (field, sdfield, higher) in _DIRECTIONS.items():
        candidates = []
        for row in rows:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value) or row.get("leakage_test") != "passed":
                continue
            if name.startswith("E_") and row.get("E_status") != "completed":
                continue
            candidates.append(row)
        if candidates:
            best = min(candidates, key=lambda row: (-float(row[field]) if higher else float(row[field]),
                                                    row["exp_id"]))
            result[name] = {"exp_id": best["exp_id"], "value": float(best[field]),
                            "seed_sd": best.get(sdfield), "n_seeds": best.get("n_seeds"),
                            "E_manifest_path": best.get("E_manifest_path")}
        else:
            result[name] = None
    return result


def direction_gain(before: dict | None, after: dict | None, *, ci_low: float | None,
                   ci_high: float | None, ci_status: str,
                   higher_is_better: bool = False) -> dict:
    """Apply the fixed 2%, paired CI and seed-SD continuation gate."""
    record = {"before": before, "after": after, "ci_low": ci_low, "ci_high": ci_high,
              "ci_status": ci_status, "higher_is_better": higher_is_better,
              "relative_threshold": .02, "significant_improvement": False}
    if before is None or after is None or before["exp_id"] == after["exp_id"]:
        return {**record, "reason": "no_new_direction_best"}
    try:
        old = float(before["value"])
        new = float(after["value"])
        low = float(ci_low)
        high = float(ci_high)
    except (TypeError, ValueError, KeyError):
        return {**record, "reason": "unavailable_paired_CI_or_metric"}
    if ci_status not in {"available", "ok"} or not np.isfinite([old, new, low, high]).all():
        return {**record, "reason": "unavailable_paired_CI_or_metric"}
    sd_values = []
    for item in (before, after):
        raw = item.get("seed_sd")
        if raw is None and item.get("n_seeds") == 1:
            raw = 0.
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return {**record, "reason": "missing_seed_standard_deviation"}
        if not math.isfinite(value) or value < 0:
            return {**record, "reason": "missing_seed_standard_deviation"}
        sd_values.append(value)
    gain = new-old if higher_is_better else old-new
    relative = gain/max(abs(old), 1e-12)
    uncertainty = math.hypot(*sd_values)
    significant = gain > 0 and relative >= .02 and low > 0 and gain > uncertainty
    return {**record, "gain": gain, "relative_gain": relative,
            "combined_seed_sd": uncertainty, "significant_improvement": bool(significant),
            "reason": "meets_all_criteria" if significant else "below_2pct_CI_or_seed_SD"}


def _verified_e_events(prepared: Any, item: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = WFRegistry(prepared.root)
    row = registry.read(item["exp_id"])
    if row is None or row.get("E_status") != "completed" or not row.get("E_manifest_path"):
        raise ValueError(f"Missing completed E evidence for {item['exp_id']}")
    destination = (prepared.out / row["E_manifest_path"]).resolve()
    if not destination.is_relative_to(prepared.out.resolve()):
        raise ValueError("Phase E evidence path escaped weekly output")
    manifest_path = destination / "manifest.json"
    if sha256(manifest_path) != row.get("E_manifest_sha256"):
        raise ValueError("Phase E manifest digest changed")
    manifest = _read(manifest_path)
    if manifest.get("candidate") != item["exp_id"] or manifest.get("arm") != "EXPLORE":
        raise ValueError("Phase E comparison identity changed")
    for name in ("alert_episode_events.csv", "score_probabilities.csv"):
        if sha256(destination / name) != manifest.get("artifacts", {}).get(name):
            raise ValueError(f"Phase E event artifact changed: {name}")
    events = pd.read_csv(destination / "alert_episode_events.csv")
    events = events.loc[events.model.eq(item["exp_id"])].copy()
    score = pd.read_csv(destination / "score_probabilities.csv")
    days = score.loc[score.model.eq(item["exp_id"]), ["fold", "target_time"]].copy()
    return events, days


def _direction_evidence(prepared: Any, direction: str,
                        before: dict | None, after: dict | None) -> dict:
    if before is None or after is None or before["exp_id"] == after["exp_id"]:
        return direction_gain(before, after, ci_low=None, ci_high=None,
                              ci_status="unavailable", higher_is_better=direction.startswith("E_"))
    if direction.startswith("E_"):
        from phase_f.wf_downstream import paired_episode_ci

        new_events, new_days = _verified_e_events(prepared, after)
        old_events, old_days = _verified_e_events(prepared, before)
        left = new_days.sort_values(["fold", "target_time"]).reset_index(drop=True)
        right = old_days.sort_values(["fold", "target_time"]).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("Two Phase E candidates do not share the same score calendar")
        paired = paired_episode_ci(pd.concat([new_events, old_events], ignore_index=True),
                                   after["exp_id"], before["exp_id"], n=1000, seed=42,
                                   represented_days=left)
        low, high = paired["recall_delta_ci_low"], paired["recall_delta_ci_high"]
        status = paired["recall_ci_status"]
        if low is not None and high is not None and low == high:
            status = "unavailable_zero_width_CI"
        return {**direction_gain(before, after, ci_low=low, ci_high=high,
                                 ci_status=status, higher_is_better=True),
                "paired_episode": paired}

    from phase_f import wf_metrics
    from phase_f.wf_evaluation import load_predictions

    new = load_predictions(prepared, after["exp_id"], "EXPLORE")
    old = load_predictions(prepared, before["exp_id"], "EXPLORE")
    rows = wf_metrics.paired_ci(new, old, "EXPLORE", n=1000, seed=42)
    if direction == "AUC_MAE":
        selected = rows.loc[rows.dataset.eq("D1") & rows.metric.eq("AUC_MAE_improvement") &
                            rows.horizon.isna()]
    elif direction == "AUC_PeakMAE":
        selected = rows.loc[rows.dataset.eq("D1") & rows.metric.eq("AUC_PeakMAE_degradation") &
                            rows.horizon.isna()]
    elif direction == "h16_PeakMAE":
        selected = rows.loc[rows.dataset.eq("D1") & rows.metric.eq("Peak_MAE_degradation") &
                            rows.horizon.eq(16)]
    else:
        raise ValueError(f"Unknown continuation direction: {direction}")
    if len(selected) != 1:
        raise ValueError(f"Missing paired CI for {direction}")
    row = selected.iloc[0]
    if direction == "AUC_MAE":
        low, high = float(row.ci_low), float(row.ci_high)
    else:
        # wf_metrics names candidate-minus-baseline peak error as degradation.
        # A reduction is the sign-reversed interval with exchanged endpoints.
        low, high = -float(row.ci_high), -float(row.ci_low)
    return {**direction_gain(before, after, ci_low=low, ci_high=high,
                             ci_status=str(row.ci_status)),
            "paired_metric": str(row.metric), "paired_estimate": float(row.estimate),
            "paired_ci_reason": str(row.ci_reason), "bootstrap_n": int(row.bootstrap_n)}


def _expansion_specs(prepared: Any, number: int) -> list[dict]:
    from phase_f import wf_plan as plan

    candidates = plan.completed(prepared.root, adapter="neural")
    result = []
    if candidates:
        parent = plan.spec(candidates[0])
        result.append(plan.child(parent, f"F5-expansion-{number}-{parent['kind']}",
                                 tier=3, loss="huber" if number % 2 == 0 else "peak_weighted_mae"))
    result.extend(plan.foundation_training_expansion(prepared.root, number))
    if not result:
        raise RuntimeError("No completed neural/foundation parent for required expansion")
    return result


def _expansion(prepared: Any, *, retry: bool) -> list[dict]:
    from phase_f.wf_contract import two_rounds_converged

    path = prepared.out / "logs/workflow/expansion_rounds.json"
    state = _read(path) if path.exists() else {
        "rounds": [], "rule": "Two consecutive waves without a >=2% CI- and seed-SD-backed improvement",
        "directions": list(_DIRECTIONS), "arm": "EXPLORE",
        "holdout_read": False}
    while not two_rounds_converged(state["rounds"]):
        number = len(state["rounds"])
        round_path = prepared.out / "logs/workflow" / f"expansion_round_{number}.json"
        if round_path.exists():
            round_lock = _read(round_path)
            if round_lock.get("round") != number or round_lock.get("arm") != "EXPLORE":
                raise ValueError("Changed expansion before snapshot")
            before = round_lock["before"]
        else:
            before = _direction_snapshot(prepared)
            write_json(round_path, {"round": number, "before": before,
                                    "arm": "EXPLORE", "locked_at": now(),
                                    "holdout_read": False}, exclusive=True)
        _execute_wave(prepared, f"expansion_{number}_ablations",
                      lambda number=number: _expansion_specs(prepared, number),
                      "Finite neural/foundation extension after all declared families",
                      retry=retry)
        for kind in ("lightgbm", "xgboost", "catboost"):
            _tune_gbdt(prepared, kind, 600+100*number,
                       label=f"stage2_{kind}", retry=retry)
        _tune_neural_families(prepared, retry=retry, target=120+100*number)
        after = _direction_snapshot(prepared)
        evidence = {name: _direction_evidence(prepared, name, before[name], after[name])
                    for name in _DIRECTIONS}
        record = {"round": number, "before": before, "after": after,
                  "direction_evidence": evidence,
                  "significant_improvement": any(item["significant_improvement"]
                                                 for item in evidence.values()),
                  "completed_at": now(), "arm": "EXPLORE", "holdout_read": False}
        state["rounds"].append(record)
        write_json(path, state)
    return state["rounds"]


def _summary(prepared: Any, stage: int) -> Path:
    registry = WFRegistry(prepared.root)
    records = [_read(path) for path in sorted(registry.records.glob("*.json"))]
    completed = [row for row in records if row.get("status") == "completed"
                 and row.get("wf_explore_AUC_MAE") is not None]
    completed.sort(key=lambda row: (float(row["wf_explore_AUC_MAE"]), row["exp_id"]))
    columns = ("exp_id", "wf_explore_AUC_MAE", "wf_explore_AUC_PeakMAE",
               "wf_explore_h16_Peak", "E_c10_22_recall")
    lines = [f"# Phase F weekly Stage {stage}", "",
             "EXPLORE development weeks only. CONFIRM and final holdout have not been evaluated here.", "",
             "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"]*len(columns)) + " |"]
    for row in completed[:10]:
        values = []
        for col in columns:
            value = row.get(col)
            values.append(f"{value:.6f}" if isinstance(value, (float, int)) and math.isfinite(value)
                          else str(value) if value is not None else "NA")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", f"Weekly registry: {len(records)} records, {len(completed)} scored complete, "
                  f"{sum(row.get('status') == 'failed' for row in records)} failed.", "",
                  "Stage 4 selection and one-time CONFIRM are handled by the separate locked evaluator.", ""])
    path = prepared.out / f"STAGE_{stage}_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    # Preserve the original protocol's top-level STAGE_n summaries bytewise.
    top = prepared.root / "outputs/phase_f" / f"STAGE_{stage}_walkforward_v2_summary.md"
    top.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_stage(prepared: Any, stage: int, *, retry: bool = False) -> dict:
    if stage not in range(4):
        raise ValueError("Weekly search stage must be 0..3")
    path = prepared.out / "logs/workflow" / f"stage_{stage}.json"
    if path.exists():
        record = _read(path)
        if record.get("status") == "completed":
            summary = prepared.out / f"STAGE_{stage}_summary.md"
            if not summary.exists() or sha256(summary) != record.get("summary_sha256"):
                raise ValueError(f"Completed weekly stage {stage} summary changed")
            return record
    for previous in range(stage):
        prior = prepared.out / "logs/workflow" / f"stage_{previous}.json"
        if not prior.exists() or _read(prior).get("status") != "completed":
            raise RuntimeError(f"Weekly Stage {previous} must complete before Stage {stage}")
    _status(prepared, stage, status="running")
    try:
        ( _stage0, _stage1, _stage2, _stage3 )[stage](prepared, retry)
        summary = _summary(prepared, stage)
        record = _status(prepared, stage, status="completed", completed_at=now(),
                         summary_sha256=sha256(summary))
        from phase_f.workflow import _git_commit
        _git_commit(prepared, f"Phase F weekly Stage {stage}: EXPLORE evidence")
        return record
    except Exception as exc:
        _status(prepared, stage, status="failed", error_type=type(exc).__name__,
                error=str(exc).replace(str(prepared.root), "<repo>"))
        raise


def run_all(prepared: Any, *, retry: bool = False) -> list[dict]:
    """Run or resume Stage 0-3, then hand the untouched CONFIRM gate to root."""
    return [run_stage(prepared, stage, retry=retry) for stage in range(4)]
