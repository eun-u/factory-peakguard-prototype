"""Reproduce the preregistered session analyses without reading test targets.

Example: .venv/Scripts/python.exe scripts/run_session_0924.py --task 2
Use --output _validation/session_0924_repro for an independent recalculation.
This entry point never calls report generation or the final evaluation gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def compare_decisions(left, right):
    """Exact structure/labels/counts, preregistered tolerance for CSV floats."""
    left = json.loads(json.dumps(left, default=str))
    right = json.loads(json.dumps(right, default=str))
    mismatches, differences = [], []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatches.append(path + ": keys differ")
                return
            for key in a:
                walk(a[key], b[key], path + "/" + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatches.append(path + ": lengths differ")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + "/" + str(i))
        elif isinstance(a, bool) or isinstance(b, bool):
            if type(a) is not type(b) or a != b:
                mismatches.append(path + ": boolean differs")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if np.isfinite(a) and np.isfinite(b):
                differences.append(abs(a-b))
            equal = a == b if isinstance(a, int) and isinstance(b, int) else np.isclose(a, b, atol=1e-10, rtol=1e-10, equal_nan=True)
            if not equal:
                mismatches.append(path + f": {a} != {b}")
        elif a != b:
            mismatches.append(path + f": {a} != {b}")

    walk(left, right, "selection")
    return {"equal": not mismatches, "max_abs_difference": max(differences, default=0), "mismatches": mismatches}


def same_json(left, right):
    return compare_decisions(left, right)["equal"]


def inputs(root=ROOT):
    from src.session_data import load_development_oof
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    predictions = load_development_oof(root)
    manifest = json.loads((root / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    return cfg, predictions, manifest


def task2(root, output):
    from src.analysis.symmetric_metrics import symmetric_peak_table
    cfg, predictions, _ = inputs(root)
    table = symmetric_peak_table(predictions, cfg)
    table.to_csv(output / "tables/T2-3_symmetric_peak_metrics.csv", index=False)
    result = {"task": 2, "rows": len(table), "horizons": sorted(table.horizon.unique().tolist()),
              "reference_rows": int(table.reference_only.sum()), "selection_used": False,
              "bootstrap_n": cfg["bootstrap"]["n"],
              "episode_ci_method": "Original fold matching; TP/FN anchored to actual start date, FP to alert start date. Dates are resampled without joining episodes.",
              "point_fp_definition": "Existing calibrated alert; auxiliary union/overpredict use pred > tau."}
    save_json(output / "logs/session_0924_task2.json", result)
    return result


def task3(root, output):
    from src.model_reporting import split_fva_table
    from src.training import _selection
    cfg, predictions, manifest = inputs(root)
    old_fva = root / "outputs/tables/T2-1_fva.csv"
    fva = pd.read_csv(old_fva)
    paths = split_fva_table(fva, output)
    comparisons = []
    keys = ["origin", "target_time", "horizon", "fold"]
    first, second = "c3_holiday_hybrid", "c3_holiday_mid_4_6"
    for horizon, group in predictions.groupby("horizon"):
        left = group.loc[group.model.eq(first)].set_index(keys).sort_index()
        right = group.loc[group.model.eq(second)].set_index(keys).sort_index()
        same_keys = left.index.equals(right.index) and len(left) > 0
        same_pred = same_keys and np.array_equal(left.pred.to_numpy(), right.pred.to_numpy(), equal_nan=True)
        same_alert = same_keys and np.array_equal(left.alert.to_numpy(), right.alert.to_numpy())
        comparisons.append({"horizon": int(horizon), "left": first, "right": second,
                            "rows": len(left), "same_keys": same_keys,
                            "same_predictions_and_missing": bool(same_pred), "same_alert": bool(same_alert),
                            "identical": bool(same_keys and same_pred and same_alert)})
    pd.DataFrame(comparisons).to_csv(output / "tables/cbl_alias_audit_0924.csv", index=False)
    identical = all(row["identical"] for row in comparisons)
    before = _selection(predictions, pd.DataFrame(), cfg)
    saved_comparison = compare_decisions(before, manifest["selection"])
    if not saved_comparison["equal"]:
        raise AssertionError(f"Current original selection does not reproduce saved decision: {saved_comparison['mismatches'][:5]}")
    consolidated = predictions.loc[~predictions.model.eq(second)] if identical else predictions
    after = _selection(consolidated, pd.DataFrame(), cfg)
    if not same_json(before, after):
        raise AssertionError("Alias consolidation changed selection")
    # Preserve the original prediction cache and training IDs. Consolidate the
    # new display table only; the original selector already excludes the alias.
    symmetric_path = output / "tables/T2-3_symmetric_peak_metrics.csv"
    if symmetric_path.exists() and identical:
        symmetric = pd.read_csv(symmetric_path)
        symmetric = symmetric.loc[~symmetric.model.eq(second)].copy()
        symmetric["model_alias"] = np.where(symmetric.model.eq(first), second, "")
        symmetric.to_csv(symmetric_path, index=False)
    result = {"task": 3, "cbl_aliases_identical_all_horizons": identical,
              "canonical_id": first if identical else None, "selection_changed": False,
              "saved_selection_reproduced": True, "source_oof_preserved": True,
              "saved_selection_comparison": saved_comparison,
              "legacy_fva_sha256": hashlib.sha256(old_fva.read_bytes()).hexdigest(),
              "replacement_tables": paths,
              "selected": {h: c["point_model"] for h, c in after["by_horizon"].items()}}
    save_json(output / "logs/session_0924_task3.json", result)
    (output / "logs/T2-1_fva_DEPRECATED.md").write_text(
        "# T2-1_fva.csv: DEPRECATED\n\n기존 파일은 이력 보존용이다. 점예측 비교는 "
        "T2-1a_fva_point.csv, 불확실성 비교는 T2-1b_fva_uncertainty.csv를 사용한다.\n"
        "q50 참고 행의 점예측 지표와 분위수 보정의 가치를 혼용하지 않는다.\n"
        "CBL 중복 별칭은 신규 대칭 지표 표에서 하나로 표시하며 원본 OOF와 선정 ID는 보존한다.\n",
        encoding="utf-8")
    return result


def task4(root, output):
    from src.session_data import load_development_history
    from src.analysis.energy_baseline_v2 import run_energy_baseline_v2
    from src.analysis.shift_v2 import prepare_h4_operational_oof, run_shift_v2
    from src.analysis.tariff import read_tariff
    cfg, predictions, manifest = inputs(root)
    history = load_development_history(root)
    operational = prepare_h4_operational_oof(predictions, manifest["selection"])
    energy = run_energy_baseline_v2(history, operational, manifest["folds"],
        n_boot=cfg["bootstrap"]["n"], seed=cfg["seed"], with_classification=False)
    tariff = read_tariff(root / cfg["tariff"]["current"])
    simulation = run_shift_v2(history, operational, manifest["folds"], energy, tariff,
        output, fractions=cfg["shift"]["fractions"], prep_minutes=cfg["alert"]["prep_minutes"])
    result = {"task": 4, "selection_changed": False,
              "scenario_label": "사후 관측 생산량 기반 사후 가정 시나리오",
              "summary": simulation.summary.to_dict("records"),
              "default_monthly": simulation.monthly.loc[simulation.monthly.fraction.eq(.2) &
                     simulation.monthly.prep_minutes.eq(30)].to_dict("records"),
              "coefficient_source": energy.fold_metrics.to_dict("records"),
              "paths": simulation.paths}
    save_json(output / "logs/session_0924_task4.json", result)
    return {"task": 4, "scenario_rows": len(simulation.summary),
            "default": simulation.summary.loc[simulation.summary.fraction.eq(.2) &
                      simulation.summary.prep_minutes.eq(30)].to_dict("records")}


def task5(root, output):
    from src.session_data import load_development_history
    from src.analysis.energy_baseline_v2 import run_energy_baseline_v2
    from src.analysis.shift_v2 import prepare_h4_operational_oof
    cfg, predictions, manifest = inputs(root)
    history = load_development_history(root)
    operational = prepare_h4_operational_oof(predictions, manifest["selection"])
    result = run_energy_baseline_v2(history, operational, manifest["folds"], output,
        n_boot=cfg["bootstrap"]["n"], seed=cfg["seed"], with_classification=True)
    save_json(output / "logs/session_0924_task5.json", {"task": 5, "selection_changed": False,
        **result.summary, "fold_metrics": result.fold_metrics.to_dict("records"),
        "coefficient_ci": result.coefficient_ci.to_dict("records"),
        "type_summary": result.type_summary.to_dict("records"), "paths": result.paths})
    return {"task": 5, **result.summary}


def task_b(root, output):
    from src.session_data import load_development_history
    from src.analysis.operational_sensitivity_0924 import management_target_sensitivity
    cfg, predictions, manifest = inputs(root)
    table = management_target_sensitivity(load_development_history(root), predictions, manifest, cfg)
    table.to_csv(output / "tables/operational_target_sensitivity_0924.csv", index=False)
    result = {"task": "B", "selection_changed": False, "rows": len(table),
              "target_quantile": .975, "event_quantile": .95,
              "default_rule": table.loc[table.rule.eq("1/1")].to_dict("records")}
    save_json(output / "logs/session_0924_taskB.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["2", "3", "4", "5", "B"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to((ROOT / name).resolve()) for name in ("outputs", "_validation")):
        raise ValueError("Analysis output is restricted to outputs/ or _validation/")
    for name in ("tables", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"2": task2, "3": task3, "4": task4, "5": task5, "B": task_b}[args.task](ROOT, output)
    print(json.dumps({**result, "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
