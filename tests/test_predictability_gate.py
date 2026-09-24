import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("p1_gate_script", Path(__file__).resolve().parents[1] / "scripts/finalize_predictability_p1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _inputs(tmp_path, a1_pass):
    out = tmp_path / "outputs/analysis_p1"
    out.mkdir(parents=True)
    (tmp_path / "outputs/logs").mkdir()
    rows = []
    def row(analysis, name, horizon=4, kind="coefficient", feature="", passed=False):
        rows.append(dict(analysis_id=analysis, hypothesis_id=name, horizon=horizon,
                         test_kind=kind, feature=feature, estimate=.7, ci_low=.65 if passed else .1,
                         ci_high=.8, null_value=.6, p_raw=.0001 if passed else .9,
                         n_positive_folds=3, eligible=True))
    row("A1", "proximity", kind="proximity", passed=a1_pass)
    for h in (16, 96):
        row("A2", f"multi{h}", h, "auc_multi")
        for f in ("slot_rate7", "slot_rate28", "origin_level", "origin_slope",
                  "previous_day_max", "previous_week_max", "previous_restart_rise"):
            row("A2", f"{f}{h}", h, "auc_single", f, passed=True)
    for i in range(4):
        row("A3", f"coef{i}")
    pd.DataFrame(rows).to_csv(out / "A1_A3_hypotheses.csv", index=False)
    rows.clear()
    for h in (16, 96):
        row("A4", f"error{h}", h, "error_concentration")
    pd.DataFrame(rows).to_csv(out / "A4_hypotheses.csv", index=False)


def test_both_primary_fail_override_even_significant_single_signals(tmp_path):
    _inputs(tmp_path, False)
    result = module.finalize(tmp_path)
    assert result["negative_override"]
    assert not result["m2_allowed"]
    assert result["allowed_by_horizon"] == {"16": [], "96": []}


def test_gate_only_licenses_registered_new_features(tmp_path):
    _inputs(tmp_path, True)
    result = module.finalize(tmp_path)
    assert result["m2_allowed"]
    assert set(result["allowed_by_horizon"]["16"]) == {
        "target_restart_slot", "minutes_to_next_restart", "slot_rate7", "slot_rate28", "previous_restart_rise"}
