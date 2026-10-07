"""Development-only boundaries and distinct risk/point roles for option C."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evening_oracle_0924 import (
    SCENARIO_LABEL, _common_cohort, _excluded_score_peaks, _gain_table,
    _point_case_status, _target_production,
    _verify_a_parity,
)


def test_oracle_uses_observed_target_production_not_origin_production():
    origins = pd.date_range("2021-01-01 09:00", periods=4, freq="15min")
    targets = origins + pd.Timedelta(hours=1)
    history = pd.DataFrame({"production_target": [100., 200., 300., 400.]}, index=targets)
    context = {"all_origins": origins,
               "targets": pd.DataFrame({"target_time": targets}, index=origins)}
    production = _target_production(history, context)
    assert production.name == "target_production_observed"
    assert production.index.equals(origins)
    assert production.tolist() == [100., 200., 300., 400.]


def test_common_cohort_drops_missing_target_production_from_every_partition():
    origins = pd.date_range("2021-01-01", periods=160, freq="15min")
    context = {"fit": origins[:40], "stop": origins[40:80],
               "cal": origins[80:120], "score": origins[120:]}
    values = pd.Series(np.ones(len(origins)), index=origins)
    values.iloc[[0, 40, 80, 159]] = np.nan
    cohort = _common_cohort(context, values)
    assert [len(cohort[name]) for name in ("fit", "stop", "cal", "score")] == [39, 39, 39, 39]
    assert all(np.isfinite(values.loc[index]).all() for index in cohort.values())


def test_excluded_score_peak_count_is_explicit():
    score = pd.date_range("2021-01-01 10:00", periods=4, freq="15min")
    context = {"score": score, "tau": 10.,
               "targets": pd.DataFrame({"y": [0., 11., 12., 0.]}, index=score)}
    assert _excluded_score_peaks(context, {"score": score[[0, 3]]}) == 2


def test_original_risk_fn_stays_fn_when_oracle_point_alert_hits():
    times = pd.date_range("2021-01-01 18:00", periods=4, freq="15min")
    base = pd.DataFrame({"origin": times-pd.Timedelta(hours=1), "target_time": times,
                         "horizon": 4, "fold": 0, "y": [0., 20., 20., 0.],
                         "tau": 10., "pred": [0., 11., 11., 0.]})
    baseline = base.assign(alert=[False, False, False, False])
    oracle = base.assign(alert=[False, True, True, False])
    cases = pd.DataFrame({"status": ["FN"], "start": [times[1]]})
    compared = _point_case_status(cases, baseline, oracle)
    assert compared.loc[0, "status"] == "FN"
    assert compared.loc[0, "baseline_point_episode_status"] == "FN"
    assert compared.loc[0, "oracle_point_episode_status"] == "TP"
    assert compared.loc[0, "reference_only"] and not compared.loc[0, "selection_used"]
    assert compared.loc[0, "scenario_label"] == SCENARIO_LABEL


def test_paired_oracle_gain_is_positive_on_identical_target_cohort():
    times = pd.date_range("2021-01-01 18:00", periods=20, freq="1D")
    common = pd.DataFrame({"origin": times-pd.Timedelta(hours=1), "target_time": times,
                           "horizon": 4, "fold": 0, "y": 20., "tau": 10.})
    baseline = common.assign(pred=10., alert=True)
    oracle = common.assign(pred=20., alert=True)
    table = _gain_table(baseline, oracle, seed=42)
    assert len(table) == 4
    assert (table.baseline_minus_oracle_mae == 10.).all()
    assert table.improvement_established.all()
    assert set(table.scenario_label) == {SCENARIO_LABEL}


def test_a_parity_evidence_must_match_current_development_artifacts(tmp_path):
    oof = tmp_path / "outputs/predictions/development_oof.csv"
    manifest = tmp_path / "outputs/logs/development_selection.json"
    oof.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    oof.write_text("origin,target_time\n", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    from scripts.evening_oracle_0924 import _digest
    record = {"original_selection_unchanged": True,
              "q95_refit_parity": {"models_checked": 39, "all_alerts_exact": True,
                                   "max_abs_difference": 0.},
              "sources_sha256": {"development_oof": _digest(oof),
                                 "development_selection": _digest(manifest)}}
    (tmp_path / "outputs/logs/peak_sensitivity_0924.json").write_text(json.dumps(record), encoding="utf-8")
    assert _verify_a_parity(tmp_path, oof, manifest)["models_checked"] == 39
    oof.write_text("changed", encoding="utf-8")
    with pytest.raises(AssertionError, match="stale"):
        _verify_a_parity(tmp_path, oof, manifest)
