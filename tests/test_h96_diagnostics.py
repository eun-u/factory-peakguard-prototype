"""Safety checks for the development-only h96 diagnostic entry point."""

import numpy as np
import pandas as pd
import pytest
import joblib
from types import SimpleNamespace

from scripts.diagnose_h96_0924 import (
    _assert_development_frame, _checked_output, _read_oof, _uniform_origins,
    audit_early_stopping,
)
from src.features import build_features


BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")


def test_oof_preflight_rejects_a_frozen_timestamp(tmp_path):
    path = tmp_path / "development_oof.csv"
    path.write_bytes(b"origin,target_time,y\n"
                     b"2021-08-09 09:30:00,2021-08-09 09:45:00,\xff\xfe\n")
    with pytest.raises(AssertionError, match="frozen-test timestamp"):
        _read_oof(path, BOUNDARY)


def test_uniform_manual_sample_spans_the_entire_score_grid():
    origins = pd.date_range("2021-05-01", periods=1001, freq="15min")
    picked = _uniform_origins(origins)
    assert len(picked) == 20 and picked[0] == origins[0] and picked[-1] == origins[-1]
    assert picked.is_unique and picked.is_monotonic_increasing


def test_development_loader_output_cannot_include_frozen_origin():
    safe = pd.DataFrame({"power": [1., 2.]},
                        index=pd.date_range(BOUNDARY - pd.Timedelta(minutes=30),
                                            periods=2, freq="15min"))
    _assert_development_frame(safe, BOUNDARY)
    unsafe = pd.concat([safe, pd.DataFrame({"power": [3.]}, index=[BOUNDARY])])
    with pytest.raises(AssertionError, match="frozen test"):
        _assert_development_frame(unsafe, BOUNDARY)


def test_h96_target_slot_feature_uses_known_current_and_prior_week():
    index = pd.date_range("2021-01-01", periods=96 * 15, freq="15min")
    frame = pd.DataFrame({"power": np.arange(len(index), dtype=float),
                          "time_repaired": False}, index=index)
    origin = pd.DatetimeIndex([index[96 * 12]])
    x, used = build_features(frame, origin, 96, {"_include_cbl": False})
    assert used.iloc[0] == origin[0]
    assert x.loc[origin[0], "target_slot_1d_ago"] == frame.loc[origin[0], "power"]
    assert x.loc[origin[0], "target_slot_7d_ago"] == frame.loc[origin[0] - pd.Timedelta(days=6), "power"]
    assert x.loc[origin[0], "lag_672"] == frame.loc[origin[0] - pd.Timedelta(days=7), "power"]


def test_diagnostic_output_is_confined_to_results_or_validation(tmp_path):
    assert _checked_output(tmp_path, None) == tmp_path / "outputs"
    assert _checked_output(tmp_path, tmp_path / "_validation" / "repeat") == tmp_path / "_validation" / "repeat"
    with pytest.raises(ValueError, match="outputs/ or _validation/"):
        _checked_output(tmp_path, tmp_path / "report" / "overwrite")
    with pytest.raises(ValueError, match="outputs/ or _validation/"):
        _checked_output(tmp_path, tmp_path.parent / "outside")


def test_early_stopping_audit_distinguishes_l1_minimum_from_joint_stop(tmp_path):
    model_dir = tmp_path / "outputs" / "models"
    model_dir.mkdir(parents=True)
    for fold in range(3):
        for name in ("lgbm", "lgbm_no_holiday"):
            model = SimpleNamespace(best_iteration_=2,
                                    evals_result_={"valid_0": {"l1": [4., 3., 1.],
                                                                 "l2": [5., 2., 4.]}})
            joblib.dump({"horizon": 96, "fold": fold, "model": model},
                        model_dir / f"development_h96_fold{fold}_{name}.joblib")
    audit = audit_early_stopping(tmp_path)
    assert len(audit) == 6
    assert set(audit.monitored_metrics) == {"l1,l2"}
    assert set(audit.best_iteration) == {2}
    assert set(audit.min_l1_iteration) == {3}
    assert set(audit.min_l2_iteration) == {2}
    assert set(audit.l1_at_best) == {3.}
    assert set(audit.l2_at_best) == {2.}
