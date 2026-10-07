import numpy as np
import pandas as pd
from src.models.conformal import fit_conformal, apply_conformal, assign_mondrian_bins, fit_mondrian_edges
from src.evaluate import score_predictions


def test_upper_conformal_finite_sample_rank_and_mondrian():
    y = np.arange(100, dtype=float)
    raw = y - 1
    fitted = fit_conformal(y, raw, .95)
    assert np.all(apply_conformal(raw, fitted) >= y)
    edges = fit_mondrian_edges(raw)
    groups = assign_mondrian_bins(raw, edges)
    mondrian = fit_conformal(y, raw, .95, groups)
    assert len(mondrian["by_bin"]) == 2  # sparse top decile uses global correction
    assert np.all(apply_conformal(raw, mondrian, groups) >= y)


def test_mixed_quantile_table_scores_raw_rows_from_finite_raw_values():
    times = pd.date_range("2021-01-01", periods=2, freq="15min")
    common = {"target_time": times, "y": [10., 20.], "pred": [10., 20.],
              "tau": [100., 100.], "alert": [False, False],
              "q10": [9., 19.], "q50": [10., 20.],
              "q90": [11., 21.], "q95": [11., 21.], "q975": [11., 21.]}
    raw = pd.DataFrame(common)
    raw["model"] = "lgbm_quantile_raw"
    calibrated = pd.DataFrame(common)
    calibrated["model"] = "lgbm_quantile_a"
    calibrated["q95_cal"] = [9., 19.]
    mixed = pd.concat([raw, calibrated], ignore_index=True)

    raw_score = score_predictions(mixed.loc[mixed.model == "lgbm_quantile_raw"])
    calibrated_score = score_predictions(mixed.loc[mixed.model == "lgbm_quantile_a"])
    assert raw_score["coverage_0.95"] == 1.0
    assert np.isclose(raw_score["pinball_0.95"], .05)
    assert calibrated_score["coverage_0.95"] == 0.0
    assert np.isclose(calibrated_score["pinball_0.95"], .95)
