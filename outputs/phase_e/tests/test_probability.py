"""Synthetic-only checks for the frozen Phase E risk and uncertainty contract."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

from outputs.phase_e.code.probability import (
    CLIP, _draws, calibrate, evaluate_probability,
)


def example(partition: str, n: int = 40, fold: int = 0) -> pd.DataFrame:
    first = "2021-01-01 00:00" if partition == "cal" else "2021-01-03 00:00"
    origin = pd.date_range(first, periods=n, freq="15min")
    i = np.arange(n)
    mu = 100 + 6 * np.sin(i / 4)
    y = 100 + 5 * np.sin(i / 4 + 0.6) + (i % 5 - 2) * 1.2
    return pd.DataFrame({"fold": fold, "partition": partition,
                         "origin": origin, "target_time": origin + pd.Timedelta("4h"),
                         "y": y, "tau": 100.0, "mu": mu, "sigma": 4.0,
                         "is_peak": y > 100, "is_d2_novel_profile": i % 3 == 0,
                         "p_climatology": .05, "model": "B5", "horizon": 16})


def test_platt_probability_and_conformal_rank() -> None:
    cal, score = example("cal"), example("score")
    scored, fit_log, conf_log = calibrate(cal, score)
    fit = fit_log["fits"][0]
    conformal = conf_log["fits"][0]
    assert fit["b"] >= 0
    assert fit["n_cal"] == len(cal)
    expected_raw = norm.sf((score.tau - score.mu) / score.sigma)
    np.testing.assert_allclose(scored.p_raw, expected_raw)
    np.testing.assert_allclose(scored.p_cal,
                               expit(fit["a"] + fit["b"] * logit(np.clip(expected_raw, CLIP, 1 - CLIP))))
    assert scored.p_cal.between(0, 1).all()
    assert conformal["rank_1_based"] == 39  # ceil((40+1)*0.95)
    standardized = np.sort((cal.y - cal.mu) / cal.sigma)
    assert conformal["q_star"] == pytest.approx(standardized[38])
    np.testing.assert_allclose(scored.U95, scored.mu + conformal["q_star"] * scored.sigma)
    np.testing.assert_array_equal(scored.uncertainty_flag, scored.U95 > scored.tau)


def test_score_labels_do_not_change_any_fitted_probability_or_bound() -> None:
    cal, score = example("cal"), example("score")
    first, fit1, conf1 = calibrate(cal, score)
    alternative = score.copy()
    alternative["y"] = 200 - alternative.y
    alternative["is_peak"] = alternative.y > alternative.tau
    second, fit2, conf2 = calibrate(cal, alternative)
    pd.testing.assert_frame_equal(first[["p_raw", "p_cal", "q95_raw", "U95"]],
                                  second[["p_raw", "p_cal", "q95_raw", "U95"]])
    assert fit1 == fit2 and conf1 == conf2


def test_invalid_calibration_rows_fail_without_dropping_or_fallback() -> None:
    cal, score = example("cal"), example("score")
    cal.loc[0, "sigma"] = 0
    with pytest.raises(ValueError, match="sigma"):
        calibrate(cal, score)
    cal = example("cal")
    cal.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite cal y"):
        calibrate(cal, score)
    cal = example("cal")
    cal["y"] = 80.0
    cal["is_peak"] = False
    with pytest.raises(ValueError, match="both classes"):
        calibrate(cal, score)
    cal = example("cal")
    cal["mu"] = np.where(cal.is_peak, 105.0, 95.0)
    with pytest.raises(RuntimeError, match="finite monotone MLE"):
        calibrate(cal, score)
    cal = example("cal")
    cal["p_raw"] = 0.25
    with pytest.raises(ValueError, match="Gaussian formula"):
        calibrate(cal, score)


def test_extreme_probabilities_use_fixed_clip_without_new_method() -> None:
    cal, score = example("cal"), example("score")
    score.loc[0, "mu"] = -1e5
    score.loc[1, "mu"] = 1e5
    result, fit, _ = calibrate(cal, score)
    a, b = fit["fits"][0]["a"], fit["fits"][0]["b"]
    assert result.p_raw.iloc[0] == 0 and result.p_raw.iloc[1] == 1
    assert result.p_cal.iloc[0] == pytest.approx(expit(a + b * logit(CLIP)))
    assert result.p_cal.iloc[1] == pytest.approx(expit(a + b * logit(1 - CLIP)))


def test_evaluation_reports_fixed_bins_and_fold_day_bootstrap() -> None:
    cal, score = example("cal", n=40), example("score", n=120)
    scored, _, _ = calibrate(cal, score)
    result = evaluate_probability(scored, n_boot=31, seed=42)
    assert set(result) == {"risk_metrics", "risk_metrics_by_fold",
                           "reliability_bins_raw", "reliability_bins_calibrated",
                           "calibration_summary", "uncertainty_metrics"}
    risk = result["risk_metrics"].set_index("subset")
    assert set(risk.index) == {"D1", "D2"}
    assert risk.loc["D1", "BS_cal"] == pytest.approx(
        np.mean((scored.p_cal - scored.is_peak.astype(int)) ** 2))
    assert risk.loc["D1", "BS_raw_minus_cal"] == pytest.approx(
        risk.loc["D1", "BS_raw"] - risk.loc["D1", "BS_cal"])
    assert risk.loc["D1", "BS_raw_minus_cal_ci_valid_draws"] == 31
    bins = result["reliability_bins_calibrated"]
    assert bins.query("subset == 'D1' and fold == 'pooled'").shape[0] == 10
    assert bins.query("subset == 'D1' and fold == 'pooled'").n.sum() == len(scored)
    assert (bins.loc[bins.n.eq(0), "observed_rate_ci_status"] == "unavailable_sparse").all()
    assert set(result["uncertainty_metrics"].method) == {"raw", "conformal"}
    again = evaluate_probability(scored, n_boot=31, seed=42)
    pd.testing.assert_frame_equal(result["risk_metrics"], again["risk_metrics"])


def test_bootstrap_resamples_whole_days_within_fold() -> None:
    data = pd.concat([example("score", n=120, fold=0),
                      example("score", n=120, fold=1)], ignore_index=True)
    draws, block = _draws(data, n_boot=23, seed=42)
    keys = pd.MultiIndex.from_arrays([data.fold, data.target_time.dt.normalize()])
    assert len(np.unique(block)) == len(keys.unique())
    for fold in (0, 1):
        in_fold = data.fold.to_numpy() == fold
        fold_blocks = np.unique(block[in_fold])
        assert (draws[:, fold_blocks].sum(axis=1) == len(fold_blocks)).all()
    # Every row in one (fold, target-day) block receives the same draw weight.
    assert np.array_equal(draws[:, block[0]], draws[:, block[1]])
