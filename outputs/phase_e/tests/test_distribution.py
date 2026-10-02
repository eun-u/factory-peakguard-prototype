"""Synthetic checks for Phase E's fixed B5 h16 distribution layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from outputs.phase_e.code.distribution import (
    HORIZON,
    _checked_artifact,
    predict_kalman_distribution,
)
from phase_c.statistical import predict_kalman


def _example() -> tuple[dict, pd.Series, pd.DatetimeIndex]:
    index = pd.date_range("2021-01-01 00:15", periods=900, freq="15min")
    positions = np.arange(len(index), dtype=float)
    power = pd.Series(100 + 10 * np.sin(positions / 37) + positions / 300, index=index)
    origins = index[[700, 715, 730]]
    return {"phi": 0.7, "q": 3.0, "r": 5.0}, power, origins


def test_predictive_mean_is_exact_phase_c_point_prediction() -> None:
    bundle, power, origins = _example()
    mean, sigma = predict_kalman_distribution(bundle, power, origins)
    assert np.array_equal(mean, predict_kalman(bundle, power, origins, HORIZON))
    assert np.isfinite(sigma).all()
    assert (sigma > 0).all()
    assert sigma.shape == mean.shape == (3,)


def test_future_observations_cannot_change_an_earlier_distribution() -> None:
    bundle, power, origins = _example()
    mean, sigma = predict_kalman_distribution(bundle, power, origins[:1])
    changed = power.copy()
    changed.loc[changed.index > origins[0]] += 10000
    later_mean, later_sigma = predict_kalman_distribution(bundle, changed, origins[:1])
    np.testing.assert_array_equal(mean, later_mean)
    np.testing.assert_array_equal(sigma, later_sigma)


def test_missing_deviation_skips_update_and_increases_uncertainty() -> None:
    bundle, power, origins = _example()
    _, complete_sigma = predict_kalman_distribution(bundle, power, origins[:1])
    incomplete = power.copy()
    incomplete.iloc[690] = np.nan
    _, missing_sigma = predict_kalman_distribution(bundle, incomplete, origins[:1])
    assert missing_sigma[0] > complete_sigma[0]


def test_observed_forecast_variance_matches_steady_state_limit() -> None:
    bundle, power, origins = _example()
    _, sigma = predict_kalman_distribution(bundle, power, origins[-1:])
    phi, q, r = bundle["phi"], bundle["q"], bundle["r"]
    coefficient = q + r - phi * phi * r
    posterior_limit = (-coefficient + np.sqrt(coefficient ** 2 + 4 * phi * phi * q * r)) / (2 * phi * phi)
    future = phi ** (2 * HORIZON) * posterior_limit + q * (1 - phi ** (2 * HORIZON)) / (1 - phi * phi) + r
    assert sigma[0] ** 2 == pytest.approx(future, abs=1e-10)


def test_fixed_horizon_and_variance_parameter_contract() -> None:
    bundle, power, origins = _example()
    with pytest.raises(ValueError, match="h16"):
        predict_kalman_distribution(bundle, power, origins, 15)
    for changed in ({**bundle, "phi": float("nan")},
                    {**bundle, "q": 0}, {**bundle, "r": float("inf")}):
        with pytest.raises(ValueError, match="finite"):
            predict_kalman_distribution(changed, power, origins)


def test_artifact_hash_is_checked_before_deserialization() -> None:
    root = Path(__file__).resolve().parents[3]
    relative = Path(__file__).resolve().relative_to(root).as_posix()
    target = root / relative
    import hashlib
    artifacts = {relative: {"bytes": target.stat().st_size,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}}
    assert _checked_artifact(root, Path(relative), artifacts) == artifacts[relative]["sha256"]
    artifacts[relative]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        _checked_artifact(root, Path(relative), artifacts)
