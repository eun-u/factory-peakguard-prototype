import numpy as np
import pandas as pd

from phase_f.goal_full_analog import Features, gate


def _history(days=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01 00:15", periods=96 * days, freq="15min")
    base = [rng.integers(20, 200, 96).astype(float) for _ in range(5)]
    values = np.concatenate([base[rng.integers(0, 5)] for _ in range(days)])
    return pd.DataFrame({"power": values}, index=idx)


def test_features_ignore_values_after_origin():
    history = _history()
    origins = history.index[96 * 30 + np.array([3, 40, 90])]
    reference = Features(history)
    for origin in origins:
        changed = history.copy()
        changed.loc[changed.index > origin, "power"] += 1000.
        perturbed = Features(changed)
        for h in (4, 16):
            a = reference.build([origin], h)
            b = perturbed.build([origin], h)
            pd.testing.assert_frame_equal(a, b)


def test_exact_copy_day_is_gated_and_exact():
    history = _history()
    F = Features(history)
    origin = history.index[96 * 35 + 50]  # mid-day, target same day for h<=16
    X = F.build([origin], 8)
    assert gate(X)[0]
    target = origin + pd.Timedelta(minutes=15 * 8)
    assert X.a_today_pred.iloc[0] == history.power.loc[target]


def test_cross_midnight_is_not_gated():
    history = _history()
    F = Features(history)
    origin = history.index[96 * 35 + 90]
    assert not gate(F.build([origin], 16))[0]
