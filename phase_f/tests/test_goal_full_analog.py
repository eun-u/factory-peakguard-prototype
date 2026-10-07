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


def test_v2_features_ignore_values_after_origin():
    from phase_f.goal_full_analog_v2 import Features as F2
    history = _history()
    history.iloc[96 * 31 + 5, 0] = np.nan  # missing slot must not leak or crash
    reference = F2(history)
    for origin in history.index[96 * 31 + np.array([3, 40, 90])]:
        changed = history.copy()
        changed.loc[changed.index > origin, "power"] += 1000.
        perturbed = F2(changed)
        for h in (4, 16):
            pd.testing.assert_frame_equal(reference.build([origin], h), perturbed.build([origin], h))


def test_v2_gate_uses_most_recent_exact_day():
    from phase_f.goal_full_analog_v2 import Features as F2, gate as gate2
    history = _history()
    F = F2(history)
    origin = history.index[96 * 35 + 50]
    X = F.build([origin], 8)
    assert gate2(X)[0]
    assert X.a_today_pred.iloc[0] == history.power.loc[origin + pd.Timedelta(minutes=120)]


def test_gate_chronos_uses_analog_when_gated_and_chronos_otherwise():
    from phase_f import goal_gate_chronos as G
    history = _history(days=45)
    F = G.Features(history)
    idx = history.index
    contexts = {}
    for h in G.HORIZONS:
        o = idx[96 * 30:96 * 44 - 20]
        y = pd.Series(history.power.reindex(o + pd.Timedelta(minutes=15 * h)).to_numpy(), index=o)
        contexts[(h, 0)] = {"cal": o[:400], "score": o[400:800], "fit": o[:400], "tau": 150.,
                            "y": y, "target_time": pd.Series(o + pd.Timedelta(minutes=15 * h), index=o),
                            "d2": pd.Series(False, index=o)}
    origins = idx[96 * 30:96 * 44]
    chronos = pd.DataFrame([(o, h, 1., 2., 3., 4., 5.) for o in origins for h in G.HORIZONS],
                           columns=["origin", "horizon", "ch_q05", "ch_q10", "ch_q50", "ch_q90", "ch_q95"])
    out = G.predict(F, contexts, chronos, [0])
    gated = out[out.gated]
    assert len(gated) and np.allclose(gated.pred, gated.analog)
    ungated = out[~out.gated]
    assert np.allclose(ungated.point_r0, 3.)
    assert set(np.round(ungated.pred - 3., 6)) <= {0., float(out["shift"].iloc[0])}
