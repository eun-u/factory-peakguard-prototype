import numpy as np
import pandas as pd

from src.training import _baseline_frames


def test_canonical_cbl_candidate_keeps_prediction_and_calibrated_alarm(monkeypatch):
    def base(_df, times, _h):
        return pd.DataFrame({"p1_latest": np.arange(len(times))}, index=times)

    def cbl(_df, times, _h):
        values = np.arange(len(times), dtype=float)
        return pd.DataFrame({"c1_mid_6_10": values, "c2_max_4_5": values+1,
                             "c3_holiday_mid_4_6": values}, index=times)

    monkeypatch.setattr("src.models.baselines.baseline_predictions", base)
    monkeypatch.setattr("src.models.cbl.cbl_all_predictions", cbl)
    cal = pd.date_range("2021-02-02", periods=40, freq="15min")
    score = pd.date_range("2021-02-03", periods=40, freq="15min")
    times = cal.append(score)
    targets = pd.DataFrame({"target_time": times+pd.Timedelta(hours=1),
                            "y": np.tile(np.arange(40), 2)}, index=times)
    frames = _baseline_frames(pd.DataFrame(), cal[:1], cal, score, 4, 0, targets, 30)
    by_name = {frame.model.iloc[0]: frame for frame in frames}
    # Historical OOF generation retains both IDs; presentation consolidation
    # is permitted only after the runner verifies their equality.
    hybrid, mid = by_name["c3_holiday_hybrid"], by_name["c3_holiday_mid_4_6"]
    np.testing.assert_array_equal(hybrid.pred, mid.pred)
    np.testing.assert_array_equal(hybrid.alert, mid.alert)
    np.testing.assert_array_equal(hybrid.alert_cutoff, mid.alert_cutoff)
