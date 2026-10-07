import numpy as np
import pandas as pd

from src.bootstrap import day_bootstrap, day_mean_ci


def test_vectorized_date_bootstrap_matches_row_resampling():
    dates = pd.to_datetime(["2021-01-01"] * 3 + ["2021-01-02"] * 2 + ["2021-01-03"] * 4)
    frame = pd.DataFrame({"target_time": dates, "value": [1., 2., 3., 10., 12., 4., 5., 6., 7.]})
    slow = day_bootstrap(frame, lambda part: float(part.value.mean()), n=1000, seed=42)
    fast = day_mean_ci(frame, frame.value, n=1000, seed=42)
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)
