import pandas as pd
from src.evaluate import match_episodes


def test_gap_separates_episodes_and_one_alert_cannot_match_two():
    times = pd.to_datetime(["2021-01-01 00:15", "2021-01-01 00:30", "2021-01-01 01:00"])
    result = match_episodes([1, 1, 1], [1, 1, 1], times)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    result = match_episodes([1, 0, 1], [1, 1, 1], pd.date_range("2021-01-01", periods=3, freq="15min"))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)


def test_matching_maximizes_number_of_episodes_before_overlap():
    times = pd.date_range("2021-01-01", periods=21, freq="15min")
    actual = [0] * 21
    alert = [0] * 21
    for i in list(range(0, 11)) + list(range(12, 21)):
        actual[i] = 1
    for i in list(range(0, 4)) + list(range(6, 16)):
        alert[i] = 1
    result = match_episodes(actual, alert, times)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
