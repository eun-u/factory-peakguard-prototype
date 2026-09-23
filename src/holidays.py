"""Explicit 2021 Korean calendar assumptions; official human check remains pending."""

from __future__ import annotations

import pandas as pd

HOLIDAYS_2021: dict[str, str] = {
    "2021-01-01": "신정",
    "2021-02-11": "설날 연휴",
    "2021-02-12": "설날",
    "2021-02-13": "설날 연휴",
    "2021-03-01": "삼일절",
    "2021-05-05": "어린이날",
    "2021-05-19": "부처님오신날",
    "2021-06-06": "현충일",
    "2021-08-15": "광복절",
    "2021-08-16": "광복절 대체공휴일",
}
CALENDAR_REVIEWED = False


def calendar_flags(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Return only calendar-known flags; no measured values are consulted."""
    times = pd.DatetimeIndex(times)
    dates = times.normalize()
    holiday_days = pd.DatetimeIndex(pd.to_datetime(list(HOLIDAYS_2021)))
    holiday = dates.isin(holiday_days)
    weekend = dates.dayofweek >= 5
    offday = holiday | weekend
    before = (dates + pd.Timedelta(days=1)).isin(holiday_days)
    after = (dates - pd.Timedelta(days=1)).isin(holiday_days)
    left_off = (dates - pd.Timedelta(days=1)).isin(holiday_days) | ((dates - pd.Timedelta(days=1)).dayofweek >= 5)
    right_off = (dates + pd.Timedelta(days=1)).isin(holiday_days) | ((dates + pd.Timedelta(days=1)).dayofweek >= 5)
    left_weekend = (dates - pd.Timedelta(days=1)).dayofweek >= 5
    right_weekend = (dates + pd.Timedelta(days=1)).dayofweek >= 5
    bridge = (~offday) & left_off & right_off & ((left_weekend & ~right_weekend) | (right_weekend & ~left_weekend))
    return pd.DataFrame({
        "is_holiday": holiday.astype(int),
        "labor_day": (dates == pd.Timestamp("2021-05-01")).astype(int),
        "pre_holiday": before.astype(int),
        "post_holiday": after.astype(int),
        "bridge_day": bridge.astype(int),
        "is_offday": offday.astype(int),
    }, index=times)
