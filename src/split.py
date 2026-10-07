"""시간순 분할. 랜덤 분할은 쓰지 않는다.

경계마다 예측거리만큼의 원점을 지워(purge), 앞 구간의 마지막 목표 시각이 다음 구간의
첫 원점보다 앞서게 한다.
"""

from __future__ import annotations

import pandas as pd


def split_time(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
               fractions: tuple[float, float] = (0.70, 0.85), purge: int = 4):
    n = len(x)
    cut1, cut2 = int(n * fractions[0]), int(n * fractions[1])
    sl = {"train": slice(0, cut1 - purge), "validation": slice(cut1, cut2 - purge), "test": slice(cut2, n)}
    parts = {k: (x.iloc[s], y.iloc[s], meta.iloc[s]) for k, s in sl.items()}
    assert_no_overlap(parts)
    return parts


def assert_no_overlap(parts: dict) -> None:
    names = list(parts)
    for earlier, later in zip(names, names[1:]):
        if parts[earlier][2]["target_time"].max() >= parts[later][0].index.min():
            raise AssertionError(f"{earlier}의 목표 시각이 {later}의 첫 원점과 겹칩니다")


def rolling_origin_folds(n: int, start_fractions=(.30, .45, .60), width: float = .10, purge: int = 4):
    """시험 구간(마지막 15%) 이전에서만 만드는 확장 창 폴드. (학습 끝, 평가 시작, 평가 끝) 인덱스."""
    folds = []
    for fraction in start_fractions:
        boundary = int(n * fraction)
        end = int(n * (fraction + width))
        folds.append((boundary - purge, boundary, end))
    return folds
