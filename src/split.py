"""Time-ordered development folds with purged target-time boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex


def make_splits(
    origins: pd.DatetimeIndex, horizon: int, dev_frac: float = .85, n_folds: int = 3,
) -> tuple[list[Fold], pd.DatetimeIndex]:
    """Use first 85% for expanding CV; reserve final 15% for frozen test.

    At every boundary, a fit label at origin+h must be strictly earlier than
    the following validation/test origin. The split is based on passed origins,
    so callers should pass the same eligible grid to every candidate model.
    """
    if horizon < 1 or not 0 < dev_frac <= 1 or n_folds < 1:
        raise ValueError("Invalid split parameters")
    origins = pd.DatetimeIndex(origins).sort_values().unique()
    if len(origins) < (n_folds + 2) * (horizon + 1):
        raise ValueError("Not enough origins for purged rolling folds")
    n_dev = int(len(origins) * dev_frac)
    dev, test = origins[:n_dev], origins[n_dev:]
    horizon_delta = pd.Timedelta(minutes=15 * horizon)
    # Frozen test starts at the original 85% boundary. Purging removes rows
    # from development fitting, never shifts the test boundary.
    if len(test):
        dev = dev[dev + horizon_delta < test.min()]
    else:
        # The caller may pass only the fixed development span. Do not allow
        # its final labels to reach past that span into a hidden test period.
        dev = dev[dev + horizon_delta <= dev.max()]
    if len(dev) <= n_folds + 1:
        raise ValueError("Development period is too short after purge")
    # 40% initial history and three contiguous 20% validation blocks.
    first_validation = max(1, int(n_dev * .4))
    edges = [first_validation + int((len(dev) - first_validation) * i / n_folds) for i in range(n_folds + 1)]
    folds: list[Fold] = []
    for i in range(n_folds):
        validation = dev[edges[i]:edges[i + 1]]
        if validation.empty:
            raise ValueError("Empty validation fold")
        train = dev[:edges[i]]
        train = train[train + horizon_delta < validation.min()]
        if train.empty:
            raise ValueError("Empty purged training fold")
        folds.append(Fold(train=train, validation=validation))
    return folds, test


def assert_purged(folds: list[Fold], test: pd.DatetimeIndex, horizon: int) -> None:
    delta = pd.Timedelta(minutes=15 * horizon)
    for fold in folds:
        if not fold.train.max() + delta < fold.validation.min():
            raise AssertionError("Training label overlaps validation origin")
    if len(test) and not folds[-1].validation.max() + delta < test.min():
        raise AssertionError("Development label overlaps test origin")
