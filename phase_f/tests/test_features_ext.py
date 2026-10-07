"""Synthetic checks for Phase F feature availability and fixed row keys."""

import numpy as np
import pandas as pd
import pytest

from phase_f.features_ext import (
    FEATURE_GROUPS, available_group_configs, build_features, causal_sequences,
)


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    index = pd.date_range("2021-04-01 00:15", periods=46 * 96, freq="15min", name="ts_end")
    positions = np.arange(len(index), dtype=float)
    power = 55 + 7 * np.sin(2 * np.pi * positions / 96) + positions / 1000
    completed = np.where(index.minute == 0, 8 + (positions // 4) % 9, np.nan)
    return pd.DataFrame({
        "power": power,
        "time_repaired": False,
        "production_completed": completed,
        "production_completed_bad": False,
    }, index=index)


def test_all_f1_feature_groups_are_causal_under_future_perturbation(history: pd.DataFrame) -> None:
    origin = pd.DatetimeIndex([pd.Timestamp("2021-05-09 13:15")], name="origin")
    before, provenance = build_features(history, origin, 16, tau=63.0)
    perturbed = history.copy()
    future = perturbed.index > origin[0]
    perturbed.loc[future, "power"] += 1_000
    perturbed.loc[future, "production_completed"] += 10_000
    after, after_provenance = build_features(perturbed, origin, 16, tau=63.0)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert before.index.equals(origin)
    assert set(before.columns) == set(provenance)
    assert set(provenance) == set(after_provenance)
    assert all(not (used > origin).fillna(False).any() for used in provenance.values())
    assert provenance["pre_holiday"].isna().all()
    assert provenance["production_last_completed_hour"].iloc[0] == pd.Timestamp("2021-05-09 13:00")


def test_each_group_has_frozen_ablation_descriptor_and_preserves_rows(history: pd.DataFrame) -> None:
    configs = available_group_configs()
    assert len({row["id"] for row in configs}) == len(configs)
    assert {group for row in configs for group in row["groups"]} == set(FEATURE_GROUPS)
    origins = pd.DatetimeIndex([history.index[4], history.index[1000], history.index[3800]], name="origin")
    for group, names in FEATURE_GROUPS.items():
        features, provenance = build_features(history, origins, 4, 63.0, groups=[group])
        assert features.index.equals(origins)
        assert set(names) <= set(features)
        assert features.columns.is_unique
        assert all(not (used > origins).fillna(False).any() for used in provenance.values())
    long, _ = build_features(history, origins, 4, 63.0, groups=["slot_7_28d"])
    assert np.isnan(long.loc[origins[0], "slot_28d"])


def test_production_requires_hour_completion_and_ignores_target_field(history: pd.DataFrame) -> None:
    origin = pd.DatetimeIndex([pd.Timestamp("2021-05-09 13:15")], name="origin")
    base, provenance = build_features(history, origin, 16, 63.0, ["production"])
    modified = history.copy()
    modified["production_target"] = -99_999
    modified.loc[pd.Timestamp("2021-05-09 13:15"), "production_target"] = 99_999
    result, _ = build_features(modified, origin, 16, 63.0, ["production"])
    pd.testing.assert_frame_equal(base, result)
    assert base.loc[origin[0], "production_last_completed_hour"] == history.loc[pd.Timestamp("2021-05-09 13:00"), "production_completed"]
    assert provenance["production_same_slot_7d_mean"].iloc[0] == origin[0] - pd.Timedelta(days=1)
    assert "production_target" not in base


def test_sequences_left_pad_and_observation_mask_without_future_fill() -> None:
    index = pd.date_range("2021-01-01 00:15", periods=5, freq="15min", name="ts_end")
    history = pd.DataFrame({"power": [1., 2., 3., 4., 5.], "time_repaired": [False, False, True, False, False]}, index=index)
    origins = pd.DatetimeIndex([index[1], index[3]], name="origin")
    values, mask = causal_sequences(history, origins, 5)
    np.testing.assert_array_equal(mask, [[False, False, False, True, True], [False, True, True, False, True]])
    assert np.isnan(values[0, :3]).all()
    assert np.isnan(values[1, 3])
    assert values[0, -1] == 2
    assert values[1, -1] == 4
    later = history.copy()
    later.loc[index[4], "power"] = 999
    later_values, later_mask = causal_sequences(later, origins, 5)
    np.testing.assert_array_equal(values, later_values)
    np.testing.assert_array_equal(mask, later_mask)


def test_fail_closed_on_future_slot_or_unknown_group(history: pd.DataFrame) -> None:
    origin = pd.DatetimeIndex([history.index[200]], name="origin")
    with pytest.raises(ValueError, match="Unknown feature groups"):
        build_features(history, origin, 4, 63.0, ["does_not_exist"])
    with pytest.raises(ValueError, match="4..16"):
        build_features(history, origin, 97, 63.0, ["slot_1_7d"])
    with pytest.raises(ValueError, match="fit-only tau"):
        build_features(history, origin, 4, np.nan, ["peak"])
