"""Small synthetic F3-5/F3-6 tests; no plant data or frozen outputs."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from phase_f.models import multihorizon


@pytest.fixture
def workdir():
    # The Windows pytest tmp_path root receives an inaccessible ACL in this
    # workspace; create an ordinary ignored scratch directory instead.
    path = Path.cwd() / "outputs/phase_f/test_scratch" / f"multihorizon_{uuid4().hex}"
    path.mkdir(parents=True)
    return path


class SyntheticPrepared:
    def __init__(self, out: Path):
        index = pd.date_range("2021-01-01", periods=1100, freq="15min")
        position = np.arange(len(index))
        power = (70 + 9 * np.sin(2 * np.pi * position / 96)
                 + 2 * np.cos(2 * np.pi * position / 672))
        self.history = pd.DataFrame({"power": power}, index=index)
        self.out = out
        self.seal = {"raw_sha256": "synthetic-only"}
        self.split_lock = {"lock_sha256": "synthetic-split-v1"}
        self.contexts = {}
        for h in multihorizon.HORIZONS:
            targets = index + pd.Timedelta(minutes=15 * h)
            slot7d = self.history.power.reindex(targets - pd.Timedelta(days=7)).to_numpy()
            context = {
                "x": pd.DataFrame({"slot7d": slot7d}, index=index),
                "y": self.history.power.shift(-h),
                "target_time": pd.Series(targets, index=index),
                "fit": index[700:800], "stop": index[840:850],
                "cal": index[880:885], "score": index[920:925],
                "tau": 77.0, "d2": pd.Series(False, index=index),
            }
            self.contexts[(h, 0)] = context

    def origins(self, horizon, fold, role):
        return self.contexts[(horizon, fold)][role]


@pytest.mark.parametrize("kind,target", [
    ("global_h", "direct"),
    ("multioutput", "weekly"),
])
def test_complete_cohort_and_actual_future_perturbation(workdir, kind, target):
    pytest.importorskip("lightgbm")
    prepared = SyntheticPrepared(workdir)
    spec = {"id": f"F3-synthetic-{kind}", "kind": kind, "target": target,
            "groups": (), "early_stopping_rounds": 3,
            "model_params": {"n_estimators": 14, "num_leaves": 7,
                             "min_child_samples": 5, "n_jobs": 1}}
    frame, audit = multihorizon.run(prepared, spec)
    assert len(frame) == len(multihorizon.HORIZONS) * 10
    assert frame.groupby(["horizon", "role"]).size().eq(5).all()
    assert frame[["horizon", "fold", "origin", "target_time", "role"]].duplicated().sum() == 0
    assert np.isfinite(frame.pred).all()
    assert audit["folds"][0]["future_perturbation_max_abs_difference"] == 0
    assert audit["folds"][0]["n_fit"] >= 30
    assert audit["folds"][0]["n_stop"] >= 2

    cached, second_audit = multihorizon.run(prepared, spec)
    assert second_audit["folds"][0]["cache_reused"] is True
    np.testing.assert_array_equal(cached.pred.to_numpy(), frame.pred.to_numpy())

    metadata = prepared.out / "models" / spec["id"] / "fold_0.json"
    old = metadata.read_text(encoding="utf-8")
    metadata.write_text(old.replace('"identity": "', '"identity": "changed-'), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        multihorizon.run(prepared, spec)


def test_fit_and_stop_boundary_excludes_late_shared_labels(workdir):
    prepared = SyntheticPrepared(workdir)
    spec = {"id": "F3-boundary", "kind": "global_h", "target": "direct", "groups": ()}
    cfg = multihorizon._as_config(spec)
    features = multihorizon._feature_map(prepared, 0, cfg)
    x, y, counts = multihorizon._matrix_and_labels(prepared, 0, cfg, features, "fit")
    assert len(x) == len(y) == sum(counts.values())
    assert all(count == 100 for count in counts.values())
    for h in multihorizon.HORIZONS:
        context = prepared.contexts[(h, 0)]
        context["fit"] = context["fit"].append(pd.DatetimeIndex([context["stop"].min()]))
    x, y, counts = multihorizon._matrix_and_labels(prepared, 0, cfg, features, "fit")
    assert len(x) == len(y) == sum(counts.values())
    assert all(count == 100 for count in counts.values())


def test_cal_and_score_labels_cannot_change_shared_model(workdir):
    pytest.importorskip("lightgbm")
    prepared = SyntheticPrepared(workdir)
    cfg = multihorizon._as_config({
        "id": "F3-label-boundary", "kind": "global_h", "target": "direct",
        "groups": (), "early_stopping_rounds": 3,
        "model_params": {"n_estimators": 12, "num_leaves": 7,
                         "min_child_samples": 5, "n_jobs": 1},
    })
    features = multihorizon._feature_map(prepared, 0, cfg)
    fitted = multihorizon._fit_fold(prepared, 0, cfg, features)
    origins = prepared.origins(16, 0, "score")
    before, _ = multihorizon._predict(fitted, features, origins, 16)
    for context in prepared.contexts.values():
        context["y"].loc[context["cal"]] = 1e6
        context["y"].loc[context["score"]] = -1e6
    refitted = multihorizon._fit_fold(prepared, 0, cfg, features)
    after, _ = multihorizon._predict(refitted, features, origins, 16)
    np.testing.assert_array_equal(before, after)
