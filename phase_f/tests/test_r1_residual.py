"""Synthetic contracts for the weekly R1 residual adapter (no Chronos/GPU)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f.models import r1_residual as adapter


def _fixture(tmp_path):
    index = pd.date_range("2021-02-20", "2021-04-18", freq="15min", name="origin")
    power = 60 + 12 * np.sin(np.arange(len(index)) / 96 * 2 * np.pi)
    history = pd.DataFrame({"power": power}, index=index)
    fit = pd.date_range("2021-03-07", periods=300, freq="15min", name="origin")
    stop = pd.date_range("2021-04-01", periods=40, freq="15min", name="origin")
    cal = pd.date_range("2021-04-09", periods=8, freq="15min", name="origin")
    score = pd.date_range("2021-04-12", periods=8, freq="15min", name="origin")
    contexts = {}
    records = []
    for h in adapter.HORIZONS:
        origins = fit.union(stop).union(cal).union(score)
        target = origins + h * adapter._SLOT
        truth = pd.Series(history.power.reindex(target).to_numpy(float), index=origins)
        contexts[(h, 0)] = {
            "fit": fit, "stop": stop, "cal": cal, "score": score,
            "y": truth, "target_time": pd.Series(target, index=origins),
            "x": pd.DataFrame({"slot7d": np.full(len(origins), 55.)}, index=origins),
            "tau": 70., "d2": pd.Series(False, index=origins),
            "summary": {"arm": "EXPLORE"},
        }
        for origin in origins:
            # These are synthetic anchored paths.  The fixture claims a mock
            # producer audit only to exercise the downstream contract.
            r1 = float(history.power.loc[origin])
            records.append({"origin": origin, "horizon": h, "r1": r1,
                            "q10": r1-5, "q50": r1, "q90": r1+5, "q95": r1+7})
    paths = pd.DataFrame(records)
    paths.attrs["causal_provenance"] = {
        "leakage_test": "passed", "input_cutoff_rule": "history.index <= origin",
        "future_perturbation_max_abs_difference": 0.0,
        "model_revision": "synthetic-pinned-revision",
    }
    view = SimpleNamespace(root=Path(__file__).resolve().parents[2], out=tmp_path,
                           history=history, contexts=contexts,
                           seal={"raw_sha256": "synthetic-seal"},
                           split_lock={"lock_sha256": "synthetic-split"})
    spec = {"id": "R1-residual-test", "adapter": "r1_residual", "arm": "EXPLORE",
            "seed": 42, "groups": (), "rolling_audit_sha256": "a"*64}
    return view, spec, paths


class _FakeLGB:
    fit_count = 0

    def __init__(self, **params):
        self.params = params
        self.best_iteration_ = 7

    def fit(self, x, y, **kwargs):
        type(self).fit_count += 1
        assert len(x) == 300 * 13 and len(kwargs["eval_set"][0][0]) == 40 * 13
        self.correction = float(np.median(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.correction)


def test_complete_cohort_checkpoint_and_upstream_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(
        LGBMRegressor=_FakeLGB, early_stopping=lambda *args, **kwargs: object()))
    _FakeLGB.fit_count = 0
    view, spec, paths = _fixture(tmp_path)
    frame, audit = adapter.run(view, spec, paths)
    assert len(frame) == 13 * (8+8)
    assert set(frame.role) == {"cal", "score"}
    assert audit["leakage_test"] == "passed"
    assert audit["cells"][0]["n_fit"] == 300 * 13
    assert audit["cells"][0]["n_stop"] == 40 * 13
    assert _FakeLGB.fit_count == 1
    again, replay = adapter.run(view, spec, paths)
    assert _FakeLGB.fit_count == 1 and replay["cells"][0]["cache_reused"]
    assert frame.equals(again)
    model_path = next((tmp_path / "models/r1_residual/R1-residual-test").glob("*.joblib"))
    with model_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(RuntimeError, match="checkpoint identity/hash mismatch"):
        adapter.run(view, spec, paths)


def test_path_and_role_guardrails(tmp_path):
    view, spec, paths = _fixture(tmp_path)
    bad = paths.iloc[1:].copy()
    with pytest.raises(ValueError, match="misses fit origins"):
        adapter.run(view, spec, bad)
    paths.loc[0, "horizon"] = 4.5
    with pytest.raises(ValueError, match="integer"):
        adapter.run(view, spec, paths)


def test_common_origin_embargo_and_future_feature_perturbation(tmp_path):
    view, spec, paths = _fixture(tmp_path)
    fit, stop, proof = adapter._common_roles(view.contexts, 0)
    assert fit.max() + 16 * adapter._SLOT < stop.min()
    assert stop.max() + 16 * adapter._SLOT < view.contexts[(16, 0)]["cal"].min()
    assert proof["common_fit_after_purge"] == 300
    indexed, _ = adapter._paths_index(paths, view.contexts)
    origin = view.contexts[(16, 0)]["score"][0]
    a = adapter._matrix(view.history, indexed, pd.DatetimeIndex([origin]), 16, 70., (), True)
    altered = view.history.copy()
    altered.loc[altered.index > origin, "power"] = 1e9
    b = adapter._matrix(altered, indexed, pd.DatetimeIndex([origin]), 16, 70., (), True)
    pd.testing.assert_frame_equal(a, b)
    # An observed error is usable only once its earlier h-step forecast has
    # matured at this origin.  A forecast issued at this origin is not used.
    indexed, _ = adapter._paths_index(paths, view.contexts)
    past = origin - 16 * adapter._SLOT
    path_error = indexed.r1.get((past, 16), np.nan)
    expected = view.history.power.loc[origin] - path_error
    actual = adapter._known_error(view.history, indexed, pd.DatetimeIndex([origin]), 16)[0]
    assert np.isclose(actual, expected, equal_nan=True)


def test_missing_or_false_upstream_audit_cannot_claim_passed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(
        LGBMRegressor=_FakeLGB, early_stopping=lambda *args, **kwargs: object()))
    view, spec, paths = _fixture(tmp_path)
    paths.attrs["causal_provenance"]["future_perturbation_max_abs_difference"] = True
    _, audit = adapter.run(view, spec, paths)
    assert audit["leakage_test"] == "requires_upstream_audit"
    assert audit["rolling_upstream_causal_manifest_verified"] is False


def test_interrupted_model_pair_is_preserved_and_refitted(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(
        LGBMRegressor=_FakeLGB, early_stopping=lambda *args, **kwargs: object()))
    _FakeLGB.fit_count = 0
    view, spec, paths = _fixture(tmp_path)
    adapter.run(view, spec, paths)
    model_path = next((tmp_path / "models/r1_residual/R1-residual-test").glob("*.joblib"))
    manifest = model_path.with_suffix(".json")
    original_bytes = model_path.read_bytes()
    manifest.unlink()
    _, audit = adapter.run(view, spec, paths)
    assert _FakeLGB.fit_count == 2
    assert audit["cells"][0]["cache_reused"] is False
    assert model_path.exists() and manifest.exists()
    preserved = list(model_path.parent.glob(f"{model_path.stem}.orphan-*{model_path.suffix}"))
    assert len(preserved) == 1 and preserved[0].read_bytes() == original_bytes


def test_stop_request_honored_only_after_durable_fold(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(
        LGBMRegressor=_FakeLGB, early_stopping=lambda *args, **kwargs: object()))
    view, spec, paths = _fixture(tmp_path)
    request = tmp_path / "logs/stop_requested.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="after durable fold 0 checkpoint"):
        adapter.run(view, spec, paths)
    model_path = next((tmp_path / "models/r1_residual/R1-residual-test").glob("*.joblib"))
    assert model_path.with_suffix(".json").exists()
