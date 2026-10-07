"""Runner-level seed aggregation and resumable identity contracts (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f import goal_r1, goal_r1_paths
from phase_f.goal_protocol import SEEDS
from phase_f.harness import make_frame
from phase_f.registry import config_hash, sha256, write_json
from phase_f.wf_evaluation import prediction_path


def _prepared(tmp_path):
    root = Path(__file__).resolve().parents[2]
    roles = {
        "fit": pd.date_range("2021-03-01", periods=5, freq="15min", name="origin"),
        "stop": pd.date_range("2021-04-01", periods=3, freq="15min", name="origin"),
        "cal": pd.date_range("2021-04-15", periods=2, freq="15min", name="origin"),
        "score": pd.date_range("2021-04-19", periods=2, freq="15min", name="origin"),
    }
    contexts = {}
    for h in range(4, 17):
        origins = pd.DatetimeIndex(sorted(set().union(*map(set, roles.values()))), name="origin")
        y = pd.Series(55 + np.arange(len(origins), dtype=float), index=origins)
        contexts[(h, 0)] = {
            **roles, "y": y,
            "target_time": pd.Series(origins + pd.Timedelta(minutes=h*15), index=origins),
            "x": pd.DataFrame({"slot7d": np.full(len(origins), 50.)}, index=origins),
            "tau": 62., "d2": pd.Series(False, index=origins),
            "summary": {"arm": "EXPLORE"},
        }
    view = SimpleNamespace(root=root, out=tmp_path, contexts=contexts,
                           split_lock={"lock_sha256": "synthetic-week-lock"})
    view.origins = lambda h, fold, role: contexts[(h, fold)][role]
    return view


def _paths(prepared):
    rows = []
    for (h, fold), c in prepared.contexts.items():
        for role in ("fit", "stop", "cal", "score"):
            for origin in c[role]:
                rows.append({"origin": origin, "horizon": h, "r1": 50.,
                             "q10": 45., "q50": 50., "q90": 55., "q95": 57.})
    return goal_r1_paths._sort_paths(pd.DataFrame(rows))


def _audit(paths, prepared):
    paths_sha = goal_r1_paths._hash_frame(paths)
    manifest = {"complete": True, "identity_hash": "synthetic-fixed-paths",
                "paths_sha256": paths_sha}
    manifest["manifest_sha256"] = config_hash(manifest)
    manifest_path = prepared.out / "logs/rolling_paths_manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "identity_hash": "synthetic-fixed-paths",
            "paths_sha256": paths_sha,
            "leakage_test": "passed", "input_cutoff_rule": "history.index <= origin",
            "future_perturbation_max_abs_difference": 0.0,
            "model_revision": "synthetic-fixed-snapshot"}


def _spec():
    return {"id": "FG-R1-runner-test", "adapter": "r1_residual", "groups": [],
            "model_params": {"objective": "regression_l1"}, "peak_weight": 1.0}


def _fake_seed(prepared, child, paths):
    seed = child["seed"]
    frames = []
    for (h, fold), c in prepared.contexts.items():
        for role in ("cal", "score"):
            origins = c[role]
            # Raw correction is deliberately unlike the applied correction.
            prediction = np.full(len(origins), 50. + seed / 1000)
            frames.append(make_frame(c, origins, h, fold, prediction, child["id"], role,
                                     r1=np.full(len(origins), 50.),
                                     correction=np.full(len(origins), 1000. + seed)))
    return pd.concat(frames, ignore_index=True), {"leakage_test": "passed", "seed": seed}


def _fake_score(prepared, spec, frame, audit):
    expected = 50 + np.mean(SEEDS) / 1000
    assert np.allclose(frame.pred, expected)
    assert "correction" not in frame
    assert np.allclose(frame.applied_correction, np.mean(SEEDS) / 1000)
    assert audit["n_seeds"] == 5
    return {"fields": {"AUC_MAE": 5., "AUC_PeakMAE": 11.,
                       "h4_MAE": 4., "h16_MAE": 6., "AUC_nMAE": .06},
            "targets_met": False}


def test_runner_means_five_distinct_seed_forecasts_and_resumes_without_fits(tmp_path, monkeypatch):
    from phase_f.models import r1_residual

    prepared = _prepared(tmp_path)
    paths = _paths(prepared)
    audit = _audit(paths, prepared)
    fitted = []

    def fake_run(view, child, rolling):
        fitted.append(child["seed"])
        assert child["rolling_audit_sha256"] == goal_r1.config_hash(audit)
        return _fake_seed(view, child, rolling)

    monkeypatch.setattr(goal_r1, "_sources", lambda root: {"unit-source": "stable"})
    monkeypatch.setattr(goal_r1, "initial_specs", lambda: [_spec()])
    monkeypatch.setattr(goal_r1, "score", _fake_score)
    monkeypatch.setattr(r1_residual, "run", fake_run)
    baselines = {"B5": "synthetic-b5", "R1": "synthetic-r1"}
    goal_r1.run_search(prepared, paths, audit, baselines)
    assert fitted == list(SEEDS) and len(set(fitted)) == 5
    mean = pd.read_parquet(prediction_path(prepared, _spec()["id"]))
    assert mean.n_seeds.eq(5).all()
    assert mean.model.eq(_spec()["id"]).all()
    assert mean.loc[mean.role.eq("score"), "arm"].eq("EXPLORE").all()
    assert len(mean) == 13 * 4
    goal_r1.run_search(prepared, paths, audit, baselines)
    assert fitted == list(SEEDS)
    assert sha256(prediction_path(prepared, _spec()["id"])) == json.loads(
        prediction_path(prepared, _spec()["id"]).with_suffix(".json").read_text(encoding="utf-8"))["sha256"]


def test_runner_rejects_changed_source_or_rolling_paths_on_resume(tmp_path, monkeypatch):
    from phase_f.models import r1_residual

    prepared = _prepared(tmp_path)
    paths = _paths(prepared)
    audit = _audit(paths, prepared)
    monkeypatch.setattr(goal_r1, "_sources", lambda root: {"unit-source": "stable"})
    monkeypatch.setattr(goal_r1, "initial_specs", lambda: [_spec()])
    monkeypatch.setattr(goal_r1, "score", _fake_score)
    monkeypatch.setattr(r1_residual, "run", _fake_seed)
    baselines = {"B5": "synthetic-b5", "R1": "synthetic-r1"}
    goal_r1.run_search(prepared, paths, audit, baselines)
    changed = paths.copy()
    changed.loc[0, "r1"] += 1.
    with pytest.raises((ValueError, RuntimeError), match="rolling|path|identity|digest"):
        goal_r1.run_search(prepared, changed, audit, baselines)
    monkeypatch.setattr(goal_r1, "_sources", lambda root: {"unit-source": "changed"})
    with pytest.raises(RuntimeError, match="locked artifact"):
        goal_r1.run_search(prepared, paths, audit, baselines)
