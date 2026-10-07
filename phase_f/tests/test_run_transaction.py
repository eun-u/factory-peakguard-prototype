"""Fault injection for a resumable 39-cell Phase F experiment transaction.

The fixture contains synthetic rows only. No plant history or model is loaded.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phase_f import metrics, registry, run
from phase_f.registry import Registry, sha256, write_json


def _score_rows(model: str) -> pd.DataFrame:
    rows = []
    for horizon in range(4, 17):
        for fold in range(3):
            origin = pd.Timestamp("2021-06-01") + pd.Timedelta(days=fold, hours=horizon)
            rows.append({"model": model, "horizon": horizon, "fold": fold,
                         "origin": origin, "target_time": origin + pd.Timedelta(minutes=15*horizon),
                         "y": 100.0 + fold, "pred": 99.0 + fold,
                         "tau": 120.0, "d2": fold == 2, "role": "score", "arm": "EXPLORE",
                         "train_seconds": 0.25, "inference_seconds": 0.01})
    return pd.DataFrame(rows)


def test_interrupt_after_summary_does_not_commit_completion_and_retry_reuses_prediction(tmp_path, monkeypatch):
    out = tmp_path / "outputs" / "phase_f"
    (out / "logs").mkdir(parents=True)
    prepared = SimpleNamespace(root=tmp_path, out=out,
                               validate_predictions=lambda frame: frame)
    monkeypatch.setattr(registry, "code_identity", lambda root: {
        "commit": "synthetic", "source_digest": "synthetic", "source_hashes": {}})
    trial = {"id": "F3-1-synthetic-t00000", "family": "F3", "tier": 2,
             "adapter": "regression", "kind": "ridge", "groups": [], "target": "direct"}
    frame = _score_rows(trial["id"])
    cells = [{"horizon": horizon, "fold": fold, "stop_MAE": 2.0}
             for horizon in range(4, 17) for fold in range(3)]
    fits = []
    summaries = []

    def fit_stub(p, spec):
        fits.append(spec["id"])
        if len(fits) > 1:
            raise AssertionError("Retry must read the sealed prediction cache")
        return frame.copy(), {"cells": cells, "leakage_test": "passed"}

    def summary_stub(p, spec, prediction, reg):
        summaries.append(spec["id"])
        assert len(prediction) == 39
        dest = p.out / "tables" / spec["id"]
        write_json(dest / "explore_manifest.json", {
            "prediction_sha256": sha256(p.out / "predictions" / f"{spec['id']}.parquet"),
            "registry_metrics": {"explore_AUC_MAE": 1.0}})
        reg.update(spec["id"], explore_AUC_MAE=1.0)
        return {"explore_AUC_MAE": 1.0}

    monkeypatch.setattr(run, "run_adapter", fit_stub)
    monkeypatch.setattr(run, "summarize_experiment", summary_stub)
    original_cost = metrics._cost
    cost_calls = []

    def interrupt_once(part, column):
        cost_calls.append(column)
        if len(cost_calls) == 1:
            raise KeyboardInterrupt("synthetic interruption after EXPLORE summary")
        return original_cost(part, column)

    monkeypatch.setattr(metrics, "_cost", interrupt_once)
    experiments = Registry(tmp_path)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        run.execute(prepared, trial, experiments)

    prediction_path = out / "predictions" / f"{trial['id']}.parquet"
    audit_path = out / "logs" / f"{trial['id']}_audit.json"
    manifest_path = out / "tables" / trial["id"] / "explore_manifest.json"
    assert prediction_path.is_file() and audit_path.is_file() and manifest_path.is_file()
    prediction_digest = sha256(prediction_path)
    audit_digest = sha256(audit_path)
    assert json.loads(audit_path.read_text(encoding="utf-8"))["prediction_sha256"] == prediction_digest
    interrupted = experiments.read(trial["id"])
    assert interrupted["status"] == "run"
    assert "stop_cells" not in interrupted and "stop_MAE" not in interrupted
    assert len(fits) == 1 and len(summaries) == 1

    completed = run.execute(prepared, trial, experiments, retry=True)
    assert completed["status"] == "completed"
    assert completed["stop_cells"] == 39
    assert np.isclose(completed["stop_MAE"], 2.0)
    assert np.isclose(completed["train_sec"], 39 * 0.25)
    assert np.isclose(completed["infer_sec"], 39 * 0.01)
    assert sha256(prediction_path) == prediction_digest and sha256(audit_path) == audit_digest
    assert len(fits) == 1 and len(summaries) == 2
