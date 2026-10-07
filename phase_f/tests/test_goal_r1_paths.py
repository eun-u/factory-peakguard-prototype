"""Small causal and transaction tests; the pinned GPU model is not loaded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phase_f import goal_r1_paths as r1


class Prepared:
    def __init__(self, root, *, n_fit=2):
        self.root = root
        self.seal = {"files": {"synthetic": "sha"}}
        self.split_lock = {"lock_sha256": "synthetic-weekly-lock"}
        index = pd.date_range("2021-04-01", "2021-04-21", freq="15min")
        self.history = pd.DataFrame({"power": np.arange(len(index), dtype=float)}, index=index)
        fit = pd.date_range("2021-04-16", periods=n_fit, freq="15min")
        stop = pd.DatetimeIndex([pd.Timestamp("2021-04-17 00:00")])
        cal = pd.DatetimeIndex([pd.Timestamp("2021-04-18 00:00")])
        score = pd.DatetimeIndex([pd.Timestamp("2021-04-19 00:00")])
        all_origins = fit.union(stop).union(cal).union(score)
        self.contexts = {(4, 1): {"fit": fit, "stop": stop, "cal": cal,
                                  "score": score,
                                  "target_time": pd.Series(all_origins + pd.Timedelta(hours=1),
                                                           index=all_origins),
                                  "summary": {"arm": "EXPLORE"}}}

    def origins(self, h, fold, role):
        return self.contexts[(h, fold)][role]


class FakePipe:
    model_context_length = 8192

    def __init__(self):
        self.calls = 0

    def predict_quantiles(self, inputs, **_):
        self.calls += 1
        array = np.empty((len(inputs), 1, 96, len(r1.QUANTILES)), dtype="float32")
        for i, inputs_i in enumerate(inputs):
            for horizon in range(96):
                array[i, 0, horizon] = float(inputs_i[-1]) + horizon + np.asarray(
                    r1.QUANTILES, dtype="float32")
        return [torch.from_numpy(array[i]) for i in range(len(inputs))], None


class WrongPipe(FakePipe):
    def predict_quantiles(self, inputs, **kwargs):
        values, other = super().predict_quantiles(inputs, **kwargs)
        return [value + 1 for value in values], other


def _anchor(prepared, *, poison_confirm=False):
    rows = []
    for role in ("cal", "score"):
        for origin in prepared.origins(4, 1, role):
            base = float(prepared.history.loc[origin, "power"]) + 3
            rows.append({"horizon": 4, "fold": 1, "origin": origin, "role": role,
                         "pred": base + .5, "q10": base + .1, "q50": base + .5,
                         "q90": base + .9, "q95": base + .95, "y": -999})
    if poison_confirm:
        rows.append({"horizon": 4, "fold": 2, "origin": pd.Timestamp("2021-04-20"),
                     "role": "score", "pred": "unread-confirm-pred", "q10": None,
                     "q50": None, "q90": None, "q95": None, "y": "unread-confirm-y"})
    return pd.DataFrame(rows)


def _fake_source(prepared, anchors, revision):
    assert revision == "a" * 40
    return {"prediction_file_sha256": "synthetic-existing-explore",
            "anchor_prediction_sha256": r1._hash_frame(anchors),
            "anchor_original_future_perturbation_max_abs_difference": 0.0,
            "anchor_original_fresh_repeat_max_abs_difference": 0.0}


def _fake_snapshot(prepared):
    return Path(prepared.root), {"model_id": "amazon/chronos-2", "revision": "a" * 40,
                                 "snapshot_files": {"synthetic": "sha"}}


def test_selected_anchor_and_causal_fit_stop_paths_are_reused(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    prepared = Prepared(root)
    # Existing observed-series gaps are NaN; identical causal windows still
    # have to pass the actual fitted-pipeline future perturbation check.
    prepared.history.loc[pd.Timestamp("2021-04-15 23:45"), "power"] = np.nan
    anchor = _anchor(prepared, poison_confirm=True)
    pipe = FakePipe()
    monkeypatch.setattr(r1, "_snapshot", _fake_snapshot)
    monkeypatch.setattr(r1, "_verify_anchor_source", _fake_source)
    monkeypatch.setattr(r1, "_load_pipeline", lambda *_: pipe)
    paths, audit = r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")
    assert list(paths.columns) == list(r1.PATH_COLUMNS)
    assert len(paths) == 5 and not paths.duplicated(["origin", "horizon"]).any()
    assert not {"y", "tau", "target_time", "role", "fold"}.intersection(paths.columns)
    assert audit["leakage_test"] == "passed"
    assert audit["input_cutoff_rule"] == "history.index <= origin"
    assert audit["future_perturbation_max_abs_difference"] == 0
    assert audit["anchor_recipe_check"]["status"] == "passed"
    assert audit["anchor_recipe_check"]["compared_quantiles"] == 8
    assert audit["anchor_rows"] == 2 and audit["inferred_rows"] == 3
    assert audit["local_training"] is False
    assert paths.attrs["causal_provenance"] == audit
    calls = pipe.calls
    (tmp_path / "r1_paths/part_999999.parquet.tmp").write_bytes(b"interrupted checkpoint")
    again, second_audit = r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")
    pd.testing.assert_frame_equal(paths, again)
    assert pipe.calls == calls and second_audit["inferred_origins"] == 3
    assert second_audit == audit
    assert len(list((tmp_path / "r1_paths").glob("part_999999.parquet.tmp.orphan-*"))) == 1
    state = json.loads((tmp_path / "r1_paths/manifest.json").read_text(encoding="utf-8"))
    assert state["complete"] and len(state["parts"]) == 1


def test_duplicate_anchor_must_be_identical_before_cache(tmp_path, monkeypatch):
    prepared = Prepared(Path(__file__).resolve().parents[2])
    anchor = _anchor(prepared)
    altered = anchor.iloc[[0]].copy()
    altered.loc[:, "q50"] += 1
    with pytest.raises(ValueError, match="median differs|Duplicate R1 anchor predictions disagree"):
        r1.load_or_build(prepared, tmp_path, anchor=pd.concat([anchor, altered], ignore_index=True),
                         device="cpu")
    assert not (tmp_path / "r1_paths").exists()


def test_corrupt_chunk_is_preserved_and_recomputed(tmp_path, monkeypatch):
    prepared = Prepared(Path(__file__).resolve().parents[2])
    anchor = _anchor(prepared)
    pipe = FakePipe()
    monkeypatch.setattr(r1, "_snapshot", _fake_snapshot)
    monkeypatch.setattr(r1, "_verify_anchor_source", _fake_source)
    monkeypatch.setattr(r1, "_load_pipeline", lambda *_: pipe)
    original, _ = r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")
    part = tmp_path / "r1_paths/part_000000.parquet"
    with part.open("ab") as stream:
        stream.write(b"tampered")
    restored, audit = r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")
    pd.testing.assert_frame_equal(original, restored)
    assert audit["inferred_origins"] == 3
    assert len(list(part.parent.glob("part_000000.parquet.corrupt-*"))) == 1


def test_context_rejects_confirm_score_before_prediction_columns_are_read():
    prepared = Prepared(Path(__file__).resolve().parents[2])
    prepared.contexts[(4, 1)]["target_time"].loc[prepared.contexts[(4, 1)]["score"]] = pd.Timestamp("2021-04-26")
    with pytest.raises(ValueError, match="CONFIRM"):
        r1._required(prepared)


def test_runtime_fingerprint_change_rejects_cache_resume(tmp_path, monkeypatch):
    prepared = Prepared(Path(__file__).resolve().parents[2])
    anchor = _anchor(prepared)
    pipe = FakePipe()
    monkeypatch.setattr(r1, "_snapshot", _fake_snapshot)
    monkeypatch.setattr(r1, "_verify_anchor_source", _fake_source)
    monkeypatch.setattr(r1, "_load_pipeline", lambda *_: pipe)
    r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")
    original = r1._execution_fingerprint("cpu")
    monkeypatch.setattr(r1, "_execution_fingerprint",
                        lambda device: {**original, "torch_version": "different-runtime"})
    with pytest.raises(RuntimeError, match="identity changed"):
        r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")


def test_wrong_current_pipeline_rejected_before_missing_inference(tmp_path, monkeypatch):
    prepared = Prepared(Path(__file__).resolve().parents[2])
    anchor = _anchor(prepared)
    pipe = WrongPipe()
    monkeypatch.setattr(r1, "_snapshot", _fake_snapshot)
    monkeypatch.setattr(r1, "_verify_anchor_source", _fake_source)
    monkeypatch.setattr(r1, "_load_pipeline", lambda *_: pipe)
    with pytest.raises(RuntimeError, match="disagrees with old R1 anchor"):
        r1.load_or_build(prepared, tmp_path, anchor=anchor, device="cpu")
    state = json.loads((tmp_path / "r1_paths/manifest.json").read_text(encoding="utf-8"))
    assert state["parts"] == []
    assert state["anchor_recipe_check"] is None


def test_chronos_list_preserves_task_axis_and_rejects_bad_batch():
    values = [torch.zeros((1, 96, len(r1.QUANTILES))),
              torch.ones((1, 96, len(r1.QUANTILES)))]
    result = r1._quantile_array(values, 2)
    assert result.shape == (2, 1, 96, len(r1.QUANTILES))
    assert result[0, 0, 0, 0] == 0 and result[1, 0, 0, 0] == 1
    with pytest.raises(ValueError, match="list length"):
        r1._quantile_array(values, 1)
