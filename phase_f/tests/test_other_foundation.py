"""Phase F F6-5 causal input and fixed-cohort adapter checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase_f.models import other_foundation as f6
from phase_f.models import other_foundation_worker as worker


def test_causal_context_is_unchanged_by_all_future_values():
    index = pd.date_range("2021-01-01", periods=40, freq="15min")
    history = pd.DataFrame({"power": np.arange(40, dtype=float)}, index=index)
    history.loc[index[5], "power"] = np.nan
    origin = index[12]
    original, gaps = f6.causal_context(history, origin, 16)
    altered = history.copy()
    altered.loc[altered.index > origin, "power"] = 999999.
    changed, _ = f6.causal_context(altered, origin, 16)
    np.testing.assert_array_equal(original, changed)
    assert gaps == 1
    assert original[-1] == 12
    assert original[5 + 3] == 4  # left padding shifts original missing slot


def test_input_transaction_rejects_changed_values(tmp_path):
    path = tmp_path / "causal_inputs.npz"
    origins = pd.date_range("2021-01-01", periods=2, freq="15min")
    inputs = np.arange(8, dtype=np.float32).reshape(2, 4)
    digest = f6._write_request(path, inputs, origins, inputs[0])
    assert f6._write_request(path, inputs, origins, inputs[0]) == digest
    changed = inputs.copy()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="changed"):
        f6._write_request(path, changed, origins, changed[0])


@pytest.mark.parametrize("missing", ["meta", "npz"])
def test_input_transaction_recovers_missing_peer_without_losing_bytes(tmp_path, missing):
    path = tmp_path / "causal_inputs.npz"
    meta = path.with_suffix(".json")
    origins = pd.date_range("2021-01-01", periods=2, freq="15min")
    inputs = np.arange(8, dtype=np.float32).reshape(2, 4)
    original_digest = f6._write_request(path, inputs, origins, inputs[0])
    stranded = path if missing == "meta" else meta
    removed = meta if missing == "meta" else path
    removed.unlink()
    stranded_digest = f6.sha256(stranded)
    changed = inputs.copy()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="changed"):
        f6._write_request(path, changed, origins, changed[0])
    assert f6.sha256(stranded) == stranded_digest
    assert not list(tmp_path.glob("*.orphan-*"))

    resumed_digest = f6._write_request(path, inputs, origins, inputs[0])
    orphans = list(tmp_path.glob(f"{stranded.stem}.orphan-*{stranded.suffix}"))
    assert len(orphans) == 1
    assert stranded_digest in orphans[0].name
    assert f6.sha256(orphans[0]) == stranded_digest
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["context"], inputs)
        np.testing.assert_array_equal(data["origin_ns"], origins.asi8)
        np.testing.assert_array_equal(data["probe"], inputs[0])
    assert json.loads(meta.read_text(encoding="utf-8"))["sha256"] == resumed_digest
    assert resumed_digest == original_digest
    npz_mtime = path.stat().st_mtime_ns
    meta_mtime = meta.stat().st_mtime_ns
    assert f6._write_request(path, inputs, origins, inputs[0]) == resumed_digest
    assert (path.stat().st_mtime_ns, meta.stat().st_mtime_ns) == (npz_mtime, meta_mtime)
    assert len(list(tmp_path.glob(f"{stranded.stem}.orphan-*{stranded.suffix}"))) == 1
    with pytest.raises(RuntimeError, match="changed"):
        f6._write_request(path, changed, origins, changed[0])


def test_input_transaction_rejects_complete_checksum_mismatch(tmp_path):
    path = tmp_path / "causal_inputs.npz"
    origins = pd.date_range("2021-01-01", periods=2, freq="15min")
    inputs = np.arange(8, dtype=np.float32).reshape(2, 4)
    f6._write_request(path, inputs, origins, inputs[0])
    meta = path.with_suffix(".json")
    saved = json.loads(meta.read_text(encoding="utf-8"))
    saved["sha256"] = "0" * 64
    meta.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        f6._write_request(path, inputs, origins, inputs[0])
    assert not list(tmp_path.glob("*.orphan-*"))


def test_worker_recovers_unregistered_chunk_and_preserves_registered_integrity(tmp_path, monkeypatch):
    request = tmp_path / "causal_inputs.npz"
    contexts = np.arange(24, dtype=np.float32).reshape(3, 8)
    np.savez_compressed(request, context=contexts, probe=contexts[0])
    output = tmp_path / "worker"
    output.mkdir()
    chunk = output / "chunk_000000_000002.npz"
    np.savez_compressed(chunk, quantiles=np.full((2, 16, 9), 999, dtype=np.float32),
                        mean=np.full((2, 16), 999, dtype=np.float32), levels=worker.LEVELS)
    stranded_digest = worker._sha256(chunk)

    def fake_forecast(kind, model, values):
        quantiles = np.broadcast_to(values[:, -1, None, None], (len(values), 16, 9)).copy()
        mean = np.broadcast_to(values[:, -1, None], (len(values), 16)).copy()
        return quantiles, mean

    monkeypatch.setattr(worker, "_load", lambda *_: object())
    monkeypatch.setattr(worker, "_forecast", fake_forecast)
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["worker", "--kind", "chronos_bolt",
                                    "--snapshot", str(tmp_path), "--request", str(request),
                                    "--output", str(output), "--cache-id", "synthetic-only",
                                    "--batch-size", "2"])
    worker.main()
    orphans = list(output.glob("chunk_000000_000002.orphan-*.npz"))
    assert len(orphans) == 1
    assert stranded_digest in orphans[0].name
    assert worker._sha256(orphans[0]) == stranded_digest
    with np.load(chunk, allow_pickle=False) as data:
        assert np.max(data["quantiles"]) < 999
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [(row["start"], row["stop"]) for row in manifest["chunks"]] == [(0, 2), (2, 3)]
    before = (worker._sha256(chunk), worker._sha256(manifest_path))
    monkeypatch.setattr(worker, "_load", lambda *_: pytest.fail("complete worker must use cache"))
    worker.main()
    assert (worker._sha256(chunk), worker._sha256(manifest_path)) == before
    chunk.write_bytes(b"changed registered chunk")
    with pytest.raises(RuntimeError, match="Prior worker chunk failed integrity check"):
        worker.main()


def test_worker_recovers_unregistered_later_chunk_without_rewriting_registered_one(tmp_path, monkeypatch):
    request = tmp_path / "causal_inputs.npz"
    contexts = np.arange(24, dtype=np.float32).reshape(3, 8)
    np.savez_compressed(request, context=contexts, probe=contexts[0])
    output = tmp_path / "worker"
    output.mkdir()
    first = output / "chunk_000000_000002.npz"
    later = output / "chunk_000002_000003.npz"
    np.savez_compressed(first, quantiles=np.ones((2, 16, 9), dtype=np.float32),
                        mean=np.ones((2, 16), dtype=np.float32), levels=worker.LEVELS)
    np.savez_compressed(later, quantiles=np.full((1, 16, 9), 999, dtype=np.float32),
                        mean=np.full((1, 16), 999, dtype=np.float32), levels=worker.LEVELS)
    first_digest = worker._sha256(first)
    later_digest = worker._sha256(later)
    worker._write_json(output / "manifest.json", {
        "cache_id": "synthetic-only", "request_sha256": worker._sha256(request),
        "chunks": [{"file": first.name, "start": 0, "stop": 2,
                    "sha256": first_digest, "forecast_seconds": 1.0}],
        "future_perturbation_max_abs_difference": 0.0,
    })
    monkeypatch.setattr(worker, "_load", lambda *_: object())
    monkeypatch.setattr(worker, "_forecast", lambda kind, model, values: (
        np.full((len(values), 16, 9), 2, dtype=np.float32),
        np.full((len(values), 16), 2, dtype=np.float32)))
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["worker", "--kind", "chronos_bolt",
                                    "--snapshot", str(tmp_path), "--request", str(request),
                                    "--output", str(output), "--cache-id", "synthetic-only",
                                    "--batch-size", "2"])
    worker.main()
    assert worker._sha256(first) == first_digest
    orphan = next(output.glob("chunk_000002_000003.orphan-*.npz"))
    assert later_digest in orphan.name and worker._sha256(orphan) == later_digest
    with np.load(later, allow_pickle=False) as data:
        assert np.all(data["quantiles"] == 2)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [(row["start"], row["stop"]) for row in manifest["chunks"]] == [(0, 2), (2, 3)]


def test_peak_probability_caps_native_tails():
    quantiles = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=float)
    assert f6._peak_probability(quantiles, 0).item() == pytest.approx(.9)
    assert f6._peak_probability(quantiles, 5).item() == pytest.approx(.5)
    assert f6._peak_probability(quantiles, 10).item() == pytest.approx(.1)


class SyntheticPrepared:
    def __init__(self, root: Path):
        self.root = root
        self.out = root / "outputs" / "phase_f"
        self.out.mkdir(parents=True)
        self.split_lock = {"lock_sha256": "synthetic-only"}
        index = pd.date_range("2021-01-01", periods=80, freq="15min")
        self.history = pd.DataFrame({"power": 5 + np.arange(80, dtype=float)}, index=index)
        fit = index[20:25]
        cal = index[30:31]
        score = index[40:42]
        self.contexts = {}
        for horizon in (4, 16):
            target = pd.Series(index.shift(horizon, freq="15min"), index=index)
            self.contexts[(horizon, 0)] = {
                "fit": fit, "cal": cal, "score": score,
                "target_time": target,
                "y": pd.Series(self.history.power.to_numpy() + horizon, index=index),
                "x": pd.DataFrame({"slot7d": self.history.power.to_numpy() - 1}, index=index),
                "tau": 20., "d2": pd.Series(False, index=index),
            }

    def origins(self, horizon, fold, role):
        return self.contexts[(horizon, fold)][role]


def test_adapter_reuses_one_path_across_horizons(tmp_path, monkeypatch):
    prepared = SyntheticPrepared(tmp_path)
    dependencies = prepared.out / "logs" / "other_foundation_dependencies.json"
    dependencies.parent.mkdir(parents=True)
    dependencies.write_text('{"synthetic": true}', encoding="utf-8")
    spec = {"id": "F6-5-synthetic", "kind": "chronos_bolt", "context_length": 16,
            "prediction_length": 16, "point": "median"}
    monkeypatch.setattr(f6, "_snapshot", lambda *_: (tmp_path, {
        "revision": "0" * 40, "snapshot_hash": "test", "files_sha256": {},
        "license": "apache-2.0", "commercially_eligible": True}))
    def fake_worker(prepared, spec, snapshot, request, cache_id, site):
        with np.load(request, allow_pickle=False) as data:
            n = len(data["context"])
        q = np.broadcast_to(np.arange(1, 10, dtype=np.float32), (n, 16, 9)).copy()
        worker = prepared.out / "models" / spec["id"] / "worker"
        worker.mkdir()
        (worker / "manifest.json").write_text(json.dumps({
            "chunks": [{"start": 0, "stop": n, "forecast_seconds": float(n)}],
            "future_perturbation_max_abs_difference": 0.0}), encoding="utf-8")
        return q, np.full((n, 16), np.nan, dtype=np.float32), f6.LEVELS.tolist()
    monkeypatch.setattr(f6, "_worker", fake_worker)
    frame, audit = f6.run(prepared, spec)
    assert len(frame) == 6
    assert set(frame.horizon) == {4, 16}
    assert set(frame.pred) == {5.0}
    assert frame.inference_seconds.eq(1).all()
    assert frame.inference_seconds_run_id.nunique() == 3
    assert audit["future_perturbation_max_abs_difference"] == 0
    assert audit["holdout_read"] is False
