"""The holdout must remain sealed until a single, recorded model freeze."""

import hashlib
import json

import pandas as pd
import pytest

from src import finalize


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def freeze_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data/raw/synthetic.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic source\r\n")
    processed = tmp_path / "data/processed/power_15min.parquet"
    processed.parent.mkdir(parents=True)
    processed.write_bytes(b"synthetic processed data")
    output = tmp_path / "outputs"
    manifest = output / "logs/development_selection.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"selection": {}}', encoding="utf-8")
    cfg = {"source": "data/raw/synthetic.csv",
           "freeze": {"not_before": "2020-01-01T00:00:00+09:00"},
           "split": {"test_start_origin": "2021-01-01 01:00:00"},
           "horizons": [4]}
    return output, cfg, manifest


def _completed_record(output, cfg, manifest):
    files = {
        "model_sha256": ("models/frozen_final.joblib", b"synthetic frozen model"),
        "final_metrics_sha256": ("tables/final_test.csv", b"model,mae\nbaseline,1\n"),
        "holdout_prediction_sha256":
            ("predictions/final_test_predictions.csv",
             b"origin,target_time,model,y,pred\n2021-01-01 01:00,2021-01-01 02:00,baseline,1,1\n"),
        "t2_metrics_sha256": ("tables/t2_final_test.csv", b"model,mae\nt2,1\n"),
        "t2_predictions_sha256":
            ("predictions/t2_final_test_predictions.csv", b"origin,target_time,model,y,pred\n"),
    }
    record = {"status": "completed", "fingerprints": finalize._fingerprints(cfg, manifest),
              "test_start_origin": cfg["split"]["test_start_origin"]}
    for key, (relative, content) in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        record[key] = _sha(path)
    lock = output / "logs/freeze_record.json"
    lock.write_text(json.dumps(record), encoding="utf-8")
    return lock


def test_date_refusal_precedes_manifest_and_holdout_access(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"freeze": {"not_before": "2099-01-01T00:00:00+09:00"}}
    with pytest.raises(RuntimeError, match="Holdout is sealed"):
        finalize.freeze_and_evaluate(object(), cfg, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_completed_freeze_reuses_matching_files_without_training(freeze_workspace, monkeypatch):
    output, cfg, manifest = freeze_workspace
    _completed_record(output, cfg, manifest)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Cached freeze must not fit, score, or read the holdout")

    monkeypatch.setattr(finalize, "_as_index", forbidden)
    monkeypatch.setattr(finalize, "_prepare_horizon", forbidden)
    monkeypatch.setattr(finalize, "_prepare_t2", forbidden)
    monkeypatch.setattr(finalize, "evaluate_all", forbidden)
    result = finalize.freeze_and_evaluate(object(), cfg, output)
    assert result["cached"] is True
    assert len(result["predictions"]) == 1
    assert result["metrics"].iloc[0]["model"] == "baseline"


@pytest.mark.parametrize("relative", ["models/frozen_final.joblib",
                                      "predictions/final_test_predictions.csv"])
def test_completed_freeze_rejects_tampered_artifact_without_scoring(
        freeze_workspace, monkeypatch, relative):
    output, cfg, manifest = freeze_workspace
    _completed_record(output, cfg, manifest)
    with (output / relative).open("ab") as stream:
        stream.write(b"tampered")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Tampered freeze must not refit or rescore")

    monkeypatch.setattr(finalize, "_as_index", forbidden)
    with pytest.raises(RuntimeError, match="Freeze reservation already exists"):
        finalize.freeze_and_evaluate(object(), cfg, output)


def test_reservation_and_model_hash_precede_holdout_then_interruption_locks_retry(
        freeze_workspace, monkeypatch):
    output, cfg, _ = freeze_workspace
    index = pd.date_range("2021-01-01 00:00", periods=16, freq="15min")
    frame = pd.DataFrame({"power": range(len(index))}, index=index)
    boundary = pd.Timestamp(cfg["split"]["test_start_origin"])
    lock = output / "logs/freeze_record.json"

    def prepare_horizon(dev, split, horizon, *_args):
        assert split == boundary and horizon == 4
        assert dev.index.max() < boundary
        assert json.loads(lock.read_text(encoding="utf-8"))["status"] == "reserved"
        return {"synthetic": "frozen point and quantile bundle"}

    def prepare_t2(dev, split, *_args):
        assert split == boundary and dev.index.max() < boundary
        return {"synthetic": "frozen T2 model"}

    class HoldoutReached(Exception):
        pass

    def first_holdout_read(all_data, origins, horizon):
        record = json.loads(lock.read_text(encoding="utf-8"))
        model_file = output / "models/frozen_final.joblib"
        assert all_data is frame
        assert len(origins) > 0 and origins.min() >= boundary and horizon == 4
        assert record["status"] == "models_frozen"
        assert model_file.is_file() and record["model_sha256"] == _sha(model_file)
        raise HoldoutReached

    monkeypatch.setattr(finalize, "_prepare_horizon", prepare_horizon)
    monkeypatch.setattr(finalize, "_prepare_t2", prepare_t2)
    monkeypatch.setattr(finalize, "point_targets", first_holdout_read)
    with pytest.raises(HoldoutReached):
        finalize.freeze_and_evaluate(frame, cfg, output)
    assert json.loads(lock.read_text(encoding="utf-8"))["status"] == "models_frozen"
    with pytest.raises(RuntimeError, match="Freeze reservation already exists"):
        finalize.freeze_and_evaluate(frame, cfg, output)
