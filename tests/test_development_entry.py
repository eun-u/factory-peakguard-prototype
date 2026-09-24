"""Entry point checks must stop before touching sealed target fields."""

import json

import pandas as pd
import pytest
import yaml

import run_all


def _workspace(tmp_path, monkeypatch):
    (tmp_path/"configs").mkdir()
    cfg = {"source": "data/raw/forbidden.csv", "output_dir": "outputs",
           "split": {"test_start_origin": "2021-08-09 09:45:00"},
           "freeze": {"not_before": "2020-01-01T00:00:00+09:00",
                      "human_approval_required": False}}
    (tmp_path/"configs/default.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(run_all, "ROOT", tmp_path)
    monkeypatch.setattr(run_all, "check_evidence", lambda *_: {})
    monkeypatch.setattr(run_all, "fingerprint", lambda *_a, **_k: {"scientific": "v1"})
    return cfg


@pytest.mark.parametrize("argv", [[], ["--development-only"], ["--only", "development"]])
def test_development_entry_uses_sealed_loader_only(tmp_path, monkeypatch, argv):
    _workspace(tmp_path, monkeypatch)
    import src.data
    import src.session_data
    import src.training
    import src.workflow

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A full-data or final-evaluation path was entered")

    monkeypatch.setattr(src.data, "load_power_data", forbidden)
    monkeypatch.setattr(src.training, "freeze_and_evaluate", forbidden)
    history = pd.DataFrame({"power": [1.0]},
                           index=pd.DatetimeIndex(["2021-08-09 09:30"], name="ts_end"))
    monkeypatch.setattr(src.session_data, "load_development_history", lambda *_: history)
    seen = {}

    def fit(frame, *_args):
        assert frame.index.max() < pd.Timestamp("2021-08-09 09:45")
        seen["fit"] = True
        return {"predictions": pd.DataFrame({"origin": [history.index[0]]}), "paths": {}}

    monkeypatch.setattr(src.training, "run_development", fit)
    monkeypatch.setattr(src.training, "run_t2_development", lambda *_a: seen.setdefault("t2", True))
    monkeypatch.setattr(run_all, "verify_development_cache", lambda *_a: False)
    monkeypatch.setattr(run_all, "development_artifacts", lambda *_a: {"synthetic": "sha"})
    monkeypatch.setattr(src.workflow, "record_run_evidence", lambda *_a: None)
    run_all.main(argv)
    assert seen == {"fit": True, "t2": True}
    status = json.loads((tmp_path/"outputs/logs/run_status.json").read_text(encoding="utf-8"))
    assert list(status["steps"]) == ["development"]
    assert status["holdout"] == "locked"


def test_dry_run_and_explicit_full_step_stop_without_data_loader(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    import src.data
    monkeypatch.setattr(src.data, "load_power_data",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("full loader called")))
    monkeypatch.setattr(run_all, "freeze_readiness", lambda *_a: {"status": "ready_for_human_approval",
                                                                    "holdout_read": False})
    monkeypatch.setattr(run_all, "require_freeze_approval",
                        lambda *_a: (_ for _ in ()).throw(RuntimeError("approval pending")))
    monkeypatch.setattr(run_all, "final_gate", lambda *_a: True)
    result = run_all.main(["--dry-run-freeze"])
    assert result["human_approval"] == "pending_or_invalid"
    assert result["date_gate_open"] is True
    with pytest.raises(RuntimeError, match="approval pending"):
        run_all.main(["--only", "data"])
    assert not (tmp_path/"outputs/tables/final_test.csv").exists()
