from datetime import datetime, timezone
import pytest
from src.workflow import final_gate

CFG = {"freeze": {"not_before": "2026-10-01T18:00:00+09:00"}}


def test_final_gate_exact_korea_time():
    assert not final_gate(CFG, datetime(2026, 10, 1, 8, 59, 59, tzinfo=timezone.utc))
    assert final_gate(CFG, datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc))


def test_naive_clock_rejected():
    with pytest.raises(ValueError):
        final_gate(CFG, datetime(2026, 10, 1))


def test_cached_prediction_tamper_and_missing_model_rejected(tmp_path):
    from src.workflow import development_artifacts, verify_development_cache, json_write
    files = ["predictions/development_oof.csv", "tables/development_cv.csv",
             "logs/development_selection.json", "predictions/t2_development_oof.csv",
             "tables/t2_development_cv.csv", "models/development_h4.joblib"]
    for name in files:
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b"original")
    json_write(tmp_path/"logs/development_cache.json", {"fingerprint": {"code": "123"},
               "artifacts": development_artifacts(tmp_path)})
    assert verify_development_cache(tmp_path, {"code": "123"})
    prediction = tmp_path/files[0]
    prediction.write_bytes(b"tampered")
    assert not verify_development_cache(tmp_path, {"code": "123"})
    prediction.write_bytes(b"original")
    (tmp_path/files[-1]).unlink()
    assert not verify_development_cache(tmp_path, {"code": "123"})


def test_report_edit_does_not_unlock_or_invalidate_scientific_freeze(tmp_path):
    from src.workflow import model_code_digest
    (tmp_path/"models").mkdir()
    (tmp_path/"features.py").write_text("past_only = True")
    (tmp_path/"reporting.py").write_text("title = 'Draft'")
    original = model_code_digest(tmp_path)
    (tmp_path/"reporting.py").write_text("title = 'Final report'")
    assert model_code_digest(tmp_path) == original
    (tmp_path/"features.py").write_text("past_only = False")
    assert model_code_digest(tmp_path) != original


def test_new_scientific_module_invalidates_development_fingerprint(tmp_path):
    from src.workflow import fingerprint
    (tmp_path/"src/analysis").mkdir(parents=True)
    module = tmp_path/"src/analysis/early_warning.py"
    module.write_text("threshold = 0.5", encoding="utf-8")
    before = fingerprint(tmp_path, development_only=True)
    module.write_text("threshold = 0.7", encoding="utf-8")
    assert fingerprint(tmp_path, development_only=True) != before


def test_run_evidence_keeps_history_after_auto_marker(tmp_path):
    from src.workflow import record_run_evidence
    progress = tmp_path/"PROGRESS.md"
    history = "# 기록\n<!-- AUTO_EXECUTION_STATUS -->\n이전 상태\n## 수동 추가\n보존할 결정"
    progress.write_text(history, encoding="utf-8")
    record_run_evidence(tmp_path, tmp_path/"outputs", {"status": "completed_development", "steps": {}})
    assert progress.read_text(encoding="utf-8").startswith(history)
    assert progress.read_text(encoding="utf-8").count("보존할 결정") == 1
