"""Archive integrity checks that must hold before publishing a frozen bundle."""
import json
from pathlib import PureWindowsPath
import subprocess
import sys
import zipfile

import pytest

from src import packaging


def test_research_archive_contains_sources_and_checks_after_extraction(tmp_path):
    from research.tests.test_project import copy_project
    from research.project import load_project, snapshot

    source = tmp_path / 'source'
    source.mkdir()
    copy_project(source)
    info = packaging.build_package(source, {'freeze': {'not_before': '2026-10-01T09:00:00+00:00'}})
    extracted = tmp_path / 'extracted'
    with zipfile.ZipFile(source / info['path']) as archive:
        names = set(archive.namelist())
        assert set(snapshot(load_project(source))['sources']) <= names
        assert 'paper/manuscript.md' in names
        assert 'experiments/plans/R2-EX02.md' in names
        assert not any(name.startswith('data/raw/') for name in names)
        archive.extractall(extracted)
    result = subprocess.run([sys.executable, '-S', '-m', 'research', 'check'],
                            cwd=extracted, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frozen_archive_rejects_unverified_development_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(packaging, "_freeze", lambda root, cfg: {"status": "completed"})
    monkeypatch.setattr(packaging, "fingerprint", lambda *args, **kwargs: {})
    monkeypatch.setattr(packaging, "verify_development_cache", lambda *args: False)
    with pytest.raises(RuntimeError, match="verified development cache"):
        packaging.build_package(tmp_path, {})
    assert not (tmp_path/"submission/source").exists()


def test_blind_scan_detects_json_escaped_private_path(tmp_path):
    source = tmp_path/"status.json"
    fixture_path = str(PureWindowsPath("C:/")/"Users"/"example"/"private"/"report.md")
    source.write_text(json.dumps({"path": fixture_path}), encoding="utf-8")
    result = packaging._scan(tmp_path, [source])
    assert result["status"] == "review_required"
    assert result["findings"] == [{"file": "status.json", "issue": "private_path_or_credential_pattern"}]


def test_interrupted_freeze_cannot_be_exported_as_unlocked_development(tmp_path):
    record = tmp_path/"outputs/logs/freeze_record.json"
    record.parent.mkdir(parents=True)
    original = json.dumps({"status": "started", "test_access_reserved": True})
    record.write_text(original, encoding="utf-8")
    with pytest.raises(RuntimeError, match="one-time lock"):
        packaging.build_package(tmp_path, {})
    assert record.read_text(encoding="utf-8") == original
    assert not (tmp_path/"submission/source").exists()
