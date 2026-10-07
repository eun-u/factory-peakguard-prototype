"""Stage 4 entry must fail before legacy selection touches evidence or locks."""
from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_f import goal_confirm_gate as gate, wf_final
from phase_f.goal_protocol import TARGETS
from phase_f.registry import config_hash


def _prepared(tmp_path):
    return SimpleNamespace(
        root=tmp_path,
        out=tmp_path / "outputs/phase_f/walkforward_v2",
    )


def _policy_path(prepared):
    path = Path(prepared.root) / gate.POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_policy(prepared, record):
    _policy_path(prepared).write_text(json.dumps(record), encoding="utf-8")


def _resign(record):
    record["policy_sha256"] = config_hash(
        {key: value for key, value in record.items() if key != "policy_sha256"}
    )
    return record


def _forbid_legacy(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Stage 4 accessed the legacy selection path")

    for name in ("WFRegistry", "search_manifest", "_fit_frozen", "_sealed", "_read"):
        monkeypatch.setattr(wf_final, name, forbidden)
    return calls


def test_pending_policy_blocks_before_legacy_reads_or_lock_writes(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    record = gate.pending_policy()
    assert record["absolute_targets"] == TARGETS
    assert record["finalist_cap"] == 7
    assert record["confirm_transaction_count"] == 1
    assert record["holdout_read"] is False
    _write_policy(prepared, record)
    calls = _forbid_legacy(monkeypatch)
    policy_bytes = _policy_path(prepared).read_bytes()

    with pytest.raises(RuntimeError, match="integration is pending"):
        wf_final.run_final(prepared)

    assert calls == []
    assert _policy_path(prepared).read_bytes() == policy_bytes
    assert not prepared.out.exists()


@pytest.mark.parametrize("dangling", (True, False))
def test_policy_symlink_fails_closed_even_when_dangling(tmp_path, monkeypatch, dangling):
    prepared = _prepared(tmp_path)
    path = _policy_path(prepared)
    target = tmp_path / "policy_target.json"
    if not dangling:
        target.write_text(json.dumps(gate.pending_policy()), encoding="utf-8")
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError):
        # Some Windows hosts deny symlink creation; simulate lstat on the link.
        if not dangling:
            path.write_bytes(target.read_bytes())
        original_lstat = Path.lstat

        def simulated_lstat(self):
            if self == path:
                return SimpleNamespace(st_mode=stat.S_IFLNK)
            return original_lstat(self)

        monkeypatch.setattr(Path, "lstat", simulated_lstat)
    calls = _forbid_legacy(monkeypatch)

    with pytest.raises(RuntimeError, match="not a regular file; Stage 4 remains closed"):
        wf_final.run_final(prepared)

    assert calls == []
    assert not prepared.out.exists()


def test_policy_filesystem_error_fails_closed(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    path = _policy_path(prepared)
    original_lstat = Path.lstat

    def denied_lstat(self):
        if self == path:
            raise PermissionError("simulated denied policy lookup")
        return original_lstat(self)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    calls = _forbid_legacy(monkeypatch)
    with pytest.raises(RuntimeError, match="cannot be inspected; Stage 4 remains closed"):
        wf_final.run_final(prepared)
    assert calls == []
    assert not prepared.out.exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "invalid_json",
        "unsigned_target_change",
        "resigned_target_change",
        "resigned_budget_change",
        "resigned_holdout_read",
        "resigned_ready",
        "missing_field",
        "extra_field",
    ),
)
def test_malformed_or_unsupported_policy_fails_closed(tmp_path, monkeypatch, mutation):
    prepared = _prepared(tmp_path)
    path = _policy_path(prepared)
    if mutation == "invalid_json":
        path.write_text("{", encoding="utf-8")
    else:
        record = gate.pending_policy()
        if mutation in ("unsigned_target_change", "resigned_target_change"):
            record["absolute_targets"]["AUC_MAE"] = 99
        elif mutation == "resigned_budget_change":
            record["stage0_3_budget"]["stochastic_seeds"] = 1
        elif mutation == "resigned_holdout_read":
            record["holdout_read"] = True
        elif mutation == "resigned_ready":
            record["status"] = "ready"
        elif mutation == "missing_field":
            del record["confirm_transaction_count"]
        elif mutation == "extra_field":
            record["unexpected"] = "bypass"
        if mutation.startswith("resigned"):
            _resign(record)
        _write_policy(prepared, record)
    before = path.read_bytes()
    calls = _forbid_legacy(monkeypatch)

    with pytest.raises(RuntimeError, match="Stage 4 remains closed"):
        wf_final.run_final(prepared)

    assert calls == []
    assert path.read_bytes() == before
    assert not prepared.out.exists()


def test_absent_policy_keeps_legacy_stage4_entry(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    assert gate.require_stage4_entry(prepared) is None

    class LegacyReached(Exception):
        pass

    monkeypatch.setattr(wf_final, "WFRegistry", lambda root: object())

    def reached(prepared):
        raise LegacyReached

    monkeypatch.setattr(wf_final, "search_manifest", reached)
    with pytest.raises(LegacyReached):
        wf_final.run_final(prepared)
    assert not prepared.out.exists()
