"""The final selection and CONFIRM transaction survive partial writes safely."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from phase_f import wf_final, wf_models
from phase_f.registry import config_hash, sha256, write_json


def _prepared(tmp_path):
    out = tmp_path / "outputs/phase_f/walkforward_v2"
    out.mkdir(parents=True)
    return SimpleNamespace(root=tmp_path, out=out)


def _audit_fields(prepared):
    fields = {}
    for field, name in wf_final._PREFLIGHT_AUDITS.items():
        path = prepared.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(field.encode())
        fields[field] = sha256(path)
    return fields


def test_finalist_views_reuse_original_optional_environments(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    expected = prepared.root / "outputs/phase_f/optional_envs"
    expected.mkdir(parents=True)
    links = []
    monkeypatch.setattr(wf_final, "_junction", lambda link, target: links.append((Path(link), Path(target))))
    monkeypatch.setattr(wf_final, "_make_junction", lambda link, target: links.append((Path(link), Path(target))))

    view = wf_final._final_view(prepared)
    assert (view.out / "optional_envs", expected) in links

    class FakePrepared:
        def __init__(self, root):
            self.root, self.out, self.contexts = root, prepared.out, {}

    from phase_f import harness
    monkeypatch.setattr(harness, "Prepared", FakePrepared)
    monkeypatch.setattr(wf_models, "_pc3_confirm_view", lambda pc3: SimpleNamespace(contexts={(4, 0): {}}))
    pc3 = wf_final._pc3_view(prepared)
    assert pc3.contexts == {(4, 0): {}}
    assert (pc3.out / "optional_envs", expected) in links


@pytest.mark.parametrize("orphan_side", ("parquet", "manifest"))
def test_partial_finalist_mean_preserved_and_refitted(tmp_path, monkeypatch, orphan_side):
    prepared = _prepared(tmp_path)
    spec = {"id": "F5-finalist", "adapter": "baseline"}
    path = wf_final.prediction_path(prepared, spec["id"])
    path.parent.mkdir(parents=True)
    metadata = path.with_suffix(".json")
    partial = path if orphan_side == "parquet" else metadata
    original = b"interrupted transaction bytes"
    partial.write_bytes(original)
    calls = []
    frame = pd.DataFrame({"pred": [42.0]})

    def fake_run(view, current, *, arm, n_seeds):
        calls.append((arm, n_seeds))
        return frame, {"n_seeds": 1}

    def fake_save(view, current, result, audit, arm):
        result.to_parquet(path, index=False)
        write_json(metadata, {"identity": "frozen", "sha256": sha256(path), "arm": arm, "audit": audit})

    monkeypatch.setattr(wf_models, "run", fake_run)
    monkeypatch.setattr(wf_final, "cache_identity", lambda view, current: "frozen")
    monkeypatch.setattr(wf_final, "save_mean", fake_save)
    actual, audit = wf_final._fit_frozen(prepared, spec, {})
    assert actual.equals(frame) and audit["n_seeds"] == 1
    assert calls == [("EXPLORE", 1)]
    assert partial.read_bytes() != original
    orphans = list(partial.parent.glob(f"{partial.stem}.orphan-*{partial.suffix}"))
    assert len(orphans) == 1 and orphans[0].read_bytes() == original
    assert sha256(path) == json.loads(metadata.read_text(encoding="utf-8"))["sha256"]


def test_complete_finalist_mean_identity_mismatch_fails_closed(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    spec = {"id": "F5-finalist", "adapter": "baseline"}
    path = wf_final.prediction_path(prepared, spec["id"])
    path.parent.mkdir(parents=True)
    pd.DataFrame({"pred": [1.]}).to_parquet(path)
    write_json(path.with_suffix(".json"), {"identity": "old", "sha256": sha256(path), "audit": {}})
    monkeypatch.setattr(wf_final, "cache_identity", lambda view, current: "new")
    monkeypatch.setattr(wf_models, "run", lambda *args, **kwargs: pytest.fail("complete cache was refitted"))
    with pytest.raises(ValueError, match="identity changed"):
        wf_final._fit_frozen(prepared, spec, {})


def test_locked_sources_block_reservation_before_confirm(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    lockpath = prepared.out / "logs/confirm_lock.json"
    wf_final._sealed(lockpath, {"source_hashes": {"phase_f/wf_final.py": "old"},
        "explore_artifacts": {}, "prediction_sha256": {}, "candidates": []})
    monkeypatch.setattr(wf_final, "_final_view", lambda p: p)
    monkeypatch.setattr(wf_final, "_source_hashes", lambda root: {"phase_f/wf_final.py": "new"})
    with pytest.raises(ValueError, match="source changed"):
        wf_final.run_final(prepared)
    assert not (prepared.out / "logs/confirm_reservation.json").exists()


def test_changed_reservation_stops_before_any_confirm_fit(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    wf_final._sealed(prepared.out / "logs/confirm_lock.json", {
        "source_hashes": {}, "explore_artifacts": {}, "prediction_sha256": {},
        "candidates": [], **_audit_fields(prepared)})
    wf_final._sealed(prepared.out / "logs/confirm_reservation.json", {
        "selection_lock_sha256": "a different selection"})
    monkeypatch.setattr(wf_final, "_source_hashes", lambda root: {})
    monkeypatch.setattr(wf_final, "_committed_selection", lambda *args: None)
    monkeypatch.setattr(wf_final, "_final_view", lambda p: p)
    monkeypatch.setattr(wf_final, "_pc3_view", lambda p: pytest.fail("CONFIRM fit was reached"))
    with pytest.raises(ValueError, match="Reserved CONFIRM candidate mismatch"):
        wf_final.run_final(prepared)


def test_changed_preflight_audit_blocks_confirm_reservation(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    wf_final._sealed(prepared.out / "logs/confirm_lock.json", {
        "source_hashes": {}, "explore_artifacts": {}, "prediction_sha256": {},
        "candidates": [], **_audit_fields(prepared)})
    monkeypatch.setattr(wf_final, "_source_hashes", lambda root: {})
    monkeypatch.setattr(wf_final, "_committed_selection", lambda *args: None)
    monkeypatch.setattr(wf_final, "_final_view", lambda p: p)
    audit = prepared.root / wf_final._PREFLIGHT_AUDITS["alias_calendar_sha256"]
    audit.write_bytes(b"changed calendar audit")
    with pytest.raises(ValueError, match="Pre-CONFIRM audit changed"):
        wf_final.run_final(prepared)
    assert not (prepared.out / "logs/confirm_reservation.json").exists()


def test_completion_seals_confirm_forecasts_and_finalist_explore_seeds(tmp_path):
    prepared = _prepared(tmp_path)
    view = SimpleNamespace(out=prepared.out / "finalists_v2")
    pc3 = SimpleNamespace(out=prepared.out / "pc3_finalists")
    names = (
        "finalists_v2/predictions/CONFIRM/F5.parquet",
        "finalists_v2/predictions/CONFIRM/F5.json",
        "finalists_v2/predictions/seeds/CONFIRM/F5/seed_1.parquet",
        "finalists_v2/predictions/seeds/EXPLORE/F5/seed_1.parquet",
        "finalists_v2/predictions/seeds/EXPLORE/F5/seed_1.json",
        "pc3_finalists/predictions/CONFIRM/F5.parquet",
        "pc3_finalists/predictions/CONFIRM/F5.json",
        "pc3_finalists/predictions/seeds/CONFIRM/F5/seed_1.json",
        "finalists_v2/tables/F5/CONFIRM/auc.csv",
    )
    for name in names:
        path = prepared.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    artifacts = wf_final._completion_artifacts(prepared, view, pc3)
    assert set(names) <= artifacts.keys()
    wf_final._verify_snapshot(prepared.out, artifacts)
    (prepared.out / names[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        wf_final._verify_snapshot(prepared.out, artifacts)


def test_cached_confirm_phase_e_checks_manifest_source_and_inputs(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    key = "F5-finalist"
    source = prepared.root / "phase_f/source.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen downstream source")
    dest = prepared.out / "tables/phase_e/CONFIRM" / key
    dest.mkdir(parents=True)
    candidate = pd.DataFrame({"pred": [2.0], "model": [key]})
    baseline = pd.DataFrame({"pred": [1.0], "model": ["F0-1-B5"]})
    for model,data in ((key,candidate),('F0-1-B5',baseline)):
        p=wf_final.prediction_path(prepared,model,'CONFIRM');p.parent.mkdir(parents=True,exist_ok=True)
        data.to_parquet(p,index=False)
    write_json(dest/'input_identity.json',{'candidate':sha256(wf_final.prediction_path(prepared,key,'CONFIRM')),
        'B5':sha256(wf_final.prediction_path(prepared,'F0-1-B5','CONFIRM'))})
    monkeypatch.setattr(wf_final, "load_predictions",
        lambda view, selected, arm: candidate if selected == key else baseline)
    from phase_f import wf_downstream
    monkeypatch.setattr(wf_downstream, "_frame_digest", lambda frame: str(frame.pred.iloc[0]))
    record = {"artifacts": {}, "source_sha256": {"phase_f/source.py": sha256(source)},
        "candidate_input_sha256": "2.0", "baseline_input_sha256": "1.0",
        "registry_fields": {"E_c10_22_recall": 0.8}}
    record["manifest_sha256"] = config_hash(record)
    write_json(dest / "manifest.json", record)
    assert wf_final._confirm_e(prepared, key)["E_status"] == "completed"
    tampered = {**record, "registry_fields": {"E_c10_22_recall": 0.0}}
    write_json(dest / "manifest.json", tampered)
    with pytest.raises(ValueError, match="manifest changed"):
        wf_final._confirm_e(prepared, key)
    write_json(dest / "manifest.json", record)
    source.write_bytes(b"new source")
    with pytest.raises(ValueError, match="source changed"):
        wf_final._confirm_e(prepared, key)


def test_unavailable_confirm_phase_e_is_sealed_and_reused(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    key = "F5-finalist"
    dest = prepared.out / "tables/phase_e/CONFIRM" / key
    candidate = pd.DataFrame({"pred": [2.0], "model": [key]})
    baseline = pd.DataFrame({"pred": [1.0], "model": ["F0-1-B5"]})
    for model, frame in ((key, candidate), ("F0-1-B5", baseline)):
        path = wf_final.prediction_path(prepared, model, "CONFIRM")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    monkeypatch.setattr(wf_final, "load_predictions",
        lambda view, selected, arm: candidate if selected == key else baseline)
    source = prepared.root / "phase_f/frozen_e.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen Phase E")
    monkeypatch.setattr(wf_final, "_phase_e_source_hashes",
        lambda root: {"phase_f/frozen_e.py": sha256(source)})
    from phase_f import wf_downstream
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise ValueError("insufficient episodes")

    monkeypatch.setattr(wf_downstream, "evaluate_candidate", unavailable)
    first = wf_final._confirm_e(prepared, key)
    second = wf_final._confirm_e(prepared, key)
    assert first == second == {"E_status": "unavailable", "E_reason": "insufficient episodes"}
    assert calls == [1]
    record = wf_final.verify_lock(json.loads((dest / "unavailable.json").read_text(encoding="utf-8")))
    assert record["input_identity_sha256"] == sha256(dest / "input_identity.json")
    source.write_bytes(b"changed Phase E")
    with pytest.raises(ValueError, match="unavailable evidence changed"):
        wf_final._confirm_e(prepared, key)
    assert calls == [1]


def test_completed_resume_recovers_report_and_git_commit(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    lockpath = prepared.out / "logs/confirm_lock.json"
    lock = wf_final._sealed(lockpath, {"source_hashes": {}, "explore_artifacts": {},
        "prediction_sha256": {}, "candidates": [], **_audit_fields(prepared)})
    reservation = prepared.out / "logs/confirm_reservation.json"
    wf_final._sealed(reservation, {"selection_lock_sha256": lock["lock_sha256"]})
    forecast = prepared.out / "finalists_v2/predictions/CONFIRM/F5.parquet"
    forecast.parent.mkdir(parents=True)
    forecast.write_bytes(b"sealed confirm prediction")
    done = wf_final._sealed(prepared.out / "logs/confirm_once.json", {
        "selection_lock_sha256": lock["lock_sha256"],
        "reservation_sha256": sha256(reservation),
        "artifacts": {forecast.relative_to(prepared.out).as_posix(): sha256(forecast)}})
    calls = []
    monkeypatch.setattr(wf_final, "_source_hashes", lambda root: {})
    monkeypatch.setattr(wf_final, "_committed_selection", lambda *args: None)
    from phase_f import workflow
    monkeypatch.setattr(workflow, "_git_commit", lambda *args: calls.append("commit"))
    reporting = types.ModuleType("phase_f.wf_reporting")
    reporting.final_report = lambda *args: calls.append("report")
    monkeypatch.setitem(sys.modules, "phase_f.wf_reporting", reporting)

    assert wf_final.run_final(prepared) == done
    assert calls == ["report", "commit"]
    assert json.loads((prepared.out / "logs/workflow/stage_4.json").read_text(encoding="utf-8"))["status"] == "completed"
    audit = prepared.root / wf_final._PREFLIGHT_AUDITS["preflight_sha256"]
    audit.write_bytes(b"changed preflight audit")
    with pytest.raises(ValueError, match="Pre-CONFIRM audit changed"):
        wf_final.run_final(prepared)
    assert calls == ["report", "commit"]
    audit.write_bytes(b"preflight_sha256")
    forecast.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        wf_final.run_final(prepared)
    assert calls == ["report", "commit"]
    forecast.write_bytes(b"sealed confirm prediction")
    revised = {"selection_lock_sha256": lock["lock_sha256"], "at": "changed"}
    write_json(reservation, {**revised, "lock_sha256": config_hash(revised)})
    with pytest.raises(ValueError, match="reservation evidence changed"):
        wf_final.run_final(prepared)
    assert calls == ["report", "commit"]
