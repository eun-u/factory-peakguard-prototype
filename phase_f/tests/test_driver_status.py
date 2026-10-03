"""Durable status remains truthful across no-op restarts and early failures."""
import json
from types import SimpleNamespace
import pytest
from phase_f import run, workflow
from phase_f.registry import write_json


def test_completed_restart_and_invalid_stage_preserve_terminal_status(tmp_path, monkeypatch):
    prepared = SimpleNamespace(out=tmp_path)
    path = tmp_path/'logs/driver_status.json'
    original = {'status':'completed','stage':'all','full_phase_complete':True}
    write_json(path, original)
    for stage in range(5):
        write_json(tmp_path/'logs/workflow'/f'stage_{stage}.json', {'status':'completed'})
    monkeypatch.setattr(workflow,'run_all',lambda *a,**k: 'done')
    monkeypatch.setattr(workflow,'run_stage',lambda *a,**k: 'done')
    assert run.dispatch_stage(prepared,'all') == 'done'
    assert run.dispatch_stage(prepared,'0') == 'done'
    for stage in ('invalid', '9'):
        with pytest.raises(ValueError):
            run.dispatch_stage(prepared, stage)
    assert json.loads(path.read_text()) == original
    write_json(path, {'status':'running','full_phase_complete':False})
    run.dispatch_stage(prepared,'all')
    recovered = json.loads(path.read_text())
    assert recovered['status']=='completed' and recovered['full_phase_complete'] is True
    assert recovered['recovered_from_completed_checkpoints'] is True
    def missing_report(*args,**kwargs):
        raise RuntimeError('Missing final report')
    monkeypatch.setattr(workflow,'run_all',missing_report)
    with pytest.raises(RuntimeError,match='Missing final report'):
        run.dispatch_stage(prepared,'all')
    failed = json.loads(path.read_text())
    assert failed['status']=='failed' and failed['full_phase_complete'] is False


def test_precondition_failure_does_not_leave_running_status(tmp_path, monkeypatch):
    prepared = SimpleNamespace(out=tmp_path)
    def fail(*args,**kwargs):
        assert json.loads((tmp_path/'logs/driver_status.json').read_text())['status']=='running'
        raise RuntimeError('Missing prior stage')
    monkeypatch.setattr(workflow,'run_stage',fail)
    with pytest.raises(RuntimeError,match='prior stage'):
        run.dispatch_stage(prepared,'2')
    assert json.loads((tmp_path/'logs/driver_status.json').read_text())['status']=='failed'
