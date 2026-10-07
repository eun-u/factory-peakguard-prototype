from pathlib import Path
import importlib.util
import pytest

SPEC = importlib.util.spec_from_file_location('phase_e_guard', Path(__file__).parents[1]/'code/guard.py')
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_historical_read_denied_without_opening():
    tmp_path = Path(__file__).parents[1]/'logs'/'synthetic_guard_root'
    g = guard.AccessGuard(tmp_path)
    for path in ('verification/results/05_summary.json','outputs/predictions/final_test.csv','outputs/logs/final_metrics.json'):
        with pytest.raises(PermissionError):
            g.check(tmp_path/path)
    assert len(g.denied)==3
    assert not g.read_paths


def test_namespace_writes():
    tmp_path = Path(__file__).parents[1]/'logs'/'synthetic_guard_root'
    g = guard.AccessGuard(tmp_path)
    for path in ('phase_c/statistical.py','outputs/phase_c/logs/a.json','PROGRESS.md'):
        with pytest.raises(PermissionError):
            g.check(tmp_path/path, True)
    g.check(tmp_path/'outputs/phase_e/logs/a.json',True)
    g.check(tmp_path/'phase_c/statistical.py')
