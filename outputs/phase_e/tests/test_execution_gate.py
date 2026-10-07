from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import run_phase_e as runner


def test_evaluation_rejects_changed_seal_before_imports_or_io(monkeypatch):
    def changed():
        raise AssertionError('Frozen analysis implementation changed')
    monkeypatch.setattr(runner,'verify_analysis_seal',changed)
    with pytest.raises(AssertionError,match='Frozen analysis'):
        runner.evaluate()


def test_analysis_seal_mismatch_is_fatal(monkeypatch):
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:'{"files":{"src/analysis/errors.py":"sealed"}}')
    monkeypatch.setattr(runner,'sha256',lambda path:'changed')
    with pytest.raises(AssertionError,match='src/analysis/errors.py'):
        runner.verify_analysis_seal()
