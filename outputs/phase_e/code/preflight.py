"""Record protocol/input/code provenance before any Phase E real-data run."""
from pathlib import Path
import sys
import datetime
import importlib.metadata
import platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from guard import sha256, write_json, git, protected_hashes


def main():
    out = ROOT / 'outputs/phase_e'
    branch = git(ROOT, 'branch', '--show-current')
    assert branch == 'Phase3_experiments'
    assert not git(ROOT, 'diff', '--name-only', 'HEAD')
    write_json(out/'logs/protected_before.json', protected_hashes(ROOT))
    write_json(out/'logs/protocol_seal.json', {
        'locked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'files': {name: sha256(out/'logs'/name) for name in ('phase_e_protocol_lock.md','phase_e_config_lock.json')},
        'before_phase_e_numeric_prediction_calibration_scoring': True,
    })
    write_json(out/'logs/run_environment.json', {
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'branch': branch, 'head': git(ROOT, 'rev-parse', 'HEAD'),
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'packages': {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow','matplotlib','pytest','joblib']},
        'development_only':True, 'phase_d_artifact_present':(ROOT/'outputs/phase_d').exists(),
        'horizon_authority':'explicit user-locked Phase E instruction',
        'new_files_scope':'outputs/phase_e', 'original_HEAD_preserved':True,
    })
    print('Protocol sealed and C/D files hashed; no Phase E observations processed.')


if __name__ == '__main__':
    main()
