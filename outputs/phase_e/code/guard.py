"""Phase E filesystem scope and immutable evidence checks (no historical reads)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write('\n')


class AccessGuard:
    """Deny historical result reads and any repository write outside Phase E.

    Raw bytes are allowed solely for the separately tested pinned prefix loader
    and SHA verification. This hook checks paths, not the semantics of decoding;
    the loader's boundary assertion and poisoned-tail tests provide that evidence.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.read_paths = set()
        self.denied = []

    def check(self, path, writing=False):
        try:
            path = Path(os.fsdecode(path)).resolve()
        except (TypeError, ValueError):
            return
        if not path.is_relative_to(self.root):
            return  # Python/environment libraries and OS temporary facilities.
        rel = path.relative_to(self.root).as_posix()
        allowed_write = rel.startswith('outputs/phase_e/')
        forbidden_read = (
            rel.startswith(('verification/', 'report/', 'slides/', 'submission/'))
            or (rel.startswith('outputs/') and not rel.startswith(('outputs/phase_c/', 'outputs/phase_e/')))
        )
        if writing and not allowed_write or not writing and forbidden_read:
            self.denied.append({'path': rel, 'writing': writing})
            raise PermissionError(f'Phase E access boundary: {rel}')
        if not writing:
            self.read_paths.add(rel)

    def hook(self, event, args):
        if event == 'open':
            path, mode, flags = args
            writing = bool((isinstance(mode, str) and any(x in mode for x in 'wax+')) or
                           (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            self.check(path, writing)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            self.check(args[0], True)
        elif event in ('os.rename',):
            self.check(args[0], True)
            self.check(args[1], True)

    def install(self):
        sys.dont_write_bytecode = True
        sys.addaudithook(self.hook)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, encoding='utf-8').strip()


def protected_hashes(root):
    """Only allowed C/D source/development files; never historical final files."""
    root = Path(root)
    paths = []
    for directory in ('phase_c', 'phase_d', 'outputs/phase_c', 'outputs/phase_d'):
        folder = root / directory
        if folder.exists():
            paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(root / p for p in ('configs/phase_c.json', 'scripts/phase_b_reference/run_phase_b_temp_probe.py'))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}


def assert_existing_preserved(root, before):
    after = protected_hashes(root)
    changed = [p for p, h in before.items() if after.get(p) != h]
    added = sorted(set(after) - set(before))
    diff = git(root, 'diff', '--name-only', 'HEAD')
    if changed or added or diff:
        raise AssertionError({'modified': changed, 'added_in_protected': added, 'tracked_diff': diff})
    return {'preserved': True, 'files_checked': len(before), 'changed': changed, 'new_in_C_D': added,
            'tracked_diff': diff, 'head': git(root, 'rev-parse', 'HEAD')}
