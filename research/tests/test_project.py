"""Protect claim provenance, fail-closed generation, and reproducible snapshots."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from research.project import ROOT, GENERATED, ResearchError, build, digest, load_project, snapshot


def copy_project(destination):
    """Copy only registered evidence and generated documents, never raw data."""
    names = set(snapshot(load_project(ROOT))['sources']) | set(GENERATED)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)


class ResearchProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        copy_project(self.root)

    def edit_json(self, name, edit):
        path = self.root / name
        value = json.loads(path.read_text(encoding='utf-8'))
        edit(value)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def test_current_selection_and_exact_evidence_row(self):
        project = load_project(self.root)
        state = snapshot(project)
        self.assertEqual(state['models']['4']['point_model'], 'p4_blend3')
        self.assertEqual(state['models']['4']['risk_model'], 'lgbm_quantile_b')
        row = project['values']['EV-P4-H4']
        self.assertEqual(row['candidate'], 'p4_blend3')
        self.assertAlmostEqual(float(row['candidate_peak_mae']), 12.579108, places=5)
        self.assertEqual(project['values']['EV-WALK']['tau_common'], 177)

    def test_rebuild_is_deterministic_and_preserves_all_inputs(self):
        before = {p: digest(p) for p in self.root.rglob('*') if p.is_file()
                  and p.relative_to(self.root).as_posix() not in GENERATED}
        build(self.root)
        first = {name: (self.root / name).read_bytes() for name in GENERATED}
        build(self.root)
        build(self.root, check=True)
        self.assertEqual(first, {name: (self.root / name).read_bytes() for name in GENERATED})
        after = {p: digest(p) for p in self.root.rglob('*') if p.is_file()
                 and p.relative_to(self.root).as_posix() not in GENERATED}
        self.assertEqual(before, after)

    def test_changed_narrative_requires_rebuild(self):
        path = self.root / 'paper/sections/06_discussion.md'
        path.write_text(path.read_text(encoding='utf-8') + '\n추가 논의.\n', encoding='utf-8')
        with self.assertRaisesRegex(ResearchError, 'stale'):
            build(self.root, check=True)
        build(self.root)
        build(self.root, check=True)
        self.assertIn('추가 논의.', (self.root / 'paper/manuscript.md').read_text(encoding='utf-8'))

    def test_missing_evidence_fails_before_writing(self):
        before = {name: (self.root / name).read_bytes() for name in GENERATED}
        (self.root / 'outputs/p6/e5_fn/summary.json').unlink()
        with self.assertRaisesRegex(ResearchError, 'Missing source'):
            build(self.root)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in GENERATED})

    def test_duplicate_comparison_rows_are_not_silently_selected(self):
        path = self.root / 'outputs/p4/p4_point_comparisons.csv'
        with path.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        selected = next(r for r in rows if r['horizon'] == '4' and r['candidate'] == 'p4_blend3'
                        and r['incumbent'] == 'lgbm_no_holiday_weight_2')
        with path.open('a', encoding='utf-8', newline='') as handle:
            csv.DictWriter(handle, fieldnames=list(selected)).writerow(selected)
        with self.assertRaisesRegex(ResearchError, 'expected one row, found 2'):
            load_project(self.root)

    def test_nonfinite_metric_is_rejected(self):
        self.edit_json('research/claims.json', lambda d: d['evidence'].append({
            'id': 'EV-NAN', 'path': 'nan.json', 'format': 'json', 'scope': 'development',
            'role': 'invalid metric', 'numeric_fields': ['score']}))
        (self.root / 'nan.json').write_text('{"score": "NaN"}', encoding='utf-8')
        with self.assertRaisesRegex(ResearchError, 'invalid numeric'):
            load_project(self.root)

    def test_exploration_cannot_be_promoted_to_confirmed_claim(self):
        self.edit_json('research/claims.json',
                       lambda d: d['claims'][0].update(evidence=['EV-WALK']))
        with self.assertRaisesRegex(ResearchError, 'exploratory evidence'):
            load_project(self.root)

    def test_unknown_reference_and_unsubstantiated_completion_fail(self):
        self.edit_json('research/claims.json',
                       lambda d: d['claims'][0].update(evidence=['EV-MISSING']))
        with self.assertRaisesRegex(ResearchError, 'unknown evidence'):
            load_project(self.root)
        copy_project(self.root)
        self.edit_json('experiments/registry.json',
                       lambda d: d['experiments'][-1].update(status='completed'))
        with self.assertRaisesRegex(ResearchError, 'completed experiment lacks evidence'):
            load_project(self.root)

    def test_changed_selection_and_incomplete_parity_are_rejected(self):
        self.edit_json('outputs/logs/development_selection.json',
                       lambda d: d['selection']['by_horizon']['4'].update(point_model='unapproved'))
        with self.assertRaisesRegex(ResearchError, 'disagree'):
            snapshot(load_project(self.root))
        copy_project(self.root)
        self.edit_json('outputs/logs/P4_integration.json',
                       lambda d: d['parity']['p4_blend3'].update(max_abs_diff={}))
        with self.assertRaisesRegex(ResearchError, 'parity'):
            snapshot(load_project(self.root))

    def test_source_traversal_and_symlink_are_rejected(self):
        self.edit_json('research/claims.json',
                       lambda d: d['evidence'][0].update(path='../outside.csv'))
        with self.assertRaisesRegex(ResearchError, 'leaves repository'):
            load_project(self.root)
        copy_project(self.root)
        path = self.root / 'outputs/p4/p4_point_comparisons.csv'
        path.unlink()
        path.symlink_to(ROOT / 'outputs/p4/p4_point_comparisons.csv')
        with self.assertRaisesRegex(ResearchError, 'leaves repository'):
            load_project(self.root)

    def test_generated_output_cannot_overwrite_source_via_symlink(self):
        target = self.root / 'paper/sections/06_discussion.md'
        before = target.read_bytes()
        output = self.root / 'paper/manuscript.md'
        output.unlink()
        output.symlink_to(target)
        with self.assertRaisesRegex(ResearchError, 'symlink'):
            build(self.root)
        self.assertEqual(target.read_bytes(), before)

    def test_cli_works_without_site_packages_or_training_imports(self):
        result = subprocess.run([sys.executable, '-S', '-m', 'research', 'check'],
                                cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('holdout_read=false; training_executed=false', result.stdout)
        self.assertFalse((self.root / 'data').exists())


if __name__ == '__main__':
    unittest.main()
