"""Research management CLI; deliberately has no train/final/execute command."""
from __future__ import annotations

import argparse
import json
import sys

from .project import ROOT, ResearchError, build, load_project, snapshot, source_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='Generate status, evidence snapshot and manuscript from existing evidence')
    sub.add_parser('check', help='Validate and require all generated files to be current')
    sub.add_parser('validate', help='Validate claims, experiments, evidence and selected model provenance')
    status = sub.add_parser('status', help='Read current research and model state')
    status.add_argument('--json', action='store_true')
    show = sub.add_parser('show', help='Read an experiment plan without executing it')
    show.add_argument('experiment_id')
    args = parser.parse_args(argv)
    try:
        if args.command in {'build', 'check'}:
            build(ROOT, check=args.command == 'check')
            print('Research documents '+('verified' if args.command == 'check' else 'generated')+'; holdout_read=false; training_executed=false')
            return 0
        p = load_project(ROOT)
        s = snapshot(p)
        if args.command == 'show':
            ex = p['experiments'].get(args.experiment_id)
            if ex is None:
                raise ResearchError('Unknown experiment: '+args.experiment_id)
            print(source_path(ROOT, ex['plan']).read_text(encoding='utf-8'))
        elif args.command == 'status' and args.json:
            print(json.dumps(s, ensure_ascii=False, indent=2))
        else:
            print(p['charter']['title'])
            print(f"Questions: {len(p['questions'])}; claims: {len(p['claims'])}; experiments: {len(p['experiments'])}")
            for key, ex in p['experiments'].items():
                print(f"{key}: {ex['status']} | {ex['next_action']}")
            print('Evidence scope: archived development; holdout_read=false; training_executed=false')
        return 0
    except (ResearchError, KeyError, TypeError, OSError) as exc:
        print(f'Research validation failed: {exc}', file=sys.stderr)
        return 1
