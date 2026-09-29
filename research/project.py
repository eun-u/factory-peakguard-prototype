"""Validate research claims and build deterministic documents from archived evidence.

Only Python's standard library is required. No training, raw-data loader, final
evaluation, or user-supplied executable is imported or invoked by this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRIES = ('research/charter.json', 'research/claims.json', 'experiments/registry.json')
GENERATED = ('research/generated/current_snapshot.json', 'research/generated/status.md',
             'paper/generated/results.md', 'paper/manuscript.md')
FIELDS = ('Background', 'Objective', 'Method', 'Dataset', 'Baselines', 'Evaluation',
          'Result', 'Conclusion', 'Threats', 'Pros', 'Cons', 'Contribution')
CLAIM_STATES = {'supported_development', 'exploratory', 'hypothesis', 'design_only', 'simulation_only'}
EXPERIMENT_STATES = {'planned', 'running', 'completed', 'blocked', 'rejected'}


class ResearchError(ValueError):
    pass


def source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or '\\' in relative:
        raise ResearchError(f'Invalid repository path: {relative!r}')
    rel = PurePosixPath(relative)
    path = (root / relative).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root.resolve()):
        raise ResearchError(f'Path leaves repository: {relative}')
    if not path.is_file():
        raise ResearchError(f'Missing source: {relative}')
    return path


def read_json(root, relative):
    try:
        return json.loads(source_path(root, relative).read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ResearchError(f'Invalid JSON: {relative}: {exc}') from exc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows, kind):
    if not isinstance(rows, list):
        raise ResearchError(f'{kind} must be a list')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ResearchError(f'{kind} requires non-empty IDs')
        if row['id'] in result:
            raise ResearchError(f'Duplicate {kind} ID: {row["id"]}')
        result[row['id']] = row
    return result


def evidence_value(root, item):
    path = source_path(root, item['path'])
    kind = item['format']
    if kind == 'csv':
        with path.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        selector = item.get('selector')
        if not isinstance(selector, dict) or not selector:
            raise ResearchError(f'CSV evidence needs a selector: {item["id"]}')
        matches = [r for r in rows if all(r.get(k) == str(v) for k, v in selector.items())]
        if len(matches) != 1:
            raise ResearchError(f'{item["id"]}: expected one row, found {len(matches)}')
        value = matches[0]
    elif kind == 'json':
        value = read_json(root, item['path'])
        for token in item.get('pointer', '').strip('/').split('/'):
            if not token:
                continue
            token = token.replace('~1', '/').replace('~0', '~')
            try:
                value = value[int(token)] if isinstance(value, list) else value[token]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ResearchError(f'{item["id"]}: missing JSON pointer') from exc
    elif kind == 'markdown':
        value = {'text': path.read_text(encoding='utf-8')}
    else:
        raise ResearchError(f'Unknown evidence format: {kind}')
    if not isinstance(value, dict):
        raise ResearchError(f'{item["id"]}: evidence value must be an object')
    for field in item.get('required_fields', []):
        if field not in value:
            raise ResearchError(f'{item["id"]}: missing field {field}')
    for field in item.get('numeric_fields', []):
        try:
            valid = math.isfinite(float(value[field]))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ResearchError(f'{item["id"]}: invalid numeric field {field}')
    return value


def load_project(root=ROOT):
    root = Path(root).resolve()
    charter, claims, registry = [read_json(root, p) for p in REGISTRIES]
    if any(d.get('schema_version') != 1 for d in (charter, claims, registry)):
        raise ResearchError('Unsupported research schema version')
    if set(charter.get('definition', {})) != set(FIELDS):
        raise ResearchError('Study definition must contain the 12 agreed fields')
    if any(not isinstance(v, str) or not v.strip() for v in charter['definition'].values()):
        raise ResearchError('Study definitions must be non-empty text')
    questions = indexed(charter.get('questions'), 'question')
    evidence = indexed(claims.get('evidence'), 'evidence')
    statements = indexed(claims.get('claims'), 'claim')
    experiments = indexed(registry.get('experiments'), 'experiment')
    values = {}
    for key, item in evidence.items():
        if item.get('scope') not in {'development', 'exploratory', 'simulation', 'historical'}:
            raise ResearchError(f'{key}: evidence scope must be explicit; no automatic final evaluation')
        if not item.get('role'):
            raise ResearchError(f'{key}: evidence role is required')
        values[key] = evidence_value(root, item)
    for key, claim in statements.items():
        if claim.get('status') not in CLAIM_STATES or claim.get('rq') not in questions:
            raise ResearchError(f'{key}: unknown claim state or research question')
        if not claim.get('statement') or not claim.get('limitation'):
            raise ResearchError(f'{key}: statement and limitation are required')
        refs = claim.get('evidence', [])
        if any(ref not in evidence for ref in refs):
            raise ResearchError(f'{key}: unknown evidence reference')
        if claim['status'] in {'supported_development', 'exploratory', 'simulation_only'} and not refs:
            raise ResearchError(f'{key}: empirical claim lacks evidence')
        if claim['status'] == 'supported_development' and any(evidence[r]['scope'] != 'development' for r in refs):
            raise ResearchError(f'{key}: exploratory evidence cannot prove a development claim')
        if any(ref not in experiments for ref in claim.get('experiments', [])):
            raise ResearchError(f'{key}: unknown experiment reference')
        if not claim.get('paper_sections') or any(section not in charter.get('paper_sections', []) for section in claim['paper_sections']):
            raise ResearchError(f'{key}: claim must map to known paper sections')
    for key, ex in experiments.items():
        if ex.get('status') not in EXPERIMENT_STATES:
            raise ResearchError(f'{key}: unknown experiment state')
        if not ex.get('questions') or any(q not in questions for q in ex['questions']):
            raise ResearchError(f'{key}: unknown research question')
        if any(c not in statements for c in ex.get('claims', [])):
            raise ResearchError(f'{key}: unknown claim')
        if any(e not in evidence for e in ex.get('evidence', [])):
            raise ResearchError(f'{key}: unknown evidence')
        if ex['status'] == 'completed' and not ex.get('evidence'):
            raise ResearchError(f'{key}: completed experiment lacks evidence')
        for field in ('hypothesis', 'comparison', 'metrics', 'decision_rule', 'next_action'):
            if not ex.get(field):
                raise ResearchError(f'{key}: missing {field}')
        source_path(root, ex['plan'])
        for relative in ex.get('implementation', []):
            source_path(root, relative)
    for section in charter.get('paper_sections', []):
        source_path(root, section)
    if not charter.get('paper_sections') or len(set(charter['paper_sections'])) != len(charter['paper_sections']):
        raise ResearchError('Paper sections must be non-empty and unique')
    return {'root': root, 'charter': charter, 'questions': questions,
            'evidence': evidence, 'values': values, 'claims': statements, 'experiments': experiments}


def snapshot(project):
    root = project['root']
    selection_path = 'outputs/logs/development_selection.json'
    selection = read_json(root, selection_path)['selection']['by_horizon']
    summary = read_json(root, 'outputs/p4/p4_summary.json')
    integration = read_json(root, 'outputs/logs/P4_integration.json')
    models = {}
    if set(selection) != set(summary['point_selection']) or set(selection) != set(summary['risk_selection']):
        raise ResearchError('Selection horizons disagree with P4 summary')
    for h, choice in sorted(selection.items(), key=lambda x: int(x[0])):
        risk = choice.get('risk_model')
        resolution = 'explicit'
        if risk is None:
            if choice.get('conformal') not in {'a', 'b'}:
                raise ResearchError(f'h{h}: risk model is unresolved')
            risk = 'lgbm_quantile_' + choice['conformal']
            resolution = 'legacy_conformal_field_confirmed_by_p4_summary'
        point = choice['point_model']
        if point != summary['point_selection'][h] or risk != summary['risk_selection'][h]:
            raise ResearchError(f'h{h}: selected models disagree with adoption evidence')
        if point in {'p4_mstl_daily', 'p4_blend3'}:
            checks = integration['parity'].get(point, {})
            differences = checks.get('max_abs_diff', {})
            if not differences or checks.get('alert_mismatch') != 0 or any(v != 0 for v in differences.values()):
                raise ResearchError(f'h{h}: missing or nonzero integration parity')
        models[h] = {'point_model': point, 'risk_model': risk, 'risk_resolution': resolution,
                     'horizon_minutes': int(h)*15, 'scope': 'development_selected'}
    source_names = list(REGISTRIES) + [selection_path, 'outputs/p4/p4_summary.json',
                                     'outputs/logs/P4_integration.json', 'eval_protocol.md', 'configs/default.yaml']
    source_names += [e['path'] for e in project['evidence'].values()]
    source_names += project['charter']['paper_sections']
    source_names += [ex['plan'] for ex in project['experiments'].values()]
    source_names += [p for ex in project['experiments'].values() for p in ex.get('implementation', [])]
    source_names += ['paper/references.bib']
    source_names += [p.relative_to(root).as_posix() for p in sorted((root/'research').glob('*.py'))]
    return {'schema_version': 1, 'scope': 'archived_development_evidence',
            'evidence_base_commit': project['charter']['evidence_base_commit'],
            'new_experiments_executed': False, 'holdout_read': False,
            'models': models,
            'claims': {k: v['status'] for k, v in project['claims'].items()},
            'experiments': {k: v['status'] for k, v in project['experiments'].items()},
            'known_limitations': project['charter']['known_limitations'],
            'sources': {p: digest(source_path(root, p)) for p in sorted(set(source_names))}}


def markdown_table(headers, rows):
    def safe(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(safe(x) for x in row)+' |' for row in rows])


def rendered(project):
    snap = snapshot(project)
    v = project['values']
    p4, p5, fn, wf = (v[k] for k in ('EV-P4-H4', 'EV-P5-H4', 'EV-FN-H4', 'EV-WALK'))
    f = lambda value: f'{float(value):.3f}'
    p = project['charter']
    result = '# 현재 결과 - 개발 증거에서 자동 생성\n\n'
    result += '이 문서는 기존 결과를 읽어 생성한다. 새 학습·정책 실험·최종 평가는 실행하지 않는다. 오차는 원자료 단위이며 단위 미확인 상태를 유지한다.\n\n'
    result += markdown_table(['비교', '기준', '후보', '개선량과 범위'], [
        ['h4 피크 MAE / P4', f(p4['incumbent_peak_mae']), f(p4['candidate_peak_mae']),
         f"{f(p4['peak_mae_gain'])}, CI [{f(p4['ci_low'])}, {f(p4['ci_high'])}], Holm p={f(p4['p_holm'])}"],
        ['점예측 경보 F1 / P4', f(p4['incumbent_episode_f1']), f(p4['candidate_episode_f1']), '확률 경보 지표와 구분'],
        ['점예측 경보 FP 위치 / P4', p4['incumbent_fp'], p4['candidate_fp'], '15분 위치 수, 알림 사건 수 아님'],
        ['h4 Chronos 피크 MAE / P5', f(p5['incumbent_peak_mae']), f(p5['candidate_peak_mae']),
         f"CI [{f(p5['ci_low'])}, {f(p5['ci_high'])}], 미채택 ({p5['decision']})"]])
    b = fn['by_category']
    result += f"\n\nh4 실제 사건 {fn['actual_episodes']}개: 적중 {b['hit']}, 늦은 적중 {b['late_hit']}, 규칙 기반 판단 미탐 {b['decision_miss']}, 예측 미탐 {b['forecast_miss']}. 미탐 중 {fn['point_model_catches_risk_misses']}개에 점예측 경보가 존재한다. 결합 정책의 추가 적중·준비시간·오경보 개선은 아직 검증하지 않았다.\n\n"
    result += 'P6 갱신 전략은 탐색 결과다. 아래 값은 주별 상위 q95 커버리지의 평균과 피크 MAE 집계이며, 전체 표본의 단일 커버리지와 구분한다.\n\n'
    result += markdown_table(['전략', '피크 MAE', '상위 q95 커버리지 평균', '경보 F1 평균'],
        [[key, f(wf['pooled'][key]['peak_mae_weighted']), f(wf['pooled'][key]['top_cov_mean']),
          f(wf['pooled'][key]['risk_f1_mean'])] for key in ('frozen', 'every4weeks', 'weekly')])
    result += f"\n\n공통 τ는 원 JSON의 {wf['tau_common']}을 인용한다. 과거 요약 문서의 176 표기와 차이가 있어 원인 확인 과제로 기록한다. 결과 파일을 수정하거나 재계산하지 않았다.\n\n"
    result += '근거: [P4 비교](../../outputs/p4/p4_point_comparisons.csv), [P5 비교](../../outputs/p4/p5_point_comparisons.csv), [미탐](../../outputs/p6/e5_fn/summary.json), [갱신](../../outputs/p6/e2_walkforward/summary.json).\n'
    status = '# 연구 현황 - 자동 생성\n\n'
    status += '정의·가설·방법·실험·결과·주장의 연결을 기준으로 관리한다. 이 파일은 `python -m research build`로 생성한다.\n\n'
    status += '## 연구 기본 정의\n\n' + markdown_table(['항목', '정의'], [[k,p['definition'][k]] for k in FIELDS])
    status += '\n\n## 연구 질문\n\n' + markdown_table(['ID','질문','주 평가'],[[k,q['question'],q['primary_metric']] for k,q in project['questions'].items()])
    status += '\n\n## 현재 선정\n\n'+markdown_table(['거리','점예측','위험 출력','근거 범위'],
              [[f'{m["horizon_minutes"]}분',m['point_model'],m['risk_model'],m['risk_resolution']] for m in snap['models'].values()])
    status += '\n\n## 주장과 근거\n\n'+markdown_table(['ID','RQ','상태','주장','한계','본문 위치'],
              [[k,c['rq'],c['status'],c['statement'],c['limitation'],
                ', '.join(f'[{Path(s).stem}](../../{s})' for s in c['paper_sections'])]
               for k,c in project['claims'].items()])
    status += '\n\n## 실험 진행\n\n'+markdown_table(['ID','상태','연구 질문','다음 행동'],
              [[k,e['status'],', '.join(e['questions']),e['next_action']] for k,e in project['experiments'].items()])
    status += '\n\n## 알려진 한계\n\n'+'\n'.join('- '+x for x in p['known_limitations'])+'\n'
    manuscript = '# '+p['title']+'\n\n연구 본문 초안 · 개발 결과 · 최종 제출본 아님\n\n'
    manuscript += '수동 집필 원본은 `paper/sections/`, 수치는 `paper/generated/results.md`에 있다. 생성된 본문을 직접 수정하지 않는다.\n\n'
    for section in p['paper_sections']:
        manuscript += source_path(project['root'], section).read_text(encoding='utf-8').strip()+'\n\n'
        if section.endswith('04_experimental_design.md'):
            manuscript += result.replace('# 현재 결과 - 개발 증거에서 자동 생성', '## 5. 결과').replace('(../../outputs/', '(../outputs/')+'\n\n'
    manuscript += '## 근거 파일\n\n'+ '\n'.join(f'- [{key}: {e["role"]}](../{e["path"]}) ({e["scope"]})' for key,e in project['evidence'].items())+'\n'
    return dict(zip(GENERATED, [json.dumps(snap, ensure_ascii=False, indent=2)+'\n', status, result, manuscript]))


def build(root=ROOT, check=False):
    project = load_project(root)
    outputs = rendered(project)  # Validate every input before writing anything.
    for name in outputs:
        if (Path(root)/name).resolve() != Path(root).resolve()/name:
            raise ResearchError(f'Generated output must not traverse a symlink: {name}')
    stale = []
    for name, text in outputs.items():
        path = Path(root)/name
        if check:
            if not path.is_file() or path.read_text(encoding='utf-8') != text:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
    if stale:
        raise ResearchError('Generated files are stale: '+', '.join(stale)+'; run python -m research build')
    return project
