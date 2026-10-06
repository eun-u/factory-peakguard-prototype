from pathlib import Path
import json
import pandas as pd
from phase_f import goal_protocol
from phase_f.registry import now, sha256, write_json

root = Path.cwd()
base = root / 'outputs/phase_f/logs/revision_20261006'
pause = base / 'user_pause_20261007'
ns = root / 'outputs/phase_f/goal_r1_recent_analog_v1'
proof_path = base / 'recent_analog_v1_full_verification.json'
proof = json.loads(proof_path.read_text(encoding='utf-8'))
assert proof['completed_candidate_count'] == 4
assert proof['holdout_read'] is False and proof['historical_final_artifact_read'] is False
results = []
artifacts = []
for row in proof['candidate_results']:
    folder = ns / 'tables' / row['id'] / 'EXPLORE'
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert sha256(manifest_path) == row['score_manifest_sha256']
    assert manifest['fields'] == row['fields']
    targets = goal_protocol.meets_targets(row['fields'])
    assert targets['individual'] == manifest['individual']
    assert targets['targets_met'] == row['targets_met'] == manifest['targets_met'] is False
    for name, expected in manifest['artifacts'].items():
        assert sha256(folder / name) == expected, (row['id'], name)
        artifacts.append({'candidate': row['id'], 'name': name, 'sha256': expected})
    assert row['n_primary_models'] == row['n_fresh_models'] == 8
    assert row['rows'] == 102427
    results.append({'id': row['id'], 'fields': row['fields'], **targets})
best = min(results, key=lambda row: row['fields']['AUC_MAE'])
ci = pd.read_csv(ns / 'tables' / best['id'] / 'EXPLORE/paired_ci.csv')
ci = ci[(ci.baseline == 'FG-R1-core-l1__guard-q95-suppress') & ci.horizon.isna()]
ci_records = json.loads(ci.to_json(orient='records'))
assert all(row['ci_low'] <= 0 <= row['ci_high'] for row in ci_records)
record = {
    'at': now(), 'user_request': '검증 후 점수가 안나와도 멈춰',
    'work_status': 'paused_after_current_validation', 'goal_status_request': 'paused',
    'search_exit_code': 0, 'root_full_verifier_exit_code': 0,
    'full_verification_sha256': sha256(proof_path),
    'score_artifact_hashes_and_target_semantics_reverified': True,
    'score_artifacts': artifacts, 'candidate_results': results, 'best_current_wave': best,
    'best_vs_existing_guard_paired_day_block_CI': ci_records,
    'decision': 'targets not met; no release promotion; peak point-estimate tradeoff not statistically confirmed',
    'original_checkpoint_proof_sha256': sha256(pause / 'original_checkpoint_after_stop.json'),
    'original_root_acceptance_sha256': sha256(pause / 'original_root_checkpoint_acceptance.json'),
    'new_waves_authorized': False, 'automatic_resume': False,
    'remaining_work': ['original Stage1 replay unresolved123/125', 'M2 model cells114/520 and seeds2/5 incomplete',
                       'original search budgets and operational/final integration incomplete',
                       'independent confirmation and commercial field validation unperformed'],
    'goal_achieved': False, 'holdout_read': False, 'historical_final_artifact_read': False,
}
write_json(pause / 'FINAL_RESULT_CHECKPOINT.json', record, exclusive=True)
write_json(ns / 'logs/stop_requested.json', {'at': now(), 'reason': record['user_request'],
    'automatic_resume': False, 'no_further_wave': True}, exclusive=True)
lines = ['# 검증 종료 및 사용자 요청에 따른 중지', '',
         'FG-R5 고정 네 설정의 실제 8주 예측, 별도 물리 재실행, 출처·체크포인트·미래값 불변성·평가값 대조를 완료했다. 모두 목표 미달이며 추가 탐색을 중지한다.', '',
         '| 설정 | 평균 MAE | 피크 MAE | h4 MAE | h16 MAE | nMAE |',
         '|---|---:|---:|---:|---:|---:|']
for row in results:
    f = row['fields']
    lines.append(f"| {row['id']} | {f['AUC_MAE']:.6f} | {f['AUC_PeakMAE']:.6f} | {f['h4_MAE']:.6f} | {f['h16_MAE']:.6f} | {f['AUC_nMAE']*100:.3f}% |")
lines += ['', '목표는 각각 4.5 / 9 / 3.5 / 5.5 / 5%다. 전력 물리 단위는 미확인이다.', '',
          '평균 최선 후보의 피크 오차 점추정치는 기존 guard보다 줄었지만 평균 오차는 높다. D1 평균/피크 및 D2 평균/피크 paired day-block CI가 모두 0을 포함하므로 확정적 개선이나 상용 성능으로 주장하지 않는다. 기존 모델을 출시 승격하지 않는다.', '',
          '현재 변형은 4설정×8개 primary 및 4설정×8개 fresh 모델, 각각102427행을 검증했다. 재실행은 독립 통계 표본이나 추가 확률 시드가 아니다. 모델 관련 root 검사29개 통과. 연구관리 build/check 통과; 관리 unittest10개 통과, 2개는 Windows symlink 권한으로 실행 불가다.', '',
          '원래 장기 작업은 406/520 M2 모델·3/5 시드 예측을 보존한 부분 완료다. 전체 예산과 최종 검증은 미완료이며 목표 달성으로 처리하지 않는다. 후보 CONFIRM 예측/평가결과·holdout·과거 final 지표는 열람하지 않았다. 과거 평가기간의 이미 관측된 raw 전력은 이후 원점의 인과 입력이 될 수 있으며, 평가 label을 학습하거나 현재/미래 값을 조회하지 않는다.', '',
          '[중지 및 재개 기록](PAUSE_AND_RESTART.md) · [구조화된 결과](FINAL_RESULT_CHECKPOINT.json)', '']
(pause / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
with (root / 'PROGRESS.md').open('a', encoding='utf-8') as f:
    f.write('\n\n### 2026-10-07 FG-R5 최종 검증 및 사용자 중지\n\n')
    f.write('- 실제 search session17563 exit0 및 root full verifier session22380 exit0. 4설정/64physical모델/각102427행 primary와 fresh exact replay 및 평가7필드 재계산, 전28개 score artifact SHA와 모든 목표판정 의미를 확인했다.\n')
    f.write('- 네 설정 모두 다섯 목표 미달. 평균 최선 w16/k3는 MAE7.528010/Peak11.621093/h4 5.607685/h16 9.663053/nMAE8.015654%. 기존 guard대비 peak 점추정 감소/평균 악화이며 D1/D2 평균·피크 CI는 모두0포함이다. 확정적 개선·출시 승격·상용성으로 표현하지 않는다.\n')
    f.write('- 사용자 지시대로 이후 변형/새 학습/자동재개를 중지한다. 원래 모델406/520·시드3/5를 물리SHA로 재확인해 보존했고 원래 Stage1 및 나머지 예산/최종 검증은 미완료다. 결과·중지/재개문·검증근거는 logs/revision_20261006/user_pause_20261007에 저장한다. 목표 paused 요청, goal_achieved=false, holdout/history-final 미열람.\n')
with (root / 'DECISIONS.md').open('a', encoding='utf-8') as f:
    f.write('\n\n## 2026-10-07 FG-R5 검증 종료 및 일시정지\n\n- 현재 고정4설정의 실제 검증은 완료했으나 다섯 성능목표 모두 미달이다. 평균 최선7.528010/11.621093는 피크/평균의 tradeoff이고 day-block CI는0을 포함한다. 출시 승격하지 않으며, 사용자 요청을 우선하여 점수와 관계없이 추가 탐색 없이 일시정지한다. 전체 예산/최종 단일union/상용성 검증은 미완료로 남긴다.\n')
print(json.dumps({'at': record['at'], 'best': best, 'current_validation_complete': True, 'no_further_wave': True}))
