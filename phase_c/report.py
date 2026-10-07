"""Render the requested Phase C report from its own new evidence only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NAMES={"B0":"Persistence","B1":"Weekly Seasonal","B2":"CBL family","B3":"MSTL (fit-only)",
       "B4":"Ridge","B5":"Weekly-deviation Kalman","M1":"LightGBM Direct",
       "M1-W":"Peak-weighted LightGBM","M2":"TCN Direct","C1":"Weekly Residual LightGBM",
       "R1":"Chronos-2 (Reference only)"}


def table(frame: pd.DataFrame) -> str:
    def fmt(value):
        if pd.isna(value):return "판정불가"
        if isinstance(value,float):return f"{value:.4f}"
        return str(value).replace("|","/")
    columns=list(frame.columns)
    return "| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"+"\n".join("| "+" | ".join(fmt(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))


def write_report(out: Path) -> Path:
    def read(name):return json.loads((out/"logs"/name).read_text(encoding="utf-8"))
    selection=read("model_selection.json");lock=read("model_config_lock.json")
    runtime=read("runtime.json");audit=read("leakage_audit.json")
    auc=pd.read_csv(out/"tables/model_auc_summary.csv")
    curves=pd.read_csv(out/"tables/model_horizon_pooled_metrics.csv")
    decisions=pd.read_csv(out/"tables/model_selection_decision.csv")
    stability=pd.read_csv(out/"tables/model_stability.csv")
    paired=pd.read_csv(out/"tables/model_pairwise_ci.csv")
    selected=selection["selected_M_star"];baseline=selection["strongest_baseline"]
    display=auc[["dataset","model","n_horizons","AUC_MAE","AUC_PeakMAE"]].copy()
    display["family"]=display.model.map(NAMES)
    anchors=curves[(curves.dataset=="D1")&curves.horizon.isin([4,8,12,16])].copy()
    anchors["minutes"]=anchors.horizon*15
    fields=[x for x in ["model","minutes","n","peak_n","MAE","Peak_MAE","RMSE"] if x in anchors]
    full=curves[(curves.dataset=="D1")].copy();full["minutes"]=full.horizon*15
    paired_focus=paired[(paired.dataset=="D1")&paired.metric.isin(["AUC_MAE","AUC_PeakMAE"])
                        &((paired.model_a==selected)|(paired.model_b==selected))]
    failures=sorted(p.name for p in (out/"logs").glob("failure_*.txt"))
    text=f"""# Phase C Model Selection 결과

Development-only. 상태: {selection['selection_status']}. M*: **{selected or '판정불가'}** ({NAMES.get(selected,'근거 부족')}). Strongest baseline: **{baseline}** ({NAMES[baseline]}). h*는 선택하지 않았다.

## 실험 목적과 고정 설계

고정된 G0+G1+G2 정보 아래 1~4시간 전체를 담당할 하나의 대표 예측 구조를 비교했다. 13개 horizon은 60~240분, 15분 간격이며 개별 Direct 모델이다. 3개 시간순 development fold의 fit/stop/cal/score 역할과 horizon purge를 유지했다. 원본 Phase B 임시 코드에서 15개 core 특징 계산식을 확인했으며, 특히 r4_std는 ddof=1이다. feature search, horizon별 configuration 탐색, ensemble, 신규 exogenous 입력을 수행하지 않았다.

원자료 SHA-256, 입력 명세, 코드, 사전등록 문서는 logs/preregistration_lock.json에 결속돼 있다. full raw 파일의 바이트 해시/복사는 무결성 확인에만 사용하고 수치 파싱은 2021-08-09 09:45 미만의 봉인 로더로 제한했다. 모든 통계적 peak는 y>Q95(y_fit)이고 계약전력 초과를 의미하지 않는다.

## 데이터 접근 및 경계 검증

Raw holdout 관측값 파싱: {audit['raw_holdout_observations_parsed']}. Cal labels model selection 사용: {audit['calibration_labels_used_for_selection']}. 현재 적용 가능한 자동 검사 통과: {audit['all_applicable_checks_pass']}.

단, 감사 하위 에이전트가 과거 verification/results/05_summary.json을 열어 오래된 테스트 지표가 표시된 경계 위반이 있었다. 이 수치는 메인 에이전트에게 전달되지 않았고 이번 모델/특징/탐색공간 결정에 쓰지 않았다는 에이전트 진술을 기록했다. 별도 광범위 소스 검색도 verification Python 코드에 매칭됐다. 따라서 '과거 final artifact를 전혀 열지 않았다'고 주장할 수 없다. 이것과 이번 raw holdout 미파싱 및 신규 최종평가 미실행은 구분한다. 상세는 logs/boundary_incident.json과 leakage_audit.json에 있다.

## 모델 역할과 hyperparameter 선택

B0~B5는 최소/계절/산업 CBL/통계/선형/동적 기준선, M1과 M2는 주요 학습모델, M1-W는 고정 peak-weight ablation, C1은 제한적 residual challenger, R1은 선정에서 제외된 사전학습 참고모델이다.

LightGBM configuration: `{lock['lgbm']['selected_id']}` / `{json.dumps(lock['lgbm']['selected_config'],ensure_ascii=False)}`.
TCN architecture: `{lock['tcn']['selected_id']}` / `{json.dumps(lock['tcn']['selected_config'],ensure_ascii=False)}`.
CBL per-horizon choices: `{json.dumps(lock['cbl_by_horizon'],ensure_ascii=False)}`.

Configuration은 60/120/180/240분 ×3fold의 stop MAE로 한 번 결정한 뒤 모든 horizon에서 고정했다. score prediction 생성 전에 model_config_lock.json을 저장했다. 각 모델의 학습 iteration/epoch early stopping은 고정된 stop-only 규칙을 따르며 cal/score로 정하지 않았다. 모든 candidate와 탈락 설정의 stop 비교표도 보존한다.

## 전체 horizon 평균오차

AUC의 정확한 정의는 **equally spaced horizon-normalized mean error**이며, 13개 pooled horizon MAE 또는 Peak-MAE의 동일 가중 평균이다.

{table(display)}

## 1/2/3/4시간 대표 지점

{table(anchors[fields])}

## 13개 horizon 전체

{table(full[fields])}

## Paired 비교와 M* 선정 과정

MAIN10 모델은 동일 valid origin/target/horizon/fold intersection에서 비교했다. R1의 별도 유효행은 MAIN10의 표본을 줄이지 않는다. 아래 차이는 model_a error minus model_b error이며 음수가 a에 유리하다. 날짜 블록 bootstrap1000회의95% CI를 사용하고 horizon 간 의존성을 같은 날짜 추출로 유지했다.

{table(paired_focus[[c for c in ['model_a','model_b','metric','difference_a_minus_b','ci_low','ci_high','ci_status'] if c in paired_focus]])}

{table(decisions[[c for c in ['model','AUC_MAE','AUC_PeakMAE','peak_safeguard_excluded','C1_rejection_reasons','eligible','complexity_rank','statistical_tie_with_primary','selected_M_star'] if c in decisions]])}

최소 AUC-MAE eligible 모델은 {selection['primary_min_auc_model']}이고, 통계적으로 구분되지 않은 후보는 {selection['statistical_tie_models']}이다. 고정 complexity tie-break 후 최종 선택은 {selected or '판정불가'}이다. 임의 가중합은 사용하지 않았다.

## Fold stability와 D2 robustness

D2는 fit 학습기간에 같은 96-slot daily profile이 없던 score 날짜의 subset이다. 시계열 날짜나 학습/lag 이력은 삭제하지 않았다. 정확한 float64 profile hash와 불완전 profile 배제는 사전등록과 fold_manifest.csv에 정의한다. 주 분석은 D1이며 D2는 동일 학습모델의 robustness 평가다.

{table(stability)}

TCN/State-space의 순위·피크 safeguard·fold 방향은 위 표대로 보고하며 이름이나 복잡도를 이유로 채택하지 않는다. Residual은 M1보다 명확한 MAE 개선, 2/3 fold 방향, D2 유지가 모두 필요한 사전 gate로 판정했다. 나쁜 결과를 삭제하거나 별도 튜닝으로 살리지 않았다.

## 기존 결과와의 차이

이번 결과는 새 namespace에서 다시 학습/추론한 Phase C 결과다. 예전 05_experiments 수치를 새 결과로 재사용하지 않았다. 첨부 Phase B 임시 script의 full-data reader, excluded-exogenous validity filtering, old-D2 training-date removal은 사용하지 않았다. 따라서 예전 표본 수/결과와 숫자 일치를 주장하지 않는다. MSTL은 기존 [96,672]/28-day/AutoETS ZZN 사양을 유지하되 cal/score 재적합 및 양방향 보간을 막는 fit-only wrapper로 변경했으므로 기존 daily-refit MSTL과 실행 조건이 다르다. Chronos는 별도 context/representation budget을 갖는 reference이다.

## 실제 실행시간과 실패 기록

누적 실제 job wall time: {runtime.get('actual_job_seconds',0):.1f}초 (병렬 job이 있으면 합산시간이며 경과시간과 다름). 세부 시작/종료·단계별 시간은 logs/runtime.json, 개별 학습/추론 시간은 metric tables와 모델 runtime metadata를 따른다. TCN tuning의 모든 후보 비용과 선택 모델의 최종 학습 비용을 구분한다.

실행 중 예외 로그: {failures or '없음'}. 실패/재시작이 있었다면 원 기록을 보존하며 구현 수정의 근거와 영향은 별도 repair log에 기록한다. synthetic 테스트의 임시폴더 권한 오류는 모델 실패와 구분한다.

## 한계와 Phase D

단일 공장·제한 기간·증강 exact profile 반복, 과거 연구 단계의 평가 열람/반복 개발에 따른 선택 편향, 공식 단위/interval boundary 미확인, holiday 공식 검토 미완료를 유지한다. Nominal bootstrap CI는 독립적인 외부 검증이나 다중비교 보정 증거가 아니다. 실제 현장/장치/생산 제어 효과를 입증하지 않는다. 경보 F1·확률 보정·conformal은 수행하지 않았으며 cal은 Phase E용으로 남겨 두었다.

Phase D 준비: {'M*의 저장된 13-horizon development curve로 진행 가능; h* 결정은 별도 단계' if selected else 'M*가 판정불가이므로 바로 진행 불가'}. Final holdout은 다음 단계에서도 별도 승인된 최종평가 전까지 열지 않는다.
"""
    path=out/"phase_c_result.md"
    path.write_text(text,encoding="utf-8")
    return path
