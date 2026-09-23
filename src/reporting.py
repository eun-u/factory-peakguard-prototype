"""Build Korean report drafts and the supplied roadmap's evidence-backed update."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(frame, limit=30):
    if frame is None or frame.empty:
        return "해당 결과가 아직 생성되지 않았습니다."
    f = frame.head(limit).copy()
    def value(v):
        if pd.isna(v):
            return "—"
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "/")
    rows = ["| " + " | ".join(map(str, f.columns)) + " |", "| " + " | ".join(["---"]*len(f.columns)) + " |"]
    rows += ["| " + " | ".join(value(v) for v in row) + " |" for row in f.itertuples(index=False, name=None)]
    return "\n".join(rows)


def _read(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def diagnostics(df, out):
    from .viz import plt, configure
    configure()
    figures = out/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, label in zip(axes, [x.index.month, x.index.dayofweek, x.index.hour], ["월", "요일 (0=월)", "종료 시각"]):
        x.groupby(key).power.mean().plot.bar(ax=ax, color="#3E6283")
        ax.set(xlabel=label, ylabel="평균 전력 (원자료 단위)")
    fig.suptitle("F1-1 전력 분포 · 전체 자료 품질 진단")
    fig.tight_layout()
    fig.savefig(figures/"F1-1_power_distribution.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(x.index, x.power, color="#3E6283", linewidth=.3)
    zero = x.power.eq(0)
    repaired = x.time_repaired.astype(bool)
    ax.scatter(x.index[zero], x.power[zero], s=8, color="#B8741A", label="0값")
    ax.scatter(x.index[repaired], x.power[repaired], s=8, color="#B02A37", label="시간 복원 · 학습 제외")
    ax.set(title="F1-2 품질 플래그 위치", ylabel="전력 (단위 미확인)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures/"F1-2_quality_timeline.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    clean = x.loc[~x.time_repaired.astype(bool)]
    axes[0].hexbin(clean.production_target, clean.power, gridsize=40, mincnt=1, cmap="Blues")
    axes[0].set(xlabel="동시간 생산량 (사후 분석)", ylabel="15분 전력 (단위 미확인)", title="생산–전력 분포")
    axes[1].hist(clean.loc[clean.production_target.eq(0), "power"].dropna(), bins=35, color="#3E6283")
    axes[1].set(xlabel="15분 전력", ylabel="구간 수", title="생산량 0의 기저부하 분포")
    fig.suptitle("F1-3 상관관계 진단 · 인과 효과 또는 설비 상태가 아님")
    fig.tight_layout()
    fig.savefig(figures/"F1-3_production_baseload.png", dpi=140)
    plt.close(fig)
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.add_patch(Rectangle((0, 3.5), .85, .65, color="#DCE6EF"))
    ax.add_patch(Rectangle((.85, 3.5), .15, .65, color="#F6E7CF"))
    ax.text(.425, 3.82, "개발 구간 · 확장 학습 + 분리된 검증", ha="center", va="center")
    ax.text(.925, 3.82, "잠금 테스트", ha="center", va="center", fontsize=9)
    for row, end in enumerate([.45, .64, .85]):
        y = 2.6-row
        ax.add_patch(Rectangle((0, y), end-.20, .48, color="#3E6283"))
        ax.add_patch(Rectangle((end-.19, y), .055, .48, color="#7D8894"))
        ax.add_patch(Rectangle((end-.125, y), .055, .48, color="#B8741A"))
        ax.add_patch(Rectangle((end-.06, y), .06, .48, color="#2B7564"))
    ax.text(0, -.1, "파랑 학습 / 회색 조기종료 / 황색 보정 / 초록 평가 · 경계에 예측거리 간격", fontsize=10)
    ax.set(xlim=(0, 1), ylim=(-.3, 4.8), title="T1-2 누수 방지 분할 개념도 · 실제 시각은 폴드 명세 참조")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures/"T1-2_split_protocol.png", dpi=140)
    plt.close(fig)
    return ["F1-1_power_distribution.png", "F1-2_quality_timeline.png", "F1-3_production_baseload.png"]


def generate_reports(df, development, cfg, outdir):
    out = Path(outdir)
    report = Path("report")
    report.mkdir(exist_ok=True)
    diagnostics(df, out)
    quality = json.loads((out/"logs/data_quality.json").read_text(encoding="utf-8"))
    metrics, selection = development["metrics"], development["selection"]
    selection_overview = pd.DataFrame([
        {"horizon_minutes": int(h)*15, **{k: v.get(k) for k in ("point_model", "persistence", "seasonal", "cbl", "conformal")}}
        for h, v in selection["by_horizon"].items()])
    from .model_reporting import build_model_report
    build_model_report(development["predictions"], selection, cfg, out)
    main = metrics.loc[metrics.horizon == cfg["primary_horizon"]]
    columns = [c for c in ["fold", "model", "n", "mae", "peak_mae", "episode_f1", "false_alarms_positions", "top_coverage_0.95"] if c in main]
    summary = main[columns]
    manifest = development.get("folds", [])
    boundary = cfg["split"]["test_start_origin"]
    disclaimer = ("이 초안의 새 모델 수치는 **개발 교차검증** 결과다. 2026-10-01 18:00 KST 전에는 최종 테스트를 실행하지 않는다. "
                  "사전 검증에서 고정 설정으로 테스트를 한 번 본 이력이 있으며 verification/의 성능은 신규 모델의 성능으로 재사용하지 않는다.\n\n")
    inventory = pd.DataFrame({"variable": df.columns, "dtype": [str(t) for t in df.dtypes], "missing": df.isna().sum().to_numpy()})
    inventory.to_csv(out/"tables/T1-1_variable_dictionary.csv", index=False)
    ch1 = ("# 1 데이터 이해 및 진단\n\n" + disclaimer +
           "## 생산단위와 측정 구조\n\n원자료 한 행은 한 시간이며 전력 네 열은 15분 종료 구간으로 해석한다(A4). "
           "설비 ID·제품·실제 교대 정보가 없어 시간대·요일·생산량을 대리변수로 사용한다.\n\n" +
           f"정규화된 관측 {len(df):,}개, 시간 복원 구간 {int(df.time_repaired.sum()):,}개, 0값 {int(df.power.eq(0).sum()):,}개. "
           f"범위는 {df.index.min()}부터 {df.index.max()}까지다.\n\n" +
           "## 단위·품질 및 전처리\n\n원본 해시와 원자료는 보존한다. 시간 복원 행과 그 관측에 의존하는 특징·목표를 제외한다. "
           "0값은 비가동 가능성이 있으나 확인된 비가동 라벨이 아니므로 제거하지 않는다. IQR 이상치는 진단이며 자동 삭제하지 않는다. "
           "이 장의 원자료 품질·분포 진단은 전체 자료를 대상으로 하며, 모델 적합·튜닝·채택 평가는 개발 구간만 사용한다. "
           "평균 열의 네 값 평균 일치 여부는 수요전력 단서이며 kW 확정 증거가 아니다.\n\n" +
           "```json\n" + json.dumps(quality, ensure_ascii=False, indent=2, default=str) + "\n```\n\n" +
           "## 변수 사전과 가용 시점\n\n" + md_table(inventory, 60) +
           "\n\n전력은 구간 종료 직후, 생산은 해당 시간 종료 후만 입력한다. 실측 기상·인원·인건비는 사후 분석 전용이다. "
           "동시간 평균 전력은 입력하지 않는다. 상세 가용 표는 PROJECT_DESIGN.md 3절이다.\n\n" +
           f"## 검증 전략\n\n고정 테스트 첫 원점 {boundary}. 개발은 그 이전만 사용한다. "
           "3개 확장 폴드에서 적합·조기종료·보정·평가를 분리하고 예측거리만큼 간격을 둔다. "
           "특징의 관측 최대 시각≤원점을 검사한다. 12월 부재와 9월 중순까지만 관측한 한계 때문에 계절 일반화를 주장하지 않는다.\n\n" +
           "원자료 분포와 품질 그림은 아래 생성 그림에 제시한다.\n")
    ch2 = ("# 2 AI 예측모델 개발\n\n" + disclaimer +
           "T1은 한 시간 뒤15분, 보조 곡선은15분·4시간·24시간이다. T2는 익일 최대15분이다. "
           "비교 모델은 persistence3종·계절 나이브3종·CBL5종·LightGBM 점예측·분위수다. "
           "CBL은 정산용 규칙을15분 자료에 준용하며 실제 DR 제외일 정보가 없어 공식 정산과 동일하지 않다.\n\n" +
           "## 폴드별 성능\n\n" + md_table(summary, 100) +
           "\n\n## 사전 규칙에 따른 선정과 채택\n\n" + md_table(selection_overview) +
           "\n\n[전체 채택·비교 CI 기록](../outputs/logs/development_selection.json). 피크 MAE CI, 에피소드 F1, FP, 상위 q95 오차, 단순성 순서로 선정한다. "
           "일 단위 F1은 시간 위치를 놓쳐도 높을 수 있는 느슨한 보조 지표다. "
           "개발 선택 후 CI에는 다중 선택 편향이 남을 수 있으므로 최종 일반화 판정은 잠금 테스트에서 한다. "
           "시계열 표본은 교환가능하지 않아 conformal의 분포무관 보장을 주장하지 않는다.\n\n" +
           "## 실행 분할 증거\n\n" + md_table(pd.DataFrame(manifest), 20) + "\n")
    files = sorted((out/"tables").glob("*.csv"))
    def table_section(patterns):
        pieces = []
        for p in files:
            if any(k in p.stem for k in patterns):
                pieces.append(f"### {p.stem}\n\n[전체 표](../{p.as_posix()})\n\n"+md_table(_read(p), 15))
        return "\n\n".join(pieces) or "결과 생성 여부는 실행 상태 파일을 확인한다."
    fva = _read(out/"tables/T2-1_fva.csv")
    final_row = fva.loc[fva.stage.eq("최종 점예측")].iloc[0]
    coverage_table = _read(out/"tables/coverage_comparison.csv")
    chosen = selection["by_horizon"][str(cfg["primary_horizon"])]
    selected_quantile = "lgbm_quantile_" + chosen.get("conformal", "a")
    top = coverage_table.loc[coverage_table.model.eq(selected_quantile) & coverage_table.alpha.eq(.95) & coverage_table.segment.eq("상위 예측")].iloc[0]
    cbl_low, cbl_high = final_row.get("vs_cbl_gain_ci_low", np.nan), final_row.get("vs_cbl_gain_ci_high", np.nan)
    cbl_judgment = "개발 표본에서는 개선 CI 하한이 0보다 크다" if cbl_low > 0 else "CBL 대비 유의한 피크 오차 개선을 입증하지 못했다"
    narrative = (f"\n\n## 핵심 결과 해석\n\n주 과제의 선택 후보는 `{chosen['point_model']}`이다. "
                 f"피크 위치 MAE는 {final_row.peak_mae:.3f}, 에피소드 F1은 {final_row.episode_f1:.3f}이다. "
                 f"CBL 대비 피크 MAE 감소량의 날짜 블록 95% CI는 [{cbl_low:.3f}, {cbl_high:.3f}]로, {cbl_judgment}. "
                 "개발 선택 후 결과이므로 최종 일반화 성능으로 해석하지 않는다.\n\n"
                 f"위험 출력은 별도의 분위수 모델과 보정 {chosen.get('conformal', 'a').upper()}를 쓴다. "
                 f"상위 예측 구간 q95 커버리지는 {top.coverage:.4f}, 목표 0.95와의 절대 차이는 {top.coverage_error:.4f}이며 "
                 f"유효 구간 수는 {int(top.scoring_n):,}개다. 구간별 불확실성과 보정 전후 차이를 아래 표에 제시한다.\n")
    ch2 += narrative
    fold_coverage = _read(out/"tables/coverage_by_fold.csv")
    qrows = fold_coverage.loc[fold_coverage.model.eq(selected_quantile)]
    if not qrows.empty:
        worst = qrows.sort_values("top_q95_coverage").iloc[0]
        ch2 += (f"\n전체 상위 구간의 날짜 블록 95% CI는 [{top.ci_low:.4f}, {top.ci_high:.4f}]이며 "
                f"관측 날짜는 {int(top.target_days)}일이다. 가장 낮은 폴드 {int(worst.fold)+1}의 상위 커버리지는 "
                f"{worst.top_q95_coverage:.4f} (상위 표본 {int(worst.top_n)}개, {int(worst.target_days)}일, "
                f"95% CI [{worst.ci_low:.4f}, {worst.ci_high:.4f}])이다. "
                "합친 구간의 CI와 특정 시기의 국소 실패를 구분한다. 시기별 고부하 분포 변화와 적은 날짜 수가 영향을 줄 수 있지만 "
                "원인은 확정할 수 없다. 날짜별 미포함 수는 coverage_top_by_day.csv에 제시한다.\n")
    ch2 += "\n\n## 예측 부가가치와 커버리지\n\n" + table_section(["T2-1", "T2-2", "coverage", "horizon", "t2_development", "expected_exceedance", "reliability"])
    ch3 = ("# 3 영향요인 및 오류분석\n\n"+disclaimer+
           "이 장의 FN·FP는 선택된 분위수 모델의 보정된 초과확률 경보를 사용한다. "
           "2장 점예측 모델의 경보 임계값과 구분하며, 연속 확인·준비시간을 적용한 운영 경보는 4장에 따로 제시한다. "
           "위치 재현율과 일대일 매칭 에피소드 재현율도 다르다. 긴 경보 하나가 여러 실제 에피소드를 덮어도 하나만 적중으로 센다. "
           "시간대3구간은 대리 교대이며 실제 교대 라벨이 아니다. 동시간 생산·기상은 사후 기준선 설명에만 쓴다. "
           "생산 연관형과 잔차 초과형은 통계적 분해이며 설비 원인이나 과다 사용의 인과 판정이 아니다. "
           "조건 탐색은 다중 비교 조정 없이 제시한다.\n\n"+
           table_section(["error", "condition", "evening", "peak_type", "energy", "importance", "applicability"]))
    conditions = _read(out/"tables/error_conditions.csv")
    if not conditions.empty:
        evening = conditions.loc[conditions.factor.eq("time_block") & conditions.value.eq("16-24")]
        episodes = _read(out/"tables/evening_episode_comparison.csv")
        if not evening.empty:
            row = evening.iloc[0]
            counts = dict(zip(episodes.status, episodes.episodes)) if not episodes.empty else {}
            ch3 += (f"\n\n## 저녁 구간의 실제 개발 결과\n\n16~24시 피크 위치 {int(row.peak_n)}개 중 적중은 {int(row.tp)}개, "
                    f"누락은 {int(row.fn)}개로 위치 재현율은 {row.peak_recall:.4f}다. "
                    f"16시 이후 시작한 에피소드는 TP {int(counts.get('TP', 0))}개, FN {int(counts.get('FN', 0))}개다. "
                    "매칭 규칙의 영향을 포함해 해석한다. 사전 검증의 0.143은 다른 평가 구간·경보 정의의 역사적 수치로, "
                    "이번 수치와 직접 비교해 개선률을 계산하지 않는다. ‘저녁이 최대 약점’이라는 문장은 현재 결과로 별도 검증해야 하며 전제로 쓰지 않는다.\n")
    ch4 = ("# 4 현장 활용방안\n\n"+disclaimer+
           "경보는 p_exceed>C/L 또는 margin<0이다. h16은 생산 일정 검토, h4는 부하 이동 조치 검토다. "
           "오경보율의 운전시간은 유효 예측 행 수×15분인 평가시간이며 실제 설비 가동시간은 미관측이다. "
           "30분 준비시간은 검증된 현장 사실이 아닌 가정이며15·60분을 비교한다. "
           "본 모델은 디맨드 컨트롤러의15분 이내 반응형 제어를 보완한다.\n\n"
           "## 비용 해석과 실행 가능성\n\n2021 관측에2026 정책을 적용하는 반사실 시간대 비교다. "
           "미확인 계약 단가 대신1·2·3 순서가중을 쓴 결과는 실제 요금 또는 비용 비중이 아니다. "
           "낮으로 이동하는 사후 안과 원점+준비시간 이후에만 이동하는 운영 안을 구분한다. "
           "저녁 경보가 발생한 뒤 지난 낮으로 이동한 결과를 실행 가능한 절감액으로 세지 않는다. "
           "적용 이동이0건이면 변화량0은 무조치 결과이며, 부하 이동 효과가 없다는 추정이 아니다. "
           "생산 이동 계수는 학습 자료 상관관계일 뿐 인과 효과가 아니며 설비·납기 제약이 없어 현장 검증이 필요하다.\n\n"+
           table_section(["alert", "rev", "tariff", "shift"])+
           "\n\n조치 후 관측 피크가 사라지는 예방의 역설 때문에 CBL 준용 기준부하와 실제 차이를 추적하도록 제안한다. "
           "독립 대조가 없으므로 그 차이만으로 인과 효과를 증명하지 않는다.\n")
    ch5 = ("# 5 창의성 및 차별성\n\n"+disclaimer+
           "CBL 현업 기준선, 공휴일 채택 판정, 이분산 분위수 보정, 확률·여유 경보, 예측거리별 조치, "
           "2026 시간대 재평가를 각각 성능 변화로 평가한다. 선택 기법은 사전 기준을 만족한 경우에만 채택하며 기각도 공개한다.\n\n"+
           table_section(["fva", "adoption", "horizon", "coverage"])+
           "\n\n선정의 전체 근거는 2장과 outputs/logs/development_selection.json에 있다. 실현되지 않은 개선을 주장하지 않는다.\n")
    ch6 = ("# 6 코드 및 재현성\n\n"+disclaimer+
           "README의 설치 후 `python run_all.py`로 데이터, 개발모델, 분석, 그림·표, 보고서, 제출 준비ZIP을 생성한다. "
           "`--only data|development|final|analysis|report|package`, `--from STEP`, `--rebuild-dev`를 지원한다. "
           "원본과 verification 해시를 실행 전후 확인하고 코드·설정·기준·원본 해시가 다르면 개발 캐시를 거부한다. "
           "최종 테스트 전에는 날짜 잠금을 확인한다.\n\n"
           "실제 실행시간·오류·캐시 상태: [run_status.json](../outputs/logs/run_status.json). "
           "자동 테스트: `python -m pytest tests -q`. "
           "신규 가상환경 설치·전체 실행 결과와 독립 검토는 PROGRESS.md에 기록한다.\n\n"
           "## 제출 경계\n\n현 단계 ZIP은 개발 재현용 준비물이다. 보고서 hwpx·PDF, 설문 캡처, 발표 PDF·PPT, "
           "최종 테스트 파일과 포털 제출 완료 화면이 모두 확보되기 전에는 최종 제출 완료라고 부르지 않는다.\n")
    cold_steps = _read(out/"tables/T6-1_cold_run_steps.csv")
    if not cold_steps.empty:
        ch6 += ("\n\n## 캐시 없는 독립 재현 실행\n\n재현 ZIP을 별도 폴더에 풀고 모델·개발 캐시 없이 실행했다. "
                "예측·지표·선정의 원본 실행과의 일치 여부는 [재현 검증](../outputs/logs/fresh_reproduction.json)에 기록한다.\n\n"
                + md_table(cold_steps) + "\n\n이 시간표는 독립 전체 실행 기록이다. 이후 캐시 사용 실행과 예약 최종 평가의 최신 상태는 run_status.json을 따른다.\n")
    # A later scheduled run appends locked holdout evidence without changing development selection.
    freeze_file = out/"logs/freeze_record.json"
    if freeze_file.exists():
        frozen = json.loads(freeze_file.read_text(encoding="utf-8"))
        if frozen.get("status") == "completed":
            final_metrics = _read(out/"tables/final_test.csv")
            ch2 += "\n\n## 동결 후 최종 테스트 결과\n\n아래 표는 동결된 모델의 일회 평가다. 위 개발표와 구분하며 결과를 보고 모델을 바꾸지 않았다.\n\n"+md_table(final_metrics, 100)
            final_t2 = _read(out/"tables/t2_final_test.csv")
            ch2 += "\n\n### T2 최종 보조과제\n\n"+md_table(final_t2, 10)
            for chapter_id, keywords in [(3, ["error", "evening", "peak_type"]), (4, ["tariff", "shift", "alert", "economic"] )]:
                text = "\n\n## 동결 테스트의 사후 분석\n\n모델·기준선 선택을 바꾸지 않는 기술통계다.\n\n"
                for path in sorted((out/"final_analysis/tables").glob("*.csv")):
                    if any(word in path.stem for word in keywords):
                        text += f"\n\n### {path.stem}\n\n[전체 표](../{path.as_posix()})\n\n"+md_table(_read(path), 12)
                if chapter_id == 3:
                    ch3 += text
                else:
                    ch4 += text
            ch6 += "\n\n동결·파일해시 기록: [freeze_record.json](../outputs/logs/freeze_record.json). 최종평가 완료는 포털 제출 완료와 다르다.\n"
    chapters = [ch1, ch2, ch3, ch4, ch5, ch6]
    groups = [["F1-", "T1-2"], ["F2-", "horizon_curve", "reliability"],
              ["condition", "evening", "peak_type", "horizon_importance"],
              ["F4-1", "alert", "rev", "tariff", "shift"], [], []]
    for index, prefixes in enumerate(groups):
        images = [path for path in sorted((out/"figures").glob("*.png")) if any(k in path.stem for k in prefixes)]
        if images:
            chapters[index] += "\n\n## 생성 그림\n\n" + "\n\n".join(f"![{path.stem}](../{path.as_posix()})" for path in images)
    names = ["ch1_data", "ch2_model", "ch3_errors", "ch4_field", "ch5_novelty", "ch6_repro"]
    for name, content in zip(names, chapters):
        (report/f"{name}.md").write_text(content, encoding="utf-8")
    (report/"REPORT_DRAFT.md").write_text("\n\n---\n\n".join(chapters), encoding="utf-8")
    Path("slides").mkdir(exist_ok=True)
    outline = ["문제와 운영 범위", "데이터 구조와 미확인 단위", "품질·누수 방지", "시간순 개발·테스트 잠금", "CBL과 통계 기준선", "FVA와 선정", "분위수 보정·커버리지", "예측거리별 성능", "저녁 피크 미탐", "생산 연관·잔차 분해", "경보·선행시간", "요금 시간대 시나리오", "이동 가능성과 한계", "재현성·결론"]
    Path("slides/outline.md").write_text("# 발표 구성안\n\n개발 결과 기반 초안. 최종 테스트 잠금 해제 후 해당 표를 대체한다.\n\n"+"\n".join(f"{i+1}. {s}" for i, s in enumerate(outline)), encoding="utf-8")
    frozen_complete = freeze_file.exists() and json.loads(freeze_file.read_text(encoding="utf-8")).get("status") == "completed"
    generate_roadmap(selection, frozen_complete=frozen_complete)
    from scripts.build_slides import build_deck
    slide_result = build_deck(Path.cwd())
    for key in ("html", "speaker_notes"):
        slide_result[key] = Path(slide_result[key]).relative_to(Path.cwd()).as_posix()
    return {"chapters": [f"report/{n}.md" for n in names], "roadmap": "docs/roadmap.html",
            "slides": slide_result,
            "scope": "frozen_test_and_development_draft" if frozen_complete else "development_draft"}


def generate_roadmap(selection=None, *, frozen_complete=False):
    phases = [
        ("0 기반 구축", "9/23–9/24", 0, 2, "원본·검증 해시 보존, 기존 MAE 감사 재현, 저장소 정리"),
        ("1 데이터 진단", "9/25–9/27", 2, 5, "15분 전개, 복원 의존 제외, 변수 사전·공휴일·단위 단서"),
        ("2-1 기준선", "9/27", 4, 5, "Persistence·계절 나이브·CBL 5종, 사전 기준 고정"),
        ("2-2~3 예측", "9/28–9/30", 5, 8, "LightGBM, 공휴일·가중, 분위수 A/B 보정"),
        ("2-4~6 위험·동결", "9/30–10/1", 7, 9, "초과확률·기대초과량·예측거리, 10/1 18시 동결"),
        ("3 오류분석", "10/1–10/3", 8, 11, "조건 FN·FP, 저녁 사례, 중요변수, 피크 유형"),
        ("4 활용·차별성", "10/3–10/4", 10, 12, "경보·REV·요금 시간대·실행 가능 이동"),
        ("5 재현·보고서", "10/4–10/6", 11, 14, "새 환경 실행, 장별 초안, hwpx 인계"),
        ("6~7 발표·제출", "10/6–10/7", 13, 15, "발표 PPT·PDF, 블라인드·포털 제출"),
        ("예비", "10/8", 15, 16, "문제 대응만 · 최종 마감23:59"),
    ]
    dates = pd.date_range("2026-09-23", periods=16)
    heads = "<th>단계</th>"+"".join(f'<th class="{"weekend" if d.dayofweek>=5 else ""}">{d.month}/{d.day}</th>' for d in dates)
    rows = []
    for label, period, start, end, text in phases:
        cells = "".join(f'<td class="{"active" if start<=i<end else "weekend" if d.dayofweek>=5 else ""}"><span class="sr">{text if start<=i<end else ""}</span></td>' for i,d in enumerate(dates))
        rows.append(f'<tr><th>{label}<small>{period}</small></th>{cells}</tr>')
    phase_list = "".join(f'<div class="phase"><div><b>{a}</b><small>{b}</small></div><p>{e}</p></div>' for a,b,c,d,e in phases)
    choices = ""
    if selection:
        choices = '<div class="scroll"><table><tr><th>예측거리</th><th>개발 점모델</th><th>CBL 대표</th><th>보정</th></tr>'+"".join(
            f'<tr><td>{int(h)*15}분</td><td>{html.escape(str(v.get("point_model","미정")))}</td><td>{html.escape(str(v.get("cbl","미정")))}</td><td>{html.escape(str(v.get("conformal","미정")))}</td></tr>' for h,v in selection.get("by_horizon",{}).items())+'</table></div>'
    text = '''<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>과제 ⑤ 실행 로드맵</title><style>
:root{--paper:#eef1f4;--panel:#f8fafb;--ink:#1c2a39;--soft:#4f5f70;--line:#c9d2db;--steel:#3e6283;--steel-bg:#dce6ef;--warn:#865007;--warn-bg:#f6e7cf;--pass:#286653;--pass-bg:#d8ede7}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 'Malgun Gothic','Apple SD Gothic Neo',sans-serif;overflow-x:clip}main{max-width:1040px;padding:40px 24px 64px;margin:auto}h1,h2,h3{line-height:1.35}h1{font-size:clamp(1.8rem,4vw,2.7rem);margin:8px 0}h2{font-size:1.4rem;margin:0 0 12px}p{margin:0 0 16px;max-width:75ch}a{color:var(--steel);text-underline-offset:3px}header{border-left:6px solid var(--steel);padding-left:22px}header p,.muted,small{color:var(--soft)}small{display:block}nav{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:24px}nav a{display:inline-flex;align-items:center;min-height:44px}section{margin-top:48px}.keys{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);background:var(--panel);margin-top:24px}.keys>div{padding:16px;border-right:1px solid var(--line);min-width:0}.keys b{display:block;font-size:1.14rem}.keys span{font-size:.85rem;color:var(--soft)}.callout{border-left:5px solid var(--warn);background:var(--warn-bg);padding:18px;margin-top:22px}.callout b{color:var(--warn)}.scroll{position:relative;overflow-x:auto;max-width:100%;border:1px solid var(--line);background:var(--panel)}table{border-collapse:collapse;width:100%}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:.88rem}th{color:var(--soft)}.gantt{min-width:900px;table-layout:fixed}.gantt th:first-child{width:190px}.gantt td,.gantt th{padding:10px 3px;border-right:1px solid var(--line)}.gantt th:first-child{padding-left:12px}.gantt .active{background:var(--steel);border-top:8px solid var(--panel);border-bottom:8px solid var(--panel)}.weekend{background:#e1e7ed}.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}.pipeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.pipeline>div{background:var(--panel);border-top:4px solid var(--steel);padding:16px;min-width:0}.pipeline b{display:block}.pipeline p{font-size:.9rem}.formula{background:var(--steel-bg);padding:16px;font-size:1.15rem}.phase{display:grid;grid-template-columns:180px minmax(0,1fr);gap:20px;padding:18px 0;border-top:1px solid var(--line)}.phase p{margin:0}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.box{background:var(--panel);padding:20px;border:1px solid var(--line)}.box h3{margin:0 0 12px}.box li{margin-bottom:8px}.band{display:grid;grid-template-columns:repeat(24,1fr);height:28px;background:var(--steel-bg);margin:12px 0}.band span{background:var(--warn);grid-column:19/22}.links{overflow-wrap:anywhere}footer{border-top:1px solid var(--line);margin-top:48px;padding-top:20px;color:var(--soft);font-size:.85rem}@media(max-width:760px){main{padding:24px 18px 48px}.keys,.pipeline{grid-template-columns:repeat(2,minmax(0,1fr))}.keys>div{border-bottom:1px solid var(--line)}.two{grid-template-columns:1fr}.phase{grid-template-columns:1fr;gap:8px}.scroll table:not(.gantt){min-width:620px}header{padding-left:16px}}@media(prefers-color-scheme:dark){:root{--paper:#141b23;--panel:#1b242e;--ink:#e4eaf0;--soft:#afbdc9;--line:#3f5162;--steel:#88b1d3;--steel-bg:#243b50;--warn:#e9ad59;--warn-bg:#3a2d19;--pass:#77cbb5}.weekend{background:#283544}}
</style></head><body><main>
<header><small>제6회 K-인공지능 제조데이터 분석 경진대회 · 과제 ⑤</small><h1>전력피크 예측 실행 로드맵</h1><p>피크의 발생 가능성과 초과 크기, 예측의 불확실성을 운영 판단으로 연결합니다. 2026년9월24일 갱신 · 설계 일정과 실제 완료 증거를 구분합니다.</p></header>
<div class="keys"><div><b>10월1일18시</b><span>모델 동결·테스트 날짜 잠금</span></div><div><b>10월5~6일</b><span>hwpx·PDF 사람 인계</span></div><div><b>10월7일</b><span>내부 제출 목표</span></div><div><b>10월8일23:59</b><span>문서상 공식 마감</span></div></div>
<nav aria-label="문서 탐색"><a href="#structure">시스템 구조</a><a href="#schedule">일정</a><a href="#evidence">실행 결과</a><a href="#rules">채택 기준</a><a href="#handoff">인계 사항</a></nav>
<div class="callout"><b>현재는 개발 검증 단계입니다.</b> 최종 테스트는10월1일18시 전까지 열지 않습니다. 설문 캡처·실제 계약 단가·포털 완료 증거는 미확보이며, 완료로 표시하지 않습니다.</div>
<section id="structure"><h2>시스템 구조</h2><p>기준선 비교에서 위험 출력까지 동일한 시점과 누수 방지 규칙을 적용합니다.</p><div class="pipeline"><div><b>1 기준선</b><p>Persistence3종<br>계절 나이브3종<br>CBL 준용5종</p></div><div><b>2 점예측</b><p>LightGBM 직접 모델<br>공휴일 채택 판단<br>피크 가중1·2·4</p></div><div><b>3 불확실성</b><p>분위수0.1~0.975<br>A 전역 보정<br>B 예측 크기별 보정</p></div><div><b>4 위험·조치</b><p>초과 확률·기대 초과량<br>4시간 전 검토<br>1시간 전 조치 검토</p></div></div><p class="formula">여유 = 관리 목표 T − 보정된 q95</p><p class="muted">T는 기본 학습95백분위, q95는95% 상단 예측입니다. 여유&lt;0 또는 초과확률&gt;C/L이면 경보 후보가 됩니다. 실제 커버리지는 별도 평가합니다.</p></section>
<section><h2>요금 재평가와 조치 가능성</h2><p>2026년4월 개편은 봄·여름·가을의 낮·저녁 시간대를 바꿉니다. 18~21시 승격은 공식 발표로 확인했으며 겨울과 주말은 별도 규칙을 적용합니다.</p><div class="band" aria-label="24시간 중18시부터21시까지 강조"><span></span></div><p>사전 검증의16~24시 재현율0.143은 역사적 결과입니다. 새 모델의 약점 여부는 개발 결과로 다시 측정합니다. 단가 미확인 시1·2·3 가중치는 비교용 가정이며 실제 비용이 아닙니다.</p><div class="callout"><b>실행 제약</b> 저녁에 경보를 받고 이미 지난 낮11~15시로 생산을 옮길 수는 없습니다. 사후 가정과 준비시간 이후에만 가능한 운영안을 구분합니다.</div><p><a href="SOURCES.md">공식 출처</a> · <a href="tariff_sources.md">요금 시간대 상세 검증</a></p></section>
<section id="schedule"><h2>일정표</h2><p class="muted">회색은 주말입니다. 표는 좌우로 스크롤할 수 있습니다. 일정은 완료 증거가 아닙니다.</p><div class="scroll" tabindex="0" aria-label="단계별 일정표"><table class="gantt"><thead><tr>__HEADS__</tr></thead><tbody>__ROWS__</tbody></table></div></section>
<section><h2>데이터 사용 규칙</h2><div class="two"><div class="box"><h3>개발 구간</h3><p>고정 테스트 경계 이전만3폴드로 평가합니다. 적합·조기종료·보정·평가를 나누고 예측거리만큼 간격을 둡니다.</p></div><div class="box"><h3>테스트 구간</h3><p>첫 원점2021-08-09 09:45. 사전 검증1회 열람을 공개하고, 새 모델은10월1일 동결 후1회 평가합니다.</p></div></div><p>관측 가용 시각≤원점. 미래 기상·생산량 입력 금지. 시간 복원48행과 해당 관측을 사용한 특징·목표는 제외합니다. CBL 참고값도 원점에 관측 가능한 자료만 씁니다.</p></section>
<section><h2>단계별 작업과 완료 기준</h2>__PHASES__</section>
<section id="evidence"><h2>개발 결과와 문서</h2><p>아래 모델은 개발 결과로 선택한 후보입니다. 최종 테스트 결과가 아닙니다.</p>__CHOICES__<nav class="links"><a href="../report/REPORT_DRAFT.md">보고서1~6장</a><a href="../PROGRESS.md">진행 기록</a><a href="../DECISIONS.md">결정 기록</a><a href="../outputs/tables/development_cv.csv">개발 성능표</a><a href="../outputs/logs/run_status.json">실행 증거</a></nav></section>
<section id="rules"><h2>사전 채택 기준</h2><div class="scroll" tabindex="0"><table><tr><th>기법</th><th>개발 채택 조건</th></tr><tr><td>공휴일 특징</td><td>휴일·전후일 MAE 개선 CI 하한&gt;0, 전체 악화 없음</td></tr><tr><td>피크 가중</td><td>피크 MAE 개선, FP 증가20% 이내</td></tr><tr><td>Mondrian B</td><td>상위 q95 커버리지 오차 감소, 전체 pinball 악화5% 이내</td></tr><tr><td>기대 초과량</td><td>에피소드 순위상관95% CI 하한&gt;0</td></tr><tr><td>확률B</td><td>F1 또는 Brier 개선 CI 하한&gt;0 · 기본 실행 비활성</td></tr></table></div></section>
<section><h2>가정과 범위</h2><p>A1 한 시간 후15분·익일 최대, A2 정량 평가 미확정, A3 단위 미확인, A4 구간 종료 시각, A5 학습 상위5% 피크, A6 시간·요일·생산 대리변수, A7 별도 테스트 없으면 내부 예측파일, A8 완료 생산량만 입력합니다.</p><p>설비 최적화·실제 절감액·계절 일반화 입증·인과 효과는 범위 밖입니다. 답변이 오면 가정을 기록하되 동결 후 테스트를 보고 모델을 바꾸지 않습니다.</p></section>
<section><h2>일정이 밀릴 때</h2><div class="two"><div class="box"><h3>삭감 순서</h3><ol><li>파운데이션 모델</li><li>확률 분류B</li><li>피크 정의 민감도</li><li>기대 초과량</li><li>4시간 예측</li><li>PDP·ICE</li><li>이동 비율 민감도</li><li>피크 시점·크기 오차</li></ol></div><div class="box"><h3>삭감 금지</h3><ul><li>CBL·FVA</li><li>Mondrian 또는 실패의 정량 보고</li><li>저녁 피크 미탐</li><li>2026 요금 시간대 분석</li><li>조건별 FN·FP</li><li>재현성·블라인드·설문 캡처</li></ul></div></div></section>
<section id="handoff"><h2>무개입 진행과 사람 인계</h2><p>10월5~6일 전까지 공개자료 확인과 코드·검증·초안은 자동으로 진행합니다. 준비시간30분은 가정으로 유지하고 민감도를 제공합니다. 동결은 사전 위임하되 날짜 잠금을 지킵니다.</p><div class="scroll" tabindex="0"><table><tr><th>항목</th><th>상태·인계</th></tr><tr><td>주최측 문의</td><td>문안 준비, 발송·답변 미확인</td></tr><tr><td>설문 캡처</td><td>본인 응답 필요·미확보. 대신 답하거나 완료 화면을 만들지 않음</td></tr><tr><td>계약 요금</td><td>전압·선택요금 미확인, 단가 공란</td></tr><tr><td>hwpx·PDF</td><td>10/5~6 사용자 서식 작업, 휴먼명조14·10,줄간격160</td></tr><tr><td>포털 제출</td><td>실제 제출·완료 화면 확보 전 미완료</td></tr></table></div><p><a href="../submission/HANDOFF.md">인계 체크리스트</a> · <a href="organizer_inquiry.md">문의 문안</a></p></section>
<footer>기준: <a href="../CLAUDE.md">CLAUDE.md</a> · <a href="../PROJECT_DESIGN.md">PROJECT_DESIGN.md</a> · verification/ 동결. 이 페이지는 python run_all.py의 보고서 단계에서 생성합니다.</footer>
</main></body></html>'''
    text = text.replace("__HEADS__", heads).replace("__ROWS__", "".join(rows)).replace("__PHASES__", phase_list).replace("__CHOICES__", choices)
    text = text.replace('</section>\n<section id="rules">',
        '<p class="links"><a href="../slides/development_deck.html">발표 초안 14장</a> · '
        '<a href="../slides/development_deck.pdf">발표 PDF</a> · <a href="DELIVERABLES.md">산출물 대응표</a></p></section>\n<section id="rules">')
    text = text.replace("<h1>전력피크 예측 실행 로드맵</h1>", "<h1>전력피크 예측 로드맵</h1>")
    text = text.replace('<span></span></div><p>사전 검증', '<span></span></div><p class="muted">0시 — 6시 — 12시 — <b>18~21시 강조</b> — 24시</p><p>사전 검증')
    text = text.replace('<h2>사전 채택 기준</h2><div', '<h2>사전 채택 기준</h2><p class="muted">좁은 화면에서는 표를 좌우로 스크롤하세요.</p><div')
    text = text.replace('<h2>무개입 진행과 사람 인계</h2>', '<h2>무개입 진행과 사람 인계</h2><p class="muted">표는 좌우로 스크롤할 수 있습니다.</p>')
    if frozen_complete:
        text = text.replace("현재는 개발 검증 단계입니다.</b> 최종 테스트는10월1일18시 전까지 열지 않습니다.",
                            "동결 모델의 최종 평가가 완료되었습니다.</b> 일회 평가 기록과 파일 해시를 보존합니다.")
        text = text.replace("최종 테스트 결과가 아닙니다.</p>", "최종 테스트 표는 <a href=\"../outputs/tables/final_test.csv\">별도 파일</a>에서 확인합니다.</p>")
    Path("docs/roadmap.html").write_text(text, encoding="utf-8")
