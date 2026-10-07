"""Phase F evidence-limited scientific reports.

EXPLORE tables are descriptive. CONFIRM is opened only after the frozen
candidate and one-time completion hashes are independently checked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_f.diagnostics import summarize_errors
from phase_f.registry import config_hash, sha256
from src.viz import configure, save

FAMILIES = tuple(f"F{i}" for i in range(12))
HORIZONS = tuple(range(4, 17))
BOUNDARY = pd.Timestamp("2021-08-09 09:45")
COLORS = ("#1C2A39", "#3E6283", "#B8741A", "#398577", "#906381")
FIGURES = ("horizon_mae_peak.png", "context_length.png",
           "feature_ablation.png", "peak_signed_errors.png",
           "representative_week.png")


def _out(root: Path) -> Path:
    out = Path(root).resolve() / "outputs" / "phase_f"
    if not out.is_dir():
        raise FileNotFoundError("Phase F output directory absent")
    return out


def _fmt(value, n=3):
    if value is None or pd.isna(value):
        return "판정불가"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{n}f}"
    return str(value)


def _md(headers, rows):
    clean = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(clean, row)) + " |" for row in rows)])


def _registry(out: Path) -> pd.DataFrame:
    path = out / "registry.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["exp_id", "family", "status", "config_hash", "explore_queries"])
    frame = pd.read_csv(path)
    needed = {"exp_id", "family", "status", "config_hash", "explore_queries"}
    if missing := needed - set(frame):
        raise ValueError(f"Registry missing {sorted(missing)}")
    if frame.exp_id.isna().any() or frame.exp_id.duplicated().any():
        raise ValueError("Registry experiment IDs are missing or duplicated")
    for col in ("holdout_read", "historical_final_artifact_read"):
        if col in frame and frame[col].fillna(False).astype(str).str.lower().eq("true").any():
            raise ValueError(f"Registry records {col}=true")
    return frame


def _explore(out: Path, registry: pd.DataFrame):
    result, omitted = {}, []
    for row in registry.itertuples(index=False):
        if row.status != "completed" or pd.isna(row.explore_queries) or int(row.explore_queries) < 1:
            continue
        base = out / "tables" / str(row.exp_id)
        paths = {name: base / f"explore_{name}.csv"
                 for name in ("auc", "pooled", "fold", "pairwise_ci", "daily_summary")}
        if not all(p.is_file() for p in paths.values()):
            omitted.append(f"{row.exp_id}: incomplete EXPLORE tables")
            continue
        tables = {name: pd.read_csv(path) for name, path in paths.items()}
        week_path = base / "explore_pairwise_week_ci.csv"
        manifest_path = base / "explore_manifest.json"
        if week_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("tables",{}).get(week_path.name) == sha256(week_path):
                tables["pairwise_week_ci"] = pd.read_csv(week_path)
        auc, pooled, fold = (tables[name] for name in ("auc", "pooled", "fold"))
        expected = {(d, h) for d in ("D1", "D2") for h in HORIZONS}
        if (set(auc.dataset) != {"D1", "D2"} or len(auc) != 2 or
                not auc.n_horizons.eq(13).all() or not auc.n_available_horizons.eq(13).all() or
                set(zip(pooled.dataset, pooled.horizon)) != expected or len(pooled) != 26 or
                len(fold) != 78 or set(fold.fold) != {0, 1, 2} or
                not np.isfinite(auc[["AUC_MAE", "AUC_PeakMAE"]].to_numpy(float)).all()):
            omitted.append(f"{row.exp_id}: incomplete D1/D2, 13-horizon, 3-fold coverage")
            continue
        result[str(row.exp_id)] = tables
    return result, omitted


def _ranking(tables):
    records = []
    for name, item in tables.items():
        auc = item["auc"].set_index("dataset")
        h16 = item["pooled"].loc[lambda f: f.dataset.eq("D1") & f.horizon.eq(16)].iloc[0]
        records.append({"exp_id": name, "D1_MAE": auc.loc["D1", "AUC_MAE"],
                        "D1_Peak": auc.loc["D1", "AUC_PeakMAE"],
                        "D2_MAE": auc.loc["D2", "AUC_MAE"],
                        "D2_Peak": auc.loc["D2", "AUC_PeakMAE"],
                        "h16_MAE": h16.MAE, "h16_Peak": h16.Peak_MAE})
    if not records:
        return pd.DataFrame(columns=["exp_id", "D1_MAE", "D1_Peak", "D2_MAE",
                                     "D2_Peak", "h16_MAE", "h16_Peak"])
    return pd.DataFrame(records).sort_values(["D1_MAE", "exp_id"], kind="stable").reset_index(drop=True)


def _frame(out: Path, registry: pd.DataFrame, candidate: str, arm: str) -> pd.DataFrame:
    if arm not in ("EXPLORE", "CONFIRM"):
        raise ValueError("Unknown score arm")
    record = registry.loc[registry.exp_id.eq(candidate)]
    if len(record) != 1 or record.iloc[0].status != "completed":
        raise ValueError("Candidate is not completed in registry")
    path = out / "predictions" / f"{candidate}.parquet"
    audit_path = out / "logs" / f"{candidate}_audit.json"
    if not path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Candidate prediction and audit both required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("prediction_sha256") != sha256(path) or audit.get("config_hash") != record.iloc[0].config_hash:
        raise ValueError("Candidate prediction/config hash mismatch")
    # PyArrow predicate pushdown: an EXPLORE report never deserializes the
    # physically adjacent CONFIRM score labels.
    frame = pd.read_parquet(path, filters=[("role", "=", "score"), ("arm", "=", arm)])
    if (frame.empty or not frame.role.eq("score").all() or not frame.arm.eq(arm).all() or
            not frame.model.eq(candidate).all() or
            pd.to_datetime(frame.target_time).ge(BOUNDARY).any()):
        raise ValueError("Filtered candidate score violates arm/model/boundary")
    if set(frame.horizon) != set(HORIZONS) or set(frame.fold) != {0, 1, 2}:
        raise ValueError("Candidate diagnostic lacks all 13 horizons and 3 folds")
    return frame


def _period(frame: pd.DataFrame, start: str, end: str) -> dict:
    target = pd.to_datetime(frame.target_time)
    part = frame.loc[target.ge(pd.Timestamp(start)) & target.lt(pd.Timestamp(end))]
    if part.empty:
        return {"period": f"{start}..{end}", "n": 0, "MAE": np.nan,
                "peak_n": 0, "Peak_MAE": np.nan, "signed_peak_error": np.nan}
    y, pred, tau = (part[col].to_numpy(float) for col in ("y", "pred", "tau"))
    peak = y > tau
    error = pred-y
    return {"period": f"{start}..{end}", "n": len(part), "MAE": float(np.abs(error).mean()),
            "peak_n": int(peak.sum()),
            "Peak_MAE": float(np.abs(error[peak]).mean()) if peak.any() else np.nan,
            "signed_peak_error": float(error[peak].mean()) if peak.any() else np.nan}


def _diagnose(out: Path, registry: pd.DataFrame, candidate: str, production_col: str | None):
    frame = _frame(out, registry, candidate, "EXPLORE")
    detail = summarize_errors(frame, arm="EXPLORE", selected=False, production_col=production_col)
    # July 26-Aug 1 is EXPLORE; Aug 2-8 is locked CONFIRM and stays unopened.
    detail["july_august_shift"] = pd.DataFrame([_period(frame, "2021-07-26", "2021-08-02")])
    dest = out / "tables" / "diagnostics" / candidate
    dest.mkdir(parents=True, exist_ok=True)
    for name, table in detail.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(dest / f"explore_{name}.csv", index=False)
    return detail, frame


def _placeholder(path: Path, title: str, message: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
            color=COLORS[0], fontsize=11, wrap=True)
    ax.set_title(title)
    ax.set_axis_off()
    save(fig, path)


def _figures(out: Path, tables: dict, rank: pd.DataFrame,
             registry: pd.DataFrame, frame: pd.DataFrame | None, candidate: str | None):
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / FIGURES[0]
    if rank.empty:
        _placeholder(path, "Horizon MAE and Peak-MAE", "No complete EXPLORE table")
    else:
        names = list(rank.exp_id.head(4))
        if "F0-1-B5" in tables and "F0-1-B5" not in names:
            names.append("F0-1-B5")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for index, name in enumerate(names):
            part = tables[name]["pooled"]
            part = part.loc[part.dataset.eq("D1")].sort_values("horizon")
            for ax, col in zip(axes, ("MAE", "Peak_MAE")):
                ax.plot(part.horizon, part[col], marker="o", lw=1.2,
                        color=COLORS[index % len(COLORS)], label=name)
        for ax, title in zip(axes, ("MAE", "Peak-MAE")):
            ax.set(xlabel="Horizon (15-minute steps)",
                   ylabel="Error (source power units)",
                   title=f"EXPLORE D1 {title}", xticks=HORIZONS)
            ax.grid(True)
        axes[0].legend(fontsize=8)
        save(fig, path)
    _figures_extra(out, rank, registry, frame, candidate)


def _render_progress(registry: pd.DataFrame, tables: dict, omitted: list[str],
                     rank: pd.DataFrame, detail: dict | None,
                     selected_id: str | None) -> str:
    counts = registry.status.value_counts().to_dict() if not registry.empty else {}
    family_rows = []
    for family in FAMILIES:
        part = registry.loc[registry.family.eq(family)]
        family_rows.append([family, len(part), int(part.status.eq("completed").sum()),
                            int(part.status.eq("failed").sum()),
                            int(part.status.isin(("unsupported", "rejected")).sum()),
                            "미해결" if part.empty or part.status.isin(("planned", "run")).any()
                            else "등록 항목 종료, 범위 검증 별도"])
    lines = [
        "# Phase F 개발 탐색 진행 보고",
        "",
        "**상태: 진행 중. full_phase_complete=false.** EXPLORE 수치는 반복 탐색으로 선택 편향이 있으며 후보 확정이나 CONFIRM 결과가 아니다.",
        "",
        "## 우선 확인할 10가지",
        "",
        f"1. 등록 {len(registry)}건, 완료 {counts.get('completed',0)}건, 실패 {counts.get('failed',0)}건, 미지원 {counts.get('unsupported',0)}건.",
        f"2. 비교 가능한 EXPLORE D1/D2, 13 horizon, 3 fold 표 {len(tables)}건. 완료 표기가 있으나 표가 불완전한 항목 {len(omitted)}건.",
        f"3. 현재 EXPLORE D1 AUC-MAE 최소 표시값: {rank.iloc[0].exp_id + ' / ' + _fmt(rank.iloc[0].D1_MAE,4) if not rank.empty else '판정불가'}. 확증 모델 선정이 아니다.",
        "4. B5 대비 paired 95% CI, 피크 악화 부재, D2 보호, 2/3 fold 개선을 함께 확인해야 한다.",
        "5. h16 MAE와 Peak-MAE는 별도로 보고한다. 평균 오차 개선만으로 피크 보호를 선언하지 않는다.",
        "6. 13 horizon × 3 fold의 Phase C MAIN10 공통 100,010 score 키를 재사용한다.",
        "7. F10의 시간대, 요일, 휴일 인접, 생산량 구간은 사후 진단이며 모델 선정 특징이나 기준이 아니다.",
        "8. 2021-07-26~08-01은 EXPLORE, 08-02~08-08은 CONFIRM이다. 완료 잠금 전에는 두 기간을 합산하지 않는다.",
        "9. 데이터 단위와 현장 허용 오차가 확인되지 않아 상용 적합성은 판정불가다. ASHRAE는 인증 기준으로 사용하지 않는다.",
        "10. 다음 단계는 남은 계열과 실패·미지원 사유 기록, primary 사전 동결, CONFIRM 1회, F9/F10/F11 후속 진단이다.",
        "",
        "## 완료된 EXPLORE 비교: D1 AUC-MAE 상위 10",
        "",
    ]
    rows = [[r.exp_id, *(_fmt(getattr(r, c),4) for c in
             ("D1_MAE","D1_Peak","D2_MAE","D2_Peak","h16_MAE","h16_Peak"))]
            for r in rank.head(10).itertuples(index=False)]
    lines += [_md(["실험 ID","D1 AUC-MAE","D1 AUC-PeakMAE","D2 AUC-MAE",
                   "D2 AUC-PeakMAE","h16 MAE","h16 Peak-MAE"], rows)
              if rows else "완료된 비교 표가 없다.", "",
              "AUC는 13개 pooled horizon metric의 동일 가중 평균이다. 탐색 최저값과 최종 동결 후보는 구분한다.", "",
              "## 계열 범위와 실패", "",
              _md(["계열","등록","완료","실패","미지원/제외","상태"], family_rows), ""]
    if omitted:
        lines += ["완료 레지스트리와 근거 표의 불일치:", "",
                  *[f"- {x}" for x in omitted], ""]
    failed = registry.loc[registry.status.eq("failed")]
    if not failed.empty:
        lines += ["실패 기록:", "",
                  _md(["ID","유형","원인"], [[r.exp_id, getattr(r,"error_type",""),
                      str(getattr(r,"error",""))[:180]] for r in failed.itertuples(index=False)]), ""]
    lines += ["## Paired 근거와 F10 오류 진단", ""]
    lines += ["### F9-3 완전 프로필 중복 민감도 (EXPLORE)","",
              _dedup_evidence(tables),"",
              "keep/drop/weight는 동일 개발 구간의 데이터 구성 민감도다. D2는 신규 프로필 "
              "부분집합이며 이 비교로 독립 일반화나 최종 적격성을 선언하지 않는다.",""]
    if selected_id is None:
        lines += ["진단용 후보 ID가 지정되지 않아 개별 후보 오류 분포를 만들지 않았다. 현재 순위를 암묵적 후보 동결로 취급하지 않는다.", ""]
    else:
        lines += [f"진단 대상: {selected_id}. EXPLORE score만 parquet predicate filter로 읽었다. 이 표시는 primary 선정이 아니다.", ""]
        ci = tables[selected_id]["pairwise_ci"]
        ci_rows = []
        for baseline in ("B5","M1","R1"):
            for domain in ("D1","D2"):
                for metric in ("AUC_MAE_improvement","AUC_PeakMAE_degradation"):
                    part = ci.loc[ci.baseline.eq(baseline) & ci.dataset.eq(domain) & ci.metric.eq(metric)]
                    if len(part) == 1:
                        r = part.iloc[0]
                        ci_rows.append([baseline,domain,metric,_fmt(r.estimate,4),
                                        _fmt(r.ci_low,4),_fmt(r.ci_high,4),r.ci_status])
        lines += [_md(["기준","집합","차이 지표","점추정","CI 하한","CI 상한","CI 상태"],ci_rows),""]
        lines += ["### ISO-week 블록 bootstrap 민감도 (EXPLORE)","",
                  _week_ci_evidence(tables[selected_id].get("pairwise_week_ci"),None,
                                    selected_id),"",
                  "주간 CI는 날짜 블록 CI와 D2/프로필 중복에 대한 민감도 확인용이며 "
                  "후보 적격성 또는 정지 기준을 바꾸지 않는다.",""]
        if detail is not None:
            for key, label in (("by_hour","시간대"),("by_weekday","요일: 0=월"),
                               ("by_holiday_context","휴일 인접"),
                               ("by_production_posthoc","생산량 사후구간")):
                part = detail.get(key)
                if not isinstance(part,pd.DataFrame) or part.empty:
                    lines += [f"{label}: 증거 없음 또는 진단 열 없음; 판정불가.",""]
                    continue
                lines += [f"### {label} EXPLORE 오류","",
                          _md(["구간","n","MAE","bias pred−actual","peak n","Peak-MAE","FP/FN"],[
                              [r.group,int(r.n),_fmt(r.mae),_fmt(r.bias_pred_minus_actual),
                               int(r.peak_n),_fmt(r.peak_mae),
                               f"{int(r.false_positive_n)}/{int(r.false_negative_n)}"]
                              for r in part.itertuples(index=False)]),""]
            shift = detail["july_august_shift"].iloc[0]
            lines += ["### 7월 말~8월 초 shift","",
                      f"EXPLORE 2021-07-26~08-01: n={int(shift.n)}, MAE={_fmt(shift.MAE)}, "
                      f"peak n={int(shift.peak_n)}, Peak-MAE={_fmt(shift.Peak_MAE)}, "
                      f"signed peak error={_fmt(shift.signed_peak_error)}. "
                      "2021-08-02~08-08 CONFIRM 값은 candidate/config hash 잠금 및 1회 평가 완료 전까지 미열람이다.","",
                      f"진단 표본: fold 1 {detail['audit']['fold_1_n']}행, D2 {detail['audit']['fold_1_d2_n']}행. "
                      f"augmented flag {detail['audit']['late_july_augmented_status']}, "
                      f"quality flag {detail['audit']['late_july_quality_status']}.",""]
    lines += ["## 그림",""]
    for name in FIGURES:
        lines += [f"![{name}](figures/{name})",""]
    lines += ["## 해석과 다음 결정 경계","",
              "동일 키 B5/M1/R1과 paired 비교하되 EXPLORE의 수백 회 조회에 따른 선택 편향을 보정한 확증 CI로 해석하지 않는다. 증강 profile의 반복과 짧은 기간도 CI 한계다. 시간별 ASHRAE CV(RMSE) 참고값은 15분 공장 미래예측의 상용 인증 기준이 아니다. 전력 단위·집계 의미, 새로운 비증강 현장자료, 허용오차·경보 부담·행동비용이 확인되기 전 상용 적합성은 판정불가다.","",
              "계열 F0–F11의 요구 coverage, 500회 이상 LightGBM 탐색, 확장 정지 기준, 실패·미지원 이유를 남긴다. EXPLORE에서 primary와 최대 4개 보조 후보를 봉인한 뒤 기존 개발자료의 CONFIRM을 한 번만 연다. F9/F10/F11 후속 진단과 보존 감사를 확인하기 전 full Phase F 완료라고 쓰지 않는다.","",
              "이 보고 경로는 final holdout이나 historical final 결과 내용을 읽지 않는다. 이전 Phase C/E 평가 및 과거 boundary incident를 지우는 의미가 아니다.",""]
    return "\n".join(lines)


def generate_progress(root: Path, *, selected_id: str | None = None,
                      production_col: str | None = None) -> dict:
    """Write EXPLORE-only progress; a supplied candidate controls diagnostics only."""
    out = _out(root)
    configure()
    registry = _registry(out)
    tables, omitted = _explore(out, registry)
    rank = _ranking(tables)
    detail = frame = None
    if selected_id is not None:
        if selected_id not in tables:
            raise ValueError("Diagnostic candidate needs complete EXPLORE tables")
        detail, frame = _diagnose(out, registry, selected_id, production_col)
    _figures(out, tables, rank, registry, frame, selected_id)
    dest = out / "progress_report.md"
    dest.write_text(_render_progress(registry,tables,omitted,rank,detail,selected_id),
                    encoding="utf-8")
    return {"path": str(dest), "full_phase_complete": False,
            "completed_comparable": len(tables), "omitted_completed": omitted,
            "diagnostic_candidate": selected_id, "confirm_read": False}


def _figures_extra(out: Path, rank: pd.DataFrame, registry: pd.DataFrame,
                   frame: pd.DataFrame | None, candidate: str | None):
    directory = out / "figures"
    path = directory / FIGURES[1]
    context = []
    for row in rank.merge(registry[["exp_id", "family"]], on="exp_id").itertuples(index=False):
        match = re.search(r"(?:^|-)c(\d+)(?:-|$)", row.exp_id)
        if match and row.family in ("F5", "F6"):
            context.append((int(match.group(1)), float(row.D1_MAE), row.exp_id, row.family))
    if len({item[0] for item in context}) < 2:
        observed = sorted({item[0] for item in context})
        _placeholder(path, "Context length",
                     f"Context-length comparison pending; observed completed lengths: {observed}")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for family, color in (("F5", COLORS[1]), ("F6", COLORS[2])):
            part = [item for item in context if item[3] == family]
            if part:
                ax.scatter([v[0] for v in part], [v[1] for v in part],
                           label=family, color=color)
                for length, mae, name, _ in part:
                    ax.annotate(name, (length, mae), xytext=(3, 3),
                                textcoords="offset points", fontsize=6)
        ax.set(xlabel="Past context (15-minute slots)", ylabel="EXPLORE D1 AUC-MAE",
               title="Completed context configurations")
        ax.grid(True)
        ax.legend()
        save(fig, path)

    path = directory / FIGURES[2]
    ablation = rank.merge(registry[["exp_id", "family"]], on="exp_id")
    ablation = ablation.loc[ablation.family.eq("F1")].sort_values("D1_MAE").head(18)
    if ablation.empty:
        _placeholder(path, "Feature ablation", "No complete F1 ablation results")
    else:
        fig, ax = plt.subplots(figsize=(10, max(5, .34*len(ablation)+1.8)))
        positions = np.arange(len(ablation))
        ax.barh(positions, ablation.D1_MAE, color=COLORS[1])
        ax.set(yticks=positions, yticklabels=ablation.exp_id,
               xlabel="EXPLORE D1 AUC-MAE", title="Completed F1 feature configurations")
        ax.invert_yaxis()
        ax.grid(axis="x")
        save(fig, path)

    path = directory / FIGURES[3]
    peak = None if frame is None else frame.loc[frame.horizon.eq(16) & frame.y.gt(frame.tau)]
    if peak is None:
        _placeholder(path, "Signed peak errors", "No diagnostic candidate supplied")
    elif peak.empty:
        _placeholder(path, "Signed peak errors", "No actual peaks in h16 EXPLORE rows")
    else:
        error = (peak.pred-peak.y).to_numpy(float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(error, bins=30, color=COLORS[1], alpha=.85)
        ax.axvline(0, color=COLORS[0], ls="--")
        ax.axvline(np.median(error), color=COLORS[2],
                   label=f"median {_fmt(np.median(error),2)}")
        ax.set(xlabel="Prediction minus actual (source power units)",
               ylabel="Actual-peak rows",
               title=f"{candidate} EXPLORE h16 signed peak error")
        ax.grid(axis="y")
        ax.legend()
        save(fig, path)

    path = directory / FIGURES[4]
    sample = None if frame is None else frame.loc[frame.horizon.eq(16) & frame.fold.eq(0)].copy()
    if sample is None or sample.empty:
        _placeholder(path, "Representative week", "No diagnostic candidate EXPLORE h16 fold 0")
    else:
        sample["target_time"] = pd.to_datetime(sample.target_time)
        sample = sample.sort_values("target_time")
        first = sample.target_time.min()
        start = first.normalize()-pd.Timedelta(days=first.weekday())
        week = sample.loc[sample.target_time.ge(start) &
                          sample.target_time.lt(start+pd.Timedelta(days=7))]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(week.target_time, week.y, color=COLORS[0], label="Observed")
        ax.plot(week.target_time, week.pred, color=COLORS[1], label=candidate)
        ax.plot(week.target_time, week.tau, color=COLORS[2], ls="--", label="fit Q0.95")
        ax.set(xlabel="Target time", ylabel="Source power units",
               title=f"First represented EXPLORE ISO week, h16 fold 0 (n={len(week)})")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate()
        save(fig, path)


def _confirmed_lock(out: Path, registry: pd.DataFrame) -> tuple[dict, dict] | None:
    selection_path = out / "logs" / "confirm_lock.json"
    complete_path = out / "logs" / "confirm_complete.json"
    if not selection_path.is_file() or not complete_path.is_file():
        return None
    lock = json.loads(selection_path.read_text(encoding="utf-8"))
    done = json.loads(complete_path.read_text(encoding="utf-8"))
    digest = lock.get("lock_sha256")
    payload = {key:value for key,value in lock.items() if key != "lock_sha256"}
    if not isinstance(digest,str) or config_hash(payload) != digest:
        return None
    candidates = lock.get("candidates")
    primary = lock.get("primary_candidate")
    if (not isinstance(candidates,list) or len(candidates)>5 or
            len(set(candidates)) != len(candidates) or
            primary != (candidates[0] if candidates else None) or
            lock.get("selection_arm") != "EXPLORE" or
            lock.get("holdout_read") is not False or
            lock.get("historical_final_artifact_read") is not False):
        return None
    split_path = out / "logs" / "split_lock.json"
    if not split_path.is_file():
        return None
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if lock.get("split_lock_sha256") != split.get("lock_sha256"):
        return None
    for candidate in candidates:
        record = registry.loc[registry.exp_id.eq(candidate)]
        path = out / "predictions" / f"{candidate}.parquet"
        metric_manifest = out / "tables" / candidate / "explore_manifest.json"
        if (len(record) != 1 or record.iloc[0].status != "completed" or
                lock.get("config_hashes",{}).get(candidate) != record.iloc[0].config_hash or
                not path.is_file() or lock.get("prediction_hashes",{}).get(candidate) != sha256(path)):
            return None
        if (not metric_manifest.is_file() or
                lock.get("explore_metric_manifest_hashes",{}).get(candidate) != sha256(metric_manifest)):
            return None
    complete_digest = done.get("confirm_once_sha256")
    complete_payload = {key:value for key,value in done.items() if key != "confirm_once_sha256"}
    if (done.get("complete") is not True or done.get("lock_sha256") != digest or
            done.get("primary_candidate") != primary or done.get("candidates") != candidates or
            done.get("holdout_read") is not False or
            done.get("historical_final_artifact_read") is not False or
            not isinstance(complete_digest,str) or config_hash(complete_payload) != complete_digest):
        return None
    tables = done.get("tables")
    if not isinstance(tables,dict) or not tables:
        return None
    for name,expected in tables.items():
        path = (out / name).resolve()
        if (not path.is_relative_to(out.resolve()) or not
                path.is_relative_to((out / "tables" / "confirm").resolve()) or
                not path.is_file() or sha256(path) != expected):
            return None
    for model in [*candidates,"B5","M1","R1"]:
        if f"tables/confirm/{model}/confirm_auc.csv" not in tables:
            return None
    return lock, done


def _coverage_gate(out: Path, done: dict, *, provisional: bool = False) -> tuple[bool,str]:
    path = out / "logs" / "completion_gate.json"
    if not path.is_file():
        return False, "completion_gate_absent"
    gate = json.loads(path.read_text(encoding="utf-8"))
    coverage = gate.get("required_family_coverage")
    if not isinstance(coverage,dict) or set(coverage) != set(FAMILIES):
        return False, "required_family_coverage_incomplete"
    for family,state in coverage.items():
        if state is True:
            continue
        if not isinstance(state,dict) or state.get("resolved") is not True or \
                state.get("status") not in ("completed","failed_explicit","unsupported_explicit"):
            return False, f"{family}_coverage_unresolved"
    if provisional:
        if gate.get("ready_for_report") is not True or gate.get("full_phase_complete") is not False:
            return False, "provisional_report_gate_absent"
    elif gate.get("full_phase_complete") is not True:
        return False, "final_completion_gate_absent"
    if (gate.get("downstream_resolved") is not True or
            gate.get("expansion_converged") is not True or
            gate.get("holdout_read") is not False or
            gate.get("historical_final_artifact_read") is not False):
        return False, "downstream_expansion_or_boundary_unresolved"
    if gate.get("confirm_once_sha256") != done["confirm_once_sha256"]:
        return False, "completion_confirm_digest_mismatch"
    return True, "verified"


def _weekly_evidence(out: Path, lock: dict, done: dict,
                     registry: pd.DataFrame) -> tuple[str,bool]:
    """Verify the locked F9 weekly diagnostic without reading numeric rows."""
    status_path = out / "logs" / "weekly_diagnostic_status.json"
    if not status_path.is_file():
        return "F9 주간 재학습 상태 기록 없음.",False
    gate = json.loads((out / "logs" / "completion_gate.json").read_text(encoding="utf-8"))
    if gate.get("weekly_diagnostic_status_sha256") != sha256(status_path):
        return "F9 주간 진단 상태 digest와 completion gate 불일치.",False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if gate.get("weekly_diagnostic") not in (None,status.get("status")):
        return "F9 주간 진단 상태와 completion gate 상태 불일치.",False
    if (status.get("selection_lock_sha256") != lock["lock_sha256"] or
            status.get("confirm_once_sha256") != done["confirm_once_sha256"] or
            status.get("holdout_read") is not False or
            status.get("historical_final_artifact_read") is not False):
        return "F9 주간 재학습 상태의 동결·CONFIRM·경계 해시 불일치.",False
    candidate = lock.get("weekly_diagnostic_candidate")
    if status.get("status") == "unavailable":
        reason = str(status.get("reason", "")).strip()
        if candidate is not None or not reason or status.get("artifacts"):
            return "F9 미지원 상태에 후보·사유·artifact 모순이 있다.",False
        return f"F9 주간 재학습: 해당 없음. 사유: {reason}.",True
    if status.get("status") != "completed" or status.get("candidate") != candidate or not candidate:
        return "F9 주간 재학습 완료 상태 또는 동결 후보 불일치.",False
    selected = registry.loc[registry.exp_id.eq(candidate)]
    if len(selected) != 1 or selected.iloc[0].config_hash != lock.get("weekly_diagnostic_config_hash"):
        return "F9 주간 진단 후보의 registry config hash 불일치.",False
    weekly_lock_path = out / "logs" / "weekly_diagnostic_lock.json"
    if not weekly_lock_path.is_file():
        return "F9 주간 진단 설정 lock 부재.",False
    if gate.get("weekly_diagnostic_lock_sha256") != sha256(weekly_lock_path):
        return "F9 주간 진단 lock digest와 completion gate 불일치.",False
    weekly_lock = json.loads(weekly_lock_path.read_text(encoding="utf-8"))
    frozen = {**json.loads(selected.iloc[0].config_json), "horizon":16}
    if (weekly_lock.get("candidate") != candidate or
            weekly_lock.get("selection_lock_sha256") != lock["lock_sha256"] or
            weekly_lock.get("confirm_once_sha256") != done["confirm_once_sha256"] or
            weekly_lock.get("frozen_config_hash") != config_hash(frozen) or
            weekly_lock.get("horizon") != 16 or
            weekly_lock.get("holdout_read") is not False or
            weekly_lock.get("historical_final_artifact_read") is not False or
            not isinstance(weekly_lock.get("score_origin_hashes"),dict) or
            set(weekly_lock["score_origin_hashes"]) != {"0","1","2"} or
            not all(weekly_lock["score_origin_hashes"].values())):
        return "F9 주간 진단 lock의 설정·origin·경계 검증 실패.",False
    artifacts = status.get("artifacts")
    expected = ({f"predictions/F9-4-fold{fold}-predictions.parquet" for fold in range(3)} |
                {f"tables/F9-4-fold{fold}-updates.csv" for fold in range(3)})
    if not isinstance(artifacts,dict) or not expected <= set(artifacts):
        return "F9 주간 진단 세 fold의 예측·업데이트 artifact 누락.",False
    for relative,digest in artifacts.items():
        path = (out / relative).resolve()
        if not path.is_relative_to(out.resolve()) or not path.is_file() or sha256(path) != digest:
            return f"F9 주간 진단 artifact 해시 불일치: {relative}.",False
    return (f"F9 주간 재학습 사후 진단: {candidate}; 3-fold 예측·업데이트 6개 기본 "
            "artifact와 추가 기록의 SHA-256, 설정·origin lock을 검증했다. "
            "후보를 재선정하지 않았다. "
            "[status](logs/weekly_diagnostic_status.json), "
            "[lock](logs/weekly_diagnostic_lock.json)."),True


def _family_attempts(out: Path, registry: pd.DataFrame) -> str:
    attempts = {family:0 for family in FAMILIES}
    path = out / "logs" / "registry_events.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                family = item.get("family")
                if item.get("event") == "run" and family in attempts:
                    attempts[family] += 1
    rows = []
    for family in FAMILIES:
        part = registry.loc[registry.family.eq(family)]
        rows.append([family,len(part),attempts[family],
                     int(part.status.eq("completed").sum()),
                     int(part.status.eq("failed").sum()),
                     int(part.status.eq("unsupported").sum()),
                     int(part.status.eq("rejected").sum()),
                     int(part.status.isin(("planned","run")).sum()),
                     int(pd.to_numeric(part.explore_queries,errors="coerce").fillna(0).sum())])
    return _md(["계열","등록","실행 시도","완료","실패","미지원","제외","미종결","EXPLORE 조회"],rows)


def _model_evidence(tables: dict, models: list[str]) -> tuple[str,str]:
    normalized, peaks = [], []
    for name in models:
        if name not in tables:
            normalized.append([name,"D1/D2","판정불가","판정불가","판정불가","판정불가"])
            peaks.append([name,"D1/D2","판정불가","판정불가","판정불가","판정불가"])
            continue
        item = tables[name]
        pooled = item["pooled"]
        auc = item["auc"].set_index("dataset")
        daily = item["daily_summary"]
        for dataset in ("D1","D2"):
            subset = pooled.loc[pooled.dataset.eq(dataset)]
            h16 = subset.loc[subset.horizon.eq(16)].iloc[0]
            annual = daily.loc[daily.dataset.eq(dataset) & daily.scope.eq("pooled") &
                               daily.subset.eq("full_96") & daily.horizon.eq(16)]
            if len(annual) != 1:
                raise ValueError(f"Missing full-day h16 summary for {name}/{dataset}")
            day = annual.iloc[0]
            normalized.append([name,dataset,_fmt(subset.nMAE.mean(),4),
                               _fmt(subset.CVRMSE.mean(),4),_fmt(subset.NMBE.mean(),4),
                               _fmt(h16.nMAE,4)])
            peaks.append([name,dataset,_fmt(auc.loc[dataset,"AUC_PeakMAE"],4),
                          _fmt(auc.loc[dataset,"AUC_PredPeakMAE"],4),
                          f"{_fmt(day.daily_max_MAE,4)} / {_fmt(day.timing_MAE_minutes,1)} min",
                          f"{int(day.n_full_days)} / {int(day.n_days)}"])
    return (
        _md(["모델","집합","13h 평균 nMAE","13h 평균 CVRMSE",
             "13h 평균 NMBE","h16 nMAE"],normalized),
        _md(["모델","집합","AUC actual-peak MAE","AUC predicted-peak MAE",
             "h16 full-day max MAE / timing MAE","full/observed days"],peaks),
    )


def _paired_evidence(out: Path, tables: dict, primary: str | None) -> str:
    if primary is None:
        return "적격 primary가 없어 primary-vs-baseline paired CI는 해당 없음."
    rows = []
    for arm in ("EXPLORE","CONFIRM"):
        path = (out / "tables" / primary /
                "explore_pairwise_ci.csv" if arm == "EXPLORE" else
                out / "tables" / "confirm" / primary / "confirm_pairwise_ci.csv")
        if not path.is_file():
            rows.append([arm,"-","-","-","판정불가","판정불가","판정불가","표 없음"])
            continue
        # EXPLORE was already checked as a completed table. CONFIRM path was
        # authenticated by confirm_complete.json before this helper is called.
        frame = tables[primary]["pairwise_ci"] if arm == "EXPLORE" else pd.read_csv(path)
        for baseline in ("B5","M1","R1"):
            for dataset in ("D1","D2"):
                for metric in ("AUC_MAE_improvement","AUC_PeakMAE_degradation"):
                    selected = frame.loc[frame.baseline.eq(baseline) &
                                         frame.dataset.eq(dataset) & frame.metric.eq(metric)]
                    if len(selected) != 1:
                        rows.append([arm,baseline,dataset,metric,
                                     "판정불가","판정불가","판정불가","행 없음"])
                    else:
                        r = selected.iloc[0]
                        rows.append([arm,baseline,dataset,metric,_fmt(r.estimate,4),
                                     _fmt(r.ci_low,4),_fmt(r.ci_high,4),
                                     str(r.ci_status) + (f": {r.ci_reason}" if
                                         pd.notna(r.ci_reason) else "")])
    return _md(["arm","기준","집합","paired 차이","점추정","95% CI 하한",
                "95% CI 상한","상태"],rows)


def _week_ci_valid(frame: pd.DataFrame, candidate: str) -> bool:
    required = {"candidate","baseline","dataset","horizon","metric","block",
                "bootstrap_n","seed","n_blocks","estimate","ci_low","ci_high",
                "ci_status","ci_reason"}
    if not isinstance(frame,pd.DataFrame) or required - set(frame):
        return False
    auc = frame.loc[frame.horizon.isna()]
    expected = {(baseline,dataset,metric) for baseline in ("B5","M1","R1")
                for dataset in ("D1","D2")
                for metric in ("AUC_MAE_improvement","AUC_PeakMAE_degradation")}
    keys = list(zip(auc.baseline,auc.dataset,auc.metric))
    return (len(auc) == len(expected) and set(keys) == expected and
            len(set(keys)) == len(keys) and frame.candidate.eq(candidate).all() and
            frame.block.eq("week").all() and frame.bootstrap_n.eq(1000).all() and
            frame.seed.eq(42).all() and auc.ci_status.isin(("available","unavailable")).all() and
            pd.to_numeric(auc.n_blocks,errors="coerce").notna().all())


def _week_ci_evidence(explore: pd.DataFrame | None,
                      confirm: pd.DataFrame | None, candidate: str) -> str:
    rows = []
    for arm,frame in (("EXPLORE",explore),("CONFIRM",confirm)):
        if frame is None:
            if arm == "CONFIRM":
                continue
            rows.append([arm,"-","-","-","-","판정불가","판정불가","판정불가",
                         "봉인된 주간 CI 표 없음"])
            continue
        if not _week_ci_valid(frame,candidate):
            rows.append([arm,"-","-","-","-","판정불가","판정불가","판정불가",
                         "주간 CI 스키마·설정 불일치"])
            continue
        auc = frame.loc[frame.horizon.isna()]
        for baseline in ("B5","M1","R1"):
            for dataset in ("D1","D2"):
                for metric in ("AUC_MAE_improvement","AUC_PeakMAE_degradation"):
                    r = auc.loc[auc.baseline.eq(baseline) & auc.dataset.eq(dataset) &
                                auc.metric.eq(metric)].iloc[0]
                    rows.append([arm,baseline,dataset,metric,int(r.n_blocks),
                                 _fmt(r.estimate,4),_fmt(r.ci_low,4),_fmt(r.ci_high,4),
                                 str(r.ci_status) + (f": {r.ci_reason}" if
                                     pd.notna(r.ci_reason) and str(r.ci_reason) else "")])
    return _md(["arm","기준","집합","paired 차이","유효 ISO 주 수","점추정",
                "95% CI 하한","95% CI 상한","상태·CI 미정의 사유"],rows)


def _information_budget(tables: dict) -> str:
    rows = []
    for kind in ("ridge","lightgbm"):
        core, union = f"F1-core-{kind}", f"F1-union-{kind}"
        if core in tables and union in tables:
            for dataset in ("D1","D2"):
                a = tables[core]["auc"].set_index("dataset").loc[dataset]
                b = tables[union]["auc"].set_index("dataset").loc[dataset]
                rows.append([f"{kind}: core → union",dataset,
                             _fmt(a.AUC_MAE,4),_fmt(b.AUC_MAE,4),
                             _fmt(a.AUC_MAE-b.AUC_MAE,4),
                             _fmt(a.AUC_PeakMAE-b.AUC_PeakMAE,4)])
        else:
            rows.append([f"{kind}: core → union","D1/D2",
                         "미완","미완","판정불가","판정불가"])
    pairs = set()
    for name in tables:
        if name.startswith("F6") and "-c512-" in name:
            long = name.replace("-c512-","-c2048-")
            if long in tables:
                pairs.add((name,long))
    if pairs:
        for short,long in sorted(pairs):
            for dataset in ("D1","D2"):
                a = tables[short]["auc"].set_index("dataset").loc[dataset]
                b = tables[long]["auc"].set_index("dataset").loc[dataset]
                rows.append([f"{short} → {long}",dataset,
                             _fmt(a.AUC_MAE,4),_fmt(b.AUC_MAE,4),
                             _fmt(a.AUC_MAE-b.AUC_MAE,4),
                             _fmt(a.AUC_PeakMAE-b.AUC_PeakMAE,4)])
    else:
        rows.append(["F6 c512 → c2048","D1/D2","미완","미완",
                     "판정불가","판정불가"])
    return _md(["비교","집합","짧은/기본 AUC-MAE","확장 AUC-MAE",
                "MAE 감소(+ 좋음)","Peak-MAE 감소(+ 좋음)"],rows)


def _dedup_evidence(tables: dict) -> str:
    rows = []
    for policy in ("keep","drop","weight"):
        model = f"F9-3-{policy}"
        if model not in tables:
            rows.append([model,"D1/D2","미완/미지원","판정불가","판정불가"])
            continue
        auc = tables[model]["auc"].set_index("dataset")
        for dataset in ("D1","D2"):
            row = auc.loc[dataset]
            rows.append([model,dataset,"완료",_fmt(row.AUC_MAE,4),
                         _fmt(row.AUC_PeakMAE,4)])
    return _md(["프로필 중복 처리","집합","EXPLORE 상태","AUC-MAE",
                "AUC-PeakMAE"],rows)


def _license_evidence(out: Path) -> str:
    rows = []
    for path in sorted((out / "logs").glob("F6-5-*_weights.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append([path.stem.removesuffix("_weights"),item.get("repo","UNKNOWN"),
                     item.get("license","UNKNOWN"),
                     _fmt(item.get("commercially_eligible")),
                     "weight lock; legal review still required"])
    return _md(["모델","weight repo","기록된 license","commercial flag",
                "해석"],rows) if rows else "Other-foundation weight/license 감사 기록 없음; 판정불가."


def _downstream_evidence(out: Path, lock: dict, done: dict,
                         primary: str | None) -> tuple[str,str,bool]:
    status_path = out / "logs" / "downstream_status.json"
    if not status_path.is_file():
        return ("적격 primary 없음: F11 후보 비교 해당 없음." if primary is None else
                "F11 상태 기록 없음: downstream_status.json 부재.",
                "",primary is None)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (status.get("selection_lock_sha256") != lock["lock_sha256"] or
            status.get("confirm_once_sha256") != done["confirm_once_sha256"]):
        return "F11 상태 기록이 동결 후보 또는 CONFIRM 해시와 불일치: 수치 사용 불가.", "",False
    if status.get("status") != "completed":
        reason = str(status.get("reason","")).strip()
        return f"F11 상태 {status.get('status','UNKNOWN')}; 사유: {reason or '기록 없음'}.", "",bool(
            status.get("status") == "unavailable" and reason)
    artifacts = status.get("artifacts")
    if not isinstance(artifacts,dict):
        return "F11 완료 표기는 있으나 artifact hash manifest가 없어 수치 사용 불가.", "",False
    for relative, expected in artifacts.items():
        path = (out / relative).resolve()
        if (not path.is_relative_to(out.resolve()) or not path.is_file() or
                sha256(path) != expected):
            return f"F11 artifact {relative} hash/파일 불일치: 수치 사용 불가.", "",False
    names = ("risk_metrics","uncertainty_metrics","alert_episode_metrics",
             "comparison_vs_B5","decision_value_curve")
    paths = {name: out / "tables" / "downstream" / f"{name}.csv" for name in names}
    for name,path in paths.items():
        relative = path.relative_to(out).as_posix()
        if relative not in artifacts or not path.is_file() or sha256(path) != artifacts[relative]:
            return f"F11 {name} hash/파일 불일치: 수치 사용 불가.", "",False
    risk = pd.read_csv(paths["risk_metrics"])
    uncertainty = pd.read_csv(paths["uncertainty_metrics"])
    episodes = pd.read_csv(paths["alert_episode_metrics"])
    models = ["B5"] + ([primary] if primary is not None else [])
    rows = []
    for model in models:
        for subset in ("D1","D2"):
            r = risk.loc[risk.model.eq(model) & risk.subset.eq(subset) & risk.fold.eq("pooled")]
            u = uncertainty.loc[uncertainty.model.eq(model) & uncertainty.subset.eq(subset) &
                                uncertainty.fold.eq("pooled") &
                                uncertainty.population.eq("all") &
                                uncertainty.method.eq("conformal")]
            if len(r)!=1 or len(u)!=1:
                return f"F11 {model}/{subset} pooled risk/coverage 행 누락: 수치 사용 불가.", "",False
            risk_row, coverage_row = r.iloc[0], u.iloc[0]
            for policy in ("1/1","2/2"):
                e = episodes.loc[episodes.model.eq(model) & episodes.subset.eq(subset) &
                                 episodes.fold.eq("pooled") & episodes.threshold.eq(.10) &
                                 episodes.policy.eq(policy)]
                if len(e)!=1:
                    return f"F11 {model}/{subset}/{policy} pooled 행 누락: 수치 사용 불가.", "",False
                e=e.iloc[0]
                fp_field = ("false_alert_episodes_per_observed_evaluation_day"
                            if "false_alert_episodes_per_observed_evaluation_day" in e.index
                            else "false_alert_episodes_per_operating_day")
                rows.append([model,subset,policy,_fmt(risk_row.BS_cal,5),
                             _fmt(risk_row.BSS,4),
                             f"{_fmt(risk_row.BSS_ci_lower,4)}–{_fmt(risk_row.BSS_ci_upper,4)}",
                             _fmt(coverage_row.coverage,4),
                             f"{_fmt(coverage_row.coverage_ci_lower,4)}–{_fmt(coverage_row.coverage_ci_upper,4)}",
                             _fmt(coverage_row.mean_width,3),_fmt(e.episode_recall,4),
                             f"{_fmt(e.episode_recall_ci_low,4)}–{_fmt(e.episode_recall_ci_high,4)}",
                             _fmt(e[fp_field],4),
                             f"{_fmt(e[f'{fp_field}_ci_low'],4)}–{_fmt(e[f'{fp_field}_ci_high'],4)}"])
    table = _md(["모델","집합","c=.10 정책","Brier cal","BSS","BSS 95% CI",
                 "U95 coverage","coverage 95% CI","mean width","episode recall",
                 "recall 95% CI","FP/평가 관측일","FP 95% CI"],rows)
    csvs = sorted(name for name in artifacts if name.startswith("tables/downstream/")
                  and name.endswith(".csv"))
    links = " ".join(f"[{Path(name).stem}]({name})" for name in csvs)
    return table, links,True


def generate_final(root: Path, *, provisional: bool = False,
                   production_col: str | None = None) -> dict:
    """Prepare a pending report or make the final report after every gate passes.

    Incomplete work returns false without reading CONFIRM numeric data or
    creating a final report. A completed but ineligible primary is a valid
    scientific failure result; no substitute candidate is invented.
    """
    out = _out(root)
    registry = _registry(out)
    pair = _confirmed_lock(out, registry)
    if pair is None:
        return {"full_phase_complete":False,
                "reason":"valid_candidate_hash_locked_confirm_absent","confirm_read":False}
    lock, done = pair
    passed, reason = _coverage_gate(out,done,provisional=provisional)
    if not passed:
        return {"full_phase_complete":False,"reason":reason,"confirm_read":False}
    weekly_text, weekly_ok = _weekly_evidence(out,lock,done,registry)
    if not weekly_ok:
        return {"full_phase_complete":False,"reason":"weekly_evidence_unresolved",
                "detail":weekly_text,"confirm_read":False}
    primary = lock["primary_candidate"]
    confirm_week = None
    if primary is not None:
        week_relative = f"tables/confirm/{primary}/confirm_pairwise_week_ci.csv"
        week_path = out / week_relative
        if (week_relative not in done["tables"] or not week_path.is_file() or
                sha256(week_path) != done["tables"][week_relative]):
            return {"full_phase_complete":False,"reason":"sealed_confirm_week_ci_absent",
                    "confirm_read":False}
        confirm_week = pd.read_csv(week_path)
        if not _week_ci_valid(confirm_week,primary):
            return {"full_phase_complete":False,"reason":"confirm_week_ci_invalid",
                    "confirm_read":True}
    downstream_text, downstream_links, downstream_ok = _downstream_evidence(
        out,lock,done,primary)
    if not downstream_ok:
        return {"full_phase_complete":False,"reason":"downstream_evidence_unresolved",
                "detail":downstream_text,"confirm_read":False}
    tables, omitted = _explore(out,registry)
    rank = _ranking(tables)
    if primary is not None and primary not in tables:
        return {"full_phase_complete":False,"reason":"primary_explore_table_incomplete",
                "confirm_read":False}
    # All table hashes above were validated before any CONFIRM CSV read.
    compare_rows = []
    confirm_models = {}
    for model in [*(lock["candidates"]), "B5", "M1", "R1"]:
        path = out / "tables" / "confirm" / model / "confirm_auc.csv"
        auc = pd.read_csv(path)
        if set(auc.dataset) != {"D1","D2"} or len(auc) != 2 or \
                not auc.n_horizons.eq(13).all() or not auc.n_available_horizons.eq(13).all():
            raise ValueError(f"Incomplete CONFIRM AUC for {model}")
        for row in auc.itertuples(index=False):
            compare_rows.append([model,row.dataset,_fmt(row.AUC_MAE,4),_fmt(row.AUC_PeakMAE,4)])
        if primary is not None and model in (primary,"B5","M1","R1"):
            paths = {name:out / "tables" / "confirm" / model / f"confirm_{name}.csv"
                     for name in ("auc","pooled","daily_summary")}
            if any(path.relative_to(out).as_posix() not in done["tables"]
                   for path in paths.values()):
                return {"full_phase_complete":False,
                        "reason":f"confirm_metrics_incomplete_{model}","confirm_read":True}
            confirm_models[model] = {name:pd.read_csv(path) for name,path in paths.items()}
    diagnostics = None
    explore_diagnostics = None
    shift = None
    if primary is not None:
        frame = _frame(out,registry,primary,"CONFIRM")
        diagnostics = summarize_errors(frame,arm="CONFIRM",selected=True,
                                       production_col=production_col)
        shift = _period(frame,"2021-08-02","2021-08-09")
        explore_frame = _frame(out,registry,primary,"EXPLORE")
        explore_diagnostics = summarize_errors(explore_frame,arm="EXPLORE",selected=False,
                                               production_col=production_col)
    progress = generate_progress(root,selected_id=primary,production_col=production_col)
    # The progress document remains intact. Final text has its own verified
    # status and links to that exploration evidence without copying false.
    lines = ["# Phase F 개발구간 검증 대기 보고" if provisional else
             "# Phase F 최종 개발구간 보고","",
             ("**상태: 보고서 검증 대기. full_phase_complete=false.** 범위·CONFIRM·후속 진단의 "
              "잠금 근거를 점검했으며 최종 봉인·검사·커밋은 아직 끝나지 않았다."
              if provisional else
              "**상태: Phase F 요구 범위·CONFIRM·후속 진단 완료 기록 검증. "
              "full_phase_complete=true.** 이는 성능 개선 성공이나 현장 상용 적합성 판정과 다르다."),"",
             "## 선행 탐색과 후보 동결","",
             f"EXPLORE 비교 가능 후보 {len(tables)}개; 완료 표기와 근거 표 불일치 {len(omitted)}개. "
             "전체 등록·실패·미지원 및 모든 실행 시도는 아래 계열 표에 나타난다. "
             "탐색 전 범위와 정지 조건은 [completion gate](logs/completion_gate.json)에 기록됐다.",
             f"잠긴 primary: {primary if primary is not None else '없음 (적격 후보 부재)'}. "
             f"보조 후보 {len(lock['candidates'])-1 if primary is not None else 0}개. "
             "선택 lock, 설정·예측 파일 해시, split lock 및 CONFIRM 1회 완료 파일을 검증했다.",""]
    if primary is None:
        lines += ["적격 primary가 없으므로 새 대표 모델을 만들거나 기준선보다 좋은 모델이 확인됐다고 주장하지 않는다. "
                  + ("완료 판정은 최종 봉인 뒤에만 가능하다." if provisional else
                     "이 상태는 완료된 실패 결과일 수 있다."),""]
    else:
        lines += [f"Primary CONFIRM 적격성: {'충족' if done.get('primary_confirmed') is True else '미충족'}. "
                  "CONFIRM 결과로 후보를 다시 고르지 않았다.",""]
    top20 = [[r.exp_id,*(_fmt(getattr(r,key),4) for key in
              ("D1_MAE","D1_Peak","D2_MAE","D2_Peak","h16_MAE","h16_Peak"))]
             for r in rank.head(20).itertuples(index=False)]
    models = [name for name in [primary,"F0-1-B5","F0-1-M1","F0-1-R1"] if name is not None]
    normalized, peak_daily = _model_evidence(tables,models)
    lines += ["## EXPLORE 상위 20개와 시도 분모","",
              _md(["ID","D1 AUC-MAE","D1 AUC-PeakMAE","D2 AUC-MAE",
                   "D2 AUC-PeakMAE","h16 MAE","h16 Peak-MAE"],top20)
              if top20 else "완료된 비교 없음.","",
              "이는 탐색 결과 정렬이다. 모든 시도·실패·미지원의 분모를 함께 공개하며 "
              "EXPLORE 최저값을 선택 편향 보정된 검증 통계로 해석하지 않는다.","",
              _family_attempts(out,registry),"",
              "## 정규화 오차와 피크·일별 최대치","",
              normalized,"",
              "nMAE·CVRMSE·NMBE는 각 fold fit 평균으로 정규화한 horizon별 값의 "
              "13-horizon 단순 평균이다. NMBE 양수는 실제 수요 과소예측 방향이다. "
              "원본 전력 단위·집계 의미는 확인되지 않았다.","",
              peak_daily,"",
              "Actual-peak는 y>fold fit τ, predicted-peak는 pred>τ다. "
              "Daily maximum은 관측된 target-day 슬롯의 최대치이며 full_96만 별도 표시한다.","",
              "## 동일 키 paired 불확실성","",
              _paired_evidence(out,tables,primary),"",
              "차이는 MAE improvement=B5/M1/R1−후보, Peak-MAE degradation=후보−기준이다. "
              "EXPLORE bootstrap CI는 탐색 횟수에 따른 선택 편향을 교정하지 못한다.","",
              "### ISO-week 블록 bootstrap 민감도","",
              _week_ci_evidence(tables.get(primary,{}).get("pairwise_week_ci")
                                if primary is not None else None,
                                confirm_week,primary) if primary is not None else
              "적격 primary가 없어 주간 paired 민감도 비교는 해당 없음.","",
              "유효 ISO 주 수와 CI 미정의 사유를 표시한다. 날짜 블록 CI 대비 기간 의존성 "
              "민감도만 확인하며, 주간 CI는 후보 적격성이나 재선정에 사용하지 않았다.","",
              "## 정보량 가설의 정량 비교","",
              _information_budget(tables),"",
              "F1 core/union 및 F6 512/2048의 동일 개발자료 점추정 비교다. "
              "동시 변경된 모델·학습조건과 반복 탐색을 고려해야 하며 긴 문맥이 "
              "사람의 판단이나 상용 성능을 인과적으로 개선했다는 주장이 아니다.","",
              "## F9-3 완전 프로필 중복 민감도","",
              _dedup_evidence(tables),"",
              "keep/drop/weight와 D2의 EXPLORE 점추정은 중복 프로필 처리의 민감도다. "
              "동결 후보 적격성, 주간 CI 또는 독립 일반화 증거와 혼동하지 않는다.",""]
    lines += ["## 동일 개발자료의 잠금 CONFIRM","",
              "CONFIRM은 Phase C/E에서 이미 사용했던 과거 개발 score 날짜의 잠금 재확인이다. "
              "독립 비증강 미래 검증 또는 final holdout이 아니다.","",
              _md(["모델","집합","AUC-MAE","AUC-PeakMAE"],compare_rows),""]
    if confirm_models:
        confirm_normalized, confirm_peak_daily = _model_evidence(
            confirm_models,[name for name in (primary,"B5","M1","R1") if name is not None])
        lines += ["### CONFIRM 정규화 오차","",confirm_normalized,"",
                  "### CONFIRM 실제·예측 피크와 일별 최대치","",confirm_peak_daily,""]
    if shift is not None:
        lines += [f"Primary의 2021-08-02~08-08 CONFIRM shift: n={int(shift['n'])}, "
                  f"MAE={_fmt(shift['MAE'])}, peak n={int(shift['peak_n'])}, "
                  f"Peak-MAE={_fmt(shift['Peak_MAE'])}, signed peak error={_fmt(shift['signed_peak_error'])}. "
                  "07-26~08-01 EXPLORE와 arm을 섞어 재선정하지 않았다.","",
                  f"F10 CONFIRM 진단 표본 {diagnostics['audit']['score_n']}행, fold 1 "
                  f"{diagnostics['audit']['fold_1_n']}행. EXPLORE의 시간대·요일·휴일 인접·"
                  "생산량 사후구간은 진행 보고에 별도로 보존된다.",""]
        earlier = _period(explore_frame,"2021-07-26","2021-08-02")
        lines += [f"앞선 2021-07-26~08-01 EXPLORE: n={int(earlier['n'])}, "
                  f"MAE={_fmt(earlier['MAE'])}, peak n={int(earlier['peak_n'])}, "
                  f"Peak-MAE={_fmt(earlier['Peak_MAE'])}, "
                  f"signed peak error={_fmt(earlier['signed_peak_error'])}. "
                  "두 arm 모두 동결 이후에만 나란히 기술했다.",""]
        for arm, detail in (("EXPLORE",explore_diagnostics),("CONFIRM",diagnostics)):
            for key,label in (("by_hour","hour"),("by_weekday","weekday 0=Mon"),
                              ("by_holiday_context","holiday-adjacent"),
                              ("by_production_posthoc","completed production bins")):
                item = detail.get(key)
                if not isinstance(item,pd.DataFrame) or item.empty:
                    lines += [f"{arm} {label}: 가능한 진단 표본·열이 없어 판정불가. "
                              "원자료/미래 loader를 추가 호출하지 않았다.",""]
                    continue
                lines += [f"### F10 {arm} {label}","",
                          _md(["구간","n","MAE","bias pred−actual","peak n","Peak-MAE","FP/FN"],[
                              [r.group,int(r.n),_fmt(r.mae),_fmt(r.bias_pred_minus_actual),
                               int(r.peak_n),_fmt(r.peak_mae),
                               f"{int(r.false_positive_n)}/{int(r.false_negative_n)}"]
                              for r in item.itertuples(index=False)]),""]
        lines += ["생산량 구간은 origin 이전 완료·가용 시각이 검증된 별도 열이 prediction artifact에 "
                  "전달된 경우에만 사후 진단으로 계산한다. 입력 특징 추가나 후보 선택에 사용하지 않는다.",""]
    lines += ["## F11 확률·상한·경보 후속 진단","",
              downstream_text,""]
    if downstream_links:
        lines += [f"전체 고정 C/L grid, 두 정책, 모든 fold와 D2의 원표: {downstream_links}",""]
    lines += ["## F9 주간 재학습 사후 진단","",weekly_text,""]
    lines += ["## Foundation 사용 허가 기록","",
              _license_evidence(out),"",
              "License flag는 저장된 weight 감사의 기계 기록이다. 법률 검토나 실제 상업 배포 승인으로 "
              "해석하지 않는다. 미지원·비상업 표기 모델을 상용 후보라고 주장하지 않는다.","",
              "## 원자료·근거 경로","",
              "[EXPLORE 진행 표](progress_report.md), [selection lock](logs/confirm_lock.json), "
              "[CONFIRM 완료 기록](logs/confirm_complete.json), "
              "[split lock](logs/split_lock.json), [downstream 상태](logs/downstream_status.json), "
              "[Phase F registry](registry.csv). 각 후보의 원표는 tables/<실험 ID>/explore_*.csv, "
              "잠금 CONFIRM 표는 tables/confirm/<실험 ID>/confirm_*.csv에 있다. "
              "최종 holdout 결과 경로는 이 보고서의 입력이 아니다.",""]
    lines += ["## 결론과 한계","",
              (("보고서 준비 단계의 full_phase_complete=false는 F0–F11 근거를 모았더라도 "
                "최종 봉인·검사·커밋이 남아 있음을 뜻한다. "
                "Primary 개선 성공 여부는 위 적격성 결과와 paired 지표로만 판단한다.")
               if provisional else
               ("Phase F의 full_phase_complete=true는 모든 F0–F11 범위, 확장 종료, "
                "F9/F10/F11 후속 진단을 확인했다는 completion gate의 범위 기록이다. "
                "Primary 개선 성공 여부는 위 적격성 결과와 paired 지표로만 판단한다.")),
              "동일 공통 cohort는 100,010 score 키다. EXPLORE 반복 탐색과 기존 개발자료 재사용은 "
              "CONFIRM을 새 독립 검증으로 만들지 않는다. 증강 profile 반복·짧은 기간·데이터 단위 및 "
              "현장 허용기준 미확인은 상용 적합성 판정의 한계다. ASHRAE 참조는 상용 인증이 아니다.",
              "이번 보고 경로는 final holdout·historical final 결과를 읽지 않았다. "
              "별도 승인과 전체 pipeline freeze 없이 final evaluation을 수행하지 않는다.",""]
    path = out / ("PHASE_F_REPORT.pending.md" if provisional else "PHASE_F_REPORT.md")
    path.write_text("\n".join(lines),encoding="utf-8")
    return {"path":str(path),"report_ready":True,
            "full_phase_complete":not provisional,
            "primary_candidate":primary,"primary_confirmed":done.get("primary_confirmed"),
            "confirm_read":True,"progress_path":str(progress["path"])}
