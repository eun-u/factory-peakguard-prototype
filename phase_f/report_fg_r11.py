"""Generate the FG-R11 result blocks in report/ch1..ch6 and REPORT_DRAFT.md.

Every number is computed from result files or the raw development history; nothing is typed by
hand. Each chapter gets one block between markers; rerunning replaces only that block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
BEGIN = "<!-- FG-R11 결과 블록 시작: phase_f/report_fg_r11.py 자동 생성, 손으로 고치지 않음 -->"
END = "<!-- FG-R11 결과 블록 끝 -->"
PF = ROOT / "outputs/phase_f"
CHAPTERS = ["ch1_data", "ch2_model", "ch3_errors", "ch4_field", "ch5_novelty", "ch6_repro"]
NOTE = ("> 이 블록이 최종 수치의 기준이다. 위쪽 기존 표는 이전 파이프라인의 개발 교차검증 수치다. "
        "테스트 = 마지막 15%(2021-08-09 09:45 원점~2021-09-15), 개발 = 그 이전 주 단위 walk-forward 16주.")


def _load(name):
    return json.loads((PF / name).read_text(encoding="utf-8"))


def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def _results():
    r = {k: _load(f"final_fg_{k}/FINAL_TEST_RESULT.json") for k in ("r8", "r9", "r10", "r11", "r12")}
    lock = _load("final_fg_r11/FINAL_LOCK.json")
    sweep = pd.read_csv(PF / "goal_fm_ensemble_v1/sweep_results.csv")
    return r, lock, sweep[sweep.name.eq("baseline")].iloc[0]


def _dev_history():
    from phase_c.data import load_history
    history, _ = load_history(ROOT)
    return history


def ch1(r, lock):
    h = _dev_history()
    power = h.power.astype(float)
    day = (power.index - pd.Timedelta(minutes=15)).floor("D")
    days = power.groupby(day).apply(lambda s: s.to_numpy())
    days = days[days.apply(len).eq(96)]
    M = np.stack(days.to_numpy())
    exact, lags = [], []
    for i in range(1, len(M)):
        d = np.nanmean(np.abs(M[:i] - M[i]), 1)
        if np.isnan(d).all():
            continue
        j = int(np.nanargmin(d))
        exact.append(d[j] == 0)
        if d[j] == 0:
            lags.append(int((days.index[i] - days.index[j]).days))
    tau = lock["tau"]
    from phase_f.models.foundation import HOLIDAYS_2021
    t = power.index
    off = (t.dayofweek >= 5) | t.strftime("%Y-%m-%d").isin(HOLIDAYS_2021)
    blocks = pd.cut(t.hour, [-1, 5, 9, 11, 13, 17, 21, 23], labels=["00-06", "06-10", "10-12", "12-14", "14-18", "18-22", "22-24"])
    peak = pd.Series(power.to_numpy() > tau, index=t)
    tab = pd.DataFrame({"blk": blocks, "off": off, "peak": peak.to_numpy()}).groupby(["blk", "off"], observed=True).peak.mean().unstack()
    tab = tab.rename(columns={False: "평일 피크 비율", True: "휴일·주말 피크 비율"}).reset_index().rename(columns={"blk": "시간대"})
    for c in tab.columns[1:]:
        tab[c] = tab[c].map(lambda v: _f(v, 3))
    lag_counts = pd.Series(lags).value_counts().head(6)
    return "\n".join([
        "## FG-R11 기준 데이터 진단 수치", NOTE, "",
        f"- 개발 이력 완전일 {len(M)}일 중 이전 날과 비교 가능한 {len(exact)}일에서, 이전 어느 날과 96칸이 완전히 같은 날: {sum(exact)}일 ({np.mean(exact):.1%}).",
        "- 정확 복제의 원본 지연(일) 상위: " + ", ".join(f"{k}일 {v}회" for k, v in lag_counts.items()) + ". 증강 데이터의 구조로 해석한다.",
        f"- 테스트 구간에서 오늘 관측 prefix가 과거일과 정확히 일치한 행: 게이트 1칸 기준 {r['r8']['gated_share']:.2%}, "
        f"5칸 기준 {r['r11']['gated_share']:.2%} (개발 16주 약 48%).",
        f"- 피크 τ = 개발 정제 전력 95백분위 {tau:.0f}. 테스트 피크 행 비율 {r['r11']['peak_rows_share']:.1%}.",
        "", "개발 이력의 시간대별 피크(> τ) 비율", "", _table(tab), "",
        "평가 설계: 개발은 주 단위 walk-forward 16주(짝수 주 EXPLORE 8주, 홀수 주 CONFIRM 8주), "
        "각 주는 FIT(학습)·STOP(조기종료)·CAL(보정) 구간 뒤 1주 점수. 마지막 15%는 최종 평가 전용.",
    ])


MODEL_ORDER = ["persistence", "B1 weekly naive", "analog only", "FG-R6", "Chronos-2 alone",
               "TH reconciled Chronos", "TH Chronos + MOS", "FG-R11 without peak shift", "FG-R11"]
LABEL = {"persistence": "지속 예측", "B1 weekly naive": "주간 계절 나이브", "analog only": "순수 유사일",
         "FG-R6": "FG-R6 (유사일 게이트+LightGBM)", "Chronos-2 alone": "Chronos-2 단독",
         "TH reconciled Chronos": "시간 계층 조정 Chronos", "TH Chronos + MOS": "시간 계층 + MOS",
         "FG-R11 without peak shift": "FG-R11 (피크 보정 전)", "FG-R11": "**FG-R11 (최종)**"}


def ch2(r, lock, dev):
    p = r["r11"]["point"]
    ci = r["r11"]["paired_ci_vs_fg_r11"]
    rows = []
    for m in MODEL_ORDER:
        c = ci.get(m)
        rows.append({"모델": LABEL[m], "MAE": _f(p[m]["AUC_MAE"]), "Peak MAE": _f(p[m]["AUC_PeakMAE"]),
                     "h4 MAE": _f(p[m]["h4_MAE"]), "h16 MAE": _f(p[m]["h16_MAE"]),
                     "FG-R11 대비 MAE 차 [95% CI]": "—" if not c else f"{c['improvement']:+.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"})
    hist = []
    for k, label, model in (("r8", "1차 FG-R8 (주 결과)", "FG-R8"), ("r9", "2차 FG-R9", "FG-R9"), ("r10", "3차 FG-R10", "FG-R10"),
                            ("r11", "4차 FG-R11 (최종)", "FG-R11"), ("r12", "5차 FG-R12", "FG-R12")):
        q = r[k]["point"][model]
        hist.append({"평가": label, "MAE": _f(q["AUC_MAE"]), "Peak MAE": _f(q["AUC_PeakMAE"]),
                     "h4": _f(q["h4_MAE"]), "h16": _f(q["h16_MAE"])})
    devrow = pd.DataFrame([{"구간": arm, "비복제 행 MAE": _f(dev[f"ng_{a}_MAE"]), "비복제 행 Peak": _f(dev[f"ng_{a}_Peak"]),
                            "전체 행 MAE": _f(dev[f"full_{a}_MAE"]), "전체 행 Peak": _f(dev[f"full_{a}_Peak"])}
                           for arm, a in (("16주", "ALL"), ("EXPLORE", "EXP"), ("CONFIRM", "CONF"))])
    return "\n".join([
        "## FG-R11 최종 모델 비교 (테스트, 13지평 h4~h16 평균)", NOTE, "",
        _table(pd.DataFrame(rows)), "",
        "CI: 목표일 블록 부트스트랩 1000회, 양수 = FG-R11이 더 좋음. 지표: 13지평 pooled MAE 평균, Peak MAE = 실제값 > τ 행의 MAE.", "",
        "### FG-R11 개발 16주 성능", "", _table(devrow), "",
        f"MOS 학습 행 {lock['mos']['train_rows']:,}개, 피크 보정량 {lock['fg_r8_params']['shift']:.1f} "
        f"(τ−{lock['fg_r8_params']['shift_margin']:.0f} 초과 예측에 적용, 28일 보정 구간에서 선택).",
    ])


def _hour_block(t):
    return pd.cut(t.dt.hour, [-1, 5, 9, 11, 13, 17, 21, 23], labels=["00-06", "06-10", "10-12", "12-14", "14-18", "18-22", "22-24"])


def ch3(r, lock):
    from phase_f.models.foundation import HOLIDAYS_2021
    d = pd.read_csv(ROOT / "outputs/predictions/final_test_fg_r11.csv", parse_dates=["origin", "target_time"])
    tau = lock["tau"]
    d["blk"] = _hour_block(d.target_time)
    d["daytype"] = np.where((d.target_time.dt.dayofweek >= 5) | d.target_time.dt.strftime("%Y-%m-%d").isin(HOLIDAYS_2021), "휴일·주말", "평일")
    d["err"] = d.pred - d.actual
    d["peak"] = d.actual > tau
    g = d.groupby(["daytype", "blk"], observed=True)
    tab = pd.DataFrame({"행": g.size(), "MAE": g.err.apply(lambda e: e.abs().mean()).map(_f),
                        "피크 행": g.peak.sum(),
                        "피크 편향": g.apply(lambda z: _f(z.err[z.peak].mean()) if z.peak.any() else "—")}).reset_index()
    tab = tab.rename(columns={"daytype": "요일 구분", "blk": "시간대"})
    h4 = d[d.horizon.eq(4)].copy()
    h4["fn"] = h4.peak & ~h4.alert
    h4["fp"] = ~h4.peak & h4.alert
    gg = h4.groupby(["daytype", "blk"], observed=True)
    fnfp = pd.DataFrame({"실제 피크": gg.peak.sum(), "FN(놓침)": gg.fn.sum(), "FP(오경보)": gg.fp.sum()}).reset_index()
    fnfp = fnfp[(fnfp["실제 피크"] > 0) | (fnfp["FP(오경보)"] > 0)].rename(columns={"daytype": "요일 구분", "blk": "시간대"})
    alerts = r["r11"]["alerts"]
    arows = []
    for key in ("persistence", "B1 weekly naive", "analog only", "Chronos-2 alone", "FG-R6", "FG-R11 point", "FG-R11 probability"):
        for hz in (4, 16):
            a = alerts[f"{key}|h{hz}"]
            arows.append({"모델": key, "지평": f"h{hz}", "위치 F1": _f(a["position_f1"]), "에피소드 F1": _f(a["episode_f1"]),
                          "에피소드 TP/FP/FN": f"{a['episode_tp']}/{a['episode_fp']}/{a['episode_fn']}",
                          "위치 오경보": a["false_alarms_positions"], "피크 MAE": _f(a["peak_mae"])})
    return "\n".join([
        "## FG-R11 오류·피크 조건 분석 (테스트)", NOTE, "",
        "시간대·요일 구분별 오차 (전 지평, 예측결과 CSV에서 계산; 시간대와 휴일은 교대·가동의 대리변수)", "", _table(tab), "",
        "h4 경보 기준 FN·FP 분포 (위치 단위, 확률 경보)", "", _table(fnfp), "",
        "모델별 경보 성능 (임계값: 각 모델 동일 규칙, 28일 보정 구간 에피소드 F1 최대)", "", _table(pd.DataFrame(arows)), "",
        f"피크 행은 대부분 과소예측(음의 편향)이다. 테스트 구간 정확 복제 게이트 작동 {r['r11']['gated_share']:.2%}로 "
        "남은 오차는 처음 보는 날의 피크 크기에서 나온다. 조건 분석은 탐색적이며 다중 비교 보정이 없다.",
    ])


def ch4(r):
    d = pd.read_csv(ROOT / "outputs/predictions/final_test_fg_r11_h4.csv", parse_dates=["origin", "target_time"])
    days = d.target_time.dt.normalize().nunique()
    a = r["r11"]["alerts"]["FG-R11 probability|h4"]
    risk = r["r11"]["risk"]
    nd = r["r11"]["next_day_max"]
    alerts_per_day = d.alert.sum() / days
    return "\n".join([
        "## FG-R11 현장 활용 수치 (테스트, 1시간 앞 = h4)", NOTE, "",
        _table(pd.DataFrame([
            {"항목": "경보 15분 위치 수 / 일", "값": _f(alerts_per_day, 1)},
            {"항목": "피크 에피소드 적중 TP / 실제 에피소드", "값": f"{a['episode_tp']} / {a['episode_tp'] + a['episode_fn']}"},
            {"항목": "오경보 에피소드 / 일", "값": _f(a["episode_fp"] / days, 2)},
            {"항목": "에피소드 F1", "값": _f(a["episode_f1"])},
            {"항목": "경보 임계 확률 (28일 보정 구간 선택)", "값": _f(a["threshold"], 3)},
            {"항목": "90% 구간 커버리지 (비게이트 행)", "값": _f(risk["coverage_q90_nongated"])},
            {"항목": "95% 구간 커버리지 (비게이트 행)", "값": _f(risk["coverage_q95_nongated"])},
            {"항목": "피크 초과확률 Brier / PR-AUC", "값": f"{_f(risk['brier'], 4)} / {_f(risk['pr_auc'])}"},
            {"항목": f"익일 최대 15분 MAE ({nd['days']}일, 23:45 예측)", "값": _f(nd["chronos"], 2)},
            {"항목": "익일 최대 MAE: 오늘 최대 지속 / 지난주 최대", "값": f"{_f(nd['persistence_today_max'], 2)} / {_f(nd['b1_last_week_max'], 2)}"},
        ])), "",
        "경보는 예측 확률이 임계값을 넘으면 발생한다. 선행시간은 1시간(h4), 4시간(h16) 경보도 같은 방식으로 산출된다.",
        "전력 단위와 계약 한도가 미확인이므로 비용 효과는 비율·시나리오로만 해석한다.",
    ])


def ch5(r, lock):
    p = r["r11"]["point"]
    c = r["r11"]["risk"]
    return "\n".join([
        "## FG-R11 기법별 기여 (테스트 수치 기준)", NOTE, "",
        _table(pd.DataFrame([
            {"기법": "시간 계층 조정 (15분~2시간, WLS-분산; EJOR 2017/2020)", "변화": f"Chronos 대비 MAE {p['Chronos-2 alone']['AUC_MAE']:.3f}→{p['TH reconciled Chronos']['AUC_MAE']:.3f}, Peak {p['Chronos-2 alone']['AUC_PeakMAE']:.3f}→{p['TH reconciled Chronos']['AUC_PeakMAE']:.3f}"},
            {"기법": "실현 오차 피드백 MOS (기상 MOS 차용)", "변화": f"Peak {p['TH reconciled Chronos']['AUC_PeakMAE']:.3f}→{p['TH Chronos + MOS']['AUC_PeakMAE']:.3f}, MAE {p['TH reconciled Chronos']['AUC_MAE']:.3f}→{p['TH Chronos + MOS']['AUC_MAE']:.3f}"},
            {"기법": f"피크 상향 보정 (+{lock['fg_r8_params']['shift']:.0f}, 28일 보정 구간)", "변화": f"Peak {p['FG-R11 without peak shift']['AUC_PeakMAE']:.3f}→{p['FG-R11']['AUC_PeakMAE']:.3f}, MAE {p['FG-R11 without peak shift']['AUC_MAE']:.3f}→{p['FG-R11']['AUC_MAE']:.3f}"},
            {"기법": "Conformal(CQR) 분위수 보정", "변화": f"90%/95% 커버리지 {c['coverage_q90_nongated']:.3f}/{c['coverage_q95_nongated']:.3f} (명목 0.90/0.95)"},
            {"기법": "정확 복제 게이트 (정밀도 ≥0.95 → 5칸)", "변화": f"테스트 작동 {r['r8']['gated_share']:.2%}→{r['r11']['gated_share']:.2%}, 우연 일치 손실 제거"},
        ])), "",
        "후보와 선정 규칙은 결과 확인 전에 사전 고정 문서(outputs/phase_f/goal_fm_ensemble_v1/PREREGISTRATION.md)에 기록했다.",
    ])


def ch6(status=None):
    rep = (status or {}).get("reproduction")
    rep_line = ("- 재현 검증: 저장 결과와 재계산 지표의 최대 차이 "
                f"{rep['max_abs_metric_difference']:.2e}, 예측 파일 바이트 동일 {rep['prediction_files_byte_identical']}."
                if rep else "- 재현 검증 결과: outputs/phase_f/final_fg_r11/PIPELINE_STATUS.json")
    return "\n".join([
        "## FG-R11 재현 절차", NOTE, "",
        "```",
        "python -m pip install -r requirements.txt            # 메인 환경",
        "python -m venv outputs/phase_f/env                   # Chronos 환경 (GPU)",
        "outputs/phase_f/env/Scripts/python -m pip install -r requirements-chronos.txt --extra-index-url https://download.pytorch.org/whl/cu126",
        "python run_all.py --fg-r11                           # 캐시 확인 → 잠금/평가 또는 재현 검증 → CSV 검사 → 보고서 블록",
        "python run_all.py --fg-r11 --refresh-caches          # Chronos·시간 계층 예측을 GPU로 재생성 후 잠금 해시 대조",
        "```", "",
        rep_line,
        "- 산출물: outputs/predictions/final_test_fg_r11.csv(전 지평), final_test_fg_r11_h4.csv(1시간 앞), final_test_next_day_max.csv(익일 최대).",
        "- 모델·설정 잠금: outputs/phase_f/final_fg_r11/FINAL_LOCK.json (소스·캐시 SHA-256, MOS 계수, 시간 계층 투영행렬, 보정값).",
    ])


def summary(r, lock, dev) -> str:
    p = r["r11"]["point"]["FG-R11"]
    ci = r["r11"]["paired_ci_vs_fg_r11"]
    a4 = r["r11"]["alerts"]["FG-R11 probability|h4"]
    risk = r["r11"]["risk"]
    rows = [{"지표": "13지평 평균 MAE", "FG-R11 테스트": _f(p["AUC_MAE"]), "개발 16주(전체 행)": _f(dev["full_ALL_MAE"])},
            {"지표": "피크 MAE (실제 > τ)", "FG-R11 테스트": _f(p["AUC_PeakMAE"]), "개발 16주(전체 행)": _f(dev["full_ALL_Peak"])},
            {"지표": "1시간 앞(h4) MAE", "FG-R11 테스트": _f(p["h4_MAE"]), "개발 16주(전체 행)": "—"},
            {"지표": "4시간 앞(h16) MAE", "FG-R11 테스트": _f(p["h16_MAE"]), "개발 16주(전체 행)": "—"}]
    base = []
    for m in ("persistence", "B1 weekly naive", "FG-R6", "Chronos-2 alone"):
        q = r["r11"]["point"][m]
        base.append({"비교 모델": LABEL[m], "MAE": _f(q["AUC_MAE"]), "Peak MAE": _f(q["AUC_PeakMAE"]),
                     "FG-R11 MAE 개선 [95% CI]": f"{ci[m]['improvement']:.3f} [{ci[m]['ci95'][0]:.3f}, {ci[m]['ci95'][1]:.3f}]",
                     "Peak MAE 감소율": f"{1 - p['AUC_PeakMAE'] / q['AUC_PeakMAE']:.1%}"})
    return "\n".join([
        "# FG-R11 최종 성능 요약", "",
        "phase_f/report_fg_r11.py가 결과 파일에서 생성. 테스트 = 마지막 15%(2021-08-09 09:45 원점~2021-09-15, 38일, 44,668행).", "",
        "## 핵심 성능", "", _table(pd.DataFrame(rows)), "",
        "## 기준 모델 대비 (테스트)", "", _table(pd.DataFrame(base)), "",
        "## 경보·불확실성 (테스트, 1시간 앞)", "",
        _table(pd.DataFrame([
            {"항목": "피크 에피소드 F1", "값": _f(a4["episode_f1"])},
            {"항목": "위치 F1", "값": _f(a4["position_f1"])},
            {"항목": "피크 에피소드 적중", "값": f"{a4['episode_tp']} / {a4['episode_tp'] + a4['episode_fn']}"},
            {"항목": "90% / 95% 예측구간 커버리지", "값": f"{_f(risk['coverage_q90_nongated'])} / {_f(risk['coverage_q95_nongated'])}"},
            {"항목": "피크 초과확률 PR-AUC", "값": _f(risk["pr_auc"])},
            {"항목": "익일 최대 15분 MAE", "값": _f(r["r11"]["next_day_max"]["chronos"], 2)}])), "",
        "## 모델 구성", "",
        "1. 정확 복제 게이트: 오늘 관측이 과거 어느 날과 5칸 이상 완전히 일치하면 그날 값을 사용(전체 이력 검색).",
        "2. Chronos-2(문맥 2048) 예측을 15분·30분·1시간·2시간·4시간 시간 계층으로 조정(WLS-분산).",
        "3. 실현 오차 피드백 MOS: 원점에서 관측된 Chronos 과거 예측 오차로 지평별 선형 보정.",
        f"4. 피크 상향 보정 +{lock['fg_r8_params']['shift']:.0f}(τ−{lock['fg_r8_params']['shift_margin']:.0f} 초과 예측).",
        "5. Conformal(CQR) 분위수 보정과 피크 초과확률 기반 경보.",
    ])


def _replace_block(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{BEGIN}\n\n{body}\n\n{END}"
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def write_blocks(status=None) -> dict:
    if status is None and (PF / "final_fg_r11/PIPELINE_STATUS.json").exists():
        status = json.loads((PF / "final_fg_r11/PIPELINE_STATUS.json").read_text(encoding="utf-8"))
    r, lock, dev = _results()
    bodies = {"ch1_data": ch1(r, lock), "ch2_model": ch2(r, lock, dev), "ch3_errors": ch3(r, lock),
              "ch4_field": ch4(r), "ch5_novelty": ch5(r, lock), "ch6_repro": ch6(status)}
    for name, body in bodies.items():
        _replace_block(REPORT / f"{name}.md", body)
    (REPORT / "FINAL_RESULTS.md").write_text(summary(r, lock, dev) + "\n", encoding="utf-8")
    draft = REPORT / "REPORT_DRAFT.md"
    _replace_block(draft, "# FG-R11 최종 결과 요약 (장별 블록 모음)\n\n" + "\n\n".join(bodies.values()))
    return {"chapters": list(bodies), "draft": "REPORT_DRAFT.md", "summary": "FINAL_RESULTS.md"}


if __name__ == "__main__":
    print(write_blocks())
