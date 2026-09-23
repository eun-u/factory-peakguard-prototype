"""Assemble the evidence-first comparison from the two task outputs."""

from __future__ import annotations

import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results"


def load_summary(number):
    path = RESULTS / f"{number}_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"headline": "결과 파일 없음", "criteria": {}}


def load_section(number, fallback):
    path = RESULTS / f"{number}_section.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else fallback


def make_report():
    t3 = load_summary("03")
    t5 = load_summary("05")
    press_detail_path = RESULTS / "03_press.json"
    press_detail = json.loads(press_detail_path.read_text(encoding="utf-8")) if press_detail_path.exists() else {}
    k3 = t3.get("criteria", {})
    k5 = t5.get("criteria", {})
    press_available = t3.get("status") == "ok"
    press_data_sentence = ("구간·세션 단위 분석 결과를 P0–P5에 기록한다."
                           if press_available else "현 ③ 원자료가 없으면 실제 수치는 확인할 수 없다.")
    press_question = ("③ 정상·이상 파일의 수집 세션과 검증 가능한 실제 고장 발생·복귀 시각은 무엇인가?"
                      if press_available else "③ 정상·이상 원본 파일의 현재 경로, 실제 라벨·타임스탬프 정의, 수집 세션 및 고장 발생 시점은 무엇인가?")
    press_assumption = ("③ candidate segment 경계는 양의 표본간격 중앙값의 5배를 초과하는 gap으로 정한다. 이는 물리적 press cycle의 확인이 아니다. 관계 Ridge는 시차 없이 같은 시점의 채널을 사용하며 P5 레벨 점수는 시험 성능을 보고 고르지 않은 raw Mahalanobis로 고정했다."
                        if press_available else "③ 원자료가 없으므로 분절 기준·원파형 여부·세션 독립성·라벨 이벤트 수를 관측값으로 채우지 않는다. 데이터가 제공되면 gap 경계와 센서 의미를 확인한다.")
    press_repro = ("20,600행 원자료에서 구간 생성·모델·주입·경보·CI까지 한 명령으로 재현된다. 정상·이상 파일이 별도 날짜라는 경계를 보고서에 남긴다."
                   if press_available else "원자료 제공 시 구간 생성부터 신뢰구간까지 한 명령으로 재실행할 수 있다. 현재 원자료 부재 여부를 명시적으로 남긴다.")
    one_hour = t5.get("one_hour", {})
    daily = t5.get("daily", {})
    one_metrics = one_hour.get("metrics", {})
    daily_metrics = daily.get("metrics", {})
    if one_metrics and daily_metrics:
        interpretation = (
            "### ⑤ 결과 해석\n\n"
            f"1시간 후 예측에서는 학습 상위 5% 초과 위치의 MAE가 선택된 나이브 {one_metrics['naive_peak_mae']:.2f}에서 "
            f"LightGBM {one_metrics['lgb_peak_mae']:.2f}로 낮아졌다. 반면 익일 최대값의 MAPE 점추정은 나이브 "
            f"{daily_metrics['naive_mape']:.1f}%보다 LightGBM {daily_metrics['lgb_mape']:.1f}%가 높았다. "
            f"0.95 분위수의 실제 누적 커버리지는 {one_metrics['q95_coverage']:.3f}로 목표 0.95보다 낮다. "
            "따라서 예측거리와 지표에 따라 우위가 다르고, 상단 위험의 보정도 충분하지 않다."
        )
    else:
        interpretation = "⑤ 결과 해석은 실행 수치가 마련되면 작성한다."
    recommendation = ("**판정 보류.** ③은 실제 이상 구분 AUROC가 높지만 이상은 한 측정 세션이고 관계 붕괴 주입을 구분하지 못해 K3-b가 해당한다. ⑤는 1시간 후 피크 구간 오차를 크게 줄였지만 완전한 하계·동계 창이 없고 익일 최대 MAPE와 95% 분위수 보정에는 약점이 있다. 어느 쪽도 새 세션·새 계절에서의 일반화와 현장 조치 효과를 입증하지 못하므로, 단일 점수로 과제 우열을 정하지 않는다."
                      if press_available else "**판정 보류.** ③ 원자료가 현재 작업 폴더에 없어 K3-a~c와 모델 비교를 실행할 수 없으므로, 두 과제의 타당성을 공정하게 서열화할 근거가 없다. ⑤ 결과는 해당 데이터에서 가능한 미래예측·경보 주장 범위를 정하는 데 사용하되 ③ 선택 여부의 대리 근거로 쓰지 않는다.")
    press_section = (load_section("03", "③ 실행 결과 없음.") if press_available else
                     "P0–P5 **판정불가**. 현재 프로젝트 폴더와 ZIP 내부에서 정상·이상 라벨을 모두 갖춘 3채널 진동·전류 원자료를 찾지 못했다. 후보 구간 수나 AUROC를 이전 DOCX의 탐색값으로 대체하지 않았다. `t3_press.py`는 파일이 제공되면 이 절의 실험을 생성한다.")
    if press_available:
        press_section = press_section.removeprefix("# ③ 사전 타당성 검증\n\n").replace("\n## ", "\n### ")
        translations = {
            "Candidate segments are defined by timestamp gaps, not verified press cycles.": "후보 구간은 시간 공백으로 나눈 것이며 확인된 프레스 사이클이 아니다.",
            "Abnormal segments may belong to one recording session; segment CIs do not capture independent-failure uncertainty.": "이상 구간은 한 기록 세션일 수 있다. 구간 재표집 CI는 독립 고장 사건의 불확실성을 반영하지 못한다.",
            "Normalized performance loss can reflect session shift or genuine fault-related level change.": "정규화 후 성능 하락은 세션 차이 또는 실제 고장 레벨 변화 모두와 양립한다.",
            "No fault onset or return-to-normal timestamp is verified; release time is unmeasurable.": "확인된 고장 시작·정상 복귀 시각이 없어 해제 시간을 측정할 수 없다.",
            "P1 score orientation uses both labels for diagnosis and is not a deployable classifier.": "P1 점수 방향은 두 라벨을 보고 정했으므로 진단용이며 운영용 분류기가 아니다.",
            "P5 detection delay is from abnormal file start; it is not validated early-warning lead time.": "P5 지연은 이상 파일 시작 기준이며 검증된 고장 전 조기경보 시간이 아니다.",
            "Channel shuffle interpolates donor segments when lengths differ; this also alters shape and autocorrelation.": "채널 교체 때 길이가 다른 제공 구간을 보간해 형태와 자기상관도 달라질 수 있다.",
            "Sequential chatter and one-session detection delay lack a defensible independent-event bootstrap CI.": "채터링과 한 세션의 탐지 지연에는 독립 사건 단위 CI를 붙이기 어렵다.",
        }
        for old, new in translations.items():
            press_section = press_section.replace(old, new)
    power_section = load_section("05", "⑤ 실행 결과가 없어 Q0–Q5 판정 불가.")
    power_section = power_section.removeprefix("## ⑤ Q0–Q5 검증 요약\n\n")
    press_extra = []
    if press_available and press_detail.get("status") == "ok":
        p0 = press_detail["p0"]
        operands = press_detail["warning_operands"]
        p4_offset = next((r for r in press_detail["p4"] if r["injection"] == "all_channel_offset_2.0sigma" and r["score"] == "relation"), None)
        p5_ref = next((r for r in press_detail["p5"] if r["alpha"] == .01 and r["rule"] == "1/1"), None)
        observed_minutes = (60 * p5_ref["false_alarm_events"] / p5_ref["alarms_per_operating_hour"]
                            if p5_ref and p5_ref["alarms_per_operating_hour"] else None)
        press_extra += [
            "### ③ 수치 해석과 경보 표",
            "",
            f"P0의 기록 간격 중앙값은 {p0['nominal_sample_seconds']:.3g}초이고 99백분위 gap은 "
            f"{p0['gap_quantiles_seconds']['0.99']:.2f}초다. gap {p0['gap_threshold_seconds']:.3g}초 초과를 candidate segment 경계로 정했다. "
            "정상과 이상은 서로 다른 날짜의 파일이므로 라벨과 수집 세션을 분리해 검증하지 못한다.",
            "",
            "라벨 | 측정 범위 | 표본 수 | 후보 구간 수 | 구간 길이 중앙값",
            ":-- | :-- | --: | --: | --:",
        ]
        for row in p0["files"]:
            label = "정상" if row["label"] == 0 else "이상"
            press_extra.append(f"{label} | {row['first']} ~ {row['last']} | {row['sample_count']:,} | {row['segment_count']} | {row['segment_length_samples']['0.5']:.0f}샘플")
        press_extra += [
            "",
            "채널 | 음수 비율 | 범위 | 구간 내 lag-1 자기상관 중앙값",
            ":-- | --: | :-- | --:",
        ]
        for name, value in p0["channels"].items():
            press_extra.append(f"{name} | {value['negative_fraction']:.3f} | {value['minimum']:.2f} ~ {value['maximum']:.2f} | {value['median_segment_lag1_autocorrelation']:.3f}")
        press_extra += [
            "",
            "부호와 짧은 기록 간격은 파형 가능성을 시사하지만 원센서 취득·집계 방식은 확인되지 않았다. press cycle이라는 물리 단위로 해석하지 않는다.",
            "",
            f"K3-a의 최고 평균·표준편차 단일특징 AUROC는 {operands['best_mean_std_auroc']:.3f}, 최고 정규화 점수는 {operands['best_normalized_auroc']:.3f}, 합산 관계 잔차는 {operands['combined_residual_auroc']:.3f}이다. "
            f"구간 길이 단독 AUROC는 {operands['segment_length_auroc']:.3f}이다. 사전 판정은 K3-a {k3.get('K3-a', '판정불가')}, K3-c {k3.get('K3-c', '판정불가')}이다.",
            "",
            "P2 표현별 AUROC 범위는 3개 정상 학습 모델의 시험 결과를 요약한 것이다. 모델 선택에 시험 라벨을 쓰지 않았다.",
            "",
            "표현 | AUROC 범위 | PR-AUC 범위",
            ":-- | :-- | :--",
        ]
        for representation in ("raw", "center", "z", "amplitude"):
            rows = [r for r in press_detail["p2"] if r["representation"] == representation]
            press_extra.append(f"{representation} | {min(r['auroc'] for r in rows):.3f}–{max(r['auroc'] for r in rows):.3f} | {min(r['pr_auc'] for r in rows):.3f}–{max(r['pr_auc'] for r in rows):.3f}")
        press_extra += [
            "",
            "P3 관계 잔차 | AUROC (95% CI) | PR-AUC (95% CI)",
            ":-- | :-- | :--",
        ]
        for row in press_detail["p3"]:
            press_extra.append(f"{row['score']} | {row['auroc']:.3f} [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}] | {row['pr_auc']:.3f} [{row['pr_auc_ci_low']:.3f}, {row['pr_auc_ci_high']:.3f}]")
        press_extra += [
            "",
            f"K3-b는 **{k3.get('K3-b', '판정불가')}**: 세 채널 교체 주입 중 관계 잔차의 최고 AUROC는 {operands['best_shuffle_relation_auroc']:.3f}이다. "
            + (f"같은 점수는 전 채널 +2σ 세션 이동에 AUROC {p4_offset['auroc']:.3f}을 보였다. " if p4_offset else "")
            + "실제 이상 분리 성능과 합성 관계 붕괴 구분 성능을 분리해서 해석해야 한다.",
            "",
            "P4 합성 주입 | 레벨 AUROC (95% CI) | 관계 AUROC (95% CI)",
            ":-- | :-- | :--",
        ]
        for injection in dict.fromkeys(row["injection"] for row in press_detail["p4"]):
            level = next(row for row in press_detail["p4"] if row["injection"] == injection and row["score"] == "level")
            relation = next(row for row in press_detail["p4"] if row["injection"] == injection and row["score"] == "relation")
            press_extra.append(f"{injection} | {level['auroc']:.3f} [{level['ci_low']:.3f}, {level['ci_high']:.3f}] | {relation['auroc']:.3f} [{relation['ci_low']:.3f}, {relation['ci_high']:.3f}]")
        press_extra += [
            "",
            "P5는 정상 시험 candidate segment를 관측한 시간만 분모로 삼는다. "
            + (f"α=1%의 1/1 규칙은 경보 사건 {p5_ref['false_alarm_events']}건, 시험 운전 약 {observed_minutes:.1f}분에 "
               f"시간당 {p5_ref['alarms_per_operating_hour']:.1f}건이었다. " if observed_minutes is not None else "")
            + "짧은 관측에서 환산한 수치는 장시간 현장 오경보율이나 고장 전 조기탐지 시간이 아니다.",
            "",
            "α | k/n | 정상 경보 사건/시간 (95% CI) | 채터링 전환 | 이상 구간 탐지율 (95% CI) | 첫 경보 지연",
            ":-- | :-- | :-- | --: | :-- | :--",
        ]
        for row in press_detail["p5"]:
            rate_ci = row["alarms_per_hour_ci95"]
            detect_ci = row["detection_ci95"]
            delay = (f"{row['delay_segments_from_abnormal_start']}구간 / {row['delay_seconds_from_abnormal_start']:.1f}초"
                     if row["delay_seconds_from_abnormal_start"] is not None else "경보 없음")
            press_extra.append(
                f"{row['alpha']:.0%} | {row['rule']} | {row['alarms_per_operating_hour']:.1f} "
                f"[{rate_ci[0]:.1f}, {rate_ci[1]:.1f}] | {row['chattering_transitions']} | "
                f"{row['abnormal_segment_detection_fraction']:.3f} [{detect_ci[0]:.3f}, {detect_ci[1]:.3f}] | {delay}"
            )
        press_extra += ["", "이상 파일은 사실상 한 세션이므로 위 탐지율 CI는 독립 고장 사건의 신뢰구간이 아니다. 해제 시간은 측정할 수 없다.", ""]
    lines = [
        "# 과제 ③·⑤ 사전 타당성 검증",
        "",
        "분석 목적은 각 데이터가 뒷받침하는 주장 범위를 확인하는 것이다. 아래 수치는 이 실행에서 재현한 값만 사용한다. 사전 판단 기준은 수정하지 않은 [00_decision_criteria.md](00_decision_criteria.md)에 있다. 파일별 구조와 출처는 [01_inventory.md](01_inventory.md)에 있다.",
        "",
        "## 1. 요약 표: 관측 증거와 경고 기준",
        "",
        "과제 | 핵심 수치(95% bootstrap CI) | 사전 경고 기준 판정",
        ":-- | :-- | :--",
        f"③ | {t3.get('headline', '결과 없음')} | K3-a {k3.get('K3-a', '판정불가')}; K3-b {k3.get('K3-b', '판정불가')}; K3-c {k3.get('K3-c', '판정불가')}",
        f"⑤ | {t5.get('headline', '결과 없음')} | K5-a {k5.get('K5-a', '판정불가')}; K5-b {k5.get('K5-b', '판정불가')}; K5-c {k5.get('K5-c', '판정불가')}",
        "",
        "`해당`은 사전 경고 조건 충족, `비해당`은 불충족, `판정불가`는 필요한 관측/실험이 없는 경우다. CI는 표본·기간·분할의 한계를 제거하지 않는다.",
        "",
        "## 2. ③ 결과: P0–P5",
        "",
        press_section,
        "",
        *press_extra,
        "## 3. ⑤ 결과: Q0–Q5",
        "",
        power_section,
        "",
        interpretation,
        "",
        "## 4. 평가표별로 이 데이터가 보여줄 수 있는 것",
        "",
        "평가 항목 | ③ | ⑤",
        ":-- | :-- | :--",
        f"데이터 이해·진단 (15) | 진동·전류 구간과 측정일 차이, 센서값 형태를 검증할 수 있다. {press_data_sentence} | 15분 위치별 전력과 계절 범위, 시간 열 오류, 파생변수 관계를 수치로 제시할 수 있다. 전력 단위와 구간 경계는 설명 문서로 확정되지 않는다.",
        "AI 모델·비교 (40) | 정상 학습 이상점수의 여러 표현·모델 비교가 가능하다. 세션과 라벨이 겹치면 고장 일반화 성능은 별도로 입증해야 한다. | 계절 나이브와 LightGBM의 시간순 미래예측, 분위수·초과사건 성능을 비교할 수 있다. 이 기간의 결과가 새로운 연도·공장으로의 일반화를 보장하지 않는다.",
        "영향요인·오류분석 (15) | 레벨·길이 shortcut, 합성 주입, 오경보 조건을 탐색할 수 있다. 합성 샘플은 실제 고장 메커니즘의 대체물이 아니다. | 시간·요일·계절·관측 생산량 조건에서 큰 오차와 피크 미탐·오경보를 정리할 수 있다. 생산량 연관을 조작 효과로 해석할 수 없다.",
        "현장 활용 (10) | 정상 구간 기준 경보 예산과 지속성 규칙을 모의 평가할 수 있다. 실제 경보 해제·정비 효용은 측정하지 못한다. | 피크 위험 경보와 비용비 민감도를 검토할 수 있다. 설비별 조치·실제 기본요금 절감액은 확인되지 않는다.",
        "창의성·차별성 (10) | 관계 붕괴 주입과 경보관리 지표가 데이터에서 살아남는지 검증할 수 있다. 살아남지 않으면 차별성 주장을 철회한다. | 분위수 보정, 확률 경보, cost-loss 가치를 시간순 비교할 수 있다. 상대 경제가치가 없으면 차별성 근거로 삼지 않는다.",
        f"코드 재현성 (10) | {press_repro} | 원본 CSV 해시·분할·특징 가용성·학습 임계값을 고정해 재실행할 수 있다. 시간 오류 복원은 가정으로 기록한다.",
        "",
        "## 5. 가정과 미해결 질문",
        "",
        "### 가정",
        "",
        "- ⑤의 `15분`·`30분`·`45분`·`60분`을 한 시간 안의 순서 있는 15분 측정으로 전개한다. 네 값의 측정 구간 시작/끝과 단위는 아직 공식 확인이 없다.",
        "- K5-a의 하계·동계 1회 포함은 각각 7~9월, 12~2월의 완전한 연속 창이 있는지로 해석한다. 이 자료의 9월은 중순까지만 있고 12월은 없다. 일부 계절 월이 있다는 사실도 Q0에 별도 표기한다.",
        "- ⑤의 0–23 밖 `시간` 48행은 날짜별 24행 순서가 유지됐다는 가정 아래 임시 복원한다. 복원 여부를 결과에 표시해야 하며 원본은 수정하지 않는다.",
        "- 과제 설명의 지정 예측구간이 별도 문서에 명시되지 않아 1시간 후 15분값과 익일 일간 최대 15분값을 실험 대상으로 둔다. 실제 운영에서 미래 생산량·실측 기상은 알 수 없다고 가정한다.",
        "- ⑤ 상위 5% 초과는 학습 구간의 통계적 사건이며 실제 계약전력 초과가 아니다. C/L 및 요금 민감도는 단위 없는 가상 비용비다.",
        f"- {press_assumption}",
        "",
        "### 확인이 필요한 질문",
        "",
        f"1. {press_question}",
        "2. ⑤ 네 전력 열의 단위와 15분 구간 경계, 시간 오류 48행의 원래 시각은 무엇인가?",
        "3. 예측 대상인 ‘지정된 시간구간’과 공식 평가 지표·테스트 구성은 무엇인가?",
        "4. 생산량은 사전 계획인가 사후 실적인가? 설비별 가동·계약전력·요금표가 따로 있는가?",
        "",
        "### 사후 의견: 사전 기준을 바꾸지 않음",
        "",
        "K5-c는 15분 초과 행 수를 기준으로 한다. 인접한 초과 행은 하나의 피크 사건일 수 있으므로 행 수가 30 이상이어도 독립 사건 30건이라고 해석하지 않는다. 이 의견은 K5-c 판정을 변경하지 않는다. K5-a는 완전한 7~9월·12~2월 창이 모두 없어 `해당`으로 판정했지만, 일부 하계·동계 월이 있다는 사실도 별도로 기록한다.",
        "",
        "## 6. 잠정 권고 (증거와 분리)",
        "",
        recommendation,
        "",
    ]
    output = RESULTS / "report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(make_report())
