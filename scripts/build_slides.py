"""Build an editable, offline development-CV presentation from saved evidence.

Only development outputs are read. The final holdout is intentionally absent.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def num(value: object, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def signed(value: object, digits: int = 1) -> str:
    return f"{float(value):+,.{digits}f}"


def horizon_label(horizon: int) -> str:
    return {1: "15분", 4: "1시간", 16: "4시간", 96: "24시간"}.get(horizon, f"{horizon*15}분")


def table(headers: list[str], rows: list[list[object]], *, cls: str = "") -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(item)}</td>" for item in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def bar_chart(labels: list[str], values: list[float], *, maximum: float | None = None,
              colors: list[str] | None = None, suffix: str = "") -> str:
    """Native SVG data chart, deliberately editable in the HTML source."""
    maximum = maximum or max(values) * 1.12 or 1
    colors = colors or ["#3E6283"] * len(values)
    lines = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 28 + i * 84
        width = max(0, min(520, 520 * float(value) / maximum))
        lines.append(f'<text x="0" y="{y+26}" class="bar-label">{esc(label)}</text>'
                     f'<rect x="240" y="{y}" width="520" height="38" fill="#E1E5E9"/>'
                     f'<rect x="240" y="{y}" width="{width:.1f}" height="38" fill="{colors[i % len(colors)]}"/>'
                     f'<text x="{min(790, 252+width):.1f}" y="{y+27}" class="bar-number">{num(value, 2)}{esc(suffix)}</text>')
    return f'<svg class="data-chart" viewBox="0 0 890 {max(110, 28+len(values)*84)}" role="img" aria-label="{esc(", ".join(labels))}">' + "".join(lines) + "</svg>"


def line_chart(labels: list[str], values: list[float], *, ymax: float, title: str,
               color: str = "#3E6283", format_value=None) -> str:
    left, top, width, height = 62, 32, 500, 240
    pts = [(left + width * i / max(1, len(values)-1), top + height * (1-float(v)/ymax))
           for i, v in enumerate(values)]
    path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    marks = []
    for (x, y), label, value in zip(pts, labels, values):
        shown = format_value(value) if format_value else num(value, 2)
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>'
                     f'<text x="{x:.1f}" y="{y-17:.1f}" text-anchor="middle" class="point-label">{esc(shown)}</text>'
                     f'<text x="{x:.1f}" y="{top+height+39}" text-anchor="middle" class="axis-label">{esc(label)}</text>')
    return (f'<figure class="chart"><figcaption>{esc(title)}</figcaption>'
            f'<svg class="data-chart" viewBox="0 0 630 350" role="img" aria-label="{esc(title)}">'
            f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#AAB7C3" stroke-width="2"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"/>' + "".join(marks) + "</svg></figure>")


def slide(index: int, title: str, body: str, *, takeaway: str = "") -> str:
    takeaway_html = f'<p class="takeaway">{esc(takeaway)}</p>' if takeaway else ""
    return (f'<section class="slide" data-title="{esc(title)}" id="slide-{index}">'
            f'<header><h2>{esc(title)}</h2><span class="page">{index:02d} / 14</span></header>'
            f'<div class="slide-body">{body}{takeaway_html}</div>'
            f'<footer>개발 교차검증 결과 · 최종 테스트 미평가 · 2026-09-24</footer></section>')


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / "outputs" / "tables" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required development evidence is missing: {path}")
    return pd.read_csv(path)


def build_deck(root: str | Path, outdir: str | Path | None = None) -> dict[str, str]:
    root = Path(root).resolve()
    out = Path(outdir) if outdir is not None else root / "slides"
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    quality = json.loads((root / "outputs/logs/data_quality.json").read_text(encoding="utf-8"))
    choice = json.loads((root / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))["selection"]
    fva = _read(root, "T2-1_fva")
    coverage = _read(root, "coverage_comparison")
    curve = _read(root, "horizon_curve")
    evening = _read(root, "evening_episode_details")
    peak_type = _read(root, "peak_type_summary")
    rev = _read(root, "relative_economic_value")
    tariff = _read(root, "tariff_counterfactual_summary")
    shift = _read(root, "shift_summary")
    selection = choice["by_horizon"]
    primary = selection["4"]
    final_name = primary["point_model"]
    main_fva = fva.set_index("stage")
    final_row = fva.loc[fva.stage.eq("최종 점예측")].iloc[0]
    cbl_row = fva.loc[fva.stage.eq("CBL 대표")].iloc[0]
    persistence_row = fva.loc[fva.stage.eq("Persistence 대표")].iloc[0]
    ci_cbl = primary["vs_cbl_peak_mae"]["ci95"]
    ci_persist = primary["vs_persistence_peak_mae"]["ci95"]
    cov95 = coverage.loc[coverage.alpha.eq(.95)]
    top_cov = cov95.loc[cov95.segment.eq("상위 예측")].set_index("model")
    selected_curve = curve.loc[curve.selected].sort_values("horizon")
    fn = evening.loc[evening.status.eq("FN")]
    fn_assignment = int(fn.missed_reason.eq("one_to_one_assignment").sum())
    fn_no_overlap = int(fn.missed_reason.eq("no_overlap").sum())
    excess = peak_type.loc[peak_type.type.eq("excess_residual")].iloc[0]
    production = peak_type.loc[peak_type.type.eq("production_explained")].iloc[0]
    r10 = rev.loc[rev.cl_ratio.eq(.1)].iloc[0]
    shift20 = shift.loc[shift.fraction.eq(.2)]

    slides: list[str] = []
    notes: list[tuple[str, str]] = []
    def add(title: str, body: str, takeaway: str, note: str):
        slides.append(slide(len(slides)+1, title, body, takeaway=takeaway))
        notes.append((title, note))

    add("1시간 앞의 15분 피크를 예측합니다",
        '<div class="hero-metric"><strong>60분</strong><span>주 과제 T1의 예측거리</span></div>'
        '<p class="hero-copy">15분 전력 관측을 이용해 피크 위치, 초과 확률, 보정된 상단을 계산합니다. '
        '비교 기준은 현업 CBL 준용과 최근값 지속 모델입니다.</p>',
        "지금 보이는 수치는 개발 교차검증입니다. 최종 테스트는 10월 1일 18시 이후 한 번 평가합니다.",
        "약 35초. 발표에서는 지정 시간이 확인되지 않아 1시간 후 15분을 주 과제로 둔 가정 A1을 먼저 밝힌다. 비용은 단위 미확인으로 상대 비교다.")

    add("원자료 6,168시간을 15분 24,672개로 전개",
        '<div class="metric-row">'
        f'<div><strong>{int(quality["source_rows"]):,}</strong><span>시간 원자료 행</span></div>'
        f'<div><strong>{int(quality["rows_15min"]):,}</strong><span>15분 구간</span></div>'
        f'<div><strong>{int(quality["time_repaired_hour_rows"])}</strong><span>시간 복원 행</span></div>'
        f'<div><strong>{int(quality["zero_power"])}</strong><span>전력 0 구간</span></div></div>'
        '<p>시간 오류 원행과 이에 의존하는 특징·목표는 학습과 평가에서 제외했습니다. 0값은 삭제하지 않고 비가동 단서로 남겼습니다.</p>',
        "‘같은 시간 평균’은 네 15분 값의 평균과 일치하지만, kW 단위 확정 증거는 아닙니다.",
        "약 35초. 원본 해시, 48행 복원, 192개 15분 플래그를 언급한다. 전력 단위와 15분 경계는 주최측 미확인이다.")

    add("예측 시점 이후의 정보는 입력하지 않았습니다",
        '<div class="split"><div class="split-dev">개발 85%<small>확장 학습 + 3개 검증 폴드</small></div>'
        '<div class="split-test">테스트 15%<small>동결 후 1회</small></div></div>'
        '<ul class="large-list"><li>전력은 원점까지의 과거 구간만 사용</li>'
        '<li>생산량은 완료된 시간의 합계만 사용</li>'
        '<li>동시각 실측 기상·인원·목표와 겹친 평균 전력은 제외</li></ul>',
        "CBL 참고일·보정창, 피크 임계값, 보정량에도 시점 제약을 적용했습니다.",
        "약 45초. 과거 사전 검증에서 고정 설정으로 테스트를 한 번 본 사실을 공개한다. 이번 모델 선택은 개발구간에서만 했다.")

    selected_rows = []
    for horizon in (1, 4, 16, 96):
        c = selection[str(horizon)]
        names = {"p1_latest": "최근값", "lgbm_no_holiday_weight_2": "LightGBM 피크가중 2",
                 "c3_holiday_hybrid": "휴일혼합 CBL"}
        selected_rows.append([horizon_label(horizon), names.get(c["point_model"], c["point_model"]),
                              "B 구간별" if c["conformal"] == "b" else "A 전역"])
    add("예측거리마다 같은 모델을 강요하지 않았습니다",
        table(["예측거리", "개발 CV 선정 점예측", "위험 상단 보정"], selected_rows),
        "15분은 최근값, 1시간은 피크가중 LightGBM, 4·24시간은 CBL이 선정됐습니다.",
        "약 45초. 복잡한 모델이 항상 이기지 않아 거리별로 독립 모델을 선정했다. T2 익일 최대는 별도 보조과제다.")

    values = [float(cbl_row.peak_mae), float(persistence_row.peak_mae), float(final_row.peak_mae)]
    cbl_significant = float(ci_cbl[0]) > 0
    add("1시간 피크 MAE: 지속모델보다 개선" + (", CBL도 개선" if cbl_significant else ", CBL 우위는 불확실"),
        bar_chart(["CBL 준용", "최근값 지속", "최종 선정"], values,
                  colors=["#7D8894", "#B8741A", "#3E6283"]) +
        f'<p>CBL 대비 개선 {signed(primary["vs_cbl_peak_mae"]["estimate"],2)} '
        f'(95% CI {signed(ci_cbl[0],2)} ~ {signed(ci_cbl[1],2)}). '
        f'지속모델 대비 개선 {signed(primary["vs_persistence_peak_mae"]["estimate"],2)} '
        f'(95% CI {signed(ci_persist[0],2)} ~ {signed(ci_persist[1],2)}).</p>',
        ("CBL 대비 개선 CI 하한이 0보다 큽니다." if cbl_significant else
         "CBL 대비 신뢰구간에 0이 포함됩니다. 필수 성공 기준은 개발 CV에서 아직 입증되지 않았습니다."),
        "약 55초. MAE는 실제 피크 위치만 계산했고 단위는 원자료 단위다. 개선 CI는 동일 날짜 블록 쌍 재표집이다.")

    fva_rows = []
    for stage in ("CBL 대표", "Persistence 대표", "LightGBM 기본", "공휴일 특징 후보",
                  "피크 가중 2 후보", "피크 가중 4 후보", "최종 점예측"):
        row = main_fva.loc[stage]
        status = ("비교 기준" if stage in ("CBL 대표", "Persistence 대표", "LightGBM 기본") else
                  "최종 선정" if stage == "최종 점예측" else
                  "기준 통과" if bool(row.adopted) else "기각")
        fva_rows.append([stage, num(row.peak_mae,2), status])
    add("기법의 부가가치는 단계별로 검증했습니다",
        table(["단계", "피크 MAE", "단계 상태"], fva_rows, cls="compact"),
        "공휴일 특징은 기각, 피크가중 후보는 기준 통과. 최종 선정은 별도 CI 순위 규칙을 따릅니다.",
        "약 45초. 채택은 선택 기법의 사전 기준 통과를 뜻한다. 최종 선정과 같지 않을 수 있다. A/B 보정은 다음 장에서 커버리지로 비교한다.")

    qlabels = {"lgbm_quantile_raw": "보정 전", "lgbm_quantile_a": "전역 A", "lgbm_quantile_b": "구간별 B"}
    qrows = [[qlabels[k], pct(top_cov.loc[k,"coverage"]), int(top_cov.loc[k,"scoring_n"])]
             for k in qlabels]
    b95 = float(top_cov.loc["lgbm_quantile_b", "coverage"])
    a95 = float(top_cov.loc["lgbm_quantile_a", "coverage"])
    add(f"상위 예측구간 q95 실제 커버리지 {pct(b95)}",
        table(["상단 예측", "상위 구간 커버리지", "표본"], qrows) +
        '<div class="target-line">목표 커버리지 95.0% · 구간별 B의 부족 폭 '
        + f'{100*(.95-b95):.1f}%p</div>',
        f"전역 A {pct(a95)}에서 구간별 B {pct(b95)}로 올랐지만 목표에는 {100*(.95-b95):.1f}%p 못 미칩니다.",
        f"약 50초. 상위 구간은 예측 중앙값 상위 10%로, {int(top_cov.loc['lgbm_quantile_b','scoring_n'])}개다. 교차검증의 경험적 커버리지이며 분포무관 보장은 주장하지 않는다.")

    hs = [horizon_label(int(h)) for h in selected_curve.horizon]
    add("멀리 볼수록 피크 MAE가 커집니다",
        '<div class="two-charts">'
        + line_chart(hs, selected_curve.peak_mae.tolist(), ymax=23, title="선정 모델 피크 MAE")
        + line_chart(hs, selected_curve.episode_f1.tolist(), ymax=.75, title="피크 에피소드 F1", color="#2B7564")
        + '</div>',
        "선정 모델은 15분 최근값, 1시간 LightGBM, 4·24시간 CBL입니다.",
        "약 45초. 장거리에서도 성능이 유지된다는 일반화 주장은 하지 않는다. 24시간 직접모델과 T2 익일 최대는 다른 과제다.")

    add(f"저녁 미탐 {len(fn)}건 중 {fn_assignment}건은 일대일 매칭의 결과",
        '<div class="metric-row two">'
        f'<div><strong>{fn_assignment}</strong><span>겹친 경보는 있으나 다른 피크에 배정</span></div>'
        f'<div><strong>{fn_no_overlap}</strong><span>실제 경보 에피소드와 겹침 없음</span></div></div>'
        f'<p>16~24시 실제 피크 에피소드 기준입니다. 겹침이 없는 {fn_no_overlap}건과 일대일 배정으로 미탐 처리된 {fn_assignment}건을 분리했습니다.</p>',
        "위치별 재현율과 에피소드 일대일 매칭 결과는 서로 다른 질문에 답합니다.",
        f"약 55초. FN {fn_assignment}건을 경보 자체가 없던 사례로 말하지 않는다. {fn_no_overlap}건만 no-overlap이다. 저녁에 취약하다는 사전 검증 문장도 개발 CV 결과로 재평가한다.")

    add(f"피크 {int(excess.episodes+production.episodes)}건 중 {int(excess.episodes)}건은 기준선 초과 잔차형",
        '<div class="metric-row two">'
        f'<div><strong>{int(excess.episodes)}</strong><span>기준선 ≤ 임계값, 실제 초과</span></div>'
        f'<div><strong>{int(production.episodes)}</strong><span>기준선도 임계값 초과</span></div></div>'
        f'<p>잔차형 비율 {pct(excess.share)} (날짜 블록 95% CI {pct(excess.share_ci_low)} ~ {pct(excess.share_ci_high)}).</p>',
        f"{pct(excess.share)}를 낭비나 설비 이상 비율로 해석하지 않습니다. 동시각 생산·기온을 쓴 사후 연관 분해입니다.",
        "약 50초. 생산량 계수는 인과효과가 아니다. 설비·제품·교대 관측이 없으므로 잔차의 원인을 단정할 수 없다.")

    add("조치 비용 가정에 따라 경보 가치가 달라집니다",
        line_chart([num(x,2) for x in rev.cl_ratio], rev.rev.tolist(), ymax=1.0,
                   title="상대 경제가치 REV · C/L 비율", color="#2B7564") +
        f'<p>C/L=0.10일 때 REV {num(r10.rev,2)} '
        f'(95% CI {num(r10.ci_low,2)} ~ {num(r10.ci_high,2)}).</p>',
        "C/L은 실제 원가가 아니라 0.01~0.50을 훑는 가정입니다.",
        "약 45초. 비교 기준은 기후평균 경보다. 전력 단위·계약단가가 확인되지 않아 원화 절감 주장은 하지 않는다.")

    tariff_rows = []
    for year in (2021, 2026):
        for band, label in (("off_peak","경부하"),("mid_peak","중간부하"),("peak","최대부하")):
            row = tariff.loc[tariff.tariff_year.eq(year)&tariff.band.eq(band)].iloc[0]
            tariff_rows.append([year,label,pct(row.weighted_missed_peak_share),pct(row.weighted_abs_error_share)])
    mid26 = float(tariff.loc[tariff.tariff_year.eq(2026)&tariff.band.eq("mid_peak"),"weighted_missed_peak_share"].iloc[0])
    evening26 = float(tariff.loc[tariff.tariff_year.eq(2026),"evening_weighted_missed_share"].max())
    add("2026 시간대 재분류의 결과는 ‘상대 가중’입니다",
        table(["분류", "시간대", "미탐 초과 가중 비중", "절대오차 가중 비중"], tariff_rows, cls="compact")
        + '<p style="font-size:21px;color:#4F5F70">가중치 1:2:3은 비교 가정입니다. 실제 계약 요금단가는 미확인입니다.</p>',
        f"2026 분류에서 미탐 초과 가중치의 {pct(mid26)}는 중간부하, 18~21시 비중은 {pct(evening26)}입니다.",
        "약 55초. 공식 시간대와 가정한 순서가중 1·2·3을 사용했다. 선택요금 단가가 없으므로 비용 비중이나 개편 효과로 해석하지 않는다. 겨울 시간대는 재확인 필요.")

    applied = int(shift20.applied_source_hours.sum())
    shift_infeasible = applied == 0
    add("같은 날 낮으로의 사후 이동은 실행 불가" if shift_infeasible else "같은 날 부하 이동의 적용 가능 시간",
        f'<div class="hero-metric"><strong>{applied}시간</strong><span>20% 이동 시나리오에서 실제 적용된 원천 시간</span></div>'
        + ('<p>저녁 경보 뒤 이미 지난 11~15시로 생산량을 옮길 수 없습니다. 이 시나리오의 월 최대 변화와 상대 가중 변화는 모두 0입니다.</p>'
           if shift_infeasible else '<p>적용 가능 시간만 이동했습니다. 생산·설비 제약과 인과효과는 검증되지 않았습니다.</p>'),
        ("시뮬레이션의 0은 모델 무효가 아니라 시간 선후 제약을 지킨 결과입니다." if shift_infeasible else
         "가상 이동 결과를 실제 절감 효과로 해석하지 않습니다."),
        "약 40초. 선제 이동은 하루 전 계획이나 다음 날 목적지, 생산 제약 정보가 있어야 다시 설계할 수 있다. 절감액을 말하지 않는다.")

    cbl_statement = ("CBL 대비 개선 CI 하한도 0보다 큼. " if cbl_significant else
                     "CBL 대비 개선의 95% CI 하한은 0 이하. ")
    add("최종 판단은 동결 후 한 번의 테스트가 결정합니다",
        '<div class="closing"><p><strong>현재 확인</strong> 지속모델 대비 1시간 피크 MAE 개선. '
        + f'<strong>CBL 비교</strong> {cbl_statement}'
        '<strong>운영 한계</strong> 단위·요금단가·설비 일정 미확인.</p></div>'
        '<p class="command">재현 명령: python run_all.py</p>'
        '<p>2026-10-01 18:00 KST 이후 동결된 모델로 테스트를 1회 평가합니다. 개발 교차검증 수치와 최종 수치를 구분해 갱신합니다.</p>',
        "개발 CV는 제출 성능이 아닙니다. hwpx·PDF, 설문, 포털 제출은 별도 인계합니다.",
        "약 50초. 성공 기준 미달을 숨기지 않는다. 테스트는 아직 열지 않았다. 보고서와 이 발표자료의 개발 수치는 10월 1일 동결 이후 별도로 검증한다.")

    if len(slides) != 14:
        raise AssertionError(f"Expected exactly 14 slides, got {len(slides)}")
    css = """
    :root{--paper:#EEF1F4;--panel:#F8FAFB;--ink:#1C2A39;--soft:#4F5F70;--line:#C9D2DB;--steel:#3E6283;--warn:#B8741A;--pass:#2B7564}
    *{box-sizing:border-box}html{background:#D7DEE5}body{margin:0;overflow-x:clip;color:var(--ink);font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;font-variant-numeric:tabular-nums}
    .deck{display:flex;flex-direction:column;align-items:center;gap:24px;padding:24px 0}.slide{position:relative;width:1280px;height:720px;overflow:hidden;padding:52px 68px 54px;background:var(--panel);box-shadow:0 4px 20px #15263822;break-after:page}
    .slide header{display:grid;grid-template-columns:minmax(0,1fr) 110px;align-items:start;border-bottom:2px solid var(--steel);padding-bottom:20px;gap:22px}.slide h2{font-size:40px;line-height:1.22;letter-spacing:-.045em;margin:0;min-width:0;overflow-wrap:anywhere}.page{font-size:22px;color:var(--soft);white-space:nowrap;padding-top:7px;text-align:right}
    .slide-body{padding-top:28px;font-size:25px;line-height:1.52}.slide-body p{margin:12px 0 0}.slide footer{position:absolute;left:68px;bottom:23px;color:var(--soft);font-size:17px}.takeaway{position:absolute;left:68px;right:68px;bottom:60px;border-top:1px solid var(--line);padding-top:13px;color:var(--steel);font-size:23px;font-weight:700;line-height:1.35}
    .hero-metric{margin:36px 0 22px}.hero-metric strong{font-size:94px;line-height:1;color:var(--steel);letter-spacing:-.06em}.hero-metric span{display:block;font-size:26px;margin-top:16px;color:var(--soft)}.hero-copy{max-width:940px;font-size:31px;line-height:1.42}
    .metric-row{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:42px 0 24px}.metric-row.two{grid-template-columns:1fr 1fr}.metric-row>div{padding:25px 20px;border-right:1px solid var(--line)}.metric-row>div:last-child{border:0}.metric-row strong{display:block;font-size:62px;color:var(--steel);line-height:1.05}.metric-row span{display:block;margin-top:14px;font-size:23px;color:var(--soft);line-height:1.32}
    table{border-collapse:collapse;width:100%;font-size:25px;line-height:1.3;margin-top:12px}th{text-align:left;color:var(--soft);font-weight:700;background:var(--paper)}td,th{padding:17px 15px;border-bottom:1px solid var(--line)}td:first-child{font-weight:700}.compact{font-size:23px}.compact td,.compact th{padding:11px 14px}.target-line{margin-top:23px;color:var(--warn);font-size:25px;font-weight:700}
    .split{display:flex;height:146px;margin:34px 0}.split>div{padding:20px 24px;font-weight:700;font-size:32px}.split small{display:block;font-weight:400;font-size:22px;margin-top:7px}.split-dev{width:85%;background:#DCE6EF;color:var(--steel)}.split-test{width:15%;background:#F6E7CF;color:var(--warn);font-size:24px!important;padding:16px 18px!important;line-height:1.18}.split-test small{font-size:19px!important;line-height:1.25}.large-list{margin:12px 0;padding-left:34px}.large-list li{margin:9px 0}
    .data-chart{width:100%;height:auto;display:block;font-family:inherit}.bar-label{font-size:25px;fill:var(--ink);font-weight:700}.bar-number{font-size:24px;fill:var(--ink)}.axis-label{font-size:23px;fill:var(--soft)}.point-label{font-size:22px;fill:var(--ink);font-weight:700}.chart{margin:0}.chart figcaption{font-size:27px;font-weight:700;margin-bottom:4px}.slide-body>.chart{max-width:630px}.two-charts{display:grid;grid-template-columns:1fr 1fr;gap:24px}.two-charts .chart .data-chart{width:100%}.closing{font-size:31px;line-height:1.55;max-width:1080px}.closing strong{color:var(--steel)}.command{font-family:Consolas,monospace;background:var(--paper);padding:14px 18px;display:inline-block}
    @page{size:1280px 720px;margin:0}@media print{html,body{background:white}.deck{display:block;padding:0}.slide{margin:0;box-shadow:none;page-break-after:always;break-after:page}.slide:last-child{page-break-after:auto}}
    @media screen and (max-width:1280px){.deck{width:1280px;padding:0}.slide{box-shadow:none}}
    """
    script = """<script>
    function fitDeck(){const scale=Math.min(1,window.innerWidth/1280);document.querySelector('.deck').style.zoom=scale}
    addEventListener('resize',fitDeck);fitDeck();
    </script>"""
    document = ('<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>과제 ⑤ 개발 결과 발표자료</title><style>' + css + '</style></head><body>'
                '<main class="deck">' + "".join(slides) + '</main>' + script + '</body></html>')
    html_path = out / "development_deck.html"
    html_path.write_text(document, encoding="utf-8")
    notes_path = out / "speaker_notes.md"
    notes_path.write_text("# 과제 ⑤ 개발 발표자 메모\n\n예상 발표 약 10분. 모든 지표는 개발 교차검증이며 최종 테스트 결과가 아닙니다.\n\n" +
                          "\n\n".join(f"## {i}. {title}\n\n{note}" for i, (title, note) in enumerate(notes, 1)) + "\n",
                          encoding="utf-8")
    return {"html": str(html_path), "speaker_notes": str(notes_path), "slides": len(slides)}


if __name__ == "__main__":
    print(json.dumps(build_deck(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
