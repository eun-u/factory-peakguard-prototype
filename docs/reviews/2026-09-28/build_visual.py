"""Build the offline, evidence-linked service/domain review visual.

Run from any working directory: python docs/reviews/2026-09-28/build_visual.py
The only output is outputs/reviews/2026-09-28/service_domain_map.html.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/reviews/2026-09-28/service_domain_map.html"


def rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def one(items: list[dict[str, str]], **where: str) -> dict[str, str]:
    matches = [item for item in items if all(item.get(k) == v for k, v in where.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one evidence row for {where}, found {len(matches)}")
    return matches[0]


def checked_metrics() -> dict[str, str]:
    decisions = rows("outputs/analysis_p2/P2_selection_before_after.csv")
    aux = rows("outputs/analysis_p2/P2_auxiliary_metrics.csv")
    comparisons = rows("outputs/analysis_p2/M1_comparisons.csv")
    folds = rows("outputs/analysis_p2/M1_fold_comparison.csv")
    operations = rows("outputs/analysis_p2/operations/A5_summary.csv")
    selected = {item["horizon"]: item for item in decisions}
    expected = {
        "1": "p1_latest",
        "4": "lgbm_no_holiday_weight_2",
        "16": "lgbm_residual_cbl",
        "96": "c3_holiday_hybrid",
    }
    for horizon, model in expected.items():
        item = selected[horizon]
        if item["point_after"] != model or item["risk_after"] != "lgbm_quantile_b":
            raise ValueError(f"Unexpected current selection at h{horizon}")
    before = one(aux, horizon="16", model="c3_holiday_hybrid")
    after = one(aux, horizon="16", model="lgbm_residual_cbl")
    comparison = one(comparisons, hypothesis_id="M1_h16")
    first_fold = one(folds, horizon="16", fold="0")
    scenario = one(
        operations,
        horizon="16",
        alarm_source="original_operational",
        prep_minutes="30",
        scenario="D",
        fraction="0.2",
    )
    metrics = {
        "fp_before": f'{float(before["position_fp"]):,.0f}',
        "fp_after": f'{float(after["position_fp"]):,.0f}',
        "mae_before": f'{float(before["peak_mae"]):.3f}',
        "mae_after": f'{float(after["peak_mae"]):.3f}',
        "ci_low": f'{float(comparison["ci_low"]):.3f}',
        "ci_high": f'{float(comparison["ci_high"]):.3f}',
        "fold_before": f'{float(first_fold["incumbent_peak_mae"]):.3f}',
        "fold_after": f'{float(first_fold["candidate_peak_mae"]):.3f}',
        "scenario_n": scenario["applied_actual_peak_source_hours"],
        "scenario_d": scenario["actual_peak_unique_source_hours"],
    }
    assert metrics == {
        "fp_before": "978",
        "fp_after": "493",
        "mae_before": "15.285",
        "mae_after": "19.778",
        "ci_low": "-12.073",
        "ci_high": "0.407",
        "fold_before": "23.760",
        "fold_after": "68.570",
        "scenario_n": "172",
        "scenario_d": "183",
    }, "Review visual claims against source CSVs before updating the template"
    return metrics


def build() -> None:
    metrics = checked_metrics()
    safe = {key: html.escape(value) for key, value in metrics.items()}
    page = TEMPLATE.format(**safe)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8", newline="\n")
    print(OUT)


TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>전력 피크 서비스 × 피지컬 AI · 중간 진단</title>
<style>
:root{{--paper:#f5f3ed;--card:#fffefa;--ink:#172c30;--muted:#587071;--line:#bac8c2;--teal:#176e68;--mint:#dcece6;--amber:#79440c;--sand:#f4e6ce;--red:#9d4f49;--rose:#f3e1dd;--slate:#edf1ed;--max:1160px}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo","Noto Sans CJK KR",system-ui,sans-serif;line-height:1.62;overflow-x:clip}}a{{color:var(--teal);text-underline-offset:3px}}a:hover{{text-decoration-thickness:2px}}:focus-visible{{outline:3px solid var(--amber);outline-offset:3px}}button{{font:inherit}}.shell{{max-width:var(--max);margin:auto;padding:0 28px}}.topline{{border-bottom:1px solid var(--ink);padding:13px 0;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}.topline .inner{{display:flex;gap:12px;justify-content:space-between;flex-wrap:wrap}}.hero{{padding:62px 0 44px;display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,.7fr);gap:56px;align-items:end}}.eyebrow{{font-size:12px;font-weight:800;letter-spacing:.13em;color:var(--teal);text-transform:uppercase}}h1{{font-family:Georgia,"Noto Serif CJK KR",serif;font-size:clamp(38px,5.3vw,70px);letter-spacing:-.06em;line-height:1.16;margin:14px 0 24px;font-weight:700}}h1 em{{font-style:normal;color:var(--teal)}}.lead{{font-size:18px;max-width:660px;margin:0;color:#354b4d;word-break:keep-all}}.hero-side{{border-top:4px solid var(--teal);border-bottom:1px solid var(--line);padding:18px 0 20px}}.hero-side strong{{display:block;font-size:24px;line-height:1.36;letter-spacing:-.035em;word-break:keep-all}}.hero-side p{{font-size:13px;color:var(--muted);margin-bottom:0}}.nav{{display:flex;gap:8px;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:13px 0;margin-bottom:54px}}.nav a{{font-size:13px;font-weight:700;text-decoration:none;color:var(--ink);padding:7px 12px;border:1px solid transparent;border-radius:2px}}.nav a:hover{{border-color:var(--line);background:var(--card)}}section{{margin:0 0 74px;scroll-margin-top:24px}}.section-head{{display:flex;align-items:baseline;justify-content:space-between;gap:16px;border-top:2px solid var(--ink);padding-top:18px;margin-bottom:26px;flex-wrap:wrap}}.section-head h2{{font-size:clamp(25px,2.7vw,34px);letter-spacing:-.045em;line-height:1.25;margin:0}}.section-head p{{font-size:13px;color:var(--muted);margin:0;max-width:520px}}.kicker{{font-size:11px;font-weight:800;letter-spacing:.12em;color:var(--teal);text-transform:uppercase}}.notice{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:54px}}.notice>div{{background:var(--card);padding:22px 25px}}.notice small{{display:block;color:var(--muted);font-weight:700;margin-bottom:7px}}.notice strong{{display:block;font-size:17px;line-height:1.4;letter-spacing:-.02em}}.notice p{{font-size:12px;color:var(--muted);margin:8px 0 0}}.service-lanes{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.lane{{background:var(--card);border:1px solid var(--line);padding:24px;min-width:0}}.lane .lane-title{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}}.lane h3{{margin:0;font-size:23px;letter-spacing:-.04em}}.lane p{{margin:11px 0 17px;font-size:14px;color:#3f5555}}.lane ul{{list-style:none;margin:0;padding:0;border-top:1px solid var(--line)}}.lane li{{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid #dce2dc;padding:10px 0;font-size:13px}}.lane li strong{{text-align:right;overflow-wrap:anywhere;font-weight:650}}.pill{{display:inline-block;white-space:nowrap;padding:3px 8px;border:1px solid currentColor;font-size:10px;font-weight:800;letter-spacing:.06em}}.pill.done{{color:var(--teal);background:var(--mint)}}.pill.partial{{color:var(--amber);background:var(--sand)}}.pill.future{{color:var(--red);background:var(--rose)}}.callout{{margin-top:18px;padding:17px 21px;border-left:4px solid var(--amber);background:var(--sand);font-size:14px}}.callout strong{{color:#79440c}}.flow{{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;align-items:stretch}}.step{{min-width:0;position:relative;border:1px solid var(--line);background:var(--card);padding:17px 13px 16px;min-height:156px}}.step:not(:last-child):after{{content:"→";position:absolute;right:-12px;top:67px;background:var(--paper);width:16px;text-align:center;color:var(--teal);font-weight:bold;z-index:1}}.step.future{{border-style:dashed;border-color:var(--red);background:#fffaf8}}.step.partial{{border-color:#c9a472;background:#fffcf6}}.step b{{display:block;font-size:17px;line-height:1.25;letter-spacing:-.04em;margin:11px 0 8px}}.step p{{font-size:11px;color:var(--muted);line-height:1.45;margin:0}}.flow-note{{display:flex;gap:10px;align-items:start;margin-top:15px;color:var(--muted);font-size:12px}}.flow-note:before{{content:"↳";color:var(--teal);font-size:20px;line-height:1}}.legend{{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:20px}}.legend span{{display:inline-flex;align-items:center;gap:6px}}.legend i{{width:13px;height:13px;display:inline-block;border:1px solid var(--teal);background:var(--mint)}}.legend .p i{{border-color:var(--amber);background:var(--sand)}}.legend .f i{{border-style:dashed;border-color:var(--red);background:var(--rose)}}.metric-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.metric-card{{background:var(--card);border:1px solid var(--line);padding:23px 24px}}.metric-card h3{{margin:0 0 4px;font-size:19px;letter-spacing:-.035em}}.metric-card p{{font-size:12px;color:var(--muted);margin:0 0 22px}}.pair{{display:grid;grid-template-columns:58px 1fr 70px;align-items:center;gap:11px;margin:13px 0;font-size:12px}}.pair .value{{font-variant-numeric:tabular-nums;text-align:right;font-size:17px;font-weight:800}}.bar-track{{height:17px;background:#e7ece8;position:relative}}.bar{{height:100%;display:block;min-width:2px}}.bar.before{{background:#a7b5ae}}.bar.after.good{{background:var(--teal)}}.bar.after.bad{{background:var(--red)}}.metric-conclusion{{padding-top:13px;margin-top:20px;border-top:1px solid var(--line);font-size:13px;font-weight:650}}.caveat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin-top:17px}}.caveat{{background:var(--slate);padding:17px 18px;border-left:3px solid var(--amber);font-size:13px}}.caveat b{{display:block;margin-bottom:6px}}.caveat p{{margin:0;color:var(--muted)}}.table-wrap{{overflow-x:auto;border:1px solid var(--line);background:var(--card)}}table{{width:100%;border-collapse:collapse;text-align:left;font-size:13px}}th{{background:#e5ece7;color:#375350;font-size:11px;letter-spacing:.04em;text-transform:uppercase}}th,td{{padding:14px 16px;border-bottom:1px solid #dce3dd;vertical-align:top}}tbody tr:last-child td{{border-bottom:0}}tbody tr:nth-child(even){{background:#fbfbf7}}td code{{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}.tiny{{font-size:12px;color:var(--muted)}}.matrix td:nth-child(1){{font-weight:700;width:22%}}.matrix td:nth-child(2){{width:15%}}.matrix td:nth-child(3){{width:35%}}.horizon td:first-child{{white-space:nowrap;font-weight:800;font-size:17px}}.horizon td:nth-child(3){{color:var(--teal);font-weight:700}}.priority-head{{display:flex;justify-content:space-between;align-items:center;gap:15px;flex-wrap:wrap;margin:0 0 18px}}.filters{{display:flex;gap:7px;flex-wrap:wrap}}.filters button{{min-width:56px;min-height:42px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 13px;cursor:pointer;font-weight:700}}.filters button[aria-pressed="true"]{{background:var(--ink);color:white;border-color:var(--ink)}}.priority-list{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.priority{{min-width:0;border:1px solid var(--line);background:var(--card);padding:20px 22px}}.priority[hidden]{{display:none}}.priority header{{display:flex;gap:9px;align-items:center;flex-wrap:wrap}}.priority h3{{font-size:18px;letter-spacing:-.035em;margin:12px 0 8px}}.priority p{{font-size:13px;color:#3f5555;margin:0 0 12px}}.gate{{padding:11px 12px;background:var(--slate);font-size:12px;color:#375350}}.gate b{{color:var(--teal)}}.book-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.book{{background:var(--card);border:1px solid var(--line);padding:17px 19px;min-width:0}}.book small{{color:var(--teal);font-weight:800}}.book h3{{font-size:16px;line-height:1.4;margin:7px 0 8px;letter-spacing:-.025em}}.book p{{margin:0;font-size:12px;color:var(--muted)}}details{{border-top:1px solid var(--line);padding:12px 0}}details:last-child{{border-bottom:1px solid var(--line)}}summary{{cursor:pointer;font-weight:700;font-size:14px}}details p,details ul{{font-size:13px;color:var(--muted);margin:8px 0 0}}details li{{margin-bottom:5px}}footer{{border-top:2px solid var(--ink);padding:28px 0 60px;font-size:12px;color:var(--muted)}}footer .footer-grid{{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap}}.print-link{{font-weight:700}}@media(max-width:980px){{.hero{{grid-template-columns:1fr;gap:30px;padding-top:46px}}.hero-side{{max-width:590px}}.flow{{grid-template-columns:repeat(4,minmax(0,1fr))}}.step:nth-child(4):after{{display:none}}.book-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:680px){{.shell{{padding:0 17px}}.hero{{padding:38px 0 32px}}h1{{font-size:38px}}.lead{{font-size:16px}}.nav{{margin-bottom:36px}}section{{margin-bottom:54px}}.notice{{grid-template-columns:1fr}}.notice>div{{padding:16px 18px}}.service-lanes,.metric-grid,.priority-list,.book-grid{{grid-template-columns:1fr}}.flow{{grid-template-columns:repeat(2,minmax(0,1fr))}}.step{{min-height:134px}}.step:nth-child(even):after{{display:none}}.step:nth-child(4):after{{display:none}}.caveat-grid{{grid-template-columns:1fr}}.section-head{{display:block}}.section-head p{{margin-top:8px}}th,td{{padding:11px 12px}}.table-wrap{{-webkit-overflow-scrolling:touch}}}}@media(max-width:370px){{.flow{{grid-template-columns:1fr}}.step:after{{display:none}}}}@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}@media print{{:root{{--paper:white;--card:white}}body{{background:white;font-size:11px}}.shell{{max-width:none;padding:0 12mm}}.topline,.nav,.filters,.print-link{{display:none}}.hero{{padding:12mm 0 8mm;grid-template-columns:1.6fr 1fr;gap:12mm}}h1{{font-size:38px}}section{{margin-bottom:11mm;break-inside:avoid}}.flow{{grid-template-columns:repeat(7,minmax(0,1fr))}}.step{{min-height:130px}}.step:nth-child(4):after{{display:block}}.priority[hidden]{{display:block}}details{{break-inside:avoid}}details:not([open])> *:not(summary){{display:block}}a{{color:var(--ink);text-decoration:none}}}}
.nav a{{display:inline-flex;align-items:center;min-height:44px}}
.footer-grid a{{display:inline-flex;align-items:center;min-height:44px}}
.filters button{{min-height:44px}}
summary{{min-height:44px;padding:11px 0}}
details:not([open])> :not(summary){{display:none}}
details a{{display:inline-flex;align-items:center;min-height:44px;vertical-align:middle}}
.lane li{{display:grid;grid-template-columns:52px minmax(0,1fr);align-items:start}}
.lane li span{{white-space:nowrap}}
.scroll-hint{{display:none}}
@media screen and (max-width:980px){{
 .flow{{grid-template-columns:1fr;gap:17px}}
 .step{{min-height:auto;padding:16px 18px}}
 .step:not(:last-child):after,.step:nth-child(4):after{{content:"↓";display:block;right:auto;left:50%;top:auto;bottom:-18px;transform:translateX(-50%);height:20px;line-height:20px;background:var(--paper)}}
}}
@media screen and (max-width:680px){{
 .scroll-hint{{display:block;margin:0 0 8px;color:var(--muted);font-size:12px;font-weight:700}}
 table.matrix{{min-width:720px}}
 table.horizon{{min-width:650px}}
}}
@media print{{.scroll-hint{{display:none}}table.matrix,table.horizon{{min-width:0}}details:not([open])> :not(summary){{display:block}}}}
</style>
</head>
<body>
<div class="topline"><div class="shell inner"><span>Factory Peakguard / service domain review</span><span>2026.09.28 · 개발 증거 기준 · 오프라인 열람</span></div></div>
<main class="shell">
<header class="hero">
  <div><div class="eyebrow">중간 진단 / 산업 전력 의사결정</div><h1>예측에서<br><em>운영 판단</em>까지</h1><p class="lead">현재 서비스는 전력 피크 예측과 위험 평가의 연구 기반을 확보했다. 다음 단계는 관측, 현장 상태, 조치 제약, 운영자 승인, 실행 결과를 한 기록으로 연결하는 것이다.</p></div>
  <aside class="hero-side" aria-label="핵심 판단"><span class="kicker">Review thesis</span><strong>예측 연구 기반 확보<br>운영 의사결정 연결은 다음 단계</strong><p>현재 프로토타입, 개발 평가, 향후 폐루프 구상을 명확히 구분한다.</p></aside>
</header>
<nav class="nav" aria-label="페이지 목차"><a href="#service">서비스 층위</a><a href="#loop">운영 흐름</a><a href="#evidence">판정 근거</a><a href="#capability">역량 지도</a><a href="#priority">우선순위</a><a href="#references">근거</a></nav>
<div class="notice" aria-label="현재 상태 요약">
 <div><small>현재 확인</small><strong>과거 시점 기반 예측·개발 CV 재현</strong><p>전력 단위와 15분 경계 의미는 미확인</p></div>
 <div><small>조건부 해석</small><strong>h16 FP 감소, 피크 오차 개선 미입증</strong><p>선정 근거와 성능의 방향을 분리</p></div>
 <div><small>미완료 게이트</small><strong>최종 홀드아웃·현장 실행 검증</strong><p>날짜 및 사람 승인 대기, 제어 연결 없음</p></div>
</div>
<section id="service"><div class="section-head"><h2>01 · 지금 존재하는 두 층위</h2><p>앱의 시연 기능과 연구 평가 모델은 같은 배포 서비스로 연결되어 있지 않다.</p></div>
 <div class="service-lanes">
  <article class="lane"><div class="lane-title"><h3>로컬 Streamlit 시제품</h3><span class="pill done">현재 구현</span></div><p>사용자가 과거 시각을 선택해 15·30·60분 전력 예측, 임시 피크 위험, 검토 상태를 본다.</p><ul><li><span>점예측</span><strong>현재값·지난주·HistGradientBoostingRegressor</strong></li><li><span>상태</span><strong>NORMAL / PEAK_WATCH / LOAD_SHIFT_REVIEW / REVIEW_REQUIRED</strong></li><li><span>역할</span><strong>운영자 검토용 탐색 시연</strong></li></ul></article>
  <article class="lane"><div class="lane-title"><h3>연구·개발 평가 경로</h3><span class="pill done">개발 검증</span></div><p>15분부터 24시간까지 예측거리별 후보와 q95 위험 출력을 rolling-origin 개발 CV로 평가한다.</p><ul><li><span>선정</span><strong>h1 / h4 / h16 / h96 점모델</strong></li><li><span>위험</span><strong>전 거리 lgbm_quantile_b 유지</strong></li><li><span>역할</span><strong>재현·사전 채택 기준·동결 준비</strong></li></ul></article>
 </div>
 <div class="callout"><strong>현재 경계</strong> · 연구 모델의 선정이 곧 시제품의 모델 교체나 현장 운영 배포를 뜻하지 않는다. README·일부 보고서에는 이전 체크포인트가 남아 있어 최신 개발 상태는 09/25 인계와 검증 기록을 우선 확인한다.</div>
</section>
<section id="loop"><div class="section-head"><h2>02 · 닫힌 운영 흐름의 목표</h2><p>실선은 현재 근거가 있는 기능, 황색은 제한된 연구 시나리오, 점선은 이 검토의 제안이다.</p></div>
 <div class="flow" role="list" aria-label="제안된 전력 운영 판단 흐름">
 <article class="step" role="listitem"><span class="pill done">현재 근거</span><b>관측</b><p>과거 전력·시각·일부 생산 자료</p></article>
 <article class="step future" role="listitem"><span class="pill future">제안</span><b>상태</b><p>설비·작업·변경 가능 구간의 구조화</p></article>
 <article class="step" role="listitem"><span class="pill done">연구 평가</span><b>예측</b><p>거리별 점예측과 q95 위험</p></article>
 <article class="step partial" role="listitem"><span class="pill partial">사후 가정</span><b>조치 후보</b><p>부분 월 부하 이동 시나리오</p></article>
 <article class="step future" role="listitem"><span class="pill future">제안</span><b>제약·승인</b><p>생산·안전·비용 검토와 운영자 승인</p></article>
 <article class="step future" role="listitem"><span class="pill future">제안</span><b>실행 결과</b><p>채택·거부·실행·실패 기록</p></article>
 <article class="step partial" role="listitem"><span class="pill partial">부분 근거</span><b>검수·재평가</b><p>개발 재현 확보; 현장 결과 기반 학습은 제안</p></article>
 </div>
 <div class="flow-note">실행 결과가 관측·검수로 돌아오는 고리는 목표 구조다. 현재 소프트웨어의 실제 자동 제어나 현장 효과를 나타내지 않는다.</div>
 <div class="legend" aria-label="상태 범례"><span><i></i>현재 확인된 구현·평가</span><span class="p"><i></i>부분 근거·사후 가정</span><span class="f"><i></i>제안·미구현</span></div>
</section>
<section id="evidence"><div class="section-head"><h2>03 · h16 결정의 양면</h2><p>같은 개발 OOF에서 나온 수치다. 서로 다른 지표를 하나의 개선 점수로 합치지 않는다.</p></div>
 <div class="metric-grid">
  <article class="metric-card"><h3>점예측 임계값 위치 FP</h3><p>낮을수록 좋음 · 개발 평가의 위치 단위 · q95 운영 경보 건수와 별개</p><div class="pair"><span>기존</span><div class="bar-track"><span class="bar before" style="width:100%"></span></div><span class="value">{fp_before}</span></div><div class="pair"><span>선정</span><div class="bar-track"><span class="bar after good" style="width:50.4%"></span></div><span class="value">{fp_after}</span></div><div class="metric-conclusion">FP 감소가 h16 잔차 모델 선정의 후순위 판정 근거</div></article>
  <article class="metric-card"><h3>피크 위치 MAE</h3><p>낮을수록 좋음 · 전력 단위 미확인</p><div class="pair"><span>기존</span><div class="bar-track"><span class="bar before" style="width:77.3%"></span></div><span class="value">{mae_before}</span></div><div class="pair"><span>선정</span><div class="bar-track"><span class="bar after bad" style="width:100%"></span></div><span class="value">{mae_after}</span></div><div class="metric-conclusion">개선량 95% CI [{ci_low}, {ci_high}] · 평균 개선 미입증</div></article>
 </div>
 <p class="tiny">두 막대는 각각 0을 기준으로 하며 지표마다 축의 최대값이 다르다. FP는 횟수, MAE는 단위 미확인 원자료 척도다. 피크 MAE = Σ|실제값 − 예측값| / 피크 표본 수. 두 표시값의 차이와 CI의 추정값은 원자료 정밀도 때문에 끝자리가 다를 수 있다.</p>
 <div class="caveat-grid"><div class="caveat"><b>초기 폴드의 악화</b><p>피크 MAE {fold_before} → {fold_after}. 후속 폴드와 전체 채택 규칙을 함께 봐야 한다.</p></div><div class="caveat"><b>위험 출력 유지</b><p>q95는 기존 B. h16 운영 조치율 개선을 새 모델 효과로 주장할 수 없다.</p></div><div class="caveat"><b>사후 조치 가능성</b><p>준비 30분·D 20% 가정에서 실제 피크 출발창 {scenario_n}/{scenario_d}. 절감액이나 실행 가능성 검증은 아니다.</p></div></div>
</section>
<section id="capability"><div class="section-head"><h2>04 · 역량과 증거의 위치</h2><p>숫자 성숙도 대신 현재 확인된 범위와 다음에 필요한 증거를 적는다.</p></div>
 <p class="scroll-hint">← 표를 좌우로 스크롤해 전체 열 확인 →</p><div class="table-wrap" role="region" tabindex="0" aria-label="역량과 증거 표, 가로 스크롤 가능"><table class="matrix"><thead><tr><th scope="col">역량</th><th scope="col">상태</th><th scope="col">확인한 범위</th><th scope="col">다음 증거</th></tr></thead><tbody>
 <tr><td>전력 관측·복원</td><td><span class="pill partial">부분</span></td><td>과거 CSV, 15분 전개, 시간 오류 표시</td><td>공식 단위·구간 경계·현장 계측 계약</td></tr>
 <tr><td>예측·위험 추정</td><td><span class="pill done">개발 검증</span></td><td>시간순 CV, 거리별 선정, 불확실성 출력</td><td>최종 홀드아웃 1회와 운영 분포 점검</td></tr>
 <tr><td>상태·제약 모델</td><td><span class="pill future">제안</span></td><td>설비·작업·안전 제약의 실행 가능한 명세 없음</td><td>현장 인터뷰와 상태/제약 데이터 사전</td></tr>
 <tr><td>조치 후보</td><td><span class="pill partial">사후 가정</span></td><td>부분 월·사후 관측 생산량·순서형 요금 시나리오</td><td>사전 입력만 쓰는 리플레이, 안전성·비용 검증</td></tr>
 <tr><td>승인·실행 기록</td><td><span class="pill future">제안</span></td><td>시제품의 검토 문구만 확인</td><td>운영자 승인/거부와 조치 결과 로그</td></tr>
 <tr><td>재현·변경 관리</td><td><span class="pill done">개발 검증</span></td><td>개발 CV 재현, 캐시 봉인, 사전등록</td><td>현장 운영 버전·입력·결정의 추적성</td></tr>
 </tbody></table></div>
 <h3 style="font-size:19px;margin:30px 0 12px">개발 평가에서 선정된 거리별 출력</h3>
 <p class="scroll-hint">← 표를 좌우로 스크롤해 전체 열 확인 →</p><div class="table-wrap" role="region" tabindex="0" aria-label="예측거리별 선정 표, 가로 스크롤 가능"><table class="horizon"><thead><tr><th scope="col">거리</th><th scope="col">예측거리(원점→목표)</th><th scope="col">선정 점모델</th><th scope="col">위험 출력</th></tr></thead><tbody><tr><td>h1</td><td>15분</td><td><code>p1_latest</code></td><td rowspan="4"><code>lgbm_quantile_b</code><br><span class="tiny">전 거리 공통·기존 B 유지</span></td></tr><tr><td>h4</td><td>1시간</td><td><code>lgbm_no_holiday_weight_2</code></td></tr><tr><td>h16</td><td>4시간</td><td><code>lgbm_residual_cbl</code></td></tr><tr><td>h96</td><td>24시간</td><td><code>c3_holiday_hybrid</code></td></tr></tbody></table></div>
</section>
<section id="priority"><div class="section-head"><h2>05 · 다음 설계 순서</h2><p>우선순위와 통과 기준은 이 중간 검토의 제안이다. 기존 연구의 사전등록 기준과 별개다.</p></div>
 <div class="priority-head"><span class="tiny" id="filter-status" aria-live="polite">전체 6개 항목 표시</span><div class="filters" role="group" aria-label="우선순위 필터"><button type="button" data-filter="all" aria-pressed="true">전체</button><button type="button" data-filter="P0" aria-pressed="false">P0</button><button type="button" data-filter="P1" aria-pressed="false">P1</button><button type="button" data-filter="P2" aria-pressed="false">P2</button></div></div>
 <div class="priority-list">
  <article class="priority" data-priority="P0"><header><span class="pill future">P0 · 제안</span><span class="tiny">R05 · 의미 계약</span></header><h3>전력 단위·경계·설비 맥락 확정</h3><p>kW/kWh, 15분 칸의 시간 의미, 계측 누락, 설비와 작업의 식별자를 현장 자료와 대조한다.</p><div class="gate"><b>제안 통과 기준</b> · 데이터 사전과 예외 표본을 현장 담당자가 승인</div></article>
  <article class="priority" data-priority="P0"><header><span class="pill future">P0 · 제안</span><span class="tiny">R01–R04 · 버전 정합</span></header><h3>시연·연구·문서의 버전 연결</h3><p>옛 보고서와 최신 h16 선정을 구분하고 예보 모델·자료 범위·점예측 FP와 피크 MAE를 함께 표시한다.</p><div class="gate"><b>제안 통과 기준</b> · 화면·보고서의 source, model, scope, 수치가 같은 검증 스냅샷을 가리킴</div></article>
  <article class="priority" data-priority="P1"><header><span class="pill future">P1 · 제안</span><span class="tiny">R07 · R09 · 상태/조건</span></header><h3>현장 상태와 조치 제약의 명세</h3><p>작업 교대, 설비 동시가동, 최소 가동·휴식, 변경 금지 구간을 상태화하고 취약·평가 불가 구간을 명시한다.</p><div class="gate"><b>제안 통과 기준</b> · 제약 위반 후보 제외, 빈 표본은 0점 대신 평가 불가로 표시</div></article>
  <article class="priority" data-priority="P1"><header><span class="pill future">P1 · 제안</span><span class="tiny">R06 · R08 · R10 · 결정 계약</span></header><h3>예보·승인·결과 기록 연결</h3><p>예보 원점·만료·운영자 선택·실행 결과를 같은 결정 ID에 연결하고 학습·추론·화면 지연을 구분해 측정한다.</p><div class="gate"><b>제안 통과 기준</b> · 미실행까지 추적하고 각 구간의 지연과 실패율을 별도 보고</div></article>
  <article class="priority" data-priority="P2"><header><span class="pill future">P2 · 제안</span><span class="tiny">R11 · 그림자 평가·효과</span></header><h3>리플레이 뒤 제한된 현장 실증</h3><p>예측 시점에 알 수 있던 입력만으로 후보를 만들고, 공식 계량·요금 정보와 생산·품질 영향을 함께 측정한다.</p><div class="gate"><b>제안 통과 기준</b> · 사후 정보 누수와 제약 위반 없이 사전 정의된 효과·불확실성 보고</div></article>
  <article class="priority" data-priority="P2"><header><span class="pill future">P2 · 제안</span><span class="tiny">R12 · 지속 개선</span></header><h3>실패 사례의 검수·오프라인 재평가</h3><p>경보 거부, 오탐, 조치 불가, 실행 실패를 사건 단위로 묶고 버전별 개선을 검증한다.</p><div class="gate"><b>제안 통과 기준</b> · 변경 전후 성능·안전·운영 비용을 같은 규칙으로 비교</div></article>
 </div>
</section>
<section id="references"><div class="section-head"><h2>06 · 도메인 연결과 출처</h2><p>책의 로봇 개념을 전력 서비스에 연결한 부분은 이 검토의 해석이다. 책에서 본 서비스 성능을 검증한 결과가 아니다.</p></div>
 <div class="book-grid">
  <article class="book"><small>3장 · p40–48</small><h3>계획과 함수 호출</h3><p>조치 제안을 구조화된 호출·승인 요청으로 표현한다는 설계 참고.</p></article>
  <article class="book"><small>7장 · p117–118</small><h3>추론층과 행동층의 분리</h3><p>예측·계획과 현장 실행 사이에 결정 경계를 둔다는 해석.</p></article>
  <article class="book"><small>8장 · p128–135</small><h3>온톨로지와 구조화된 상태</h3><p>설비·작업·제약 관계를 명시적으로 기록하는 설계 참고.</p></article>
  <article class="book"><small>9장 · p144–147</small><h3>사람 행동 기록의 데이터화</h3><p>운영자 승인·거부·수정 사유를 학습 가능한 사건으로 남긴다는 해석.</p></article>
  <article class="book"><small>14장 · p214–229</small><h3>현실과 시뮬레이션의 연결</h3><p>리플레이와 반사실 시나리오를 실제 계측·제약으로 교정한다는 설계 참고.</p></article>
  <article class="book"><small>15장 · p234–241</small><h3>현장 실패에서 개선</h3><p>검수된 실패 사례를 오프라인 재평가와 버전 변경에 연결한다는 해석.</p></article>
 </div>
 <div style="margin-top:30px"><details><summary>서비스 수치와 현재 상태의 근거</summary><ul><li><a href="../../logs/session_0925_summary.md">09/25 개발 인계</a> — 최신 개발 선정, 한계, 최종 홀드아웃 미실행, 1881.419초는 전체 개발 재현 시간이며 온라인 응답 시간이 아님. 30분 목표는 미충족.</li><li><a href="../../analysis_p2/P2_selection_before_after.csv">P2 선정 전후</a>, <a href="../../analysis_p2/P2_auxiliary_metrics.csv">P2 보조 지표</a>, <a href="../../analysis_p2/M1_comparisons.csv">M1 비교</a>, <a href="../../analysis_p2/M1_fold_comparison.csv">M1 폴드</a> — h16 수치.</li><li><a href="../../analysis_p2/operations/A5_summary.csv">A5 사후 시나리오</a> — 172/183. 실제 현장 조치, kW·금전 절감, 인과 효과를 확인한 수치가 아님.</li><li><a href="../../../prototype/README.md">Streamlit 시제품 설명</a> — 앱의 별도 모델과 상태.</li></ul></details><details><summary>책 자료의 범위와 검수 상태</summary><p>책 연결은 <code>C:\Project\피지컬AI시스템설계\knowledge\books\physical_ai_system_design\</code>의 목차·페이지 기록을 참고했다. 159개 캡처 처리 QA의 <code>PASS_WITH_DOCUMENTED_REVIEW</code>는 출처 관리 상태이며 책의 과학적 주장이나 재현 검증을 뜻하지 않는다. OCR은 <code>UNVERIFIED_OCR</code>이고 판권면 하단은 <code>REVIEW_REQUIRED</code>다. 이번 작업에서 모든 이미지 판독을 다시 검수하지 않았다.</p></details><details><summary>검토에서 열어 둔 불일치</summary><p>기존 README·보고서·로드맵의 이전 체크포인트와 최신 P2/P3 개발 인계가 혼재한다. 시제품의 15·30·60분 HistGradientBoostingRegressor 경로와 연구의 h1/h4/h16/h96 선정도 별개다. 최종 홀드아웃은 날짜·사람 승인 전까지 미실행이며, 개발 재현 31분 21초는 온라인 추론 지연으로 해석하지 않는다.</p></details></div>
</section>
</main>
<footer><div class="shell footer-grid"><span>현재 증거와 제안 경계를 보존한 서비스 도메인 중간 시각화 · 2026.09.28</span><span><a href="../../../docs/reviews/2026-09-28/service_review.md">상세 검토 문서</a> · <a class="print-link" href="#" onclick="window.print();return false">인쇄 / PDF</a></span></div></footer>
<script>
(function(){{
 const buttons=Array.from(document.querySelectorAll('[data-filter]'));
 const cards=Array.from(document.querySelectorAll('.priority'));
 const status=document.getElementById('filter-status');
 buttons.forEach(button=>button.addEventListener('click',()=>{{
  const filter=button.dataset.filter;
  buttons.forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  let visible=0;
  cards.forEach(card=>{{card.hidden=filter!=='all'&&card.dataset.priority!==filter;if(!card.hidden)visible++;}});
  status.textContent=(filter==='all'?'전체':filter)+' '+visible+'개 항목 표시';
 }}));
}})();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    build()
