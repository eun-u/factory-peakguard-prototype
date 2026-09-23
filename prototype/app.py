"""Streamlit view of the historical, review-only power forecasting prototype."""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from prototype.forecast import HORIZONS, MIN_HISTORY_DAYS, MODEL_NAMES, decide, load_series, run_snapshot


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "task05_power" / "source.zip"
MODEL_LABELS = {
    "HistGradientBoosting": "간단한 트리 모델",
    "Persistence": "현재값 유지",
    "Same time last week": "지난주 같은 시각",
}
STATUS_LABELS = {
    "NORMAL": "정상 관찰",
    "PEAK_WATCH": "피크 주의",
    "LOAD_SHIFT_REVIEW": "부하 이동 검토",
    "REVIEW_REQUIRED": "운영자 확인 필요",
}


@st.cache_data(show_spinner=False)
def get_series() -> pd.DataFrame:
    return load_series(DATA_PATH)


@st.cache_data(show_spinner=False, max_entries=12)
def get_snapshot(cutoff: pd.Timestamp) -> dict:
    return run_snapshot(get_series(), cutoff)


def power_chart(snapshot: dict, model_name: str) -> go.Figure:
    cutoff = snapshot["cutoff"]
    current = snapshot["current"]
    forecasts = snapshot["results"][model_name]
    future_times = [cutoff + timedelta(minutes=15 * horizon) for horizon in HORIZONS]
    x_band = [cutoff, *future_times]
    y_lower = [current, *(forecasts[h]["lower"] for h in HORIZONS)]
    y_upper = [current, *(forecasts[h]["upper"] for h in HORIZONS)]
    y_predict = [current, *(forecasts[h]["prediction"] for h in HORIZONS)]

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.06,
        subplot_titles=("지난 24시간", "향후 60분"),
    )
    figure.add_trace(
        go.Scatter(
            x=snapshot["history"].index,
            y=snapshot["history"]["power"],
            mode="lines",
            name="최근 24시간 실제 전력",
            line=dict(color="#274b66", width=2),
            hovertemplate="%{x|%m/%d %H:%M}<br>실제 %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=x_band, y=y_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=x_band,
            y=y_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(214, 111, 70, 0.20)",
            name="검증오차 기반 예측 범위",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=x_band,
            y=y_predict,
            mode="lines+markers",
            name="미래 예측",
            line=dict(color="#d66f46", width=3),
            marker=dict(size=8),
            hovertemplate="%{x|%m/%d %H:%M}<br>예측 %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_hline(y=snapshot["threshold"], line_color="#aa3c3c", line_dash="dash", row=1, col=1)
    figure.add_hline(
        y=snapshot["threshold"], line_color="#aa3c3c", line_dash="dash", row=1, col=2,
        annotation_text="임시 피크 기준", annotation_position="top right",
    )
    figure.update_layout(
        height=430,
        margin=dict(l=10, r=18, t=72, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.22, x=0),
        xaxis=dict(title=None, showgrid=False, tickformat="%m/%d %H:%M"),
        xaxis2=dict(title=None, showgrid=False, tickformat="%H:%M"),
        yaxis=dict(title="전력 원자료 값 (단위 미확인)", gridcolor="#e7ebef", rangemode="tozero"),
        hovermode="closest",
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="전력 피크 판단 흐름", page_icon="⚡", layout="wide")
    st.title("전력 피크 판단 흐름")
    st.caption("과거 전력 → 15·30·60분 예측 → 임시 피크 위험 → 예측 불확실성 → 운영 판정")

    if not DATA_PATH.exists():
        st.error("data/raw/task05_power/source.zip이 없습니다. README의 데이터 배치 방법을 확인하세요.")
        st.stop()
    try:
        series = get_series()
    except Exception as exc:
        st.error(f"데이터를 읽지 못했습니다: {exc}")
        st.stop()

    minimum = series.index[0] + timedelta(days=MIN_HISTORY_DAYS)
    available = series.loc[minimum:].loc[lambda frame: frame["power"].notna()].index
    date_col, time_col, model_col = st.columns([1, 1, 1.4])
    with date_col:
        chosen_date = st.date_input(
            "과거 시점 · 날짜",
            value=available[-1].date(),
            min_value=available[0].date(),
            max_value=available[-1].date(),
        )
    day_times = [time for time in available if time.date() == chosen_date]
    with time_col:
        cutoff = st.selectbox(
            "과거 시점 · 시간",
            day_times,
            index=len(day_times) - 1,
            format_func=lambda value: value.strftime("%H:%M"),
        )
    with model_col:
        model_name = st.selectbox(
            "예측 방법",
            MODEL_NAMES,
            index=2,
            format_func=lambda value: MODEL_LABELS[value],
        )

    try:
        with st.spinner("선택 시점 이전 데이터로 학습·검증 중..."):
            snapshot = get_snapshot(cutoff)
    except Exception as exc:
        st.error(f"예측을 만들지 못했습니다: {exc}")
        st.stop()
    decision = decide(snapshot, model_name)

    top = st.columns([1.25, 1, 1, 1.4])
    top[0].metric("가정한 현재 시각", cutoff.strftime("%Y-%m-%d %H:%M"))
    top[1].metric("현재 전력", f"{snapshot['current']:.1f}")
    top[2].metric("임시 피크 임계값", f"{snapshot['threshold']:.1f}")
    top[3].metric("현재 판정", decision["status"])
    st.caption(
        "전력값의 kW/kWh 단위는 확인되지 않았습니다. 상위 5%는 학습 구간에서만 계산한 "
        "프로토타입용 임시 기준이며 실제 산업 피크 또는 요금 기준이 아닙니다."
    )

    st.subheader("최근 24시간과 향후 60분")
    st.plotly_chart(power_chart(snapshot, model_name), width="stretch")

    st.subheader("시점별 예측과 판단")
    rows = []
    for horizon in HORIZONS:
        item = snapshot["results"][model_name][horizon]
        crosses = item["lower"] <= snapshot["threshold"] <= item["upper"]
        rows.append(
            {
                "시점": f"{horizon * 15}분 후",
                "예상 전력": f"{item['prediction']:.1f}",
                "예측 범위": f"{item['lower']:.1f} ~ {item['upper']:.1f}",
                "피크 위험": "높음" if item["prediction"] >= snapshot["threshold"] else "낮음",
                "불확실성": "높음" if 2 * item["radius"] >= 0.30 * snapshot["threshold"] else "보통",
                "임계선과 범위": "겹침" if crosses else "겹치지 않음",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.info(f"**{decision['status']} · {STATUS_LABELS[decision['status']]}** — " + "; ".join(decision["reasons"]) + ".")
    if decision["status"] == "LOAD_SHIFT_REVIEW":
        st.caption("이동 가능한 작업이나 동시가동 부하를 운영자가 검토하라는 뜻입니다. 설비 이동·자동 제어 명령이 아닙니다.")
    else:
        st.caption("모든 상태는 운영자 판단 지원용입니다. 설비 자동 제어는 하지 않습니다.")

    with st.expander("모델 비교 · 예측 범위 · 데이터 처리 방식"):
        scores = snapshot["metrics"].copy()
        scores["모델"] = scores["model"].map(MODEL_LABELS)
        scores["시점"] = scores["minutes"].astype(str) + "분 후"
        scores["검증 MAE"] = scores["mae"].round(2)
        st.dataframe(scores[["모델", "시점", "검증 MAE", "calibration_rows"]].rename(
            columns={"calibration_rows": "검증 관측 수"}
        ), hide_index=True, width="stretch")
        st.write(
            f"선택 시각 **{cutoff:%Y-%m-%d %H:%M}** 이전만 사용했습니다. "
            f"학습 끝: **{snapshot['train_end']:%Y-%m-%d %H:%M}**. "
            "그 뒤부터 선택 시각까지를 시간순 검증 구간으로 두고, 각 예측 거리별 "
            "절대잔차의 90% 분위수를 예측값 양쪽에 더해 범위를 그렸습니다. "
            "이는 간단한 경험적 범위이며 미래 적중률을 보장하지 않습니다."
        )
        st.write(
            "원본은 시간당 1행이며 15/30/45/60분 값을 HH:15/HH:30/HH:45/다음 시각 HH:00에 배치했습니다. "
            "2021-07-13·15의 시간 이상값 48행은 각 날짜의 24행 순서로 탐색용 시각을 복원하고 표시했습니다. "
            "원본 ZIP은 수정하지 않았고, 미래 생산량·기상 실측값·같은 시간의 평균값은 입력하지 않았습니다."
        )
        st.write(
            "피크 위험은 예측점이 임계값 이상인지로 표시합니다. 범위가 임계선에 걸치는지는 별도로 보여줍니다. "
            "판정은 입력 품질 또는 전체 예측구간 폭 30% 이상이면 REVIEW_REQUIRED, "
            "그 외 현재·예측값이 임계값보다 5% 이상 높으면 LOAD_SHIFT_REVIEW, "
            "임계값 이상이면 PEAK_WATCH, 나머지는 NORMAL입니다."
        )


if __name__ == "__main__":
    main()
