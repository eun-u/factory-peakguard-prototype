"""Feasibility audit for task ② robot welding, without inventing weld labels.

Run: python -X utf8 verification/t2_weld.py
Dependencies are limited to the Python standard library, pandas, numpy, and
matplotlib. The XLSX reader below avoids adding an Excel engine dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
SEED = 42
BOOTSTRAPS = 1000
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XL_NS = {"x": MAIN_NS}


def _cell_column(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    index = 0
    for letter in letters.group():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_xlsx_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Read simple worksheet values with standard-library OOXML parsing.

    This workbook contains values, not formula-dependent fields. Formula-only
    cells without a cached value remain missing, which inventory will expose.
    """
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in shared.findall(f"{{{MAIN_NS}}}si")]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.get("Id"): rel.get("Target")
            for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
        }
        result: dict[str, pd.DataFrame] = {}
        for sheet in workbook.findall("x:sheets/x:sheet", XL_NS):
            target = targets[sheet.get(f"{{{DOC_REL_NS}}}id")]
            member = target.lstrip("/") if target.startswith("/") else "xl/" + target
            member = member.replace("\\", "/")
            root = ET.fromstring(archive.read(member))
            records: list[dict[int, object]] = []
            for row in root.findall("x:sheetData/x:row", XL_NS):
                values: dict[int, object] = {}
                for cell in row.findall("x:c", XL_NS):
                    position = _cell_column(cell.get("r", ""))
                    value_node = cell.find("x:v", XL_NS)
                    value: object = value_node.text if value_node is not None else None
                    kind = cell.get("t")
                    if kind == "s" and value is not None:
                        value = strings[int(value)]
                    elif kind == "inlineStr":
                        value = "".join(cell.find("x:is", XL_NS).itertext())
                    elif kind not in {"str", "e"} and value is not None:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    values[position] = value
                records.append(values)
            if not records:
                result[sheet.get("name", "unnamed")] = pd.DataFrame()
                continue
            width = max(max(row, default=-1) for row in records) + 1
            headers = [str(records[0].get(col) or f"Unnamed: {col + 1}") for col in range(width)]
            frame = pd.DataFrame(
                [[row.get(col) for col in range(width)] for row in records[1:]],
                columns=headers,
            )
            frame = frame.dropna(how="all").reset_index(drop=True)
            result[sheet.get("name", "unnamed")] = frame
        return result


def _table_inventory(df: pd.DataFrame, date_column: str | None = None) -> list[str]:
    lines = [
        f"- 크기: {len(df):,}행 × {len(df.columns):,}열; 완전 중복 행: {df.duplicated().sum():,}개",
        "",
        "| 컬럼 | dtype | 결측 | 고유값 |",
        "| :-- | :-- | --: | --: |",
    ]
    for name in df.columns:
        lines.append(
            f"| {name} | {df[name].dtype} | {df[name].isna().sum():,} | {df[name].nunique(dropna=True):,} |"
        )
    if date_column and date_column in df:
        numeric = pd.to_numeric(df[date_column], errors="coerce")
        dates = pd.to_datetime(numeric, origin="1899-12-30", unit="D", errors="coerce")
        lines.extend(["", (
            f"- `{date_column}` 엑셀 일련번호 해석: {dates.min().date()}~{dates.max().date()}; "
            f"날짜 {dates.nunique()}개; 날짜 파싱 실패 {dates.isna().sum()}개"
        )])
    return lines


def _date_values(frame: pd.DataFrame) -> pd.Series:
    serial = pd.to_numeric(frame["working time"], errors="coerce")
    return pd.to_datetime(serial, origin="1899-12-30", unit="D", errors="coerce").dt.normalize()


def _bootstrap_reported_count_per_date(daily: pd.DataFrame) -> tuple[float, float, float]:
    """Describe reported counts across reporting dates, never individual welds."""
    counts = daily["defect_count"].dropna().to_numpy(dtype=float)
    if len(counts) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = counts.mean()
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(counts, size=(BOOTSTRAPS, len(counts)), replace=True)
    lower, upper = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return float(estimate), float(lower), float(upper)


def _plot_daily(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dates = daily["date"].dt.strftime("%m-%d")
    x = np.arange(len(daily))
    bottom = np.zeros(len(daily))
    colors = ["#446b85", "#d99042", "#a04e4f"]
    for kind, color in zip((1, 2, 3), colors):
        counts = daily[f"type_{kind}"].fillna(0).to_numpy()
        ax.bar(x, counts, bottom=bottom, color=color, label=f"Type {kind}")
        bottom += counts
    for i, row in daily.iterrows():
        if pd.isna(row["defect_count"]):
            ax.text(i, 0.3, "No result row", ha="center", va="bottom", rotation=90, fontsize=8)
        else:
            missing_types = [str(kind) for kind in (1, 2, 3) if pd.isna(row[f"type_{kind}"])]
            if missing_types:
                ax.text(
                    i, row["defect_count"] + 0.15,
                    f"Type {','.join(missing_types)} not reported",
                    ha="center", va="bottom", rotation=90, fontsize=8,
                )
    ax.set_xticks(x, dates, rotation=45)
    ax.set_ylabel("Reported defect count (not weld-level labels)")
    ax.set_xlabel("2020 date")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "02_defect_over_time.png", dpi=160)
    plt.close(fig)


def _plot_process(daily: pd.DataFrame, variables: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    x = np.arange(len(daily))
    for ax, variable in zip(axes.flat, variables):
        ax.plot(x, daily[f"{variable} mean"], marker="o", color="#355c7d")
        ax.set_title(variable)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xticks(x, daily["date"].dt.strftime("%m-%d"), rotation=45)
    fig.suptitle("Daily process means; descriptive only", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "02_process_by_date.png", dpi=160)
    plt.close(fig)


def _write_comparison(counts: dict[str, object], k2_a: str) -> None:
    """Quote the existing ③·⑤ report without recomputing its results."""
    previous = (RESULTS / "report.md").read_text(encoding="utf-8")
    cited_values = (
        "raw Mahalanobis AUROC 0.999", "관계 잔차의 최고 AUROC는 0.500",
        "59.26", "17.77", "147개", "26일",
    )
    missing = [value for value in cited_values if value not in previous]
    if missing:
        raise ValueError(f"Previous report changed; verify quoted ③·⑤ figures: {missing}")
    content = f"""# 과제 ②·③·⑤ 사전 타당성 비교

## 증거 비교

②의 근거는 [02 보고서](02_report.md)와 원자료 구조 진단이다. ③·⑤의 수치와 판정은 기존 [③·⑤ 보고서](report.md)에서 인용했으며 재계산하지 않았다. 서로 다른 예측 목표의 AUROC, F1, MAE를 한 점수로 환산하지 않는다.

| 과제 | 걸린 사전 경고 | 모델 항목의 변별력 여지 | 오류분석 재료 | 현장 활용 결과물 | 식별 가능한 독립 사건 | 주요 데이터 한계 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| ② 로봇용접 | K2-a {k2_a}. K2-b·c·d 판정불가 | 개별 불량 라벨이 없어 F1과 모델 포화 여부를 측정할 수 없다. | 날짜·유형별 불량 건수는 있으나 개별 FN·FP를 계산할 수 없다. | 관측 공정값 범위와 검사 데이터 연결 요구사항은 제시할 수 있다. 낮은 위험의 안정조건은 검증할 수 없다. | 불량 기록이 있는 날짜 {counts['positive_result_dates']}일. 불량 집계 {counts['reported_defect_count']}건을 독립 부품 {counts['reported_defect_count']}건으로 해석하지 않는다. | {counts['raw_rows']:,}개 공정 행과 {counts['result_rows']}개 불량 집계 행을 1:1로 잇는 키가 없다. 두께는 관측상 고정이며 주요 설정 범위가 좁다. |
| ③ 프레스 펌프 | K3-b 해당. K3-a·c 비해당 | raw Mahalanobis AUROC 0.999 [0.996, 1.000]이지만 한 이상 세션 평가다. 관계 붕괴 주입 AUROC 최고 0.500으로 관계 기반 차별성은 뒷받침되지 않는다. | 정상 오경보와 21개 이상 후보 구간은 분석 가능하다. 고장 전 FN·해제시간은 평가할 수 없다. | 정상 학습 경보 임계값·지속성 규칙과 짧은 관측구간의 경보 부담을 보여줄 수 있다. | 이상 측정 세션 사실상 1건. 21개 후보 구간은 독립 고장 21건이 아니다. | 정상·이상 수집일이 다르고 원파형/집계값 여부가 미확인이다. |
| ⑤ 전력 피크 | K5-a 해당. K5-b·c 비해당 | 1시간 후 피크 구간 MAE는 나이브 59.26 [24.65, 91.09], LightGBM 17.77 [13.88, 22.28]로 개선됐다. 익일 최대 MAPE는 LightGBM이 더 높았다. | 테스트 피크 초과 372개 15분 위치, 연속 147개 에피소드·26일. 요일·시간·생산량별 오차와 FN·FP를 볼 수 있다. | 확률 경보와 가상 C/L 의사결정 민감도를 보일 수 있다. 실제 요금 절감이나 설비별 일정 변경 효과는 입증할 수 없다. | 피크 발생일 26일. 372개 위치를 독립 요금 사건으로 해석하지 않는다. | 완전한 7~9월·12~2월 창이 없고 전력 단위·15분 경계·시간 오류 복원 방식이 확정되지 않았다. |

### 비교의 한계

- ②의 검사 집계는 개별 용접 정답이 아니다. 집계 건수를 {counts['raw_rows']:,}개 공정 행에 복제하면 허위 라벨과 누수가 생긴다. 일부 날짜에는 불량 유형 결과 행도 빠져 있다.
- ③의 높은 이상 구분 AUROC와 ⑤의 낮은 예측 MAE는 문제·측정단위가 달라 직접 우열 비교가 불가능하다.
- ②의 불량 {counts['positive_result_dates']}일, ③의 이상 1세션, ⑤의 피크 26일은 서로 같은 사건 정의가 아니다.

## 잠정 권고

현재 제공된 자료만으로는 ②의 필수 산출물인 **개별 용접 불량예측 F1, FN·FP, 안정 공정조건의 테스트 검증**을 만들 수 없다. ②를 선택하려면 생산 기록과 최종 검사 결과를 같은 용접·부품 ID로 연결한 라벨 자료를 먼저 확보해야 한다. ③과 ⑤도 각각 독립 고장 세션과 새 계절의 검증이 부족하므로, 세 과제의 최종 우열은 **판정 보류**한다.
"""
    (RESULTS / "comparison_2_3_5.md").write_text(content, encoding="utf-8")


def main() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "data" / "raw" / "task02_welding"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Task ② raw data directory is missing: {source_dir}")
    source_files = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".xlsx", ".csv", ".pdf", ".hwp", ".hwpx", ".txt", ".md"}
    )
    workbook_files = [p for p in source_files if p.suffix.lower() == ".xlsx"]
    if len(workbook_files) != 1:
        raise ValueError(f"Expected one discovered welding workbook; found {len(workbook_files)}")
    workbook = workbook_files[0]
    sheets = read_xlsx_sheets(workbook)
    required = {"Raw data", "result", "data set"}
    if not required.issubset(sheets):
        raise ValueError(f"Workbook sheets changed; found {list(sheets)}")
    raw, outcomes, dictionary = (sheets["Raw data"], sheets["result"], sheets["data set"])

    csv_files = [p for p in source_files if p.suffix.lower() == ".csv"]
    csv_data: dict[Path, pd.DataFrame] = {}
    for csv_path in csv_files:
        for encoding in ("utf-8-sig", "cp949"):
            try:
                csv_data[csv_path] = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeError:
                continue
        if csv_path not in csv_data:
            raise ValueError(f"Cannot decode {csv_path}")

    # The workbook dictionary is the only supplied variable-definition source.
    inventory = ["# ② 로봇용접 데이터 인벤토리", "", "## 발견한 파일", "", "파일 | 바이트 | SHA-256 앞 12자리", ":-- | --: | :--"]
    for path in source_files:
        inventory.append(
            f"{path.relative_to(ROOT).as_posix()} | {path.stat().st_size:,} | "
            f"{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        )
    for name, frame in sheets.items():
        inventory.extend(["", f"## XLSX 시트: {name}", ""])
        inventory.extend(_table_inventory(frame, "working time" if name != "data set" else None))
    for path, frame in csv_data.items():
        inventory.extend(["", f"## CSV: {path.name}", ""])
        inventory.extend(_table_inventory(frame))
    inventory.extend(["", "## 시트 내 데이터 사전", "", "항목 | 설명 | 문서상 수집 범위", ":-- | :-- | :--"])
    for _, row in dictionary.iterrows():
        if pd.isna(row.get("Data")):
            continue
        inventory.append(
            f"{row.get('Data')} | {row.get('항목 설명', '')} | {row.get('수집 범위', '')}"
        )
    inventory.extend([
        "", "`data set` 시트의 두께는 컬럼명에 mm, 가압력은 bar, 전류는 kA, 전압은 V, 통전시간은 ms가 명시되어 있다.",
        "`working time`은 실제로 날짜 일련번호만 있으며 시각은 없다. 용접속도는 제공되지 않았다.",
        "`result`의 `defect`는 0~3 정수이며 날짜·유형별 보고 건수로 추정된다(데이터 사전에 정의 없음). `defect type`은 1~3 코드다. G열은 제목 없이 첫 세 행에서만 각각 파임불량, 용접부족, 크랙발생을 적는다. 불량 유형 외에 개별 용접 양불 정의나 검사 규격은 없다.",
        "2020-03-31에는 유형 3 결과 행이 없으므로 그날의 유형 3을 0건으로 해석하지 않는다.",
        "", "## 행 단위와 결합 가능성", "",
        "`Raw data`는 날짜별로 `idx`가 다시 시작하고 행마다 용접조건 한 세트가 기록되어 있어 용접 1회의 요약 행으로 **추정**된다. 파형 시계열과 초 단위 타임스탬프는 없다.",
        "`result`는 `idx`가 1~23인 날짜·유형별 건수 집계 행이다. 두 시트의 `idx`는 서로 다른 순번이며, 동일 날짜에는 원자료 행이 수백~수천 개 있다. 개별 검사 결과를 원자료의 한 행에 결합할 키가 없다.",
        "`scaled_data.csv`에는 불량 라벨·날짜·용접 ID가 없고 변환 절차도 없다. 동일 행 수만으로 원자료와 정렬되었다고 가정하지 않는다.",
        "따라서 결과 건수를 해당 날짜의 모든 용접 행에 방송하거나 누락 날짜를 정상으로 채우지 않는다.",
    ])
    (RESULTS / "02_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")

    raw = raw.copy()
    outcomes = outcomes.copy()
    raw["date"] = _date_values(raw)
    outcomes["date"] = _date_values(outcomes)
    raw["idx"] = pd.to_numeric(raw["idx"], errors="coerce")
    outcomes["defect"] = pd.to_numeric(outcomes["defect"], errors="coerce")
    outcomes["defect type"] = pd.to_numeric(outcomes["defect type"], errors="coerce")
    process = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
    thickness = ["Thickness 1(mm)", "Thickness 2(mm)"]
    for variable in process + thickness:
        raw[variable] = pd.to_numeric(raw[variable], errors="coerce")

    daily = raw.groupby("date", dropna=False).size().rename("measurement_rows").to_frame()
    for variable in process:
        daily[f"{variable} mean"] = raw.groupby("date")[variable].mean()
        daily[f"{variable} std"] = raw.groupby("date")[variable].std()
    daily["defect_count"] = outcomes.groupby("date")["defect"].sum()
    for kind in (1, 2, 3):
        daily[f"type_{kind}"] = outcomes.loc[outcomes["defect type"] == kind].groupby("date")["defect"].sum()
    daily = daily.reset_index().sort_values("date").reset_index(drop=True)
    daily.to_csv(RESULTS / "02_daily_aggregate.csv", index=False, encoding="utf-8-sig")

    # Repeated settings are about input discreteness, never observed weld labels.
    setting_cols = thickness + process
    settings = raw.groupby(setting_cols, dropna=False).size().sort_values(ascending=False)
    unique_settings = int(len(settings))
    repeated_setting_rows = int(settings.loc[settings > 1].sum())
    id_duplicates = int(raw.duplicated(["date", "idx"]).sum())
    _plot_daily(daily)
    _plot_process(daily, process)

    positive_dates = int(outcomes.loc[outcomes["defect"] > 0, "date"].nunique())
    result_dates = int(outcomes["date"].nunique())
    missing_result_dates = [str(date.date()) for date in daily.loc[daily["defect_count"].isna(), "date"]]
    observed_count = int(outcomes["defect"].sum())
    type_counts = {str(kind): int(outcomes.loc[outcomes["defect type"] == kind, "defect"].sum()) for kind in (1, 2, 3)}
    reported_mean, reported_low, reported_high = _bootstrap_reported_count_per_date(daily)
    outcome_types = outcomes.groupby("date")["defect type"].apply(lambda values: set(values.dropna().astype(int)))
    incomplete_type_dates = [str(date.date()) for date, types in outcome_types.items() if types != {1, 2, 3}]
    k2_a = "해당" if positive_dates < 30 else "비해당"
    k2_a_reason = (
        f"식별 가능한 최상위 독립 후보 단위인 날짜에서 불량 보고일 {positive_dates}일 "
        f"{'< 30' if positive_dates < 30 else '≥ 30'}. "
        f"{observed_count}건을 독립 부품 {observed_count}개로 해석할 수 없음"
    )
    counts = {
        "raw_rows": int(len(raw)), "raw_dates": int(raw["date"].nunique()),
        "result_rows": int(len(outcomes)), "result_dates": result_dates,
        "positive_result_dates": positive_dates, "reported_defect_count": observed_count,
        "defect_type_counts": type_counts, "missing_result_dates": missing_result_dates,
        "incomplete_type_dates": incomplete_type_dates,
        "date_idx_duplicate_rows": id_duplicates,
        "unique_process_setting_combinations": unique_settings,
        "rows_in_repeated_process_setting_combinations": repeated_setting_rows,
    }
    summary = {
        "task": "2_robot_welding", "seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
        "criteria": {"K2-a": k2_a, "K2-b": "판정불가", "K2-c": "판정불가", "K2-d": "판정불가"},
        "counts": counts,
        "metrics": {
            "reported_defect_counts_per_reporting_date": {
                "estimate": reported_mean, "ci95": [reported_low, reported_high],
                "unit": "reported counts per reporting date; not weld defect rate",
                "resampling_unit": "date", "n_dates": result_dates,
            },
            "supervised_f1": None, "supervised_pr_auc": None,
        },
        "supervised_available": False,
        "label_join_key_available": False,
        "source_files": [str(path.relative_to(ROOT).as_posix()) for path in source_files],
    }
    (RESULTS / "02_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    range_text = "; ".join(
        f"{column} {raw[column].min():.3f}~{raw[column].max():.3f}"
        for column in process
    )
    report = f"""# 과제 ② 로봇용접 불량예측 사전 타당성 검증

## 1. 요약

| 핵심 증거·경고 | 결과 |
| :-- | :-- |
| 원자료 | {len(raw):,}개 측정 행, {counts['raw_dates']}일, 설비·품목 각각 1종 |
| 검사 결과 | {len(outcomes)}개 날짜·유형별 집계 행, {result_dates}일에 불량 {observed_count}건 보고; 유형 1/2/3 = {type_counts['1']}/{type_counts['2']}/{type_counts['3']}건 |
| K2-a | **{k2_a}** — {k2_a_reason} |
| K2-b | **판정불가** — 용접 행별 불량 라벨·결합 키 없음 |
| K2-c | **판정불가** — 같은 이유로 LightGBM 분할별 F1 산출 불가 |
| K2-d | **판정불가** — 학습·테스트의 개별 양불과 범위 안/밖 불량률 산출 불가 |
| 제한적 기술 지표 | 결과 보고일 8일의 `defect` 합계 평균: **{reported_mean:.2f}건/보고일** [95% 날짜 부트스트랩 CI {reported_low:.2f}, {reported_high:.2f}]. 이는 용접 불량률이나 미보고 유형을 포함한 전체 불량 건수가 아님 |

**결론:** 이 파일만으로 최종검사 전 개별 용접 불량예측의 F1, 실패조건, 안정 공정조건을 검증할 수 없다. 집계 불량 건수를 같은 날짜의 모든 측정 행에 라벨로 붙이면 가짜 성능이 나온다.

## 2. R0 구조 진단

- [파일·시트·dtype·결측·중복·데이터 사전](02_inventory.md). XLSX의 `data set` 시트가 변수 정의 자료다. 별도 PDF/HWP/TXT 설명 문서는 발견되지 않았다.
- 한 `Raw data` 행에는 설비, 품목, 날짜, 두 소재 두께, 가압력, 전류, 전압, 통전시간이 한 번씩 기록되어 있고 `idx`가 날짜마다 다시 시작한다. 한 행은 한 용접의 요약 관측으로 **추정**된다. 시계열 파형이 아니므로 파형 요약특징이나 에너지 적분을 만들지 않았다. 행마다 단일 용접인지, 여러 점의 부품 검사인지 원자료만으로 확정할 수 없다.
- 원자료 기간: {raw['date'].min().date()}~{raw['date'].max().date()}, {counts['raw_dates']}개 날짜. 시각·로트·부품 ID는 없다. `(날짜, idx)` 중복은 {id_duplicates}개다. 설비와 품목은 모두 한 종류이고 소재 두께 1·2는 모두 0.7 mm다. 용접속도 컬럼은 없다.
- `result`의 `defect`는 개별 0/1 라벨이 아니라 유형별 **보고 건수로 추정**된다. 유형 코드는 1 파임불량, 2 용접부족, 3 크랙발생으로 첫 세 행의 G열 메모에서 확인했다. `defect`의 공식 정의, G열의 공식 컬럼명, 검사 판정 기준은 제공되지 않았다. 불량 보고 날짜는 {positive_dates}일이며 {', '.join(missing_result_dates)}에는 결과 행이 없어서 정상일이라고 간주할 수 없다. {', '.join(incomplete_type_dates)}에는 일부 유형 행이 빠져 있어 그날의 합계도 보고된 유형만 포함한다.
- 관측 공정값 범위: {range_text}. 사전의 수집 범위보다 실제 관측 범위가 좁고, 두께·품목·설비에는 변동이 없어 이 축을 따라 일반화할 근거가 없다.
- 공정조건 {unique_settings:,}개 조합 중 반복 조합에 속한 행은 {repeated_setting_rows:,}/{len(raw):,}개다. 같은 조건의 개별 불량률이 0 또는 1로 갈리는지는 라벨 부재로 확인할 수 없다. 품목·설비별 불량률 비교도 한 종류라 불가능하다.
- [날짜별 측정 행 수·공정 평균·집계 불량 건수](02_daily_aggregate.csv). 데이터 사전상 `working time`은 작업시간이지만 실제 값은 엑셀 **날짜** 일련번호로만 기록되어 있다.

## 3. R1 누수·shortcut 진단

- [날짜·불량유형별 집계 건수](02_defect_over_time.png), [날짜별 공정조건 평균](02_process_by_date.png). 불량 건수는 날짜별로 변하지만 개별 용접의 양불 배치는 보이지 않는다. {', '.join(missing_result_dates)}의 막대 부재는 **미보고**이며 0건이 아니다.
- 단변량 AUROC 상위 10개를 계산하지 않았다. 결과가 있는 날짜는 모두 불량 건수가 양수이고, 개별 정상/불량 행 라벨이 없다. 날짜 집계 건수를 개별 행에 복제하거나 미보고일을 음성으로 만들면 AUROC는 정의상 잘못된다.
- 예측 시점 배제 변수: `result`의 `defect`, `defect type`, 유형 설명(G열)은 검사 후 정보다. `idx`는 생산순번이지만 날짜마다 재시작하는 식별자이며 라벨 대체정보가 될 수 있어 모델 특성에서 제외 대상이다. `working time`은 날짜 수준 정보로 드리프트 진단에만 사용할 수 있다. `scaled_data.csv`에는 라벨·날짜·용접 ID와 변환 출처가 없으므로 모델 입력으로 결합하지 않았다.

## 4. R2 분할 비교

랜덤 층화 70/15/15, 그룹 분할, 시간순 70/15/15 모두 **판정불가**. 라벨과 용접·검사 결합 키가 없어 독립 테스트 집합을 구성할 수 없다. 식별 가능한 그룹 후보는 날짜뿐이고, 설비·품목은 상수다. 따라서 LightGBM F1·PR-AUC 비교 및 K2-c 판단을 하지 않았다.

## 5. R3 모델 비교

다수 클래스, 단일변수 임계 규칙, 로지스틱 회귀, RandomForest, LightGBM을 학습하지 않았다. 용접별 양불이 없으므로 검증구간에서 F1 임계값 또는 확률 보정을 정할 수 없다. F1·정밀도·재현율·PR-AUC·ROC-AUC·Brier score, 신뢰도 곡선, 검사 상위 5%/10%/20% 적발률, 검사 100건당 적발 건수에는 보고 가능한 추정치나 부트스트랩 CI가 없다. K2-b도 판정불가다.

## 6. R4 영향요인·실패조건

permutation importance, PDP/ICE, 두 조건 조합별 불량률, FN·FP 집중 조건 및 불량 유형별 재현율을 계산하지 않았다. 날짜 평균과 유형별 총건수의 동시 변화는 특정 용접조건이 불량에 영향을 준다는 증거가 아니다. 특히 두께는 고정되어 두께별 실패조건 비교 자체가 불가능하다. 소수 날짜에서 탐색적 비교를 해도 다중 비교 조정 없이 확증할 수 없다.

## 7. R5 안정 공정조건 범위

학습 구간에서 저위험 조건을 도출하지 않았다. 해당 조건의 용접별 양불률과 테스트 범위 안/밖 불량률을 측정할 수 없어 K2-d는 판정불가다. 관측 연관조차 검증되지 않았고 공정조건 변경의 인과 효과는 더욱 주장할 수 없다.

## 8. 평가표 여섯 항목에서 보여줄 수 있는 것

| 항목 | 이 데이터로 가능한 증거와 한계 |
| :-- | :-- |
| 데이터 이해·진단 15 | 측정 행과 검사 집계의 다른 단위, 날짜·품목·설비 범위, 변수 단위, 결합 키 부재를 재현 가능하게 진단할 수 있다. |
| AI 모델 40 | 개별 양불 라벨이 없어 정당한 베이스라인/비교 모델 F1과 선정 근거를 만들 수 없다. |
| 영향요인·오류분석 15 | 날짜별 조건 분포와 유형별 집계 건수만 가능하다. FN/FP·조건별 개별 불량률은 불가하다. |
| 현장 활용 10 | 검사 결과에 용접 ID·시각·부품/로트 키를 추가하는 데이터 수집 요구사항을 제시할 수 있다. 개별 검사 우선순위나 안정 범위는 제시할 수 없다. |
| 창의성·차별성 10 | 행 단위 불일치와 집계 라벨 방송의 위험을 사전 검증으로 보여줄 수 있다. 불량 저감 효과는 입증할 수 없다. |
| 코드 재현성 10 | `python -X utf8 verification/t2_weld.py`로 인벤토리·그림·일자 집계·요약을 재생성한다. 지도학습은 입력 데이터가 갖춰져야 활성화할 수 있다. |

## 9. 가정

1. 엑셀 일련번호는 1899-12-30 기준의 날짜로 해석했다. `working time`에 시각 소수부가 없으므로 날짜 단위만 사용했다.
2. `Raw data` 한 행은 조건 한 세트의 관측으로 취급했으며 단일 용접 1건이라는 것은 **추정**이다. 검사 모집단과 측정 행의 대응이 불명확해 둘의 비율은 계산하지 않았다.
3. `defect` 값은 데이터 사전에 정의되지 않았지만 날짜·유형별 비음수 정수라서 보고 건수로 추정했다. 검사 단위별 건수인지, 한 검사에서 중복 유형이 가능한지는 알 수 없다.
4. 유형 1/2/3 설명은 제목 없는 G열의 첫 세 행에만 있으나 같은 코드가 모든 날짜에 같은 뜻이라고 가정했다.
5. 날짜를 식별 가능한 가장 큰 독립 후보 단위로 사용해 K2-a를 판정했다. 날짜 간 독립성도 실제로 확인되지 않았다.
6. 결과 행이 없는 날짜·유형은 미보고로 두었다. 불량 0건이라는 가정은 하지 않았다. 그림에서 빠진 막대도 0건의 증거가 아니다. {', '.join(incomplete_type_dates)}의 일자별 합계는 전체 유형의 확정 합계가 아니다.
7. `scaled_data.csv`의 행 정렬과 변환 절차는 불명확해 원자료에 결합하지 않았다.
8. 날짜 부트스트랩 1,000회(seed 42)는 **기술 지표**인 보고일별 기록 건수의 변동만 나타낸다. 전체 기간·독립 부품의 불확실성이나 라벨 성능 신뢰구간이 아니다.

## 10. 확인이 필요한 미해결 질문

- 개별 검사 기록에 용접 `idx`, 작업 날짜·시각, 부품 ID 또는 로트 ID를 연결할 키가 있는가? 검사 단위가 용접점인가, 부품인가?
- `result`의 날짜별 기록은 모든 불량 유형을 빠짐없이 포함하는가? 결과 행이 없는 날짜는 미검사인가 0건인가?
- 개별 불량 판정 규격과 불량 유형 코드북, 중복 불량 유형 허용 여부는 무엇인가?
- 두께·소재·품목·설비 변동이 있는 별도 데이터와 용접속도 정보가 있는가? `scaled_data.csv`의 변환·결측·행 정렬 근거는 무엇인가?

### 사후 의견

K2-a의 날짜 기준 {positive_dates}일은 용접 불량예측이 요구하는 독립 부품 수의 대체치가 아니다. 기준의 신뢰성 한계는 사전 판정과 별도로 검토해야 한다.
"""
    output = RESULTS / "02_report.md"
    output.write_text(report, encoding="utf-8")
    _write_comparison(counts, k2_a)
    return output


if __name__ == "__main__":
    print(main())
