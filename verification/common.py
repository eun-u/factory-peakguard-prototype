"""Source inventory and document extraction for the feasibility check."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache"}


def project_files():
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts):
            yield path


def read_csv(source):
    errors = []
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            if isinstance(source, bytes):
                return pd.read_csv(io.BytesIO(source), encoding=encoding), encoding
            return pd.read_csv(source, encoding=encoding), encoding
        except (UnicodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV decoding failed: " + "; ".join(errors))


def docx_paragraphs(path):
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return [
        "".join(node.text or "" for node in para.findall(".//w:t", ns)).strip()
        for para in root.findall(".//w:p", ns)
    ]


def temporal_summary(df):
    columns = list(df.columns)
    for name in columns:
        if name.lower() in {"timestamp", "datetime", "date_time"}:
            times = pd.to_datetime(df[name], errors="coerce")
            return f"{times.min()}–{times.max()} (파싱 실패 {times.isna().sum()})"
    if "날짜" in columns:
        dates = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d", errors="coerce")
        detail = f"날짜 {dates.min()}–{dates.max()} (파싱 실패 {dates.isna().sum()})"
        if "시간" in columns:
            hours = pd.to_numeric(df["시간"], errors="coerce")
            detail += f"; 원본 시간 {hours.min()}–{hours.max()}, 0–23 외 {((hours < 0) | (hours > 23) | hours.isna()).sum()}행"
        return detail
    for name in columns:
        if "time" in name.lower() or "date" in name.lower():
            times = pd.to_datetime(df[name], errors="coerce")
            return f"{times.min()}–{times.max()} (파싱 실패 {times.isna().sum()})"
    return "시간 열 식별 불가"


def table_details(df):
    lines = [
        f"- 행×열: {len(df):,} × {len(df.columns):,}",
        f"- 시간 범위: {temporal_summary(df)}",
        f"- 완전 중복 행: {df.duplicated().sum():,}",
        "- 컬럼 | dtype | 결측 수 | 고유값 수",
        "  :-- | :-- | --: | --:",
    ]
    for name in df.columns:
        lines.append(f"  {name} | {df[name].dtype} | {df[name].isna().sum():,} | {df[name].nunique(dropna=True):,}")
    return "\n".join(lines)


def build_inventory():
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = list(project_files())
    sections = [
        "# 01 데이터·파일 인벤토리",
        "",
        "이 파일은 실행 시점의 프로젝트 스냅샷이다. `.git`, `.venv`, `__pycache__` 등 실행환경 내부 파일은 제외했다. ZIP 내부 CSV와 풀린 CSV는 별도 관측자료로 합산하지 않는다. 실행 뒤 생성되는 결과 파일은 다음 재실행에서 목록에 나타날 수 있다.",
        "",
        "## 전체 파일 목록",
        "",
        "파일 | 바이트 | SHA-256 앞 12자리 | 형태",
        ":-- | --: | :-- | :--",
    ]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        digest = "self" if path == RESULTS / "01_inventory.md" else hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        size = "self" if path == RESULTS / "01_inventory.md" else f"{path.stat().st_size:,}"
        kind = "CSV 표" if path.suffix.lower() == ".csv" else "ZIP 자료" if path.suffix.lower() == ".zip" else "문서" if path.suffix.lower() in {".docx", ".pdf", ".hwp", ".hwpx", ".txt", ".md"} else "코드/설정/기타"
        sections.append(f"{rel} | {size} | {digest} | {kind}")
    sections.extend(["", "## 표 파일의 구조", ""])
    seen = 0
    for path in files:
        if path.suffix.lower() == ".csv":
            df, enc = read_csv(path)
            sections.extend([f"### {path.relative_to(ROOT).as_posix()}", "", f"- 인코딩: {enc}", table_details(df), ""])
            seen += 1
        elif path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                for member in members:
                    if member.filename.lower().endswith(".csv"):
                        payload = archive.read(member)
                        df, enc = read_csv(payload)
                        duplicates = [
                            item.relative_to(ROOT).as_posix() for item in files
                            if item.suffix.lower() == ".csv" and item.read_bytes() == payload
                        ]
                        duplicate_note = "동일한 비압축 사본: " + ", ".join(duplicates) if duplicates else "동일한 비압축 사본 없음"
                        sections.extend([f"### {path.relative_to(ROOT).as_posix()}::{member.filename}", "", f"- ZIP 내부 압축 전 크기: {member.file_size:,}바이트; 인코딩: {enc}; {duplicate_note}", table_details(df), ""])
                        seen += 1
                    else:
                        sections.extend([f"- ZIP 내부 비CSV: {path.name}::{member.filename} ({member.file_size:,}바이트)", ""])
    if not seen:
        sections.append("표 파일 없음.\n")
    sections.extend(["## 설명 문서와 변수 정의", ""])
    docs = [p for p in files if p.suffix.lower() == ".docx"]
    for path in docs:
        lines = docx_paragraphs(path)
        sections.extend([
            f"### {path.relative_to(ROOT).as_posix()}", "",
            f"- DOCX 문단 {len(lines):,}개를 확인했다. 내부 의사결정·탐색 기록이며 공식 데이터 사전으로 확인되지 않았다.",
            "- ⑤: `날짜`·`시간`은 시간축, `15분`·`30분`·`45분`·`60분`은 시간별 네 전력 위치, `평균`은 그 파생 평균으로 기술한다. `생산량`은 생산 실적/계획 여부 미확정, `기온`·`풍속`·`습도`·`강수량`은 환경 관측, `전기요금(계절)`·`인건비`는 비용·시간 대리 가능성, `day`·`d`·`m`은 달력 파생, `공장인원`은 생성관계 의심이다.",
            "- ⑤의 전력 kW/kWh 단위와 네 15분 열의 구간 시작·종료 의미는 문서도 미확정으로 표기한다. 미래 생산량·미래 실측 기상은 사전에 안다고 볼 근거가 없다.",
            "- ③은 이전 대화 탐색값(정상 20,000행·이상 600행, 파일별 라벨 고정, 짧은 공백 분절)을 기록하지만 재현 코드·원자료가 이 문서에 없다. 공식 변수 정의와 원파형 여부는 확인되지 않았다.",
            "- 이 문서의 ⑤ 선택 결론과 과거 성능 수치는 이번 중립적 타당성 검증의 결론·재현 수치로 채택하지 않는다.", "",
        ])
    if not docs:
        sections.append("읽을 수 있는 DOCX 설명 문서 없음.\n")
    sections.extend([
        "## 과제 판별", "",
        "- ⑤: `okm_augumented_2021.csv`는 `15분`·`30분`·`45분`·`60분`, `생산량`, `전기요금(계절)`을 포함하고 ⑤ 폴더/ZIP에 위치한다. 동일한 압축·비압축 사본은 중복 입력으로 취급하지 않는다.",
    ])
    press = []
    for path in files:
        if path.suffix.lower() == ".csv":
            df, _ = read_csv(path)
            if {"AI0_Vibration", "AI1_Vibration", "AI2_Current"}.issubset(df.columns):
                press.append(path.relative_to(ROOT).as_posix())
    if press:
        sections.append("- ③: AI0/AI1 진동·AI2 전류 컬럼으로 확인: " + ", ".join(press))
    else:
        sections.append("- ③: 현재 작업 폴더에서 진동·전류 3채널을 갖춘 CSV를 찾지 못했다. ③의 P0–P5 수치는 원자료 부재로 판정 불가다.")
    output = RESULTS / "01_inventory.md"
    output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build_inventory())
