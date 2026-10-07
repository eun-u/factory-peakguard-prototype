"""제출용 소스코드 ZIP 생성과 블라인드 점검.

    python tools/package_submission.py                 # submission/source/task5_source.zip
    python tools/package_submission.py --name 파일명.zip

허용 목록에 있는 파일만 담는다. 내부 의사결정 문서, 과제 ②·③ 자료, 사전 검증 작업물, 화면 시제품은 넣지 않는다.
ZIP에 담길 모든 텍스트 파일에서 소속·개인 경로로 보이는 문자열을 찾아, 하나라도 있으면 ZIP을 만들지 않는다.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "README.md", "requirements.txt", "run_all.py",
    "configs/default.yaml",
    "src/**/*.py",
    "tests/*.py",
    "data/raw/task05_power/okm_augumented_2021.csv",   # 학습용 데이터
    "outputs/predictions/*.csv",                       # 테스트데이터 예측결과 파일
    "outputs/tables/*.csv", "outputs/figures/*.png", "outputs/logs/*.json",
]
REQUIRED = ["README.md", "requirements.txt", "run_all.py", "configs/default.yaml",
            "data/raw/task05_power/okm_augumented_2021.csv",
            "outputs/predictions/test_predictions_1h.csv", "outputs/predictions/test_predictions_daily_max.csv"]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv"}
# 소속·개인 식별로 이어질 수 있는 문자열. 필요하면 팀이 아는 학교·기관명을 --extra로 추가한다.
PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",   # 이메일
    r"[A-Za-z]:\\Users\\", r"/Users/[^/\s]+", r"/home/[^/\s]+",   # 개인 계정 경로
    r"대학교", r"대학원", r"University", r"\bUniv\b", r"Institute", r"연구소", r"주식회사", r"\(주\)",
    r"\bInc\.", r"Co\., ?Ltd",
]


def collect() -> list[Path]:
    files: set[Path] = set()
    for pattern in INCLUDE:
        files.update(p for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def blind_scan(files: list[Path], extra: list[str]) -> list[str]:
    regex = re.compile("|".join(PATTERNS + [re.escape(e) for e in extra]), re.IGNORECASE)
    hits = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for number, line in enumerate(text.splitlines(), 1):
            match = regex.search(line)
            if match:
                hits.append(f"{path.relative_to(ROOT)}:{number}: {match.group(0)}")
    return hits


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="task5_source.zip")
    parser.add_argument("--extra", nargs="*", default=[], help="추가로 금지할 문자열(학교·기관명 등)")
    args = parser.parse_args(argv)
    files = collect()
    missing = [r for r in REQUIRED if not (ROOT / r).is_file()]
    if missing:
        print("필수 파일이 없습니다. 원자료 배치와 python run_all.py 실행을 먼저 하세요:")
        print("\n".join(f"  - {m}" for m in missing))
        return 1
    hits = blind_scan(files, args.extra)
    if hits:
        print("블라인드 점검 실패. 아래 문자열을 지운 뒤 다시 실행하세요:")
        print("\n".join(f"  - {h}" for h in hits))
        return 2
    out = ROOT / "submission" / "source" / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"{len(files)}개 파일 → {out.relative_to(ROOT)}")
    print("다음: ZIP을 새 폴더에 풀고 README의 설치·실행 명령으로 run_all.py를 다시 실행해 결과를 대조하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
