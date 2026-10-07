"""원자료와 같은 열 구조의 합성 CSV. 파이프라인 실행 점검 전용이며 결과 해석에 쓰지 않는다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["날짜", "시간", "15분", "30분", "45분", "60분", "평균", "생산량", "기온", "풍속", "습도", "강수량",
           "전기요금(계절)", "day", "d", "m", "공장인원", "인건비"]


def make_synthetic(path: Path, days: int = 257, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    holidays = set(rng.choice(days, 6, replace=False))
    for di, date in enumerate(dates):
        weekday = date.dayofweek
        evening_shift = weekday < 5 and rng.random() < 0.25
        for hour in range(24):
            working = weekday < 5 and di not in holidays and (8 <= hour < 18 or (evening_shift and 18 <= hour < 23))
            production = int(max(0, rng.normal(900, 250))) if working else int(rng.random() < .1) * 30
            base = 20 + 15 * np.sin(2 * np.pi * (date.dayofyear - 30) / 365) ** 2
            values = []
            for _ in range(4):
                v = base + 0.15 * production + rng.normal(0, 6) + (25 if working and rng.random() < .15 else 0)
                values.append(0 if di in holidays and 2 <= hour < 5 else int(max(0, round(v))))
            temp = 5 + 20 * np.sin(np.pi * (date.dayofyear - 20) / 365) + rng.normal(0, 2)
            rows.append([int(date.strftime("%Y%m%d")), hour, *values, int(round(np.mean(values))), production,
                         round(temp, 1), round(abs(rng.normal(2, 1)), 1), int(rng.integers(30, 90)),
                         round(max(0, rng.normal(0, 1)), 1), 1.0 if date.month in (6, 7, 8) else 0.0,
                         weekday, date.day, date.month, float(rng.integers(50, 120)), 1.0])
    df = pd.DataFrame(rows, columns=COLUMNS)
    for bad in ("2021-07-13", "2021-07-15"):
        idx = df.index[df["날짜"] == int(bad.replace("-", ""))]
        df.loc[idx, "시간"] = np.arange(165, 165 + len(idx))
    df.loc[rng.choice(len(df), 3, replace=False), "풍속"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


if __name__ == "__main__":
    import sys
    print(make_synthetic(Path(sys.argv[1])))
