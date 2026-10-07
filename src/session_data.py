"""Read only the sealed development prefix for the unattended 2026-09-24 session.

The boundary hour is read field by field. At 09:45 the reader stops immediately
after the 30-minute field and never decodes the 45/60-minute holdout fields.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import ANALYSIS_COLUMNS, MINUTES, SOURCE_COLUMNS


SEALED_BOUNDARY = pd.Timestamp("2021-08-09 09:45:00")
PROTECTED_DIRECTORIES = ("report", "slides", "submission", "verification", "prototype", "data/raw")
PROTECTED_FILES = ("docs/roadmap.html", "eval_protocol.md",
                   "outputs/logs/adoption_criteria.md",
                   "outputs/logs/adoption_criteria_addendum_0924.md")


def _field(stream) -> bytes:
    """Read exactly one comma-terminated metadata or permitted power field."""
    data = bytearray()
    while True:
        char = stream.read(1)
        if char == b",":
            return bytes(data)
        if not char or char in (b"\r", b"\n"):
            raise ValueError("Source row ended before a required safe field")
        data.extend(char)


def _safe_rows(path: Path, cutoff: pd.Timestamp) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    with path.open("rb") as stream:
        header = next(csv.reader([stream.readline().decode("utf-8-sig").strip()]))
        if len(header) != len(set(header)) or not SOURCE_COLUMNS <= set(header):
            raise ValueError("Source CSV columns are missing or duplicated")
        if header[:6] != ["날짜", "시간", "15분", "30분", "45분", "60분"]:
            raise ValueError("Safe prefix reader requires the pinned source column order")
        last_day: pd.Timestamp | None = None
        position = -1
        while True:
            first = stream.read(1)
            if not first:
                break
            # No power field is consumed until date and hour metadata are checked.
            if first in (b"\r", b"\n"):
                raise ValueError("Blank source rows are not allowed")
            day_bytes = first + _field(stream)
            hour_bytes = _field(stream)
            day = pd.Timestamp(datetime.strptime(day_bytes.decode("ascii"), "%Y%m%d"))
            raw_hour = float(hour_bytes.decode("ascii"))
            if last_day is not None and day < last_day:
                raise ValueError("Source dates are not chronological")
            if day != last_day:
                if last_day is not None and position != 23:
                    raise ValueError("A completed source day has fewer than 24 ordered rows")
                last_day, position = day, 0
            else:
                position += 1
            if position > 23:
                raise ValueError("A source day has more than 24 hourly rows")
            safe_minutes = [minute for minute in MINUTES
                            if day + pd.Timedelta(minutes=position * 60 + minute) < cutoff]
            if not safe_minutes:
                break
            if len(safe_minutes) == 4:
                line = day_bytes + b"," + hour_bytes + b"," + stream.readline()
                values = next(csv.reader([line.decode("utf-8")]))
                if len(values) != len(header):
                    raise ValueError("Full safe source row has an unexpected column count")
                record = dict(zip(header, values))
            else:
                # The next comma ends the last safe power value. Never read
                # the remaining bytes on this boundary row.
                record = {name: None for name in header}
                record["날짜"], record["시간"] = day_bytes.decode("ascii"), hour_bytes.decode("ascii")
                for minute in safe_minutes:
                    record[f"{minute}분"] = _field(stream).decode("ascii")
            record.update(_date=day, _position=position, _source_hour=raw_hour,
                          _safe_minutes=tuple(safe_minutes), _partial=len(safe_minutes) < 4)
            rows.append(record)
            if len(safe_minutes) < 4:
                break
        if not rows:
            raise ValueError("No safe source observations precede the sealed cutoff")
        if rows[-1]["_date"] < cutoff.normalize() and rows[-1]["_position"] != 23:
            raise ValueError("Source ended before the sealed development boundary")
    return rows, header


def _normalize(rows: list[dict], cutoff: pd.Timestamp) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    # Full days follow the same positional timestamp repair as src.data.
    for day, block in raw.groupby("_date", sort=False):
        positions = block["_position"].to_numpy(dtype=int)
        complete = day < cutoff.normalize()
        if complete and (len(block) != 24 or not np.array_equal(positions, np.arange(24))):
            raise ValueError("Cannot repair hours safely: require 24 ordered rows per complete date")
        observed = block["_source_hour"].to_numpy(dtype=float)
        plausible = np.isfinite(observed) & (observed >= 0) & (observed <= 23)
        if not np.array_equal(observed[plausible], positions[plausible]):
            raise ValueError("Source hours conflict with row-order timestamps")
        if not complete and not plausible.all():
            raise ValueError("Boundary date has an unverified repaired hour")
    raw["time_repaired"] = ~raw["_source_hour"].between(0, 23)
    records: list[dict] = []
    for row in rows:
        repaired = not (np.isfinite(row["_source_hour"]) and 0 <= row["_source_hour"] <= 23)
        for minute in row["_safe_minutes"]:
            rec = {"ts_end": row["_date"] + pd.Timedelta(minutes=row["_position"] * 60 + minute),
                   "power": pd.to_numeric(row[f"{minute}분"], errors="coerce"),
                   "time_repaired": repaired,
                   "production_target": pd.to_numeric(row.get("생산량"), errors="coerce")}
            for source, dest in ANALYSIS_COLUMNS.items():
                if source in raw:
                    rec[dest] = pd.to_numeric(row.get(source), errors="coerce")
            records.append(rec)
    frame = pd.DataFrame(records).set_index("ts_end").sort_index()
    frame.index.name = "ts_end"
    if frame.index.has_duplicates or frame.index.max() >= cutoff:
        raise AssertionError("The safe loader produced a duplicate or sealed timestamp")
    regular = pd.date_range(frame.index.min(), frame.index.max(), freq="15min", name="ts_end")
    frame = frame.reindex(regular)
    if frame.index.max() >= cutoff:
        raise AssertionError("Reindexing crossed the sealed timestamp")
    frame.loc[frame.power.lt(0), "power"] = np.nan
    frame["time_repaired"] = frame.time_repaired.fillna(False).astype(bool)
    frame["missing_power"] = frame.power.isna()
    frame["zero_power"] = frame.power.eq(0)
    clean = frame.loc[~frame.time_repaired, "power"].dropna()
    q1, q3 = clean.quantile([.25, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    frame["iqr_outlier"] = frame.power.lt(low) | frame.power.gt(high)
    complete = raw.loc[~raw["_partial"]]
    hourly_end = pd.DatetimeIndex(complete["_date"] + pd.to_timedelta(complete["_position"] + 1, unit="h"))
    production = pd.Series(pd.to_numeric(complete.get("생산량"), errors="coerce").to_numpy(), index=hourly_end)
    repaired_hour = pd.Series(complete.time_repaired.to_numpy(dtype=bool), index=hourly_end)
    if production.index.has_duplicates:
        raise ValueError("Duplicate hourly production completion timestamp")
    frame["production_completed"] = production.reindex(frame.index)
    frame["production_completed_bad"] = repaired_hour.reindex(frame.index).fillna(False).astype(bool)
    frame["production_known"] = production.reindex(frame.index).ffill()
    frame["production_known_bad"] = repaired_hour.reindex(frame.index).astype("boolean").ffill().fillna(True).astype(bool)
    frame["production_hour_date"] = (frame.index - pd.Timedelta(nanoseconds=1)).normalize()
    for field in ("production_target", *ANALYSIS_COLUMNS.values()):
        if field in frame:
            frame[f"missing_{field}"] = frame[field].isna()
    zero_group = frame.zero_power.ne(frame.zero_power.shift()).cumsum()
    frame["zero_run_length"] = frame.zero_power.groupby(zero_group).transform("sum").where(frame.zero_power, 0).astype(int)
    frame.attrs["source_mode"] = "development_prefix_only"
    frame.attrs["sealed_boundary"] = str(cutoff)
    frame.attrs["boundary_partial_hour"] = bool(raw["_partial"].iloc[-1])
    return frame


def load_development_history(root: str | Path) -> pd.DataFrame:
    """Return a source-normalized history with every ts_end strictly before 09:45.

    Unlike the original full-data loader this routine stops inside the final
    hourly CSV row, after its 15- and 30-minute fields. The partial row's
    production and weather values are not parsed because they were not yet
    available at 09:30.
    """
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(cfg["split"]["test_start_origin"])
    if cutoff != SEALED_BOUNDARY:
        raise ValueError("The session cutoff must equal the sealed 2021-08-09 09:45 boundary")
    source = root / cfg["source"]
    rows, _ = _safe_rows(source, cutoff)
    frame = _normalize(rows, cutoff)
    if frame.index.max() != cutoff - pd.Timedelta(minutes=15):
        raise AssertionError("Development history does not end at the expected 09:30 slot")
    if (frame.index >= cutoff).any():
        raise AssertionError("A test timestamp entered development history")
    return frame


def load_development_oof(root: str | Path) -> pd.DataFrame:
    """Verify OOF timestamps before decoding any prediction or target column."""
    path = Path(root).resolve() / "outputs/predictions/development_oof.csv"
    return read_development_oof(path)


def read_development_oof(path: str | Path) -> pd.DataFrame:
    """Metadata-first reader also used by isolated diagnostic fixtures."""
    path = Path(path)
    verified_stamps: set[bytes] = set()

    def verify_stamp(value: bytes) -> None:
        # Candidate models repeat exactly the same timestamps. Memoising only
        # their validation avoids parsing the same date hundreds of times;
        # every row's metadata is still checked before its body is consumed.
        if value not in verified_stamps:
            stamp = pd.Timestamp(value.decode("ascii"))
            if pd.isna(stamp) or stamp >= SEALED_BOUNDARY:
                raise ValueError("OOF metadata crosses the sealed development boundary or is missing")
            verified_stamps.add(value)

    with path.open("rb") as stream:
        header = next(csv.reader([stream.readline().decode("utf-8-sig").strip()]))
        if header[:2] != ["origin", "target_time"]:
            raise ValueError("OOF metadata must occupy the first two CSV columns")
        count = 0
        while True:
            first = stream.read(1)
            if not first:
                break
            if first in (b"\r", b"\n"):
                raise ValueError("OOF has a blank row")
            verify_stamp(first + _field(stream))
            verify_stamp(_field(stream))
            stream.readline()
            count += 1
    if count == 0:
        raise ValueError("Development OOF is empty")
    frame = pd.read_csv(path, parse_dates=["origin", "target_time"], low_memory=False)
    if (len(frame) != count or frame[["origin", "target_time"]].isna().any().any()
            or frame.origin.ge(SEALED_BOUNDARY).any() or frame.target_time.ge(SEALED_BOUNDARY).any()):
        raise AssertionError("OOF changed during verified read or crosses the sealed boundary")
    return frame


def write_preflight_manifest(root: str | Path) -> Path:
    """Byte-hash protected artifacts without parsing their contents."""
    root = Path(root).resolve()
    files: set[Path] = set()
    for directory in PROTECTED_DIRECTORIES:
        files.update(path for path in (root / directory).rglob("*") if path.is_file())
    files.update(root / name for name in PROTECTED_FILES if (root / name).is_file())
    digests: dict[str, str] = {}
    for path in sorted(files):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Protected path escapes the workspace: {path}")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        digests[path.relative_to(root).as_posix()] = sha.hexdigest()
    missing = [name for name in PROTECTED_FILES if name not in digests]
    if missing:
        raise FileNotFoundError(f"Required protected files are missing: {missing}")
    target = root / "outputs/logs/session_0924_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
        if prior.get("protected_sha256") != digests or prior.get("boundary") != str(SEALED_BOUNDARY):
            raise ValueError("Protected files changed; refusing to replace the session preflight")
        return target
    target.write_text(json.dumps({"boundary": str(SEALED_BOUNDARY),
                                  "protected_file_count": len(digests),
                                  "protected_sha256": digests}, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[1]
    safe = load_development_history(workspace)
    manifest = write_preflight_manifest(workspace)
    print(json.dumps({"safe_rows": len(safe), "safe_max_ts_end": str(safe.index.max()),
                      "preflight_manifest": str(manifest)}, ensure_ascii=False))
