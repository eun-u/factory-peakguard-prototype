"""End-to-end FG-R11 pipeline: caches -> lock -> one-time test evaluation, or reproduction.

FG-R11 = exact-copy gate (>=5 observed slots) + Chronos-2 (context 2048) reconciled by an
origin-aligned temporal hierarchy (WLS-variance) + per-horizon realized-error MOS + CAL peak
shift + conformal risk. Model code: phase_f/final_fg_r11.py (and the modules it hashes).

Stages
1. Caches (Chronos environment, GPU): dev Chronos forecasts, test Chronos forecasts,
   temporal-hierarchy base forecasts (dev, full). Created only when missing or with
   --refresh-caches; every cache with a locked hash is verified against the lock.
2. Lock (dev data only) when FINAL_LOCK.json is missing.
3. Evaluation once when FINAL_TEST_RESULT.json is missing. When it already exists the
   stored result is never overwritten: the locked model is re-evaluated in a temporary
   directory and compared with the stored metrics (reproduction, eval_protocol item 9).
4. Prediction CSV checks and report blocks (phase_f.report_fg_r11).

Usage: python run_all.py --fg-r11 [--refresh-caches]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R11 = ROOT / "outputs/phase_f/final_fg_r11"
R8 = ROOT / "outputs/phase_f/final_fg_r8"
DEV_CHRONOS = ROOT / "outputs/phase_f/goal_gate_chronos_v1/cache/chronos_c2048.parquet"
TEST_CHRONOS = R8 / "cache/chronos_test.parquet"
DAY_AHEAD = R8 / "cache/chronos_day_ahead.parquet"
TH_DEV = R11 / "cache/th_dev.parquet"
TH_FULL = R11 / "cache/th_full.parquet"
PRED_DIR = ROOT / "outputs/predictions"
CSV_COLUMNS = ["origin", "target_time", "horizon", "actual", "pred", "q05", "q10", "q90", "q95",
               "p_exceed", "gated", "alert"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chronos_python() -> str:
    env = os.environ.get("PEAKGUARD_CHRONOS_PYTHON")
    if env:
        return env
    for candidate in (ROOT / "outputs/phase_f/env/Scripts/python.exe", ROOT / "outputs/phase_f/env/bin/python"):
        if candidate.exists():
            return str(candidate)
    raise SystemExit("Chronos environment not found. Create it from requirements-chronos.txt "
                     "(see README) or set PEAKGUARD_CHRONOS_PYTHON.")


def _run(args: list[str], python: str | None = None) -> None:
    cmd = [python or sys.executable, "-W", "ignore", "-m", *args]
    print("  $", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_caches(refresh: bool) -> dict:
    lock = json.loads((R11 / "FINAL_LOCK.json").read_text(encoding="utf-8")) if (R11 / "FINAL_LOCK.json").exists() else None
    r8_lock = json.loads((R8 / "FINAL_LOCK.json").read_text(encoding="utf-8"))
    jobs = [
        (DEV_CHRONOS, ["phase_f.gate_chronos_infer", "--context", "2048", "--out", str(DEV_CHRONOS)],
         lock["chronos_dev_cache_sha256"] if lock else None),
        (TEST_CHRONOS, ["phase_f.final_fg_r8", "infer"], lock["r8_test_chronos_sha256"] if lock else None),
        (TH_DEV, ["phase_f.temporal_hierarchy", "--scope", "dev", "--out", str(TH_DEV)],
         lock["th"]["th_dev_sha256"] if lock else None),
        (TH_FULL, ["phase_f.temporal_hierarchy", "--scope", "full", "--out", str(TH_FULL)], None),
    ]
    report = {}
    for path, args, expected in jobs:
        if refresh and path.exists():
            keep = path.with_suffix(".previous.parquet")
            shutil.copy2(path, keep)
            path.unlink()
            if path == TEST_CHRONOS and DAY_AHEAD.exists():
                DAY_AHEAD.unlink()
        if not path.exists():
            if path == TEST_CHRONOS and r8_lock is None:
                raise SystemExit("FG-R8 lock missing; cannot regenerate the test Chronos cache")
            _run(args, chronos_python())
        digest = _sha(path)
        if expected and digest != expected:
            raise SystemExit(f"{path.relative_to(ROOT)} differs from the locked hash; restore the archived cache")
        report[str(path.relative_to(ROOT).as_posix())] = {"sha256": digest, "locked_hash": "verified" if expected else "not_locked"}
    return report


def check_prediction_csv(result: dict) -> dict:
    """Validate the A7 prediction files against the evaluation result."""
    full = pd.read_csv(PRED_DIR / "final_test_fg_r11.csv", parse_dates=["origin", "target_time"])
    h4 = pd.read_csv(PRED_DIR / "final_test_fg_r11_h4.csv", parse_dates=["origin", "target_time"])
    day = pd.read_csv(PRED_DIR / "final_test_next_day_max.csv")
    problems = []
    if list(full.columns) != CSV_COLUMNS or list(h4.columns) != CSV_COLUMNS:
        problems.append("unexpected columns")
    if len(full) != result["rows"]:
        problems.append(f"row count {len(full)} != result rows {result['rows']}")
    if not h4.horizon.eq(4).all() or len(h4) != int(full.horizon.eq(4).sum()):
        problems.append("h4 file is not the horizon-4 subset")
    if full[["pred", "q05", "q10", "q90", "q95", "p_exceed"]].isna().any().any():
        problems.append("missing prediction values")
    if full.duplicated(["origin", "horizon"]).any():
        problems.append("duplicate origin/horizon rows")
    if not (full.target_time == full.origin + pd.to_timedelta(15 * full.horizon, unit="m")).all():
        problems.append("target_time != origin + 15min * horizon")
    if (full.origin < pd.Timestamp("2021-08-09 09:45")).any():
        problems.append("prediction before the test boundary")
    mae = full.assign(e=(full.pred - full.actual).abs()).groupby("horizon").e.mean().mean()
    expected = result["point"]["FG-R11"]["AUC_MAE"]
    if abs(mae - expected) > 1e-9:
        problems.append(f"CSV AUC MAE {mae} != result {expected}")
    if day.isna().any().any() or len(day) != result["next_day_max"]["days"]:
        problems.append("next-day max file incomplete")
    return {"rows": int(len(full)), "rows_h4": int(len(h4)), "next_day_rows": int(len(day)),
            "first_origin": str(full.origin.min()), "last_target": str(full.target_time.max()),
            "auc_mae_from_csv": float(mae), "problems": problems, "ok": not problems}


def reproduce(stored: dict) -> dict:
    """Re-evaluate the locked model in a temporary directory; never touch the stored result."""
    from phase_f import final_fg_r11 as model
    keep = {name: (PRED_DIR / name).read_bytes() for name in
            ("final_test_fg_r11.csv", "final_test_fg_r11_h4.csv", "final_test_next_day_max.csv")
            if (PRED_DIR / name).exists()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("FINAL_LOCK.json", "fg_r6_models.joblib"):
            shutil.copy2(R11 / name, tmp / name)
        old_out, old_lock = model.OUT, model.LOCK
        model.OUT, model.LOCK = tmp, tmp / "FINAL_LOCK.json"
        try:
            model.stage_evaluate()
        finally:
            model.OUT, model.LOCK = old_out, old_lock
        again = json.loads((tmp / "FINAL_TEST_RESULT.json").read_text(encoding="utf-8"))
    diffs = {}
    for name, metrics in stored["point"].items():
        for key in ("AUC_MAE", "AUC_PeakMAE", "h4_MAE", "h16_MAE"):
            diffs[f"{name}|{key}"] = abs(metrics[key] - again["point"][name][key])
    worst = max(diffs.values())
    regenerated_same = all((PRED_DIR / n).read_bytes() == b for n, b in keep.items())
    return {"max_abs_metric_difference": worst, "reproduced": worst < 1e-6,
            "prediction_files_byte_identical": regenerated_same}


def main(refresh_caches: bool = False) -> dict:
    t0 = time.time()
    print("[fg-r11] caches", flush=True)
    status = {"caches": ensure_caches(refresh_caches)}
    from phase_f import final_fg_r11 as model
    if not (R11 / "FINAL_LOCK.json").exists():
        print("[fg-r11] lock (development data only)", flush=True)
        model.stage_lock()
    result_path = R11 / "FINAL_TEST_RESULT.json"
    if not result_path.exists():
        print("[fg-r11] one-time evaluation", flush=True)
        model.stage_evaluate()
        status["evaluation"] = "performed"
    else:
        print("[fg-r11] evaluation exists: reproduction check", flush=True)
        status["evaluation"] = "existing"
        status["reproduction"] = reproduce(json.loads(result_path.read_text(encoding="utf-8")))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print("[fg-r11] prediction CSV check", flush=True)
    status["prediction_csv"] = check_prediction_csv(result)
    print("[fg-r11] report blocks", flush=True)
    from phase_f.report_fg_r11 import write_blocks
    status["report_blocks"] = write_blocks(status)
    status["seconds"] = round(time.time() - t0, 1)
    (R11 / "PIPELINE_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    point = result["point"]["FG-R11"]
    print(json.dumps({"FG-R11 test": {k: round(point[k], 4) for k in ("AUC_MAE", "AUC_PeakMAE", "h4_MAE", "h16_MAE")},
                      "reproduction": status.get("reproduction"), "csv_ok": status["prediction_csv"]["ok"],
                      "seconds": status["seconds"]}, ensure_ascii=False, indent=2))
    if not status["prediction_csv"]["ok"] or (status.get("reproduction") and not status["reproduction"]["reproduced"]):
        raise SystemExit("FG-R11 pipeline check failed; see outputs/phase_f/final_fg_r11/PIPELINE_STATUS.json")
    return status


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-caches", action="store_true")
    main(ap.parse_args().refresh_caches)
