"""Verify an isolated cold run and optionally promote its audited development cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.workflow import (digest, fingerprint, development_artifacts,
                          verify_development_cache, json_write, check_evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    if candidate == ROOT or not candidate.is_relative_to(ROOT/"_validation"):
        raise ValueError("Candidate must be an isolated directory below this workspace's _validation")
    out = candidate/"outputs"
    status = json.loads((out/"logs/full_run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed_development" or list(status["steps"]) != [
            "data", "development", "final", "analysis", "report", "package"]:
        raise AssertionError("The isolated full development run has not completed")
    if status["steps"]["development"].get("detail", {}).get("cache") == "verified":
        raise AssertionError("Expected a cold development run, not cached reuse")
    expected = fingerprint(ROOT, development_only=True)
    if expected != fingerprint(candidate, development_only=True):
        raise AssertionError("Scientific source/config/source-data fingerprints differ")
    if not verify_development_cache(out, expected):
        raise AssertionError("Isolated artifact cache failed integrity validation")
    check_evidence(candidate)
    compared = []
    for relative in ("predictions/development_oof.csv", "tables/development_cv.csv",
                     "predictions/t2_development_oof.csv", "tables/t2_development_cv.csv"):
        left = pd.read_csv(ROOT/"outputs"/relative, low_memory=False)
        right = pd.read_csv(out/relative, low_memory=False)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False,
                                      rtol=1e-10, atol=1e-10)
        compared.append({"file": relative, "rows": len(right), "numerically_identical": True,
                         "cold_sha256": digest(out/relative)})
    selection = "logs/development_selection.json"
    if json.loads((ROOT/"outputs"/selection).read_text(encoding="utf-8")) != json.loads(
            (out/selection).read_text(encoding="utf-8")):
        raise AssertionError("Independent model selection differs")
    artifacts = development_artifacts(out)
    if args.promote:
        for relative in artifacts:
            source, target = (out/relative).resolve(), (ROOT/"outputs"/relative).resolve()
            if not source.is_relative_to(out) or not target.is_relative_to(ROOT/"outputs"):
                raise ValueError("Artifact path escapes the owned output directories")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(out/"logs/development_cache.json", ROOT/"outputs/logs/development_cache.json")
        if not verify_development_cache(ROOT/"outputs", expected):
            raise AssertionError("Promoted artifact validation failed")
    info = {"status": "verified_cold_reproduction", "command": "python run_all.py",
            "isolated_directory": candidate.relative_to(ROOT).as_posix(),
            "source_fingerprint": expected, "elapsed_seconds": status.get("seconds"),
            "stage_seconds": {name: value["seconds"] for name, value in status["steps"].items()},
            "prediction_and_metric_comparison": compared, "selection_identical": True,
            "test_opened": False, "promoted": args.promote,
            "numerical_tolerance": {"rtol": 1e-10, "atol": 1e-10},
            "core_environment": "requirements-pipeline.lock.txt",
            "note": "Report/layout-only maintenance is validated separately; forecasting source and results match."}
    json_write(ROOT/"outputs/logs/fresh_reproduction.json", info)
    pd.DataFrame([{"step": name, "seconds": value["seconds"], "cache": "cold_run"}
                  for name, value in status["steps"].items()]).to_csv(
        ROOT/"outputs/tables/T6-1_cold_run_steps.csv", index=False)
    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith(str(candidate)):
            return Path(value).relative_to(candidate).as_posix()
        return value
    json_write(ROOT/"outputs/logs/cold_run_status.json", portable(status))
    print(json.dumps({k: info[k] for k in ("status", "elapsed_seconds", "promoted")}, indent=2))


if __name__ == "__main__":
    main()
