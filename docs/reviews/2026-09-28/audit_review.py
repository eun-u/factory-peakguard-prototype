"""Recheck the review's evidence without training models or decoding holdout data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/reviews/2026-09-28"
KB = Path("C:/Project/피지컬AI시스템설계/knowledge/books/physical_ai_system_design")
sys.path.insert(0, str(ROOT))
from src.workflow import check_evidence, fingerprint, verify_development_cache


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = check_evidence(ROOT)
    science = fingerprint(ROOT, development_only=True)
    if not verify_development_cache(ROOT / "outputs", science):
        raise RuntimeError("Review verification requires a valid cache; no automatic retraining")
    files = ["README.md", "PROGRESS.md", "DECISIONS.md", "CLAUDE.md", "run_all.py",
             "prototype/app.py", "prototype/forecast.py", "src/session_data.py", "src/workflow.py",
             "src/analysis/shift_v2.py", "src/analysis/alerting.py", "src/analysis/decision.py",
             "outputs/logs/session_0925_summary.md", "outputs/logs/run_status.json",
             "outputs/logs/development_selection.json", "outputs/logs/analysis_status.json",
             "outputs/logs/session_0925_repro.json", "outputs/logs/P3_operations_repro.json",
             "outputs/logs/scheduled_freeze.json", "outputs/analysis_p2/M1_comparisons.csv",
             "outputs/analysis_p2/M1_fold_comparison.csv", "outputs/analysis_p2/P2_candidate_decisions.csv",
             "outputs/analysis_p2/P2_selection_before_after.csv", "outputs/analysis_p2/operations/A5_summary.csv",
             "outputs/analysis_p2/P2_auxiliary_metrics.csv", "report/ch2_model.md", "report/ch4_field.md"]
    knowledge = ["QA_REPORT.md", "PROCESSING_STATE.md", "chapters/chapter_03.md",
                 "chapters/chapter_07.md", "chapters/chapter_08.md", "chapters/chapter_09.md",
                 "chapters/chapter_10.md", "chapters/chapter_14.md", "chapters/chapter_15.md",
                 "pages/page_0019.md", "pages/page_0023.md", "pages/page_0066.md",
                 "pages/page_0078.md", "pages/page_0106.md", "pages/page_0111.md", "pages/page_0113.md"]
    manifest = {
        "review_date": "2026-09-28", "captured_at": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "service_sources": {p: sha(ROOT / p) for p in files},
        "domain_source_root": str(KB), "domain_sources": {p: sha(KB / p) for p in knowledge},
        "prior_training_record": json.loads((ROOT / "outputs/logs/run_status.json").read_text(encoding="utf-8")),
        "source_values_decoded": False,
    }
    write("source_manifest.json", manifest)
    results = {}
    for name, args in [("pytest", ["-m", "pytest", "tests", "-q"]),
                       ("freeze_dry_run", ["run_all.py", "--dry-run-freeze"])]:
        result = subprocess.run([sys.executable, "-X", "utf8", *args], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8")
        (OUT / f"{name}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        results[name] = {"command": "python " + " ".join(args), "direct_exit_code": result.returncode}
        if name == "pytest":
            count = re.search(r"(\d+) passed", result.stdout)
            results[name]["passed"] = int(count.group(1)) if count else None
        if result.returncode != 0:
            write("validation.json", results)
            raise SystemExit(result.returncode)
    task = subprocess.run(["powershell", "-NoProfile", "-Command",
                           "(Get-ScheduledTask -TaskName 'FactoryPeakguard-Task05-Freeze-20261001' -ErrorAction Stop).State.ToString()"],
                          capture_output=True, text=True)
    results["scheduled_task"] = {"exit_code": task.returncode,
                                 "state": task.stdout.strip() if task.returncode == 0 else "unverified"}
    readiness = json.loads((ROOT / "outputs/logs/freeze_dry_run.json").read_text(encoding="utf-8"))
    results.update({"checked_at": datetime.now().astimezone().isoformat(), "freeze_readiness": readiness,
                    "development_cache_valid": verify_development_cache(ROOT / "outputs", science),
                    "protected_evidence_unchanged": protected == check_evidence(ROOT),
                    "scientific_fingerprint_unchanged": science == fingerprint(ROOT, development_only=True),
                    "full_training_run": False, "prototype_full_data_tests_run": False,
                    "holdout_decoded": False})
    assert results["development_cache_valid"] and results["protected_evidence_unchanged"]
    assert results["scientific_fingerprint_unchanged"] and readiness["holdout_read"] is False
    write("validation.json", results)
    print(json.dumps({k: v for k, v in results.items() if k != "freeze_readiness"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
