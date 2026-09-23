"""Reproducible development or frozen-analysis archive; never a blind submission."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import zipfile

from .workflow import digest, json_write, model_code_digest, fingerprint, verify_development_cache

FINAL = {
    "model_sha256": "outputs/models/frozen_final.joblib",
    "holdout_prediction_sha256": "outputs/predictions/final_test_predictions.csv",
    "final_metrics_sha256": "outputs/tables/final_test.csv",
    "t2_predictions_sha256": "outputs/predictions/t2_final_test_predictions.csv",
    "t2_metrics_sha256": "outputs/tables/t2_final_test.csv",
}
TRANSIENT = {"development_cache.json", "evidence_manifest.json", "run_status.json",
             "freeze_record.json", "blind_scan.json"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _freeze(root: Path, cfg: dict) -> dict | None:
    path = root / "outputs/logs/freeze_record.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError("Incomplete freeze reservation must be preserved; refuse an archive that would remove its one-time lock")
    if datetime.now(timezone.utc) < datetime.fromisoformat(cfg["freeze"]["not_before"]):
        raise RuntimeError("Completed freeze exists before the authorized date")
    if record.get("test_start_origin") != cfg["split"]["test_start_origin"]:
        raise RuntimeError("Frozen test boundary differs from configuration")
    for key, relative in FINAL.items():
        artifact = root / relative
        if not artifact.is_file() or digest(artifact) != record.get(key):
            raise RuntimeError(f"Frozen artifact missing or hash mismatch: {relative}")
    src = root / "src"
    selection = root / "outputs/logs/development_selection.json"
    source = root / cfg["source"]
    processed = root / "data/processed/power_15min.parquet"
    expected = {"code_sha256": model_code_digest(src),
                "config_sha256": _sha(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")),
                "selection_sha256": digest(selection) if selection.is_file() else None,
                "source_sha256": digest(source) if source.is_file() else None,
                "processed_sha256": digest(processed) if processed.is_file() else None}
    if record.get("fingerprints") != expected:
        raise RuntimeError("Freeze fingerprints differ from current code, data, or selection")
    return record


def _gather(root: Path, frozen: bool) -> list[Path]:
    paths = []
    for name in ("README.md", "CLAUDE.md", "PROJECT_DESIGN.md", "PROGRESS.md",
                 "DECISIONS.md", "eval_protocol.md", "requirements.txt",
                 "requirements-pipeline.lock.txt", "requirements-artifacts.txt", "run_all.py"):
        path = root / name
        if path.is_file():
            paths.append(path)
    for folder in ("src", "tests", "configs", "report", "slides", "scripts"):
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts
                     and (p.suffix in {".py", ".md", ".json", ".yaml"}
                          or (folder == "slides" and p.suffix in {".html", ".pdf"})
                          or (folder == "scripts" and p.suffix == ".ps1")))
    paths.extend((root / "data/raw/task05_power").glob("*.csv"))
    paths.extend((root / "data/raw/task05_power").glob("*.zip"))
    for name in ("docs/roadmap.html", "docs/INDEX.md", "docs/DELIVERABLES.md", "docs/SOURCES.md", "docs/tariff_sources.md",
                 "docs/organizer_inquiry.md", "submission/HANDOFF.md"):
        path = root / name
        if path.is_file():
            paths.append(path)
    paths.extend(p for p in (root / "verification").rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts)
    for folder in ("figures", "tables", "predictions", "logs"):
        paths.extend(p for p in (root / "outputs" / folder).glob("*") if p.is_file()
                     and p.suffix in {".png", ".csv", ".json", ".md"}
                     and p.name not in TRANSIENT
                     and (frozen or not p.name.startswith(("final_test", "t2_final_test", "final_analysis"))))
    if frozen:
        paths.extend(root / relative for relative in FINAL.values())
        paths.append(root / "outputs/logs/freeze_record.json")
        paths.append(root / "data/processed/power_15min.parquet")
        paths.append(root / "outputs/logs/development_cache.json")
        paths.extend((root / "outputs/models").glob("development_*.joblib"))
        paths.extend(p for p in (root / "outputs/final_analysis").rglob("*")
                     if p.is_file() and p.suffix in {".png", ".csv", ".json"})
    return sorted(set(paths))


def _scan(root: Path, paths: list[Path]) -> dict:
    findings, historical_paths = [], []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if path.suffix in {".py", ".md", ".json", ".yaml", ".html", ".ps1"} and not rel.startswith("verification/"):
            body = path.read_text(encoding="utf-8-sig")
            body = body.replace("\\\\", "\\")
            for pattern in (r"[A-Za-z]:[\\/]Users[\\/][^\s\"']+", r"[A-Za-z]:[\\/]Project[\\/]",
                            r"gh[pousr]_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{30,}"):
                if re.search(pattern, body):
                    findings.append({"file": rel, "issue": "private_path_or_credential_pattern"})
        if "verification" in path.relative_to(root).parts and path.suffix in {".py", ".md", ".json"}:
            if re.search(r"[A-Za-z]:[\\/]", path.read_text(encoding="utf-8-sig")):
                historical_paths.append(rel)
    return {"status": "review_required" if findings or historical_paths else "automated_scan_passed",
            "findings": findings, "historical_provenance_paths": historical_paths,
            "note": "Source archive preserves historical evidence; blind submission and document metadata review remain separate."}


def build_package(root, cfg):
    root = Path(root).resolve()
    record = _freeze(root, cfg)
    frozen = record is not None
    if frozen and not verify_development_cache(root / cfg.get("output_dir", "outputs"),
                                               fingerprint(root, development_only=True)):
        raise RuntimeError("Frozen archive requires a verified development cache and model bundles")
    paths = _gather(root, frozen)
    scan = _scan(root, paths)
    scan_path = root / "outputs/logs/blind_scan.json"
    json_write(scan_path, scan)
    paths = sorted(set(paths + [scan_path]))
    state = "frozen_reproduction" if frozen else "development_reproduction_only"
    metadata = {"status": state,
                "final_test": "evaluated_once" if frozen else f"locked_until_{cfg['freeze']['not_before']}",
                "freeze_artifact_sha256": {key: record[key] for key in FINAL} if frozen else None,
                "survey_capture": "missing", "portal_submission": "not_performed",
                "blind_scan": scan,
                "run_status": "packaged_stage_snapshot; rerun python run_all.py for authoritative completion status"}
    current_status = root / "outputs/logs/run_status.json"
    run_snapshot = {"status": "packaged_stage_snapshot", "archive_state": state,
                    "note": "Archive is written before run_all records its final status. Rerun the pipeline for completion evidence.",
                    "source_run_status_sha256": digest(current_status) if current_status.is_file() else None}
    filename = "task05_frozen_reproduction.zip" if frozen else "task05_development_reproduction.zip"
    target = root / "submission/source" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("PACKAGE_STATUS.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("outputs/logs/run_status.json", json.dumps(run_snapshot, ensure_ascii=False, indent=2))
    info = {"path": target.relative_to(root).as_posix(), "sha256": digest(target),
            "files": len(paths) + 2, "bytes": target.stat().st_size, "status": state,
            "blind_scan": scan["status"]}
    json_write(root / "submission/source/package_manifest.json", info)
    return info
