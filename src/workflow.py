"""Integrity, reproducibility and one-time holdout gates."""
from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_code_digest(src):
    """Freeze forecasting/scoring code, allowing later report-only maintenance."""
    src = Path(src)
    report_only = {"reporting.py", "model_reporting.py", "packaging.py", "viz.py"}
    paths = [p for p in sorted(src.rglob("*.py")) if p.name not in report_only]
    result = hashlib.sha256()
    for path in paths:
        result.update(path.relative_to(src).as_posix().encode("utf-8"))
        result.update(path.read_bytes())
    return result.hexdigest()


def evidence_digest(path):
    path = Path(path)
    data = path.read_bytes()
    if "verification" in path.parts and path.suffix in {".py", ".md", ".json", ".csv", ".txt"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def immutable_evidence(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): evidence_digest(p)
            for folder in ("verification", "data/raw/task05_power")
            for p in sorted((root/folder).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def check_evidence(root):
    root = Path(root)
    current = immutable_evidence(root)
    expected_csv = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
    if current.get("data/raw/task05_power/okm_augumented_2021.csv") != expected_csv:
        raise RuntimeError("Raw CSV differs from independently archived source digest")
    pinned = root/"configs/verification_manifest.json"
    if pinned.exists():
        expected = json.loads(pinned.read_text(encoding="utf-8"))
        if any(current.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Verification evidence differs from pinned base-commit manifest")
    path = root/"outputs/logs/evidence_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != current:
            changed = sorted(k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k))
            raise RuntimeError(f"Frozen/raw evidence changed: {changed}")
    else:
        json_write(path, current)
    return current


def fingerprint(root, *, development_only=False):
    root = Path(root)
    files = [root/"requirements.txt", root/"eval_protocol.md",
             root/"outputs/logs/adoption_criteria.md"]
    # Preregistered candidate identities and the P1 feature gate are scientific
    # inputs too. Editing either must invalidate a previously sealed CV cache.
    files += sorted((root/"outputs/logs").glob("preregistration_0925_P*.md"))
    files += [root/"outputs/logs/feature_spec_P2.md", root/"outputs/logs/feature_spec_P2.json"]
    if development_only:
        # New scientific modules must invalidate the old OOF cache too.  The
        # previous hand-maintained list omitted session_data and Phase 2 code.
        report_only = {"reporting.py", "model_reporting.py", "packaging.py", "viz.py"}
        files += [root/"run_all.py"] + [p for p in sorted((root/"src").rglob("*.py"))
                                        if p.name not in report_only]
        files += [root/"configs/default.yaml"]
    else:
        files += [root/"run_all.py"] + sorted((root/"src").rglob("*.py")) + sorted((root/"configs").glob("*.yaml"))
    files += sorted((root/"data/raw/task05_power").glob("*.csv"))
    return {p.relative_to(root).as_posix(): digest(p) for p in files if p.exists()}


def final_gate(cfg, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Freeze clock must include a timezone")
    deadline = datetime.fromisoformat(cfg["freeze"]["not_before"])
    return now >= deadline


def development_artifacts(outdir):
    """Hash all reusable development outputs, including importance model bundles."""
    out = Path(outdir)
    names = ["predictions/development_oof.csv", "tables/development_cv.csv",
             "logs/development_selection.json", "predictions/t2_development_oof.csv",
             "tables/t2_development_cv.csv"]
    paths = [out/name for name in names]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("Development outputs are incomplete")
    paths += sorted((out/"models").glob("development_*.joblib"))
    paths += sorted((out/"models/p2_adaptive").glob("*.joblib"))
    if len(paths) == len(names):
        raise RuntimeError("Development model bundles are missing")
    return {path.relative_to(out).as_posix(): digest(path) for path in paths}


def verify_development_cache(outdir, expected_fingerprint):
    path = Path(outdir)/"logs/development_cache.json"
    if not path.is_file():
        return False
    cache = json.loads(path.read_text(encoding="utf-8"))
    if cache.get("fingerprint") != expected_fingerprint or not cache.get("artifacts"):
        return False
    try:
        return development_artifacts(outdir) == cache["artifacts"]
    except RuntimeError:
        return False


def freeze_readiness(root, cfg, outdir, expected_fingerprint=None, *, allow_completed=False):
    """Verify development provenance without parsing any test observation."""
    from .session_data import SEALED_BOUNDARY

    root, out = Path(root).resolve(), Path(outdir)
    config_path = root/"configs/default.yaml"
    stored_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg != stored_cfg:
        raise RuntimeError("Freeze configuration differs from the pinned default YAML")
    if pd_timestamp(cfg["split"]["test_start_origin"]) != SEALED_BOUNDARY:
        raise RuntimeError("The configured test origin differs from the sealed boundary")
    source_path = root/cfg["source"]
    if not source_path.is_file():
        raise FileNotFoundError("Pinned freeze source is missing")
    expected = expected_fingerprint or fingerprint(root, development_only=True)
    if not verify_development_cache(out, expected):
        raise RuntimeError("Development cache is missing, stale, or incomplete")
    selection = out/"logs/development_selection.json"
    cache = out/"logs/development_cache.json"
    manifest = json.loads(selection.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("selection"), dict) or not manifest.get("folds"):
        raise RuntimeError("Development selection manifest is incomplete")
    if any(choice.get("adaptive_q95") is not None for choice in
           manifest["selection"].get("by_horizon", {}).values()):
        raise RuntimeError("Unadopted adaptive q95 cannot enter the prepared final path")
    lock = out/"logs/freeze_record.json"
    completed = False
    if lock.exists():
        record = json.loads(lock.read_text(encoding="utf-8"))
        completed = allow_completed and record.get("status") == "completed"
        if not completed:
            raise RuntimeError("Freeze reservation already exists")
    final_names = ("tables/final_test.csv", "predictions/final_test_predictions.csv",
                   "tables/t2_final_test.csv", "predictions/t2_final_test_predictions.csv")
    if not completed and any((out/name).exists() for name in final_names):
        raise RuntimeError("Holdout artifacts exist without a completed freeze")
    return {"status": "completed" if completed else "ready_for_human_approval", "holdout_read": False,
            "development_cache_sha256": digest(cache),
            "selection_sha256": digest(selection),
            "config_sha256": digest(config_path),
            "source_sha256": digest(source_path),
            "fingerprint_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode("utf-8")).hexdigest(),
            "test_start_origin": str(SEALED_BOUNDARY)}


def pd_timestamp(value):
    """Local import keeps the ordinary hash helpers free of pandas startup."""
    import pandas as pd
    return pd.Timestamp(value)


def require_freeze_approval(root, cfg, outdir, expected_fingerprint=None):
    """Require a named, hash-bound human decision before any holdout loader."""
    readiness = freeze_readiness(root, cfg, outdir, expected_fingerprint, allow_completed=True)
    path = Path(outdir)/"logs/human_freeze_approval.json"
    if not path.is_file():
        raise RuntimeError("Explicit human freeze approval is required")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("decision") != "approve_freeze" or not str(decision.get("approved_by", "")).strip():
        raise RuntimeError("A named human freeze approval is required")
    approved_at = datetime.fromisoformat(str(decision.get("approved_at_kst", "")))
    if approved_at.tzinfo is None or approved_at > datetime.now(timezone.utc):
        raise RuntimeError("Freeze approval timestamp must be timezone-aware and not future-dated")
    for name in ("development_cache_sha256", "selection_sha256", "config_sha256",
                 "source_sha256", "fingerprint_sha256"):
        if decision.get(name) != readiness[name]:
            raise RuntimeError(f"Freeze approval is stale: {name}")
    return readiness


def record_run_evidence(root, outdir, status):
    """Keep dated machine status separate from the written historical progress log."""
    root, out = Path(root), Path(outdir)
    json_write(out/"logs/run_status.json", status)
    if "development" in status.get("steps", {}) and status.get("status", "").startswith("completed"):
        json_write(out/"logs/full_run_status.json", status)
        path = out/"tables/T6-1_run_steps.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "seconds", "state"])
            writer.writeheader()
            writer.writerows({"step": name, "seconds": item["seconds"], "state": status["status"]}
                             for name, item in status["steps"].items())
    progress = root/"PROGRESS.md"
    body = progress.read_text(encoding="utf-8") if progress.exists() else "# 진행 기록\n"
    marker = "<!-- AUTO_EXECUTION_STATUS -->"
    # Append only. Earlier session entries can follow the first auto marker.
    body = body.rstrip()
    freeze = out/"logs/freeze_record.json"
    frozen = json.loads(freeze.read_text(encoding="utf-8")) if freeze.exists() else {}
    current = "동결 평가 완료 · 파일 해시 검증 기록 참조" if frozen.get("status") == "completed" else "최종 테스트 미평가 · 날짜/예약 잠금 유지"
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    progress.write_text(body + f"\n\n{marker}\n## 자동 실행 상태\n\n갱신: {stamp}\n\n"
        + f"- 최근 실행: `{status.get('status')}`; 실행 단계: {', '.join(status.get('steps', {}))}.\n"
        + f"- {current}.\n- [최신 실행 기록](outputs/logs/run_status.json) · [전체 실행 기록](outputs/logs/full_run_status.json).\n",
        encoding="utf-8")


def audit_historical_predictions(root):
    """Recompute archived prediction metrics, without retraining or reopening targets."""
    import numpy as np
    import pandas as pd
    root = Path(root)
    archived = pd.read_csv(root/"verification/results/05b_predictions.csv")
    summary = json.loads((root/"verification/results/05b_summary.json").read_text(encoding="utf-8"))
    threshold = summary["peak_threshold"]
    mask = archived.y > threshold
    result = {name: float(np.mean(np.abs(archived.loc[mask, "y"]-archived.loc[mask, name])))
              for name in ("persistence", "lgb")}
    if round(result["persistence"], 2) != 27.73 or round(result["lgb"], 2) != 17.77:
        raise AssertionError("Historical metrics do not match archived evidence")
    result.update(status="verified_from_frozen_predictions", new_test_evaluation=False,
                  note="과거 저장 예측의 산술 재현이며 재학습 또는 신규 모델의 테스트 성능이 아님")
    json_write(root/"outputs/logs/historical_reproduction.json", result)
    return result
