"""Replay registered P2 operating evidence against the integrated P3 OOF."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    original = ROOT / "outputs/analysis_p2"
    destination = ROOT / "_validation/session_0925_operational_repro"
    analysis = destination / "analysis_p2"
    analysis.mkdir(parents=True, exist_ok=True)
    for name in ("M1_comparisons.csv", "M4_hypotheses.csv"):
        shutil.copyfile(original / name, analysis / name)
    paths = [original / "P2_auxiliary_metrics.csv", original / "P2_hypotheses_holm.csv",
             *sorted((original / "operations").glob("A5_*.csv"))]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    checksums = {str(path): digest(path) for path in paths}
    subprocess.run([sys.executable, "-X", "utf8", "scripts/finalize_models_p2.py",
                    "--output-dir", str(destination),
                    "--manifest", "outputs/logs/development_selection.json",
                    "--predictions", "outputs/predictions/development_oof.csv"], cwd=ROOT, check=True)
    evidence = []
    for path in paths:
        fresh_path = analysis / path.relative_to(original)
        before, fresh = pd.read_csv(path), pd.read_csv(fresh_path)
        pd.testing.assert_frame_equal(before, fresh, check_exact=False, atol=1e-10, rtol=0)
        numeric = before.select_dtypes(include="number").columns
        differences = np.abs(before[numeric].to_numpy(float) - fresh[numeric].to_numpy(float))
        finite = differences[np.isfinite(differences)]
        maximum = float(finite.max()) if finite.size else 0.
        assert checksums[str(path)] == digest(path), f"Original evidence changed: {path}"
        evidence.append({"source": path.relative_to(ROOT).as_posix(), "rows": len(before),
                         "max_abs_difference": maximum, "sha256": checksums[str(path)]})
    result = {"status": "pass", "scope": "sealed_development_oof_only",
              "absolute_tolerance": 1e-10, "holdout_read": False,
              "source_unchanged": True, "retrospective_ridge_fits": 6,
              "forecast_model_fits": 0, "seconds": time.perf_counter()-started, "tables": evidence}
    (ROOT / "outputs/logs/P3_operations_repro.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "tables": len(evidence),
                      "max_abs_difference": max(row["max_abs_difference"] for row in evidence)}))


if __name__ == "__main__":
    main()
