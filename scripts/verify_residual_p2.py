"""Verify M1 uses exactly the frozen CBL representative at every OOF row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.session_data import load_development_oof, read_development_oof


def main() -> None:
    original_path = ROOT/"outputs/predictions/development_oof.csv"
    manifest_path = ROOT/"outputs/logs/development_selection.json"
    candidate_path = ROOT/"outputs/predictions/p2_residual_oof.csv"
    original = load_development_oof(ROOT)
    candidate = read_development_oof(candidate_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = ["origin", "target_time", "horizon", "fold"]
    comparisons = []
    for horizon in (4, 16, 96):
        baseline_name = manifest["selection"]["by_horizon"][str(horizon)]["cbl"]
        left = candidate.loc[candidate.horizon.eq(horizon)]
        right = original.loc[original.horizon.eq(horizon) & original.model.eq(baseline_name),
                             keys+["pred"]]
        if right.duplicated(keys).any():
            raise AssertionError("Original CBL OOF has duplicate keys")
        joined = left.merge(right, on=keys, how="left", suffixes=("", "_original"),
                            validate="one_to_one")
        if len(joined) != len(left) or joined.pred_original.isna().any():
            raise AssertionError("M1 baseline does not align with original CBL OOF")
        difference = np.abs(joined.residual_baseline.to_numpy(float)-joined.pred_original.to_numpy(float))
        maximum = float(difference.max()) if len(difference) else np.nan
        if not np.isfinite(maximum) or maximum > 1e-10:
            raise AssertionError(f"M1 h{horizon} baseline changed from original CBL: {maximum}")
        comparisons.append({"horizon": horizon, "baseline_model": baseline_name,
                            "rows": len(joined), "max_abs_difference": maximum})
    result = {"status": "passed", "comparisons": comparisons,
              "original_oof_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
              "original_selection_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "candidate_oof_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest()}
    path = ROOT/"outputs/analysis_p2/M1_parity.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
