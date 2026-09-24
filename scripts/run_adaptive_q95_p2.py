"""Reproduce preregistered Phase 2 M4 candidates on sealed development data."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adaptive_q95 import fit_adaptive_candidates
from src.session_data import load_development_history, load_development_oof


def run_adaptive_q95_p2(root: str | Path):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    history = load_development_history(root)
    original = load_development_oof(root)
    manifest = json.loads((root / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    return fit_adaptive_candidates(history, cfg, manifest, original, root / "outputs")


if __name__ == "__main__":
    result = run_adaptive_q95_p2(ROOT)
    print({"candidate_rows": len(result["predictions"]),
           "hypotheses": len(result["hypotheses"]),
           "quantile_fits": result["model_fit_count"]})
