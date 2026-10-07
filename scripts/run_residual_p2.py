"""Run preregistered M1 residual candidates on sealed development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.research_cv import fit_residual_candidates
from src.session_data import load_development_history, load_development_oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT/"configs/default.yaml").read_text(encoding="utf-8"))
    history = load_development_history(ROOT)
    original_oof = load_development_oof(ROOT)
    manifest = json.loads((ROOT/"outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    result = fit_residual_candidates(history, cfg, manifest, original_oof,
                                     output_dir=args.output_dir)
    print(json.dumps({"status": "completed", "fits": len(result["fit_metadata"]),
                      "candidate_rows": len(result["predictions"]),
                      "baseline_exclusion_rows": len(result["audit"]),
                      "hypotheses": len(result["comparisons"]),
                      "paths": result["paths"]}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
