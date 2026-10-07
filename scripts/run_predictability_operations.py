"""Run preregistered P1 A4/A5 on sealed development inputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.analysis.predictability import detect_restart_events, fit_restart_thresholds
from src.analysis.predictability_operations import run_a4, run_a5
from src.session_data import load_development_history, load_development_oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("A4", "A5", "both"), default="both")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/analysis_p1")
    args = parser.parse_args()
    history = load_development_history(ROOT)
    oof = load_development_oof(ROOT)
    manifest = json.loads((ROOT / "outputs/logs/development_selection.json").read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.analysis in ("A4", "both"):
        result = run_a4(history, oof, manifest,
                        (fit_restart_thresholds, detect_restart_events), args.outdir)
        summary["A4"] = result["hypotheses"].to_dict("records")
    if args.analysis in ("A5", "both"):
        tariff = yaml.safe_load((ROOT / "configs/tariff_2026.yaml").read_text(encoding="utf-8"))
        result = run_a5(history, oof, manifest, tariff, args.outdir)
        summary["A5"] = result["summary"].to_dict("records")
    (args.outdir / "A4_A5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in summary.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
