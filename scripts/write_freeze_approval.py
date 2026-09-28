"""Record the named human freeze approval bound to the current development hashes.

Run this yourself after reviewing `python run_all.py --dry-run-freeze`:

    python scripts/write_freeze_approval.py --approved-by "이름"

It never reads holdout data. It refuses when the development cache is stale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.workflow import fingerprint, freeze_readiness, json_write  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-by", required=True, help="Name of the person approving the freeze")
    args = parser.parse_args()
    if not args.approved_by.strip():
        raise SystemExit("A named approver is required")
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    out = ROOT / cfg.get("output_dir", "outputs")
    ready = freeze_readiness(ROOT, cfg, out, fingerprint(ROOT, development_only=True))
    if ready.get("status") != "ready_for_human_approval" or ready.get("holdout_read") is not False:
        raise SystemExit(f"Not ready for approval: {ready}")
    kst = timezone(timedelta(hours=9))
    record = {"decision": "approve_freeze", "approved_by": args.approved_by.strip(),
              "approved_at_kst": datetime.now(kst).isoformat(timespec="seconds"),
              **{key: ready[key] for key in ("development_cache_sha256", "selection_sha256",
                                             "config_sha256", "source_sha256", "fingerprint_sha256")}}
    json_write(out / "logs/human_freeze_approval.json", record)
    print(record)


if __name__ == "__main__":
    main()
