"""One-command reproduction: python verification/run_all.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import build_inventory
from report import make_report


HERE = Path(__file__).resolve().parent


def main():
    build_inventory()
    for script in ("t3_press.py", "t5_power.py"):
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE.parent)
    print(make_report())
    build_inventory()


if __name__ == "__main__":
    main()
