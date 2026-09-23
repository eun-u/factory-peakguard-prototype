"""Shared plotting defaults and small, reproducible Korean-language figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure() -> None:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "AppleGothic", "NanumGothic") if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font,
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
    })


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
