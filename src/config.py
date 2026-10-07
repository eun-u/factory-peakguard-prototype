"""설정 파일 읽기와 출력 폴더 준비."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def output_dirs(cfg: dict) -> dict[str, Path]:
    base = resolve(cfg["output_dir"])
    dirs = {name: base / name for name in ("figures", "tables", "predictions", "logs")}
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    dirs["base"] = base
    return dirs
