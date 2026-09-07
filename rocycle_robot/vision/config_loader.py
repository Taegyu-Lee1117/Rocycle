"""load_vision_config() -- config/vision.yaml 로더. 원칙3(외부화) 준수."""

from __future__ import annotations

from pathlib import Path

import yaml

_PKG_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PKG_ROOT / "config"
_MODELS_DIR = _PKG_ROOT / "models"


def load_vision_config(path: Path | None = None) -> dict:
    path = path or (_CONFIG_DIR / "vision.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["vision"]


def resolve_model_path(model_path: str) -> str:
    """vision.yaml의 model_path(패키지 상대 경로)를 절대경로로 변환."""
    p = Path(model_path)
    if p.is_absolute():
        return str(p)
    return str(_PKG_ROOT / p)
