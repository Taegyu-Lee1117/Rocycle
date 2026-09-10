"""load_item_routing() -- config/item_routing.yaml 로더 + bin_teach_points 조회.

상태머신 골격(6일차) 전용. 원칙3(외부화) 준수 -- 품목명에 대한
분기를 코드에 두지 않고 이 표를 조회해서 동작한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)


def _resolve_config_dir() -> Path:
    try:
        return Path(get_package_share_directory("rocycle_robot")) / "config"
    except PackageNotFoundError:
        return Path(__file__).resolve().parents[1] / "config"


_CONFIG_DIR = _resolve_config_dir()


def load_item_routing(path: Path | None = None) -> dict:
    path = path or (_CONFIG_DIR / "item_routing.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _load_bin_teach_points():
    """bin_teach_points.py는 일반 .py 모듈이라 importlib로 직접 로드."""
    path = _CONFIG_DIR / "bin_teach_points.py"
    spec = importlib.util.spec_from_file_location("bin_teach_points", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_bin_pose(bin_name: str, which: str) -> list[float]:
    """which: 'hover' or 'place'. Returns [x, y, z, rx, ry, rz]."""
    module = _load_bin_teach_points()
    points = module.BIN_HOVER_POINTS if which == "hover" else module.BIN_PLACE_POINTS
    for _sticker, design_name, x, y, z, rx, ry, rz in points:
        if design_name == bin_name:
            return [x, y, z, rx, ry, rz]
    raise KeyError(f"bin {bin_name!r} not found in bin_teach_points.py ({which})")
