"""predict_ahead_sec(item) = travel + descent + close_wait — 예측 시점 계산.

v41/v44에서 정의한 "예측 시점 재정의"(예측 시간 = 이동+하강+닫힘)를
그대로 구현한다. 값은 `config/motion_timing.yaml`과
`config/gripper_profiles.yaml`에서 읽는다 — 하드코딩하지 않는다
(CLAUDE.md 외부화 원칙).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_motion_timing(path: Path | None = None) -> dict:
    path = path or (_CONFIG_DIR / "motion_timing.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["motion_timing"]


def load_gripper_profiles(path: Path | None = None) -> dict:
    path = path or (_CONFIG_DIR / "gripper_profiles.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["gripper"]


def predict_ahead_sec(item_key: str, motion_timing: dict | None = None, gripper_profiles: dict | None = None) -> float:
    """Seconds from "now" (detection time) to "gripper fully closed"."""
    motion_timing = motion_timing or load_motion_timing()
    gripper_profiles = gripper_profiles or load_gripper_profiles()

    if item_key not in gripper_profiles:
        raise KeyError(f"unknown item_key {item_key!r}, not in gripper_profiles.yaml")

    close_wait = gripper_profiles[item_key]["close_wait_sec"]
    return motion_timing["travel_time_sec"] + motion_timing["descent_time_sec"] + close_wait
