"""Intent -> ROS2 action mapping (설계문서 v31 4-2절).

Kept separate from IntentClassifier so it's testable without hitting the
OpenAI API: this module is pure data + a dispatch function.

Node wiring is a stub for now (log only) -- actual /conveyor/stop etc. calls
land here once state_machine_node exists (작업 순서 E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandAction:
    intent: str
    ros_call: str  # descriptive, not yet a live call -- see class docstring
    tts_response: str | None  # None = no spoken reply for this intent


# 설계문서 v25 3-1a-1절: 속도 고정 안내 문구는 팀결정 사항, 그대로 재사용.
_SPEED_FIXED_RESPONSE = "현재 고정 속도로 운영 중입니다"

COMMAND_TABLE: dict[str, CommandAction] = {
    "start": CommandAction("start", "state_machine_node: IDLE -> TRACKING", None),
    "pause": CommandAction("pause", "conveyor_node: /conveyor/stop", None),
    "resume": CommandAction("resume", "state_machine_node: resume approval -> /conveyor/set_speed", None),
    "stop": CommandAction("stop", "state_machine_node: begin 6-4절 종료 시퀀스", None),
    "speed_inquiry": CommandAction("speed_inquiry", "(no ROS call)", _SPEED_FIXED_RESPONSE),
    "handoff_ack": CommandAction("handoff_ack", "8절 ASSIST: release_force trigger ack", None),
    "unknown": CommandAction("unknown", "(no ROS call)", None),
}


def dispatch(intent: str, log_fn: Callable[[str], None] = print) -> CommandAction:
    """Look up the action for a classified intent and log it (stub for the
    real ROS2 call, which needs state_machine_node to exist)."""
    action = COMMAND_TABLE.get(intent, COMMAND_TABLE["unknown"])
    log_fn(f"[voice intent={action.intent}] -> {action.ros_call}")
    if action.tts_response:
        log_fn(f"[voice intent={action.intent}] TTS: {action.tts_response!r}")
    return action
