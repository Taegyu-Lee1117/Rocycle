"""Intent -> ROS2 action mapping (설계문서 v31 4-2절).

Kept separate from IntentClassifier so it's testable without hitting the
OpenAI API: this module is pure data + a dispatch function.

Node wiring is a stub for now (log only) -- actual /conveyor/stop etc. calls
land here once state_machine_node exists (작업 순서 E).

**[정정 — 6일차, v55/v56 회신]** 팀원 STT/TTS 실제 구현은 원문 텍스트가
아니라 이미 분류된 명령(`/voice_command`, 대문자 START/PAUSE/RESUME/
STATUS/STOP/UNKNOWN)을 발행한다 -- `IntentClassifier`(OpenAI 기반)는
이 경로에서 호출하지 않는다(삭제는 안 함, 팀원 쪽 분류기 장애 시
폴백으로 남겨둠). 여기 키는 소문자로 유지하고, 수신 측(`voice_bridge_
node`/`tracking_node`)에서 `.lower()`로 흡수한다.
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
    "start": CommandAction("start", "tracking_node: IDLE -> RUNNING, conveyor ON", None),
    "pause": CommandAction("pause", "tracking_node: RUNNING -> PAUSED, conveyor OFF", None),
    "resume": CommandAction("resume", "tracking_node: PAUSED -> RUNNING, conveyor ON", None),
    "stop": CommandAction("stop", "tracking_node: -> STOPPING, 종료 시퀀스", None),
    # [팀결정 — v56 회신] 지금까지 처리 개수 안내. tts_response는 실제
    # 카운트에 따라 동적으로 만들어야 하므로 여기 정적 문자열로는 못
    # 넣는다 -- tracking_node가 직접 구성해서 /voice/tts/say에 발행한다
    # (dispatch()가 반환하는 이 항목의 tts_response=None은 "정적 응답
    # 없음"이 아니라 "동적으로 별도 처리"라는 뜻).
    "status": CommandAction("status", "tracking_node: 누적 처리 개수 응답(동적)", None),
    # [팀결정 — v56 회신] 오늘 범위에서 제외 -- speed_inquiry는 팀원
    # STT 발행 목록에 없음(여유 되면 추가 요청함), handoff_ack는 이미
    # 변위 감지로 핸드오버가 동작해 음성 응답이 필수가 아님(v51 5절).
    # 코드에는 남겨두되(향후 팀원 쪽이 지원하면 바로 씀) 죽은 경로.
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
