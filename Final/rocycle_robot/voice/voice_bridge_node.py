"""voice_bridge_node -- 이미 분류된 명령 -> 정적 TTS 응답 발행.

**[정정 — 6일차, v55/v56 회신] 경계는 텍스트 레벨이 아니었다.** 팀원
STT/TTS 실제 구현(`STT_TTS_현재연동정보_전달용.txt`)은 원문 텍스트가
아니라 이미 의도분류까지 끝낸 명령을 발행한다:
    구독: /voice_command  (std_msgs/String, 값: START/PAUSE/RESUME/
                            STATUS/STOP/UNKNOWN -- 대문자)
    발행: /voice/tts/say   (std_msgs/String) -- 팀원 쪽 TTS 토픽명은
                            아직 미확정(v55 회신 "예시 A: /voice/tts"),
                            확정되면 이 발행 토픽명만 바꾸면 됨.

`IntentClassifier`(OpenAI 기반)는 이 경로에서 더 이상 호출하지 않는다
-- 팀원 쪽이 이미 분류를 하므로 중복이고, API 크레딧/네트워크 지연
문제도 함께 없어진다(v56 1절). 삭제는 안 하고 `intent_classification.py`
그대로 둔다 -- 팀원 쪽 분류기 장애 시 폴백 경로로 쓸 수 있다.

이 노드가 하는 일: `/voice_command`를 구독해 `.lower()`로 정규화한
뒤 `command_mapping.dispatch()`로 조회하고, **정적** TTS 응답이 있는
품목(예: speed_inquiry)만 발행한다. `status`처럼 실제 로봇 상태(누적
처리 개수)가 필요한 동적 응답은 이 노드가 모른다 -- 그건 `tracking_
node`가 `/voice_command`를 직접 별도로 구독해서 처리한다(START/PAUSE
/RESUME/STOP의 실제 상태 전이도 마찬가지로 tracking_node 쪽 몫,
아래 참고). 이 노드와 tracking_node가 같은 토픽을 각자 구독하는 건
ROS2 토픽 특성상 문제없다(다중 구독자 허용, 서비스가 아니므로 경합
없음).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rocycle_robot.voice.command_mapping import dispatch


class VoiceBridgeNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("voice_bridge_node", **kwargs)

        self._tts_pub = self.create_publisher(String, "/voice/tts/say", 10)
        self.create_subscription(String, "/voice_command", self._on_voice_command, 10)

        self.get_logger().info("voice_bridge_node up (listening on /voice_command).")

    def _on_voice_command(self, msg: String) -> None:
        intent = msg.data.strip().lower()
        self.get_logger().info(f"[VOICE] received={msg.data!r} -> intent={intent!r}")

        action = dispatch(intent, log_fn=self.get_logger().info)
        if action.tts_response:
            out = String()
            out.data = action.tts_response
            self._tts_pub.publish(out)
            self.get_logger().info(f"[TTS publish] {action.tts_response!r}")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
