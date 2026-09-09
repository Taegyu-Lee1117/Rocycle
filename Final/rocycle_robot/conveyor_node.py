"""conveyor_node -- ~/bin/conveyor를 감싸는 최소 ROS2 노드 (v58 회신, 팀결정).

설계문서(v32) 5절 인터페이스 이름을 따른다:
    /conveyor/set_speed  Service
    /conveyor/stop       Service
    /conveyor/status     Topic

**[전제, 의도된 단순화]** `/conveyor/set_speed`는 원래 임의 속도
(Req: speed)를 받는 설계지만, (a) 실제로는 100 steps/sec 고정값만
실사용 확정됐고(CLAUDE.md "현재 확정값" 표, 탈조 없음 확인됨), (b)
이 패키지(`rocycle_robot`)가 순수 `ament_python`이라 커스텀 `.srv`
(float 필드) 생성 인프라가 없다(이 세션에서 `colcon build` 자체가
안 되는 상태로 확인됨 -- 새 인터페이스 패키지를 만드는 건 검증
못할 위험이 큰 별도 작업). 그래서 `std_srvs/srv/Trigger`로 구현하고
"set_speed 호출 = 확정 속도(100)로 시작"으로 단순화했다. 가변 속도가
실제로 필요해지면 그때 커스텀 인터페이스 패키지를 새로 만들 것.

**[팀결정 — v58 회신] 시리얼 포트는 이 노드만 소유한다.** 다른 어떤
프로세스도 `~/bin/conveyor`를 직접 서브프로세스로 호출하면 안 된다
(`tracking_node._call_conveyor()`를 이 노드의 서비스 호출로 바꾸는
연동은 다음 단계 -- 이 노드 자체는 지금 독립적으로 완성해둔다).

**[팀결정 — v58 회신, 3-1절] 워치독은 자리만 만들어두고 기본은
비활성화.** 지금은 로봇 PC/관제 PC가 물리적으로 한 대라 `/system/
heartbeat`를 아무도 발행하지 않는다 -- 기본으로 켜두면 "하트비트가
없다"는 게 항상 참이라 매번 오탐으로 정지해버린다. `conveyor.
watchdog_enabled`(기본 False) 파라미터로 나중에 PC를 분리할 때만
켠다.
"""

from __future__ import annotations

import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

_FIXED_SPEED = 100  # steps/sec -- CLAUDE.md "현재 확정값", 탈조 없음 확인됨


class ConveyorNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("conveyor_node", **kwargs)

        self.declare_parameter("conveyor.watchdog_enabled", False)
        self.declare_parameter("conveyor.watchdog_timeout_sec", 2.0)
        # [완료 — 6일차, v61 회신] 기계별 값(시리얼을 여는 스크립트
        # 경로) -- 예전엔 특정 계정 홈 디렉터리가 코드에 박혀 있었다
        # (`/home/rokey/bin/conveyor`). 파라미터로 빼서 관제 PC 등
        # 다른 계정/경로에서도 동작하게 한다 -- 기본값은 기존 값 그대로
        # 유지(하위 호환), 다른 기계에서는 `config/local.yaml`로
        # 오버라이드할 것(config/local.example.yaml 참고).
        self.declare_parameter("conveyor.script_path", "/home/rokey/bin/conveyor")
        self._conveyor_bin = self.get_parameter("conveyor.script_path").value
        self._watchdog_enabled = self.get_parameter("conveyor.watchdog_enabled").value
        self._watchdog_timeout_sec = self.get_parameter(
            "conveyor.watchdog_timeout_sec"
        ).value

        self._state = "STOP"  # "STOP" | "RUN"
        self._last_speed = 0
        self._last_command_time = time.monotonic()
        self._last_heartbeat_time = time.monotonic()

        self._status_pub = self.create_publisher(String, "/conveyor/status", 10)
        self.create_subscription(Empty, "/system/heartbeat", self._on_heartbeat, 10)

        self.create_service(Trigger, "/conveyor/set_speed", self._on_set_speed)
        self.create_service(Trigger, "/conveyor/stop", self._on_stop)

        # [3-3절] 노드가 뜰 때 벨트는 정지 상태로 둔다 -- 재시작 시
        # 이전 상태를 복원하면 사람이 예상 못 한 순간에 벨트가 돈다.
        self._run_conveyor_cmd("off")
        self._state = "STOP"

        self.create_timer(1.0, self._publish_status)
        if self._watchdog_enabled:
            self.create_timer(0.2, self._check_watchdog)

        self.get_logger().info(
            f"conveyor_node up. watchdog_enabled={self._watchdog_enabled} "
            f"(timeout={self._watchdog_timeout_sec}s)"
        )

    def _run_conveyor_cmd(self, cmd: str) -> bool:
        """[3-5절] 시리얼 열기 실패 등으로 죽지 않는다 -- 에러만 로그로
        남기고 계속 동작(명령 전송 실패 시 1회 재시도 후 실패 반환)."""
        for attempt in range(2):
            try:
                subprocess.run([self._conveyor_bin, cmd], check=True, timeout=5.0)
                self._last_command_time = time.monotonic()
                return True
            except Exception as e:
                self.get_logger().error(
                    f"conveyor {cmd} failed (attempt {attempt + 1}/2): {e}"
                )
        return False

    def _on_set_speed(self, request, response):
        ok = self._run_conveyor_cmd(str(_FIXED_SPEED))
        self._state = "RUN" if ok else "STOP"
        self._last_speed = _FIXED_SPEED if ok else 0
        response.success = ok
        response.message = (
            f"conveyor RUN at {_FIXED_SPEED} steps/sec" if ok else "conveyor command failed"
        )
        return response

    def _on_stop(self, request, response):
        ok = self._run_conveyor_cmd("off")
        self._state = "STOP" if ok else self._state
        self._last_speed = 0 if ok else self._last_speed
        response.success = ok
        response.message = "conveyor STOP" if ok else "conveyor stop command failed"
        return response

    def _on_heartbeat(self, msg: Empty) -> None:
        self._last_heartbeat_time = time.monotonic()

    def _check_watchdog(self) -> None:
        gap = time.monotonic() - self._last_heartbeat_time
        if gap > self._watchdog_timeout_sec and self._state == "RUN":
            self.get_logger().warn(
                f"[WATCHDOG] heartbeat missing {gap:.1f}s > "
                f"{self._watchdog_timeout_sec}s -- stopping conveyor "
                "(복구는 수동, 자동 재개 안 함)"
            )
            self._run_conveyor_cmd("off")
            self._state = "STOP"
            self._last_speed = 0

    def _publish_status(self) -> None:
        msg = String()
        msg.data = (
            f"state={self._state} speed={self._last_speed} "
            f"last_cmd_age={time.monotonic() - self._last_command_time:.1f}s"
        )
        self._status_pub.publish(msg)

    def stop_on_shutdown(self) -> None:
        """[3-4절] 노드가 죽을 때 벨트를 멈춘다 -- 노드만 죽고 워치독
        (다른 PC 이야기)이 없는 상황을 여기서 막는다."""
        self.get_logger().info("conveyor_node shutting down -- stopping conveyor")
        self._run_conveyor_cmd("off")


def main(args=None):
    rclpy.init(args=args)
    node = ConveyorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
