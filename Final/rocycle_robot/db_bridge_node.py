"""Persist completed robot cycles from /ui/state through the local web API."""

from __future__ import annotations

import json
from collections import deque
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .db_bridge_logic import build_processing_payload


class DatabaseBridgeNode(Node):
    """Observe UI state only; never sends commands to the robot."""

    def __init__(self) -> None:
        super().__init__("rocycle_db_bridge")
        self.declare_parameter("api_url", "http://127.0.0.1:8000")
        self.declare_parameter("request_timeout_sec", 1.5)
        self._api_url = str(self.get_parameter("api_url").value).rstrip("/")
        self._timeout = float(self.get_parameter("request_timeout_sec").value)
        self._queue: deque[tuple[str, dict]] = deque()
        self._queued_ids: set[str] = set()
        self._sent_ids: set[str] = set()
        self._sent_order: deque[str] = deque()
        self._last_error: str | None = None

        self.create_subscription(String, "/ui/state", self._on_ui_state, 10)
        self.create_timer(0.5, self._flush_one)
        self.get_logger().info(f"DB bridge ready: {self._api_url}/api/processing")

    def _on_ui_state(self, message: String) -> None:
        try:
            state = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warn("/ui/state JSON 파싱 실패")
            return

        try:
            built = build_processing_payload(state)
        except ValueError as exc:
            self.get_logger().warn(f"DB 저장 제외: {exc}")
            return
        if built is None:
            return
        external_id, payload = built
        if external_id in self._sent_ids or external_id in self._queued_ids:
            return
        if len(self._queue) >= 100:
            dropped_id, _ = self._queue.popleft()
            self._queued_ids.discard(dropped_id)
            self.get_logger().warning(
                f"DB queue full; dropped oldest event {dropped_id}"
            )
        self._queue.append((external_id, payload))
        self._queued_ids.add(external_id)

    def _flush_one(self) -> None:
        if not self._queue:
            return
        external_id, payload = self._queue[0]
        request = Request(
            f"{self._api_url}/api/processing",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                if response.status not in (200, 201):
                    raise RuntimeError(f"unexpected HTTP status {response.status}")
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            error = str(exc)
            if error != self._last_error:
                self.get_logger().warn(f"DB API 전송 대기: {error}")
                self._last_error = error
            return

        self._queue.popleft()
        self._queued_ids.discard(external_id)
        self._sent_ids.add(external_id)
        self._sent_order.append(external_id)
        if len(self._sent_order) > 2000:
            expired_id = self._sent_order.popleft()
            self._sent_ids.discard(expired_id)
        self._last_error = None
        self.get_logger().info(
            f"처리 기록 저장 완료: {payload['predicted_class']} -> {payload['destination']}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatabaseBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
