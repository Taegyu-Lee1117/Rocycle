"""tracking_node -- fixed-camera detection tracking + pixel->base 3D (설계문서 5절).

Status (3일차): only the plane-intersection math (geometry/plane_intersection.py)
is implemented and unit-tested. Detection input, extrinsic calibration (camera->
base) and the belt plane are not measured yet -- this node loads them as config
and will refuse to publish points until all three are present (see
_require_calibration below), rather than silently using wrong defaults.

Not yet implemented (tracked in 설계문서 13절 작업 순서):
  - Kalman filter / predict service (/tracking/predict) -- 작업 순서 이후 단계
  - Actual /track/detections subscription from a YOLO detector node
  - Belt speed feed for velocity state
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from rocycle_robot.geometry.plane_intersection import (
    CameraIntrinsics,
    Plane,
    RayPlaneParallelError,
    pixel_to_base_point,
)


class TrackingNode(Node):
    def __init__(self) -> None:
        super().__init__("tracking_node")

        # camera intrinsics -- [실측] 2일차 C270 캘리브레이션 결과
        # (재활용로봇_설계문서 13절 26번, ~/Downloads/c270_calibration_ost.yaml)
        self.declare_parameter("camera.fx", 1427.528)
        self.declare_parameter("camera.fy", 1429.802)
        self.declare_parameter("camera.cx", 674.333)
        self.declare_parameter("camera.cy", 359.179)
        self.declare_parameter(
            "camera.dist_coeffs", [0.083788, -0.006721, -0.005860, 0.007959, 0.0]
        )

        # extrinsic (camera -> base) and belt plane -- NOT measured yet (작업 C/B).
        # Left unset on purpose; see _require_calibration.
        self.declare_parameter("camera.cam_to_base_path", "")
        self.declare_parameter("belt_plane.abcd", [0.0, 0.0, 0.0, 0.0])

        self._intrinsics = CameraIntrinsics(
            fx=self.get_parameter("camera.fx").value,
            fy=self.get_parameter("camera.fy").value,
            cx=self.get_parameter("camera.cx").value,
            cy=self.get_parameter("camera.cy").value,
            dist_coeffs=np.array(self.get_parameter("camera.dist_coeffs").value),
        )
        self._cam_to_base: np.ndarray | None = self._load_cam_to_base()
        self._belt_plane: Plane | None = self._load_belt_plane()

        # TODO(작업 이후 단계): 실제 detection 토픽 구독 — 지금은 인터페이스만
        # self.create_subscription(..., "/track/detections", self._on_detection, 10)

        self.get_logger().info(
            "tracking_node up. calibration ready=%s" % self._is_calibrated()
        )

    def _load_cam_to_base(self) -> np.ndarray | None:
        path = self.get_parameter("camera.cam_to_base_path").value
        if not path:
            return None
        return np.load(path)

    def _load_belt_plane(self) -> Plane | None:
        a, b, c, d = self.get_parameter("belt_plane.abcd").value
        if (a, b, c, d) == (0.0, 0.0, 0.0, 0.0):
            return None
        return Plane(a=a, b=b, c=c, d=d)

    def _is_calibrated(self) -> bool:
        return self._cam_to_base is not None and self._belt_plane is not None

    def pixel_to_base(self, u: float, v: float) -> np.ndarray:
        """Public entry point used by _on_detection (and by 작업 E test scripts)."""
        if not self._is_calibrated():
            raise RuntimeError(
                "cam_to_base / belt_plane not set -- run 작업 B/C first "
                "(설계문서 13절 작업 순서) before calling pixel_to_base"
            )
        try:
            return pixel_to_base_point(
                u, v, self._intrinsics, self._cam_to_base, self._belt_plane
            )
        except RayPlaneParallelError:
            self.get_logger().warn(f"pixel ({u},{v}): ray grazes belt plane, dropping")
            raise


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
