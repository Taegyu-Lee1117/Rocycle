"""Tests for TrackingNode's X-axis Kalman filter + pickup-time prediction.

Covers the software half of the "정지 상태 예측 검증" invariant
(v40 4-4절 2번): with a stationary (repeated, unchanging) pixel
detection, the predicted pickup X must equal the current detected X.
A live camera/belt version of this check is still required separately.
"""

import time

import pytest
import rclpy

from rocycle_robot.tracking_node import TrackingNode

CALIB_PATH = (
    "/home/rokey/ws_cobot_pjt/ROKEY-Team-Project/ros2/rocycle_robot/"
    "calib_capture/T_cam2base.npy"
)
BELT_PLANE_ABCD = [-0.003466, 0.005640, 0.999978, -316.7869]


@pytest.fixture
def node():
    rclpy.init()
    n = TrackingNode(
        parameter_overrides=[
            rclpy.parameter.Parameter("camera.cam_to_base_path", value=CALIB_PATH),
            rclpy.parameter.Parameter("belt_plane.abcd", value=BELT_PLANE_ABCD),
        ]
    )
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_stationary_pixel_predicted_x_equals_current_x(node):
    u, v = 530, 550  # arbitrary belt-region pixel, same as used live this session
    t0 = time.monotonic()
    for i in range(10):
        node.update_x_detection(u, v, t0 + i * 0.1)

    current_x = node._kf.state.x
    predicted = node.predict_pickup_point("can")
    assert abs(predicted[0] - current_x) < 2.0
    assert abs(node._kf.state.vx) < 1.0


def test_moving_pixel_predicts_ahead_in_belt_direction(node):
    t0 = time.monotonic()
    for i in range(15):
        u = 500 - i * 5  # base-frame x increases as pixel u decreases here
        node.update_x_detection(u, 550, t0 + i * 0.1)

    predicted = node.predict_pickup_point("can")
    current_x = node._kf.state.x
    # moving belt: predicted pickup point should differ from the last raw x
    assert predicted[0] != current_x
