"""Unit tests for rocycle_robot.geometry.obstacles.

Pure logic, no ROS/hardware needed -- run with:
    python3 -m pytest ros2/rocycle_robot/test/test_obstacles.py -v
"""

import pytest

from rocycle_robot.geometry.obstacles import (
    Obstacle,
    ObstacleCollisionError,
    check_target_safe,
    load_obstacles_from_dict,
)


def _belt_obstacle():
    return Obstacle(
        name="belt",
        x_min=164.0,
        x_max=670.0,
        y_min=-289.0,
        y_max=-230.0,
        z_floor=321.0,
        note="테이블면(약 241.8)부터 벨트면(약 316.8~320.6)까지 롤러/프레임 구간",
    )


def test_target_above_floor_is_safe():
    # z above the floor, inside xy bounds -> no exception
    check_target_safe(400.0, -250.0, 400.0, [_belt_obstacle()])


def test_target_below_floor_inside_bounds_raises():
    with pytest.raises(ObstacleCollisionError):
        check_target_safe(400.0, -250.0, 294.69, [_belt_obstacle()])


def test_regression_actual_collision_case():
    """3일차 실제 충돌 좌표: (444.14, -240.56, 294.69) 벨트를 눌렀다."""
    with pytest.raises(ObstacleCollisionError):
        check_target_safe(444.14, -240.56, 294.69, [_belt_obstacle()])


def test_target_outside_xy_bounds_is_safe_even_if_low():
    # z very low, but xy well outside the belt footprint
    check_target_safe(0.0, 0.0, 50.0, [_belt_obstacle()])


def test_target_exactly_at_floor_is_safe():
    check_target_safe(400.0, -250.0, 321.0, [_belt_obstacle()])


def test_multiple_obstacles_first_violation_reported():
    bin1 = Obstacle(name="can_bin", x_min=0, x_max=100, y_min=0, y_max=100, z_floor=200.0)
    bin2 = Obstacle(name="paper_bin", x_min=200, x_max=300, y_min=0, y_max=100, z_floor=200.0)
    with pytest.raises(ObstacleCollisionError, match="paper_bin"):
        check_target_safe(250.0, 50.0, 100.0, [bin1, bin2])


def test_load_obstacles_from_dict_skips_null_entries():
    # flat schema: 통은 "bins" 아래 중첩하지 않고 최상위에 개별 항목으로
    # 둔다(v32 원안은 중첩이었으나 로더 단순화를 위해 CLI가 평탄화함,
    # 설계문서 v29 9절 참고)
    data = {
        "belt": {
            "xy_bounds": [164.0, 670.0, -289.0, -230.0],
            "z_floor": 321.0,
            "note": "belt region",
        },
        "plastic_bin": None,
        "can_bin": None,
        "camera_mount": None,
    }
    obstacles = load_obstacles_from_dict(data)
    assert len(obstacles) == 1
    assert obstacles[0].name == "belt"
    assert obstacles[0].z_floor == 321.0


def test_load_obstacles_from_dict_empty():
    assert load_obstacles_from_dict({}) == []
