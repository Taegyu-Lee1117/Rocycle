"""Unit tests for rocycle_robot.geometry.plane_intersection.

Pure math, no ROS/hardware needed -- run with:
    python3 -m pytest ros2/rocycle_robot/test/test_plane_intersection.py -v
"""

import numpy as np
import pytest

from rocycle_robot.geometry.plane_intersection import (
    CameraIntrinsics,
    Plane,
    RayPlaneParallelError,
    fit_plane_with_residuals,
    pixel_to_base_point,
    undistort_to_camera_ray,
)


def _top_down_camera_setup():
    """Camera at base (0,0,500mm), looking straight down (-Z), no roll/skew.

    R = diag(1,-1,-1): camera +x -> base +x, camera +z (forward) -> base -z.
    Belt plane is base Z=0.
    """
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0,
                             dist_coeffs=np.zeros(5))
    R = np.diag([1.0, -1.0, -1.0])
    cam_to_base = np.eye(4)
    cam_to_base[:3, :3] = R
    cam_to_base[:3, 3] = [0.0, 0.0, 500.0]
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    return intr, cam_to_base, plane


class TestPixelToBasePoint:
    def test_principal_point_maps_to_point_directly_below_camera(self):
        intr, cam_to_base, plane = _top_down_camera_setup()
        point = pixel_to_base_point(intr.cx, intr.cy, intr, cam_to_base, plane)
        np.testing.assert_allclose(point, [0.0, 0.0, 0.0], atol=1e-6)

    def test_offset_pixel_matches_hand_calculation(self):
        # height 500mm, focal 1000px, 100px offset -> real offset = 500*100/1000 = 50mm
        intr, cam_to_base, plane = _top_down_camera_setup()
        point = pixel_to_base_point(intr.cx + 100.0, intr.cy, intr, cam_to_base, plane)
        np.testing.assert_allclose(point, [50.0, 0.0, 0.0], atol=1e-6)

    def test_offset_pixel_negative_and_y_axis(self):
        intr, cam_to_base, plane = _top_down_camera_setup()
        # +v in image -> camera +y -> base -y (R has -1 on y)
        point = pixel_to_base_point(intr.cx, intr.cy + 200.0, intr, cam_to_base, plane)
        np.testing.assert_allclose(point, [0.0, -100.0, 0.0], atol=1e-6)

    def test_grazing_ray_raises_instead_of_diverging(self):
        """Ray nearly parallel to the belt plane must raise, not return garbage.

        This is exactly the failure mode 설계문서 3-1c절 avoids by mounting
        the camera at 30~40 degrees instead of near-horizontal.
        """
        intr, cam_to_base, plane = _top_down_camera_setup()
        # extreme horizontal pixel offset -> ray nearly along camera's x-axis,
        # which is horizontal in base frame (perpendicular to the plane normal)
        grazing_u = intr.cx + 1000.0 * 1000.0  # x_norm ~= 1000
        with pytest.raises(RayPlaneParallelError):
            pixel_to_base_point(grazing_u, intr.cy, intr, cam_to_base, plane)

    def test_result_always_lies_on_the_plane(self):
        intr, cam_to_base, plane = _top_down_camera_setup()
        for du, dv in [(0, 0), (37, -82), (-150, 60), (250, 250)]:
            point = pixel_to_base_point(intr.cx + du, intr.cy + dv, intr, cam_to_base, plane)
            residual = plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d
            assert abs(residual) < 1e-6


class TestUndistortToCameraRay:
    def test_zero_distortion_principal_point_is_forward(self):
        intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0,
                                 dist_coeffs=np.zeros(5))
        ray = undistort_to_camera_ray(320.0, 240.0, intr)
        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)

    def test_ray_is_unit_length(self):
        intr = CameraIntrinsics(fx=1427.5, fy=1429.8, cx=674.3, cy=359.2,
                                 dist_coeffs=np.array([0.0838, -0.0067, -0.0059, 0.0080, 0.0]))
        for u, v in [(0, 0), (1280, 720), (674, 359), (100, 600)]:
            ray = undistort_to_camera_ray(u, v, intr)
            assert abs(np.linalg.norm(ray) - 1.0) < 1e-9


class TestFitPlaneWithResiduals:
    def test_exact_flat_points_have_zero_residual(self):
        points = np.array([
            [0.0, 0.0, 100.0],
            [50.0, 0.0, 100.0],
            [0.0, 50.0, 100.0],
            [50.0, 50.0, 100.0],
        ])
        plane, residuals = fit_plane_with_residuals(points)
        np.testing.assert_allclose(residuals, np.zeros(4), atol=1e-6)
        # normal should be +-Z
        assert abs(abs(plane.c) - 1.0) < 1e-6
        assert abs(plane.a) < 1e-6 and abs(plane.b) < 1e-6

    def test_noisy_points_give_small_nonzero_residuals(self):
        rng = np.random.default_rng(0)
        base = np.array([
            [0.0, 0.0, 100.0], [50.0, 0.0, 100.0],
            [0.0, 50.0, 100.0], [50.0, 50.0, 100.0],
        ])
        noisy = base + rng.normal(scale=0.5, size=base.shape)
        _, residuals = fit_plane_with_residuals(noisy)
        assert np.all(np.abs(residuals) < 5.0)  # well within a few mm for 0.5mm noise

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_plane_with_residuals(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
