"""Pixel -> robot-base 3D coordinate via ray/belt-plane intersection.

Used by tracking_node for the fixed camera (C270, no depth) — this is the
only 3D localization path available for that camera (설계문서 2절/3-3a절).

Pipeline: pixel (u,v) -> undistort -> camera-frame ray -> base-frame ray
          -> intersect with belt plane -> base-frame (x, y, z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

# Below this, the ray is considered near-parallel to the plane and the
# intersection is numerically unreliable (division blows up). This is the
# reason the camera is mounted at 30~40 degrees rather than near-horizontal
# (설계문서 3-1c절).
MIN_RAY_PLANE_COSINE = 0.05


class RayPlaneParallelError(ValueError):
    """Raised when the camera ray is too close to parallel with the belt plane."""


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # shape (5,), plumb_bob: [k1, k2, p1, p2, k3]

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Plane:
    """Plane in base frame: a*x + b*y + c*z + d = 0."""

    a: float
    b: float
    c: float
    d: float

    @property
    def normal(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c], dtype=np.float64)

    @classmethod
    def from_points(cls, points: np.ndarray) -> "Plane":
        """Least-squares plane fit from an (N,3) array of base-frame points, N>=3.

        Returns the plane and (via `fit_residuals`) lets the caller separately
        inspect residuals — see `fit_plane_with_residuals`.
        """
        plane, _ = fit_plane_with_residuals(points)
        return plane


def fit_plane_with_residuals(points: np.ndarray) -> tuple[Plane, np.ndarray]:
    """Least-squares plane fit (SVD) + per-point signed residual distances (mm).

    points: (N,3) array of base-frame xyz, N >= 3. Units are whatever the
    input points use (mm, matching get_current_posx()).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        raise ValueError("need at least 3 points to fit a plane")

    centroid = points.mean(axis=0)
    centered = points - centroid
    # Smallest singular vector of the centered points = plane normal.
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    a, b, c = normal
    d = -float(normal @ centroid)
    plane = Plane(a=float(a), b=float(b), c=float(c), d=float(d))

    residuals = (points @ normal) + d  # signed distance of each point to plane
    return plane, residuals


def undistort_to_camera_ray(u: float, v: float, intr: CameraIntrinsics) -> np.ndarray:
    """Pixel (u,v) -> unit-length ray direction in the camera frame (+z forward)."""
    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, intr.camera_matrix, intr.dist_coeffs)
    x_norm, y_norm = undistorted[0, 0]
    ray = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def pixel_to_base_point(
    u: float,
    v: float,
    intr: CameraIntrinsics,
    cam_to_base: np.ndarray,
    belt_plane: Plane,
) -> np.ndarray:
    """Full pipeline: pixel -> base-frame 3D point on the belt plane.

    cam_to_base: 4x4 homogeneous transform, camera frame -> base frame
                 (rotation R = cam_to_base[:3,:3], origin O = cam_to_base[:3,3]).
    Raises RayPlaneParallelError if the ray grazes the plane (see module docstring).
    """
    ray_cam = undistort_to_camera_ray(u, v, intr)

    R = cam_to_base[:3, :3]
    origin_base = cam_to_base[:3, 3]
    ray_base = R @ ray_cam
    ray_base = ray_base / np.linalg.norm(ray_base)

    n = belt_plane.normal
    denom = float(n @ ray_base)
    cosine = abs(denom) / np.linalg.norm(n)
    if cosine < MIN_RAY_PLANE_COSINE:
        raise RayPlaneParallelError(
            f"ray-plane cosine {cosine:.4f} < {MIN_RAY_PLANE_COSINE} "
            "(camera ray nearly parallel to belt plane -- check mount angle)"
        )

    t = -(n @ origin_base + belt_plane.d) / denom
    if t < 0:
        raise RayPlaneParallelError(
            f"intersection behind camera (t={t:.3f}) -- check cam_to_base/plane signs"
        )

    return origin_base + t * ray_base
