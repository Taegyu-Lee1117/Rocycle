"""Belt plane teaching data — 3일차 실측, 5점 (펜던트 수동 조작 + 육성 좌표 판독).

Captured via teach-pendant manual jog (ROS2 service layer was unresponsive
during the manual<->auto control-mode switch, so points were read directly
off the pendant display rather than via get_current_posx()). Units: mm/deg,
DR_BASE frame, matches get_current_posx(ref=0) convention.

Regenerate the fitted plane with:
    python3 -c "from config.belt_plane_teach_points import fit_and_report; fit_and_report()"
"""

import numpy as np

# (label, x, y, z, rx, ry, rz)
TEACH_POINTS = [
    ("near-center", 171.18, -259.15, 319.13, 122.80, 179.82, 123.48),
    ("near-left", 164.91, -231.98, 318.34, 124.29, -179.87, 124.76),
    ("near-right", 171.94, -288.04, 319.08, 165.46, 179.91, 166.11),
    ("far-left", 668.99, -231.19, 320.61, 142.40, -179.75, 142.07),
    ("far-right", 648.56, -286.16, 320.45, 149.39, -179.51, 149.41),
]

# far-end note: this was the robot's *reach* limit, not the belt or camera
# FOV limit -- consistent with the A-7 finding that reach and camera FOV
# both bound the usable ~70cm working section (설계문서 3-1c절).

FITTED_PLANE_ABCD = (-0.003466, 0.005640, 0.999978, -316.7869)  # a,b,c,d
RESIDUALS_MM = {
    "near-center": 0.281,
    "near-left": -0.334,
    "near-right": 0.066,
    "far-left": 0.193,
    "far-right": -0.206,
}


def fit_and_report():
    from rocycle_robot.geometry.plane_intersection import fit_plane_with_residuals

    points = np.array([[p[1], p[2], p[3]] for p in TEACH_POINTS])
    labels = [p[0] for p in TEACH_POINTS]
    plane, residuals = fit_plane_with_residuals(points)
    print(f"plane: a={plane.a:.6f} b={plane.b:.6f} c={plane.c:.6f} d={plane.d:.4f}")
    for lbl, r in zip(labels, residuals):
        print(f"  {lbl:12s} residual = {r:+.3f} mm")
    print(f"max |residual| = {np.max(np.abs(residuals)):.3f} mm")
    print(f"rms residual   = {np.sqrt(np.mean(residuals ** 2)):.3f} mm")
    return plane, residuals
