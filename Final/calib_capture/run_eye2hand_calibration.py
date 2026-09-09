#!/usr/bin/env python3
"""Run eye-to-hand calibration on the 3일차 captured samples.

Uses the algorithm from ~/Downloads/corecode/corecode/Calibration_Tutorial/
eye2hand_calibration.py verbatim (imported, not copy-pasted) with our actual
board size (10x7, matches the same physical checkerboard used for the C270
intrinsic calibration, 2일차) and data path.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.expanduser("~/Downloads/corecode/corecode/Calibration_Tutorial")
)
from eye2hand_calibration import (  # noqa: E402
    Calibrate,
    calibrate_camera_from_chessboard,
    compose_transformation_matrices,
    find_checkerboard_pose,
    get_robot_pose_matrix,
)
import cv2  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKERBOARD_SIZE = (10, 7)
SQUARE_SIZE = 25  # mm, matches 2일차 intrinsic calibration board

# [실측 — 2일차] C270 intrinsics, ~/Downloads/c270_calibration_ost.yaml
C270_CAMERA_MATRIX = np.array(
    [
        [1427.528173, 0.0, 674.333213],
        [0.0, 1429.801911, 359.179031],
        [0.0, 0.0, 1.0],
    ]
)
C270_DIST_COEFFS = np.array([0.083788, -0.006721, -0.005860, 0.007959, 0.0])


def main():
    with open(os.path.join(DATA_DIR, "calibrate_data.json")) as f:
        data = json.load(f)
    robot_poses = np.array(data["poses"])
    image_paths = [os.path.join(DATA_DIR, fn) for fn in data["file_name"]]

    valid_indices = []
    for i, pose in enumerate(robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)
        det_T = np.linalg.det(T_base2gripper)
        print(f"sample {i}: det(T_base2gripper) = {det_T:.4f}")
        if np.abs(det_T) > 1e-6:
            valid_indices.append(i)
        else:
            print(f"  WARNING: singular T_base2gripper at sample {i}, skipping")
    robot_poses = robot_poses[valid_indices]
    image_paths = [image_paths[i] for i in valid_indices]

    # NOTE: unlike the reference __main__, we already have known-good C270
    # intrinsics from 2일차 -- reuse them instead of recalibrating from these
    # 9 (fewer, less diverse) samples, which would be a weaker intrinsic fit.
    camera_matrix = C270_CAMERA_MATRIX
    dist_coeffs = C270_DIST_COEFFS

    R_gripper2base_list, t_gripper2base_list = [], []
    R_checker2camera_list, t_checker2camera_list = [], []

    used = 0
    for img_path, pose in zip(image_paths, robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  could not read {img_path}, skipping")
            continue

        R_cam2checker, t_cam2checker = find_checkerboard_pose(
            image, CHECKERBOARD_SIZE, SQUARE_SIZE, camera_matrix, dist_coeffs
        )
        if R_cam2checker is None:
            print(f"  checkerboard not found in {img_path}, skipping")
            continue

        T_gripper2base = np.linalg.inv(T_base2gripper)
        R_gripper2base_list.append(T_gripper2base[:3, :3].copy())
        t_gripper2base_list.append(T_gripper2base[:3, 3].reshape(-1, 1).copy())

        T_cam2checker = np.eye(4)
        T_cam2checker[:3, :3] = R_cam2checker
        T_cam2checker[:3, 3] = t_cam2checker.flatten()
        T_checker2cam = np.linalg.inv(T_cam2checker)
        R_checker2camera_list.append(T_checker2cam[:3, :3].copy())
        t_checker2camera_list.append(T_checker2cam[:3, 3].copy())
        used += 1

    print(f"\n{used}/{len(image_paths)} samples usable for hand-eye solve")

    T_gripper2base_list = compose_transformation_matrices(
        R_gripper2base_list, t_gripper2base_list
    )
    T_checker2cam_list = compose_transformation_matrices(
        R_checker2camera_list, t_checker2camera_list
    )
    num_pairs = min(len(T_gripper2base_list), len(T_checker2cam_list))

    A_list, B_list = [], []
    for i in range(num_pairs - 1):
        A_i = np.dot(
            np.linalg.inv(T_gripper2base_list[i]), T_gripper2base_list[i + 1]
        )
        B_i = np.dot(
            np.linalg.inv(T_checker2cam_list[i]), T_checker2cam_list[i + 1]
        )
        A_list.append(A_i)
        B_list.append(B_i)

    theta, b_x = Calibrate(A_list, B_list)
    X = np.eye(4)
    X[:3, :3] = theta
    X[:3, 3] = b_x.flatten()
    T_cam2base = X

    print("\nT_cam2base =")
    print(T_cam2base)
    print("\ntranslation (mm):", T_cam2base[:3, 3])

    out_path = os.path.join(DATA_DIR, "..", "T_cam2base.npy")
    np.save(out_path, T_cam2base)
    print(f"\nsaved to {os.path.abspath(out_path)}")
    return T_cam2base


if __name__ == "__main__":
    main()
