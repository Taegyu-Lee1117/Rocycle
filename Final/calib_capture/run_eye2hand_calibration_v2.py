#!/usr/bin/env python3
"""Eye-to-hand calibration, corrected version.

corecode의 eye2hand_calibration.py에서 버그 2개를 발견해 우회한다
(가상 정답 데이터로 검증 완료, 3일차 실측 보고서 참고):

1. `Calibrate()` 커스텀 AX=XB 솔버 자체가 틀린 값을 반환한다
   (노이즈 없는 가상 데이터로도 회전오차 94도, 위치오차 908mm) —
   OpenCV 내장 `cv2.calibrateHandEye()`로 교체.
2. `find_checkerboard_pose()`가 반환한 solvePnP 결과(이미 정확히
   checker2cam 관계)를 호출부에서 다시 한 번 `inv()`해서 뒤집는다
   — 이 여분의 inv()를 제거해야 한다(코드 안 변수명이 반대로
   붙어있어서 생긴 버그로 추정).

eye-to-hand 트릭(카메라 고정, 체커보드가 그리퍼에 부착)에서 올바른
cv2.calibrateHandEye 입력:
  R_gripper2base 슬롯 <- inv(raw 로봇 pose) = T_base2gripper(i)
  R_target2cam   슬롯 <- solvePnP 결과 그대로(checker2cam, 여분 inv 없음)
  반환값이 곧 T_cam2base(추가 반전 불필요)
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(
    0, os.path.expanduser("~/Downloads/corecode/corecode/Calibration_Tutorial")
)
from eye2hand_calibration import get_robot_pose_matrix  # noqa: E402

CHECKERBOARD_SIZE = (10, 7)
SQUARE_SIZE = 25  # mm

C270_CAMERA_MATRIX = np.array(
    [
        [1427.528173, 0.0, 674.333213],
        [0.0, 1429.801911, 359.179031],
        [0.0, 0.0, 1.0],
    ]
)
C270_DIST_COEFFS = np.array([0.083788, -0.006721, -0.005860, 0.007959, 0.0])


def find_checkerboard_pose_fixed(image, board_size, square_size, camera_matrix, dist_coeffs):
    """corecode의 find_checkerboard_pose와 동일하되, 반환값 의미를 명확히 한다:
    solvePnP(objp, corners, ...)의 (R,t)는 P_cam = R @ P_obj + t 이므로
    이미 그 자체로 'checker2cam' 변환이다(추가 반전 불필요).
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None
    corners_sub = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    retval, rvec, tvec = cv2.solvePnP(objp, corners_sub, camera_matrix, dist_coeffs)
    if not retval:
        return None, None
    R_checker2cam, _ = cv2.Rodrigues(rvec)
    t_checker2cam = tvec.flatten()
    return R_checker2cam, t_checker2cam


def main(data_dir):
    with open(os.path.join(data_dir, "calibrate_data.json")) as f:
        data = json.load(f)
    robot_poses = np.array(data["poses"])
    image_paths = [os.path.join(data_dir, fn) for fn in data["file_name"]]

    R_gripper2base_list, t_gripper2base_list = [], []
    R_checker2cam_list, t_checker2cam_list = [], []

    used = 0
    for img_path, pose in zip(image_paths, robot_poses):
        image = cv2.imread(img_path)
        if image is None:
            continue
        R_checker2cam, t_checker2cam = find_checkerboard_pose_fixed(
            image, CHECKERBOARD_SIZE, SQUARE_SIZE, C270_CAMERA_MATRIX, C270_DIST_COEFFS
        )
        if R_checker2cam is None:
            print(f"  checkerboard not found in {img_path}, skipping")
            continue

        T_robot = get_robot_pose_matrix(*pose)  # raw robot pose = gripper2base
        T_base2gripper = np.linalg.inv(T_robot)  # input slot for cv2: "gripper2base"

        R_gripper2base_list.append(T_base2gripper[:3, :3])
        t_gripper2base_list.append(T_base2gripper[:3, 3])
        R_checker2cam_list.append(R_checker2cam)
        t_checker2cam_list.append(t_checker2cam)
        used += 1

    print(f"{used}/{len(image_paths)} samples usable")

    R_x, t_x = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_checker2cam_list, t_checker2cam_list,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_x
    T_cam2base[:3, 3] = t_x.flatten()

    print("\nT_cam2base (corrected) =")
    print(T_cam2base)
    print("\ntranslation (mm):", T_cam2base[:3, 3])

    import math
    z_axis = T_cam2base[:3, 2]
    tilt = math.degrees(math.acos(abs(z_axis[2])))
    print(f"tilt from vertical: {tilt:.2f} deg (design intent 30-40)")

    return T_cam2base


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_v2_belt_extrap_suspect"
    T = main(data_dir)
    out_path = "T_cam2base_corrected.npy"
    np.save(out_path, T)
    print(f"\nsaved to {os.path.abspath(out_path)}")
