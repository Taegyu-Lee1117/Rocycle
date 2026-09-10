"""tracking_node -- fixed-camera detection tracking + pixel->base 3D (설계문서 5절).

Status (6일차): plane-intersection math + belt_correction 전부 구현·검증
완료(실제 파지로 확인). X축 1차원 칼만 필터 + 예측 시점 계산 구현
완료(v40/v41/v44 설계 반영) — Y/Z는 상수로 둔다(4일차 실측: 그리퍼
자기 정렬이 Y축을 흡수, X축은 안 흡수 — CLAUDE.md 참고).

**[통합 완료 — 6일차, v47] YOLO12s 실시간 검출 연동.** `/image_raw`를
구독해 `vision.RecycleDetector`로 프레임마다 검출한다 — 좌표변환
(`pixel_to_base_point`/`belt_correction`)·칼만필터·예측 로직은 전혀
건드리지 않았다(통합 프롬프트 7/11번 준수). `vision.dry_run` 파라미터로
로봇 실행 여부를 제어(기본 True).

**[완료 — 6일차, 순차 처리 인덱싱] 다중 물체 트랙 기반으로 재설계.**
기존엔 `detect()`(`max_det=1`)로 프레임당 최고-신뢰도 물체 하나만
보고 물체 1개 기준 단일 상태(`_pending_class`/`_locked_class`/
`self._kf`)로 추적했다 — 신뢰도가 낮은 물체는 다른 물체가 화면에
있는 동안 추적 자체가 안 되는 문제가 실측으로 확인됨(CLAUDE.md
"현재 단계" 10번). **해결**: `detect_all()`로 여러 물체를 동시에
받아 `self._tracks`(물체별 트랙 리스트)로 관리하고, Pick 우선순위는
신뢰도가 아니라 벨트 진행방향(가장 앞선 물체)으로 정한다. 트랙은
파지 사이클(블로킹 구간) 동안에도 상태가 유지되므로, 블로킹이
끝난 직후 이미 확인된 트랙은 재확인 없이 곧바로 Pick 후보가 된다.

**[완료 — 6일차] 로봇 연결 + 상태머신 골격.** `dry_run=False`면
`_execute_pick()`이 실제로 파지하고, `_place_item()`이
`config/item_routing.yaml`을 조회해 통에 배치하거나(bin) 사람전달
경로는 placeholder로 처리한다(원칙3 준수 — 품목명 분기 없음).
같은 벨트 위 실물 검증 완료(can, 이동 중 파지 성공/실패 사례와
원인 규명은 CLAUDE.md 참고). 아직 완전한 상태머신은 아니다 —
그리퍼가 하나뿐이라 물리적 파지/배치는 여전히 한 번에 하나씩
순차 진행(위 다중 트랙은 "누구를 다음에 집을지" 우선순위만 개선),
별도 노드로 분리 안 됨, 사람전달은 실제 순응제어 핸드오버 구현
완료(아래 참고).

**[완료 — 6일차, v48 회신] 파지 사이클 중 컨베이어 정지/재가동
(v23 원칙 개정).** 순차 처리 시 벨트를 안 세우면 다음 물체가
처리 중(약 24초) 도달범위를 벗어나는 문제(NOT REACHABLE 알람으로
확정)가 있어, 검출~추적~예측~파지 구간은 벨트가 돌고(칼만 예측이
실제로 필요한 구간, 그대로 유지), 파지 성공 직후부터 배치+복귀
완료까지만 `_call_conveyor("off"/"on")`로 정지·재가동한다. 배치
실패 시(도달 불가 등)는 벨트를 정지 상태로 유지(사람 확인 후
재시작). 아직 실물 재검증 전 — 다음 시행에서 확인 필요.

**[완료 — 6일차] 실제 순응제어 기반 사람 핸드오버.** `_handoff_item()`
— `task_compliance_ctrl`(저강성)로 순응 모드 진입, `get_current_posx`
폴링으로 사람이 당기는 변위(release_threshold_mm)를 감지하면 그리퍼
오픈 후 `release_compliance_ctrl`로 복귀. 새 x/y는 만들지 않고
`_execute_pick`이 이미 도달 검증한 벨트 상공 컬럼에서 z만 올려
제시 높이로 쓴다(안전 절 참고). 설정은 `config/item_routing.yaml`의
`handoff` 절. **[전제] 오늘 최초 실물 시행 예정 — 10회 연속 무사고
검증 전이므로 "안전 확정"으로 표현하지 않는다.**

아직 구현 안 됨:
  - 정지 상태(벨트 v=0) 라이브 검증(작업5, 예정)
  - 별도 상태머신 노드로 분리(현재는 tracking_node에 같이 있음)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

try:
    from dsr_msgs2.srv import MoveLine, ReleaseComplianceCtrl, TaskComplianceCtrl
    from onrobot_rg_msgs.srv import SetCommand
except ImportError:  # pragma: no cover -- only needed when dry_run=False
    MoveLine = None
    SetCommand = None
    TaskComplianceCtrl = None
    ReleaseComplianceCtrl = None

from rocycle_robot.geometry.belt_correction import correct_belt_xy
from rocycle_robot.geometry.kalman_1d import ConstantVelocityKalman1D
from rocycle_robot.geometry.pickup_timing import (
    load_gripper_profiles,
    load_motion_timing,
    predict_ahead_sec,
)
from rocycle_robot.geometry.plane_intersection import (
    CameraIntrinsics,
    Plane,
    RayPlaneParallelError,
    pixel_to_base_point,
)
from rocycle_robot.routing import get_bin_pose, load_item_routing


_CROSS_CLASS_DUPLICATE_IOU = 0.8


def _bbox_iou(lhs, rhs) -> float:
    """Return IoU for two YOLO xyxy boxes, or zero for invalid boxes."""
    if not (
        isinstance(lhs, (list, tuple))
        and isinstance(rhs, (list, tuple))
        and len(lhs) >= 4
        and len(rhs) >= 4
    ):
        return 0.0

    lx1, ly1, lx2, ly2 = map(float, lhs[:4])
    rx1, ry1, rx2, ry2 = map(float, rhs[:4])
    if lx2 <= lx1 or ly2 <= ly1 or rx2 <= rx1 or ry2 <= ry1:
        return 0.0

    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _deduplicate_cross_class_detections(detections):
    """Keep the highest-confidence class for nearly identical YOLO boxes."""
    indexed = [
        (index, detection)
        for index, detection in enumerate(detections)
        if isinstance(detection, dict)
    ]
    indexed.sort(
        key=lambda item: float(item[1].get("confidence", 0.0)),
        reverse=True,
    )

    kept = []
    for index, detection in indexed:
        class_name = detection.get("class_name")
        is_cross_class_duplicate = any(
            class_name != kept_detection.get("class_name")
            and _bbox_iou(detection.get("bbox"), kept_detection.get("bbox"))
            >= _CROSS_CLASS_DUPLICATE_IOU
            for _, kept_detection in kept
        )
        if not is_cross_class_duplicate:
            kept.append((index, detection))

    kept.sort(key=lambda item: item[0])
    return [detection for _, detection in kept]


class TrackingNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("tracking_pick_node", **kwargs)

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

        # X축 1차원 칼만 필터 -- Y/Z는 상수 취급(설계 근거는 모듈
        # docstring 및 CLAUDE.md 참고). process_var/measurement_var는
        # [전제, 잠정치] -- belt_correction RMS(4.5mm)를 measurement
        # 표준편차로 역산한 값. 실측 데이터가 쌓이면 재튜닝할 것.
        self.declare_parameter("kalman.process_var", 1.0)
        self.declare_parameter("kalman.measurement_var", 20.0)
        self._kalman_process_var = self.get_parameter("kalman.process_var").value
        self._kalman_measurement_var = self.get_parameter("kalman.measurement_var").value

        self._motion_timing = load_motion_timing()
        self._gripper_profiles = load_gripper_profiles()
        self._item_routing = load_item_routing()  # 상태머신 골격 (6일차)

        # Docker YOLO detection 토픽 기반 설정
        # /recycle_detection/detections 의 JSON 결과를 사용한다.
        self.declare_parameter(
            "vision.min_consecutive_frames",
            9
        )

        self.declare_parameter(
            "vision.object_lost_frames",
            5
        )

        # 실제 로봇 실행 전까지 반드시 True
        self.declare_parameter(
            "vision.dry_run",
            True
        )

        # C270 영상에서 벨트 영역 최소 y pixel
        self.declare_parameter(
            "vision.belt_pixel_y_min",
            470
        )

        self.declare_parameter(
            "vision.track_match_gate_mm",
            500.0
        )

        # [9일차, v139] 벨트 무정지 전환용. 1단계에서는 stop_during_pick을
        # true로 두어 기존 인터록 동작을 그대로 유지한다 -- 2단계에서
        # false로 바꿔 무정지 시험을 하고, 문제가 있으면 true로 되돌린다.
        self.declare_parameter("conveyor.stop_during_pick", True)
        # 로봇 도달한계 590mm에서 50mm 앞. [전제] 실측으로 조정할 것.
        self.declare_parameter("conveyor.reach_limit_stop_x", 540.0)
        # 사유별이 아니라 집합 전체에 대한 워치독. 사유가 늘어도 코드를
        # 안 고쳐도 되고, interlock/weighing도 노드 예외로 영영 안 풀릴
        # 수 있다. 120초 근거: reach_limit은 두 사이클에 걸쳐 유지될 수
        # 있다(A 처리 중 B가 한계 도달 -> A가 끝나도 B가 한계에 서 있어
        # B 처리 완료 시점에야 풀린다). 재시도 없이 24초x2=48초, 한쪽에
        # 재시도가 끼면 70초 이상이라 90초는 여유가 20초 안팎뿐이었다.
        self.declare_parameter("conveyor.hold_watchdog_sec", 120.0)
        # [9일차] 이 간격보다 짧게 도착한 검출은 파지 중 쌓였다가
        # 드레인된 백로그로 보고 버린다. 카메라 7.5Hz = 약 133ms의
        # 절반인 66ms를 기본값으로 둔다 -- 실시간 프레임이 이보다
        # 촘촘히 올 수는 없다.
        self.declare_parameter("vision.detection_min_gap_sec", 0.066)
        # [9일차] 검출 박스가 화면 가로 경계에서 이 여유 안쪽에 닿으면
        # 잘린 것으로 보고 버린다. 물체가 화면에 온전히 들어온 뒤에야
        # 추적·파지 대상이 된다. 벨트가 2.9mm/s로 느려 몇 초 늦게
        # 잡히는 정도이고, 도달한계(590mm)까지 여유가 충분하다.
        self.declare_parameter("vision.edge_margin_px", 15)
        self.declare_parameter("vision.image_width_px", 1280)
        # [9일차, v143] 예측에 쓰는 칼만 속도의 허용 범위.
        # **벨트 실측 3.06mm/s**(캔, 중앙 구간, 표본 151개/20초,
        # 잔차 RMS 0.25mm)의 약 2배를 상한으로 둔다. 벨트는 정속
        # 장치이고 역주행하지 않으므로 하한은 0이다.
        self.declare_parameter("kalman.vx_min_mm_s", 0.0)
        self.declare_parameter("kalman.vx_max_mm_s", 6.0)

        # ============================================================
        # [9일차 야간, v155] **파지 자세 검증 1단계 -- 손목캠 기록만.**
        # ============================================================
        # 목적: 관절위치 기반 파지판정(`_check_grasp`)의 사각지대를
        # 메우기 위한 데이터 수집이다. `plastic`(뚜껑)/`plastic_bag`
        # (비닐)은 같은 물체라도 어디를 무느냐에 따라 관절각이 크게
        # 벌어져 임계값 자체를 못 잡는다(9일차 실측: 찌그러진 뚜껑을
        # 성공적으로 물었는데 파지값이 빈 손 +0.7496과 같게 나옴).
        #
        # **이 단계는 판정하지 않는다.** ROI를 캡처해 파일로 저장하고
        # 통계만 로그에 남긴다. 사이클 동작은 전혀 바뀌지 않는다.
        # 임계값은 품목별 표본이 쌓인 뒤에 정한다 -- 근거 없는 임계값을
        # 넣어두면 다음 단계에서 그걸 믿고 동작을 바꿀 때 위험하다.
        #
        # **ROI 근거(실측)**: 손목 리얼센스는 그리퍼를 거의 못 본다.
        # 그리퍼를 열고 닫으며 프레임을 차분한 결과 변화 영역이
        # x 330~540 / y 420~480(640x480 기준)뿐이었다 -- 화면 맨 아래
        # 가장자리에 손가락 끝(노란 패드)만 걸친다. 열면 손가락이
        # 화면 밖으로 나가고, 나머지 화면은 전부 벨트와 바닥이다.
        # ROI가 화면 고정 좌표라 좌표변환이 없고, 따라서 eye-in-hand
        # 캘리브레이션도 필요 없다.
        self.declare_parameter("grasp_check.enabled", True)
        self.declare_parameter(
            "grasp_check.image_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter("grasp_check.roi", [330, 410, 540, 480])
        self.declare_parameter("grasp_check.save_dir", "")
        self.declare_parameter("grasp_check.wait_sec", 0.6)
        # 배경 노출 비율(채도·밝기 기준) 임계값. 이 값보다 높으면
        # "그리퍼에 아무것도 없음"으로 본다. **로그에만 쓴다(1단계).**
        # 실측 8표본 기준 빈손 37.6~62.2% / 파지 7.6~9.6%의 중간값.
        self.declare_parameter("grasp_check.bg_threshold", 0.23)
        # ============================================================
        # [9일차 야간, v156] **배치 확인 + depth 1단계 -- 전부 기록만.**
        # ============================================================
        # (a) 배치 확인 개방폭: 배치 후 그리퍼가 실제로 열렸는지 확인.
        #     설계문서는 `/onrobot/pose`로 읽으라고 하는데 **실측 결과
        #     그 서비스는 그리퍼 상태와 무관하게 항상 같은 값을 준다**
        #     (열림/닫힘/재열림 모두 x=0.0843, y=0.1904). 개방폭 신호는
        #     `/onrobot_joint_states`뿐이다 -- 열림 ±0.4793rad, 빈손
        #     닫힘 +0.7496rad. 이미 `_gripper_joint`로 받고 있으므로
        #     그 값을 쓴다.
        # (b) 배치 스냅샷: 배치 직후 손목캠 ROI를 같이 남긴다. 그리퍼가
        #     열렸는데도 물체가 남아 있는 경우(끼임)를 눈으로 확인할
        #     자료가 된다.
        # (c) depth 1단계: 파지 시점 ROI의 거리 통계를 로그에 남긴다.
        #     고정 하강깊이(100/110/115mm)의 타당성을 사후 검증하기
        #     위한 자료다. **좌표 변환을 하지 않으므로 캘리브레이션이
        #     필요 없다.** [한계] depth와 color는 정렬돼 있지 않아
        #     같은 ROI 좌표가 정확히 같은 영역은 아니다 -- 1단계에서는
        #     경향만 본다.
        self.declare_parameter("place_check.enabled", True)
        # [정정 -- 10일차, v159] **"기준값 +- 허용범위"에서 "경계값"으로
        # 바꾼다.** 처음엔 열림 기준 -0.4793에서 +-0.15로 잡았는데,
        # 실측 11건에서 열림 값 자체가 넓게 퍼져 오탐이 났다:
        #     배치 성공(열림)  -0.4793 x4, -0.4668 x2, -0.4638,
        #                      -0.3953, -0.2856   (폭 0.194)
        #     파지 실패(빈손)  +0.1622 x2
        # -0.2856이 허용범위를 벗어나 "안 열림(의심)" 오탐이 났다
        # (실제로는 정상 배치). 두 군집은 0.448 떨어져 있으므로
        # 경계값 하나로 가르는 것이 맞다 -- 중간값 -0.06을 쓴다
        # (양쪽 여유 각각 약 0.22).
        self.declare_parameter("place_check.open_max_rad", -0.06)
        # ============================================================
        # [9일차 야간, v157] **회전각 추정 1단계 -- 로그만.**
        # ============================================================
        # 고정캠(C270) 원본 프레임에서 검출 bbox 안의 물체 장축 각도를
        # 추정한다. 벨트가 균일한 초록이라 색으로 물체를 분리하고,
        # 최소 회전 사각형(minAreaRect)의 장축 각도를 쓴다.
        # 0도 = 화면 가로 = 벨트 진행 방향.
        #
        # **[중요] 이 기능은 지금 실효가 없다.** 전제가 "물체가 비스듬히
        # 놓이면 그리퍼를 그 각도로 돌린다"인데, 9일차 야간 실측에서
        # **비스듬한 캔은 애초에 검출되지 않는다**는 것이 확인됐다:
        #     벨트와 나란히   77/77 프레임 (conf 0.90)
        #     45도 비스듬히    0/77 프레임
        #     벨트 가로지름    1/77 프레임
        # 같은 캔·같은 자리에서 각도만 바꾼 결과이고, 같은 프레임의
        # 뚜껑/건전지는 세 번 다 77/77로 안정적이었다(카메라·조명
        # 문제가 아니다). 학습 사진을 캔이 가로로 놓인 상태로만
        # 찍었기 때문일 가능성이 크다 -- 즉 **모델 학습 데이터 문제**다.
        # 돌려야 할 물체가 검출되지 않으므로 2단계(rz 적용)로 갈 이유가
        # 없고, 1단계는 과제 구색과 기록 목적으로만 둔다(사용자 판단).
        #
        # 추정기 자체는 동작이 확인됐다(같은 장면 3회 반복):
        #     뚜껑     세장비 1.08  -> 원형이라 각도 무의미(자동 배제 근거)
        #     490캔    세장비 2.79  (실제 168/66 = 2.5와 일치)
        #     건전지   세장비 2.90, 각도 -5.1 ~ -5.7도로 일관
        # [v158] `/ui/detections` 규격의 h. 폭은 vision.image_width_px 재사용.
        self.declare_parameter("vision.image_height_px", 720)
        self.declare_parameter("rotation_check.enabled", True)
        self.declare_parameter("rotation_check.image_topic", "/image_raw")
        self.declare_parameter("rotation_check.min_elongation", 1.2)
        self.declare_parameter("depth_check.enabled", True)
        self.declare_parameter(
            "depth_check.image_topic", "/camera/camera/depth/image_rect_raw"
        )

        self._min_consecutive_frames = (
            self.get_parameter(
                "vision.min_consecutive_frames"
            ).value
        )

        self._object_lost_frames = (
            self.get_parameter(
                "vision.object_lost_frames"
            ).value
        )

        self._dry_run = self.get_parameter(
            "vision.dry_run"
        ).value

        self._belt_pixel_y_min = (
            self.get_parameter(
                "vision.belt_pixel_y_min"
            ).value
        )

        self._track_match_gate_mm = (
            self.get_parameter(
                "vision.track_match_gate_mm"
            ).value
        )

        self._stop_during_pick = self.get_parameter(
            "conveyor.stop_during_pick"
        ).value
        self._reach_limit_stop_x = self.get_parameter(
            "conveyor.reach_limit_stop_x"
        ).value
        self._hold_watchdog_sec = self.get_parameter(
            "conveyor.hold_watchdog_sec"
        ).value
        self._detection_min_gap_sec = self.get_parameter(
            "vision.detection_min_gap_sec"
        ).value
        self._edge_margin_px = self.get_parameter("vision.edge_margin_px").value
        self._image_width = self.get_parameter("vision.image_width_px").value
        self._vx_min = self.get_parameter("kalman.vx_min_mm_s").value
        self._vx_max = self.get_parameter("kalman.vx_max_mm_s").value

        # [v155] 손목캠 파지 기록. `_wrist_frame`은 마지막 수신 프레임을
        # (monotonic 수신시각, encoding, height, width, bytes)로 들고 있다.
        # 변환은 캡처 시점에만 한다 -- 15Hz 콜백에서 매번 numpy 변환하면
        # 검출 콜백과 같은 executor를 쓰는 구간에서 낭비가 크다.
        self._grasp_check_enabled = self.get_parameter("grasp_check.enabled").value
        self._grasp_check_topic = self.get_parameter("grasp_check.image_topic").value
        _roi = list(self.get_parameter("grasp_check.roi").value or [])
        self._grasp_check_roi = tuple(int(v) for v in _roi) if len(_roi) == 4 else None
        _sdir = self.get_parameter("grasp_check.save_dir").value or ""
        self._grasp_check_dir = (
            Path(_sdir).expanduser() if _sdir else Path.home() / "grasp_check"
        )
        self._grasp_check_wait = float(self.get_parameter("grasp_check.wait_sec").value)
        self._grasp_check_bg_th = float(
            self.get_parameter("grasp_check.bg_threshold").value
        )
        self._place_check_enabled = self.get_parameter("place_check.enabled").value
        self._place_open_max = float(
            self.get_parameter("place_check.open_max_rad").value
        )
        self._image_height = self.get_parameter("vision.image_height_px").value
        # [v158] `reach_limit_px`는 계산 비용이 있어 한 번만 구해 캐시한다.
        self._reach_limit_px = None
        self._rot_check_enabled = self.get_parameter("rotation_check.enabled").value
        self._rot_check_topic = self.get_parameter("rotation_check.image_topic").value
        self._rot_min_elong = float(
            self.get_parameter("rotation_check.min_elongation").value
        )
        self._fixed_frame = None
        self._depth_check_enabled = self.get_parameter("depth_check.enabled").value
        self._depth_check_topic = self.get_parameter("depth_check.image_topic").value
        self._wrist_depth = None
        self._wrist_frame = None
        self._grasp_check_seq = 0

        # [완료 — 6일차, 순차 처리 인덱싱] 물체 1개 기준 단일 상태
        # (_pending_class/_locked_class/self._kf 하나)를 물체별 트랙
        # 리스트로 교체 -- 여러 물체를 동시에 추적하고, Pick 우선순위는
        # 신뢰도가 아니라 벨트 진행방향(가장 앞선 물체)으로 정한다.
        # 트랙 하나 = {class_name, kf, pending_count, last_y, last_z,
        # last_detection_time, lost_count}. 상세: _on_image/_match_track.
        self._tracks: list[dict] = []
        self._next_track_id = 0

        # ------------------------------------------------------------
        # [9일차, v139] 컨베이어 정지 사유 집합 (벨트 무정지 전환 1단계)
        # ------------------------------------------------------------
        # 인터록/도달한계/계량이 각자 on/off를 부르면 경쟁상태가 생긴다
        # (도달한계로 서 있는데 계량이 먼저 끝나 재가동 -> 사유가 안
        # 풀렸는데 벨트가 돎). **사유 집합이 빌 때만 실제로 재가동하는
        # 단일 소유자 구조**로 만든다.
        #
        # 세 사유가 모두 이 노드에서 발생하므로 집합을 여기 둔다.
        # 다른 노드(예: 미구현 손검출 안전정지)가 벨트를 세울 필요가
        # 생기면 conveyor_node로 옮길 것. 지금 conveyor_node에 두지 않는
        # 이유는 colcon 커스텀 .srv 구성이 없어 사유 이름을 실어보낼
        # 메시지를 못 만들기 때문(v59 제약).
        self._hold_reasons: set[str] = set()
        self._hold_since: dict[str, float] = {}
        # 비상정지는 **집합 밖에 별도로 둔다.** 집합에 넣으면 다른 사유가
        # 다 풀렸을 때 집합이 비어서 벨트가 돌아버린다. 사람이 명시적으로
        # 해제할 때만 풀려야 한다.
        self._emergency_stop = False
        # 워치독이 강제 재가동시킨 트랙. 도달한계 판정과 파지 후보
        # **양쪽에서** 제외한다(한쪽만 빼면 즉시 재정지 또는 무한 실패).
        self._abandoned_track_ids: set[int] = set()
        # [9일차, v118 2단계] 품목별 파지 실패 누적(진단용).
        self._grasp_fail_counts: dict[str, int] = {}
        # [9일차] 검출 백로그 폐기용. 카메라 7.5Hz(약 133ms)의 절반보다
        # 짧은 간격은 실시간 프레임일 수 없다.
        self._last_detection_arrival = 0.0
        self._stale_detection_drops = 0
        self._vx_clamp_count = 0
        self._edge_clipped_drops = 0

        # [안전 — 6일차, 사고 후 추가] 배치/상태머신이 아직 없어 파지
        # 후 놓는 동작이 없다 -- 들고 있는 채로 다음 물체를 또 집으려
        # 하면 그리퍼가 닫힌 채로 눌러 찌그러뜨리는 사고가 남(실측).
        # 하나를 들고 있는 동안은 새 Pick을 아예 막는다.
        self._holding = False
        # [최적화 — 6일차, v49 회신] 힘 정규화가 매번 "최대까지 올린
        # 뒤 목표까지 내리기"(최대 28회 명령, 실측 1.7~2.2초)라 순차
        # 처리 시 confirmed~완료 구간을 늘려 뒤 물체가 도달범위를
        # 벗어나는 문제(위 CLAUDE.md 12번 항목)에 기여했다. 세션
        # 내내 힘 설정이 유지된다는 것(이미 알려진 함정)을 거꾸로
        # 이용해 마지막 설정값을 코드가 직접 추적 -- 알면 차이만큼만
        # 조정. 첫 파지(None)는 절대값을 모르니 기존 방식(캘리브레이션)
        # 그대로 유지.
        self._last_force_n: float | None = None

        # [완료 — 6일차, v55/v56 회신] 음성 명령 게이팅. START를 받기
        # 전까지는 검출·추적은 하되 Pick 트리거는 막는다(팀원 회신
        # 5절: "START를 받았다고 즉시 Pick 실행하면 안 됨, 컨베이어
        # 먼저 켜고 Detection 전까지 로봇은 안 움직여야 함" -- 이
        # 요구사항이 지금까지 아예 없었다는 게 v55에서 드러난 갭).
        # PAUSE/STOP은 `_execute_pick`/`_place_item`이 블로킹이라
        # 콜백 자체가 그동안 안 불리므로, 자연히 "현재 사이클 완료
        # 후"에만 반영된다(v56 4-2절 확인 -- 물체를 든 채로 멈추면
        # 그리퍼에 물체가 남는 문제를 막기 위해 의도된 동작이지,
        # 버그가 아니다). 상태: IDLE(대기) / RUNNING(가동) /
        # PAUSED(일시정지) / STOPPING(종료 시퀀스 중, 오늘은 즉시
        # IDLE로 전이 -- 실제 "현재 사이클 마무리 후 종료"는
        # _place_item/_handoff_item 쪽에서 자연히 처리됨).
        # [전제] 기본값 IDLE이 실제 음성 연동 시 맞는 동작이지만,
        # 이번 세션 내내 써온 단독 테스트 스크립트들(voice 노드 없이
        # TrackingNode만 띄워 바로 파지 테스트)이 전부 깨진다 -- START를
        # 아무도 안 보내면 영원히 Pick이 안 나감. 테스트 편의를 위해
        # 파라미터로 초기 상태를 오버라이드할 수 있게 둔다(기본값은
        # 실제 배포 의도대로 IDLE 유지).
        self.declare_parameter("voice.initial_state", "IDLE")
        self._voice_state = self.get_parameter("voice.initial_state").value
        self._pick_counts: dict[str, int] = {}
        # [8일차] 핸드오버 타임아웃(사람이 안 받아감) -> review_bin 재배치
        # 횟수만 따로 센다. `_pick_counts`는 성공 인계와 동일하게 계속
        # 올라가므로(기존 집계 호환 유지) 여기서 구분 데이터를 남긴다 --
        # 나중에 실측 보정할 때 이 수치가 있어야 인계 실패율을 알 수 있다.
        self._handoff_timeout_counts: dict[str, int] = {}
        # [8일차] 무게 측정 진단 로그용 -- 직전 이동 종료 시각, 컨베이어
        # 가동 여부. baseline만 벨트 가동 중에 측정되는 문제를 로그에서
        # 바로 확인할 수 있게 한다.
        self._last_move_done_t: float | None = None
        self._conveyor_running = False

        # [8일차, v118 1단계] 그리퍼 개폐 위치 -- 파지 성공/실패 자동 판정용.
        # 지금까지 파지 성공률은 사람 눈으로만 확인했지 자동 측정이 없었다.
        # **무게로는 판정이 안 된다**(실측: 빈 캔 16g 파지 시 net이
        # +14.4/-10.9/+35.9g로 0을 사이에 두고 흩어짐 -- measure_pose로
        # 개선한 뒤에도 그렇다). 반면 그리퍼 폭은 물체 유무가 반대편
        # 끝으로 갈린다(빈 캔 +0.1780 vs 파지 실패 +0.7496).
        # **[버그 발견·수정 -- 8일차, 실물 1회차]** 처음엔 메인 노드에
        # 구독을 걸었는데, 파지 사이클 중 `_call_move_line` 등이 메인
        # executor를 블로킹해 콜백이 안 돌았다 -- 실측에서 파지 후인데
        # `joint=-0.4638`(완전 열림, 파지 시작 전 값)이 읽혀 "파지 성공"
        # 오판이 났다(실제 값은 +0.1405). 토픽은 50Hz로 정상 발행 중이었다.
        # **수정**: 로봇 서비스 호출에 쓰는 별도 노드/executor
        # (`_robot_node`/`_robot_executor`)에 구독을 걸어, 블로킹 구간
        # 에서도 `spin_until_future_complete`가 도는 동안 같이 갱신되게
        # 한다. CLAUDE.md "알려진 함정"의 executor 분리와 같은 이유다.
        self._gripper_joint: float | None = None

        # 로봇 실행 -- dry_run=False일 때만 생성. [버그 발견·수정 —
        # 6일차] 처음엔 self(=TrackingNode)에 클라이언트를 만들고
        # rclpy.spin_until_future_complete(self, future)로 기다렸는데,
        # 이 호출 자체가 `_on_image` 콜백(=self를 스핀 중인 외부
        # spin_once 안)에서 실행되다 보니 "Executor is already
        # spinning"으로 즉시 실패했다(재진입 스핀 금지). **완전히
        # 별도의 Node+SingleThreadedExecutor**를 만들어 로봇 제어
        # 전용으로 쓰는 것으로 해결 — 이러면 TrackingNode의 구독
        # 콜백 스핀과 겹치지 않는다. CLAUDE.md "알려진 함정" 준수:
        # MoveLine success=True를 믿지 않고 get_current_posx로 검증.
        self._robot_node = None
        self._robot_executor = None
        self._move_line_client = None
        self._gripper_client = None
        self._get_posx_client = None

        # 현재 검증 완료된 conveyor_node 인터페이스 사용
        self._conveyor_pub = self.create_publisher(
            String,
            "/conveyor_command",
            10,
        )

        # [8일차] 관제 화면(web_ui) 연동 -- `tracking_node.py`에 있던
        # `/ui/*` 발행을 이 노드로 이식. web_ui/index.html이 실제로
        # 구독하는 건 `/ui/state`와 `/ui/alert` 두 개다.
        # `/ui/detections`(bbox 오버레이)는 화면에서 아직 안 그리므로
        # (web_ui/README.md "알려진 제약") 이번엔 넣지 않았다.
        self._bin_counts: dict[str, int] = {}
        self._stage = "idle"
        self._last_result: dict | None = None
        self._last_image_time: float | None = None

        self._ui_state_pub = self.create_publisher(String, "/ui/state", 10)
        self._ui_alert_pub = self.create_publisher(String, "/ui/alert", 10)
        # [9일차 -> 10일차, v158] **`/ui/detections` 누락을 메운다.**
        # UI 설계안(3절)은 `/ui/state`, `/ui/alert`, `/ui/detections`
        # 세 토픽을 규격으로 정하는데 이 노드는 앞의 둘만 발행하고
        # 있었다. 화면에 bbox 오버레이를 그리려면 이 토픽이 필요하다.
        self._ui_det_pub = self.create_publisher(String, "/ui/detections", 10)

        # 카메라 생존 신호. 원본(`tracking_node.py`)은 `/image_raw`를
        # 직접 구독하니까 그 콜백에서 시각을 찍었는데, 이 노드는 Docker
        # YOLO의 검출 토픽만 받으므로 그 경로가 없다. `/recycle_detection/
        # detections` 도착 시각으로 대신하면 **벨트가 비어서 검출이 0건인
        # 정상 상황**에도 카메라가 죽은 것처럼 보인다. 그래서 v4l2_camera가
        # 영상과 같은 주기로 내보내는 `/camera_info`(수백 바이트, 디코딩
        # 비용 없음)를 따로 구독해서 판정한다.
        self.create_subscription(
            CameraInfo, "/camera_info", self._on_camera_info, qos_profile_sensor_data
        )

        # [v157] 고정캠 원본. **메인 노드에 건다** -- 회전각 추정은
        # PICK TRIGGER 시점(=`_on_detection` 안, 블로킹 이전)에 하므로
        # 손목캠처럼 `_robot_node`로 옮길 필요가 없다.
        if self._rot_check_enabled:
            self.create_subscription(
                Image, self._rot_check_topic, self._on_fixed_image,
                qos_profile_sensor_data,
            )

        self.create_timer(0.5, self._publish_ui_state)
        # [9일차, v139] 사유집합 워치독 -- 사유가 안 풀린 채 오래 지나면
        # 통째로 비우고 경고한다. 사유별이 아니라 집합 전체에 건다.
        self.create_timer(1.0, self._check_hold_watchdog)

        if not self._dry_run:
            if MoveLine is None or SetCommand is None:
                raise RuntimeError(
                    "dry_run=False인데 dsr_msgs2/onrobot_rg_msgs를 import 못함 "
                    "-- 로봇 스택 소싱 확인할 것"
                )
            from dsr_msgs2.srv import GetCurrentPosx, GetWorkpieceWeight

            self._robot_node = rclpy.create_node("tracking_node_robot_client")
            self._robot_executor = SingleThreadedExecutor()
            self._robot_executor.add_node(self._robot_node)

            self._move_line_client = self._robot_node.create_client(
                MoveLine, "/dsr01/dsr_controller2/motion/move_line"
            )
            self._gripper_client = self._robot_node.create_client(
                SetCommand, "/onrobot/sendCommand"
            )
            self._robot_node.create_subscription(
                JointState, "/onrobot_joint_states", self._on_gripper_joint, 10
            )
            # [v155] 손목캠도 **`_robot_node`에** 건다. TrackingNode 쪽에
            # 걸면 `_execute_pick`의 블로킹 구간 동안 콜백이 안 돌아
            # 파지 직후에 파지 이전 프레임이 잡힌다 -- `_gripper_joint`가
            # 똑같은 이유로 오판을 냈던 전례(6일차)와 같은 구조다.
            if self._grasp_check_enabled:
                self._robot_node.create_subscription(
                    Image,
                    self._grasp_check_topic,
                    self._on_wrist_image,
                    qos_profile_sensor_data,
                )
            if self._depth_check_enabled:
                self._robot_node.create_subscription(
                    Image,
                    self._depth_check_topic,
                    self._on_wrist_depth,
                    qos_profile_sensor_data,
                )
            self._get_posx_client = self._robot_node.create_client(
                GetCurrentPosx, "/dsr01/dsr_controller2/aux_control/get_current_posx"
            )
            self._weight_client = self._robot_node.create_client(
                GetWorkpieceWeight, "/dsr01/dsr_controller2/force/get_workpiece_weight"
            )
            self._task_compliance_client = self._robot_node.create_client(
                TaskComplianceCtrl, "/dsr01/dsr_controller2/force/task_compliance_ctrl"
            )
            self._release_compliance_client = self._robot_node.create_client(
                ReleaseComplianceCtrl, "/dsr01/dsr_controller2/force/release_compliance_ctrl"
            )
            for c, name in (
                (self._move_line_client, "move_line"),
                (self._gripper_client, "sendCommand"),
                (self._get_posx_client, "get_current_posx"),
                (self._weight_client, "get_workpiece_weight"),
                (self._task_compliance_client, "task_compliance_ctrl"),
                (self._release_compliance_client, "release_compliance_ctrl"),
            ):
                if not c.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError(f"service {name} not available -- 로봇 스택 확인")

        # 디버깅/검증용 Robot Base 좌표 출력
        self._base_detection_pub = self.create_publisher(
            String,
            "/tracking/base_detection",
            10,
        )

        # Docker YOLO 결과
        self.create_subscription(
            String,
            "/recycle_detection/detections",
            self._on_detection,
            10,
        )
        # [완료 — 6일차, v55/v56 회신] voice_bridge_node와 별개로 이
        # 토픽을 직접 구독한다 -- 실제 상태 전이·컨베이어 제어·처리
        # 개수 집계는 여기(로봇/컨베이어를 이미 제어하고 있는 노드)
        # 몫이고, voice_bridge_node는 정적 TTS 응답만 담당한다(다중
        # 구독자, 토픽이라 경합 없음).
        self._voice_tts_pub = self.create_publisher(String, "/voice/tts/say", 10)
        self.create_subscription(String, "/voice_command", self._on_voice_command, 10)

        self.get_logger().info(
            "tracking_node up. calibration ready=%s dry_run=%s"
            % (self._is_calibrated(), self._dry_run)
        )

    def _load_cam_to_base(self) -> np.ndarray | None:
        """[9일차] 상대경로를 패키지 share 기준으로 해석한다.

        저장소 `config/tracking.yaml`은 6일차 v61 회신에서 "특정 계정의
        홈 디렉터리가 박힌 절대경로"를 패키지 상대경로로 바꿨는데,
        **`tracking_pick_node`에는 그걸 해석하는 코드가 없어서**
        `np.load("calib_capture/T_cam2base.npy")`가 실행 CWD 기준으로
        찾다가 실패한다. 지금까지는 params 파일의 최상위 키가 노드
        이름과 달라 이 파라미터 자체가 적용되지 않았기 때문에
        드러나지 않았다(같은 날 함께 발견).

        절대경로는 그대로 쓰고, 상대경로만 share 디렉터리 기준으로
        해석한다.
        """
        path = self.get_parameter("camera.cam_to_base_path").value
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = Path(
                get_package_share_directory("rocycle_robot")
            ) / p
        if not p.exists():
            self.get_logger().error(
                f"[CALIB] cam_to_base 파일 없음: {p} -- 캘리브레이션 미적용"
            )
            return None
        self.get_logger().info(f"[CALIB] cam_to_base 로드: {p}")
        return np.load(str(p))

    def _load_belt_plane(self) -> Plane | None:
        a, b, c, d = self.get_parameter("belt_plane.abcd").value
        if (a, b, c, d) == (0.0, 0.0, 0.0, 0.0):
            return None
        return Plane(a=a, b=b, c=c, d=d)

    def _is_calibrated(self) -> bool:
        return self._cam_to_base is not None and self._belt_plane is not None

    def pixel_to_base(self, u: float, v: float) -> np.ndarray:
        """Public entry point used by _on_detection (and by 작업 E test scripts).

        [수정 — 6일차] belt_correction 적용 누락 버그 수정 -- 이전에는
        raw pixel_to_base_point 결과를 그대로 반환해 4~5일차에 실측한
        벨트 평면 보정(RMS 4.5mm 개선)이 이 노드에는 적용되지 않고
        있었다. 3일차 심야 결정대로 XY만 보정하고 Z는 그대로 둔다.
        """
        if not self._is_calibrated():
            raise RuntimeError(
                "cam_to_base / belt_plane not set -- run 작업 B/C first "
                "(설계문서 13절 작업 순서) before calling pixel_to_base"
            )
        try:
            raw = pixel_to_base_point(
                u, v, self._intrinsics, self._cam_to_base, self._belt_plane
            )
        except RayPlaneParallelError:
            self.get_logger().warn(f"pixel ({u},{v}): ray grazes belt plane, dropping")
            raise
        cx, cy = correct_belt_xy(raw[0], raw[1])
        return np.array([cx, cy, raw[2]])

    def _new_track(self, class_name: str) -> dict:
        self._next_track_id += 1
        return {
            "track_id": self._next_track_id,
            "class_name": class_name,
            "last_confidence": None,
            "kf": ConstantVelocityKalman1D(
                process_var=self._kalman_process_var,
                measurement_var=self._kalman_measurement_var,
            ),
            "pending_count": 0,
            "last_y": None,
            "last_z": None,
            "last_detection_time": None,
            "lost_count": 0,
            # [v157] 회전각 추정용. 마지막으로 관측된 검출 bbox.
            "last_bbox": None,
        }

    def _match_track(
        self, class_name: str, base_x: float, now: float, exclude_ids: set
    ) -> dict | None:
        """같은 클래스 기존 트랙 중 지금 위치(base_x)에 가장 가까운 것을 찾는다.

        [완료 — 6일차, 순차 처리 인덱싱] `_execute_pick`/`_place_item`은
        블로킹 호출이라 실행되는 동안(~17~24초) `_on_image` 콜백 자체가
        전혀 안 불린다(single-threaded executor) -- 그래서 단순히
        "마지막 관측 위치"와 비교하면 그 시간 동안 실제로 이동한
        거리를 놓친다. 마지막 갱신 이후 흐른 시간만큼 칼만필터로
        외삽한 예측 위치를 기준으로 비교해야 블로킹 구간을 넘어서도
        같은 물체로 재매칭할 수 있다. `exclude_ids`는 이번 프레임에
        이미 다른 검출과 매칭된 트랙 -- 같은 클래스 물체가 프레임 안에
        여러 개 있을 때 전부 같은 트랙 하나로 몰리는 것을 막는다.
        """
        best = None
        best_dist = None
        for t in self._tracks:
            if t["class_name"] != class_name or id(t) in exclude_ids:
                continue
            dt = now - t["last_detection_time"] if t["last_detection_time"] else 0.0
            predicted_x = t["kf"].predict_position_at(dt) if dt > 0 else t["kf"].state.x
            dist = abs(predicted_x - base_x)
            # [완료 — 6일차, v53 요청] 재매칭 판단 근거를 로그로 남긴다
            # -- 블로킹 구간 이후 재매칭이 실제로 되는지 로그만으로도
            # 추적 가능하게 한다.
            self.get_logger().debug(
                "[MATCH] class=%s candidate_track predicted_x=%.1f "
                "(dt=%.2fs) det_x=%.1f dist=%.1f gate=%.0f"
                % (class_name, predicted_x, dt, base_x, dist, self._track_match_gate_mm)
            )
            if best_dist is None or dist < best_dist:
                best, best_dist = t, dist
        if best is not None and best_dist <= self._track_match_gate_mm:
            self.get_logger().debug(
                "[MATCH] class=%s -> matched existing track (dist=%.1f <= gate=%.0f)"
                % (class_name, best_dist, self._track_match_gate_mm)
            )
            return best
        reason = "no candidate of this class" if best is None else (
            f"best_dist={best_dist:.1f} > gate={self._track_match_gate_mm:.0f}"
        )
        self.get_logger().debug(f"[MATCH] class={class_name} -> new track ({reason})")
        return None

    def _on_gripper_joint(self, msg: JointState) -> None:
        """그리퍼 개폐 위치 갱신 -- `finger_joint` 하나만 본다."""
        try:
            self._gripper_joint = msg.position[msg.name.index("finger_joint")]
        except (ValueError, IndexError):
            pass

    def _on_fixed_image(self, msg) -> None:
        """[v157] 고정캠(C270) 원본 프레임 보관 -- 변환 없이 원본만."""
        self._fixed_frame = (msg.encoding, msg.height, msg.width, msg.data)

    def _estimate_rotation(self, track: dict) -> None:
        """[9일차 야간, v157] **회전각 추정 1단계 -- 로그만 남긴다.**

        검출 bbox 안에서 초록 벨트가 아닌 화소를 물체로 보고, 가장 큰
        외곽선의 최소 회전 사각형에서 장축 각도를 얻는다.
        0도 = 화면 가로 = 벨트 진행 방향.

        세장비(장축/단축)가 `rotation_check.min_elongation` 미만이면
        원형에 가까워 각도가 무의미하므로 "회전 불필요"로 기록한다
        (실측: 컵 뚜껑 1.08).

        **rz를 바꾸지 않는다.** 2단계(적용)는 새 rz 조합마다 도달성을
        재검증해야 하고, 무엇보다 비스듬한 캔이 검출되지 않아 적용할
        대상 자체가 없다 -- 파라미터 선언부 주석 참고.
        """
        if not self._rot_check_enabled:
            return
        try:
            import cv2  # 없으면 조용히 건너뛴다

            frame = self._fixed_frame
            bbox = track.get("last_bbox")
            if frame is None or bbox is None:
                return
            enc, h, w, data = frame
            arr = np.frombuffer(bytes(data), dtype=np.uint8)
            if arr.size != h * w * 3:
                return
            img = arr.reshape(h, w, 3)
            bgr = img[:, :, ::-1] if enc == "rgb8" else img
            bgr = np.ascontiguousarray(bgr)

            pad = 8
            x1 = max(0, int(bbox[0]) - pad); y1 = max(0, int(bbox[1]) - pad)
            x2 = min(w, int(bbox[2]) + pad); y2 = min(h, int(bbox[3]) + pad)
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            H, S = hsv[:, :, 0], hsv[:, :, 1]
            belt = (H >= 35) & (H <= 95) & (S > 60)
            obj = (~belt).astype(np.uint8) * 255
            k5 = np.ones((5, 5), np.uint8); k9 = np.ones((9, 9), np.uint8)
            obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k5)
            obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k9)
            cs, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                return
            c = max(cs, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area < 200.0:
                return
            (_cx, _cy), (rw, rh), ang = cv2.minAreaRect(c)
            if rw < rh:
                ang += 90.0; rw, rh = rh, rw
            ang = ((ang + 90.0) % 180.0) - 90.0
            elong = rw / max(rh, 1.0)
            note = (
                "회전 불필요(원형에 가까움)"
                if elong < self._rot_min_elong
                else "장축 방향 유효"
            )
            self.get_logger().info(
                "[ROTATE] %s 각도=%+.1f도 장축=%.0f 단축=%.0f 세장비=%.2f "
                "면적=%.0f %s -- 기록 전용, rz 변경 없음"
                % (track.get("class_name"), ang, rw, rh, elong, area, note)
            )
        except Exception as exc:  # 진단 기능이 사이클을 깨선 안 된다
            self.get_logger().warn(f"[ROTATE] 추정 실패(무시하고 계속): {exc}")

    def _on_wrist_image(self, msg) -> None:
        """[v155] 손목 리얼센스 컬러 프레임 보관 -- 변환 없이 원본만."""
        self._wrist_frame = (
            time.monotonic(), msg.encoding, msg.height, msg.width, msg.data
        )

    def _on_wrist_depth(self, msg) -> None:
        """[v156] 손목 리얼센스 depth 프레임 보관 -- 변환 없이 원본만."""
        self._wrist_depth = (
            time.monotonic(), msg.encoding, msg.height, msg.width, msg.data
        )

    def _depth_roi_stats(self) -> str:
        """[v156, depth 1단계] ROI 거리 통계 문자열. 실패해도 예외를 내지
        않는다 -- 호출부가 이미 try 안이지만 여기서도 방어한다.

        [한계] depth와 color는 정렬돼 있지 않아 같은 ROI 좌표가 정확히
        같은 영역은 아니다. 1단계는 경향만 본다 -- 고정 하강깊이의
        타당성 확인이 목적이고 좌표 변환은 하지 않는다.
        """
        if not self._depth_check_enabled or self._grasp_check_roi is None:
            return "depth=off"
        try:
            frame = self._wrist_depth
            if frame is None:
                return "depth=미수신"
            _t, enc, h, w, data = frame
            arr = np.frombuffer(bytes(data), dtype=np.uint16)
            if arr.size != h * w:
                return f"depth=크기이상({arr.size},enc={enc})"
            d = arr.reshape(h, w).astype(np.float32)
            x1, y1, x2, y2 = self._grasp_check_roi
            x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
            roi = d[y1:y2, x1:x2]
            v = roi[roi > 0]
            if v.size < 20:
                return f"depth=유효화소부족({v.size})"
            return (
                "depth_valid=%.0f%% depth_min=%.0fmm depth_p50=%.0fmm"
                % (100.0 * v.size / roi.size, float(v.min()),
                   float(np.percentile(v, 50)))
            )
        except Exception as exc:
            return f"depth=실패({exc})"

    def _check_place_open(self, item_key: str, bin_name: str) -> None:
        """[9일차 야간, v156] **배치 확인(a) -- 기록만 한다.**

        배치 후 그리퍼가 실제로 열렸는지 관절값으로 확인한다. 열림은
        약 -0.4793rad, 빈손 닫힘은 +0.7496rad이라 구분이 명확하다.
        열리지 않았다면 물체가 통에 안 들어가고 아직 물려 있다는 뜻
        이므로 경고를 남긴다 -- **다만 동작은 바꾸지 않는다(1단계).**

        설계문서가 지정한 `/onrobot/pose`는 쓰지 않는다. 실측 결과
        그리퍼 상태와 무관하게 항상 같은 값을 반환한다.
        """
        if not self._place_check_enabled:
            return
        try:
            if self._robot_executor is not None:
                deadline = time.monotonic() + 0.3
                while time.monotonic() < deadline:
                    self._robot_executor.spin_once(timeout_sec=0.05)
            joint = self._gripper_joint
            if joint is None:
                self.get_logger().warn(
                    "[PLACECHK] 그리퍼 위치 미수신 -- 개방 확인 건너뜀"
                )
                return
            ok = joint <= self._place_open_max
            msg = (
                "[PLACECHK] %s -> %s 개방폭 joint=%+.4f (경계 %+.4f 이하면 열림) "
                "판정=%s -- 기록 전용"
                % (item_key, bin_name, joint, self._place_open_max,
                   "열림" if ok else "안 열림(의심)")
            )
            if ok:
                self.get_logger().info(msg)
            else:
                self.get_logger().warn(msg)
        except Exception as exc:
            self.get_logger().warn(f"[PLACECHK] 확인 실패(무시하고 계속): {exc}")

    def _capture_grasp_view(self, item_key: str, label: str = "grasp") -> None:
        """[9일차 야간, v155] **파지 자세 검증 1단계 -- 기록만 한다.**

        파지 상승 직후 손목캠 ROI를 저장하고 통계를 로그에 남긴다.
        **판정하지 않고, 사이클도 바꾸지 않는다.** 관절위치 판정이
        원리적으로 못 쓰는 품목(`plastic`/`plastic_bag`)의 임계값을
        정하려면 라벨이 붙은 표본이 먼저 필요하기 때문이다 -- 완주
        1회를 돌리면 관절각 판정과 사용자 육안으로 품목별 정답이
        붙는다.

        전 구간이 try/except로 감싸여 있다. 이 기능의 어떤 실패도
        파지 사이클을 깨서는 안 된다(진단 기능이 운용을 망가뜨리면
        안 된다는 `_weigh_with_log`와 같은 원칙).
        """
        if not self._grasp_check_enabled or self._grasp_check_roi is None:
            return
        try:
            # 블로킹 직후라 마지막 콜백이 오래됐을 수 있다 -- `_check_grasp`
            # 과 같은 이유로 executor를 잠깐 돌려 최신 프레임을 받는다.
            # 15Hz 발행이라 0.6초면 여러 장이 들어온다.
            if self._robot_executor is not None:
                deadline = time.monotonic() + self._grasp_check_wait
                while time.monotonic() < deadline:
                    self._robot_executor.spin_once(timeout_sec=0.05)

            frame = self._wrist_frame
            if frame is None:
                self.get_logger().warn(
                    "[GRIPVIEW] 손목캠 프레임 미수신 -- 기록 건너뜀"
                )
                return
            recv_t, enc, h, w, data = frame
            age = time.monotonic() - recv_t

            arr = np.frombuffer(bytes(data), dtype=np.uint8)
            if arr.size != h * w * 3:
                self.get_logger().warn(
                    f"[GRIPVIEW] 예상 못 한 프레임 크기 {arr.size} "
                    f"(h={h} w={w} enc={enc}) -- 기록 건너뜀"
                )
                return
            img = arr.reshape(h, w, 3)

            x1, y1, x2, y2 = self._grasp_check_roi
            x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
            roi = img[y1:y2, x1:x2].astype(np.float32)

            # rgb8 기준 채널 분리. 다른 인코딩이면 채널 순서만 다르고
            # 통계의 의미는 유지되므로 그대로 계산하되 로그에 남긴다.
            r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            # **판정 원리**: 물체를 물면 그것이 ROI를 가려 배경이 안 보인다.
            # 배경(초록 벨트 + 나무 상판)은 **색이 있고 밝다**. 물린
            # 물체는 흰 뚜껑(채도 낮음)이든 검은 비닐(밝기 낮음)이든
            # 둘 다 이 조건에서 빠진다. 그래서 지표는 OpenCV HSV 정의의
            # `S > 50 AND V > 50`인 화소 비율이다.
            #   V = max(R,G,B),  S = 255 * (max - min) / max
            # cv2 없이 numpy로 계산해 저장 실패와 무관하게 늘 남는다.
            #
            # 실측 8표본(9일차 야간, 라벨 있음):
            #   빈손(벨트 위 x=200/275/300/350/425/500)  37.6 ~ 62.2%
            #   뚜껑 파지 성공                             7.6%
            #   비닐 파지 성공                             9.6%
            # 분리 여유 +28.1%p. 기본 임계 0.23은 그 중간이다.
            #
            # **[기각된 지표] "초록 벨트 + 나무 상판" RGB 색 판정.**
            # x=200에서 24.9%까지 떨어져 빈손인데 "있음"으로 오판됐다
            # (다른 위치는 52~63%). 원인은 나무 판정의 `r > 120` 조건
            # 으로, 그 자리에서 상판이 조금 어둡게 잡히자 항이 통째로
            # 빠졌다. 사진은 육안으로 거의 동일했다 -- 장면이 아니라
            # 지표가 불안정했던 것이다.
            #
            # **[한계] 밝고 채도 높은 물체를 물면 오판한다.** 빨간 장갑,
            # 노란/초록 캔 같은 것은 "배경처럼" 읽혀 빈손으로 판정될 수
            # 있다. 지금 표본은 뚜껑(흰색)과 비닐(검은색)뿐이다.
            vmax = roi.max(axis=2)
            vmin = roi.min(axis=2)
            sat = np.where(vmax > 0, 255.0 * (vmax - vmin) / np.maximum(vmax, 1.0), 0.0)
            bg = float(((sat > 50.0) & (vmax > 50.0)).mean())
            # 진단용 보조 지표(판정에는 쓰지 않는다)
            green = ((g > r + 15.0) & (g > b + 15.0)).mean()
            wood = ((r > b + 25.0) & (r > 120.0) & (g > b + 10.0)).mean()
            edge = (
                np.abs(np.diff(gray, axis=0)).mean()
                + np.abs(np.diff(gray, axis=1)).mean()
            ) / 2.0
            # **판정은 파지 시점(label="grasp")에만 유효하다.** 배경 기준이
            # "초록 벨트 + 나무 상판"이라 벨트 위에서만 성립한다. 배치
            # 시점(label="place")은 통 위에 있어 배경이 흰 플라스틱
            # 바구니이고, 그리퍼가 열려 물체를 놓은 뒤인데도 배경비율이
            # 0.1~4.4%로 나와 "있음"으로 잘못 읽힌다(9일차 야간 실측).
            # 그 자리에서는 이미지만 기록하고 판정은 내지 않는다.
            # **[철회 -- 9일차 야간] 판정을 내지 않는다.** 실패 표본이
            # 실제로 나온 뒤 전부 뒤집혔다:
            #   파지 실패(뚜껑) joint +0.7542  배경  9.6% -> "있음" 오판
            #   파지 실패(뚜껑) joint +0.7542  배경  9.2% -> "있음" 오판
            #   파지 성공(비닐) joint +0.7478  배경 26.0% -> "없음" 오판
            #   파지 성공(뚜껑) joint -0.0095  배경  0.5% -> 정답
            # 4건 중 1건만 맞았다.
            #
            # **원인은 구조적이다.** 파지에 실패하면 물체가 그리퍼 바로
            # 아래 벨트에 그대로 남는다. 카메라는 그리퍼를 지나쳐 아래를
            # 보므로 "물린 흰 뚜껑"과 "빈 그리퍼 아래 놓인 흰 뚜껑"이
            # 거의 같은 그림이다. 앞서 모은 빈손 표본 6건은 벨트가
            # 깨끗한 상태에서 찍은 것이라 이 경우를 대표하지 못했다.
            #
            # 후보 지표 4개(채도·밝기 / 배경색·밝기 / 표준편차 / 엣지)가
            # 전부 겹친다. depth도 안 된다 -- 검은 비닐은 IR을 흡수해
            # 거리가 안 잡히고 뒤쪽 벨트 거리(391mm)가 찍혀 실패
            # 사례(392mm)와 구별되지 않는다.
            #
            # **다음 설계 후보**: 상승 후 물체가 아래에 있을 수 없는
            # 전용 검사 자세(예: 통 위나 충분히 높은 곳)로 옮겨 촬영.
            # 그러면 "물림"과 "빈손"의 배경이 확실히 갈린다. 사이클
            # 시간이 늘어나므로 설계 결정이 필요하다.
            #
            # 그때까지는 **지표만 기록**한다. 근거 없는 판정이 로그에
            # 남으면 다음 사람이 그걸 믿는다.
            verdict = "미판정(지표 검증 실패 -- 위 주석 참고)"

            saved = "-"
            try:
                import cv2  # 저장 실패가 사이클을 깨면 안 되므로 지역 import

                self._grasp_check_dir.mkdir(parents=True, exist_ok=True)
                self._grasp_check_seq += 1
                name = (
                    f"{time.strftime('%H%M%S')}_{self._grasp_check_seq:03d}_"
                    f"{label}_{item_key}.png"
                )
                path = self._grasp_check_dir / name
                bgr = img[:, :, ::-1] if enc == "rgb8" else img
                out = np.ascontiguousarray(bgr).copy()
                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
                cv2.imwrite(str(path), out)
                saved = str(path)
            except Exception as exc:
                saved = f"<저장 실패: {exc}>"

            self.get_logger().info(
                "[GRIPVIEW/%s] %s 판정=%s (배경 %.1f%% / 임계 %.0f%%) "
                "green=%.1f%% wood=%.1f%% gray_mean=%.1f gray_std=%.1f "
                "edge=%.2f %s roi=(%d,%d,%d,%d) enc=%s age=%.2fs saved=%s "
                "-- 기록 전용, 동작 변경 없음"
                % (label, item_key, verdict, bg * 100.0,
                   self._grasp_check_bg_th * 100.0,
                   float(green) * 100.0, float(wood) * 100.0,
                   float(gray.mean()), float(gray.std()), float(edge),
                   self._depth_roi_stats(), x1, y1, x2, y2, enc, age, saved)
            )
        except Exception as exc:  # 진단 기능이 사이클을 깨선 안 된다
            self.get_logger().warn(f"[GRIPVIEW] 기록 실패(무시하고 계속): {exc}")

    def _check_grasp(self, item_key: str) -> bool | None:
        """[8일차, v118 1단계] 파지 성공/실패 판정 -- **로그만, 동작 무변경.**

        RG2는 평행 그리퍼라 닫기 명령 후 손가락이 멈춘 위치가 곧 물체
        유무다: 허공이면 끝까지 닫히고(+0.7496), 물체가 있으면 그 폭에서
        멈춘다. 임계값은 품목별로 `gripper_profiles.yaml`의
        `grasp_fail_joint_rad`에 둔다 -- 단일 임계값은 불가능하다(건전지
        +0.6232가 페트병 기준 임계 +0.5089를 넘어 매번 실패로 오판된다).

        반환: True=파지 성공, False=파지 실패, None=판정 불가/미측정.

        **`plastic_bag`은 항상 None**이다 -- 가장 두껍게 접은 상태가
        +0.7141로 실패값과 0.0355rad 차이뿐이라(종이 재현 편차 0.007의
        5배) 구분이 안 된다. 억지로 판정하면 파지 성공을 실패로 오탐해
        성공률 통계를 오염시킨다.

        **이 반환값으로 사이클을 바꾸지 않는다**(1단계). 2단계(실패 시
        스킵 + 재추적)는 리허설 이후로 미뤄져 있다.
        """
        # 블로킹 직후라 마지막 콜백이 조금 오래됐을 수 있다 -- 판정
        # 직전에 executor를 잠깐 돌려 최신값을 받는다(50Hz 발행이므로
        # 0.3초면 충분하고도 남는다).
        if self._robot_executor is not None:
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                self._robot_executor.spin_once(timeout_sec=0.05)

        th = self._gripper_profiles.get(item_key, {}).get("grasp_fail_joint_rad")
        joint = self._gripper_joint
        if joint is None:
            self.get_logger().warn(
                f"[GRASP] {item_key}: 그리퍼 위치 미수신 -- 판정 건너뜀"
            )
            return None
        if th is None:
            self.get_logger().info(
                f"[GRASP] {item_key}: joint={joint:+.4f} (임계값 없음 -- 판정 제외, 기록만)"
            )
            return None

        ok = joint < th
        self.get_logger().info(
            "[GRASP] %s: joint=%+.4f th=%+.4f margin=%+.4f -> %s"
            % (item_key, joint, th, th - joint, "파지 성공" if ok else "파지 실패")
        )
        if not ok:
            # 1단계에서는 알림만 -- 사이클은 그대로 진행된다.
            self._publish_ui_alert(
                "warn",
                f"{item_key} 파지 실패 의심 (그리퍼 {joint:+.3f} >= {th:+.3f})",
            )
        return ok

    def _on_camera_info(self, msg: CameraInfo) -> None:
        """카메라 생존 확인 전용 -- 내용은 안 쓰고 도착 시각만 기록한다."""
        self._last_image_time = self.get_clock().now().nanoseconds / 1e9

    def _publish_ui_alert(self, level: str, msg: str) -> None:
        """이벤트성 알림 발행 -- 도달 불가, 배치 실패, 무게 초과 라우팅 등
        운영자가 놓치면 안 되는 순간에만 호출한다(주기 발행 아님,
        `/ui/state`와 역할 분리)."""
        payload = {"ts": time.time(), "level": level, "msg": msg}
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self._ui_alert_pub.publish(out)

    def _get_reach_limit_px(self) -> float | None:
        """[10일차, v158] `conveyor.reach_limit_stop_x`(베이스 mm)를 화면
        픽셀 x로 환산해 돌려준다. UI가 "이 선을 넘으면 포기" 경계를
        그리는 데 쓴다(설계안 3절 `reach_limit_px`).

        **역투영 함수를 새로 쓰지 않고 `pixel_to_base`를 스캔한다.**
        정투영 경로는 왜곡 계수와 벨트 평면 보정(`correct_belt_xy`)까지
        검증된 코드인데, 역변환을 따로 구현하면 그 보정을 다시 뒤집어야
        해서 새 오차원이 생긴다. u를 훑어 base_x가 임계를 넘는 지점을
        찾으면 같은 경로를 그대로 재사용할 수 있다.

        결과는 카메라·캘리브레이션이 바뀌지 않는 한 고정이므로 한 번만
        계산해 캐시한다. 계산 실패 시 None을 돌려주고(규격상 필드는
        유지), 다음 호출에서 다시 시도하지 않는다 -- 0.5초마다 도는
        타이머에서 실패를 반복하면 로그만 더럽힌다.
        """
        if self._reach_limit_px is not None:
            return self._reach_limit_px if self._reach_limit_px >= 0 else None
        if not self._is_calibrated():
            return None
        try:
            v = float(self._belt_pixel_y_min) + 120.0   # 벨트 앵커가 실제로 놓이는 행 부근
            prev_u, prev_x = None, None
            found = None
            for u in range(0, int(self._image_width) + 1, 4):
                try:
                    bx = float(self.pixel_to_base(float(u), v)[0])
                except Exception:
                    continue
                if prev_x is not None and prev_x < self._reach_limit_stop_x <= bx:
                    # 선형 보간으로 교차점 픽셀을 구한다
                    t = (self._reach_limit_stop_x - prev_x) / (bx - prev_x)
                    found = prev_u + t * (u - prev_u)
                    break
                prev_u, prev_x = u, bx
            if found is None:
                self.get_logger().warn(
                    "[UI] reach_limit_px 산출 실패 -- base_x가 화면 안에서 "
                    f"{self._reach_limit_stop_x:.0f}mm를 넘지 않는다. null로 발행한다."
                )
                self._reach_limit_px = -1.0
                return None
            self._reach_limit_px = round(found, 1)
            self.get_logger().info(
                "[UI] reach_limit_px=%.1f (base_x %.0fmm, 기준행 v=%.0f)"
                % (self._reach_limit_px, self._reach_limit_stop_x, v)
            )
            return self._reach_limit_px
        except Exception as exc:
            self.get_logger().warn(f"[UI] reach_limit_px 계산 실패: {exc}")
            self._reach_limit_px = -1.0
            return None

    def _publish_ui_state(self) -> None:
        """0.5초 주기(2Hz) 상태 스냅샷 발행.

        `health.robot`은 dry_run 중엔 판단 불가라 `None`(JSON null)로 둔다
        (실제 로봇 없이 True/False로 단정하면 과대 청구 원칙 위반).
        `health.conveyor`는 원본이 `service_is_ready()`를 썼지만 이 노드는
        컨베이어를 서비스가 아니라 `/conveyor_command` 토픽으로 제어하므로,
        구독자(=conveyor_node) 존재 여부로 대체했다 -- 상태 확인을 위해
        실제로 컨베이어를 움직이는 건 부작용이 커서 부적절하다.

        **[알려진 제약]** `stage`가 pick/measure/place/handoff인 블로킹
        구간(17~24초)에는 단일 스레드 executor라 이 타이머 자체가 안 돈다
        (web_ui/README.md에 기록된 아키텍처 제약) -- 화면의 경과 시간이
        클라이언트 로컬 시계로 계산되는 이유다.
        """
        now_ros = self.get_clock().now().nanoseconds / 1e9
        camera_ok = (
            self._last_image_time is not None
            and (now_ros - self._last_image_time) < 2.0
        )
        counts = dict(self._bin_counts)
        payload = {
            "ts": time.time(),
            "state": self._voice_state,
            "stage": self._stage,
            "counts": counts,
            "total": sum(counts.values()),
            "last": self._last_result,
            "reach_limit_px": self._get_reach_limit_px(),
            "health": {
                "camera": camera_ok,
                "robot": None if self._dry_run else True,
                "conveyor": self.count_subscribers("/conveyor_command") > 0,
                "stt": self.count_publishers("/voice_command") > 0,
                "tts": self.count_subscribers("/voice/tts/say") > 0,
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._ui_state_pub.publish(msg)

    def _apply_place_pose_override(
        self, item_key: str, bin_name: str, hover_pose: list, place_pose: list
    ) -> tuple[list, list]:
        """[완료 — 8일차, v87 회신] `config/item_routing.yaml`의
        `place_pose_override[item_key][bin_name]`을 조회해 rx/ry/rz만
        덮어쓴다(x/y/z는 그 통의 원래 좌표 그대로) -- 오버라이드가
        없으면 입력을 그대로 반환(기존 동작 무변화). **품목별이
        아니라 (품목, 통)별**인 이유: `can`처럼 정상 경로(can_bin)와
        무게초과 경로(review_bin)로 같은 품목이 다른 통에 갈 수
        있어서, 품목 하나에 자세 하나로는 부족하다.

        오버라이드 dict에 rx/ry/rz 중 없는 축은 파지 자세
        (`motion_timing.yaml`의 belt_hover_rx/ry/rz)를 기본값으로
        쓴다 -- "회전 없이 파지 각도 그대로 넣는다"는 의도를
        축 단위로 부분 지정할 수 있게 하기 위함.
        """
        override = self._item_routing.get("place_pose_override", {}).get(
            item_key, {}
        ).get(bin_name)
        if not override:
            return hover_pose, place_pose

        mt = self._motion_timing
        rpy = [
            override.get("rx", mt["belt_hover_rx"]),
            override.get("ry", mt["belt_hover_ry"]),
            override.get("rz", mt["belt_hover_rz"]),
        ]
        self.get_logger().info(
            f"[PLACE] place_pose_override 적용: item={item_key} bin={bin_name} rpy={rpy}"
        )
        return hover_pose[:3] + rpy, place_pose[:3] + rpy

    def _say(self, text: str) -> None:
        """TTS 발행 + 콘솔/로그 폴백(v56 요청3) -- TTS가 아직 팀원 쪽과
        최종 연결 전이라 로그로도 항상 확인 가능하게 한다. 팀원 쪽
        TTS 토픽명이 확정되면 `/voice/tts/say` 발행 부분만 바꾸면 됨."""
        self.get_logger().info(f"[TTS] {text}")
        msg = String()
        msg.data = text
        self._voice_tts_pub.publish(msg)

    def _handoff_timeout_phrase(self) -> str:
        """[8일차] STATUS/종료 보고에 붙일 핸드오버 타임아웃 안내.

        타임아웃 재배치는 `_pick_counts`에도 성공 인계와 똑같이 잡히므로
        (기존 집계 호환), 사람이 실제로 못 받아간 건이 몇 건인지 여기서
        따로 알린다. 0건이면 빈 문자열이라 기존 문구가 그대로 유지된다."""
        total = sum(self._handoff_timeout_counts.values())
        if total == 0:
            return ""
        return f" 이 중 {total}개는 받아가지 않아 확인 필요 통으로 옮겼습니다."

    def _report_status(self) -> None:
        total = sum(self._pick_counts.values())
        if total == 0:
            self._say("아직 처리한 물체가 없습니다.")
            return
        parts = ", ".join(f"{k} {v}개" for k, v in self._pick_counts.items())
        self._say(f"지금까지 총 {total}개 처리했습니다. {parts}.{self._handoff_timeout_phrase()}")

    def _report_end_of_run(self) -> None:
        total = sum(self._pick_counts.values())
        if total == 0:
            self._say("작업을 종료합니다. 처리한 물체가 없습니다.")
            return
        parts = ", ".join(f"{k} {v}개" for k, v in self._pick_counts.items())
        self._say(
            f"작업을 종료합니다. 총 {total}개를 처리했습니다. {parts}."
            f"{self._handoff_timeout_phrase()}"
        )

    def _on_voice_command(self, msg: String) -> None:
        """[완료 — 6일차, v55/v56 회신] START/PAUSE/RESUME/STOP/STATUS
        게이팅. 팀원 쪽 STT/TTS가 이미 분류한 명령(`/voice_command`,
        대문자)을 그대로 받는다 -- `.lower()`로 흡수.

        [전제, 미검증] 컨베이어/로봇이 지금 연결 해제 상태라 상태
        전이 로직 자체는 짰지만 실물 검증은 다음 세션. IDLE<->RUNNING
        전이는 로그로 확인 가능(컨베이어 연결만 있으면 `_call_conveyor`
        까지 실물로 확인 가능, 로봇은 불필요).
        """
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"[VOICE] state={self._voice_state} cmd={cmd!r}")

        if cmd == "start":
            if self._voice_state == "IDLE":
                stale_track_count = len(self._tracks)
                self._tracks.clear()
                self.get_logger().info(
                    "[VOICE] start: cleared %d pre-run track(s); "
                    "requiring fresh moving-belt detections"
                    % stale_track_count
                )
                self._voice_state = "RUNNING"
                self._apply_conveyor_state()
                self._say("분리수거를 시작합니다.")
            else:
                self.get_logger().info(f"[VOICE] start 무시(state={self._voice_state})")
        elif cmd == "pause":
            if self._voice_state == "RUNNING":
                self._voice_state = "PAUSED"
                self._apply_conveyor_state()
                # [v56 4-2절] 파지 사이클(_execute_pick/_place_item)은
                # 블로킹이라 이 콜백이 그 도중엔 안 불린다 -- 즉 이
                # PAUSE는 항상 "현재 사이클이 없을 때"에만 처리되므로
                # 물체를 든 채로 멈추는 일은 구조적으로 없다. 다만
                # 사이클 도중 말한 PAUSE는 그 사이클이 끝날 때까지
                # 반영이 늦어질 수 있어 미리 안내한다.
                self._say("현재 작업을 마무리한 뒤 정지합니다.")
            else:
                self.get_logger().info(f"[VOICE] pause 무시(state={self._voice_state})")
        elif cmd == "resume":
            if self._voice_state == "PAUSED":
                stale_track_count = len(self._tracks)
                self._tracks.clear()
                self.get_logger().info(
                    "[VOICE] resume: cleared %d paused track(s); "
                    "requiring fresh moving-belt detections"
                    % stale_track_count
                )
                self._voice_state = "RUNNING"
                self._apply_conveyor_state()
                self._say("작업을 재개합니다.")
            else:
                self.get_logger().info(f"[VOICE] resume 무시(state={self._voice_state})")
        elif cmd == "stop":
            # [팀결정 — v56 5-3절] STOP=작업 종료(마무리 후 보고),
            # 즉시 정지가 아님 -- 음성은 인식 지연이 있어 비상정지에
            # 부적합하다는 근거로 팀원에게 전달함. 물리 비상정지 버튼이
            # 즉시 정지를 담당.
            self._voice_state = "STOPPING"
            self._report_end_of_run()
            self._apply_conveyor_state()
            self._voice_state = "IDLE"
        elif cmd == "status":
            self._report_status()
        else:
            self.get_logger().info(f"[VOICE] unknown command: {cmd!r}")

    def _on_detection(self, msg: String) -> None:
        """Docker YOLO detection -> multi tracking -> predicted Pick."""

        try:
            detections = json.loads(msg.data)

        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f"invalid detection JSON: {exc}"
            )
            return

        # dict 하나가 들어오는 경우에도 대응
        if isinstance(detections, dict):
            detections = [detections]

        if not isinstance(detections, list):
            return

        detections = _deduplicate_cross_class_detections(detections)

        ui_items = []   # [v158] `/ui/detections` 항목 수집용

        # [9일차] **파지 중 큐에 쌓인 검출 버스트를 버린다.**
        # 파지 동작이 메인 executor를 블로킹하는 동안 검출 메시지가
        # 큐에 쌓였다가, 파지가 끝나는 순간 한꺼번에 드레인된다.
        # 실측: 확정에 필요한 9프레임이 **0.008초** 만에 채워졌다
        # (정상은 카메라 7.5Hz 기준 약 1.2초).
        # 결과: (a) 연속프레임 확정 조건이 시간 정보를 전혀 담지 못하고
        # (b) 칼만이 8ms 구간에서 추정한 속도로 8.8초 앞을 외삽해
        # predicted_x가 702mm(도달한계 590 초과)까지 튀었으며
        # (c) 큐의 위치는 파지 전 옛 위치인데 도착 시각은 '지금'이라
        # extra_gap 보정도 안 먹는다.
        # 인터록(파지 중 벨트 정지) 하에서는 옛 위치가 여전히 유효해
        # 드러나지 않았고, **무정지로 바꾸는 순간 드러난 결함**이다.
        #
        # 검출 JSON에 프레임 촬영 시각이 없어서(std_msgs/String, header
        # 없음) 나이로 거를 수가 없다 -- 도착 간격으로 대신 거른다.
        # 카메라가 7.5Hz(약 0.133초)이므로 그보다 훨씬 짧은 간격으로
        # 도착한 것은 실시간 프레임이 아니라 드레인된 백로그다.
        now_arrival = time.monotonic()
        gap = now_arrival - self._last_detection_arrival
        self._last_detection_arrival = now_arrival
        if gap < self._detection_min_gap_sec:
            self._stale_detection_drops += 1
            if self._stale_detection_drops % 20 == 1:
                self.get_logger().warn(
                    f"[DETECT] 백로그 검출 폐기 (도착간격 {gap*1000:.1f}ms "
                    f"< {self._detection_min_gap_sec*1000:.0f}ms, 누적 "
                    f"{self._stale_detection_drops}건)"
                )
            return

        now = (
            self.get_clock()
            .now()
            .nanoseconds
            / 1e9
        )

        matched_ids = set()

        for det in detections:

            if not isinstance(det, dict):
                continue

            if "class_name" not in det:
                continue

            class_name = str(
                det["class_name"]
            )

            confidence = float(
                det.get(
                    "confidence",
                    0.0,
                )
            )

            # ----------------------------------------
            # ZIP 내부 detector와 동일한 anchor 계산
            # bbox 하단 95% 지점
            # ----------------------------------------

            bbox = det.get("bbox")

            if (
                isinstance(
                    bbox,
                    (list, tuple),
                )
                and len(bbox) >= 4
            ):

                x1, y1, x2, y2 = map(
                    float,
                    bbox[:4],
                )

                u = (
                    x1 + x2
                ) / 2.0

                v = (
                    y1
                    + (y2 - y1) * 0.95
                )
                bbox_x1, bbox_x2 = x1, x2
                det_bbox = [x1, y1, x2, y2]   # [v157] 회전각 추정용

            elif (
                "cx" in det
                and "cy" in det
            ):

                # bbox가 없는 경우 fallback
                u = float(det["cx"])
                v = float(det["cy"])
                bbox_x1 = bbox_x2 = None
                det_bbox = None

            else:
                continue

            # 벨트 영역 밖 detection 제외
            if v < self._belt_pixel_y_min:
                continue

            # [9일차] **화면 가로 경계에 걸친 검출을 버린다.**
            #
            # 사용자 지적("캔의 머리가 나올 때 미리 예측해서 잡는다")을
            # 앵커 픽셀로 검증했다 -- 파지 확정 시점의 앵커 u가 12~154로
            # 화면(폭 1280)의 왼쪽 1~12% 구간이었다. 490ml 캔은 화면상
            # 약 336px인데 앵커가 u=63이면 캔의 왼쪽 1/3이 화면 밖이다.
            #
            # 박스가 경계에서 잘리면 그 중심은 물체의 진짜 중심이 아니라
            # **보이는 부분의 중심**이다. 그걸 목표로 삼으니 그리퍼가
            # 물체 중심에서 벗어난 곳에 내려앉고, 무거운 물체는 기울어
            # 벨트에 끌린다(끌리면 무게 일부가 벨트에 실려 net이 실제보다
            # 가볍게 찍힌다 -- 실측 +448g -> -130g).
            #
            # **잘리는 정도가 매번 달라서 상수 보정으로는 못 맞춘다.**
            # 9일차에 can의 prediction_bias_mm을 0 -> 25 -> 45로 세 번
            # 조정했는데 전부 이 잘린 박스를 전제로 맞춘 값이라 무효였다
            # (보정 0에서 27mm 왼쪽, 보정 45에서 37mm 오른쪽 -- 차이 64mm가
            # 보정 변화량 45mm보다 크다).
            #
            # 세로(belt_pixel_y_min)는 이미 거르고 있었는데 가로는 안
            # 걸러지고 있었다. 양쪽 경계 모두 같은 여유값으로 거른다.
            if bbox_x1 is not None and self._edge_margin_px > 0:
                if (
                    bbox_x1 <= self._edge_margin_px
                    or bbox_x2 >= self._image_width - self._edge_margin_px
                ):
                    self._edge_clipped_drops += 1
                    if self._edge_clipped_drops % 50 == 1:
                        self.get_logger().info(
                            "[DETECT] 화면 경계 검출 제외 "
                            "(x1=%.0f x2=%.0f, 여유 %dpx, 누적 %d건)"
                            % (bbox_x1, bbox_x2, self._edge_margin_px,
                               self._edge_clipped_drops)
                        )
                    continue

            # ----------------------------------------
            # Pixel → Robot Base
            # ----------------------------------------

            try:
                base = self.pixel_to_base(
                    u,
                    v,
                )

            except RayPlaneParallelError:
                continue

            except Exception as exc:

                self.get_logger().error(
                    f"pixel_to_base failed "
                    f"for {class_name}: {exc}"
                )

                continue

            # ----------------------------------------
            # 기존 Track 매칭
            # ----------------------------------------

            track = self._match_track(
                class_name,
                float(base[0]),
                now,
                matched_ids,
            )

            if track is None:

                track = self._new_track(
                    class_name
                )

                self._tracks.append(
                    track
                )

            track["last_y"] = float(
                base[1]
            )

            track["last_z"] = float(
                base[2]
            )

            track["last_bbox"] = det_bbox   # [v157]
            track["last_confidence"] = confidence

            # ----------------------------------------
            # Kalman
            # ----------------------------------------

            if (
                track[
                    "last_detection_time"
                ]
                is None
            ):

                track["kf"].reset(
                    float(base[0])
                )

            else:

                dt = (
                    now
                    - track[
                        "last_detection_time"
                    ]
                )

                if dt > 0:

                    track["kf"].update(
                        float(base[0]),
                        dt,
                    )

            track[
                "last_detection_time"
            ] = now

            track[
                "pending_count"
            ] += 1

            track[
                "lost_count"
            ] = 0

            matched_ids.add(
                id(track)
            )

            # ----------------------------------------
            # 좌표 확인용 topic
            # ----------------------------------------

            result = {
                "class_id":
                    det.get("class_id"),

                "class_name":
                    class_name,

                "confidence":
                    confidence,

                "cx":
                    u,

                "cy":
                    v,

                "base_x":
                    float(base[0]),

                "base_y":
                    float(base[1]),

                "base_z":
                    float(base[2]),

                "kf_x":
                    float(
                        track["kf"].state.x
                    ),

                "kf_vx":
                    float(
                        track["kf"].state.vx
                    ),

                "pending_count":
                    int(
                        track[
                            "pending_count"
                        ]
                    ),
            }

            out = String()

            out.data = json.dumps(
                result,
                ensure_ascii=False,
            )

            self._base_detection_pub.publish(
                out
            )

            self.get_logger().info(
                "class=%s conf=%.2f "
                "anchor=(%.1f,%.1f) "
                "kf_x=%.1f "
                "kf_vx=%.2f "
                "pending=%d "
                "confirmed=%s "
                "holding=%s"
                % (
                    class_name,
                    confidence,
                    u,
                    v,
                    track["kf"].state.x,
                    track["kf"].state.vx,
                    track[
                        "pending_count"
                    ],
                    track[
                        "pending_count"
                    ]
                    >=
                    self._min_consecutive_frames,
                    self._holding,
                )
            )

            # [v158] `/ui/detections` 항목 수집. 규격:
            # items[{cls, conf, bbox, anchor, pending, required, locked}]
            _pend = int(track["pending_count"])
            ui_items.append({
                "cls": class_name,
                "conf": round(float(confidence), 4),
                "bbox": det_bbox,
                "anchor": [round(float(u), 1), round(float(v), 1)],
                "pending": _pend,
                "required": int(self._min_consecutive_frames),
                "locked": _pend >= self._min_consecutive_frames,
            })

        # [v158] 프레임 단위 발행. 검출이 0건이어도 발행한다 --
        # UI가 "이전 프레임 잔상"을 지우려면 빈 목록이 필요하다.
        try:
            _m = String()
            _m.data = json.dumps({
                "ts": time.time(),
                "w": self._image_width,
                "h": self._image_height,
                "items": ui_items,
            }, ensure_ascii=False)
            self._ui_det_pub.publish(_m)
        except Exception as exc:  # UI 발행이 파지 파이프라인을 깨선 안 된다
            self.get_logger().warn(f"[UI] /ui/detections 발행 실패: {exc}")

        # --------------------------------------------
        # 미검출 Track 관리
        # --------------------------------------------

        for track in list(
            self._tracks
        ):

            if (
                id(track)
                not in matched_ids
            ):

                track[
                    "lost_count"
                ] += 1

                if (
                    track[
                        "lost_count"
                    ]
                    >=
                    self._object_lost_frames
                ):

                    self._tracks.remove(
                        track
                    )

        # [9일차, v139] 도달한계 안전정지 -- 가장 앞선 미처리 물체가
        # 로봇 도달한계에 근접하면 벨트를 세운다. 들고 있는 중에도
        # 평가해야 한다(파지 중에 뒤 물체가 한계에 닿을 수 있음).
        self._update_reach_limit_hold()

        # 물체를 들고 있으면 새 Pick 금지
        if self._holding:
            # stage는 지금 진행 중인 단계(pick/measure/place/handoff)를
            # 유지한다 -- 여기서 덮어쓰면 화면 진행 표시가 뒤로 튄다.
            return

        # START 전에는
        # Detection/Tracking만 수행
        if (
            self._voice_state
            != "RUNNING"
        ):
            self._stage = "idle"
            return

        self._stage = "track" if self._tracks else "detect"

        candidates = [
            track
            for track
            in self._tracks
            if (
                track[
                    "pending_count"
                ]
                >=
                self._min_consecutive_frames
                # [9일차] 워치독이 포기한 트랙은 파지 후보에서도 뺀다.
                # 도달한계 판정에서만 빼면 계속 선택됐다 실패한다.
                and track["track_id"] not in self._abandoned_track_ids
                # [9일차, v141] 이미 도달한계를 넘은 트랙은 즉시 제외한다
                # -- 워치독 120초를 기다릴 이유가 없다.
                and not self._is_unreachable(track)
            )
        ]

        if not candidates:
            return

        self._stage = "predict"

        # 벨트 진행방향에서
        # 가장 앞선 객체 우선
        best = max(
            candidates,
            key=lambda track:
                track["kf"].state.x,
        )

        self._tracks.remove(
            best
        )

        try:

            predicted = (
                self.predict_pickup_point(
                    best
                )
            )

            # [v157] 회전각 추정 -- 트리거 시점(블로킹 이전)이라 최신
            # 고정캠 프레임이 들어와 있다. 로그만 남기고 rz는 안 바꾼다.
            self._estimate_rotation(best)

            self.get_logger().info(
                "PICK TRIGGER "
                "class=%s "
                "predicted_pickup(preview)=%s "
                "dry_run=%s "
                "remaining_tracks=%d"
                % (
                    best[
                        "class_name"
                    ],
                    predicted,
                    self._dry_run,
                    len(
                        self._tracks
                    ),
                )
            )

            # 실제 로봇은 dry_run=False일 때만
            if not self._dry_run:

                self._execute_pick(
                    best[
                        "class_name"
                    ],
                    best,
                )

        except (
            RuntimeError,
            KeyError,
        ) as exc:

            self.get_logger().warn(
                "pick prediction/execution "
                f"skipped: {exc}"
            )


    def predict_pickup_point(self, track: dict) -> np.ndarray:
        """Predicted (x, y, z) at the moment the gripper will be fully closed.

        예측 시간 = 이동 + 하강 + 닫힘 (v41/v44 설계). `track["class_name"]`
        은 `config/gripper_profiles.yaml`의 키(예: "can", "battery")와
        일치해야 한다. [완료 — 6일차, 순차 처리 인덱싱] 물체별 트랙을
        인자로 받도록 변경(이전엔 전역 `self._kf`/`self._last_y/z` 참조).
        """
        if track["last_y"] is None or track["last_z"] is None:
            raise RuntimeError("track has no detection yet")
        ahead = predict_ahead_sec(
            track["class_name"], self._motion_timing, self._gripper_profiles
        )
        predicted_x = self._predict_x(track, ahead)
        return np.array([predicted_x, track["last_y"], track["last_z"]])

    # -- 로봇 실행 (dry_run=False 전용) -----------------------------------

    def _call_move_line(self, pos, vel, acc, mode=0):
        req = MoveLine.Request()
        req.pos = list(pos)
        req.vel = list(vel)
        req.acc = list(acc)
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0
        req.mode = mode
        req.blend_type = 0
        req.sync_type = 0
        future = self._move_line_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=20.0)
        # [8일차] 무게 측정 로그에 "직전 이동 종료 후 경과시간"을 남기기
        # 위한 시각. settle 부족이 측정에 섞이는지 사후 판정용이다.
        self._last_move_done_t = time.monotonic()
        return future.result()

    def _call_gripper(self, command):
        """[9일차, v147] 응답 대기를 8.0 -> 2.0초로 단축.

        **무정지 전환으로 드러난 문제**: 그리퍼가 응답을 안 주면 8초를
        꽉 채워 기다리는데, 그 사이 벨트는 계속 흐른다. 실측 60회 중
        4회가 여기서 걸려 `close gripper -> rise`가 10초(=8초 타임아웃
        + close_wait 2초)였고, 벨트 물체속도 2.9mm/s 기준 **물체가 약
        23mm 끌렸다.** 인터록에서는 파지 중 벨트가 서 있어 드러나지
        않던 항목이다.

        **단축이 안전한 근거**: 그리퍼 `'c'`는 비동기라 응답이 물리적
        완료를 뜻하지 않는다는 게 이 프로젝트의 확립된 전제이고
        (CLAUDE.md), 물리적 완료는 호출부의 `close_wait_sec` sleep이
        타임아웃과 **별개로** 보장한다. `call_async`라 명령 자체는
        이미 전송돼 있고 응답만 유실/지연된 것이며, 실제로 그 4회 모두
        파지에 성공했다. 정상 사이클은 1~3초 안에 끝나므로 2초면
        정상 응답에는 여유가 충분하다.

        근본 원인(Modbus 폴링 지연 또는 응답 유실)은 시연 이후 조사.
        """
        req = SetCommand.Request()
        req.command = command
        future = self._gripper_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=2.0)
        return future.result()

    def _get_current_posx(self):
        from dsr_msgs2.srv import GetCurrentPosx
        req = GetCurrentPosx.Request()
        req.ref = 0
        future = self._get_posx_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=10.0)
        result = future.result()
        return list(result.task_pos_info[0].data)

    def _get_workpiece_weight(self) -> float | None:
        """Doosan 표준 API로 현재 들고 있는 물체 무게 추정(kg), 1회 측정.

        원칙7(비전으로 못 보는 걸 힘으로 본다) 구현. 실패(success=False
        또는 음수 weight)시 None 반환.

        **[실측, 정정 — 6일차, 순차 처리 인덱싱 테스트]** "관성력 섞임
        방지" 가설(로봇/벨트 정지 후 settle 대기)은 기각됨 -- 벨트
        완전 정지+로봇 완전 정지 상태에서도 빈 그리퍼 단일 측정이
        8회에 걸쳐 45g 폭(1.310~1.355kg)으로 흔들렸고, 벨트 가동
        상태(21g 폭)와 비교해도 큰 차이가 없었다. **정정된 진단:
        힘센서 자체의 고유 노이즈가 threshold(30g)와 비슷한 규모다**
        -- 벨트/로봇 정지 여부와 무관. 단일 측정 대신 `_get_workpiece_
        weight_avg()`(평균) 사용을 권장, 이 메서드는 단발 측정이
        필요한 경우에만 남겨둔다.
        """
        from dsr_msgs2.srv import GetWorkpieceWeight
        req = GetWorkpieceWeight.Request()
        future = self._weight_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.success or result.weight < 0:
            self.get_logger().warn(f"get_workpiece_weight failed: {result}")
            return None
        return float(result.weight)

    def _get_workpiece_weight_avg(self, n: int = 5, interval: float = 0.15) -> float | None:
        """`_get_workpiece_weight()`를 n회 측정해 평균 -- 힘센서 고유
        노이즈(단일 측정 20~45g 폭 확인됨) 완화용. 유효한 측정이 하나도
        없으면 None."""
        readings = []
        for _ in range(n):
            w = self._get_workpiece_weight()
            if w is not None:
                readings.append(w)
            time.sleep(interval)
        if not readings:
            return None
        return sum(readings) / len(readings)

    def _weigh_with_log(self, label: str, n: int = 5, interval: float = 0.15,
                        drop_first: int = 1, n_log: int | None = None,
                        use_median: bool = False):
        """[8일차] 무게 측정 + **원시샘플·자세·직전이동 경과시간 로깅**.

        `_get_workpiece_weight_avg()`와 계산 결과는 동일하고(평균), 진단에
        필요한 맥락을 로그로 남기는 것만 다르다. **판정에는 관여하지
        않는다** -- 호출부는 반환된 평균값을 기존과 똑같이 쓴다.

        남기는 이유: `net = post-pick - baseline`의 요동이 힘센서 고유
        노이즈가 아니라 **두 측정의 조건 차이**(자세 109~144mm 차이,
        baseline만 벨트 가동 중)라는 게 8일차에 규명됐다. 원인을 확정
        하려면 평균값이 아니라 **각 샘플의 원시값**과 그 시점의 자세가
        필요하다(6일차엔 평균만 남겨서 사후 분석이 불가능했다).

        `_last_move_done_t`는 직전 이동이 끝난 시각으로, 측정까지의
        settle 시간을 재기 위한 것이다.

        **[완료 — 8일차, v112] `drop_first`로 첫 샘플을 버린다.**
        실측 3건에서 첫 샘플이 나머지와 60~107g 어긋났고(0.75초 측정
        구간 안에서 단조 증가/감소), 전체평균과 뒤 3개 평균이 30g
        차이났다 -- threshold 80g 기준 무시 못 할 크기다.
        `since_last_move`가 4.35초로 충분히 길었는데도 그렇다는 건
        이동 후 settle이 아니라 **측정 자체의 초기 과도**로 보인다
        [추론]. 버린 뒤 유효 샘플이 1개 이하로 남으면 버리지 않는다.
        로그에는 `avg_all`(버리기 전 평균)과 원시샘플 전량을 같이
        남겨 사후 비교가 가능하게 한다.
        """
        # [완료 — 8일차, v115] **판정은 앞 n개로만, 로그는 n_log개까지.**
        # 8일차 실측에서 샘플 내 추세가 회차마다 갈렸다(단조 감소/단조
        # 증가/무작위). 웹 클로드 가설: "물리 정지 미보장"이 아니라
        # **물체 형상별 잔류진동 주파수 차이**일 수 있다 -- 페트병은
        # 길쭉해서 저주파, 캔은 짧은 원통이라 고주파라, 0.75초(5샘플)
        # 구간에 담기는 진동 주기 수가 달라 단조로도 무작위로도 보인다.
        # 검증하려면 더 긴 구간이 필요한데, **판정 구간을 늘리면 사이클
        # 시간이 늘고 지금까지 쌓은 데이터와 조건이 달라진다** -- 그래서
        # 판정은 앞 n개 그대로 두고 로그만 n_log까지 이어 받는다.
        # 품목별(can vs pet_labeled)로 패턴이 갈리면 가설이 지지된다.
        total = n_log if n_log is not None and n_log > n else n
        raw = []
        for _ in range(total):
            w = self._get_workpiece_weight()
            if w is not None:
                raw.append(w)
            time.sleep(interval)

        judged = raw[:n]  # 판정에 쓰는 구간 -- 기존과 동일
        readings = judged[drop_first:] if len(judged) > drop_first + 1 else judged

        try:
            pos = self._get_current_posx()
            pos_s = "(%.1f,%.1f,%.1f,%.1f,%.1f,%.1f)" % tuple(pos[:6])
        except Exception as exc:  # 진단 로깅이 파지 사이클을 깨선 안 된다
            pos_s = f"<자세 읽기 실패: {exc}>"

        since_move = (
            time.monotonic() - self._last_move_done_t
            if self._last_move_done_t is not None
            else float("nan")
        )
        # [9일차] baseline은 **중앙값**을 쓴다. 벨트 가동 중에 재야 하는
        # 구간이라(예측 구간 total_ahead 안이므로 세울 수 없다) 진동으로
        # 한 샘플이 크게 튀면 평균이 통째로 끌려간다 -- 실측에서
        # baseline 1373.5g / post-pick 1235.9g로 net이 -137.6g까지
        # 틀어졌다. 샘플 수와 소요 시간은 그대로 두고 집계만 바꾼다.
        if use_median and judged:
            # **판정 표본 전체(drop_first 미적용)에 중앙값을 쓴다.**
            # baseline은 n=3인데 drop_first=1을 적용하면 표본이 2개가
            # 되어 중앙값이 평균과 같아진다(가운데 두 값의 평균).
            # 중앙값은 초기 과도 샘플도 자연히 배제하므로 drop_first가
            # 하던 일을 겸한다.
            avg = statistics.median(judged)
        else:
            avg = sum(readings) / len(readings) if readings else None
        avg_judged = sum(judged) / len(judged) if judged else None
        self.get_logger().info(
            "[WEIGHLOG] %s n=%d/%d avg=%s (avg_nodrop=%s dropped=%d) "
            "samples_all=%s (총 %d개, 판정은 앞 %d개) "
            "pose=%s since_last_move=%.2fs conveyor_running=%s"
            % (label, len(readings), n,
               f"{avg*1000:.1f}g" if avg is not None else "None",
               f"{avg_judged*1000:.1f}g" if avg_judged is not None else "None",
               len(judged) - len(readings),
               "[" + ", ".join(f"{r*1000:.1f}" for r in raw) + "]",
               len(raw), n,
               pos_s, since_move, self._conveyor_running)
        )
        return avg

    def _call_task_compliance_ctrl(self, stx, ref: int = 0, time_: float = 0.0):
        req = TaskComplianceCtrl.Request()
        req.stx = list(stx)
        req.ref = ref
        req.time = time_
        future = self._task_compliance_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=5.0)
        return future.result()

    def _call_release_compliance_ctrl(self):
        req = ReleaseComplianceCtrl.Request()
        future = self._release_compliance_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=5.0)
        return future.result()

    # ----------------------------------------------------------------
    # [9일차, v139] 정지 사유 집합 기반 컨베이어 제어
    # ----------------------------------------------------------------
    # 이름을 stop/resume이 아니라 hold/release로 둔 이유: **release가 곧
    # 가동은 아니다**(집합이 안 비면 계속 정지). 코드를 읽을 때 바로
    # 드러나게 하기 위한 것.

    def _hold(self, reason: str) -> None:
        if reason not in self._hold_reasons:
            self._hold_reasons.add(reason)
            self._hold_since[reason] = time.time()
            self.get_logger().info(
                f"[HOLD] +{reason} (사유집합={sorted(self._hold_reasons)})"
            )
        self._apply_conveyor_state()

    def _release(self, reason: str, cause: str = "normal") -> None:
        if reason in self._hold_reasons:
            held = time.time() - self._hold_since.pop(reason, time.time())
            self._hold_reasons.discard(reason)
            self.get_logger().info(
                f"[HOLD] -{reason} 유지 {held:.1f}s 해제원인={cause} "
                f"(사유집합={sorted(self._hold_reasons)})"
            )
        self._apply_conveyor_state()

    def _apply_conveyor_state(self) -> None:
        """사유집합/비상정지/음성상태를 종합해 실제 모터 상태를 정한다."""
        if self._emergency_stop:
            desired = False
        elif self._voice_state != "RUNNING":
            desired = False
        elif self._hold_reasons:
            desired = False
        else:
            desired = True

        if desired != self._conveyor_running:
            self._call_conveyor("on" if desired else "off")

    def _check_hold_watchdog(self) -> None:
        """사유가 안 풀린 채 오래 지나면 통째로 비우고 경고."""
        if not self._hold_reasons:
            return
        oldest = min(self._hold_since.values())
        held = time.time() - oldest
        if held < self._hold_watchdog_sec:
            return

        stuck = sorted(self._hold_reasons)
        self.get_logger().warn(
            f"[HOLD] 워치독 {held:.1f}s 초과 -- 사유집합 강제 해제 {stuck}"
        )
        self._publish_ui_alert(
            "warn",
            f"컨베이어 정지 사유가 {held:.0f}초간 안 풀려 강제 재가동 ({', '.join(stuck)})",
        )
        if "reach_limit" in stuck:
            # 이 트랙을 포기하지 않으면 재가동 직후 다시 한계에 서 있어
            # 즉시 재정지된다. 파지 후보에서도 빼야 계속 선택됐다 실패하는
            # 낭비가 안 생긴다.
            abandoned = []
            for track in self._tracks:
                x = self._track_front_x(track)
                if x is not None and x >= self._reach_limit_stop_x:
                    self._abandoned_track_ids.add(track["track_id"])
                    abandoned.append(f"{track['class_name']}@{x:.0f}mm")
            if abandoned:
                self.get_logger().warn(
                    f"[HOLD] 도달한계 포기 트랙: {', '.join(abandoned)}"
                )
                self._publish_ui_alert(
                    "warn",
                    f"물체 포기: 도달한계 대기 시간 초과 ({', '.join(abandoned)})",
                )
        for reason in stuck:
            self._release(reason, cause="워치독 강제")

    def _track_front_x(self, track: dict) -> float | None:
        try:
            return float(track["kf"].state.x)
        except Exception:
            return None

    def _predict_x(self, track: dict, ahead: float) -> float:
        """[9일차, v143] 속도를 실측 범위로 클램프한 위치 예측.

        **문제**: 관측 구간은 약 1.0초(9프레임 확정)인데 외삽 구간은
        7~11초다 -- 관측의 8배를 외삽한다. 그 1초 창에서 추정한
        속도가 참값(3.06mm/s)에서 2~24배씩 양방향으로 벗어난다:
            실측 3.18mm/s 구간에서 칼만이 34.12mm/s로 추정
        34mm/s를 9초 외삽하면 307mm -- 물체보다 30cm 앞을 집는다.
        실제로 "허공 파지"와 "도달한계 초과(632mm)"가 이것 때문이었다.

        **조치**: 벨트는 정속 장치이고 역주행하지 않으므로, 예측에
        쓰는 속도만 [vx_min, vx_max]로 자른다. 필터 상태 자체는
        건드리지 않는다(추적/매칭에는 원래 추정을 그대로 쓴다).
        참값 기준 9초 이동량은 약 28mm로 그리퍼 여유 안이므로,
        이상치만 잘라내도 예측은 충분히 맞는다.

        근본 대책(벨트 속도를 상수로 고정)은 prediction_bias_mm
        의존관계까지 정리해야 해서 시연 이후 과제로 미뤄져 있다.
        """
        st = track["kf"].state
        vx = min(max(st.vx, self._vx_min), self._vx_max)
        if vx != st.vx:
            self._vx_clamp_count += 1
            if self._vx_clamp_count % 10 == 1:
                self.get_logger().info(
                    f"[PREDICT] vx 클램프 {st.vx:+.2f} -> {vx:+.2f}mm/s "
                    f"(누적 {self._vx_clamp_count}회)"
                )
        return st.x + vx * ahead

    def _is_unreachable(self, track: dict) -> bool:
        """[9일차, v141] 이미 도달한계를 넘어선 트랙인가.

        무정지에서는 파지가 블로킹하는 동안 도달한계 판정이 돌지 않아
        물체가 한계를 넘어가 버린다(실측: 540 정지선을 지나 632mm).
        그 트랙은 `_execute_pick`이 매번 "도달 불가"로 중단하는데,
        호출부가 트랙을 지우고 다음 프레임에 새 트랙이 생기므로
        **1.6초마다 무한 재시도**가 된다(워치독 120초까지).

        트랙 id로는 못 막는다 -- 매번 새 id가 부여되기 때문이다.
        그래서 **위치로 판정**한다: 예측은 거리를 더하기만 하므로
        현재 x가 이미 한계를 넘었으면 어떤 경우에도 도달 불가다.

        놓친 물체는 벨트 끝으로 흘러가 사람이 회수한다 -- 원 설계가
        이미 안전한 실패모드로 분류해둔 경로다(낙하·충돌 아님).
        """
        max_x = self._motion_timing.get("max_reachable_x")
        if max_x is None:
            return False
        x = self._track_front_x(track)
        if x is None or x <= max_x:
            return False
        tid = track["track_id"]
        if tid not in self._abandoned_track_ids:
            self._abandoned_track_ids.add(tid)
            self.get_logger().warn(
                f"[EXEC] {track['class_name']} 도달한계 초과(x={x:.1f} > {max_x}) "
                "-- 즉시 포기, 벨트 끝으로 흘려보냄"
            )
            self._publish_ui_alert(
                "warn", f"{track['class_name']} 도달 범위 밖 — 포기"
            )
        return True

    def _update_reach_limit_hold(self) -> None:
        """가장 앞선 미처리 물체가 도달한계에 근접하면 벨트를 세운다."""
        front = None
        for track in self._tracks:
            if track["track_id"] in self._abandoned_track_ids:
                continue
            # 이미 한계를 넘은 물체 때문에 벨트를 세우면 영원히 안 풀린다.
            if self._is_unreachable(track):
                continue
            x = self._track_front_x(track)
            if x is not None and (front is None or x > front):
                front = x

        if front is not None and front >= self._reach_limit_stop_x:
            self._hold("reach_limit")
        else:
            self._release("reach_limit")

    def _call_conveyor(self, cmd: str) -> None:
        """현재 검증된 /conveyor_command 방식."""

        msg = String()

        if cmd == "off":
            msg.data = "STOP"
            self._conveyor_running = False
        else:
            msg.data = "START"
            self._conveyor_running = True

        self._conveyor_pub.publish(
            msg
        )

        self.get_logger().info(
            f"[CONVEYOR] "
            f"/conveyor_command -> "
            f"{msg.data}"
        )


    def _execute_pick(self, item_key: str, track: dict) -> None:
        """벨트에서 예측 좌표로 이동->하강->파지->상승.

        [완료 — 6일차, 순차 처리 인덱싱] 물체별 트랙(`track`)을 인자로
        받도록 변경(이전엔 전역 `self._kf`/`self._last_detection_time`
        참조 -- 여러 물체를 동시에 추적하게 되면서 어떤 트랙을 파지
        중인지 명시적으로 넘겨야 한다).

        **[버그 발견·수정 — 6일차, 벨트 이동 3차 시도]** 처음엔 목표
        좌표를 `_on_image`에서 미리 계산해 여기 인자로 넘겨받았다 --
        그런데 그리퍼 열기(0.5s)+파지력 정규화(최대 28회 명령,
        실측 약 1.7s)를 하는 동안 로봇은 아직 안 움직이는데, 이
        약 2.2초가 `predict_ahead_sec()`(이동+하강+닫힘)에는 전혀
        반영이 안 됐다 -- 그 결과 예측이 항상 실제보다 덜 간
        지점으로 나와(과소예측) 캔 하단부 끝만 겨우 잡는 실패가
        재현됨(사용자 실측 지적). **수정: 목표 좌표를 힘 정규화가
        다 끝난 "지금"을 기준으로 다시 계산**한다 -- 칼만필터의
        마지막 갱신 시각부터 지금까지 흐른 실제 시간을 예측 시간에
        더해서 그 시점 기준으로 이동을 시작하면, 힘 정규화에 걸린
        시간도 예측에 자연히 포함된다.

        CLAUDE.md "알려진 함정" 준수: (1) MoveLine 이후 항상
        get_current_posx로 실제 이동 확인, (2) 파지력은 세션 내내
        유지되므로 매 파지 직전 명시적으로 정규화(우선 'i'로 최대까지
        올린 뒤 목표 force_n까지 'd'로 내림), (3) 닫기 후 close_wait_sec
        만큼 대기.
        """
        mt = self._motion_timing
        profile = self._gripper_profiles[item_key]
        depth = profile.get("pick_depth_mm", 100.0)
        h_vel = [mt["horizontal_vel"], mt["horizontal_acc"]]
        v_vel = [mt["vertical_vel"], mt["vertical_acc"]]

        # [실측 — 6일차, 사고] 그리퍼가 열려있다고 가정하면 안 된다 --
        # 이전 사이클이 닫힌 채로 끝났다면 그 상태 그대로 하강해
        # 캔을 찌그러뜨린 사고가 실제로 났다. 매 파지 시작 시 반드시
        # 먼저 명시적으로 연다.
        self.get_logger().info("[EXEC] open gripper (approach 전 확인)")
        self._call_gripper("o")
        time.sleep(0.5)

        force_n = profile.get("force_n")
        target_force = force_n if force_n is not None else 40.0
        if self._last_force_n is None:
            self.get_logger().info(
                f"[EXEC] force normalize for {item_key} (첫 파지, 절대값 모름 -- "
                "최대까지 올린 뒤 목표까지 내림)"
            )
            for _ in range(16):  # 3~40N 범위를 확실히 덮도록 최대까지 올림
                self._call_gripper("i")
                time.sleep(0.05)
            steps_down = round((40.0 - target_force) / 2.5)
            for _ in range(max(steps_down, 0)):
                self._call_gripper("d")
                time.sleep(0.05)
        else:
            diff = target_force - self._last_force_n
            steps = round(diff / 2.5)
            self.get_logger().info(
                f"[EXEC] force adjust for {item_key}: {self._last_force_n}N -> "
                f"{target_force}N ({steps:+d} steps, 마지막 설정값 추적)"
            )
            cmd = "i" if steps > 0 else "d"
            for _ in range(abs(steps)):
                self._call_gripper(cmd)
                time.sleep(0.05)
        self._last_force_n = target_force

        # [버그 발견·수정 — 6일차, 무게 평균측정 도입 직후] baseline
        # 측정(`_get_workpiece_weight_avg`, 5회 평균 -- 힘센서 노이즈
        # 대응으로 오늘 추가)이 원래 "move to hover" 이후·"descend"
        # 이전에 있었는데, 이 구간(~1~2초)이 `extra_gap` 계산 시점
        # (아래) 이후에 발생해서 예측 시간에 전혀 반영이 안 됐다 --
        # 그 결과 실제 하강 시점이 예측보다 늦어져 캔 하부(트레일링
        # 엣지)만 잡는 실패가 재현됨(사용자 실측 지적: "하강 전
        # 대기시간이 너무 길다", "예상 위치 빗나가서 하부만 잡음").
        # **수정**: baseline 측정을 재예측(extra_gap) 계산 *이전*으로
        # 옮긴다 -- 이러면 이 시간이 자동으로 extra_gap에 포함된다.
        # **[추가 조정]** 그런데도 이 구간이 예측 구간(total_ahead)을
        # 늘려 칼만필터 속도 추정 오차를 증폭시킨다(실측: extra_gap
        # =6.5초, total_ahead=13.4초까지 늘어나며 하부만 잡는 실패
        # 재현) -- 파지 성공이 무게 정밀도보다 우선이므로, baseline은
        # 더 빠른 설정(3회, 0.08초 간격)으로 줄인다. post-pick 측정
        # (벨트 이미 정지, 그리퍼-크리티컬 경로 아님)은 기본값(5회)
        # 그대로 유지 -- 거기는 시간이 늘어도 다음 파지에 영향 없다.
        weight_check = self._item_routing.get("weight_check", {}).get(item_key)
        baseline_weight = (
            # [정정 — 8일차, v118] **can-baseline만 n_log를 안 준다.**
            # baseline은 벨트 가동 중, extra_gap 스냅샷 이전에 실행되므로
            # 측정 시간이 그대로 `total_ahead`에 더해진다 -- 시간 자체는
            # extra_gap에 자동 반영되어 under-prediction 버그는 안 나지만,
            # **total_ahead가 커지면 칼만필터가 더 먼 미래를 예측해야 해서
            # 오차가 증폭된다.** 6일차에 baseline을 5샘플->3샘플로 줄인
            # 이유가 정확히 이것이었다. 15샘플(1.2초)은 3샘플(0.24초)의
            # 5배라 그 회귀를 그대로 재현한다.
            # 진동 패턴 확인은 시간이 안 중요한 지점(can-postpick,
            # handoff-*-presented)에서만 한다.
            self._weigh_with_log(f"{item_key}-baseline", n=3, interval=0.08, use_median=True)
            if weight_check is not None
            else None
        )

        now = self.get_clock().now().nanoseconds / 1e9
        extra_gap = now - track["last_detection_time"]
        total_ahead = extra_gap + predict_ahead_sec(
            item_key, self._motion_timing, self._gripper_profiles
        )
        predicted_x = self._predict_x(track, total_ahead)
        bias = profile.get("prediction_bias_mm", 0.0)
        if bias:
            self.get_logger().info(f"[EXEC] applying prediction_bias_mm={bias} for {item_key}")
            predicted_x += bias
        hover = [
            float(predicted_x), mt["belt_hover_y"], mt["belt_hover_z"],
            mt["belt_hover_rx"], mt["belt_hover_ry"], mt["belt_hover_rz"],
        ]
        self.get_logger().info(
            "[EXEC] re-predicted after force-normalize: extra_gap=%.3fs "
            "total_ahead=%.3fs predicted_x=%.2f (힘정규화 지연 반영됨)"
            % (extra_gap, total_ahead, predicted_x)
        )

        max_x = mt.get("max_reachable_x")
        if max_x is not None and predicted_x > max_x:
            self.get_logger().warn(
                f"[EXEC] predicted_x={predicted_x:.1f} > max_reachable_x={max_x} "
                "-- 표준 자세로 도달 불가, MoveLine 시도 없이 중단(물체가 벨트 "
                "끝쪽에서 너무 늦게 확정됨 -- 벨트 시작쪽에 더 가깝게 놓고 재시도할 것)"
            )
            self._publish_ui_alert(
                "warn", f"{item_key} 도달 범위 밖 — 파지 건너뜀"
            )
            # [완료 — 6일차, 순차 처리 인덱싱] 이 트랙은 `_on_image`에서
            # 이미 self._tracks에서 제거된 뒤 여기로 넘어왔다(호출부
            # 참고) -- 별도로 락을 풀 필요 없이, 물체가 계속 보이면
            # 다음 프레임에 새 트랙으로 자연스럽게 재생성돼 재시도된다.
            return

        self.get_logger().info(f"[EXEC] move to hover {hover}")
        self._call_move_line(hover, h_vel, h_vel, mode=0)
        pos = self._get_current_posx()
        if abs(pos[0] - hover[0]) > 5.0 or abs(pos[1] - hover[1]) > 5.0:
            self.get_logger().error(
                f"[EXEC] hover move did not reach target (got {pos[:3]}, "
                f"wanted {hover[:3]}) -- aborting pick"
            )
            self._publish_ui_alert("error", "벨트 상공 이동 실패 — 파지 중단")
            self._stage = "idle"
            # [버그 발견·수정 — 6일차, 핸드오버 1차 실물 시도] 예전 단일
            # 트랙(`_locked_class`) 설계에서는 여기서 락을 안 풀면 물체가
            # 계속 화면에 보이는 한 재시도 자체가 영원히 막히는 버그가
            # 있었다(사용자가 "로봇 안움직여"로 발견). **[완료 — 순차
            # 처리 인덱싱 재설계]** 이제는 트랙이 `_on_image`에서 이미
            # self._tracks에서 제거된 뒤 여기로 넘어오므로, 물체가 계속
            # 보이면 다음 프레임에 새 트랙으로 자연스럽게 재생성돼
            # 재시도된다 -- 같은 효과를 구조적으로 유지.
            return

        self._stage = "pick"
        self.get_logger().info(f"[EXEC] descend {depth}mm")
        z_before = None
        try:
            z_before = self._get_current_posx()[2]
        except Exception:
            pass
        self._call_move_line([0.0, 0.0, -depth, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)

        # [9일차 진단] **하강 도달 z를 로그로 남긴다.** `hover` 이동에는
        # 도달 검증이 있는데 하강에는 없어서, 명령값과 실제 도달값이
        # 다른지 확인할 방법이 아예 없었다. 동작은 바꾸지 않고 기록만
        # 한다(도달 실패해도 중단하지 않음 -- 이미 집으러 내려온 상태에서
        # 중단하면 오히려 어정쩡하게 멈춘다).
        #
        # 참고 기준: 벨트 평면 z는 config상 316.8, 실측 파지점 319.6~320.2.
        # battery 110mm(TCP z=320)가 "표면을 거의 스치는" 높이라는 것이
        # 사용자 실물 감각이고 config 좌표와도 일치한다. z가 310 아래로
        # 내려가면 벨트를 누르는 영역이다.
        try:
            z_after = self._get_current_posx()[2]
            moved = (z_before - z_after) if z_before is not None else float("nan")
            self.get_logger().info(
                "[EXEC] descend 도달 z=%.1f (명령 %.0fmm, 실제 %.1fmm, "
                "벨트면 약 320)" % (z_after, float(depth), moved)
            )
            if z_after < 310.0:
                self.get_logger().warn(
                    "[EXEC] 하강 z=%.1f -- 벨트면(약 320)보다 10mm 이상 아래다. "
                    "파지 실패 시 벨트를 누른다." % z_after
                )
        except Exception as exc:
            self.get_logger().warn(f"[EXEC] 하강 후 z 읽기 실패: {exc}")

        self.get_logger().info("[EXEC] close gripper")
        self._call_gripper("c")
        time.sleep(profile["close_wait_sec"])

        self.get_logger().info(f"[EXEC] rise {depth}mm")
        self._call_move_line([0.0, 0.0, depth, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)

        # [8일차, v118 1단계] 파지 성공/실패 자동 판정 -- 로그만 남기고
        # 사이클은 그대로 진행한다. 상승 후에 재는 이유: 물체를 든 상태의
        # 그리퍼 폭이 실제 운용 상태이고, 이 시점이면 닫기 동작이 확실히
        # 끝나 있다.
        # [9일차, v118 2단계] **파지 실패면 이후 동작을 전부 생략하고
        # 곧바로 다음 파지 위치로 복귀한다**(사용자 요청 — 8일차 제안의
        # 본래 목적). 빈 손으로 통까지 갔다가 벨트를 세우고 계량까지
        # 하는 낭비를 없앤다.
        #
        # **`False`(명시적 실패)일 때만 동작한다.** `None`(판정 불가)에는
        # 절대 반응하지 않는다 -- `plastic`/`plastic_bag`은 임계값이
        # 없고, 오늘 실측에서 찌그러진 뚜껑을 성공적으로 물었는데도
        # 파지값이 빈 손(+0.7496)과 같게 나온 사례가 있다. 그런 품목에
        # 억지로 반응하면 성공한 파지를 버린다.
        #
        # **[주의] 오탐 위험**: battery는 파지값 +0.6232, 임계값 +0.6864로
        # 여유가 0.0493rad뿐이다(실측 1회, 눕힌 자세만). 정상 파지가
        # 실패로 읽히면 그 물체를 놓고 지나간다. 오탐이 관찰되면 우선
        # battery의 `grasp_fail_joint_rad`를 null로 바꿔 판정에서 빼는
        # 것으로 대응할 것 -- 2단계 전체를 끄기보다 품목 단위로 끄는
        # 편이 손실이 적다.
        # [v155] 손목캠 기록 -- `_check_grasp`보다 먼저 부른다. 상승
        # 직후에 가까울수록 물체 자세가 파지 직후 상태에 가깝고,
        # 이 호출이 executor를 돌려주므로 뒤이은 관절값 읽기에도
        # 최신 콜백이 반영된다.
        self._capture_grasp_view(item_key)

        grasp_ok = self._check_grasp(item_key)
        if grasp_ok is False:
            self.get_logger().warn(
                f"[EXEC] {item_key} 파지 실패 -- 배치/계량 생략하고 다음 대상으로 복귀"
            )
            self._publish_ui_alert("warn", f"{item_key} 파지 실패 — 재시도 대기")
            self._abort_pick_and_return(item_key)
            return

        self._holding = True

        # [v48 회신, v23 원칙 개정] 파지 성공 -- 배치+복귀 완료까지
        # 벨트 정지(다음 물체가 처리 중 도달범위를 벗어나는 문제
        # 방지). 검출~추적~예측~파지 구간은 벨트가 돌았음(칼만
        # 예측이 실제로 필요한 구간, 그대로 유지).
        # [9일차] 인터록도 사유 집합의 원소 하나로 넣는다 -- 1단계에서
        # 구조가 자연스럽게 검증된다. stop_during_pick=false면 파지 중에도
        # 벨트를 세우지 않는다(2단계 무정지 시험).
        if self._stop_during_pick:
            self.get_logger().info("[EXEC] pick complete, stopping conveyor for placement")
            self._hold("interlock")
        self._stage = "measure"

        # [완료 — 8일차, v103] **`measure_pose` 복원 — 무게는 항상
        # 같은 자세에서 잰다.**
        #
        # 8일차 실측: `baseline`은 이전 사이클 복귀 위치(x=300)에서,
        # `post-pick`은 파지점(x=155~191)에서 재고 있었다 -- **X가 매
        # 사이클 109~144mm 달라** 팔 뻗은 정도가 바뀌고 중력보상
        # 계통오차가 그대로 `net`에 섞였다(진짜 ~16g인 빈 캔이 net
        # +168.6g으로 찍혀 review_bin 오배치). 반례로 #7 사이클은
        # 우연히 Δx=+9.0mm였는데 net +235.1g으로 내용물을 정확히
        # 잡아냈다 -- **센서는 멀쩡하고 측정 자세가 문제**라는 방증.
        #
        # **[확인 — 설계문서 v32:2982 원문 대조]** 새 개념이 아니라
        # 원래 설계(v10, `measure_pose`/`HOLD_AND_MEASURE`)의 복원이다:
        # "관절 토크 역산 무게 측정이 로봇 자세에 의존(1일차 발견) --
        # 파지 후 고정 `measure_pose`로 이동해 항상 동일 자세에서
        # 측정". 1일차에 원인도 해법도 알고 있었는데 구현에서 이
        # 단계가 누락됐던 것.
        #
        # 좌표는 `default_return_x`(300.0) + 벨트 상공 자세 -- 매
        # 사이클 복귀에 이미 쓰던 위치라 도달성 리스크가 낮다.
        # **파지 성공률에는 영향이 없다**(파지 완료 이후 동작이라
        # 예측 구간 `total_ahead`와 무관) -- baseline을 앞으로 옮기는
        # 안과의 결정적 차이이고, 6일차 extra_gap 버그가 재발하지
        # 않는 이유다.
        if weight_check is not None:
            # [9일차, v139] **계량 중에는 벨트를 잠깐 세운다.**
            # 무정지 전환(stop_during_pick=false) 시험에서 벨트 진동이
            # 계량에 그대로 실려 빈 캔이 net 106.2g으로 측정됐다
            # (실제 약 16g, 기준 80g -> review_bin 오배치).
            # post-pick 샘플 폭 61.2g vs baseline 폭 7.4g.
            # 정지 상태 계량은 이미 검증된 조건(net 오차 1.6g)이므로
            # 그 조건을 보존한다. 파지 사이클 17~24초 중 1~2초라
            # 무정지의 시각적 이득은 거의 그대로 유지된다.
            # **baseline 쪽은 절대 세우지 않는다** -- 예측 구간
            # (extra_gap/total_ahead) 안이라 파지 타이밍이 틀어진다.
            self._hold("weighing")
            measure_pose = [
                mt.get("default_return_x", 300.0),
                mt["belt_hover_y"], mt["belt_hover_z"],
                mt["belt_hover_rx"], mt["belt_hover_ry"], mt["belt_hover_rz"],
            ]
            self.get_logger().info(
                f"[EXEC] move to measure_pose {[round(v, 1) for v in measure_pose[:3]]} "
                "(무게는 항상 같은 자세에서 측정 -- v103)"
            )
            self._call_move_line(measure_pose, h_vel, h_vel, mode=0)
            pos = self._get_current_posx()
            if abs(pos[0] - measure_pose[0]) > 5.0 or abs(pos[1] - measure_pose[1]) > 5.0:
                # 도달 실패해도 측정은 진행한다 -- 자세가 다르면 net이
                # 부정확할 뿐이지만, 여기서 중단하면 이미 집은 물체를
                # 든 채 멈춰버린다. 대신 로그로 남겨 그 회차를 사후에
                # 걸러낼 수 있게 한다.
                self.get_logger().warn(
                    f"[EXEC] measure_pose 도달 실패 (got {[round(v, 1) for v in pos[:3]]}, "
                    f"wanted {[round(v, 1) for v in measure_pose[:3]]}) -- "
                    "측정은 진행하되 net 값 신뢰도 낮음"
                )

        # [완료 — 6일차] 무게 측정으로 내용물 여부 판정(원칙7,
        # v31 팀결정: can으로 무게측정). Doosan 표준 API 사용, 파지
        # 전(baseline_weight)과의 차이를 실제 물체 무게로 쓴다.
        # **[버그 발견 -> 진단 정정 -> 수정 — 6일차, 속도 상향 후]**
        # 수직 속도를 100->130(vel)로 올린 뒤 빈 캔인데도 review_bin
        # 오배치가 반복됨(net 153.8g/125.3g, threshold 30g의 4~5배).
        # 처음엔 "정지 직후 관성/진동 settle 부족"으로 진단해 0.3->
        # 0.6초로 늘렸지만 오히려 더 나빠지는 등 settle 시간과 무관하게
        # 값이 들쭉날쭉했다. **재진단**: 벨트/로봇을 완전히 정지시킨
        # 채 빈 그리퍼로 단일 측정을 8회 반복해도 45g 폭(1.310~1.355kg)
        # 으로 흔들렸고, 벨트 가동 중(21g 폭)과 비교해도 벨트 상태가
        # 주된 원인이 아니었다 -- **힘센서 자체의 고유 노이즈가
        # threshold와 비슷한 규모**라는 게 진짜 원인. **수정**: settle
        # sleep 대신 `_get_workpiece_weight_avg()`(5회 평균)로 교체
        # (baseline/post-pick 둘 다) -- 노이즈를 시간이 아니라 반복
        # measurement 평균으로 줄인다.
        forced_bin = None
        review_reason = None
        # [9일차] 아래 파지폭 교차검증을 사람 인계 경로에만 걸기 위한 조회.
        _route = self._item_routing["routing"].get(item_key) or {}
        route_type_is_handoff = _route.get("type") == "human_handoff"
        if weight_check is not None:
            weight_kg = self._weigh_with_log(f"{item_key}-postpick")
            # [9일차] n_log=15 제거. 15샘플이 판정에 쓰이는 건 앞 5개뿐인데,
            # 나머지 10개가 계량 구간을 약 8.8초 늘리고 있었다(측정 1회당
            # 서비스 호출 지연 약 0.7초 + interval 0.15초 = 약 0.85초).
            # 무정지 전환으로 이 구간만큼 벨트를 세우게 되면서 그 비용이
            # 그대로 드러났다. 진동 패턴 분석용 표본은 8일차에 충분히
            # 쌓였다.
            threshold = weight_check["threshold_kg"]
            net_weight = None
            if weight_kg is not None and baseline_weight is not None:
                net_weight = weight_kg - baseline_weight
            self.get_logger().info(
                f"[EXEC] weight check for {item_key}: raw={weight_kg}kg "
                f"baseline={baseline_weight}kg net={net_weight}kg "
                f"(threshold={threshold}kg)"
            )
            # [9일차] **계측 신뢰불가 판정.** net이 물리적으로 불가능한
            # 음수면(물체를 들었는데 가벼워짐) baseline이 벨트 진동으로
            # 튄 것이다 -- 실측 -137.6g. 이때 값을 그대로 믿으면 반대
            # 부호일 때 내용물 든 캔이 가볍게 측정돼 사람에게 갈 수 있다.
            # 무게초과와 같은 처리(review_bin 강제)를 하되 **로그는
            # 구분**한다. bin행 품목도 같이 걸려 정상 캔이 가끔
            # review_bin으로 갈 수 있는데, 성가심 수준이고 안전 문제가
            # 아니므로 감수한다(v141 지시).
            # [9일차, v142 선택항목] **파지폭 교차검증.**
            # 그리퍼가 물체를 확실히 물고 있는데(파지 판정 True) net이
            # 10g 미만이면 계측이 틀린 것이다 -- 물체가 있는데 무게가
            # 안 잡혔다는 뜻이므로 baseline이 튀었을 가능성이 높다.
            # 파지실패감지가 이미 읽는 신호를 재사용할 뿐 새 폴링은 없다.
            # 실제로 적용되는 품목은 can뿐이다(weight_check가 정의된
            # 품목 중 pet_labeled는 판정이 null이라 grasp_ok가 None).
            # [정정 -- 9일차] **`human_handoff` 경로 품목에만 적용한다.**
            # 처음엔 모든 weight_check 품목에 걸었는데, `can`에서 오작동
            # 했다 -- 빈 캔이 실제 16g이라 드리프트가 조금만 껴도 10g
            # 아래로 떨어져 매번 "신뢰불가"로 review_bin에 갔다(완주에서
            # 빈 캔 2개가 net +1.2g / -79.4g로 둘 다 오배치).
            #
            # 이 체크의 원래 목적은 "무거운 물체가 끌려서 가볍게 찍히는
            # 것을 잡아 사람에게 가는 걸 막는 것"이다. `can`은 bin행이라
            # 사람에게 가지 않으므로 여기선 안전에 기여하지 않고 오작동만
            # 한다. 가벼운 것이 정상인 품목에는 걸면 안 된다.
            # [정정 -- 9일차 야간, v153] **음수 하한을 -30g -> -150g로 넓힌다.**
            # -30g은 "물체를 들었는데 가벼워짐 = 물리적으로 불가능"이라는
            # 전제로 잡은 값인데, 그 전제가 틀렸다. 로드셀 자체가 느리게
            # 진동하기 때문에 **빈 물체도 정상적으로 음수 net이 나온다.**
            #
            # 이날 밤 실측(로봇 정지, 그리퍼 빈 상태, measure_pose 고정,
            # 30초 연속 샘플링):
            #   - 주기 약 27초, 진폭 약 +-45g의 완만한 진동
            #     (1285g -> 1376g -> 1285g -> 다시 상승)
            #   - 표준편차 28.7g, 30초 내 폭 91.6g
            #   - 벨트 가동 여부는 무관(중앙값 이동 -9.5g, 표준편차 동일)
            #   - 창을 8초까지 늘려도 중앙값 폭 69g -- **평균/중앙값으로
            #     제거할 수 없다.** 드리프트 주기가 측정 구간보다 길다.
            #   - get_workpiece_weight 서비스 1회 호출에 약 770ms 걸린다.
            #     `interval` 인자는 사실상 의미가 없고 실제 샘플 간격은
            #     0.8초다(baseline n=3은 0.24초가 아니라 약 2.5초).
            #
            # baseline과 postpick 사이는 실측 10.4~15.3초로 드리프트
            # 반주기에 해당한다 -- 최악 조건이다. 그래서 빈 캔(16g)의
            # net이 +1.2g / -79.4g처럼 흩어진다. -30g 하한은 이 정상
            # 범위를 "고장"으로 오판해 **빈 캔을 review_bin으로 보냈고,
            # 그 결과 빈 캔과 샌드 캔이 둘 다 review_bin에 들어가
            # 구분이 안 됐다**(완주시험에서 사용자가 지적한 증상).
            #
            # -150g은 item_routing.yaml에 문서화된 드리프트 포락선
            # (사이클간 최대 125g) 바로 바깥이다. 무거운 물체(400g)는
            # 드리프트 최악에서도 275g으로 읽히므로 음수로 내려갈 수
            # 없다 -- 즉 -150g 이하는 여전히 진짜 계측 고장을 뜻한다.
            # 품목별로 재정의하려면 weight_check에 net_min_kg를 준다.
            net_min = float(weight_check.get("net_min_kg", -0.15))

            if (
                route_type_is_handoff
                and grasp_ok is True
                and net_weight is not None
                and net_weight < 0.010
            ):
                forced_bin = weight_check["review_bin"]
                review_reason = "weight_measurement_unreliable"
                self.get_logger().warn(
                    f"[EXEC] {item_key} 무게판정 신뢰불가(파지는 성공인데 "
                    f"net={net_weight*1000:.1f}g < 10g) -- {forced_bin} 강제"
                )
                self._publish_ui_alert(
                    "warn",
                    f"{item_key} 파지·무게 불일치({net_weight*1000:.0f}g) — {forced_bin}으로 보냄",
                )
            elif net_weight is not None and net_weight < net_min:
                forced_bin = weight_check["review_bin"]
                review_reason = "weight_measurement_unreliable"
                self.get_logger().warn(
                    f"[EXEC] {item_key} 무게판정 신뢰불가(net={net_weight*1000:.1f}g "
                    f"< {net_min*1000:.0f}g, 계측 고장 의심) -- {forced_bin} 강제"
                )
                self._publish_ui_alert(
                    "warn",
                    f"{item_key} 무게 측정 신뢰불가({net_weight*1000:.0f}g) — {forced_bin}으로 보냄",
                )
            elif net_weight is not None and net_weight > threshold:
                forced_bin = weight_check["review_bin"]
                review_reason = "weight_exceeded"
                self.get_logger().warn(
                    f"[EXEC] {item_key} over weight threshold -- routing to "
                    f"{forced_bin} instead (내용물 있는 것으로 의심)"
                )
                self._publish_ui_alert(
                    "warn",
                    f"{item_key} 무게 초과({net_weight:.3f}kg) — {forced_bin}으로 보냄",
                )

        if weight_check is not None:
            self._release("weighing")

        # [8일차] `/ui/state`의 `last` 필드용 -- 실제 배치 통(`dest`)은
        # 배치가 끝난 뒤에 채운다(여기서는 아직 미확정).
        self._last_result = {
            "item": item_key,
            "dest": None,
            "confidence": track.get("last_confidence"),
            "net_weight_kg": net_weight if weight_check is not None else None,
            "weight_g": (
                round(net_weight * 1000.0, 1)
                if weight_check is not None and net_weight is not None
                else None
            ),
            "reason": review_reason,
            "ts": time.time(),
        }

        self.get_logger().info(f"[EXEC] pick complete for {item_key}, placing...")
        self._place_item(item_key, forced_bin=forced_bin)

    def _abort_pick_and_return(self, item_key: str) -> None:
        """[9일차, v118 2단계] 파지 실패 시 사이클을 즉시 접고 복귀한다.

        통 이동·계량·벨트 정지를 전부 생략한다 -- 빈 손으로 그 과정을
        도는 것은 순수한 낭비다(사용자 지적). 복귀 위치는 배치 경로와
        같은 규칙을 쓴다: 남은 트랙 중 가장 앞선 물체 쪽, 없으면 기본값.
        """
        mt = self._motion_timing
        h_vel = [mt["horizontal_vel"], mt["horizontal_acc"]]

        # [정정 -- 9일차] **여기서 그리퍼를 열지 않는다.**
        # 처음엔 "허공을 문 채로 두지 않는다"는 이유로 열었는데, 실제로
        # 돌려보니 정반대 결과가 났다: 종이박스를 실제로 물고 있었는데
        # (joint +0.6038/+0.6057, 빈 손은 +0.7496) 오분류로 페트병
        # 임계값(0.5089)을 적용받아 실패로 판정됐고, 이 open이 물체를
        # 그 자리에 떨어뜨렸다. **판정이 틀렸을 때 물체를 놓는 것이
        # 그냥 들고 복귀하는 것보다 훨씬 나쁘다.**
        # 어차피 다음 파지 직전에 "open gripper (approach 전 확인)"이
        # 있으므로 여기서 열 필요가 없다.

        next_x = max((t["kf"].state.x for t in self._tracks), default=None)
        if next_x is None:
            next_x = mt.get("default_return_x", 300.0)
        self.get_logger().info(
            f"[EXEC] 파지 실패 복귀 -- belt hover (x={next_x:.1f})"
        )
        belt_hover = [
            next_x, mt["belt_hover_y"], mt["belt_hover_z"],
            mt["belt_hover_rx"], mt["belt_hover_ry"], mt["belt_hover_rz"],
        ]
        self._call_move_line(belt_hover, h_vel, h_vel, mode=0)

        # 이 시점엔 아직 interlock/weighing을 걸기 전이지만, 경로가
        # 바뀌어도 사유가 남지 않도록 방어적으로 푼다(사유 집합은
        # 없는 원소를 지워도 안전하다).
        self._release("interlock")
        self._release("weighing")

        self._holding = False
        self._stage = "idle"
        self._grasp_fail_counts[item_key] = (
            self._grasp_fail_counts.get(item_key, 0) + 1
        )
        self.get_logger().info(
            f"[EXEC] 파지 실패 누적 {item_key}={self._grasp_fail_counts[item_key]}회"
        )

    def _place_item(self, item_key: str, forced_bin: str | None = None) -> None:
        """상태머신 골격(6일차) — 품목별 통 배치 또는 사람전달 placeholder.

        `config/item_routing.yaml`을 조회해서 동작한다(원칙3, 품목명
        분기를 코드에 두지 않음 — `forced_bin`은 품목명이 아니라
        무게 측정값에 대한 분기이므로 원칙3 위반이 아님, 원칙7과
        일치: "무게 측정으로 내용물 여부 판정"). "수직 상승 -> 수평
        이동 -> 수직 하강" 3단계 시퀀스를 지킨다(CLAUDE.md 안전 절)
        — `_execute_pick`이 이미 벨트 상공(z=430)까지 상승을 끝낸
        상태에서 호출되므로, 여기서는 그 높이를 유지한 채 수평
        이동만 하고 그다음에 하강한다.
        """
        mt = self._motion_timing
        h_vel = [mt["horizontal_vel"], mt["horizontal_acc"]]
        v_vel = [mt["vertical_vel"], mt["vertical_acc"]]
        route = self._item_routing["routing"].get(item_key)
        if route is None:
            self.get_logger().error(f"no routing entry for {item_key!r} -- 들고 대기")
            self._publish_ui_alert("error", f"{item_key} 라우팅 설정 없음 — 들고 대기")
            return

        # [9일차] `forced_bin`(무게 초과 판정)이 `human_handoff`보다 우선한다.
        # 이 순서가 뒤집혀 있으면 -- 원래 그랬다 -- `weight_check`를
        # `pet_labeled`에 켜도 무게 초과 판정이 그냥 버려지고 사람에게
        # 내밀어진다. `human_handoff` 분기가 `forced_bin`을 보기 전에
        # return해버리기 때문. `can`은 routing이 bin이라 이 결함이
        # 드러나지 않았다(잠복 상태였음).
        #
        # 원칙7("비전으로 못 보는 걸 힘으로 본다")이 실제로 성립하려면
        # 힘 판정이 비전 라우팅을 덮어쓸 수 있어야 한다. 8일차에 드러난
        # "내용물 든 캔이 pet_labeled로 오분류되면 무게 안전장치를
        # 우회한다"는 결함의 본체가 이것이다.
        if forced_bin is not None:
            self.get_logger().warn(
                f"[PLACE] {item_key}: 무게 판정으로 {forced_bin} 우선 "
                f"(routing={route['type']} 무시)"
            )
        elif route["type"] == "human_handoff":
            self._stage = "handoff"
            self._handoff_item(item_key)
            return

        bin_name = forced_bin if forced_bin is not None else route["bin"]
        hover_pose = get_bin_pose(bin_name, "hover")
        place_pose = get_bin_pose(bin_name, "place")
        hover_pose, place_pose = self._apply_place_pose_override(
            item_key, bin_name, hover_pose, place_pose
        )

        self._stage = "place"
        self.get_logger().info(f"[PLACE] {item_key} -> {bin_name}: move to bin hover")
        self._call_move_line(hover_pose, h_vel, h_vel, mode=0)
        pos = self._get_current_posx()
        if abs(pos[0] - hover_pose[0]) > 5.0 or abs(pos[1] - hover_pose[1]) > 5.0:
            self.get_logger().error(
                f"[PLACE] bin hover move did not reach target (got {pos[:3]}, "
                f"wanted {hover_pose[:3]}) -- 들고 대기, 배치 중단, 벨트도 정지 유지"
                "(사람이 확인 후 재시작할 것)"
            )
            self._publish_ui_alert(
                "error", f"{bin_name} 상공 이동 실패 — 배치 중단, 사람 확인 필요"
            )
            return

        self.get_logger().info(f"[PLACE] descend to {bin_name} place pose")
        self._call_move_line(place_pose, v_vel, v_vel, mode=0)

        self.get_logger().info("[PLACE] open gripper")
        self._call_gripper("o")
        time.sleep(self._item_routing.get("place_open_wait_sec", 1.0))

        # [v156] 배치 확인(a) 개방폭 + (b) 배치 스냅샷 -- 둘 다 기록만.
        # 그리퍼가 실제로 열렸는지, 열렸는데도 물체가 남아 있지는
        # 않은지를 사후에 확인할 자료를 남긴다.
        self._check_place_open(item_key, bin_name)
        self._capture_grasp_view(item_key, label="place")

        self.get_logger().info(f"[PLACE] rise back to {bin_name} hover")
        self._call_move_line(hover_pose, v_vel, v_vel, mode=0)

        # [완료 — 6일차, 순차 처리 인덱싱] 단일 트랙(self._kf) 대신
        # 남아있는 트랙 중 가장 앞선(x가 가장 큰, 다음에 처리될) 물체
        # 방향으로 복귀한다 -- 다음 Pick 대상에 더 가까운 곳에서
        # 대기하는 효과. 남은 트랙이 없으면 벨트 시작쪽 기본값으로.
        next_x = max((t["kf"].state.x for t in self._tracks), default=None)
        if next_x is None:
            next_x = mt.get("default_return_x", 300.0)
        self.get_logger().info(
            f"[PLACE] return to belt hover (x={next_x:.1f}, ready for next pick)"
        )
        belt_hover = [
            next_x, mt["belt_hover_y"], mt["belt_hover_z"],
            mt["belt_hover_rx"], mt["belt_hover_ry"], mt["belt_hover_rz"],
        ]
        self._call_move_line(belt_hover, h_vel, h_vel, mode=0)

        self._holding = False
        self._pick_counts[item_key] = self._pick_counts.get(item_key, 0) + 1
        self._bin_counts[bin_name] = self._bin_counts.get(bin_name, 0) + 1
        if self._last_result is not None and self._last_result.get("item") == item_key:
            self._last_result["dest"] = bin_name
            if bin_name == "review_bin" and not self._last_result.get("reason"):
                self._last_result["reason"] = "manual_review_required"
        self._stage = "idle"
        self.get_logger().info(f"[PLACE] {item_key} placed in {bin_name}, restarting conveyor")
        self._release("interlock")

    def _handoff_item(self, item_key: str) -> None:
        """실제 순응제어 기반 사람 핸드오버 (6일차 구현, 8일차 안전 재작성).

        새 목표 좌표(x/y)를 만들지 않는다 -- `_execute_pick`이 이미
        도달을 검증한 벨트 상공 (predicted_x, y=-264, z=430) 컬럼
        에서 z만 더 올린다(같은 x/y, 더 높은 곳은 덜 뻗는 자세라
        도달성 측면에서 더 안전할 것으로 판단, 새 자세를 탐색하는
        위험을 피함).

        절차: 그리퍼 닫힌 채 z만 상승(제시 높이) -> **상승 도달 확인**
        -> task_compliance_ctrl(저강성)으로 순응 모드 진입 -> 기준점
        캡처 -> get_current_posx를 폴링해 **수평 이동 속도**가 연속 N회
        임계값 이상이고 AND 누적 변위도 임계값을 넘으면(사람이 당김)
        그리퍼 오픈 -> release_compliance_ctrl -> 원래 높이로 복귀 ->
        벨트 재가동. 타임아웃 시에는 그 자리에서 열지 않고 review_bin
        으로 정상 배치한다.

        [전제] 10회 연속 무사고 검증 전이므로 "안전이 확정됐다"고
        표현하지 않는다(CLAUDE.md 과대 청구 금지 원칙).
        """
        mt = self._motion_timing
        v_vel = [mt["vertical_vel"], mt["vertical_acc"]]
        cfg = self._item_routing.get("handoff", {})
        extra_rise_mm = cfg.get("extra_rise_mm", 120.0)
        stiffness = cfg.get("stiffness", [500.0, 500.0, 500.0, 100.0, 100.0, 100.0])
        release_threshold_mm = cfg.get("release_threshold_mm", 10.0)
        timeout_sec = cfg.get("timeout_sec", 15.0)
        poll_interval_sec = cfg.get("poll_interval_sec", 0.3)
        compliance_transition_sec = cfg.get("compliance_transition_sec", 0.5)

        # ------------------------------------------------------------------
        # 1) 제시 높이 상승 -- 도달 확인 필수
        # ------------------------------------------------------------------
        # [버그 발견·수정 — 8일차, 관제PC 실기] 상승 이동은 상대좌표
        # (mode=1)라 `_execute_pick`의 hover 도달 확인(절대좌표 비교)에
        # 걸리지 않는다. 실제로 `[x=561.72, z=550]`가 NOT REACHABLE인데도
        # 코드가 그대로 순응모드 진입까지 진행한 사례가 관측됐다
        # (Codex 실기 로그, 2026-09-08). 못 올라간 자세에서 저강성으로
        # 들어가면 어디서 무엇을 놓게 될지 예측할 수 없으므로, 여기서
        # 끊고 `_holding=True`인 채 사람 개입을 기다린다.
        pos_before_rise = self._get_current_posx()
        self.get_logger().info(
            f"[HANDOFF] {item_key}: rise {extra_rise_mm}mm to presentation height "
            "(같은 x/y 컬럼, 새 자세 탐색 안 함)"
        )
        self._call_move_line([0.0, 0.0, extra_rise_mm, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)
        pos_after_rise = self._get_current_posx()
        actual_rise = pos_after_rise[2] - pos_before_rise[2]
        if abs(actual_rise - extra_rise_mm) > 5.0:
            self.get_logger().error(
                f"[HANDOFF] presentation-height rise did not reach target "
                f"(wanted +{extra_rise_mm}mm, got +{actual_rise:.1f}mm) -- "
                "순응모드 진입 안 함, 들고 대기(벨트 정지 유지, 사람이 확인 후 재시작할 것)"
            )
            self._publish_ui_alert(
                "error", f"{item_key} 제시 높이 상승 실패 — 핸드오버 중단, 사람 확인 필요"
            )
            self._say("핸드오버 자세 이동에 실패했습니다. 물체를 든 채 대기합니다. 확인이 필요합니다.")
            return  # _holding=True 그대로 유지

        # ------------------------------------------------------------------
        # 1-b) [8일차, v99] 무게 측정 -- **로그만, 판정 없음**
        # ------------------------------------------------------------------
        # 8일차에 드러난 결함: `weight_check`가 `can`에만 정의돼 있어서,
        # 내용물 든 캔이 `pet_labeled`로 오분류되면 무게 측정 자체를
        # 건너뛰고 이 핸드오버 경로로 들어온다 -- **내용물 든 캔을
        # 사람에게 내밀게 된다**(실측: 같은 캔인데 내용물 유무만으로
        # 5회 중 3회 오분류). 원칙7("비전으로 못 보는 걸 힘으로 본다")이
        # 비전 판정에 의존하는 구조라 비전이 틀리면 힘 판정에 도달조차
        # 못 한다.
        #
        # 최종 방어는 "사람에게 주기 전엔 품목과 무관하게 무게를 확인"
        # 하는 것인데, **지금은 판정을 켜지 않는다.** 두 가지가 먼저다:
        #   (1) baseline/post-pick의 측정 조건 차이(자세 109~144mm,
        #       벨트 가동 여부)를 잡아야 net 값을 믿을 수 있다.
        #   (2) 정상 라벨 페트병의 무게 분포 실측치가 아직 없다.
        # 그래서 여기서는 **측정하고 로그만 남긴다** -- 정상 핸드오버가
        # 무게 때문에 막히는 회귀 없이 임계값 산정용 데이터를 모은다.
        self._weigh_with_log(f"handoff-{item_key}-presented")
        # [9일차] n_log=15 제거 -- 이 호출은 순응제어 진입 **전**이라
        # 사람이 받으러 오기까지 약 13초를 그냥 기다리게 만들고 있었다.

        # ------------------------------------------------------------------
        # 2) 순응모드 진입
        # ------------------------------------------------------------------
        # [사용자 지시 -- 8일차] 사람에게 받아가라고 **먼저 알린다.**
        # 이전엔 제시 높이로 올라간 뒤 아무 안내 없이 15초를 세다가,
        # 시간이 지나야 "받아가지 않아 확인 필요 통으로 옮깁니다"만
        # 나왔다 -- 작업자 입장에선 로봇이 왜 멈춰 서 있는지 모른 채
        # 타임아웃을 맞게 된다. 왜 건네는지(라벨 제거가 필요한 물건)와
        # 무엇을 해야 하는지(받아가기)를 순응모드 진입 직전에 말한다.
        self._say("라벨 제거가 필요한 물건입니다. 받아가 주세요.")

        self.get_logger().info(
            f"[HANDOFF] entering compliance mode stx={stiffness} "
            f"(release_threshold_mm={release_threshold_mm}, timeout_sec={timeout_sec})"
        )
        self._call_task_compliance_ctrl(stx=stiffness, ref=0, time_=compliance_transition_sec)
        time.sleep(compliance_transition_sec)

        # ------------------------------------------------------------------
        # 3) 기준점 캡처
        # ------------------------------------------------------------------
        # [버그 이력]
        # 6일차: start_pos를 순응모드 진입 *이전*에 캡처 -> 저강성
        #   진입 자체의 중력 처짐을 "당김"으로 오인(2.9초만에
        #   disp=10.7mm로 오탐 릴리즈, 물체 낙하). 수정: 진입 +
        #   고정 0.5초 settle 후 캡처.
        # 8일차(1차 재발): 관제PC 실물에서 사람 없이 약 3초 만에
        #   또 오탐 릴리즈 -- 고정 0.5초로는 처짐이 다 안 멈춘
        #   상태였을 것으로 추정. 적응형 settle로 교체했으나,
        # 8일차(재정정): **적응형 settle도 근본 해결이 아니다** --
        #   "이동량이 매 폴링 2mm 이하"는 변화율 조건일 뿐 총량
        #   조건이 아니라서, 폴링마다 조금씩 처지는 **느린 크리프**는
        #   settle 조건을 계속 통과하면서도 누적되면 결국 총 변위
        #   임계값을 넘는다. 기준점을 한 번 잡고 누적 변위만 재는
        #   구조인 한 settle을 아무리 정교하게 해도 이 문제가 남는다.
        #
        # **[완료 — 8일차] 판별 기준을 "누적 변위(총량)"에서 "수평
        # 속도"로 전환.** 사람이 당기는 동작은 빠르고(추정 20~50mm/s),
        # 중력 처짐/크리프는 느리다(6일차 사고 실측 평균 약 3.7mm/s)
        # -- 총량이 아니라 **속도**로 보면 느린 크리프가 아무리 오래
        # 누적돼도 걸리지 않는다. 그래서 적응형 settle 로직 자체가
        # 불필요해져 제거했다(진입 직후의 급격한 첫 처짐 스냅은
        # `compliance_transition_sec` 대기로 흡수).
        start_pos = self._get_current_posx()
        start_t = time.monotonic()
        self.get_logger().info(
            f"[HANDOFF] baseline captured, start_pos="
            f"{[round(v, 1) for v in start_pos[:3]]}"
        )

        pull_speed_mm_per_s = cfg.get("pull_speed_mm_per_s", 15.0)
        pull_speed_consecutive = cfg.get("pull_speed_consecutive", 2)

        # ------------------------------------------------------------------
        # 4) 릴리즈 판정 -- 누적변위 단독 X, "수평속도 AND 누적변위" 조합
        # ------------------------------------------------------------------
        # 릴리즈 조건(AND): (a) **수평** 방향(dx,dy만, dz 제외 -- 중력
        # 처짐은 -Z가 주성분이라 애초에 속도 계산에서 배제) 이동 속도가
        # 연속 `pull_speed_consecutive`회 이상 `pull_speed_mm_per_s` 이상
        # AND (b) 시작점 대비 누적 3D 변위가 `release_threshold_mm` 이상
        # (미세한 흔들림 배제용 하한선, 기존 로직 유지).
        # [전제, 미검증] `pull_speed_mm_per_s`=15.0은 실측 3.7mm/s와
        # 추정 20~50mm/s 사이의 중간값 -- 로봇 실물 재검증(느린 처짐
        # 조건/실제 당김 조건 둘 다) 전까지 확정값 아님.
        released = False
        fast_count = 0
        prev_pos = start_pos
        prev_t = start_t
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            time.sleep(poll_interval_sec)
            now_t = time.monotonic()
            cur_pos = self._get_current_posx()
            dt = now_t - prev_t  # 고정 poll_interval_sec을 가정하지 않고
            # 실제 경과 시간으로 나눈다.

            step_horizontal = (
                (cur_pos[0] - prev_pos[0]) ** 2 + (cur_pos[1] - prev_pos[1]) ** 2
            ) ** 0.5
            speed_horizontal = step_horizontal / dt if dt > 0 else 0.0

            # [완료 — 8일차, v87] 누적변위에서 **아래 방향(-Z, 처짐)만
            # 0으로 취급**한다. 기존 3D 전체 변위는 처짐만으로도
            # release_threshold_mm(10mm)를 넘어 AND 조건의 절반이 상시
            # 통과 상태였다 -- 1회차 실측이 그 증거다(처짐만으로 19.4mm,
            # 사람 손 없음). 사람이 받아갈 땐 물체를 수평으로 끌거나
            # 위로 올리지(+Z), 아래로 누르진 않는다는 물리적 근거.
            dz_total = cur_pos[2] - start_pos[2]
            dz_effective = max(dz_total, 0.0)  # 처짐(음수)은 0 취급, 들어올림(양수)은 그대로
            total_disp = (
                (cur_pos[0] - start_pos[0]) ** 2
                + (cur_pos[1] - start_pos[1]) ** 2
                + dz_effective ** 2
            ) ** 0.5
            total_disp_3d = (
                (cur_pos[0] - start_pos[0]) ** 2
                + (cur_pos[1] - start_pos[1]) ** 2
                + dz_total ** 2
            ) ** 0.5  # 로그 참고용(수정 전 값과 비교하기 위해 같이 남김)
            step_vertical = cur_pos[2] - prev_pos[2]  # 로그 참고용(부호로 처짐/들어올림 구분)

            is_fast = speed_horizontal >= pull_speed_mm_per_s
            fast_count = fast_count + 1 if is_fast else 0

            # 다음 실물 테스트에서 처짐/당김 속도 분포를 실측할 수 있도록
            # 매 폴링 상세 로그를 남긴다 -- **절대 제거하지 말 것**
            # (로그 없이 사고가 나면 원인을 추정으로만 답하게 된다).
            self.get_logger().info(
                "[HANDOFF] poll elapsed=%.2fs pos=(%.1f,%.1f,%.1f) "
                "total_disp=%.1fmm (3d=%.1fmm) step_horiz=%.1fmm step_vert=%+.1fmm "
                "dt=%.2fs speed_horiz=%.1fmm/s fast_count=%d/%d"
                % (now_t - start_t, cur_pos[0], cur_pos[1], cur_pos[2],
                   total_disp, total_disp_3d, step_horizontal, step_vertical, dt,
                   speed_horizontal, fast_count, pull_speed_consecutive)
            )

            prev_pos, prev_t = cur_pos, now_t

            if fast_count >= pull_speed_consecutive and total_disp > release_threshold_mm:
                self.get_logger().info(
                    f"[HANDOFF] pull detected: speed_horiz={speed_horizontal:.1f}mm/s "
                    f">= {pull_speed_mm_per_s}mm/s ({fast_count}회 연속) AND "
                    f"total_disp={total_disp:.1f}mm > {release_threshold_mm}mm -- releasing"
                )
                released = True
                break

        if released:
            self._say("전달 완료했습니다.")
            self._call_gripper("o")
            time.sleep(self._item_routing.get("place_open_wait_sec", 1.0))

            self.get_logger().info("[HANDOFF] releasing compliance mode")
            self._call_release_compliance_ctrl()
            time.sleep(compliance_transition_sec)

            self.get_logger().info("[HANDOFF] returning to belt hover height")
            self._call_move_line([0.0, 0.0, -extra_rise_mm, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)

            self._holding = False
            self._pick_counts[item_key] = self._pick_counts.get(item_key, 0) + 1
            self._bin_counts["human_handoff"] = self._bin_counts.get("human_handoff", 0) + 1
            if self._last_result is not None and self._last_result.get("item") == item_key:
                self._last_result["dest"] = "human_handoff"
            self._stage = "idle"
            self.get_logger().info(f"[HANDOFF] {item_key} handed off, restarting conveyor")
            self._release("interlock")
            return

        # ------------------------------------------------------------------
        # 5) 타임아웃 -- 그 자리에서 열지 말고 review_bin으로 정상 배치
        # ------------------------------------------------------------------
        # [완료 — 8일차, 팀결정] **이전 동작**: 제시 높이에서 바로
        # 그리퍼를 열어 물체가 그 자리에서 낙하했다. **지금 동작**:
        # 정상 배치와 같은 "수직 상승 -> 수평 이동 -> 수직 하강" 3단계
        # 시퀀스(CLAUDE.md 안전 절)를 그대로 따라 review_bin("확인 필요"
        # 통)에 배치한다 -- 사람이 못 받아간 물체를 허공에 떨어뜨리지
        # 않는다.
        self.get_logger().warn(
            f"[HANDOFF] {item_key}: no pull detected within {timeout_sec}s -- "
            "routing to review_bin instead of releasing in place"
        )
        self._publish_ui_alert(
            "warn", f"{item_key} 인계 시간 초과 — review_bin으로 재배치"
        )
        self._say("받아가지 않아 확인 필요 통으로 옮깁니다.")
        self.get_logger().info("[HANDOFF] releasing compliance mode before re-routing")
        self._call_release_compliance_ctrl()
        time.sleep(compliance_transition_sec)

        h_vel = [mt["horizontal_vel"], mt["horizontal_acc"]]
        bin_name = "review_bin"

        # [완료 — 8일차, v87] review_bin 배치 자세를 `place_pose_override`
        # (config/item_routing.yaml)로 조회해 덮어쓴다. 티치 자세
        # (rz=157.11)와 파지 자세(rz=89.78)가 67.33° 차이나는데, 그
        # 각도로 돌리면 라벨 페트병 길이가 통 가로폭을 넘어 안 들어간다
        # -- 1회차 실물에서 통에 부딪혀 **그리퍼가 SAFE 모드로 진입**했다
        # (소프트웨어는 성공으로 기록 -- x/y 도달 확인만 하므로
        # 그리퍼-통벽 간섭은 이 체크에 안 걸린다).
        #
        # [전제, 미검증] 이 (좌표, 자세) 조합은 한 번도 실행된 적이
        # 없다 -- 실물 투입 전 반드시 빈 그리퍼로 먼저 이동시켜
        # 도달성/그리퍼-통벽 간섭을 확인할 것(4일차 "상승 없이
        # 이동하면 그리퍼가 통을 밀어버린" 사고 전례 있음,
        # obstacles.yaml의 review_bin은 미실측 상태).
        hover_pose = get_bin_pose(bin_name, "hover")
        place_pose = get_bin_pose(bin_name, "place")
        hover_pose, place_pose = self._apply_place_pose_override(
            item_key, bin_name, hover_pose, place_pose
        )

        self.get_logger().info(f"[HANDOFF] {item_key} -> {bin_name} (timeout): move to bin hover")
        self._call_move_line(hover_pose, h_vel, h_vel, mode=0)
        pos = self._get_current_posx()
        if abs(pos[0] - hover_pose[0]) > 5.0 or abs(pos[1] - hover_pose[1]) > 5.0:
            self.get_logger().error(
                f"[HANDOFF] {bin_name} hover move did not reach target (got {pos[:3]}, "
                f"wanted {hover_pose[:3]}) -- 들고 대기, 배치 중단, 벨트도 정지 유지"
                "(사람이 확인 후 재시작할 것)"
            )
            self._publish_ui_alert(
                "error", f"{bin_name} 상공 이동 실패 — 배치 중단, 사람 확인 필요"
            )
            return

        self.get_logger().info(f"[HANDOFF] descend to {bin_name} place pose")
        self._call_move_line(place_pose, v_vel, v_vel, mode=0)

        self.get_logger().info("[HANDOFF] open gripper")
        self._call_gripper("o")
        time.sleep(self._item_routing.get("place_open_wait_sec", 1.0))

        self.get_logger().info(f"[HANDOFF] rise back to {bin_name} hover")
        self._call_move_line(hover_pose, v_vel, v_vel, mode=0)

        # `_place_item`과 같은 복귀 규칙 -- 남은 트랙 중 가장 앞선(x가
        # 가장 큰) 물체 방향으로 복귀, 없으면 벨트 시작쪽 기본값.
        next_x = max((t["kf"].state.x for t in self._tracks), default=None)
        if next_x is None:
            next_x = mt.get("default_return_x", 300.0)
        self.get_logger().info(
            f"[HANDOFF] return to belt hover (x={next_x:.1f}, ready for next pick)"
        )
        belt_hover = [
            next_x, mt["belt_hover_y"], mt["belt_hover_z"],
            mt["belt_hover_rx"], mt["belt_hover_ry"], mt["belt_hover_rz"],
        ]
        self._call_move_line(belt_hover, h_vel, h_vel, mode=0)

        self._holding = False
        self._pick_counts[item_key] = self._pick_counts.get(item_key, 0) + 1
        self._handoff_timeout_counts[item_key] = (
            self._handoff_timeout_counts.get(item_key, 0) + 1
        )
        self._bin_counts[bin_name] = self._bin_counts.get(bin_name, 0) + 1
        if self._last_result is not None and self._last_result.get("item") == item_key:
            self._last_result["dest"] = bin_name
            self._last_result["reason"] = "handoff_timeout"
        self._stage = "idle"
        self.get_logger().info(
            f"[HANDOFF] {item_key} placed in {bin_name} (사람이 안 받아감), restarting conveyor "
            f"-- handoff_timeout_count[{item_key}]="
            f"{self._handoff_timeout_counts[item_key]}"
        )
        self._release("interlock")


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
