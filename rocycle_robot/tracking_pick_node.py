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
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, JointState
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

        # [완료 — 6일차, 순차 처리 인덱싱] 물체 1개 기준 단일 상태
        # (_pending_class/_locked_class/self._kf 하나)를 물체별 트랙
        # 리스트로 교체 -- 여러 물체를 동시에 추적하고, Pick 우선순위는
        # 신뢰도가 아니라 벨트 진행방향(가장 앞선 물체)으로 정한다.
        # 트랙 하나 = {class_name, kf, pending_count, last_y, last_z,
        # last_detection_time, lost_count}. 상세: _on_image/_match_track.
        self._tracks: list[dict] = []
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

        self.create_timer(0.5, self._publish_ui_state)

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
        return {
            "class_name": class_name,
            "kf": ConstantVelocityKalman1D(
                process_var=self._kalman_process_var,
                measurement_var=self._kalman_measurement_var,
            ),
            "pending_count": 0,
            "last_y": None,
            "last_z": None,
            "last_detection_time": None,
            "lost_count": 0,
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
                self._call_conveyor("on")
                self._say("분리수거를 시작합니다.")
            else:
                self.get_logger().info(f"[VOICE] start 무시(state={self._voice_state})")
        elif cmd == "pause":
            if self._voice_state == "RUNNING":
                self._voice_state = "PAUSED"
                self._call_conveyor("off")
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
                self._call_conveyor("on")
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
            self._call_conveyor("off")
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

            elif (
                "cx" in det
                and "cy" in det
            ):

                # bbox가 없는 경우 fallback
                u = float(det["cx"])
                v = float(det["cy"])

            else:
                continue

            # 벨트 영역 밖 detection 제외
            if v < self._belt_pixel_y_min:
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
        predicted_x = track["kf"].predict_position_at(ahead)
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
        req = SetCommand.Request()
        req.command = command
        future = self._gripper_client.call_async(req)
        self._robot_executor.spin_until_future_complete(future, timeout_sec=8.0)
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
                        drop_first: int = 1, n_log: int | None = None):
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
            self._weigh_with_log("can-baseline", n=3, interval=0.08)
            if weight_check is not None
            else None
        )

        now = self.get_clock().now().nanoseconds / 1e9
        extra_gap = now - track["last_detection_time"]
        total_ahead = extra_gap + predict_ahead_sec(
            item_key, self._motion_timing, self._gripper_profiles
        )
        predicted_x = track["kf"].predict_position_at(total_ahead)
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
        self._call_move_line([0.0, 0.0, -depth, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)

        self.get_logger().info("[EXEC] close gripper")
        self._call_gripper("c")
        time.sleep(profile["close_wait_sec"])

        self.get_logger().info(f"[EXEC] rise {depth}mm")
        self._call_move_line([0.0, 0.0, depth, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)

        # [8일차, v118 1단계] 파지 성공/실패 자동 판정 -- 로그만 남기고
        # 사이클은 그대로 진행한다. 상승 후에 재는 이유: 물체를 든 상태의
        # 그리퍼 폭이 실제 운용 상태이고, 이 시점이면 닫기 동작이 확실히
        # 끝나 있다.
        self._check_grasp(item_key)
        self._holding = True

        # [v48 회신, v23 원칙 개정] 파지 성공 -- 배치+복귀 완료까지
        # 벨트 정지(다음 물체가 처리 중 도달범위를 벗어나는 문제
        # 방지). 검출~추적~예측~파지 구간은 벨트가 돌았음(칼만
        # 예측이 실제로 필요한 구간, 그대로 유지).
        self.get_logger().info("[EXEC] pick complete, stopping conveyor for placement")
        self._call_conveyor("off")
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
        if weight_check is not None:
            weight_kg = self._weigh_with_log("can-postpick", n_log=15)
            threshold = weight_check["threshold_kg"]
            net_weight = None
            if weight_kg is not None and baseline_weight is not None:
                net_weight = weight_kg - baseline_weight
            self.get_logger().info(
                f"[EXEC] weight check for {item_key}: raw={weight_kg}kg "
                f"baseline={baseline_weight}kg net={net_weight}kg "
                f"(threshold={threshold}kg)"
            )
            if net_weight is not None and net_weight > threshold:
                forced_bin = weight_check["review_bin"]
                self.get_logger().warn(
                    f"[EXEC] {item_key} over weight threshold -- routing to "
                    f"{forced_bin} instead (내용물 있는 것으로 의심)"
                )
                self._publish_ui_alert(
                    "warn",
                    f"{item_key} 무게 초과({net_weight:.3f}kg) — {forced_bin}으로 보냄",
                )

        # [8일차] `/ui/state`의 `last` 필드용 -- 실제 배치 통(`dest`)은
        # 배치가 끝난 뒤에 채운다(여기서는 아직 미확정).
        self._last_result = {
            "item": item_key,
            "dest": None,
            "net_weight": net_weight if weight_check is not None else None,
            "ts": time.time(),
        }

        self.get_logger().info(f"[EXEC] pick complete for {item_key}, placing...")
        self._place_item(item_key, forced_bin=forced_bin)

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

        if route["type"] == "human_handoff":
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
        self._stage = "idle"
        self.get_logger().info(f"[PLACE] {item_key} placed in {bin_name}, restarting conveyor")
        self._call_conveyor("on")

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
        self._weigh_with_log(f"handoff-{item_key}-presented", n_log=15)

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
            self._call_conveyor("on")
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
        self._call_conveyor("on")


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
