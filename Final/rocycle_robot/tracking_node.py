"""[중요 — 8일차, 팀결정] **이 파일은 관제PC 실제 운영에 쓰이지
않는다.** 관제PC는 `tracking_pick_node.py`(Docker YOLO 검출 소비
아키텍처, `/recycle_detection/detections` 구독)를 돌린다 — 여기
`tracking_node.py`를 고쳐도 **시연에 전혀 반영되지 않는다.**

이 파일은 (a) GPU/Docker 없는 개발PC에서의 dry-run 테스트용,
(b) 안전 로직(핸드오버 등)을 먼저 설계·검증한 뒤
`tracking_pick_node.py`로 이식하는 원본으로만 쓴다. 안전/동작
관련 수정은 반드시 두 파일 모두에(또는 관제PC Claude에게 이식
요청) 반영할 것 — 한쪽만 고치면 조용히 어긋난다.

tracking_node -- fixed-camera detection tracking + pixel->base 3D (설계문서 5절).

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
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

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
from rocycle_robot.vision.config_loader import load_vision_config, resolve_model_path
from rocycle_robot.vision.detector import RecycleDetector


class TrackingNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("tracking_node", **kwargs)

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
        self._reach_limit_px = self._compute_reach_limit_px()

        # YOLO12s 검출 연동 (v47) -- config/vision.yaml에서 conf 등 로드.
        vision_cfg = load_vision_config()
        self.declare_parameter("vision.model_path", vision_cfg["model_path"])
        self.declare_parameter("vision.conf", vision_cfg["conf"])
        self.declare_parameter("vision.iou", vision_cfg["iou"])
        self.declare_parameter("vision.imgsz", vision_cfg["imgsz"])
        self.declare_parameter(
            "vision.min_consecutive_frames", vision_cfg["min_consecutive_frames"]
        )
        self.declare_parameter(
            "vision.object_lost_frames", vision_cfg["object_lost_frames"]
        )
        self.declare_parameter("vision.dry_run", True)
        self.declare_parameter("vision.belt_pixel_y_min", vision_cfg["belt_pixel_y_min"])
        self.declare_parameter("vision.max_det", vision_cfg.get("max_det", 3))
        self.declare_parameter(
            "vision.track_match_gate_mm", vision_cfg.get("track_match_gate_mm", 250.0)
        )

        model_path = resolve_model_path(self.get_parameter("vision.model_path").value)
        self._detector = RecycleDetector(
            model_path,
            conf=self.get_parameter("vision.conf").value,
            imgsz=self.get_parameter("vision.imgsz").value,
        )
        self._min_consecutive_frames = self.get_parameter(
            "vision.min_consecutive_frames"
        ).value
        self._object_lost_frames = self.get_parameter("vision.object_lost_frames").value
        self._dry_run = self.get_parameter("vision.dry_run").value
        self._belt_pixel_y_min = self.get_parameter("vision.belt_pixel_y_min").value
        self._max_det = self.get_parameter("vision.max_det").value
        self._track_match_gate_mm = self.get_parameter("vision.track_match_gate_mm").value

        self._bridge = CvBridge()
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
        # [완료 — 8일차, v67 회신] `_pick_counts`(품목명 키, 음성
        # 상태보고 "캔 3개, 종이 2개"용)와 별개로 **실제 목적지
        # 키(통 이름 또는 "human_handoff")**로 집계하는 딕셔너리를
        # 하나 더 둔다 -- UI 막대그래프는 "어느 통에 몇 개"가 필요한데,
        # `item_routing.yaml`을 보면 pet_unlabeled/pet_labeled처럼
        # 서로 다른 품목이 같은 통(plastic_bin)에 몰리거나 아예 통이
        # 아닌 곳(human_handoff)으로 가는 경우가 있어 품목명 키로는
        # 안 맞는다.
        self._bin_counts: dict[str, int] = {}

        # [완료 — 8일차, v64 회신] 관제 UI용 상태 발행. `stage`는
        # idle/detect/track/predict/pick/measure/place/handoff 중 하나
        # (v64 원안엔 handoff가 없었으나 실제 코드엔 사람전달 경로가
        # 있어 추가함 -- 웹 클로드 확인 필요, 요청2 참고). **[중요,
        # 정직하게 밝힐 것] `_execute_pick`/`_place_item`/`_handoff_item`
        # 은 전부 `_on_image` 콜백 안에서 블로킹 호출된다(단일 스레드
        # executor) -- 그 ~17~24초 구간 동안은 이 타이머 콜백 자체가
        # 실행될 기회가 없다. 즉 "pick"/"measure"/"place"/"handoff"
        # 단계는 UI에 실시간으로는 절대 보이지 않고, 블로킹이 끝나는
        # 순간 한꺼번에 다음 상태로 점프한다 -- `ts` 필드로 클라이언트가
        # staleness를 스스로 판단해야 한다(진짜 해결은 파지 실행을
        # 별도 스레드/executor로 분리해야 하는데, 그건 오늘 범위 밖의
        # 아키텍처 변경). 상세 설명은 응답문서 참고.
        self._stage = "idle"
        self._last_result: dict | None = None
        self._last_image_time: float | None = None
        self._ui_state_pub = self.create_publisher(String, "/ui/state", 10)
        self._ui_alert_pub = self.create_publisher(String, "/ui/alert", 10)
        self._ui_detections_pub = self.create_publisher(String, "/ui/detections", 10)
        self.create_timer(0.5, self._publish_ui_state)

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

        # [완료 — 6일차, v58 회신] 컨베이어 서비스 클라이언트 -- dry_run
        # 여부와 무관하게 항상 만든다(음성 명령은 dry_run 중에도 들어올
        # 수 있음). 같은 재진입 스핀 문제(위 주석 참고)가 있어 전용
        # Node+SingleThreadedExecutor를 별도로 둔다(로봇 executor와도
        # 분리 -- 로봇이 없어도 컨베이어만 단독으로 쓸 수 있어야 함).
        self._conveyor_node = rclpy.create_node("tracking_node_conveyor_client")
        self._conveyor_executor = SingleThreadedExecutor()
        self._conveyor_executor.add_node(self._conveyor_node)
        self._conveyor_set_speed_client = self._conveyor_node.create_client(
            Trigger, "/conveyor/set_speed"
        )
        self._conveyor_stop_client = self._conveyor_node.create_client(
            Trigger, "/conveyor/stop"
        )

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

        # [완료 -- 8일차, v72 회신] `create_subscription(..., 10)`은
        # depth=10/RELIABLE(기본값)이라 CPU 추론이 입력 속도를 못
        # 따라가면(실측: 30Hz 입력 vs CPU 추론 약 1fps) 큐에 최대
        # 10프레임이 도착 순서대로 쌓이고 오래된 프레임부터 처리하게
        # 된다 -- 시간이 지날수록 점점 과거 프레임을 처리하는 구조적
        # 지연(v45에서 겪은 "인식 지연"과 같은 유형)으로 이어질 수
        # 있다는 지적을 받고 확인. `qos_profile_sensor_data`
        # (BEST_EFFORT + KEEP_LAST depth=5, ROS2에서 카메라류 토픽에
        # 쓰는 표준 프로파일)로 교체 -- 큐가 차면 오래된 프레임이
        # 자동으로 버려지고 항상 최신에 가까운 프레임을 처리한다.
        self.create_subscription(
            Image, "/image_raw", self._on_image, qos_profile_sensor_data
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
        """[완료 — 6일차, v61 회신 반영] 절대경로 대신 패키지 상대
        경로도 허용 -- 기존엔 `config/tracking.yaml`에 특정 사용자의
        홈 디렉터리가 박힌 절대경로(`/home/rokey/...`)가 들어있어서
        다른 계정/경로(예: 관제 PC)로 옮기면 그대로 깨졌다.
        `resolve_model_path`(YOLO 모델 경로에 이미 쓰던 것)를 그대로
        재사용 -- 상대경로면 패키지 루트 기준으로 풀고, 절대경로면
        그대로 쓴다(기존 설정과 호환)."""
        path = self.get_parameter("camera.cam_to_base_path").value
        if not path:
            return None
        return np.load(resolve_model_path(path))

    def _load_belt_plane(self) -> Plane | None:
        a, b, c, d = self.get_parameter("belt_plane.abcd").value
        if (a, b, c, d) == (0.0, 0.0, 0.0, 0.0):
            return None
        return Plane(a=a, b=b, c=c, d=d)

    def _is_calibrated(self) -> bool:
        return self._cam_to_base is not None and self._belt_plane is not None

    def _compute_reach_limit_px(self) -> float | None:
        """[완료 — 8일차, v66 3-3절] `max_reachable_x`(로봇 좌표, 590mm)
        에 해당하는 화면 x픽셀을 기동 시 1회 계산 -- 카메라·벨트가
        고정이라 매 프레임 다시 계산할 필요 없음. `belt_correction`
        (forward 방향의 경험적 XY 보정)의 역변환은 코드베이스에
        없어 적용하지 않았다 -- **[전제, 미검증]** 그 보정값 자체가
        RMS 4.5mm/최대 8.6mm로 작아 화면 픽셀 단위 오차는 미미할
        것으로 판단했다(실측 검증 안 함). 캘리브레이션 미완료거나
        `max_reachable_x` 설정이 없으면 `None` 반환."""
        if self._cam_to_base is None or self._belt_plane is None:
            return None
        max_x = self._motion_timing.get("max_reachable_x")
        if max_x is None:
            return None
        p = self._belt_plane
        if abs(p.c) < 1e-9:
            return None
        y = self._motion_timing["belt_hover_y"]
        z = -(p.a * max_x + p.b * y + p.d) / p.c
        R = self._cam_to_base[:3, :3]
        origin = self._cam_to_base[:3, 3]
        cam_pt = R.T @ (np.array([max_x, y, z]) - origin)
        if cam_pt[2] <= 0:
            return None
        return float(self._intrinsics.fx * cam_pt[0] / cam_pt[2] + self._intrinsics.cx)

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

    def _publish_ui_alert(self, level: str, msg: str) -> None:
        """[완료 — 8일차, v64 회신] 이벤트성 알림 발행 -- 도달 불가,
        배치 실패, 무게 초과 라우팅 등 운영자가 놓치면 안 되는
        순간에만 호출한다(주기 발행 아님, `/ui/state`와 역할 분리)."""
        payload = {"ts": time.time(), "level": level, "msg": msg}
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self._ui_alert_pub.publish(out)

    def _publish_ui_state(self) -> None:
        """[완료 — 8일차, v64/v66 회신] 0.5초 주기(2Hz) 상태 스냅샷 발행.

        **[정정 — v66 2절] `passed`(원안: 미검출로 벨트를 그냥
        통과한 개수) 필드는 뺐다.** YOLO 검출 기반 파이프라인에는
        "무언가 벨트를 지나갔다"를 클래스 무관하게 감지할 방법이
        없다(안 잡히면 애초에 아무 이벤트도 안 생김 -- 이 카운터
        자체가 구조적으로 셀 수 없음). v66 2-3절의 폴백 지침
        ("세기 어려우면 필드를 빼고 알려줄 것")을 그대로 따름 --
        클래스 무관 일반 물체 감지(예: 모션/블롭 기반)를 새로 붙이면
        가능하지만 데모 임박 시점에 들일 작업은 아니라고 판단.
        `health.robot`은 dry_run 중엔 판단 불가라 `None`(JSON null)로
        둔다(실제 로봇 없이 True/False로 단정하면 과대 청구 원칙
        위반). `health.conveyor`는 "최근 성공한 호출"이 아니라
        `service_is_ready()`(연결 여부)로 대체했다 -- 상태 확인을
        위해 실제로 컨베이어를 움직이는 건 부작용이 커서 부적절
        하다고 판단.
        """
        now_ros = self.get_clock().now().nanoseconds / 1e9
        camera_ok = (
            self._last_image_time is not None
            and (now_ros - self._last_image_time) < 2.0
        )
        counts = dict(self._bin_counts)
        total = sum(counts.values())
        payload = {
            "ts": time.time(),
            "state": self._voice_state,
            "stage": self._stage,
            "counts": counts,
            "total": total,
            "last": self._last_result,
            "reach_limit_px": self._reach_limit_px,
            "health": {
                "camera": camera_ok,
                "robot": None if self._dry_run else True,
                "conveyor": self._conveyor_set_speed_client.service_is_ready(),
                "stt": self.count_publishers("/voice_command") > 0,
                "tts": self.count_subscribers("/voice/tts/say") > 0,
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._ui_state_pub.publish(msg)

    def _report_status(self) -> None:
        total = sum(self._pick_counts.values())
        if total == 0:
            self._say("아직 처리한 물체가 없습니다.")
            return
        parts = ", ".join(f"{k} {v}개" for k, v in self._pick_counts.items())
        self._say(f"지금까지 총 {total}개 처리했습니다. {parts}.")

    def _report_end_of_run(self) -> None:
        total = sum(self._pick_counts.values())
        if total == 0:
            self._say("작업을 종료합니다. 처리한 물체가 없습니다.")
            return
        parts = ", ".join(f"{k} {v}개" for k, v in self._pick_counts.items())
        self._say(f"작업을 종료합니다. 총 {total}개를 처리했습니다. {parts}.")

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

    def _on_image(self, msg: Image) -> None:
        """YOLO 다중 검출 -> 물체별 트랙 매칭/갱신 -> 우선순위 Pick.

        **[완료 — 6일차, 순차 처리 인덱싱 재설계]** 기존엔 물체 1개
        기준 단일 상태(`_pending_class`/`_locked_class`/`self._kf`
        하나)였다 -- `detect()`가 `max_det=1`로 프레임당 최고-신뢰도
        물체 하나만 반환해서, 신뢰도가 낮은 물체(예: 건전지)는 다른
        물체(예: 페트병)가 화면에 있는 동안 아예 추적 자체가 안
        됐다(CLAUDE.md "현재 단계" 10번 실측). **해결**: `detect_all()`
        로 여러 물체를 동시에 받아 물체별 트랙(`self._tracks`)으로
        관리하고, Pick 우선순위는 신뢰도가 아니라 **벨트 진행방향
        (x가 가장 큰, 도달범위에 가장 먼저 걸릴 물체)**으로 정한다.
        트랙은 파지 사이클(confirmed~완료, ~17~24초, 그동안 이
        콜백 자체가 안 불림)을 넘어서도 유지되므로, 앞선 물체가
        처리되는 동안 뒤 물체가 이미 몇 프레임 확인돼 있었다면
        블로킹이 끝난 직후 곧바로(재확인 없이) Pick을 시도할 수
        있다 -- confirmed~완료 구간을 실질적으로 단축하는 효과.

        dry-run 전용 여부는 `vision.dry_run` 파라미터로 제어.
        """
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        frame_h, frame_w = frame.shape[:2]
        dets = self._detector.detect_all(frame, max_det=self._max_det)
        now = self.get_clock().now().nanoseconds / 1e9
        self._last_image_time = now

        # [완료 — 8일차, v66 회신] 클라이언트측(브라우저) bbox 오버레이용.
        # `locked`은 원안(v64)처럼 "이번 프레임에 Pick으로 선택된 그
        # 트랙"이 아니라 **"pending_count가 확정 기준을 이미 넘긴
        # 트랙 전체"**로 단순화했다 -- 전자는 `best` 선택 시점에만
        # 잠깐 참이었다가 바로 `_execute_pick`이 블로킹을 시작해
        # 버려서(2-2절과 같은 한계) 사실상 UI에 보일 일이 없다.
        # 후자가 "확정까지 남은 프레임" 표시 목적(v66 3-1절)에
        # 더 부합한다고 판단 -- 웹 클로드 확인 필요.
        det_items: list[dict] = []

        matched_ids = set()
        for det in dets:
            on_belt = det.anchor_pixel[1] >= self._belt_pixel_y_min
            if not on_belt:
                det_items.append({
                    "cls": det.class_name, "conf": round(det.confidence, 3),
                    "bbox": det.bbox, "anchor": det.anchor_pixel,
                    "pending": 0, "required": self._min_consecutive_frames,
                    "locked": False,
                })
                continue
            try:
                base = self.pixel_to_base(*det.anchor_pixel)
            except RayPlaneParallelError:
                det_items.append({
                    "cls": det.class_name, "conf": round(det.confidence, 3),
                    "bbox": det.bbox, "anchor": det.anchor_pixel,
                    "pending": 0, "required": self._min_consecutive_frames,
                    "locked": False,
                })
                continue

            track = self._match_track(det.class_name, float(base[0]), now, matched_ids)
            if track is None:
                track = self._new_track(det.class_name)
                self._tracks.append(track)

            track["last_y"] = float(base[1])
            track["last_z"] = float(base[2])
            if track["last_detection_time"] is None:
                track["kf"].reset(float(base[0]))
            else:
                dt = now - track["last_detection_time"]
                if dt > 0:
                    track["kf"].update(float(base[0]), dt)
            track["last_detection_time"] = now
            track["pending_count"] += 1
            track["lost_count"] = 0
            matched_ids.add(id(track))

            det_items.append({
                "cls": det.class_name, "conf": round(det.confidence, 3),
                "bbox": det.bbox, "anchor": det.anchor_pixel,
                "pending": track["pending_count"], "required": self._min_consecutive_frames,
                "locked": track["pending_count"] >= self._min_consecutive_frames,
            })

            self.get_logger().info(
                "class=%s conf=%.2f anchor=%s kf_x=%.1f kf_vx=%.2f pending=%d "
                "confirmed=%s holding=%s"
                % (det.class_name, det.confidence, det.anchor_pixel,
                   track["kf"].state.x, track["kf"].state.vx, track["pending_count"],
                   track["pending_count"] >= self._min_consecutive_frames, self._holding)
            )

        detections_payload = {"ts": time.time(), "w": frame_w, "h": frame_h, "items": det_items}
        det_msg = String()
        det_msg.data = json.dumps(detections_payload, ensure_ascii=False)
        self._ui_detections_pub.publish(det_msg)

        for track in list(self._tracks):
            if id(track) not in matched_ids:
                track["lost_count"] += 1
                if track["lost_count"] >= self._object_lost_frames:
                    self.get_logger().debug(
                        "[MATCH] class=%s track dropped (lost_count=%d >= %d)"
                        % (track["class_name"], track["lost_count"], self._object_lost_frames)
                    )
                    self._tracks.remove(track)

        if self._holding:
            return

        # [완료 — 6일차, v55/v56 회신] RUNNING 상태가 아니면(IDLE/
        # PAUSED/STOPPING) 검출·추적은 계속하되(위에서 이미 처리됨)
        # Pick 트리거만 막는다.
        if self._voice_state != "RUNNING":
            self._stage = "idle"
            return

        candidates = [
            t for t in self._tracks if t["pending_count"] >= self._min_consecutive_frames
        ]
        if not candidates:
            self._stage = "track" if self._tracks else "detect"
            return

        # 인덱싱 -- 신뢰도 기준이 아니라 벨트 진행방향(가장 앞선 물체)
        # 기준으로 다음 Pick 대상을 정한다.
        best = max(candidates, key=lambda t: t["kf"].state.x)
        self._tracks.remove(best)
        self._stage = "predict"

        try:
            predicted = self.predict_pickup_point(best)
            self.get_logger().info(
                "PICK TRIGGER class=%s predicted_pickup(preview)=%s dry_run=%s "
                "remaining_tracks=%d"
                % (best["class_name"], predicted, self._dry_run, len(self._tracks))
            )
            if not self._dry_run:
                self._stage = "pick"
                self._execute_pick(best["class_name"], best)
        except (RuntimeError, KeyError) as e:
            self._publish_ui_alert("warn", f"파지 예측/실행 건너뜀: {e}")
            self.get_logger().warn(f"pick prediction/execution skipped: {e}")

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
        """벨트 정지/재가동 (v48 회신, v23 원칙 개정 -- 파지 사이클
        중(파지 성공~배치~복귀)에만 정지).

        **[정정 — 6일차, v58 회신, 팀결정] 더 이상 `~/bin/conveyor`를
        직접 서브프로세스로 호출하지 않는다.** 시리얼 포트를 두
        프로세스가 동시에 만지면 충돌 위험이 있어(v56/v57/v58에서
        논의), 이제 `conveyor_node`(신규, `conveyor_node.py`)가
        시리얼을 단독 소유하고 이 메서드는 그 노드의 `/conveyor/
        set_speed`("on"에 대응)/`/conveyor/stop`("off"에 대응)
        서비스를 호출한다. `conveyor_node`가 안 떠 있으면(예: 이
        노드만 단독 테스트하는 경우) 짧은 타임아웃 후 에러 로그만
        남기고 계속 진행 -- 기존 서브프로세스 방식과 같은 "안 죽는다"
        동작을 유지한다. 재진입 스핀 방지(CLAUDE.md 알려진 함정)를
        위해 전용 `_conveyor_executor`를 쓴다(로봇 executor와 분리,
        dry_run 여부와 무관하게 항상 필요 -- 음성 명령은 dry_run
        중에도 들어올 수 있음).
        """
        client = self._conveyor_stop_client if cmd == "off" else self._conveyor_set_speed_client
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f"conveyor {cmd} failed: conveyor_node 서비스 응답 없음 "
                "(conveyor_node가 안 떠 있는지 확인할 것)"
            )
            return
        req = Trigger.Request()
        future = client.call_async(req)
        self._conveyor_executor.spin_until_future_complete(future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error(f"conveyor {cmd} failed: {result}")

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
            self._get_workpiece_weight_avg(n=3, interval=0.08)
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
            self._publish_ui_alert(
                "warn",
                f"예측 좌표 도달 범위 초과 — 건너뜀 (predicted_x={predicted_x:.1f}mm)",
            )
            self.get_logger().warn(
                f"[EXEC] predicted_x={predicted_x:.1f} > max_reachable_x={max_x} "
                "-- 표준 자세로 도달 불가, MoveLine 시도 없이 중단(물체가 벨트 "
                "끝쪽에서 너무 늦게 확정됨 -- 벨트 시작쪽에 더 가깝게 놓고 재시도할 것)"
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
            self._publish_ui_alert("error", "벨트 상공 이동 실패 — 파지 중단")
            self.get_logger().error(
                f"[EXEC] hover move did not reach target (got {pos[:3]}, "
                f"wanted {hover[:3]}) -- aborting pick"
            )
            # [버그 발견·수정 — 6일차, 핸드오버 1차 실물 시도] 예전 단일
            # 트랙(`_locked_class`) 설계에서는 여기서 락을 안 풀면 물체가
            # 계속 화면에 보이는 한 재시도 자체가 영원히 막히는 버그가
            # 있었다(사용자가 "로봇 안움직여"로 발견). **[완료 — 순차
            # 처리 인덱싱 재설계]** 이제는 트랙이 `_on_image`에서 이미
            # self._tracks에서 제거된 뒤 여기로 넘어오므로, 물체가 계속
            # 보이면 다음 프레임에 새 트랙으로 자연스럽게 재생성돼
            # 재시도된다 -- 같은 효과를 구조적으로 유지.
            return

        self.get_logger().info(f"[EXEC] descend {depth}mm")
        self._call_move_line([0.0, 0.0, -depth, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)

        self.get_logger().info("[EXEC] close gripper")
        self._call_gripper("c")
        time.sleep(profile["close_wait_sec"])

        self.get_logger().info(f"[EXEC] rise {depth}mm")
        self._call_move_line([0.0, 0.0, depth, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)
        self._holding = True

        # [v48 회신, v23 원칙 개정] 파지 성공 -- 배치+복귀 완료까지
        # 벨트 정지(다음 물체가 처리 중 도달범위를 벗어나는 문제
        # 방지). 검출~추적~예측~파지 구간은 벨트가 돌았음(칼만
        # 예측이 실제로 필요한 구간, 그대로 유지).
        self.get_logger().info("[EXEC] pick complete, stopping conveyor for placement")
        self._call_conveyor("off")

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
        self._stage = "measure"
        forced_bin = None
        net_weight = None
        if weight_check is not None:
            weight_kg = self._get_workpiece_weight_avg()
            threshold = weight_check["threshold_kg"]
            if weight_kg is not None and baseline_weight is not None:
                net_weight = weight_kg - baseline_weight
            self.get_logger().info(
                f"[EXEC] weight check for {item_key}: raw={weight_kg}kg "
                f"baseline={baseline_weight}kg net={net_weight}kg "
                f"(threshold={threshold}kg)"
            )
            if net_weight is not None and net_weight > threshold:
                forced_bin = weight_check["review_bin"]
                self._publish_ui_alert(
                    "warn", f"{item_key} 무게 초과 — {forced_bin}으로 라우팅"
                )
                self.get_logger().warn(
                    f"[EXEC] {item_key} over weight threshold -- routing to "
                    f"{forced_bin} instead (내용물 있는 것으로 의심)"
                )

        # [완료 — 8일차, v64 회신] `/ui/state`의 `last` 필드용 -- 실제
        # 통 이름(dest)은 아직 모른다(_place_item/_handoff_item이
        # 성공해야 확정), 여기서는 item/weight_g/reason만 먼저 채운다.
        self._last_result = {
            "item": item_key,
            "dest": None,
            "weight_g": None if net_weight is None else round(net_weight * 1000, 1),
            "reason": "over_weight" if forced_bin else None,
        }

        self.get_logger().info(f"[EXEC] pick complete for {item_key}, placing...")
        self._stage = "place"
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
            self._publish_ui_alert("error", f"{item_key} 라우팅 설정 없음 — 들고 대기")
            self.get_logger().error(f"no routing entry for {item_key!r} -- 들고 대기")
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

        self.get_logger().info(f"[PLACE] {item_key} -> {bin_name}: move to bin hover")
        self._call_move_line(hover_pose, h_vel, h_vel, mode=0)
        pos = self._get_current_posx()
        if abs(pos[0] - hover_pose[0]) > 5.0 or abs(pos[1] - hover_pose[1]) > 5.0:
            self._publish_ui_alert(
                "error", f"{bin_name} 상공 이동 실패 — 배치 중단, 사람 확인 필요"
            )
            self.get_logger().error(
                f"[PLACE] bin hover move did not reach target (got {pos[:3]}, "
                f"wanted {hover_pose[:3]}) -- 들고 대기, 배치 중단, 벨트도 정지 유지"
                "(사람이 확인 후 재시작할 것)"
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
        """실제 순응제어 기반 사람 핸드오버 (6일차 구현).

        새 목표 좌표(x/y)를 만들지 않는다 -- `_execute_pick`이 이미
        도달을 검증한 벨트 상공 (predicted_x, y=-264, z=430) 컬럼
        에서 z만 더 올린다(같은 x/y, 더 높은 곳은 덜 뻗는 자세라
        도달성 측면에서 더 안전할 것으로 판단, 새 자세를 탐색하는
        위험을 피함). CLAUDE.md 안전 절의 검증 기준(저강성 500+
        당김 방향+핸드오버 자세+라벨 페트병, 10회 연속 SAFE_OFF
        0회)과 v25 예비 시행(TCP 변위 36.8mm까지 SAFE_OFF 없음,
        0.3초 폴링 75회 전부 정상 응답)을 그대로 적용 -- release
        임계값은 그 예비 시행에서 쓴 잠정값(10mm)을 기본값으로 둔다.

        절차: 그리퍼 닫힌 채 z만 상승(제시 높이) -> task_compliance_ctrl
        (저강성)으로 순응 모드 진입 -> 기준점 캡처 -> get_current_posx
        를 폴링해 **수평 이동 속도**가 연속 N회 임계값 이상이고
        AND 누적 변위도 임계값을 넘으면(사람이 당김) 그리퍼 오픈 ->
        release_compliance_ctrl로 순응 모드 해제 -> 원래 높이로 복귀
        -> 벨트 재가동. 판별 기준이 "누적 변위(총량)"에서 "수평
        속도"로 바뀐 경위는 아래 `_handoff_item` 본문의 주석(8일차,
        v76 회신) 참고 -- 총량 기준으로는 느린 크리프를 못 걸러낸다는
        지적을 받고 전환했다.

        **[완료 — 8일차, 팀결정] `timeout_sec` 안에 당김이 감지되지
        않으면** 그 자리에서 그냥 여는 대신(이전 동작 -- 물체가 그
        자리에서 낙하) **review_bin("확인 필요")으로 정상 배치
        시퀀스를 거쳐 옮긴다.**

        [전제] 10회 연속 무사고 검증 전이므로 "안전이 확정됐다"고
        표현하지 않는다(CLAUDE.md 과대 청구 금지 원칙) -- 오히려
        8일차에 관제PC 실물 테스트에서 오탐 릴리즈(사람 없이 약
        3초 만에 릴리즈, 물체 낙하)가 재현됐고, 그 다음 수정(적응형
        settle)도 근본 해결이 아니라는 지적을 받아 판별 방식 자체를
        속도 기반으로 바꿨다 — 이 역시 로봇 실물 재검증 전.
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

        self.get_logger().info(
            f"[HANDOFF] {item_key}: rise {extra_rise_mm}mm to presentation height "
            "(같은 x/y 컬럼, 새 자세 탐색 안 함)"
        )
        # [버그 발견·수정 — 8일차, 관제PC 팀원 실물 테스트에서 실측]
        # 이 상승 이동(mode=1, 상대좌표)에 도달 확인이 아예 없었다 --
        # `_execute_pick`/`_place_item`의 다른 MoveLine 호출은 전부
        # 도달 여부를 검증하는데(CLAUDE.md "알려진 함정": MoveLine은
        # success=True를 반환하면서 실제로는 안 움직일 수 있음) 이
        # 호출만 빠져 있었다. **실제로 도달 실패가 났다**(팀원 실측:
        # x=561.72, z=550 NOT REACHABLE) -- 확인 없이 다음 단계
        # (순응모드 진입)로 그냥 넘어가면 물체가 원치 않는 자세/
        # 높이에서 그대로 순응제어에 들어가게 되는 위험한 상태다.
        # mode=1(상대좌표)이라 절대좌표 비교가 아니라 "이동 전후
        # z 변화량이 의도한 만큼인지"로 확인한다.
        pos_before_rise = self._get_current_posx()
        self._call_move_line([0.0, 0.0, extra_rise_mm, 0.0, 0.0, 0.0], v_vel, v_vel, mode=1)
        pos_after_rise = self._get_current_posx()
        actual_rise = pos_after_rise[2] - pos_before_rise[2]
        if abs(actual_rise - extra_rise_mm) > 5.0:
            self._publish_ui_alert(
                "error", f"{item_key} 제시 높이 상승 실패 — 핸드오버 중단, 사람 확인 필요"
            )
            self.get_logger().error(
                f"[HANDOFF] presentation-height rise did not reach target "
                f"(원했던 상승 +{extra_rise_mm}mm, 실제 +{actual_rise:.1f}mm, "
                f"pos_after={[round(v, 1) for v in pos_after_rise[:3]]}) -- "
                "순응모드 진입 안 함, 들고 대기(사람이 확인 후 재시작할 것)"
            )
            return

        self.get_logger().info(
            f"[HANDOFF] entering compliance mode stx={stiffness} "
            f"(release_threshold_mm={release_threshold_mm}, timeout_sec={timeout_sec})"
        )
        self._call_task_compliance_ctrl(stx=stiffness, ref=0, time_=compliance_transition_sec)
        time.sleep(compliance_transition_sec)

        # [버그 이력]
        # 6일차: start_pos를 순응모드 진입 *이전*에 캡처 -> 저강성
        #   진입 자체의 중력 처짐을 "당김"으로 오인(2.9초만에
        #   disp=10.7mm로 오탐 릴리즈, 물체 낙하). 수정: 진입 +
        #   고정 0.5초 settle 후 캡처.
        # 8일차(1차 재발): 관제PC 실물에서 사람 없이 약 3초 만에
        #   또 오탐 릴리즈 -- 고정 0.5초로는 처짐이 다 안 멈춘
        #   상태였을 것으로 추정. 수정: 위치가 실제로 멈출 때까지
        #   (연속 폴링 간 이동량이 임계값 이하) 적응적으로 기다린
        #   뒤 캡처하는 방식으로 교체.
        # 8일차(v76 회신, 재정정): **적응형 settle도 근본 해결이
        #   아니다** -- "이동량이 매 폴링 2mm 이하"는 변화율 조건일
        #   뿐 총량 조건이 아니라서, 물체가 폴링마다 2mm씩 계속
        #   처지는 **느린 크리프(creep)**는 settle 조건을 계속
        #   통과하면서도 몇 초에 걸쳐 누적되면 결국 총 변위 임계값을
        #   넘는다 -- 기준점을 한 번 잡고 누적 변위만 재는 구조인 한
        #   settle을 아무리 정교하게 해도 이 문제가 남는다는 지적을
        #   받고 판별 방식 자체를 바꿨다.
        #
        # **[완료 — 8일차, v76 회신] 판별 기준을 "누적 변위(총량)"
        # 에서 "수평 속도"로 전환.** 물리적으로 사람이 당기는 동작은
        # 빠르고(추정 20~50mm/s), 중력에 의한 처짐/크리프는 느리다
        # (6일차 사고 실측 기준 평균 약 3.7mm/s) -- 총량이 아니라
        # **속도**로 보면 느린 크리프가 아무리 오래 누적돼도 걸리지
        # 않는다. 그래서 적응형 settle 로직 자체가 불필요해져
        # 제거했다(속도 게이팅이 스스로 초기 전이 구간의 빠른 튐도
        # 걸러낼 수 있음 -- 단, 진입 직후 급격한 첫 처짐 스냅은
        # `compliance_transition_sec` 대기로 흡수).
        #
        # 릴리즈 조건(AND, v76 1-5절 권장 조합): (a) **수평** 방향
        # (dx,dy만, dz 제외 -- 중력 처짐은 -Z가 주성분이라 애초에
        # 속도 계산에서 배제, v76 1-4절) 이동 속도가 연속
        # `pull_speed_consecutive`회 이상 `pull_speed_mm_per_s` 이상
        # AND (b) 시작점 대비 누적 3D 변위가 `release_threshold_mm`
        # 이상(미세한 흔들림 배제용 하한선, 기존 로직 유지).
        # [전제, 미검증] `pull_speed_mm_per_s`=15.0은 실측 3.7mm/s와
        # 추정 20~50mm/s 사이의 중간값(v76 1-3절) -- 로봇 실물
        # 재검증(느린 처짐 조건/실제 당김 조건 둘 다) 전까지 확정
        # 값 아님. 매 폴링 로그를 남겨(아래) 다음 실물 테스트에서
        # 분포를 실측할 수 있게 했다.
        start_pos = self._get_current_posx()
        start_t = time.monotonic()
        self.get_logger().info(
            f"[HANDOFF] baseline captured, start_pos="
            f"{[round(v, 1) for v in start_pos[:3]]}"
        )

        pull_speed_mm_per_s = cfg.get("pull_speed_mm_per_s", 3.0)
        pull_speed_consecutive = cfg.get("pull_speed_consecutive", 2)

        released = False
        fast_count = 0
        prev_pos = start_pos
        prev_t = start_t
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            time.sleep(poll_interval_sec)
            now_t = time.monotonic()
            cur_pos = self._get_current_posx()
            dt = now_t - prev_t  # [v76 1-6절] 고정 poll_interval_sec을
            # 가정하지 않고 실제 경과 시간으로 나눈다.

            step_horizontal = (
                (cur_pos[0] - prev_pos[0]) ** 2 + (cur_pos[1] - prev_pos[1]) ** 2
            ) ** 0.5
            speed_horizontal = step_horizontal / dt if dt > 0 else 0.0

            # [버그 발견·수정 — 8일차, v87 회신] 누적변위(AND 조건의
            # 절반)가 3D 전체(dx,dy,dz)라서 처짐(-Z)만으로도 이미
            # 10mm를 넘어버렸다(1회차 실측: 3D 누적 19.4mm, 거의
            # 전부 -Z). 즉 AND가 사실상 속도 조건 하나에만 기대는
            # 상태였다. **수정**: 아래쪽(-Z, 처짐 방향)만 누적에서
            # 제외하고 수평+위쪽(+Z, 들어올림)은 그대로 포함한다 --
            # 사람이 받을 때는 자기 쪽으로 당기거나(수평) 들어
            # 올리지(+Z), 아래로 누르진 않는다는 물리적 근거.
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

            # [v76 요청2] 다음 실물 테스트에서 처짐/당김 속도 분포를
            # 실측할 수 있도록 매 폴링 상세 로그를 남긴다.
            # total_disp_3d(수정 전 정의)도 같이 남겨서 v87 수정
            # 전후 값을 로그만으로 비교할 수 있게 한다.
            self.get_logger().info(
                "[HANDOFF] poll elapsed=%.2fs pos=(%.1f,%.1f,%.1f) "
                "total_disp=%.1fmm total_disp_3d=%.1fmm step_horiz=%.1fmm "
                "step_vert=%+.1fmm dt=%.2fs speed_horiz=%.1fmm/s fast_count=%d/%d"
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

            self.get_logger().info(f"[HANDOFF] returning to belt hover height")
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

        # [완료 — 8일차, 팀결정] 타임아웃(사람이 안 받아감) 시 그
        # 자리에서 그냥 여는 대신 review_bin("확인 필요")으로 옮긴다.
        # **이전 동작**: 제시 높이에서 바로 그리퍼를 열어 물체가 그
        # 자리에서 낙하했다(실측 -- 관제PC 테스트에서 약 3초 만에
        # 라벨 페트병 낙하 확인, 단 이 사고 자체는 아래 STABLE_EPS_MM
        # settle 수정이 원인이었을 가능성이 높음 -- 타임아웃 15초를
        # 다 채우기 전에 오탐 릴리즈였을 것으로 추정, 실측 로그 없어
        # 확정은 아님). **지금 동작**: 정상 배치와 같은 "수직 상승 ->
        # 수평 이동 -> 수직 하강" 3단계 시퀀스(CLAUDE.md 안전 절)를
        # 그대로 따라 review_bin에 배치한다 -- 사람이 못 받아간
        # 물체를 허공에 떨어뜨리지 않는다.
        self.get_logger().warn(
            f"[HANDOFF] {item_key}: no pull detected within {timeout_sec}s -- "
            "routing to review_bin instead of releasing in place"
        )
        self.get_logger().info("[HANDOFF] releasing compliance mode before re-routing")
        self._call_release_compliance_ctrl()
        time.sleep(compliance_transition_sec)

        h_vel = [mt["horizontal_vel"], mt["horizontal_acc"]]
        bin_name = "review_bin"
        # [버그 발견·수정 — 8일차, 관제PC 실물+사용자 육안 확인,
        # v87 회신] `get_bin_pose`가 주는 자세(rx/ry/rz)를 그대로
        # 쓰면 파지 자세 대비 67.33° 돌아간 채로 들어가는데, 라벨
        # 페트병은 그 자세에서 길이가 review_bin 가로폭을 넘어
        # 안 들어간다(로그는 도달 확인만 보므로 정상 완료로 보였음
        # -- 실제로는 그리퍼가 열렸을 때 통에 안 들어가고 걸치거나
        # 떨어짐). **[팀결정 — 사용자 확인 + v87] 무게초과 캔/
        # plastic_bag은 기존 회전 배치 그대로 두고(원통형이라 자세
        # 무관, 비닐도 문제 없음 확인됨), pet_labeled만 파지 각도
        # 그대로 진입하도록 예외 처리한다.** 이제 `_apply_place_
        # pose_override`(범용, `_place_item`도 같이 씀)로 처리 --
        # `config/item_routing.yaml`의 `place_pose_override.
        # pet_labeled.review_bin`에 설정돼 있음. [전제, 미검증]
        # 이 (좌표, 자세) 조합은 한 번도 실행된 적 없다 -- 실물
        # 투입 전 반드시 빈 그리퍼로 먼저 이동시켜 도달성/그리퍼-
        # 통벽 간섭을 확인할 것(4일차 "상승 없이 이동하면 그리퍼가
        # 통을 밀어버린" 사고 전례 있는 구역, obstacles.yaml의
        # review_bin은 미실측 상태).
        hover_pose = get_bin_pose(bin_name, "hover")
        place_pose = get_bin_pose(bin_name, "place")
        hover_pose, place_pose = self._apply_place_pose_override(
            item_key, bin_name, hover_pose, place_pose
        )

        self.get_logger().info(f"[HANDOFF] {item_key} -> {bin_name} (timeout): move to bin hover")
        self._call_move_line(hover_pose, h_vel, h_vel, mode=0)
        pos = self._get_current_posx()
        if abs(pos[0] - hover_pose[0]) > 5.0 or abs(pos[1] - hover_pose[1]) > 5.0:
            self._publish_ui_alert(
                "error", f"{bin_name} 상공 이동 실패 — 배치 중단, 사람 확인 필요"
            )
            self.get_logger().error(
                f"[HANDOFF] {bin_name} hover move did not reach target (got {pos[:3]}, "
                f"wanted {hover_pose[:3]}) -- 들고 대기, 배치 중단, 벨트도 정지 유지"
                "(사람이 확인 후 재시작할 것)"
            )
            return

        self.get_logger().info(f"[HANDOFF] descend to {bin_name} place pose")
        self._call_move_line(place_pose, v_vel, v_vel, mode=0)

        self.get_logger().info("[HANDOFF] open gripper")
        self._call_gripper("o")
        time.sleep(self._item_routing.get("place_open_wait_sec", 1.0))

        self.get_logger().info(f"[HANDOFF] rise back to {bin_name} hover")
        self._call_move_line(hover_pose, v_vel, v_vel, mode=0)

        next_x = max((t["kf"].state.x for t in self._tracks), default=None)
        if next_x is None:
            next_x = mt.get("default_return_x", 300.0)
        belt_hover = [
            next_x, mt["belt_hover_y"], mt["belt_hover_z"],
            mt["belt_hover_rx"], mt["belt_hover_ry"], mt["belt_hover_rz"],
        ]
        self.get_logger().info(f"[HANDOFF] return to belt hover (x={next_x:.1f})")
        self._call_move_line(belt_hover, h_vel, h_vel, mode=0)

        self._holding = False
        self._pick_counts[item_key] = self._pick_counts.get(item_key, 0) + 1
        self._bin_counts[bin_name] = self._bin_counts.get(bin_name, 0) + 1
        if self._last_result is not None and self._last_result.get("item") == item_key:
            self._last_result["dest"] = bin_name
            self._last_result["reason"] = "handoff_timeout"
        self._publish_ui_alert("warn", f"{item_key} 인계 시간 초과 — {bin_name}으로 재배치")
        self._stage = "idle"
        self.get_logger().info(
            f"[HANDOFF] {item_key} timeout -> placed in {bin_name}, restarting conveyor"
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
