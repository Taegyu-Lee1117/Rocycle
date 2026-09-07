# tracking_node(검출+추적+예측+로봇제어) 컨테이너 이미지.
#
# [팀결정 — 6일차] 시연 시 YOLO(ultralytics)는 Docker로 실행한다.
# **[전제, 미검증] 이 Dockerfile은 GPU가 없는 개발 PC에서 작성됐고
# 실제 빌드/실행 검증을 못 했다** — 관제 PC(GPU 있음)에서 빌드해
# 확인이 필요하다.
#
# 왜 통째로(tracking_node 전체를) 컨테이너에 넣는가: 원래 설계문서는
# 검출 전용 노드(fixed_camera_node)를 따로 두고 /track/detections
# 토픽으로 분리하는 구조지만, 지금 tracking_node는 검출부터 로봇
# 제어까지 한 노드에 있다(v58/v59에서 "통째로 이관 먼저" 팀결정과
# 같은 이유로, 지금 분리하면 검증된 코드를 크게 건드리게 됨). 이
# 컨테이너도 같은 방침으로 tracking_node 전체를 담는다 — 나중에
# 문제가 생기면 검출만 분리하는 방향을 검토할 것(아래 "향후" 참고).
#
# 왜 컨테이너가 필요한가(numpy 충돌, CLAUDE.md 알려진 함정과 동일
# 원인): ultralytics가 최신 opencv-python을 끌어오면서 numpy를
# 2.x로 올리는데, 이게 cv_bridge(numpy 1.x 기준 컴파일)를 깨뜨린다.
# 베어메탈에서는 "pip opencv-python 제거 + 시스템 python3-opencv
# 사용"으로 우회했는데, 컨테이너 안에서도 **같은 조치가 여전히
# 필요하다**(컨테이너가 이 충돌 자체를 없애주진 않음 — 컨테이너의
# 이점은 이 격리를 호스트의 다른 환경과 분리해서 재현 가능하게
# 만드는 것).

FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=30
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-opencv \
        ros-jazzy-cv-bridge \
        ros-jazzy-usb-cam \
        ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# [중요] numpy<2 고정 -- ultralytics/torch가 최신 numpy를 끌어와도
# 이 핀 때문에 못 올라간다. opencv-python(pip)은 절대 설치하지
# 말 것 -- 위에서 설치한 시스템 python3-opencv를 쓴다.
RUN pip install --break-system-packages --no-cache-dir \
        "numpy<2" \
        ultralytics==8.4.140 \
        pyyaml

# [확인 필요, GPU 전용] 기본 pip 설치로 받는 torch가 이미 CUDA
# 지원 빌드일 수 있으나(리눅스 x86_64 기본 휠), 호스트 CUDA 드라이버
# 버전과 안 맞으면 아래처럼 맞는 인덱스로 명시 설치할 것 — 정확한
# 버전은 관제 PC의 `nvidia-smi`로 CUDA 버전 확인 후 채울 것:
#   pip install --break-system-packages --no-cache-dir torch \
#     --index-url https://download.pytorch.org/whl/cu121   # 예시, 버전 확인 필요

WORKDIR /workspace
COPY rocycle_robot/ /workspace/rocycle_robot/

ENV PYTHONPATH=/workspace/rocycle_robot:$PYTHONPATH

# entrypoint에서 ROS2 환경 소싱 후 실행 -- 아직 colcon 설치 체계가
#없어(CLAUDE.md 참고) source-path 방식 그대로 유지.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "rocycle_robot.tracking_node"]
