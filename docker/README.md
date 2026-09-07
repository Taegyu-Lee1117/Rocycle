# tracking_node Docker 실행

**[전제, 미검증] GPU 없는 개발 PC에서 작성 — 관제 PC(GPU 있음)에서
빌드·실행 검증이 필요합니다.**

## 왜 Docker인가

`ultralytics`(YOLO)가 최신 `opencv-python`을 끌어오면서 `numpy`를
2.x로 올리는데, 이게 ROS2 `cv_bridge`(numpy 1.x 기준 컴파일)를
깨뜨립니다(CLAUDE.md "알려진 함정" 참고). 베어메탈에서는 "pip
opencv-python 제거 + 시스템 python3-opencv 사용"으로 우회했는데,
**컨테이너 안에서도 같은 조치가 그대로 필요합니다** — Docker가 이
충돌 자체를 없애주지는 않습니다. 대신 이 환경을 호스트의 다른
설정과 분리해서 재현 가능하게 만들어줍니다.

## 빌드

```bash
cd rocycle_robot   # docker/Dockerfile을 이 디렉터리 기준으로 작성함
docker build -f docker/Dockerfile -t rocycle-tracking-node .
```

## GPU 확인 (빌드 전에)

```bash
nvidia-smi   # CUDA 버전 확인
```

pip 기본 torch가 이미 CUDA 지원 빌드일 가능성이 높지만, 안 맞으면
`Dockerfile` 안의 주석 처리된 torch 설치 줄을 확인된 CUDA 버전에
맞게 채워 넣고 다시 빌드하세요.

## 실행

```bash
docker run --rm -it \
  --network host \
  --gpus all \
  --device=/dev/video2 \
  -e ROS_DOMAIN_ID=30 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  rocycle-tracking-node
```

**`--network host` 필수** — ROS2 DDS는 컨테이너의 기본 브리지
네트워크에서는 디스커버리가 잘 안 됩니다(STT/TTS 팀원 Docker
연동 때와 같은 문제, `STT_TTS_연동_요청사항.txt` 참고).

**`--device=/dev/video2`는 실제 카메라 장치 번호로 바꾸세요** —
재부팅하면 번호가 바뀔 수 있습니다, `/dev/v4l/by-id/` 경로 고정을
권장합니다(CLAUDE.md 4-4절과 동일한 주의사항).

**usb_cam 노드는 이 컨테이너 안에 없습니다** — 별도로(컨테이너
안이든 호스트든) 띄워야 `/image_raw`가 발행됩니다. 컨테이너
안에서 함께 띄우려면:
```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video2 -p image_width:=1280 \
  -p image_height:=720 -p pixel_format:=mjpeg2rgb -p framerate:=30.0 &
python3 -m rocycle_robot.tracking_node
```
(이미지가 위 Dockerfile CMD를 오버라이드하는 예시 -- 실제로는
`docker run`의 커맨드 인자나 별도 진입 스크립트로 두 프로세스를
같이 띄우는 방법을 팀에서 정할 것)

## 확인되지 않은 것 (팀에서 실제로 테스트 필요)

```
[ ] 이미지가 실제로 빌드되는가(numpy/torch 버전 충돌 없이)
[ ] GPU가 컨테이너 안에서 인식되는가(torch.cuda.is_available())
[ ] --network host로 호스트의 다른 ROS2 노드(conveyor_node 등)와
    실제로 통신되는가
[ ] 카메라 장치가 컨테이너 안에서 정상 인식되는가
```
