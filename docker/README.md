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

## [중요] 이미지에는 코드만, 설정·캘리브레이션은 볼륨 마운트

**[정정 — 팀 피드백 반영]** 처음엔 `rocycle_robot/` 전체를 이미지에
구웠는데, 그러면 캘리브레이션 값이 바뀔 때마다(관제 PC 물리 배치가
바뀌거나 재캘리브레이션할 때) 이미지를 다시 빌드해야 합니다.
```
이미지에 굽는 것        코드만: rocycle_robot/, setup.py, package.xml
볼륨 마운트하는 것      config/, calib_capture/, models/
```
캘리브레이션 값이 바뀌면 파일만 바꾸고 컨테이너를 재시작하면
됩니다(이미지 재빌드 불필요) — 아래 실행 명령 참고.

**[안전] 볼륨을 안 걸면 config_loader가 파일을 못 찾아 노드가 죽습니다
(에러 로그와 함께 즉시 종료)** — v61이 우려했던 "캘리브레이션 없어도
조용히 뜨는" 실패 모드와 달리, 이 경우는 시끄럽게 죽으므로 오히려
바로 알아차릴 수 있습니다.

## [팀 피드백 — 순서 중요] Docker 빌드 *전에* 호스트에서 먼저 확인

**Docker 안에서 뭔가 안 되면 원인이 "컨테이너 문제"인지 "하드웨어
자체 문제"인지 구분이 안 됩니다.** 호스트에서 먼저 되는 걸
확인해두면 나중에 컨테이너에서 문제가 생겨도 하드웨어는 이미
검증됐으니 컨테이너 설정만 의심하면 됩니다. 순서:

```
1. main pull
2. [호스트] conveyor_node 실행 -> 벨트가 실제로 도는가
   (확인_체크리스트.md 1절, 로봇/카메라 불필요)
3. [호스트] 카메라 확인 -> /dev/video* 인식, usb_cam으로 30fps 나오는가
   (확인_체크리스트.md 2절, 로봇 불필요)
4. [호스트] nvidia-smi 실행 -> CUDA 버전 확인
   **GPU가 컨테이너 안에서만 안 보이면 nvidia-container-toolkit
   미설치가 가장 흔한 원인입니다** -- 호스트에서 nvidia-smi가
   되는지부터 확인하고, 안 되면 드라이버 문제, 되는데 컨테이너
   안에서만 안 보이면 nvidia-container-toolkit 설치 여부를 볼 것
   (`dpkg -l | grep nvidia-container-toolkit`).
5. 그 다음에 Docker 빌드
```

## 빌드

```bash
cd rocycle_robot   # 이 디렉터리(패키지 루트)를 빌드 컨텍스트로 사용
docker build -f docker/Dockerfile -t rocycle-tracking-node .
```

pip 기본 torch가 이미 CUDA 지원 빌드일 가능성이 높지만, 위 4번에서
확인한 CUDA 버전과 안 맞으면 `Dockerfile` 안의 주석 처리된 torch
설치 줄을 채워 넣고 다시 빌드하세요.

## 실행

```bash
docker run --rm -it \
  --network host \
  --gpus all \
  --device=/dev/video2 \
  -v "$(pwd)/config:/workspace/rocycle_robot/config:ro" \
  -v "$(pwd)/calib_capture:/workspace/rocycle_robot/calib_capture:ro" \
  -v "$(pwd)/models:/workspace/rocycle_robot/models:ro" \
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

**`-v` 세 줄은 호스트 경로(`$(pwd)/config` 등)를 실제 위치로
바꾸세요** — 위 예시는 패키지 루트에서 실행한다고 가정한 것입니다.
`:ro`(read-only)로 마운트해서 컨테이너 안에서 캘리브레이션 파일을
실수로 고치는 걸 막았습니다.

**usb_cam 노드는 이 컨테이너 안에 없습니다** — 별도로(컨테이너
안이든 호스트든) 띄워야 `/image_raw`가 발행됩니다. 컨테이너
안에서 함께 띄우려면 위 `docker run`의 커맨드를 오버라이드해서
`usb_cam_node_exe`와 `tracking_node`를 같이 실행하는 방법을
팀에서 정할 것.

## [팀 확인 필요] conveyor_node는 컨테이너 밖(호스트)에서 실행 권장

**시리얼 포트(`/dev/ttyUSB*`/`/dev/ttyACM*`)를 컨테이너에 넘기려면
`--device`가 필요하지만, `conveyor_node`는 의존성이 가볍고
(`std_srvs`만 필요) 시리얼 장치 하나만 쓰므로 굳이 컨테이너에
넣지 않고 호스트에서 그냥 `python3 -m rocycle_robot.conveyor_node`
로 실행하는 게 더 단순합니다.** `tracking_node`(컨테이너 안)는
`/conveyor/set_speed`·`/conveyor/stop` 서비스를 ROS2 네트워크로
호출할 뿐 시리얼을 직접 열지 않으므로, `--network host`만 걸려
있으면 컨테이너 안에서도 호스트의 `conveyor_node`를 정상적으로
호출할 수 있습니다(서비스 호출이지 시리얼 직접 접근이 아님).

정리하면:
```
컨테이너 안   tracking_node (+ 필요시 usb_cam)
호스트        conveyor_node, (팀원 STT/TTS 컨테이너와는 별개)
```

## 확인되지 않은 것 (팀에서 실제로 테스트 필요)

```
[ ] 이미지가 실제로 빌드되는가(numpy/torch 버전 충돌 없이)
[ ] GPU가 컨테이너 안에서 인식되는가(torch.cuda.is_available())
[ ] --network host로 호스트의 conveyor_node와 실제로 통신되는가
    (tracking_node 컨테이너 -> 호스트 conveyor_node 서비스 호출)
[ ] 카메라 장치가 컨테이너 안에서 정상 인식되는가
[ ] 볼륨 마운트한 config/calib_capture/models가 정상 로드되는가
    (기동 로그에 "calibration ready=True" 확인)
```
