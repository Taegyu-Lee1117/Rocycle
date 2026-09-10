# control_pc — 관제PC에서 **실제로 실행되는** 노드들

이 디렉터리는 관제PC(MS-03)에서 시연 시 실제로 구동되는 파일 중
**저장소 다른 위치에 없거나, 있어도 구현이 다른 것**만 모은 것이다.
2026-09-10 아침 점검에서 "관제PC 디스크가 유일한 사본"인 파일이
확인되어 추가했다.

## object_detection/

YOLO 검출 노드 패키지. **저장소 어디에도 없었다**(`git ls-files`로
확인, 0건). Docker 이미지 `rocycle-yolo:jazzy` 안에 빌드돼 있어
컨테이너가 살아 있는 동안에는 동작하지만, 이미지를 잃으면 복구
경로가 없었다.

실행 방법(관제PC 기준):
```
docker exec -d rocycle-yolo-cpu bash -lc \
  'source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && \
   exec ros2 run object_detection recycle_detection --ros-args \
   -p model_path:=/models/recycle_best.pt -p device:="cpu"'
```
`device`는 GPU가 살아 있으면 `"0"`, 아니면 `"cpu"`. 2026-09-10 현재
커널 업데이트(7.0.0-30 -> 7.0.0-31)로 NVIDIA 커널 모듈이 빌드되지
않아 `cpu`로 운용 중이다. 검출 속도는 GPU 사용 때와 같다(원래도
CPU 추론이었다).

## recycle_robot/conveyor_node.py

컨베이어 노드. 저장소 루트의 `rocycle_robot/conveyor_node.py`와
**완전히 다른 구현**이다(약 270줄 차이).

```
저장소 rocycle_robot/conveyor_node.py : /conveyor/set_speed, /conveyor/stop 서비스 방식
여기 있는 실행본                       : /conveyor_command 토픽 방식 (START / STOP)
```

`tracking_pick_node`가 `/conveyor_command` 토픽으로 제어하므로
**실제 시연에서 도는 것은 이쪽**이다. 실행: `ros2 run recycle_robot conveyor_node`
(패키지 이름이 `rocycle_robot`이 아니라 `recycle_robot`인 점에 주의).

기존 파일을 덮어쓰지 않고 나란히 둔 이유: 어느 구현을 정본으로 삼을지는
별도 정리 과제이며, 시연 직전에 판단할 사안이 아니다.

## 알려진 저장소/실행본 불일치 (시연 이후 정리 과제)

관제PC 실행 트리(`~/Rocycle_Project/ros2_ws/src/rocycle_robot`)와
저장소 `rocycle_robot/`은 위 파일 외에도 `routing.py`, `tracking_node.py`,
`geometry/pickup_timing.py`, `voice/command_mapping.py`, `package.xml`,
`setup.py`, 테스트 파일이 서로 다르고 한쪽에만 있는 파일도 있다.
어느 쪽이 최신인지 확인 없이 맞추면 다른 사람 작업을 지울 수 있어
손대지 않았다.

## 모델 가중치는 여기 없다

`object_detection/resource/` 안에 있던 `.pt` 파일 두 개는 **일부러
제외했다**:

```
recycle_best.pt        md5 16fe1922  = v5 (구버전, 실제로 안 쓰임)
yolov8n_tools_0122.pt                 (무관한 잔재)
```

실제 운용 모델은 **v6**(`md5 8b4d25fa`)이며 컨테이너가
`/home/rokey/Rocycle_Project/models/recycle_best.pt`를
`/models/recycle_best.pt`로 마운트해서 쓴다. 패키지 안의 v5 사본을
같이 올리면 나중에 그것을 정본으로 착각할 위험이 있어 뺐다.
v5 가중치는 저장소 `models/v5/`에 이미 있다.
