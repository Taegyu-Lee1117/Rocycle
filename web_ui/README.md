# 관제 화면 (web_ui)

`index.html` 하나짜리 정적 페이지(별도 빌드 과정 없음, roslibjs만
CDN에서 로드). 설계 배경/논의 이력은 저장소 루트의
`재활용로봇_v65_응답.txt`~`v75_응답.txt`, 현재 유효한 설계는
`관제화면_UI설계안.md`(이 폴더) 참고 — **문서가 여러 버전
쌓여 헷갈리면 `관제화면_UI설계안.md`가 최신 확정본이다.**

## 이관 체크리스트 (관제 PC에서 `git pull` 후 할 일)

### 1) pull로 오는 것 / 오지 않는 것

```
[pull로 온다]
  web_ui/ 전체(이 폴더)
  rocycle_robot/tracking_node.py (QoS 수정, /ui/* 발행,
    handoff 타임아웃 처리 등)
  config/item_routing.yaml
  그 외 이번 세션에서 커밋된 파일 전부

[pull로 안 온다 -- 관제 PC에서 별도 설치 필요]
  ros-jazzy-rosbridge-suite      (apt install)
  ros-jazzy-web-video-server     (apt install)
  두 패키지 다 이 개발 PC에는 이미 설치돼 있어서(사전 확인 없이도
  됨) 별도 설치 스텝을 실제로 검증하진 못했다 -- 관제 PC에 없으면
  `sudo apt install ros-jazzy-rosbridge-suite ros-jazzy-web-video-server`
  로 설치.
```

### 2) 기동 절차 (포트 3개)

```bash
# 1. rosbridge (WebSocket, 포트 9090)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# 2. web_video_server (HTTP 영상, 포트 8080)
ros2 run web_video_server web_video_server

# 3. tracking_node (이미 떠 있어야 함 -- /ui/state 등 발행 주체)
#    (평소 기동 방식 그대로, dry_run 여부와 무관하게 /ui/* 발행됨)

# 4. web_ui 정적 서버 (포트 8000)
cd ros2/rocycle_robot/web_ui
python3 -m http.server 8000
```

브라우저에서 `http://<관제PC IP>:8000/`을 연다. 기본값은 이 페이지를
서빙하는 호스트(=관제 PC)를 rosbridge(9090)/web_video_server(8080)
주소로 그대로 쓴다. 다른 머신에 떠 있다면 쿼리 파라미터로 지정:

```
http://<PC>:8000/?rosbridge_host=192.168.1.50&video_host=192.168.1.50
```

### 3) 확인 절차

```
□ 브라우저 콘솔에 에러 없는지 확인(F12)
□ 좌상단 상태 텍스트가 "연결 중..."이 아니라 실제 상태(대기 중/
  작업 중 등)로 바뀌는지 -- 안 바뀌면 rosbridge(9090) 연결 실패
□ 영상이 뜨는지 -- 안 뜨면:
    1. curl http://localhost:8080/snapshot?topic=/image_raw&type=jpeg
       로 web_video_server 자체가 정상인지 먼저 확인
    2. 정상인데 화면에 안 뜨면 브라우저 콘솔에서 CORS/네트워크
       에러 확인(video_host가 맞는 주소인지)
□ "상세" 버튼 눌러서 막대그래프/진행단계/연결상태가 뜨는지
□ 우상단 조작 버튼(시작/일시정지/재개/종료) 눌렀을 때 실제로
  로봇/컨베이어가 반응하는지(터미널 로그에 [VOICE] 라인 확인)
□ 비상정지 버튼 눌렀을 때 컨베이어가 실제로 멈추는지(로봇 쪽은
  아직 인터럽트 여부 미검증 -- 아래 참고)
```

## 구성 (상세는 `관제화면_UI설계안.md` 참고)

- **메인 화면**: 영상(canvas, snapshot 폴링) + 상태(대형 텍스트,
  색 구분) + 경과 시간 + "상세" 버튼 + 비상정지 버튼 + 조작 버튼
  4개(시작/일시정지/재개/종료, `/voice_command`로 발행) + 알림
  패널(`/ui/alert` 구독, 활성 알림 있을 때만 표시).
- **상세 페이지**: 막대그래프 6개(정상분류 4·사람전달 1·확인필요
  1, 색 3종) + 진행 단계 칩 + 마지막 처리 결과 + 연결 상태 점 5개
  + 자체 비상정지 버튼. 30초 무조작 시 메인으로 자동 복귀.

## [전제, 미검증] 관제 PC에서 재확인 필요 (환경 의존)

이번 세션은 GPU 없는 개발 PC + 합성 카메라로 진행했고, 세션
후반엔 이 개발 PC 자체가 자원 소진 상태(TCP TIME_WAIT 수백 개,
여유 메모리 부족)였던 것까지 확인됐다 — 즉 아래 항목은 "이
환경에서 이랬다"이지 "관제 PC에서도 이럴 것"이 아니다:

- ~~snapshot 폴링이 실제로 안정적으로 뜨는지~~ **[정정 이력 —
  8일차]** 처음엔 관제PC에서 curl로 확인해 `http=200` 나온 걸 보고
  "개발PC 자원 소진 때문이었다"고 결론 냈는데(1차 해소 판정),
  **이게 틀렸다** — curl 테스트 시 `topic=/image_raw`(미인코딩)로
  쳐서 우연히 통과했을 뿐, 실제 브라우저(`Image()`)는
  `encodeURIComponent(topic)`로 `%2Fimage_raw`를 보내고 있었고,
  `web_video_server`가 `%2F`를 디코딩하지 않아 응답 없이 연결을
  끊는(`net::ERR_EMPTY_RESPONSE`) **진짜 코드 버그**였다 — 자원
  상태와 무관하게 정상 환경(관제PC)에서도 100% 재현됨을 확인.
  **[완료]** `encodeURIComponent` 대신 실제로 쿼리 구조를 깨는
  문자만 이스케이프하는 `encodeTopicForQuery()`로 교체(`/`는
  인코딩 안 함) — 수정 후 관제PC 재검증 필요(다음 항목).
- canvas 갱신 성능(10fps 유지), YOLO 실제 추론 fps에 따른 예측
  타이밍 재조정. (미검증 유지)
- 화면 해상도별 레이아웃 — 16:9 기준으로 만들었다, 관제 PC
  모니터 비율 확인할 것. (미검증 유지)

## [전제, 미검증] 환경과 무관하게 유효한 사실

- `<img>` 태그는 `multipart/x-mixed-replace`(MJPEG 스트림)를
  최신 Chrome에서 렌더링하지 않는다(fetch/naturalWidth로 직접
  확인) -- 그래서 `<iframe>`을 거쳐 최종적으로 **snapshot 폴링 +
  canvas**로 정착했다.
- `/dsr01/dsr_controller2/motion/move_stop` 서비스 이름은 로봇
  드라이버 소스코드로 존재를 확인했으나(`dsr_msgs2/srv/MoveStop`),
  로봇이 실제로 떠 있을 때 `ros2 service list`로 정확한 네임스페이스
  재확인 필요 -- 비상정지 버튼의 로봇 정지 부분은 "전달만 되고
  즉시 정지는 미확인"이라는 문구를 화면에 그대로 유지한다(과대
  청구 금지 원칙, 실동작 검증 전까지 지우지 말 것).
- `/ui/detections`(bbox 오버레이)는 아직 화면에 안 그림 -- canvas
  구조는 이미 준비돼 있어(snapshot 폴링과 같은 canvas) 다음
  단계에서 그 위에 그리면 된다.

## 알려진 제약 (아키텍처, 코드로 못 고침)

`stage`가 "pick"/"measure"/"place"/"handoff"로 바뀌는 블로킹
구간(17~24초)에는 `/ui/state`/`/ui/alert`/`/ui/detections` 발행
자체가 멈춘다(`tracking_node`가 단일 스레드 executor라서). 화면의
"경과 시간"은 이 사실을 감안해 **클라이언트 로컬 시계**로 계산한다
(서버가 안 보내도 클라이언트가 직접 잰다).
