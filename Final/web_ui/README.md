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
□ 상단바 상태 칩이 "연결 중..."이 아니라 실제 상태(대기 중/
  작업 중 등)로 바뀌는지 -- 안 바뀌면 rosbridge(9090) 연결 실패
□ 영상이 뜨는지 -- 안 뜨면:
    1. curl "http://localhost:8080/snapshot?topic=/recycle_detection/image&type=jpeg"
       로 web_video_server 자체가 정상인지 먼저 확인
    2. 정상인데 화면에 안 뜨면 브라우저 콘솔에서 CORS/네트워크
       에러 확인(video_host가 맞는 주소인지)
□ 우측 사이드바에 처리현황 막대그래프/진행단계/연결상태 칩이
  뜨는지(별도 "상세" 화면 없음 -- v104에서 메인 한 화면으로 통합)
□ 조작 버튼(시작/일시정지/재개/종료) 눌렀을 때 실제로 로봇/
  컨베이어가 반응하는지(터미널 로그에 [VOICE] 라인 확인)
□ 비상정지 버튼 눌렀을 때 컨베이어가 실제로 멈추는지(로봇 쪽은
  아직 인터럽트 여부 미검증 -- 아래 참고)
```

## 구성 (상세는 `관제화면_UI설계안.md` 참고)

**[완료 -- 8일차, v104/v107 회신] 레이아웃 전면 교체.** 5분할/
상세페이지 분리 구조를 버리고 **상단바 + 좌우분할(좌: 고정카메라
전체화면, 우: 사이드바) + 하단 알림레일** 한 화면으로 통합했다.

- **상단바**: 로고 + 상태 칩(IDLE/RUNNING/PAUSED/STOPPING, 색
  구분) + 경과 시간 + 연결상태 pill 5개(카메라/로봇/컨베이어/
  STT/TTS) + 시계.
- **좌측(메인 영상)**: 고정 카메라(C270 + 검출오버레이) 영역
  전체를 채움(`object-fit` 상당의 canvas cover 렌더링, 가장자리
  일부 잘림 있음 -- 아래 참고). 하단 스트립에 카메라명/해상도/
  fps/현재 단계 표시.
- **우측 사이드바**: 손목캠(RealSense) 소형 미리보기 + 진행 단계
  (검출~인계 8단계, 완료/현재/미도달 3색) + 처리 현황(막대그래프
  6개: 정상분류 4·사람전달 1·확인필요 1, 3색) + 마지막 처리
  결과 + 조작 버튼 4개(`/voice_command` 발행) + 비상정지.
- **하단 알림 레일**: `/ui/alert` 구독, 최근 5건 가로 배치,
  error/warn/info 3색 구분, 10초 후 서서히 사라짐.
- 상세 페이지(별도 화면)는 v104에서 제거됨 -- 위 내용이 전부
  메인 화면에 상시 표시되므로 중복이었음.

### 데모 모드 (`?demo=1`)

**[완료 -- 8일차, v107 요청] 로봇/카메라 없이 화면 검토용.**
```
http://localhost:8000/               실제 운용. 데모 값 없음
http://localhost:8000/?demo=1        데모 값 표시
```
- **데모 값은 `?demo=1`에서만 표시된다.** 실제 운용 시에는
  파라미터 없이 접속한다 — **시연 전 반드시 주소창에 `demo`가
  없는지 확인할 것.**
- 기본값은 꺼짐(안전 쪽) — 코드 상단 상수가 아니라 URL 쿼리로만
  켜지므로, 끄는 걸 잊고 git에 커밋될 위험이 없다.
- 데모 모드에서는 **rosbridge 연결 자체를 시도하지 않는다**
  (`ros` 객체를 생성하지 않음) — 실제 로봇에 연결된 채로 데모
  URL이 켜져 있어도 조작 버튼이 진짜 명령을 내보내지 않도록
  원천 차단.
- 실제 렌더 함수(`render()`/`pushAlert()`)를 그대로 재사용 —
  데모 전용 DOM을 따로 하드코딩하지 않으므로 데모 화면이 실제
  화면의 정확한 예고가 된다.
- 데모 값에는 의도적으로 극단값을 섞어 넣었다: 플라스틱 0건
  (막대 0일 때 표시 확인), TTS 꺼짐(연결점 off 상태 확인),
  오류·경고 각 1건(세 알림 색이 한 화면에 나오는지 확인).

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
  모니터 비율 확인할 것. **[실측 — 8일차] 1600×900 헤드리스
  기준 좌우 각 4.6% 잘림(cover 렌더링, 상하 0%)** — 벨트는
  가로지르는 구도라 남지만 양 끝(투입 시작점/끝단)이 잘리는지는
  실제 모니터 비율에 좌우돼 실물 확인 필요(미검증 유지).

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
