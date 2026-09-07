# YOLO12s 재활용 로봇 팀원 전달 패키지

이 폴더는 기존 파지 프로그램의 영상 인식 부분을 YOLO12s로 교체하기 위한 패키지입니다.

## 포함 파일

- `models/yolo12s_best.pt`: 학습된 모델
- `detector.py`: 기존 코드에서 불러올 검출 모듈
- `example_camera_test.py`: 카메라 동작 확인 예제
- `model_info.json`: 클래스, 성능, 입출력 규격
- `requirements.txt`: 필요한 Python 패키지
- `CLAUDE_INTEGRATION_PROMPT.md`: 기존 코드를 Claude에 수정 요청할 때 함께 제공할 설명

## 설치

이 폴더에서 터미널을 열고 실행합니다.

```bash
python -m pip install -r requirements.txt
```

## 카메라 테스트

```bash
python example_camera_test.py --camera 0
```

다른 카메라가 열리면 다음을 사용합니다.

```bash
python example_camera_test.py --camera 1
```

- `S`: 현재 검출 화면 저장
- `Q`: 종료

## 기존 코드에서 사용

```python
from detector import RecycleDetector

detector = RecycleDetector("models/yolo12s_best.pt", conf=0.25)
detection = detector.detect(frame)

if detection is not None:
    print(detection.class_name)
    print(detection.confidence)
    print(detection.bbox)
    print(detection.anchor_pixel)
```

`anchor_pixel`은 세워진 페트병도 벨트 평면상의 위치로 변환할 수 있도록 bbox 하단 중앙에 가깝게 잡은 픽셀입니다. 기존 카메라-로봇 좌표 변환에 넣을 후보 좌표이며, 실제 TCP 파지점과 높이는 기존 로봇 코드의 클래스별 설정을 유지해야 합니다.

## 클래스와 처리 방향

| 클래스 | 실제 대상 | 처리 방향 |
|---|---|---|
| `battery` | AA 건전지 | 건전지 통 |
| `can` | 알루미늄 캔 | 파지 후 무게 측정 |
| `paper` | 초코 종이상자 | 종이 통 |
| `pet_labeled` | 라벨 있는 페트병 | 사람 전달 |
| `pet_unlabeled` | 라벨 없는 페트병 | 플라스틱 통 |
| `plastic_bag` | 묶인 비닐 | 확인 필요 통 |

## 통합 시 주의사항

1. `.pt` 모델만으로 로봇 좌표는 나오지 않습니다. 기존 카메라-로봇 좌표 변환을 유지해야 합니다.
2. 동일한 물체가 매 프레임 검출되므로 검출 직후 매번 Pick 함수를 호출하면 안 됩니다.
3. 동일 클래스 연속 확인, Pick Zone 진입, 객체 잠금 조건을 통과했을 때 한 번만 Pick 명령을 발행해야 합니다.
4. 한 번에 물체 하나만 투입하므로 `max_det=1`을 사용합니다.
5. 카메라 주변에는 분류 대상 물체를 두지 않으므로 벨트 ROI는 사용하지 않습니다.
6. `pet_unlabeled` 일부가 `pet_labeled`로 판단될 수 있으므로 실제 영상에서 추가 확인해야 합니다.
