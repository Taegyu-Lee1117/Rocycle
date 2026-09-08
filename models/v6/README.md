# YOLO12s Ver6

재활용품 분류 프로젝트의 YOLO12s 최종 학습 모델과 핵심 평가 결과입니다.

## 클래스

1. `battery`
2. `can`
3. `paper`
4. `pet_labeled`
5. `plastic`
6. `plastic_bag`

## 학습 조건

- epochs: 50
- imgsz: 640
- batch: 8
- seed: 42
- optimizer: AdamW
- pretrained: true

## 내부 Test 결과

| 항목 | 결과 |
|---|---:|
| Precision | 0.9967 |
| Recall | 1.0000 |
| mAP50 | 0.9950 |
| mAP50-95 | 0.9691 |

`conf=0.25` 혼동행렬에서는 Test 213개 객체가 모두 정답 클래스에 검출되었습니다. 다만 이는 Roboflow 내부 Test 결과이므로, 실제 C270 웹캠 및 컨베이어 환경에서 별도 검증이 필요합니다.

## 파일 안내

- `yolo12s_best.pt`: 최종 추론용 가중치
- `yolo12s_ver6_train_eval_colab.ipynb`: 학습·평가 Colab 코드
- `args.yaml`: 학습 설정
- `results.csv`: epoch별 학습 지표
- `results.png`: 학습 곡선
- `test_metrics.json`: Test 전체·클래스별 지표
- `confusion_matrix_conf025.png`: `conf=0.25` Test 혼동행렬
- `dataset_audit.json`: 클래스, 분할, 라벨 및 중복 검사 결과

