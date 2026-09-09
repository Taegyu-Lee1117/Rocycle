from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO


EXPECTED_CLASSES = [
    "battery",
    "can",
    "paper",
    "pet_labeled",
    "pet_unlabeled",
    "plastic_bag",
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list
    anchor_pixel: list

    def to_dict(self):
        return asdict(self)


class RecycleDetector:
    """YOLO12s detector interface for the recycling robot project."""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf: float = 0.25,
        imgsz: int = 640,
        device=None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        actual_classes = [self.model.names[i] for i in sorted(self.model.names)]
        if actual_classes != EXPECTED_CLASSES:
            raise ValueError(
                "Unexpected model classes. "
                f"expected={EXPECTED_CLASSES}, actual={actual_classes}"
            )

    def detect(self, frame) -> Optional[Detection]:
        """Return the highest-confidence object in one OpenCV frame."""
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.70,
            max_det=1,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        box = result.boxes[0]
        class_id = int(box.cls[0].item())
        class_name = self.model.names[class_id]
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().tolist())

        # This point lies close to the belt plane even for an upright bottle.
        anchor_x = int((x1 + x2) / 2)
        anchor_y = int(y1 + (y2 - y1) * 0.95)

        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=[x1, y1, x2, y2],
            anchor_pixel=[anchor_x, anchor_y],
        )

    def detect_all(self, frame, max_det: int = 3) -> list:
        """Return up to max_det objects in one frame (다중 물체 순차 처리용,
        6일차 추가). `detect()`(팀 원안, max_det=1, "한 번에 물체 하나만
        투입" 전제)는 그대로 두고, 순차 처리 인덱싱에 필요한 다중 검출
        경로만 별도로 추가한다.
        """
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.70,
            max_det=max_det,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().tolist())
            anchor_x = int((x1 + x2) / 2)
            anchor_y = int(y1 + (y2 - y1) * 0.95)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    anchor_pixel=[anchor_x, anchor_y],
                )
            )
        return detections
