import argparse
from datetime import datetime
from pathlib import Path

import cv2

from detector import RecycleDetector


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolo12s_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO12s camera example")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera {index} could not be opened. Try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def main():
    args = parse_args()
    detector = RecycleDetector(DEFAULT_MODEL, conf=args.conf)
    cap = open_camera(args.camera)
    screenshots = ROOT / "test_output"
    screenshots.mkdir(exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detection = detector.detect(frame)

            if detection is not None:
                x1, y1, x2, y2 = detection.bbox
                anchor_x, anchor_y = detection.anchor_pixel

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (anchor_x, anchor_y), 7, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"{detection.class_name} {detection.confidence:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "S: screenshot  Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("YOLO12s Team Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("s"), ord("S")):
                name = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")
                path = screenshots / name
                cv2.imwrite(str(path), frame)
                print(f"Saved: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
