import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.pt"
DEFAULT_IMAGE = ROOT / "images" / "train" / "00002.jpg"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25


def main():
    parser = argparse.ArgumentParser(
        description="Run traffic sign detection on an image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="Path to input image",
    )
    args = parser.parse_args()
    image_path = Path(args.image)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        verbose=False,
    )

    result = results[0]
    print(f"Detections: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"Class: {model.names[class_id]} | Confidence: {confidence:.3f}")

    annotated = result.plot()
    cv2.imshow("Traffic Sign Detection - YOLOv8", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
