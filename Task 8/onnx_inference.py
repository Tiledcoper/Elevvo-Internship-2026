from pathlib import Path

import cv2
import torch
import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.onnx"
IMAGE_PATH = ROOT / "images" / "train" / "00002.jpg"
CONFIDENCE_THRESHOLD = 0.05
IMAGE_SIZE = 640


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print("========== SYSTEM INFO ==========")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if hasattr(ort, "preload_dlls") and torch.cuda.is_available():
        ort.preload_dlls()

    print(f"ONNX Runtime: {ort.__version__}")
    print("Available providers:", ort.get_available_providers())

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        verbose=False,
    )

    result = results[0]
    print(f"\nDetections: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"Class: {model.names[class_id]} | Confidence: {confidence:.3f}")

    annotated = result.plot()
    cv2.imshow("ONNX Traffic Sign Detection", annotated)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
