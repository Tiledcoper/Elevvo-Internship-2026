from pathlib import Path

import cv2
import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.onnx"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 960
CAMERA_INDEX = 0


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

    providers = ort.get_available_providers()
    use_gpu = "CUDAExecutionProvider" in providers
    device = 0 if use_gpu else "cpu"

    print("========== ONNX WEBCAM INFERENCE ==========")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available providers: {providers}")
    print(f"Inference device: {'NVIDIA GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Press Q to quit.")

    # Ultralytics automatically uses ONNX Runtime for the .onnx model.
    model = YOLO(str(MODEL_PATH), task="detect")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=device,
                verbose=False,
            )

            annotated_frame = results[0].plot()
            cv2.imshow("Traffic Sign Recognition - ONNX Webcam", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
