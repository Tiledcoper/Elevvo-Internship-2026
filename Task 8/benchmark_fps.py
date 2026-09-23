import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.pt"
TEST_DIR = ROOT / "images" / "test"
IMAGE_SIZE = 640
NUM_WARMUP = 10


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {TEST_DIR}")

    for path in image_paths[:NUM_WARMUP]:
        frame = cv2.imread(str(path))
        if frame is not None:
            model.predict(source=frame, imgsz=IMAGE_SIZE, device=device, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    processed = 0

    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        model.predict(source=frame, imgsz=IMAGE_SIZE, device=device, verbose=False)
        processed += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    fps = processed / elapsed if elapsed > 0 else 0.0
    avg_ms = (elapsed / processed) * 1000 if processed else 0.0

    print("\n========== FPS BENCHMARK ==========")
    print(f"Images tested : {processed}")
    print(f"Total time    : {elapsed:.3f} sec")
    print(f"Average time  : {avg_ms:.2f} ms/image")
    print(f"FPS            : {fps:.2f}")
    print(f"Device         : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
