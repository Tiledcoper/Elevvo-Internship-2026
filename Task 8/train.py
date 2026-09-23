from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "yolov8n.pt"
DATA_PATH = ROOT / "data.yaml"


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_PATH}")

    device = get_device()

    print("========== TRAINING SYSTEM ==========")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_PATH),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        pretrained=True,
        patience=10,
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.5,
        project=str(ROOT / "runs"),
        name="gtsdb_yolov8n_final",
        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
