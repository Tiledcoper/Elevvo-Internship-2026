from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.yaml"
MODEL = ROOT / "weights" / "yolov8n.pt"

IMG_SIZE = 960
EPOCHS = 120
BATCH = 8
WORKERS = 0  # Windows-safe
PATIENCE = 30
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH}")
    print(f"Workers: {WORKERS}")

    model = YOLO(str(MODEL))

    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        cache=False,
        amp=True,
        mosaic=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="traffic_sign_v3_960",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    best = ROOT / "runs" / "traffic_sign_v3_960" / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Best model: {best}")
    print("For final evaluation: python evaluate_v3.py")
    print("For ONNX export: python export_v3.py")


if __name__ == "__main__":
    main()
