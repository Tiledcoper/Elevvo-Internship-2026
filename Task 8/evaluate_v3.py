from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "weights" / "best.pt"
DATA = ROOT / "data.yaml"
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    model = YOLO(str(MODEL))
    metrics = model.val(
        data=str(DATA),
        imgsz=960,
        device=DEVICE,
        split="val",
        plots=True,
        workers=0,
    )

    print("\n=== V3 Validation Results ===")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
