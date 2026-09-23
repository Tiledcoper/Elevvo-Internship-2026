from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "weights" / "best.pt"


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    model = YOLO(str(MODEL))
    out = model.export(
        format="onnx",
        imgsz=960,
        simplify=True,
        opset=12,
    )
    print(f"ONNX exported to: {out}")


if __name__ == "__main__":
    main()
