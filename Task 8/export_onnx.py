from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.pt"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    onnx_path = model.export(
        format="onnx",
        imgsz=960,
        opset=12,
        simplify=True,
    )

    print("\nONNX export completed.")
    print(f"Saved to: {onnx_path}")


if __name__ == "__main__":
    main()
