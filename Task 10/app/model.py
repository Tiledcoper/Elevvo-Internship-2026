from pathlib import Path
from typing import Any
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

_model = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model

def predict_image(image: Any, conf: float = 0.25) -> list[dict]:
    model = get_model()
    results = model.predict(source=image, conf=conf, verbose=False)

    detections = []
    for result in results:
        names = result.names
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]

            detections.append({
                "class_id": cls_id,
                "class_name": str(names.get(cls_id, cls_id)),
                "confidence": round(confidence, 4),
                "bbox": xyxy
            })

    return detections
