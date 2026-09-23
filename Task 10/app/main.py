from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.model import predict_image
from app.schemas import PredictionResponse

app = FastAPI(
    title="Traffic Sign Recognition API",
    description="FastAPI model-serving API for the Task 8 YOLOv8 traffic-sign detector.",
    version="1.0.0",
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.get("/")
def root():
    return {
        "message": "Traffic Sign Recognition API is running",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "POST /predict",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded_on_first_prediction"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Explicit validation so bad files return HTTP 400.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Allowed formats: JPG, JPEG, PNG, WEBP."
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file extension. Allowed: .jpg, .jpeg, .png, .webp."
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Image is too large. Maximum size is 10 MB.")

    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        detections = predict_image(image)
    except Exception as exc:
        # Keep internal implementation details out of the API response.
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    return {
        "filename": file.filename,
        "detections": detections,
        "detection_count": len(detections),
    }

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    # FastAPI/Pydantic validation failures are surfaced as 400 for this task.
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request schema or missing required input."},
    )
