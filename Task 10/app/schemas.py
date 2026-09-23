from pydantic import BaseModel, Field

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)

class PredictionResponse(BaseModel):
    filename: str
    detections: list[Detection]
    detection_count: int
