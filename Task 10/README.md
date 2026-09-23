# Task 10 — End-to-End MLOps Pipeline: Traffic Sign Recognition

This project converts the YOLOv8 traffic-sign detector from Task 8 into a deployable
FastAPI model-serving service, containerized with Docker and tested with automated CI.

## Architecture

```text
Client / Postman / Streamlit
          |
          v
      FastAPI API
          |
          v
 Pydantic / Request Validation
          |
          v
     Image Validation
          |
          v
     YOLOv8 best.pt
          |
          v
   JSON Predictions
```

## Requirements covered

- FastAPI model serving
- YOLOv8 `.pt` model artifact from Task 8
- Pydantic response schema validation
- Explicit HTTP 400 validation for unsupported/corrupt images
- Docker containerization
- Postman collection
- Streamlit frontend bonus
- GitHub Actions CI/CD bonus
- Automated API tests

## 1. Run locally

Use Python 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 2. API

### Health

`GET /health`

Response:

```json
{
  "status": "ok",
  "model": "loaded_on_first_prediction"
}
```

### Prediction

`POST /predict`

Content type:

`multipart/form-data`

Field:

`file`

Allowed image formats:

- JPG
- JPEG
- PNG
- WEBP

Example response:

```json
{
  "filename": "traffic_sign.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "example",
      "confidence": 0.91,
      "bbox": [100.0, 80.0, 240.0, 190.0]
    }
  ],
  "detection_count": 1
}
```

## 3. Validation

The API returns HTTP 400 when:

- the uploaded file is not an allowed image MIME type
- the extension is unsupported
- the file is empty
- the image is corrupted/unreadable
- the image is larger than 10 MB
- FastAPI/Pydantic request validation fails

## 4. Docker

Build:

```bash
docker build -t traffic-sign-api .
```

Run:

```bash
docker run --rm -p 8000:8000 traffic-sign-api
```

Or:

```bash
docker compose up --build
```

Then open:

`http://localhost:8000/docs`

## 5. Postman

Import:

`postman_collection.json`

For `POST /predict`, choose:

Body → form-data → key `file` → type `File`

Select a Task 8 test image.

## 6. Streamlit bonus

Start the API first.

Then:

```bash
streamlit run frontend/streamlit_app.py
```

Enter:

`http://localhost:8000`

Upload an image and click Predict.

## 7. Tests

Run:

```bash
pytest -q
```

The tests cover:

- health endpoint
- invalid file type → 400
- corrupt image → 400
- valid image request → 200

## 8. GitHub Actions

Every push and pull request triggers:

`.github/workflows/ci.yml`

The workflow installs dependencies and runs the test suite.

## 9. Notes

The model is loaded lazily on the first prediction request. This keeps the API startup
lighter and avoids loading the model when only the health endpoint is used.

The Docker image uses a CPU-compatible Python base image. GPU acceleration can be added
later with a CUDA-enabled runtime if the deployment environment provides a compatible
NVIDIA GPU and container runtime.
