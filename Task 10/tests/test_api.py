from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

def make_image_bytes():
    image = Image.new("RGB", (100, 100), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_invalid_file_type_returns_400():
    response = client.post(
        "/predict",
        files={"file": ("data.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400

def test_corrupt_image_returns_400():
    response = client.post(
        "/predict",
        files={"file": ("bad.png", b"not really an image", "image/png")}
    )
    assert response.status_code == 400

def test_valid_image_request():
    with patch("app.main.predict_image", return_value=[]):
        response = client.post(
            "/predict",
            files={"file": ("test.png", make_image_bytes(), "image/png")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "test.png"
    assert body["detection_count"] == 0
    assert body["detections"] == []
