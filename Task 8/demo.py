from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.pt"
DEMO_IMAGE = ROOT / "images" / "train" / "00002.jpg"
IMAGE_SIZE = 960
DEMO_CONFIDENCE = 0.05
USER_CONFIDENCE = 0.25
CAMERA_INDEX = 0


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def check_model():
    if not MODEL_PATH.exists():
        print(f"\nError: model not found: {MODEL_PATH}")
        return False
    return True


def print_system_info():
    print("\n========== SYSTEM ==========")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")


def run_image(model, image_path: Path, confidence: float):
    if not image_path.exists():
        print(f"\nError: image not found: {image_path}")
        return

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=confidence,
        device=get_device(),
        verbose=False,
    )

    result = results[0]
    print(f"\nImage: {image_path}")
    print(f"Confidence threshold: {confidence:.2f}")
    print(f"Detections: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        print(f"  - {model.names[class_id]} | Confidence: {score:.3f}")

    annotated = result.plot()
    cv2.imshow("GTSDB Traffic Sign Detection", annotated)
    print("Press any key in the image window to return to the menu.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(model):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("\nError: Could not open webcam.")
        return

    print("\nWebcam started. Press Q to return to the menu.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read webcam frame.")
                break

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=USER_CONFIDENCE,
                device=get_device(),
                verbose=False,
            )

            annotated = results[0].plot()
            cv2.imshow("GTSDB Traffic Sign Recognition - Webcam", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    if not check_model():
        raise SystemExit(1)

    print_system_info()
    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))

    while True:
        print("\n========================================")
        print(" GTSDB Traffic Sign Recognition Demo")
        print("========================================")
        print("1. Run verified demo image")
        print("2. Test your own image")
        print("3. Webcam")
        print("4. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            run_image(model, DEMO_IMAGE, DEMO_CONFIDENCE)
        elif choice == "2":
            raw_path = input("Enter image path: ").strip().strip('"')
            run_image(model, Path(raw_path), USER_CONFIDENCE)
        elif choice == "3":
            run_webcam(model)
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
