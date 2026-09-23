import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(
    page_title="Traffic Sign Detector",
    page_icon="🚦"
)

st.title("🚦 Traffic Sign Recognition")
st.write(
    "Upload a traffic-sign image and send it to the FastAPI model-serving API."
)

api_url = st.text_input(
    "FastAPI URL",
    "http://localhost:8000"
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded:

    # Read uploaded image
    image = Image.open(uploaded).convert("RGB")

    # Show original image
    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("Predict", type="primary"):

        try:
            # Send image to FastAPI
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type
                )
            }

            response = requests.post(
                f"{api_url.rstrip('/')}/predict",
                files=files,
                timeout=60
            )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    f"Found {data['detection_count']} detection(s)."
                )

                # Create a copy for drawing
                annotated_image = image.copy()

                draw = ImageDraw.Draw(annotated_image)

                # Default font
                font = ImageFont.load_default()

                # Draw every detection
                for detection in data["detections"]:

                    bbox = detection["bbox"]

                    x1, y1, x2, y2 = map(int, bbox)

                    class_name = detection["class_name"]
                    confidence = detection["confidence"]

                    label = f"{class_name} {confidence:.1%}"

                    # Bounding box
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline="red",
                        width=5
                    )

                    # Calculate text size
                    text_box = draw.textbbox(
                        (0, 0),
                        label,
                        font=font
                    )

                    text_width = text_box[2] - text_box[0]
                    text_height = text_box[3] - text_box[1]

                    # Label background
                    label_y = max(0, y1 - text_height - 8)

                    draw.rectangle(
                        [
                            x1,
                            label_y,
                            x1 + text_width + 8,
                            label_y + text_height + 8
                        ],
                        fill="red"
                    )

                    # Label text
                    draw.text(
                        (x1 + 4, label_y + 4),
                        label,
                        fill="white",
                        font=font
                    )

                # Show annotated image
                st.subheader("Prediction Result")

                st.image(
                    annotated_image,
                    caption="Detected Traffic Sign",
                    use_container_width=True
                )

                # Optional JSON result
                with st.expander("View API Response"):
                    st.json(data)

            else:
                st.error(
                    f"API error {response.status_code}: {response.text}"
                )

        except requests.RequestException as exc:

            st.error(
                f"Could not connect to FastAPI: {exc}"
            )