# GTSDB Traffic Sign Recognition — YOLOv8 V3 Final

A real-time traffic sign detection and classification project based on the German Traffic Sign Detection Benchmark (GTSDB).

The model detects and classifies traffic signs directly from **full road-scene images**, including small signs and complex backgrounds.

## Final V3 Model

- Architecture: **YOLOv8n**
- Training input size: **960 × 960**
- Number of classes: **43 GTSDB classes**
- Pretrained initialization: **Yes**
- Training device: **NVIDIA GeForce RTX 4060 Laptop GPU**
- CUDA training: **Yes**
- Best checkpoint: `weights/best.pt`
- Deployment model: `weights/best.onnx`
- ONNX opset: **12**

## Dataset

The project uses 900 GTSDB images:

| Split | Images |
|---|---:|
| Train | 480 |
| Validation | 120 |
| Test | 300 |
| **Total** | **900** |

The original 300-image test set is preserved. Background images without labels are intentionally included in the training/validation data. See `SPLIT_REPORT.txt` for the split details and class distribution.

## V3 Training Configuration

- Epochs: **120 maximum**
- Batch size: **8**
- Image size: **960**
- Patience: **30**
- Optimizer: `auto`
- Cosine learning-rate schedule: enabled
- AMP: enabled
- Mosaic: enabled
- Horizontal flip: enabled
- HSV augmentation: enabled
- Translation: `0.1`
- Scale: `0.5`
- Windows DataLoader workers: **0** for stability

The included `train_v3.py` reproduces the V3 training configuration without requiring multiprocessing workers on Windows.

## Final V3 Validation Results

Validation was run on the 120-image validation split at **960 × 960** using `weights/best.pt`.

| Metric | V3 Result |
|---|---:|
| Precision | **34.78%** |
| Recall | **33.51%** |
| mAP@50 | **33.92%** |
| mAP@50-95 | **29.59%** |

### Important note

Object detection is evaluated with Precision, Recall, and mAP rather than ordinary classification accuracy.

The main limitation of the final results is the **limited and imbalanced amount of labeled data per class**. The project contains 43 traffic-sign classes but only 900 images in total (480 train / 120 validation / 300 test), and several classes have very few examples. This makes it difficult for the model to learn and generalize equally well across all classes. In addition, many signs are very small inside full road scenes with complex backgrounds.

Further improvement would primarily require a larger and more balanced labeled dataset. Longer training, higher-resolution experiments, and larger models would also require substantially more GPU time/resources, so a stronger GPU/server would make those experiments more practical. The hardware is therefore mainly a constraint on experimentation speed; the primary limitation is the available training data.

## Inference Speed

A fixed-image benchmark was run on the complete **300-image test set** using the V3 `best.pt` model with the benchmark script at **640 × 640**.

| Metric | Result |
|---|---:|
| Images tested | **300** |
| Total time | **7.530 sec** |
| Average time | **25.10 ms/image** |
| FPS | **39.84** |
| Device | **CUDA** |
| GPU | **NVIDIA GeForce RTX 4060 Laptop GPU** |

**Note:** 39.84 FPS is a fixed-image inference benchmark, not a measured webcam FPS.

## ONNX Deployment Bonus

The final trained model has been exported to ONNX:

```text
weights/best.onnx
```

The exported model uses a **960 × 960** input and ONNX opset 12.

The project includes local webcam inference using the ONNX model with a confidence threshold of **0.25**.

Run:

```powershell
python webcam.py
```

Press `Q` to exit.

If `onnxruntime-gpu` is installed and CUDA is available, the webcam script automatically selects the CUDA execution provider; otherwise it falls back to CPU.

## Quick Demo

Run:

```powershell
python demo.py
```

Options:

1. Verified demo image
2. Test your own image
3. Webcam
4. Exit

The verified demo uses a lower confidence threshold only to make small traffic signs easier to visualize. Normal image prediction and webcam inference use a confidence threshold of **0.25**.

## Image Prediction

```powershell
python predict.py
```

Or:

```powershell
python predict.py path\to\image.jpg
```

The prediction script displays bounding boxes, class names, and confidence scores.

## FPS Benchmark

```powershell
python benchmark_fps.py
```

This benchmarks the 300 images in `images/test`.

## ONNX Export

The final ONNX file is already included. To reproduce the export from the final PyTorch checkpoint:

```powershell
python export_v3.py
```

For ONNX inference:

```powershell
python onnx_inference.py
```

## Evaluation

To reproduce the V3 validation evaluation:

```powershell
python evaluate_v3.py
```

The evaluation script uses `workers=0`, which avoids the Windows multiprocessing error encountered during validation.

## Training

The training script is included for reproducibility:

```powershell
python train_v3.py
```

**Warning:** training is not required for normal project demonstration because the trained `best.pt` is already included.

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the base requirements:

```powershell
pip install -r requirements.txt
```

For ONNX Runtime GPU acceleration:

```powershell
pip install -r requirements-gpu.txt
```

For NVIDIA GPU training/inference, install a compatible CUDA-enabled PyTorch build for your system if it is not already installed.

## Project Structure

```text
GTSDB_Traffic_Sign_Recognition_FINAL_V3/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
├── weights/
│   ├── best.pt
│   ├── best.onnx
│   └── yolov8n.pt
│
├── results/
│   ├── sample_input.jpg
│   └── secondary_input.jpg
│
├── demo.py
├── predict.py
├── webcam.py
├── benchmark_fps.py
├── train.py
├── train_v3.py
├── evaluate_v3.py
├── export_v3.py
├── export_onnx.py
├── onnx_inference.py
├── data.yaml
├── SPLIT_REPORT.txt
├── V3_RESULTS.txt
├── FINAL_SUBMISSION_CHECKLIST.txt
├── README.md
├── requirements.txt
├── requirements-gpu.txt
└── .gitignore
```

## Technologies

- Python
- Ultralytics YOLO
- YOLOv8
- PyTorch
- OpenCV
- ONNX
- ONNX Runtime
- CUDA

## Limitations

Traffic signs can be very small in full-resolution road scenes, and some classes have limited training examples. Therefore, per-class performance is uneven. The lightweight YOLOv8n architecture was retained to balance detection quality and inference speed.

## Final Status

**Submission-ready V3 package**

- [x] Full-scene object detection
- [x] 43 traffic-sign classes
- [x] Trained YOLOv8n V3 model
- [x] Validation metrics recorded
- [x] 300-image FPS benchmark recorded
- [x] PyTorch checkpoint included
- [x] ONNX model included
- [x] Webcam inference included
- [x] Confidence threshold filtering included
- [x] Reproducible training/evaluation/export scripts included
- [x] README and project structure included
