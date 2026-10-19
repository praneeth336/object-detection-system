# YOLO Object Detection & Multi-Object Tracking System

A full-featured, modular object detection, tracking, zone analysis, and custom model training system powered by **Ultralytics YOLO (v8 / 11)**, **OpenCV**, **Streamlit**, and **FastAPI**.

---

## 🌟 Key Features

1. **Multi-Model Support**: Pre-trained YOLOv8/YOLO11 models (Nano, Small, Medium, Large) + Custom fine-tuned weights support.
2. **Multi-Object Tracking (MOT)**: Real-time object tracking using ByteTrack / BoT-SORT algorithms with unique track IDs.
3. **Region of Interest (ROI) & Line Crossing**:
   - Custom polygon zone detection filtering.
   - Line-crossing entry/exit object counter HUD overlay.
4. **Interactive Web Application (Streamlit)**:
   - Drag & drop image detection with side-by-side visualization.
   - Video file and real-time webcam feed processing with live analytics dashboard.
   - Parameter sliders for Confidence Threshold & IoU (NMS).
   - In-app custom model training interface.
5. **RESTful API Backend (FastAPI)**:
   - High-throughput programmatic endpoints returning JSON predictions and base64 annotated images.
6. **Command Line Interface (CLI)**:
   - Command-line tool for batch processing image folders, video files, running training, or starting web/API servers.
7. **Analytics Export**: Automatic export of detection history to CSV & JSON formats.

---

## 📁 Directory Structure

```
.
├── app/
│   ├── api.py            # FastAPI REST API server
│   └── streamlit_app.py  # Interactive Web Dashboard
├── core/
│   ├── detector.py       # YOLODetector wrapper engine
│   ├── roi.py            # ROI zone manager & line crossing counter
│   ├── trainer.py        # YOLOTrainer custom training pipeline
│   └── utils.py          # Image annotation, FPS tracker, CSV/JSON exports
├── sample_data/
│   └── data.yaml         # Dataset configuration template
├── cli.py                # Command Line Interface (CLI) entry point
├── train.py              # Standalone model training script
├── demo.py               # Demo & end-to-end verification script
├── config.yaml           # System configuration settings
└── requirements.txt      # Project dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate Virtual Environment (Windows)
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

---

## 🖥️ 1. Interactive Web Application (Streamlit)

Launch the interactive dashboard to process images, videos, webcams, or fine-tune models in your browser:

```bash
python cli.py serve-ui
# Or directly via streamlit:
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚡ 2. REST API Server (FastAPI)

Launch the REST API server:

```bash
python cli.py serve-api --port 8000
# Or directly via uvicorn:
python app/api.py
```
View interactive Swagger API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Sample API Request (Python)

```python
import requests

url = "http://localhost:8000/api/v1/detect"
files = {"file": open("sample_data/sample.jpg", "rb")}
params = {"conf_threshold": 0.25, "return_annotated": True}

response = requests.post(url, files=files, params=params)
data = response.json()

print(f"Detected Objects: {data['total_objects']}")
print(f"Class Breakdown: {data['counts_by_class']}")
```

---

## 💻 3. Command Line Interface (CLI)

### Image Detection (Single or Directory)
```bash
python cli.py detect-image --input sample_data/sample.jpg --model yolov8n.pt --save-csv --save-json
```

### Video / Webcam Detection & Tracking
```bash
# Process video file:
python cli.py detect-video --input input_video.mp4 --model yolov8s.pt --save-csv

# Process local webcam (Index 0):
python cli.py detect-video --input 0 --model yolov8n.pt
```

### Custom Model Fine-Tuning
```bash
python cli.py train --data sample_data/data.yaml --epochs 25 --batch 16
```

---

## 🏋️ Custom Model Fine-Tuning Guide

To train YOLO on your custom dataset:
1. Format dataset with YOLO format (`images/train`, `labels/train`, `images/val`, `labels/val`).
2. Update `sample_data/data.yaml` with your custom class names and directory paths.
3. Run training:
   ```bash
   python train.py --data sample_data/data.yaml --epochs 50 --batch 16 --model yolov8n.pt
   ```
4. Best trained weights will be saved in `runs/train/exp/weights/best.pt`. You can load this `.pt` file directly into the Web UI or API.

