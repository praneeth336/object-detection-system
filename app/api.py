import base64
import os
import sys
import io
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import YOLODetector
from core.utils import load_config

app = FastAPI(
    title="YOLO Object Detection & Analytics REST API",
    description="High-performance API for object detection, multi-object tracking, and image analysis powered by Ultralytics YOLO.",
    version="1.0.0"
)

# Global Detector Instance
config = load_config()
default_model_path = config.get("models", {}).get("default_model", "yolov8n.pt")
detector_instance: Optional[YOLODetector] = None

@app.on_event("startup")
def startup_event():
    global detector_instance
    try:
        detector_instance = YOLODetector(model_path=default_model_path)
    except Exception as e:
        print(f"Warning: Failed to load default YOLO model on startup: {e}")

class DetectionItem(BaseModel):
    bbox: List[float] # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

class DetectionResponse(BaseModel):
    filename: str
    total_objects: int
    counts_by_class: Dict[str, int]
    detections: List[DetectionItem]
    annotated_image_base64: Optional[str] = None

@app.get("/api/v1/health")
def health_check():
    """Check API server and model readiness."""
    return {
        "status": "online",
        "model_loaded": detector_instance.model_path if detector_instance else None,
        "class_count": len(detector_instance.class_names) if detector_instance else 0
    }

@app.get("/api/v1/classes")
def get_supported_classes():
    """Retrieve list of object classes supported by the loaded YOLO model."""
    if not detector_instance:
        raise HTTPException(status_code=500, detail="Detector model not initialized.")
    return {"classes": detector_instance.class_names}

@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_object(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.01, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.01, le=1.0),
    return_annotated: bool = Query(True),
    classes: Optional[List[int]] = Query(None)
):
    """
    Perform object detection on an uploaded image file.
    Returns JSON bounding box predictions, class labels, counts, and optionally base64 annotated image.
    """
    if not detector_instance:
        raise HTTPException(status_code=500, detail="Detector model not initialized.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    # Update threshold dynamically
    detector_instance.update_parameters(conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    result = detector_instance.detect_frame(
        frame=image,
        target_classes=classes,
        annotate=return_annotated
    )

    base64_str = None
    if return_annotated and result.annotated_image is not None:
        _, buffer = cv2.imencode(".jpg", result.annotated_image)
        base64_str = base64.b64encode(buffer).decode("utf-8")

    return DetectionResponse(
        filename=file.filename or "uploaded_image.jpg",
        total_objects=result.total_count,
        counts_by_class=result.counts_by_class,
        detections=[DetectionItem(**d) for d in result.detections],
        annotated_image_base64=base64_str
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
