import os
import time
from typing import List, Dict, Any, Optional, Generator, Union
import numpy as np
import cv2
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .utils import load_config, draw_detections, FPSCounter
from .roi import ROIManager, LineCounter

@dataclass
class DetectionResult:
    """Dataclass holding detection output for a single image/frame."""
    frame_index: int
    timestamp: float
    detections: List[Dict[str, Any]] = field(default_factory=list)
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    annotated_image: Optional[np.ndarray] = None
    fps: float = 0.0

class YOLODetector:
    """Unified YOLO Object Detection & Multi-Object Tracking Engine."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        tracker_type: str = "bytetrack.yaml"
    ):
        if YOLO is None:
            raise ImportError("Ultralytics library is not installed. Run 'pip install ultralytics'.")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.tracker_type = tracker_type
        
        # Load YOLO Model
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, "names") else {}
        self.fps_counter = FPSCounter()

    def update_parameters(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        model_path: Optional[str] = None
    ):
        """Dynamically update model inference settings."""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if model_path is not None and model_path != self.model_path:
            self.model_path = model_path
            self.model = YOLO(model_path)
            self.class_names = self.model.names

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        enable_tracking: bool = False,
        target_classes: Optional[List[int]] = None,
        roi_manager: Optional[ROIManager] = None,
        line_counter: Optional[LineCounter] = None,
        annotate: bool = True
    ) -> DetectionResult:
        """
        Run object detection or tracking on a single OpenCV BGR image matrix.
        """
        h, w = frame.shape[:2]
        self.fps_counter.start() if frame_index == 0 else None
        
        # Run inference or tracking
        if enable_tracking:
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=target_classes,
                tracker=self.tracker_type,
                persist=True,
                verbose=False
            )
        else:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=target_classes,
                verbose=False
            )

        detections = []
        counts_by_class = {}
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist() # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                track_id = int(box.id[0].cpu().item()) if box.is_track and box.id is not None else None
                
                det_record = {
                    "frame_index": frame_index,
                    "timestamp": time.time(),
                    "bbox": [round(c, 2) for c in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "track_id": track_id
                }
                detections.append(det_record)
                counts_by_class[cls_name] = counts_by_class.get(cls_name, 0) + 1

        # Apply ROI Filtering if enabled
        if roi_manager and not roi_manager.is_empty():
            inside_dets, _ = roi_manager.filter_detections(detections, w, h)
            detections = inside_dets
            # Recalculate class counts for ROI
            counts_by_class = {}
            for d in detections:
                cn = d["class_name"]
                counts_by_class[cn] = counts_by_class.get(cn, 0) + 1

        # Apply Line Crossing Counter if enabled
        if line_counter:
            line_counter.update(detections, w, h)

        # Draw Annotations
        annotated_img = None
        if annotate:
            annotated_img = draw_detections(
                frame,
                detections,
                show_labels=True,
                show_conf=True,
                show_tracks=enable_tracking
            )
            if roi_manager and not roi_manager.is_empty():
                annotated_img = roi_manager.draw_roi(annotated_img)
            if line_counter:
                annotated_img = line_counter.draw_line(annotated_img)

        fps = self.fps_counter.update()

        return DetectionResult(
            frame_index=frame_index,
            timestamp=time.time(),
            detections=detections,
            counts_by_class=counts_by_class,
            total_count=len(detections),
            annotated_image=annotated_img,
            fps=round(fps, 1)
        )

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        target_classes: Optional[List[int]] = None
    ) -> DetectionResult:
        """Process a static image file from disk."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        result = self.detect_frame(image, target_classes=target_classes)

        if output_path and result.annotated_image is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, result.annotated_image)

        return result

    def process_video_stream(
        self,
        video_source: Union[str, int],
        enable_tracking: bool = True,
        target_classes: Optional[List[int]] = None,
        roi_manager: Optional[ROIManager] = None,
        line_counter: Optional[LineCounter] = None,
        max_frames: Optional[int] = None
    ) -> Generator[DetectionResult, None, None]:
        """
        Generator for frame-by-frame processing of video file or webcam stream.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect_frame(
                    frame=frame,
                    frame_index=frame_idx,
                    enable_tracking=enable_tracking,
                    target_classes=target_classes,
                    roi_manager=roi_manager,
                    line_counter=line_counter,
                    annotate=True
                )

                yield result
                frame_idx += 1

                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
